"""
Drop-in-ish replacement for the yfinance calls in the dashboard, backed by
Dhan's Data APIs instead. Takes plain NSE trading symbols (no ".NS" suffix)
and handles the symbol -> security_id lookup and access-token refresh
internally.

Implemented as direct REST calls (not the dhanhq SDK) because the SDK's
market-data methods have changed shape across versions (older installs
like 1.3.3 don't even expose a live-quote method). Hitting Dhan's
documented v2 Data API endpoints directly works regardless of which SDK
version — or none at all — is installed.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Optional

import pandas as pd
import pytz
import requests

from dhan_auth import DhanTokenManager
from dhan_instruments import NSE_EQ_EXCHANGE_SEGMENT, DhanInstrumentLookup

logger = logging.getLogger("dhan_market_data")

API_BASE_URL = "https://api.dhan.co/v2"
REQUEST_TIMEOUT_SECONDS = 15
INSTRUMENT_TYPE_EQUITY = "EQUITY"
IST = pytz.timezone("Asia/Kolkata")

# Observed empirically: Dhan's /charts/* endpoints rate-limit after only
# ~7 rapid sequential calls (HTTP 429, errorCode DH-904). MIN_CALL_INTERVAL
# is the same safe aggregate spacing as before -- 1 request initiated per
# this many seconds, across ALL callers combined. What changed is HOW it's
# enforced: previously a single thread slept this long AFTER each response
# came back (so total time per call was network_latency + interval,
# compounding serially). Now a shared _RateLimiter paces request
# *initiation* times only, so multiple symbols' network waits can overlap
# instead of stacking -- same total requests/second sent to Dhan, less
# wall-clock time spent idle.
MIN_CALL_INTERVAL_SECONDS = 0.35
MAX_429_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.5
BATCH_FETCH_WORKERS = 5


class _RateLimiter:
    """
    Thread-safe pacing gate. Each call to wait_turn() reserves the next
    available time slot (spaced min_interval apart from the previous
    reservation) and returns once that slot arrives. Reservation is a brief
    locked operation; the actual waiting happens outside the lock, so
    threads don't block each other's network I/O -- only the request
    *initiation* times are kept evenly spaced.
    """

    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait_turn(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_allowed)
            self._next_allowed = start + self._min_interval
        sleep_time = start - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)


# Module-level singleton: shared across every DhanMarketData instance in
# this process, so multiple scanners (CPR PRO, PivotBoss, Ichimoku, ...)
# can't collectively exceed the same safe aggregate rate even if each
# creates its own DhanMarketData object.
_RATE_LIMITER = _RateLimiter(MIN_CALL_INTERVAL_SECONDS)

# Option Chain is a completely separate, much stricter limit per Dhan's own
# docs: 1 request per 3 seconds. Must NOT share the limiter above, or every
# other endpoint would get throttled down to option-chain speed (or this
# endpoint would blow through its own limit using the faster one).
OPTION_CHAIN_MIN_INTERVAL_SECONDS = 3.0
_OPTION_CHAIN_RATE_LIMITER = _RateLimiter(OPTION_CHAIN_MIN_INTERVAL_SECONDS)
DEFAULT_FNO_EXCHANGE_SEGMENT = "NSE_FNO"


class DhanMarketData:
    # Dhan error codes meaning the access token itself was rejected server-side
    # (not a rate limit, not a data issue) -- e.g. a second script/process
    # regenerated a token for the same client, silently invalidating this
    # process's in-memory one. Worth a forced token refresh + one retry
    # rather than failing every symbol in a batch identically.
    _AUTH_ERROR_CODES = {"DH-906", "DH-901"}

    def __init__(
        self,
        token_manager: Optional[DhanTokenManager] = None,
        instrument_lookup: Optional[DhanInstrumentLookup] = None,
    ):
        self._token_manager = token_manager or DhanTokenManager()
        self._instruments = instrument_lookup or DhanInstrumentLookup()

    # -- public API ---------------------------------------------------------

    def get_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Last traded price for a list of NSE equity symbols."""
        security_ids = self._instruments.resolve_many(symbols)
        id_to_symbol = {v: k for k, v in security_ids.items()}

        response = self._post(
            "/marketfeed/ltp",
            {NSE_EQ_EXCHANGE_SEGMENT: [int(sid) for sid in security_ids.values()]},
        )
        return self._extract_ltp(response, id_to_symbol)

    def get_historical_daily(
        self, symbol: str, from_date: date, to_date: date,
        security_id: Optional[str] = None, exchange_segment: str = NSE_EQ_EXCHANGE_SEGMENT,
    ) -> pd.DataFrame:
        """
        OHLCV daily candles for one symbol, as a DataFrame indexed by date.
        Defaults to resolving `symbol` on NSE, same as always. Pass an
        explicit `security_id`/`exchange_segment` (e.g. from
        DhanInstrumentLookup.resolve_with_exchange()) for a BSE-only stock
        that has no NSE equity listing at all.
        """
        if security_id is None:
            security_id = self._instruments.resolve(symbol)

        response = self._post(
            "/charts/historical",
            {
                "securityId": security_id,
                "exchangeSegment": exchange_segment,
                "instrument": INSTRUMENT_TYPE_EQUITY,
                "fromDate": from_date.isoformat(),
                "toDate": to_date.isoformat(),
            },
        )
        return self._candles_to_dataframe(response)

    def get_historical_intraday(
        self, symbol: str, from_dt: datetime, to_dt: datetime, interval_minutes: int = 5,
        security_id: Optional[str] = None, exchange_segment: str = NSE_EQ_EXCHANGE_SEGMENT,
    ) -> pd.DataFrame:
        """OHLCV intraday candles. Dhan limits this to 90 days per request."""
        if security_id is None:
            security_id = self._instruments.resolve(symbol)

        response = self._post(
            "/charts/intraday",
            {
                "securityId": security_id,
                "exchangeSegment": exchange_segment,
                "instrument": INSTRUMENT_TYPE_EQUITY,
                "interval": interval_minutes,
                "fromDate": from_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "toDate": to_dt.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        return self._candles_to_dataframe(response)

    def get_historical_daily_batch(
        self, symbols: list[str], from_date: date, to_date: date
    ) -> dict[str, pd.DataFrame]:
        """
        Concurrent, rate-limited fetch of daily history for many symbols.
        Same safe aggregate request rate as a sequential loop (enforced by
        the shared _RateLimiter inside _post()), but multiple symbols'
        network waits overlap instead of stacking serially. Failures for
        individual symbols are logged and skipped rather than aborting the
        whole batch.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=BATCH_FETCH_WORKERS) as executor:
            futures = {
                executor.submit(self.get_historical_daily, symbol, from_date, to_date): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as exc:
                    logger.warning("Historical daily fetch failed for %s: %s", symbol, exc)
        return results

    def get_historical_intraday_batch(
        self, symbols: list[str], from_dt: datetime, to_dt: datetime, interval_minutes: int = 5
    ) -> dict[str, pd.DataFrame]:
        """Concurrent, rate-limited fetch of intraday history for many symbols."""
        results = {}
        with ThreadPoolExecutor(max_workers=BATCH_FETCH_WORKERS) as executor:
            futures = {
                executor.submit(self.get_historical_intraday, symbol, from_dt, to_dt, interval_minutes): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as exc:
                    logger.warning("Historical intraday fetch failed for %s: %s", symbol, exc)
        return results

    def get_option_expiry_list(
        self, security_id: str, exchange_segment: str = DEFAULT_FNO_EXCHANGE_SEGMENT
    ) -> list[str]:
        """Available expiry dates (YYYY-MM-DD) for an underlying's options."""
        response = self._post(
            "/optionchain/expirylist",
            {"UnderlyingScrip": int(security_id), "UnderlyingSeg": exchange_segment},
            rate_limiter=_OPTION_CHAIN_RATE_LIMITER,
        )
        return response.get("data", [])

    def get_option_chain(
        self, security_id: str, expiry: str, exchange_segment: str = DEFAULT_FNO_EXCHANGE_SEGMENT
    ) -> dict:
        """
        Full option chain (OI, greeks, IV, LTP per strike) for one
        underlying + expiry. Returns Dhan's raw response -- shape isn't
        independently verified yet, so callers should inspect `data`
        rather than assume a specific nesting.
        """
        return self._post(
            "/optionchain",
            {
                "UnderlyingScrip": int(security_id),
                "UnderlyingSeg": exchange_segment,
                "Expiry": expiry,
            },
            rate_limiter=_OPTION_CHAIN_RATE_LIMITER,
        )

    # -- internals ------------------------------------------------------

    def _post(self, path: str, body: dict, rate_limiter: "_RateLimiter" = None) -> dict:
        limiter = rate_limiter or _RATE_LIMITER
        auth_retry_used = False
        for attempt in range(1, MAX_429_RETRIES + 1):
            limiter.wait_turn()
            access_token = self._token_manager.get_access_token()
            response = requests.post(
                f"{API_BASE_URL}{path}",
                json=body,
                headers={
                    "access-token": access_token,
                    "client-id": self._token_manager.client_id,
                    "Content-Type": "application/json",
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 429:
                wait_seconds = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "%s rate-limited (attempt %d/%d); waiting %.1fs before retry.",
                    path, attempt, MAX_429_RETRIES, wait_seconds,
                )
                time.sleep(wait_seconds)
                continue
            if not response.ok:
                if not auth_retry_used and self._is_auth_token_error(response):
                    logger.warning(
                        "%s rejected the current access token (%s); forcing a "
                        "fresh PIN+TOTP regeneration and retrying once. This "
                        "usually means another script/process using the same "
                        "Dhan credentials generated a newer token, invalidating "
                        "this one server-side.",
                        path, response.text[:200],
                    )
                    auth_retry_used = True
                    self._token_manager.get_access_token(force_refresh=True)
                    continue
                raise RuntimeError(f"{path} returned HTTP {response.status_code}: {response.text[:300]}")
            return response.json()
        raise RuntimeError(f"{path} still rate-limited after {MAX_429_RETRIES} retries")

    @classmethod
    def _is_auth_token_error(cls, response) -> bool:
        if response.status_code not in (400, 401):
            return False
        try:
            body = response.json()
        except ValueError:
            return False
        return body.get("errorCode") in cls._AUTH_ERROR_CODES

    @staticmethod
    def _extract_ltp(response: dict, id_to_symbol: dict[str, str]) -> dict[str, float]:
        logger.debug("marketfeed/ltp raw response: %s", response)
        result = {}
        segment_data = response.get("data", {}).get(NSE_EQ_EXCHANGE_SEGMENT, {})
        for security_id, payload in segment_data.items():
            symbol = id_to_symbol.get(str(security_id))
            if symbol and isinstance(payload, dict) and "last_price" in payload:
                result[symbol] = float(payload["last_price"])
        if not result:
            logger.warning(
                "Could not extract LTP values from response: %s. "
                "Run with DHAN_LOG_LEVEL=DEBUG and check the raw payload above.",
                response,
            )
        return result

    @staticmethod
    def _candles_to_dataframe(response: dict) -> pd.DataFrame:
        logger.debug("historical data raw response: %s", response)
        data = response.get("data", response)
        try:
            df = pd.DataFrame({
                "Open": data["open"],
                "High": data["high"],
                "Low": data["low"],
                "Close": data["close"],
                "Volume": data["volume"],
            })
            df.index = pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert(IST)
            df.index.name = "Date"
            return df
        except KeyError as exc:
            raise RuntimeError(
                f"Unexpected historical-data response shape (keys: {list(data.keys()) if isinstance(data, dict) else type(data)}). "
                "Adjust _candles_to_dataframe() in dhan_market_data.py to match."
            ) from exc

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
import time
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
# is a proactive pause between calls in a batch loop to mostly avoid ever
# hitting it; the backoff-retry in _post() is the reactive fallback for
# whatever slips through.
MIN_CALL_INTERVAL_SECONDS = 0.35
MAX_429_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.5


class DhanMarketData:
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

    def get_historical_daily(self, symbol: str, from_date: date, to_date: date) -> pd.DataFrame:
        """OHLCV daily candles for one symbol, as a DataFrame indexed by date."""
        security_id = self._instruments.resolve(symbol)

        response = self._post(
            "/charts/historical",
            {
                "securityId": security_id,
                "exchangeSegment": NSE_EQ_EXCHANGE_SEGMENT,
                "instrument": INSTRUMENT_TYPE_EQUITY,
                "fromDate": from_date.isoformat(),
                "toDate": to_date.isoformat(),
            },
        )
        return self._candles_to_dataframe(response)

    def get_historical_intraday(
        self, symbol: str, from_dt: datetime, to_dt: datetime, interval_minutes: int = 5
    ) -> pd.DataFrame:
        """OHLCV intraday candles. Dhan limits this to 90 days per request."""
        security_id = self._instruments.resolve(symbol)

        response = self._post(
            "/charts/intraday",
            {
                "securityId": security_id,
                "exchangeSegment": NSE_EQ_EXCHANGE_SEGMENT,
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
        Sequential, throttled fetch of daily history for many symbols.
        Failures for individual symbols are logged and skipped rather than
        aborting the whole batch (matches this codebase's existing
        tolerant-scan pattern elsewhere).
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_historical_daily(symbol, from_date, to_date)
            except Exception as exc:
                logger.warning("Historical daily fetch failed for %s: %s", symbol, exc)
            time.sleep(MIN_CALL_INTERVAL_SECONDS)
        return results

    def get_historical_intraday_batch(
        self, symbols: list[str], from_dt: datetime, to_dt: datetime, interval_minutes: int = 5
    ) -> dict[str, pd.DataFrame]:
        """Sequential, throttled fetch of intraday history for many symbols."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_historical_intraday(symbol, from_dt, to_dt, interval_minutes)
            except Exception as exc:
                logger.warning("Historical intraday fetch failed for %s: %s", symbol, exc)
            time.sleep(MIN_CALL_INTERVAL_SECONDS)
        return results

    # -- internals ------------------------------------------------------

    def _post(self, path: str, body: dict) -> dict:
        for attempt in range(1, MAX_429_RETRIES + 1):
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
                raise RuntimeError(f"{path} returned HTTP {response.status_code}: {response.text[:300]}")
            return response.json()
        raise RuntimeError(f"{path} still rate-limited after {MAX_429_RETRIES} retries")

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
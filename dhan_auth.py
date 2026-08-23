"""
Dhan API authentication and access-token lifecycle management.

Auth model (per https://dhanhq.co/docs/v2/authentication/):
  - Access tokens are valid for a hard-capped 24 hours (SEBI/exchange rule,
    not a Dhan choice) — no token can be made to live longer than that.
  - RenewToken can extend a *still-active* token by another 24h without
    needing PIN/TOTP again. It fails once the token has actually expired.
  - generateAccessToken (PIN + live TOTP code) mints a brand-new token from
    scratch and is the only way to recover once a token has expired.

Strategy used here: try the cheap renew first, fall back to full
regeneration, and cache the result locally so we don't hit Dhan's auth
servers on every process start.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pyotp
import requests

logger = logging.getLogger("dhan_auth")

# Called directly via REST (not through the dhanhq SDK) because the
# DhanLogin/DhanContext helper classes only exist in newer SDK releases —
# older installs (e.g. dhanhq==1.3.3) don't have them at all. Hitting these
# documented endpoints directly works regardless of SDK version.
AUTH_BASE_URL = "https://auth.dhan.co"
API_BASE_URL = "https://api.dhan.co/v2"
REQUEST_TIMEOUT_SECONDS = 15
TOTP_INTERVAL_SECONDS = 30  # standard TOTP code lifetime; pyotp's default

# Dhan's response has been observed under both of these key styles in
# different doc snapshots. We check both rather than assuming one, and log
# the raw keys at DEBUG on first use so a mismatch is easy to spot.
_TOKEN_KEYS = ("accessToken", "access_token")
_EXPIRY_KEYS = ("expiryTime", "expiry_time")

# Renew before the token is actually dead, so a mid-request expiry never
# happens. RenewToken only works on an active (non-expired) token.
RENEW_BUFFER = timedelta(hours=2)


@dataclass
class DhanToken:
    access_token: str
    expires_at: datetime

    def is_valid(self, buffer: timedelta = timedelta(minutes=5)) -> bool:
        return datetime.now(timezone.utc) < (self.expires_at - buffer)

    def needs_renewal(self, buffer: timedelta = RENEW_BUFFER) -> bool:
        return datetime.now(timezone.utc) >= (self.expires_at - buffer)


class DhanTokenManager:
    """
    Obtains and maintains a valid Dhan access token automatically using
    PIN + TOTP, so no manual daily token copy-paste is required.

    Required environment variables:
        DHAN_CLIENT_ID    - your Dhan client / UCC id
        DHAN_PIN          - your Dhan trading PIN
        DHAN_TOTP_SECRET  - the base32 secret shown when you enabled TOTP
                             on web.dhan.co (NOT a 6-digit code — the
                             underlying secret key used to generate them)
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        pin: Optional[str] = None,
        totp_secret: Optional[str] = None,
        cache_path: Optional[str] = None,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ):
        self.client_id = client_id or os.environ.get("DHAN_CLIENT_ID")
        self.pin = pin or os.environ.get("DHAN_PIN")
        self.totp_secret = totp_secret or os.environ.get("DHAN_TOTP_SECRET")

        if not all([self.client_id, self.pin, self.totp_secret]):
            raise ValueError(
                "DHAN_CLIENT_ID, DHAN_PIN and DHAN_TOTP_SECRET must all be set "
                "(env vars or constructor args). Never hardcode these."
            )

        self.cache_path = Path(
            cache_path or os.environ.get("DHAN_TOKEN_CACHE_PATH", ".dhan_token_cache.json")
        )
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

        # Guards every read/write of self._token. Without this, concurrent
        # callers (e.g. multiple worker threads in a batch fetch) that all
        # see a stale token at the same moment will each independently try
        # to renew/regenerate at once -- racing each other against Dhan's
        # auth endpoint and producing malformed/rejected responses.
        self._lock = threading.Lock()
        self._token: Optional[DhanToken] = self._load_cache()

    # -- public API ---------------------------------------------------

    def get_access_token(self, force_refresh: bool = False) -> str:
        """
        Return a valid access token, refreshing/regenerating as needed.
        Thread-safe: only one caller at a time actually performs a
        renew/regenerate network call. Others block briefly and then reuse
        whatever that call produced, instead of racing their own attempts.
        """
        with self._lock:
            if not force_refresh and self._token and self._token.is_valid():
                if self._token.needs_renewal():
                    self._try_renew()
                return self._token.access_token

            if self._token and not force_refresh:
                if self._try_renew():
                    return self._token.access_token

            self._regenerate()
            return self._token.access_token

    # -- internals ------------------------------------------------------

    def _current_totp(self) -> str:
        return pyotp.TOTP(self.totp_secret).now()

    def _with_retries(self, description: str, fn):
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except Exception as exc:  # network/API errors from the SDK
                last_exc = exc
                logger.warning(
                    "%s failed (attempt %d/%d): %s",
                    description, attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
        raise RuntimeError(f"{description} failed after {self.max_retries} attempts") from last_exc

    def _try_renew(self) -> bool:
        if not self._token:
            return False
        try:
            logger.info("Attempting to renew existing Dhan access token.")
            response = self._with_retries("RenewToken", self._call_renew_token)
            self._apply_response(response)
            logger.info("Token renewed successfully. New expiry: %s", self._token.expires_at)
            return True
        except Exception as exc:
            logger.warning(
                "Token renewal failed (likely already expired); "
                "falling back to full regeneration via PIN+TOTP. Reason: %s", exc
            )
            return False

    def _regenerate(self) -> None:
        """
        Retries here are TOTP-window-aware, not just a short generic
        backoff: a TOTP code is only valid for a fixed ~30s window, so a
        retry that fires quickly reuses the SAME code Dhan may have just
        rejected as a replay. Each retry waits until a genuinely fresh
        code is guaranteed before trying again, rather than hammering the
        auth endpoint with what looks like the same submission repeated.
        """
        logger.info("Generating a new Dhan access token via PIN + TOTP.")
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._call_generate_access_token()
                self._apply_response(response)
                logger.info("New token generated. Expiry: %s", self._token.expires_at)
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "generateAccessToken failed (attempt %d/%d): %s",
                    attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    wait_seconds = self._seconds_until_fresh_totp()
                    logger.info("Waiting %.1fs for a fresh TOTP code before retrying.", wait_seconds)
                    time.sleep(wait_seconds)
        raise RuntimeError(f"generateAccessToken failed after {self.max_retries} attempts") from last_exc

    @staticmethod
    def _seconds_until_fresh_totp(buffer: float = 1.0) -> float:
        elapsed_in_window = time.time() % TOTP_INTERVAL_SECONDS
        return (TOTP_INTERVAL_SECONDS - elapsed_in_window) + buffer

    def _call_generate_access_token(self) -> dict:
        # Documented as a POST with dhanClientId/pin/totp passed as query
        # params: https://dhanhq.co/docs/v2/authentication/
        totp_code = self._current_totp()
        response = requests.post(
            f"{AUTH_BASE_URL}/app/generateAccessToken",
            params={
                "dhanClientId": self.client_id,
                "pin": self.pin,
                "totp": totp_code,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if not response.ok:
            raise RuntimeError(
                f"generateAccessToken returned HTTP {response.status_code}: {response.text[:300]}"
            )
        return response.json()

    def _call_renew_token(self) -> dict:
        response = requests.get(
            f"{API_BASE_URL}/RenewToken",
            headers={
                "access-token": self._token.access_token,
                "dhanClientId": self.client_id,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if not response.ok:
            raise RuntimeError(f"RenewToken returned HTTP {response.status_code}: {response.text[:300]}")
        return response.json()

    def _apply_response(self, response: dict) -> None:
        logger.debug("Auth response: %s", response)

        token = next((response[k] for k in _TOKEN_KEYS if k in response), None)
        if not token:
            raise RuntimeError(
                f"Could not find access token in Dhan response: {response}. "
                "If this shows a real error message/status from Dhan (not just unfamiliar "
                "key names), that's the actual reason auth failed -- check its content "
                "rather than assuming the response format changed."
            )

        expiry_raw = next((response[k] for k in _EXPIRY_KEYS if k in response), None)
        expires_at = self._parse_expiry(expiry_raw)

        self._token = DhanToken(access_token=token, expires_at=expires_at)
        self._save_cache()

    @staticmethod
    def _parse_expiry(expiry_raw) -> datetime:
        if expiry_raw:
            try:
                # Dhan documents expiryTime as an ISO-8601-ish timestamp.
                parsed = datetime.fromisoformat(str(expiry_raw).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                logger.warning("Unparseable expiry '%s' from Dhan; assuming 24h from now.", expiry_raw)
        return datetime.now(timezone.utc) + timedelta(hours=24)

    def _load_cache(self) -> Optional[DhanToken]:
        if not self.cache_path.exists():
            return None
        try:
            data = json.loads(self.cache_path.read_text())
            if data.get("client_id") != self.client_id:
                return None
            return DhanToken(
                access_token=data["access_token"],
                expires_at=datetime.fromisoformat(data["expires_at"]),
            )
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Could not read token cache (%s); ignoring.", exc)
            return None

    def _save_cache(self) -> None:
        payload = {
            "client_id": self.client_id,
            "access_token": self._token.access_token,
            "expires_at": self._token.expires_at.isoformat(),
        }
        self.cache_path.write_text(json.dumps(payload))
        try:
            os.chmod(self.cache_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        except OSError:
            pass

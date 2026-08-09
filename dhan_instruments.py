"""
Resolves NSE trading symbols (e.g. "RELIANCE", "TCS") to the numeric
Dhan security_id required by the market-data APIs.

Source: Dhan's official instrument master, published at
https://dhanhq.co/docs/v2/instruments/ as a downloadable CSV:
    https://images.dhan.co/api-data/api-scrip-master-detailed.csv

Column names in that CSV are matched fuzzily (substring match, not exact
equality) because Dhan has renamed columns before without notice. On first
run, log at DEBUG to confirm which columns were picked for your file.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("dhan_instruments")

SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"

# Candidate substrings for each logical column we need. Matched
# case-insensitively against the CSV's actual header row.
#
# NOTE on the symbol column: Dhan's current CSV has no "TRADING_SYMBOL"
# column. For a cash-equity row, the plain ticker (e.g. "RELIANCE") lives
# in UNDERLYING_SYMBOL, not SYMBOL_NAME (which holds the full company name,
# e.g. "RELIANCE INDUSTRIES LTD") or DISPLAY_NAME. Verified directly
# against the live CSV for RELIANCE (2885), TCS (11536), HDFCBANK (1333).
_SYMBOL_COL_CANDIDATES = ("UNDERLYING_SYMBOL",)
_SECURITY_ID_COL_CANDIDATES = ("SECURITY_ID",)
_EXCH_COL_CANDIDATES = ("EXCH_ID", "EXCH_EXCH_ID", "EXM_EXCH_ID")
_SEGMENT_COL_CANDIDATES = ("SEGMENT",)
_INSTRUMENT_COL_CANDIDATES = ("INSTRUMENT_NAME", "INSTRUMENT")

NSE_EQ_EXCHANGE_SEGMENT = "NSE_EQ"


def _find_column(columns, candidates) -> Optional[str]:
    for col in columns:
        upper = col.upper()
        if any(cand in upper for cand in candidates):
            return col
    return None


class DhanInstrumentLookup:
    def __init__(self, cache_path: str = ".dhan_scrip_master.csv", cache_ttl_hours: float = 24.0):
        self.cache_path = Path(cache_path)
        self.cache_ttl_seconds = cache_ttl_hours * 3600
        self._symbol_to_security_id: dict[str, str] = {}
        self._load()

    def resolve(self, symbol: str) -> str:
        """Return the Dhan security_id for an NSE equity trading symbol."""
        key = symbol.strip().upper()
        try:
            return self._symbol_to_security_id[key]
        except KeyError:
            raise KeyError(
                f"No Dhan security_id found for symbol '{symbol}'. "
                "Check the exact trading symbol matches Dhan's instrument master "
                "(e.g. 'M&M' vs 'M_M', no '.NS' suffix like yfinance uses)."
            ) from None

    def resolve_many(self, symbols: list[str]) -> dict[str, str]:
        resolved, missing = {}, []
        for symbol in symbols:
            try:
                resolved[symbol] = self.resolve(symbol)
            except KeyError:
                missing.append(symbol)
        if missing:
            logger.warning("Could not resolve %d symbols: %s", len(missing), missing)
        return resolved

    # -- internals --------------------------------------------------------

    def _load(self) -> None:
        df = self._get_scrip_master()

        symbol_col = _find_column(df.columns, _SYMBOL_COL_CANDIDATES)
        security_id_col = _find_column(df.columns, _SECURITY_ID_COL_CANDIDATES)
        exch_col = _find_column(df.columns, _EXCH_COL_CANDIDATES)
        segment_col = _find_column(df.columns, _SEGMENT_COL_CANDIDATES)
        instrument_col = _find_column(df.columns, _INSTRUMENT_COL_CANDIDATES)

        missing_cols = [
            name for name, col in [
                ("symbol", symbol_col), ("security_id", security_id_col),
                ("exchange", exch_col), ("segment", segment_col),
            ] if col is None
        ]
        if missing_cols:
            raise RuntimeError(
                f"Could not identify columns {missing_cols} in Dhan's scrip master. "
                f"Actual columns were: {list(df.columns)}. "
                "Dhan likely changed their CSV format — update the *_COL_CANDIDATES "
                "lists in dhan_instruments.py to match."
            )

        logger.debug(
            "Resolved columns -> symbol=%s security_id=%s exchange=%s segment=%s instrument=%s",
            symbol_col, security_id_col, exch_col, segment_col, instrument_col,
        )

        nse_equity = df[
            (df[exch_col].astype(str).str.upper() == "NSE")
            & (df[segment_col].astype(str).str.upper() == "E")  # Equity segment
        ]

        for _, row in nse_equity.iterrows():
            symbol = str(row[symbol_col]).strip().upper()
            security_id = str(row[security_id_col]).strip()
            if symbol and security_id and security_id != "nan":
                self._symbol_to_security_id[symbol] = security_id

        logger.info("Loaded %d NSE equity symbol mappings from Dhan instrument master.", len(self._symbol_to_security_id))

    def _get_scrip_master(self) -> pd.DataFrame:
        if self._is_cache_fresh():
            logger.info("Using cached Dhan instrument master (%s).", self.cache_path)
            return pd.read_csv(self.cache_path, low_memory=False)

        logger.info("Downloading fresh Dhan instrument master from %s", SCRIP_MASTER_URL)
        response = requests.get(SCRIP_MASTER_URL, timeout=30)
        response.raise_for_status()
        self.cache_path.write_bytes(response.content)
        return pd.read_csv(self.cache_path, low_memory=False)

    def _is_cache_fresh(self) -> bool:
        if not self.cache_path.exists():
            return False
        age_seconds = time.time() - self.cache_path.stat().st_mtime
        return age_seconds < self.cache_ttl_seconds
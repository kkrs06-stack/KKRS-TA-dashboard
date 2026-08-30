"""
Drop-in replacement for fetch_all_ohlcv() in pivotboss_dashboard.py, backed
by Dhan instead of yfinance.

Same contract as the original:
    fetch_all_ohlcv_dhan(ticker_list, base_interval) -> dict[symbol, DataFrame]

- ticker_list: yfinance-style tickers as they appear in fo_stock_list.csv
  (e.g. "RELIANCE.NS"). The returned dict is keyed by these SAME strings,
  so nothing else in pivotboss_dashboard.py needs to change.
- base_interval: "1d" (daily, ~1 year lookback) or "60m" (60-day intraday).
- Each DataFrame has Open/High/Low/Close/Volume, IST-indexed, with the same
  cleaning rules as the original (drop NaN/duplicate/zero-volume/flat rows).

To integrate: in pivotboss_dashboard.py, replace
    data_dict = fetch_all_ohlcv(ticker_list, base_interval)
with
    from dhan_pivotboss_adapter import fetch_all_ohlcv_dhan
    data_dict = fetch_all_ohlcv_dhan(ticker_list, base_interval)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytz
import streamlit as st

from dhan_auth import DhanTokenManager
from dhan_instruments import DhanInstrumentLookup
from dhan_market_data import DhanMarketData

IST = pytz.timezone("Asia/Kolkata")

DAILY_LOOKBACK_DAYS = 400  # comfortably covers ~250 trading days/year
INTRADAY_LOOKBACK_DAYS = 60
INTRADAY_INTERVAL_MINUTES = 60

_token_manager = DhanTokenManager()
_lookup = DhanInstrumentLookup()
_market = DhanMarketData(token_manager=_token_manager, instrument_lookup=_lookup)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning rules as the original yfinance-based fetch_all_ohlcv."""
    if df is None or df.empty:
        return df

    df = df.dropna()
    df = df[~df.index.duplicated(keep="first")]

    if "Volume" in df.columns:
        df = df[df["Volume"] > 0]

    df = df[
        ~(
            (df["Open"] == df["High"])
            & (df["High"] == df["Low"])
            & (df["Low"] == df["Close"])
        )
    ]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)

    return df


@st.cache_data(show_spinner="Downloading all symbol data (Dhan)...")
def fetch_all_ohlcv_dhan(ticker_list: list[str], base_interval: str) -> dict[str, pd.DataFrame]:
    plain_symbols = [t.replace(".NS", "") for t in ticker_list]
    ticker_by_plain = dict(zip(plain_symbols, ticker_list))

    if base_interval == "1d":
        from_date = date.today() - timedelta(days=DAILY_LOOKBACK_DAYS)
        to_date = date.today()
        raw = _market.get_historical_daily_batch(plain_symbols, from_date, to_date)
    else:
        from_dt = datetime.now() - timedelta(days=INTRADAY_LOOKBACK_DAYS)
        to_dt = datetime.now()
        raw = _market.get_historical_intraday_batch(
            plain_symbols, from_dt, to_dt, interval_minutes=INTRADAY_INTERVAL_MINUTES
        )

    data_dict = {}
    for plain_symbol, df in raw.items():
        cleaned = _clean(df)
        if cleaned is not None and not cleaned.empty:
            original_ticker = ticker_by_plain[plain_symbol]
            data_dict[original_ticker] = cleaned

    return data_dict
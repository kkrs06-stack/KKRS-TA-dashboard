"""
Ichimoku Cloud scanner - Streamlit tab.

Scans your F&O list for LONG/SHORT signals (per ichimoku_engine.py) using
live Dhan data. Shows every signal from the most recent trading bar (same
"latest trading day, not strictly today" logic as CPR PRO -- works
regardless of when you run it).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import pandas as pd
import pytz
import streamlit as st
from dotenv import load_dotenv

load_dotenv("dhan.env")

from ichimoku_engine import DEFAULT_CONFIG, compute_ichimoku_signals
from dhan_auth import DhanTokenManager
from dhan_instruments import DhanInstrumentLookup
from dhan_market_data import DhanMarketData

IST = pytz.timezone("Asia/Kolkata")

CHART_TF_OPTIONS = ["1 Hour", "4 Hour", "1 Day", "1 Week", "1 Month"]

# Lookback per timeframe: generous enough that rolling 52+52-period
# calculations have real history behind them, not just bare minimum warmup.
DAILY_LOOKBACK_DAYS = 500       # ~2 years, for "1 Day" charts directly
WEEKLY_SOURCE_LOOKBACK_DAYS = 1825   # ~5 years of daily data, resampled to weekly
MONTHLY_SOURCE_LOOKBACK_DAYS = 3650  # ~10 years of daily data, resampled to monthly
INTRADAY_LOOKBACK_DAYS = 90    # Dhan's per-request cap on intraday history
INTRADAY_BASE_INTERVAL_MIN = 60  # Dhan's largest native intraday interval

_token_manager = DhanTokenManager()
_lookup = DhanInstrumentLookup()
_market = DhanMarketData(token_manager=_token_manager, instrument_lookup=_lookup)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.dropna()
    df = df[~df.index.duplicated(keep="first")]
    if "Volume" in df.columns:
        df = df[df["Volume"] > 0]
    df = df[~((df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"]))]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)
    return df


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(rule).agg(agg).dropna()


@st.cache_data(show_spinner="Downloading Ichimoku scan data (Dhan)...", ttl=3600)
def fetch_ohlcv_for_ichimoku(ticker_list: list[str], chart_tf: str) -> dict[str, pd.DataFrame]:
    plain_symbols = [t.replace(".NS", "") for t in ticker_list]
    ticker_by_plain = dict(zip(plain_symbols, ticker_list))

    if chart_tf == "1 Hour":
        from_dt = datetime.now() - timedelta(days=INTRADAY_LOOKBACK_DAYS)
        to_dt = datetime.now()
        raw = _market.get_historical_intraday_batch(
            plain_symbols, from_dt, to_dt, interval_minutes=INTRADAY_BASE_INTERVAL_MIN
        )
        resample_rule = None

    elif chart_tf == "4 Hour":
        from_dt = datetime.now() - timedelta(days=INTRADAY_LOOKBACK_DAYS)
        to_dt = datetime.now()
        raw = _market.get_historical_intraday_batch(
            plain_symbols, from_dt, to_dt, interval_minutes=INTRADAY_BASE_INTERVAL_MIN
        )
        resample_rule = "240min"

    elif chart_tf == "1 Day":
        from_date = date.today() - timedelta(days=DAILY_LOOKBACK_DAYS)
        to_date = date.today()
        raw = _market.get_historical_daily_batch(plain_symbols, from_date, to_date)
        resample_rule = None

    elif chart_tf == "1 Week":
        from_date = date.today() - timedelta(days=WEEKLY_SOURCE_LOOKBACK_DAYS)
        to_date = date.today()
        raw = _market.get_historical_daily_batch(plain_symbols, from_date, to_date)
        resample_rule = "W-FRI"

    else:  # "1 Month"
        from_date = date.today() - timedelta(days=MONTHLY_SOURCE_LOOKBACK_DAYS)
        to_date = date.today()
        raw = _market.get_historical_daily_batch(plain_symbols, from_date, to_date)
        resample_rule = "M"

    data_dict = {}
    for plain_symbol, df in raw.items():
        cleaned = _clean(df)
        if cleaned is None or cleaned.empty:
            continue
        if resample_rule:
            cleaned = _resample(cleaned, resample_rule)
        if not cleaned.empty:
            data_dict[ticker_by_plain[plain_symbol]] = cleaned
    return data_dict


def process_fo_stock_list() -> pd.DataFrame:
    try:
        fo_df = pd.read_csv("fo_stock_list.csv")
        if "lot_size" in fo_df.columns and "lotsize" not in fo_df.columns:
            fo_df = fo_df.rename(columns={"lot_size": "lotsize"})
        return fo_df
    except Exception:
        st.error("Could not read fo_stock_list.csv")
        return pd.DataFrame()


def getfirsttwotext(text) -> str:
    return " ".join(str(text).split()[:2])


def _latest_bar_signals(symbol: str, df: pd.DataFrame, config: dict, lot, name) -> list[dict]:
    try:
        signals_df = compute_ichimoku_signals(df, config)
    except Exception as exc:
        print(f"{symbol}: ERROR in compute_ichimoku_signals -> {exc}")
        return []

    if signals_df.empty:
        return []

    # For intraday chart_tf, treat "latest trading day" (all bars on that
    # date) same as CPR PRO. For Daily/Weekly/Monthly this naturally
    # degenerates to just the single latest bar.
    latest_date = signals_df.index.date.max()
    latest_rows = signals_df[signals_df.index.date == latest_date]
    if latest_rows.empty:
        return []

    occurrences = []
    for ts, row in latest_rows.iterrows():
        for side, flag_col in (("LONG", "LongSignal"), ("SHORT", "ShortSignal")):
            if bool(row[flag_col]):
                occurrences.append({
                    "Symbol": symbol.replace(".NS", ""),
                    "Name": getfirsttwotext(name),
                    "Lot": lot,
                    "Time": ts,
                    "Side": side,
                    "Price": round(float(row["Close"]), 2),
                    "Tenkan": round(float(row["Tenkan"]), 2) if pd.notna(row["Tenkan"]) else "NA",
                    "Kijun": round(float(row["Kijun"]), 2) if pd.notna(row["Kijun"]) else "NA",
                    "CloudTop": round(float(row["CloudTop"]), 2) if pd.notna(row["CloudTop"]) else "NA",
                    "CloudBottom": round(float(row["CloudBottom"]), 2) if pd.notna(row["CloudBottom"]) else "NA",
                    "FutureCloud": "Green" if row["FutureCloudGreen"] else "Red",
                    "Trigger": row["TriggerType"],
                })
    return occurrences


def _process_single_symbol(args):
    symbol, lot, name, data_dict, config = args
    df = data_dict.get(symbol)
    min_bars = config["senkou_b_period"] + config["displacement"] * 2 + 5
    if df is None or len(df) < min_bars:
        return []
    return _latest_bar_signals(symbol, df, config, lot, name)


@st.cache_data(show_spinner="Scanning for Ichimoku signals...", ttl=3600)
def batch_scan_ichimoku(data_dict: dict, fo_df: pd.DataFrame, config: dict) -> tuple[list[dict], list[dict]]:
    args_list = []
    for _, row in fo_df.iterrows():
        symbol = row["symbol"]
        lot = row.get("lotsize", "")
        name = row.get("name", symbol)
        args_list.append((symbol, lot, name, data_dict, config))

    all_occurrences = []
    if not args_list:
        return [], []

    max_workers = min(10, len(args_list))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_symbol, args): args for args in args_list}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    all_occurrences.extend(result)
            except Exception:
                pass

    longs = sorted([o for o in all_occurrences if o["Side"] == "LONG"], key=lambda o: o["Time"], reverse=True)
    shorts = sorted([o for o in all_occurrences if o["Side"] == "SHORT"], key=lambda o: o["Time"], reverse=True)
    return longs, shorts


# =====================================================
# Tile rendering
# =====================================================

def _render_tiles(occurrences: list[dict], side: str):
    if not occurrences:
        st.write("No signals on the latest trading bar.")
        return

    for i in range(0, len(occurrences), 2):
        row_occ = occurrences[i:i + 2]
        cols = st.columns(2)
        for col, occ in zip(cols, row_occ):
            sig_bg = "#18AA47" if side == "LONG" else "#E53935"
            time_str = occ["Time"].strftime("%d-%b %H:%M")
            cloud_color = "#37F553" if occ["FutureCloud"] == "Green" else "#FF3A3A"

            tviewurl = f"https://www.tradingview.com/chart/WGKkLmP8/?symbol=NSE%3A{occ['Symbol']}"

            card_html = f"""
            <div style="background:#252525;border-radius:14px;width:380px;min-height:380px;position:relative;
                        box-shadow:1px 2px 8px #111;margin-bottom:15px;border:1px solid #333;overflow:hidden;padding-bottom:10px">
                <div style="width:100%;text-align:center;padding-top:6px">
                    <a href="{tviewurl}" target="_blank"
                       style="color:#fff;font-size:1.08em;font-weight:700;text-decoration:none">{occ['Name']}</a>
                </div>
                <div style="position:absolute;left:14px;top:6px;font-size:1.10em;
                            background:{sig_bg};color:#fff;padding:2px 8px;border-radius:10px;font-weight:700">
                    {side}
                </div>
                <div style="position:absolute;right:16px;top:6px;font-size:1.10em;color:#ECECEC">
                    Lot <span style="font-weight:bold">{occ['Lot']}</span>
                </div>
                <div style="width:100%;text-align:center;margin-top:24px;margin-bottom:2px">
                    <span style="font-size:1.10em;color:#37F553;font-weight:700">₹{occ['Price']}</span>
                    <span style="font-size:1.10em;color:#FFD700;margin-left:8px">at {time_str}</span>
                </div>
                <div style="width:100%;text-align:center;margin-bottom:6px;font-size:1.10em;color:#ECECEC">
                    {occ['Trigger']}
                </div>
                <div style="width:90%;border-top:1px solid #444;margin:6px auto"></div>
                <div style="width:100%;text-align:center;font-size:1.10em;color:#ECECEC;margin-bottom:4px">
                    Tenkan {occ['Tenkan']} &nbsp; Kijun {occ['Kijun']}
                </div>
                <div style="width:100%;text-align:center;font-size:1.08em;color:#ECECEC;margin-bottom:4px">
                    Cloud {occ['CloudBottom']} - {occ['CloudTop']}
                </div>
                <div style="width:100%;text-align:center;font-size:1.08em;margin-bottom:4px">
                    Future Cloud: <span style="color:{cloud_color};font-weight:700">{occ['FutureCloud']}</span>
                </div>
            </div>
            """
            col.markdown(card_html, unsafe_allow_html=True)


# =====================================================
# Main Streamlit tab
# =====================================================

def run_ichimoku_tab():
    st.title("Ichimoku Cloud Scanner")

    st.sidebar.header("Ichimoku Settings")
    chart_tf = st.sidebar.selectbox("Chart Timeframe", CHART_TF_OPTIONS, index=2)
    lagging_mode = st.sidebar.selectbox(
        "Lagging Line Comparison",
        ["true", "simple"],
        format_func=lambda x: "True-to-chart (52-bar, recommended)" if x == "true" else "Simple proxy (26-bar)",
        index=0,
    )

    with st.sidebar.expander("Advanced Ichimoku Settings"):
        tenkan_period = st.number_input("Tenkan Period", 3, 20, DEFAULT_CONFIG["tenkan_period"])
        kijun_period = st.number_input("Kijun Period", 10, 60, DEFAULT_CONFIG["kijun_period"])
        senkou_b_period = st.number_input("Senkou B Period", 20, 120, DEFAULT_CONFIG["senkou_b_period"])
        displacement = st.number_input("Displacement", 10, 60, DEFAULT_CONFIG["displacement"])

    max_symbols = st.sidebar.slider("Max symbols to scan", 10, 250, 250, 10)

    if st.button(" Refresh Data Cache"):
        fetch_ohlcv_for_ichimoku.clear()
        batch_scan_ichimoku.clear()
        st.session_state.pop("ichimoku_results", None)

    fo_df = process_fo_stock_list()
    if fo_df.empty:
        return
    fo_df = fo_df.iloc[:max_symbols].copy()
    ticker_list = list(fo_df["symbol"])

    data_dict = fetch_ohlcv_for_ichimoku(ticker_list, chart_tf)

    run = st.button("Run Ichimoku Scan")
    if run:
        config = {
            **DEFAULT_CONFIG,
            "lagging_compare_mode": lagging_mode,
            "tenkan_period": tenkan_period,
            "kijun_period": kijun_period,
            "senkou_b_period": senkou_b_period,
            "displacement": displacement,
        }
        st.session_state["ichimoku_results"] = batch_scan_ichimoku(data_dict, fo_df, config)

    # Persisted across tab switches -- see exhaustion_dashboard.py's
    # run_exhaustion_tab() for why this is needed (Streamlit reruns the
    # whole script on every tab switch, and st.button() is only True on
    # the exact rerun it was clicked).
    results = st.session_state.get("ichimoku_results")
    if results is None:
        st.info("Click 'Run Ichimoku Scan' to scan for signals.")
        return
    longs, shorts = results

    total = len(longs) + len(shorts)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Signals (latest bar)", total)
    with col2:
        st.metric("LONG", len(longs))
    with col3:
        st.metric("SHORT", len(shorts))

    st.markdown("---")
    cols = st.columns(2)
    for idx, (title, occurrences) in enumerate(zip(["LONG", "SHORT"], [longs, shorts])):
        with cols[idx]:
            bg = "#18AA47" if idx == 0 else "#E53935"
            st.markdown(
                f"<div style='background:{bg};padding:13px 0;border-radius:13px;margin-bottom:12px;"
                f"text-align:center;width:99%'>"
                f"<span style='color:#FFF;font-size:1.19em;font-weight:700;letter-spacing:2px'>{title}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            _render_tiles(occurrences, title)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    run_ichimoku_tab()

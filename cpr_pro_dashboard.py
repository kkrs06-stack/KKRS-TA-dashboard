"""
PivotBoss CPR PRO scanner - Streamlit tab.

Scans your F&O list for A+ Buy/Sell signals (per cpr_engine.py) using live
Dhan data, and shows EVERY A+ occurrence from the most recent trading day
(not just the latest bar) as one tile each, matching PivotBoss's tile-grid
style. "Most recent trading day" naturally falls back to the last real
session if run on a weekend/holiday or after hours.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import pandas as pd
import pytz
import streamlit as st
from dotenv import load_dotenv

load_dotenv("dhan.env")

from cpr_engine import DEFAULT_CONFIG, compute_cpr_signals
from dhan_auth import DhanTokenManager
from dhan_instruments import DhanInstrumentLookup
from dhan_market_data import DhanMarketData

IST = pytz.timezone("Asia/Kolkata")

DAILY_LOOKBACK_DAYS = 500
INTRADAY_LOOKBACK_DAYS = 60  # within Dhan's 90-day/request cap on intraday history

CHART_TF_OPTIONS = ["1 min", "5 min", "15 min", "25 min", "60 min", "Daily"]
CPR_TF_OPTIONS = ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "9 Months", "12 Months"]

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


@st.cache_data(show_spinner="Downloading CPR scan data (Dhan)...")
def fetch_ohlcv_for_cpr(ticker_list: list[str], chart_tf: str) -> dict[str, pd.DataFrame]:
    plain_symbols = [t.replace(".NS", "") for t in ticker_list]
    ticker_by_plain = dict(zip(plain_symbols, ticker_list))

    if chart_tf == "Daily":
        from_date = date.today() - timedelta(days=DAILY_LOOKBACK_DAYS)
        to_date = date.today()
        raw = _market.get_historical_daily_batch(plain_symbols, from_date, to_date)
    else:
        interval_minutes = int(chart_tf.split()[0])
        from_dt = datetime.now() - timedelta(days=INTRADAY_LOOKBACK_DAYS)
        to_dt = datetime.now()
        raw = _market.get_historical_intraday_batch(plain_symbols, from_dt, to_dt, interval_minutes=interval_minutes)

    data_dict = {}
    for plain_symbol, df in raw.items():
        cleaned = _clean(df)
        if cleaned is not None and not cleaned.empty:
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


DASHBOARD_FIELDS = ["State", "Width", "Relation", "StateShift", "Shift", "Opportunity", "Energy", "Conviction", "MTF", "Chop"]


def _latest_trading_day_aplus_occurrences(symbol: str, df: pd.DataFrame, config: dict, lot, name) -> list[dict]:
    """
    Returns every A+ occurrence from the most recent trading day present in
    the data -- NOT strictly "today". On weekends/holidays (or if run after
    hours), this naturally falls back to the last real trading session,
    since Dhan simply won't have returned bars for non-trading days.
    """
    try:
        signals_df = compute_cpr_signals(df, config)
    except Exception as exc:
        print(f"{symbol}: ERROR in compute_cpr_signals -> {exc}")
        return []

    if signals_df.empty:
        return []

    latest_date = signals_df.index.date.max()
    latest_day_rows = signals_df[signals_df.index.date == latest_date]
    if latest_day_rows.empty:
        return []

    occurrences = []
    for ts, row in latest_day_rows.iterrows():
        for side, flag_col, score_col in (("BUY", "aPlusBuy", "buyScore"), ("SELL", "aPlusSell", "sellScore")):
            if bool(row[flag_col]):
                occ = {
                    "Symbol": symbol.replace(".NS", ""),
                    "Name": getfirsttwotext(name),
                    "Lot": lot,
                    "Time": ts,
                    "Side": side,
                    "Price": round(float(row["Close"]), 2),
                    "Score": int(row[score_col]),
                    "PP": round(float(row["PP"]), 2) if pd.notna(row["PP"]) else "NA",
                    "MID": round(float(row["MID"]), 2) if pd.notna(row["MID"]) else "NA",
                    "PR": round(float(row["PR"]), 2) if pd.notna(row["PR"]) else "NA",
                }
                for field in DASHBOARD_FIELDS:
                    occ[field] = row.get(field, "NA")
                occurrences.append(occ)
    return occurrences


def _process_single_symbol(args):
    symbol, lot, name, data_dict, config = args
    df = data_dict.get(symbol)
    if df is None or len(df) < max(config["trendEMALength"], config["proLookback"]) + 5:
        return []
    return _latest_trading_day_aplus_occurrences(symbol, df, config, lot, name)


@st.cache_data(show_spinner="Scanning for A+ signals...")
def batch_scan_cpr(data_dict: dict, fo_df: pd.DataFrame, config: dict) -> tuple[list[dict], list[dict]]:
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

    buys = sorted([o for o in all_occurrences if o["Side"] == "BUY"], key=lambda o: o["Time"], reverse=True)
    sells = sorted([o for o in all_occurrences if o["Side"] == "SELL"], key=lambda o: o["Time"], reverse=True)
    return buys, sells


# =====================================================
# Tile rendering
# =====================================================

def _render_tiles(occurrences: list[dict], side: str):
    if not occurrences:
        st.write("No A+ signals on the latest trading day.")
        return

    for i in range(0, len(occurrences), 2):
        row_occ = occurrences[i:i + 2]
        cols = st.columns(2)
        for col, occ in zip(cols, row_occ):
            sig_bg = "#18AA47" if side == "BUY" else "#E53935"
            time_str = occ["Time"].strftime("%H:%M")

            enhanced_rows = "".join(
                f'<div style="font-size:0.9em;color:#ECECEC;margin-bottom:2px">{field}: {occ[field]}</div>'
                for field in DASHBOARD_FIELDS
            )

            tviewurl = f"https://www.tradingview.com/chart/?symbol=NSE%3A{occ['Symbol']}"

            card_html = f"""
            <div style="background:#252525;border-radius:14px;width:380px;min-height:420px;position:relative;
                        box-shadow:1px 2px 8px #111;margin-bottom:15px;border:1px solid #333;overflow:hidden;padding-bottom:10px">
                <div style="width:100%;text-align:center;padding-top:6px">
                    <a href="{tviewurl}" target="_blank"
                       style="color:#fff;font-size:1.08em;font-weight:700;text-decoration:none">{occ['Name']}</a>
                </div>
                <div style="position:absolute;left:14px;top:6px;font-size:0.82em;
                            background:{sig_bg};color:#fff;padding:2px 8px;border-radius:10px;font-weight:700">
                    A+ {side}
                </div>
                <div style="position:absolute;right:16px;top:6px;font-size:0.88em;color:#ECECEC">
                    Lot <span style="font-weight:bold">{occ['Lot']}</span>
                </div>
                <div style="width:100%;text-align:center;margin-top:24px;margin-bottom:2px">
                    <span style="font-size:1.10em;color:#37F553;font-weight:700">₹{occ['Price']}</span>
                    <span style="font-size:0.85em;color:#FFD700;margin-left:8px">at {time_str}</span>
                </div>
                <div style="width:100%;text-align:center;margin-bottom:6px">
                    <span style="font-size:0.9em;color:#ECECEC">Score {occ['Score']}/10</span>
                </div>
                <div style="width:90%;border-top:1px solid #444;margin:6px auto"></div>
                <div style="width:100%;text-align:center;font-size:0.85em;color:#ECECEC;margin-bottom:4px">
                    PP {occ['PP']} &nbsp; MID {occ['MID']} &nbsp; PR {occ['PR']}
                </div>
                <div style="width:90%;border-top:1px solid #444;margin:6px auto"></div>
                <div style="width:100%;padding:0 14px">
                    {enhanced_rows}
                </div>
            </div>
            """
            col.markdown(card_html, unsafe_allow_html=True)


# =====================================================
# Main Streamlit tab
# =====================================================

def run_cpr_pro_tab():
    st.title("CPR PRO Scanner")

    st.sidebar.header("CPR PRO Settings")
    cpr_tf = st.sidebar.selectbox("CPR Timeframe", CPR_TF_OPTIONS, index=0)
    chart_tf = st.sidebar.selectbox("Chart Timeframe", CHART_TF_OPTIONS, index=2)
    mtf_tf = st.sidebar.selectbox("MTF CPR Timeframe", CPR_TF_OPTIONS, index=1)
    mtf_mode = st.sidebar.selectbox("MTF Mode", ["Off", "Soft", "Strict"], index=1)

    with st.sidebar.expander("Advanced CPR Settings"):
        cpr_width_lookback = st.number_input("CPR Width Lookback", 5, 20, DEFAULT_CONFIG["cprWidthLookback"])
        narrow_thresh = st.number_input("Narrow CPR Threshold", 0.1, 2.0, DEFAULT_CONFIG["narrowCPRThreshold"], 0.05)
        wide_thresh = st.number_input("Wide CPR Threshold", 0.5, 3.0, DEFAULT_CONFIG["wideCPRThreshold"], 0.05)
        very_wide_thresh = st.number_input("Very Wide CPR Threshold", 0.5, 4.0, DEFAULT_CONFIG["veryWideCPRThreshold"], 0.05)
        trend_ema_len = st.number_input("Trend EMA Length", 5, 200, DEFAULT_CONFIG["trendEMALength"])
        min_score = st.number_input("Minimum Signal Score", 0, 10, DEFAULT_CONFIG["minimumSignalScore"])
        aplus_score = st.number_input("A+ Score Threshold", 0, 10, DEFAULT_CONFIG["aPlusScoreThreshold"])
        vol_mult = st.number_input("Volume Multiplier", 0.5, 3.0, DEFAULT_CONFIG["volumeMultiplier"], 0.1)
        block_chop = st.checkbox("Block Signals in Chop", DEFAULT_CONFIG["blockSignalsInChop"])
        chop_thresh = st.number_input("Chop Score Threshold", 0, 5, DEFAULT_CONFIG["chopScoreThreshold"])

    max_symbols = st.sidebar.slider("Max symbols to scan", 10, 200, 50, 10)

    if st.button(" Refresh Data Cache"):
        fetch_ohlcv_for_cpr.clear()
        batch_scan_cpr.clear()
        st.session_state.pop("cpr_pro_results", None)

    fo_df = process_fo_stock_list()
    if fo_df.empty:
        return
    fo_df = fo_df.iloc[:max_symbols].copy()
    ticker_list = list(fo_df["symbol"])

    data_dict = fetch_ohlcv_for_cpr(ticker_list, chart_tf)

    run = st.button("Run CPR PRO Scan")
    if run:
        config = {
            **DEFAULT_CONFIG,
            "cpr_tf": cpr_tf,
            "mtf_tf": mtf_tf,
            "mtfMode": mtf_mode,
            "cprWidthLookback": cpr_width_lookback,
            "narrowCPRThreshold": narrow_thresh,
            "wideCPRThreshold": wide_thresh,
            "veryWideCPRThreshold": very_wide_thresh,
            "trendEMALength": trend_ema_len,
            "minimumSignalScore": min_score,
            "aPlusScoreThreshold": aplus_score,
            "volumeMultiplier": vol_mult,
            "blockSignalsInChop": block_chop,
            "chopScoreThreshold": chop_thresh,
        }
        st.session_state["cpr_pro_results"] = batch_scan_cpr(data_dict, fo_df, config)

    # Persisted across tab switches -- see exhaustion_dashboard.py's
    # run_exhaustion_tab() for why this is needed (Streamlit reruns the
    # whole script on every tab switch, and st.button() is only True on
    # the exact rerun it was clicked).
    results = st.session_state.get("cpr_pro_results")
    if results is None:
        st.info("Click 'Run CPR PRO Scan' to scan for signals.")
        return
    buys, sells = results

    total = len(buys) + len(sells)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total A+ (latest day)", total)
    with col2:
        st.metric("A+ BUY", len(buys))
    with col3:
        st.metric("A+ SELL", len(sells))

    st.markdown("---")
    cols = st.columns(2)
    for idx, (title, occurrences) in enumerate(zip(["A+ BUY", "A+ SELL"], [buys, sells])):
        with cols[idx]:
            bg = "#18AA47" if idx == 0 else "#E53935"
            st.markdown(
                f"<div style='background:{bg};padding:13px 0;border-radius:13px;margin-bottom:12px;"
                f"text-align:center;width:99%'>"
                f"<span style='color:#FFF;font-size:1.19em;font-weight:700;letter-spacing:2px'>{title}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            _render_tiles(occurrences, "BUY" if idx == 0 else "SELL")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    run_cpr_pro_tab()

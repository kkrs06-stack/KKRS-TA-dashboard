"""
Parabolic Exhaustion scanner - Streamlit tab.

Two-stage design:
  Stage 1 (fast, all F&O symbols): equity-based exhaustion scan.
  Stage 2 (slow, ~3s/call): for symbols that qualify on the latest bar
    only, fetch the option chain and recommend an OTM strike by OI
    concentration.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import pytz
import streamlit as st
from dotenv import load_dotenv

load_dotenv("dhan.env")

from exhaustion_engine import DEFAULT_CONFIG, compute_exhaustion_signals
from option_strike_selector import DEFAULT_OTM_MAX_PCT, DEFAULT_OTM_MIN_PCT, select_otm_strike
from dhan_auth import DhanTokenManager
from dhan_instruments import DhanInstrumentLookup
from dhan_market_data import DhanMarketData

IST = pytz.timezone("Asia/Kolkata")
DAILY_LOOKBACK_DAYS = 250

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


@st.cache_data(show_spinner="Downloading exhaustion scan data (Dhan)...", ttl=3600)
def fetch_ohlcv_for_exhaustion(ticker_list: list[str]) -> dict[str, pd.DataFrame]:
    plain_symbols = [t.replace(".NS", "") for t in ticker_list]
    ticker_by_plain = dict(zip(plain_symbols, ticker_list))

    from_date = date.today() - timedelta(days=DAILY_LOOKBACK_DAYS)
    to_date = date.today()
    raw = _market.get_historical_daily_batch(plain_symbols, from_date, to_date)

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


def _stage1_check(symbol: str, df: pd.DataFrame, config: dict, lot, name) -> list[dict]:
    """
    Looks at the trailing `signal_lookback_bars` bars (default 3 = today
    plus the prior 2 sessions), not just the latest one -- catches a real
    exhaustion signal even if the scan wasn't run on the exact day it
    fired. Returns one candidate per side (CALL/PUT) that had a match
    within the window, using the most recent matching bar for that side.
    """
    try:
        signals = compute_exhaustion_signals(df, config)
    except Exception as exc:
        print(f"{symbol}: ERROR in compute_exhaustion_signals -> {exc}")
        return []

    if signals.empty:
        return []

    lookback_bars = config.get("signal_lookback_bars", 1)
    window = signals.iloc[-lookback_bars:]
    last_pos = len(signals) - 1

    results = []
    for col, side in (("BearishExhaustion", "CALL"), ("BullishExhaustion", "PUT")):
        matches = window[window[col]]
        if matches.empty:
            continue
        signal_ts = matches.index[-1]  # most recent match within the window
        row = signals.loc[signal_ts]
        bars_ago = last_pos - signals.index.get_loc(signal_ts)
        results.append({
            "symbol": symbol.replace(".NS", ""),
            "name": getfirsttwotext(name),
            "lot": lot,
            "time": signal_ts,
            "bars_ago": int(bars_ago),
            "side": side,
            "price": round(float(row["Close"]), 2),
            "pct_change": round(float(row["PctChangeFull"]), 2),
            "rsi": round(float(row["RSI"]), 2),
            "stretch_short": round(float(row["StretchShortPct"]), 2),
            "stretch_long": round(float(row["StretchLongPct"]), 2),
        })
    return results


def _process_single_symbol_stage1(args):
    symbol, lot, name, data_dict, config = args
    df = data_dict.get(symbol)
    min_bars = config["lookback_sessions"] + config["sma_long_period"] + 5
    if df is None or len(df) < min_bars:
        return []
    return _stage1_check(symbol, df, config, lot, name)


@st.cache_data(show_spinner="Scanning for parabolic exhaustion...", ttl=3600)
def batch_scan_stage1(data_dict: dict, fo_df: pd.DataFrame, config: dict) -> list[dict]:
    args_list = []
    for _, row in fo_df.iterrows():
        symbol = row["symbol"]
        lot = row.get("lotsize", "")
        name = row.get("name", symbol)
        args_list.append((symbol, lot, name, data_dict, config))

    candidates = []
    if not args_list:
        return []

    max_workers = min(10, len(args_list))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_symbol_stage1, args): args for args in args_list}
        for future in as_completed(futures):
            try:
                results = future.result()
                candidates.extend(results)
            except Exception:
                pass

    return candidates


def run_stage2(candidates: list[dict], otm_min_pct: float, otm_max_pct: float) -> list[dict]:
    """
    Sequential by necessity -- option chain is rate-limited to 1 request
    per 3 seconds. Only runs for the (typically small) Stage 1 shortlist.
    """
    results = []
    for cand in candidates:
        symbol = cand["symbol"]
        try:
            security_id = _lookup.resolve(symbol)
            expiries = _market.get_option_expiry_list(security_id)
            if not expiries:
                cand["strike_info"] = None
                cand["strike_error"] = "No expiries returned"
                results.append(cand)
                continue

            nearest_expiry = expiries[0]
            chain = _market.get_option_chain(security_id, nearest_expiry)
            chain_data = chain.get("data", {})

            strike_info = select_otm_strike(chain_data, cand["side"], otm_min_pct, otm_max_pct)
            cand["expiry"] = nearest_expiry
            cand["strike_info"] = strike_info
            cand["strike_error"] = None if strike_info else "No liquid OTM strike found in band"
        except Exception as exc:
            cand["strike_info"] = None
            cand["strike_error"] = str(exc)
        results.append(cand)
    return results


# =====================================================
# Tile rendering
# =====================================================

def _render_tiles(candidates: list[dict], side: str):
    if not candidates:
        st.write("No exhaustion candidates on the latest trading day.")
        return

    for i in range(0, len(candidates), 2):
        row_cands = candidates[i:i + 2]
        cols = st.columns(2)
        for col, cand in zip(cols, row_cands):
            sig_bg = "#E53935" if side == "CALL" else "#18AA47"
            side_label = "SELL CALL" if side == "CALL" else "SELL PUT"
            signal_color = "#FFD700" if cand.get("bars_ago", 0) > 0 else "#8FD68F"
            signal_text = (
                f"Signal: Today ({cand['time'].strftime('%d-%b')})"
                if cand.get("bars_ago", 0) == 0
                else f"Signal: {cand['bars_ago']} session(s) ago ({cand['time'].strftime('%d-%b')})"
            )

            strike_info = cand.get("strike_info")
            strike_lines = []
            if strike_info:
                strike_lines.append(
                    f'<div style="width:100%;text-align:center;font-size:1.2em;color:#FFD700;font-weight:700;margin-bottom:3px">{strike_info["wall_label"]} at ₹{strike_info["strike"]:.1f} ({cand["side"]})</div>'
                )
                strike_lines.append(
                    f'<div style="width:100%;text-align:center;font-size:1.2em;color:#ECECEC;margin-bottom:2px">OI: {strike_info["oi"]:,} ({strike_info["oi_trend"]}) &nbsp; {strike_info["pct_away_from_spot"]:+.1f}% OTM</div>'
                )
                strike_lines.append(
                    f'<div style="width:100%;text-align:center;font-size:1.2em;color:#ECECEC;margin-bottom:2px">Delta: {strike_info["delta"]:.2f} &nbsp; IV: {strike_info["implied_volatility"]:.1f}% &nbsp; Premium: ₹{strike_info["last_price"]}</div>'
                )
                if strike_info.get("delta_flag"):
                    strike_lines.append(
                        f'<div style="width:100%;text-align:center;font-size:1.2em;color:#FF3A3A;margin-bottom:2px"> {strike_info["delta_flag"]}</div>'
                    )
            else:
                strike_lines.append(
                    f'<div style="width:100%;text-align:center;font-size:0.92em;color:#FF3A3A">{cand.get("strike_error", "Strike data unavailable")}</div>'
                )
            strike_html = "".join(strike_lines)

            card_lines = [
                '<div style="background:#252525;border-radius:14px;width:380px;min-height:380px;position:relative;box-shadow:1px 2px 8px #111;margin-bottom:15px;border:1px solid #333;overflow:hidden;padding-bottom:8px">',
                f'<div style="width:100%;text-align:center;padding:6px 90px 0 90px;box-sizing:border-box"><span style="color:#fff;font-size:1.15em;font-weight:700">{cand["name"]}</span></div>',
                f'<div style="position:absolute;left:14px;top:6px;font-size:0.8em;background:{sig_bg};color:#fff;padding:2px 9px;border-radius:10px;font-weight:700">{side_label}</div>',
                f'<div style="position:absolute;right:16px;top:6px;font-size:0.92em;color:#ECECEC">Lot <span style="font-weight:bold">{cand["lot"]}</span></div>',
                f'<div style="width:100%;text-align:center;margin-top:22px;margin-bottom:2px"><span style="font-size:1.2em;color:#37F553;font-weight:700">₹{cand["price"]}</span> <span style="font-size:1.2em;color:#FFD700;margin-left:8px">{cand["pct_change"]:+.1f}% / {DEFAULT_CONFIG["lookback_sessions"]}d</span></div>',
                f'<div style="width:100%;text-align:center;font-size:1.2em;color:#ECECEC;margin-bottom:2px">RSI {cand["rsi"]} &nbsp; Stretch(9): {cand["stretch_short"]:+.1f}% &nbsp; Stretch(20): {cand["stretch_long"]:+.1f}%</div>',
                f'<div style="width:100%;text-align:center;font-size:1.2em;color:{signal_color};margin-bottom:5px">{signal_text}</div>',
                '<div style="width:90%;border-top:1px solid #444;margin:5px auto"></div>',
                strike_html,
                '</div>',
            ]
            card_html = "".join(card_lines)
            col.markdown(card_html, unsafe_allow_html=True)


# =====================================================
# Main Streamlit tab
# =====================================================

def run_exhaustion_tab():
    st.title("Parabolic Exhaustion Scanner")
    st.caption("Sell OTM options against stocks showing an exhausted, accelerating move. Screener only -- no order placement.")

    st.sidebar.header("Exhaustion Settings")
    lookback = st.sidebar.number_input("Lookback Sessions", 15, 30, DEFAULT_CONFIG["lookback_sessions"])
    min_move = st.sidebar.number_input("Min Move %", 5.0, 20.0, DEFAULT_CONFIG["min_move_pct"], 0.5)
    signal_lookback_bars = st.sidebar.number_input(
        "Signal Window (sessions, incl. today)",
        1, 5, 3, 1,
        help="Catches a real signal even if the scan wasn't run on the exact day it fired -- "
             "e.g. 3 = today plus the prior 2 sessions.",
    )

    with st.sidebar.expander("Advanced Exhaustion Settings"):
        stretch_long = st.number_input("Stretch from 20-SMA %", 1.0, 25.0, DEFAULT_CONFIG["stretch_long_pct"], 0.5)
        rsi_overbought = st.number_input("RSI Overbought", 60.0, 85.0, DEFAULT_CONFIG["rsi_overbought"], 1.0)
        rsi_oversold = st.number_input("RSI Oversold", 15.0, 40.0, DEFAULT_CONFIG["rsi_oversold"], 1.0)
        weak_close_threshold = st.number_input(
            "Weak-Close Threshold (close within X of the day's low/high)",
            0.1, 0.5, DEFAULT_CONFIG["weak_close_threshold"], 0.05,
        )

        st.markdown("---")
        st.caption(
            "Stretch-9 and Acceleration are shown as context on every candidate "
            "but OFF by default -- both proved unreliable for sustained multi-week "
            "moves during validation. Turn on only if you want to gate signals "
            "on them again."
        )
        require_stretch_short = st.checkbox(
            "Require Stretch from 9-SMA", value=DEFAULT_CONFIG["require_stretch_short"],
        )
        stretch_short = st.number_input(
            "Stretch from 9-SMA %", 1.0, 30.0, DEFAULT_CONFIG["stretch_short_pct"], 0.5,
            disabled=not require_stretch_short,
        )
        require_acceleration = st.checkbox(
            "Require Acceleration", value=DEFAULT_CONFIG["require_acceleration"],
        )
        acceleration_tolerance = st.number_input(
            "Acceleration Tolerance (1.0 = strict steepening, lower = allows some deceleration)",
            0.5, 1.0, DEFAULT_CONFIG["acceleration_tolerance"], 0.05,
            disabled=not require_acceleration,
        )

    st.sidebar.header("Strike Selection")
    otm_min = st.sidebar.number_input("OTM Band Min %", 1.0, 15.0, DEFAULT_OTM_MIN_PCT, 0.5)
    otm_max = st.sidebar.number_input("OTM Band Max %", 2.0, 25.0, DEFAULT_OTM_MAX_PCT, 0.5)

    max_symbols = st.sidebar.slider("Max symbols to scan (Stage 1)", 10, 250, 50, 10)

    if st.button(" Refresh Data Cache"):
        fetch_ohlcv_for_exhaustion.clear()
        batch_scan_stage1.clear()

    fo_df = process_fo_stock_list()
    if fo_df.empty:
        return
    fo_df = fo_df.iloc[:max_symbols].copy()
    ticker_list = list(fo_df["symbol"])

    data_dict = fetch_ohlcv_for_exhaustion(ticker_list)

    run = st.button("Run Exhaustion Scan")
    if not run:
        return

    config = {
        **DEFAULT_CONFIG,
        "lookback_sessions": lookback,
        "min_move_pct": min_move,
        "stretch_short_pct": stretch_short,
        "stretch_long_pct": stretch_long,
        "rsi_overbought": rsi_overbought,
        "rsi_oversold": rsi_oversold,
        "weak_close_threshold": weak_close_threshold,
        "acceleration_tolerance": acceleration_tolerance,
        "require_stretch_short": require_stretch_short,
        "require_acceleration": require_acceleration,
        "signal_lookback_bars": signal_lookback_bars,
    }

    candidates = batch_scan_stage1(data_dict, fo_df, config)

    if not candidates:
        st.info("No parabolic exhaustion candidates found on the latest trading day.")
        return

    st.info(f"Stage 1 found {len(candidates)} candidate(s). Fetching option chains (rate-limited to ~3s each)...")
    with st.spinner(f"Fetching option chains for {len(candidates)} candidate(s)..."):
        candidates = run_stage2(candidates, otm_min, otm_max)

    call_candidates = sorted([c for c in candidates if c["side"] == "CALL"], key=lambda c: c["time"], reverse=True)
    put_candidates = sorted([c for c in candidates if c["side"] == "PUT"], key=lambda c: c["time"], reverse=True)

    total = len(call_candidates) + len(put_candidates)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Candidates", total)
    with col2:
        st.metric("Sell CALL (Bearish Exh.)", len(call_candidates))
    with col3:
        st.metric("Sell PUT (Bullish Exh.)", len(put_candidates))

    st.markdown("---")
    cols = st.columns(2)
    for idx, (title, cands, side) in enumerate(zip(
        ["SELL CALLS", "SELL PUTS"], [call_candidates, put_candidates], ["CALL", "PUT"]
    )):
        with cols[idx]:
            bg = "#E53935" if idx == 0 else "#18AA47"
            st.markdown(
                f"<div style='background:{bg};padding:13px 0;border-radius:13px;margin-bottom:12px;"
                f"text-align:center;width:99%'>"
                f"<span style='color:#FFF;font-size:1.19em;font-weight:700;letter-spacing:2px'>{title}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            _render_tiles(cands, side)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    run_exhaustion_tab()
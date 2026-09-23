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


# =====================================================
# LOOK & FEEL -- same dark "ledger" palette as Ribbon Scanner/Portfolio/
# PivotBoss/Ichimoku, applied to Exhaustion's own subject: a labeled
# "20-SMA to Price" stretch diagram replaces PivotBoss's VWAP ladder,
# since Exhaustion's whole signal is "how far has price stretched from
# its own average," not a band ladder or a cross chart.
#
# Glyph color follows the ACTUAL PRICE DIRECTION (green = price stretched
# up, red = price stretched down) -- NOT the trade's own bias (SELL CALL
# is a bearish-leaning trade taken *because* price stretched up). Coloring
# by trade bias instead was tried first and confused a real review: a
# glyph showing a genuine upward stretch rendered red reads as wrong
# against the "green=up" convention every other tile in this product
# uses. The SELL CALL / SELL PUT badge still carries the trade-bias color
# separately, so nothing is lost by keeping the glyph on price direction.
#
# _CSS below is rendered via st.markdown(unsafe_allow_html=True), not
# st.html() -- it (and the tiles) contain inline SVG, which st.html()
# silently strips. Kept deliberately comment-free and any HTML that gets
# concatenated across many tiles is built as single-line strings -- both
# are load-bearing: a stray comment or a 4-space-indented multi-line
# f-string broke this exact rendering path building Ribbon Scanner. See
# the project_streamlit_html_patterns memory for the full writeup.
#
# No logic changes anywhere in this file -- every engine-facing function
# above this section (fetch/clean/Stage 1/Stage 2) is untouched. Only the
# rendering layer (_render_tiles, run_exhaustion_tab) and new display-only
# helpers below are new.
# =====================================================

_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@1,500;1,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
.ex-wrap{
  --ink:#0e1512; --panel:#141d19; --panel-2:#182220; --line:#26332e; --line-soft:#1d2925;
  --text:#e9ede8; --text-dim:#93a39a; --text-faint:#5f7269;
  --long:#4fd1a0; --short:#e8785f; --fresh:#dcae53; --hold:#7d93c2;
  font-family:"IBM Plex Sans",system-ui,sans-serif;
  color:var(--text);
}
.ex-wrap .eyebrow{
  font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--text-faint);margin:0 0 6px;
}
.ex-wrap h1{
  font-family:"Fraunces",Georgia,serif;font-style:italic;font-weight:500;font-size:32px;
  line-height:1.05;margin:0 0 8px;color:var(--text);
}
.ex-wrap .rule-line{
  font-size:16px;color:var(--text-dim);width:100%;max-width:none;line-height:1.5;margin:0 0 14px;
}
.ex-wrap .rule-line b{color:var(--text);font-weight:600;}
div[data-testid="stMarkdownContainer"] p{max-width:none!important;}
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:#141d19!important;border:1px solid #26332e!important;
  border-radius:3px!important;padding:6px 10px!important;
}
div[data-testid="stButton"] button{
  background:#141d19!important;border:1px solid #26332e!important;
  color:#93a39a!important;font-family:"IBM Plex Sans",sans-serif!important;
}
div[data-testid="stButton"] button:hover{border-color:#dcae53!important;color:#e9ede8!important;}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label{
  font-family:"IBM Plex Mono",monospace!important;font-size:11px!important;
  letter-spacing:.1em;text-transform:uppercase;color:#5f7269!important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div{
  background:#182220!important;border-color:#26332e!important;color:#e9ede8!important;
  font-family:"IBM Plex Sans",sans-serif!important;font-size:13px!important;
}
div[data-testid="stSelectbox"] svg{fill:#93a39a!important;}
div[data-testid="stNumberInput"] input{
  background:#182220!important;border:1px solid #26332e!important;
  color:#e9ede8!important;font-family:"IBM Plex Mono",monospace!important;font-size:13px!important;
}
.ex-wrap .summary{display:flex;align-items:center;padding:14px 20px;background:var(--panel-2);border:1px solid var(--line-soft);border-radius:3px;margin-bottom:22px;font-family:"IBM Plex Mono",monospace;font-size:16px;color:var(--text-dim);flex-wrap:wrap;row-gap:8px;}
.ex-wrap .summary .step{display:flex;align-items:baseline;gap:8px;padding:0 16px;border-right:1px solid var(--line-soft);}
.ex-wrap .summary .step:last-child{border-right:none;}
.ex-wrap .summary .step b{font-size:16px;color:var(--text);font-weight:700;}
.ex-wrap .summary .step .lbl{color:var(--text-faint);font-family:"IBM Plex Sans",sans-serif;font-size:16px;}
.ex-wrap .long-tag{color:var(--long);}
.ex-wrap .short-tag{color:var(--short);}
.ex-wrap .grid-label{font-family:"IBM Plex Mono",monospace;font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--text-faint);margin:0 0 14px;}
.ex-wrap .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px;}
.ex-wrap .tile{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:18px;position:relative;}
.ex-wrap .tile-top{display:flex;justify-content:space-between;align-items:flex-start;}
.ex-wrap .symbol{font-weight:700;font-size:21px;}
.ex-wrap .symbol a{color:inherit;text-decoration:none;border-bottom:1px dotted var(--text-faint);}
.ex-wrap .symbol a:hover{border-bottom-color:var(--fresh);color:var(--fresh);}
.ex-wrap .company{font-size:14px;color:var(--text-faint);margin-top:2px;}
.ex-wrap .dir-badge{font-family:"IBM Plex Mono",monospace;font-size:13px;font-weight:700;letter-spacing:.06em;padding:4px 9px;border-radius:2px;white-space:nowrap;}
.ex-wrap .dir-badge.long{color:var(--long);background:rgba(79,209,160,.12);}
.ex-wrap .dir-badge.short{color:var(--short);background:rgba(232,120,95,.12);}
.ex-wrap .tag-row{display:flex;gap:8px;margin:10px 0 11px;flex-wrap:wrap;}
.ex-wrap .pill{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:13.5px;color:var(--text-dim);border:1px solid var(--line);border-radius:20px;padding:3px 10px;}
.ex-wrap .glyph-box svg{display:block;width:100%;height:66px;}
.ex-wrap .price-row{display:flex;align-items:baseline;gap:10px;font-family:"IBM Plex Mono",monospace;margin:8px 0 12px;font-variant-numeric:tabular-nums;flex-wrap:wrap;}
.ex-wrap .price{font-size:22px;font-weight:700;}
.ex-wrap .price-note{font-size:12.5px;color:var(--text-faint);font-family:"IBM Plex Sans",sans-serif;}
.ex-wrap .chg{font-size:15px;font-weight:600;}
.ex-wrap .chg.up{color:var(--long);}
.ex-wrap .chg.down{color:var(--short);}
.ex-wrap .stat-row{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;font-size:15px;padding:7px 0;border-top:1px solid var(--line-soft);font-variant-numeric:tabular-nums;gap:10px;}
.ex-wrap .stat-row .k{color:var(--text-faint);white-space:nowrap;}
.ex-wrap .stat-row .v{color:var(--text);font-family:"IBM Plex Mono",monospace;font-weight:600;white-space:nowrap;}
.ex-wrap .stat-row .v.long{color:var(--long);}
.ex-wrap .stat-row .v.short{color:var(--short);}
.ex-wrap .stat-row .v.gold{color:var(--fresh);}
.ex-wrap .strike-plan{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px 0;margin-top:10px;padding-top:10px;border-top:1px solid var(--line-soft);}
.ex-wrap .strike-plan div{text-align:center;}
.ex-wrap .strike-plan .sp-label{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--text-faint);font-family:"IBM Plex Mono",monospace;margin-bottom:4px;}
.ex-wrap .strike-plan .sp-value{font-family:"IBM Plex Mono",monospace;font-size:15px;font-weight:600;}
.ex-wrap .wall-note{font-family:"IBM Plex Mono",monospace;font-size:13.5px;color:var(--fresh);margin-top:10px;padding-top:10px;border-top:1px solid var(--line-soft);text-align:center;}
.ex-wrap .warn-note{font-family:"IBM Plex Mono",monospace;font-size:12.5px;color:var(--short);margin-top:8px;display:flex;align-items:center;gap:6px;}
.ex-wrap .error-note{font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--short);margin-top:10px;padding-top:10px;border-top:1px solid var(--line-soft);}
</style>
"""

_HEADER_HTML = """
<div class="ex-wrap">
  <p class="eyebrow">Trading Suite</p>
  <h1>Parabolic Exhaustion</h1>
  <p class="rule-line">
    Screener only &mdash; flags F&amp;O names that made a large, accelerating move and show a same-day
    exhaustion signature, then proposes an OTM strike to sell against the move by open-interest
    concentration. <b>No automated order placement.</b>
  </p>
</div>
"""


def _exhaustion_glyph_svg(stretch_long: float, price: float) -> str:
    up = stretch_long >= 0
    color = "#4fd1a0" if up else "#e8785f"
    try:
        sma20 = price / (1 + stretch_long / 100.0)
    except ZeroDivisionError:
        sma20 = price
    mag = min(max(abs(stretch_long), 5.0), 45.0)
    delta = 6.0 + (mag - 5.0) / 40.0 * 28.0
    if up:
        base_y, price_y = 48, max(48 - delta, 8)
        mid_y = base_y - delta * 0.55
        price_label_y = price_y - 7
    else:
        base_y, price_y = 16, min(16 + delta, 58)
        mid_y = base_y + delta * 0.55
        price_label_y = price_y + 13
    return (
        f'<svg viewBox="0 0 280 66">'
        f'<circle cx="20" cy="{base_y}" r="4" fill="#5f7269"/>'
        f'<text x="30" y="{base_y - 3:.0f}" fill="#5f7269" font-size="10" font-family="IBM Plex Mono">20-SMA</text>'
        f'<text x="30" y="{base_y + 9:.0f}" fill="#5f7269" font-size="9" font-family="IBM Plex Mono">&#8377;{sma20:,.2f}</text>'
        f'<line x1="24" y1="{base_y}" x2="248" y2="{price_y:.0f}" stroke="{color}" stroke-width="1.6" stroke-dasharray="4,3"/>'
        f'<text x="130" y="{mid_y:.0f}" fill="{color}" font-size="11" font-family="IBM Plex Mono" font-weight="600" text-anchor="middle">{stretch_long:+.1f}%</text>'
        f'<circle cx="255" cy="{price_y:.0f}" r="5.5" fill="{color}"/>'
        f'<text x="255" y="{price_label_y:.0f}" fill="{color}" font-size="10" font-family="IBM Plex Mono" text-anchor="end">Price &#8377;{price:,.2f}</text>'
        f'</svg>'
    )


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
            "prev_close": round(float(signals["Close"].iloc[-1]), 2),
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

            # cand["price"] is the Close on the day the signal fired (could be
            # several sessions back within the lookback window) -- this is the
            # live spot from today's option chain, shown separately so the
            # tile never implies the signal-day price is current.
            cand["current_price"] = chain_data.get("last_price")

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

def _ex_tile_html(cand: dict) -> str:
    """Built as one continuous line (no embedded newlines) -- see
    project_streamlit_html_patterns memory for why."""
    side = cand["side"]
    dir_class = "short" if side == "CALL" else "long"
    side_label = "SELL CALL" if side == "CALL" else "SELL PUT"
    symbol_clean = cand["symbol"]

    glyph = _exhaustion_glyph_svg(cand["stretch_long"], cand["price"])
    tag_row = f'<span class="pill">Lot {cand["lot"]}</span>'

    tv_url = f"https://www.tradingview.com/chart/RaPnty9s/?symbol=NSE%3A{symbol_clean}"
    symbol_html = f'<a href="{tv_url}" target="_blank">{symbol_clean}</a>'

    chg_cls = "up" if cand["pct_change"] >= 0 else "down"
    arrow = "&#9650;" if cand["pct_change"] >= 0 else "&#9660;"
    price_row = (
        f'<span class="price">&#8377;{cand["price"]:,.2f}</span>'
        f'<span class="price-note">(at signal)</span>'
        f'<span class="chg {chg_cls}">{arrow} {cand["pct_change"]:+.1f}% / {DEFAULT_CONFIG["lookback_sessions"]}d</span>'
    )

    bars_ago = cand.get("bars_ago", 0)
    if bars_ago == 0:
        signal_html = f'Today &middot; {cand["time"].strftime("%d-%b")}'
        signal_cls = "long"
    else:
        signal_html = f'{bars_ago} session(s) ago &middot; {cand["time"].strftime("%d-%b")}'
        signal_cls = "gold"
        current_price = cand.get("current_price")
        if current_price:
            reference_close = cand.get("prev_close", cand["price"])
            if current_price > reference_close:
                now_cls = "long"
            elif current_price < reference_close:
                now_cls = "short"
            else:
                now_cls = ""
            signal_html += f'&nbsp;&middot;&nbsp;Now: <span class="v {now_cls}">&#8377;{current_price:.2f}</span>'

    strike_info = cand.get("strike_info")
    if strike_info:
        strike_block = (
            f'<div class="strike-plan">'
            f'<div><div class="sp-label">Wall</div><div class="sp-value">{strike_info["wall_label"]}</div></div>'
            f'<div><div class="sp-label">Strike</div><div class="sp-value">{strike_info["strike"]:.1f}</div></div>'
            f'<div><div class="sp-label">OTM</div><div class="sp-value">{strike_info["pct_away_from_spot"]:+.1f}%</div></div>'
            f'<div><div class="sp-label">OI</div><div class="sp-value">{strike_info["oi"]:,}</div></div>'
            f'<div><div class="sp-label">Delta</div><div class="sp-value">{strike_info["delta"]:.2f}</div></div>'
            f'<div><div class="sp-label">IV / Prem.</div><div class="sp-value">{strike_info["implied_volatility"]:.1f}% &middot; {strike_info["last_price"]}</div></div>'
            f'</div>'
            f'<div class="wall-note">{strike_info["wall_label"]} at &#8377;{strike_info["strike"]:.1f} ({side}) &middot; OI {strike_info["oi_trend"]}</div>'
        )
        if strike_info.get("delta_flag"):
            strike_block += f'<div class="warn-note">&#9888; {strike_info["delta_flag"]}</div>'
    else:
        strike_block = f'<div class="error-note">{cand.get("strike_error", "Strike data unavailable")}</div>'

    return (
        f'<div class="tile">'
        f'<div class="tile-top">'
        f'<div><div class="symbol">{symbol_html}</div><div class="company">{cand["name"]}</div></div>'
        f'<span class="dir-badge {dir_class}">{side_label}</span>'
        f'</div>'
        f'<div class="tag-row">{tag_row}</div>'
        f'<div class="glyph-box">{glyph}</div>'
        f'<div class="price-row">{price_row}</div>'
        f'<div class="stat-row"><span class="k">RSI (14)</span><span class="v {dir_class}">{cand["rsi"]}</span></div>'
        f'<div class="stat-row"><span class="k">Stretch (9 / 20)</span><span class="v {dir_class}">{cand["stretch_short"]:+.1f}%&nbsp;&middot;&nbsp;{cand["stretch_long"]:+.1f}%</span></div>'
        f'<div class="stat-row"><span class="k">Signal</span><span class="v {signal_cls}">{signal_html}</span></div>'
        f'{strike_block}'
        f'</div>'
    )


def _render_tiles(candidates: list[dict]):
    if not candidates:
        st.write("No exhaustion candidates on the latest trading day.")
        return

    tiles_html = ""
    for cand in candidates:
        tiles_html += _ex_tile_html(cand)

    # SVG is in these tiles -- st.markdown(unsafe_allow_html=True), not
    # st.html() (which sanitizes <svg> out). tiles_html has no embedded
    # newlines, so this is also safe from the markdown-code-block trap.
    st.markdown(f'<div class="ex-wrap"><div class="grid">{tiles_html}</div></div>', unsafe_allow_html=True)


# =====================================================
# Main Streamlit tab
# =====================================================

def run_exhaustion_tab():
    st.markdown(_CSS + _HEADER_HTML, unsafe_allow_html=True)

    st.sidebar.header("Exhaustion Settings")
    # Lookback Sessions, Min Move %, Signal Window, and Max symbols moved
    # to the headline controls row below (the 4 you actually touch
    # often); OTM Band stays as its own secondary row, still outside any
    # expander (matches its original visibility) -- everything else here
    # is unchanged, same sidebar, same defaults.
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

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            lookback = st.number_input("Lookback Sessions", 15, 30, DEFAULT_CONFIG["lookback_sessions"])
        with col2:
            min_move = st.number_input("Min Move %", 5.0, 20.0, DEFAULT_CONFIG["min_move_pct"], 0.5)
        with col3:
            signal_lookback_bars = st.number_input(
                "Signal Window (sessions, incl. today)",
                1, 5, 3, 1,
                help="Catches a real signal even if the scan wasn't run on the exact day it "
                     "fired -- e.g. 3 = today plus the prior 2 sessions.",
            )
        with col4:
            max_symbols = st.slider("Max symbols to scan (Stage 1)", 10, 250, 250, 10)

    otm_col1, otm_col2 = st.columns(2)
    with otm_col1:
        otm_min = st.number_input("OTM Band Min %", 1.0, 15.0, DEFAULT_OTM_MIN_PCT, 0.5)
    with otm_col2:
        otm_max = st.number_input("OTM Band Max %", 2.0, 25.0, DEFAULT_OTM_MAX_PCT, 0.5)

    refresh_col, run_col = st.columns(2)
    with refresh_col:
        if st.button(" Refresh Data Cache", use_container_width=True):
            fetch_ohlcv_for_exhaustion.clear()
            batch_scan_stage1.clear()
            st.session_state.pop("exhaustion_results", None)

    fo_df = process_fo_stock_list()
    if fo_df.empty:
        return
    fo_df = fo_df.iloc[:max_symbols].copy()
    ticker_list = list(fo_df["symbol"])

    data_dict = fetch_ohlcv_for_exhaustion(ticker_list)

    with run_col:
        run = st.button("Run Exhaustion Scan", use_container_width=True)
    if run:
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

        if candidates:
            st.info(f"Stage 1 found {len(candidates)} candidate(s). Fetching option chains (rate-limited to ~3s each)...")
            with st.spinner(f"Fetching option chains for {len(candidates)} candidate(s)..."):
                candidates = run_stage2(candidates, otm_min, otm_max)

        st.session_state["exhaustion_results"] = candidates

    # Persisted across tab switches -- Streamlit reruns the whole script on
    # every interaction (including changing which strategy tab is selected
    # in the sidebar), and a button's "clicked" state is only True on the
    # exact rerun it was clicked. Without storing the result in
    # session_state, navigating away and back would silently drop the
    # tiles until "Run Exhaustion Scan" was clicked again.
    candidates = st.session_state.get("exhaustion_results")
    if candidates is None:
        st.info("Click 'Run Exhaustion Scan' to scan for candidates.")
        return
    if not candidates:
        st.info("No parabolic exhaustion candidates found on the latest trading day.")
        return

    call_candidates = sorted([c for c in candidates if c["side"] == "CALL"], key=lambda c: c["time"], reverse=True)
    put_candidates = sorted([c for c in candidates if c["side"] == "PUT"], key=lambda c: c["time"], reverse=True)
    all_candidates = sorted(candidates, key=lambda c: c["time"], reverse=True)

    st.markdown(
        f'<div class="ex-wrap"><div class="summary">'
        f'<div class="step"><b>{len(fo_df):,}</b><span class="lbl">F&amp;O universe scanned (Stage 1)</span></div>'
        f'<div class="step"><b class="short-tag">{len(call_candidates)} Sell CALL</b>&nbsp;&middot;&nbsp;'
        f'<b class="long-tag">{len(put_candidates)} Sell PUT</b></div>'
        f'<div class="step"><span class="lbl">{signal_lookback_bars}-session window</span></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ex-wrap"><p class="grid-label">Candidates</p></div>', unsafe_allow_html=True)
    _render_tiles(all_candidates)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    run_exhaustion_tab()

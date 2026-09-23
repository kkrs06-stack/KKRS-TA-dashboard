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


# =====================================================
# LOOK & FEEL -- same dark "ledger" palette as Ribbon Scanner/Portfolio/
# PivotBoss, applied to Ichimoku's own subject: a two-lane "Daily / Lagging"
# cross chart replaces PivotBoss's VWAP ladder, since Ichimoku's trigger is
# about WHICH of those two references crossed already vs is crossing right
# now on this bar (see ichimoku_engine.py's TriggerType), not a band ladder.
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
# above this section (fetch/clean/resample/scan) is untouched. Only the
# rendering layer (_render_tiles, run_ichimoku_tab) and new display-only
# helpers below are new.
# =====================================================

_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@1,500;1,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
.ic-wrap{
  --ink:#0e1512; --panel:#141d19; --panel-2:#182220; --line:#26332e; --line-soft:#1d2925;
  --text:#e9ede8; --text-dim:#93a39a; --text-faint:#5f7269;
  --long:#4fd1a0; --short:#e8785f; --fresh:#dcae53; --hold:#7d93c2;
  font-family:"IBM Plex Sans",system-ui,sans-serif;
  color:var(--text);
}
.ic-wrap .eyebrow{
  font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--text-faint);margin:0 0 6px;
}
.ic-wrap h1{
  font-family:"Fraunces",Georgia,serif;font-style:italic;font-weight:500;font-size:32px;
  line-height:1.05;margin:0 0 8px;color:var(--text);
}
.ic-wrap .rule-line{
  font-size:16px;color:var(--text-dim);width:100%;max-width:none;line-height:1.5;margin:0 0 14px;
}
.ic-wrap .rule-line b{color:var(--text);font-weight:600;}
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
.ic-wrap .summary{display:flex;align-items:center;padding:14px 20px;background:var(--panel-2);border:1px solid var(--line-soft);border-radius:3px;margin-bottom:22px;font-family:"IBM Plex Mono",monospace;font-size:16px;color:var(--text-dim);flex-wrap:wrap;row-gap:8px;}
.ic-wrap .summary .step{display:flex;align-items:baseline;gap:8px;padding:0 16px;border-right:1px solid var(--line-soft);}
.ic-wrap .summary .step:last-child{border-right:none;}
.ic-wrap .summary .step b{font-size:16px;color:var(--text);font-weight:700;}
.ic-wrap .summary .step .lbl{color:var(--text-faint);font-family:"IBM Plex Sans",sans-serif;font-size:16px;}
.ic-wrap .long-tag{color:var(--long);}
.ic-wrap .short-tag{color:var(--short);}
.ic-wrap .grid-label{font-family:"IBM Plex Mono",monospace;font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--text-faint);margin:0 0 14px;}
.ic-wrap .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px;}
.ic-wrap .tile{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:18px;position:relative;}
.ic-wrap .tile-top{display:flex;justify-content:space-between;align-items:flex-start;}
.ic-wrap .symbol{font-weight:700;font-size:21px;}
.ic-wrap .symbol a{color:inherit;text-decoration:none;border-bottom:1px dotted var(--text-faint);}
.ic-wrap .symbol a:hover{border-bottom-color:var(--fresh);color:var(--fresh);}
.ic-wrap .company{font-size:14px;color:var(--text-faint);margin-top:2px;}
.ic-wrap .dir-badge{font-family:"IBM Plex Mono",monospace;font-size:13px;font-weight:700;letter-spacing:.06em;padding:4px 9px;border-radius:2px;white-space:nowrap;}
.ic-wrap .dir-badge.long{color:var(--long);background:rgba(79,209,160,.12);}
.ic-wrap .dir-badge.short{color:var(--short);background:rgba(232,120,95,.12);}
.ic-wrap .tag-row{display:flex;gap:8px;margin:10px 0 11px;flex-wrap:wrap;}
.ic-wrap .pill{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:13.5px;color:var(--text-dim);border:1px solid var(--line);border-radius:20px;padding:3px 10px;}
.ic-wrap .glyph-box svg{display:block;width:100%;height:66px;}
.ic-wrap .price-row{display:flex;align-items:baseline;gap:10px;font-family:"IBM Plex Mono",monospace;margin:8px 0 12px;font-variant-numeric:tabular-nums;flex-wrap:wrap;}
.ic-wrap .price{font-size:22px;font-weight:700;}
.ic-wrap .chg{font-size:15px;font-weight:600;}
.ic-wrap .chg.up{color:var(--long);}
.ic-wrap .chg.down{color:var(--short);}
.ic-wrap .stat-row{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;font-size:15px;padding:7px 0;border-top:1px solid var(--line-soft);font-variant-numeric:tabular-nums;gap:10px;}
.ic-wrap .stat-row .k{color:var(--text-faint);white-space:nowrap;}
.ic-wrap .stat-row .v{color:var(--text);font-family:"IBM Plex Mono",monospace;font-weight:600;white-space:nowrap;}
.ic-wrap .stat-row .v.long{color:var(--long);}
.ic-wrap .stat-row .v.short{color:var(--short);}
</style>
"""

_HEADER_HTML = """
<div class="ic-wrap">
  <p class="eyebrow">Trading Suite</p>
  <h1>Ichimoku Cloud</h1>
  <p class="rule-line">
    Tenkan/Kijun cross confirmed by the daily and lagging-line reference, with the <b>future cloud</b>
    (Senkou A/B, plotted forward) as the trend filter. A <b>LONG</b> needs a green future cloud and
    Tenkan above Kijun; a <b>SHORT</b> needs the reverse &mdash; each tagged with which of the two
    confirmations (daily, lagging) fired the trigger.
  </p>
</div>
"""


def _fresh_flags_from_trigger(trigger: str) -> tuple[bool, bool]:
    """Maps ichimoku_engine.py's TriggerType text to which of the two
    lanes (daily, lagging) is crossing right now on this bar vs already
    crossed earlier. Returns (daily_fresh, lagging_fresh)."""
    if trigger == "Daily Established, Lagging Confirms":
        return False, True
    if trigger == "Lagging Established, Daily Confirms":
        return True, False
    return True, True  # "Double Cross (Daily+Lagging)" -- both fresh


def _ichimoku_lane_svg(lane: str, direction: str, fresh: bool, color: str) -> str:
    y = 20 if lane == "daily" else 48
    label_y = 9 if lane == "daily" else 37
    label_text = "DAILY" if lane == "daily" else "LAGGING"
    caption_y = 4 if lane == "daily" else 63
    up = direction == "LONG"
    near = (y + 7) if up else (y - 7)
    far = (y - 7) if up else (y + 7)
    dash = f'<line x1="10" y1="{y}" x2="270" y2="{y}" stroke="#26332e" stroke-width="1" stroke-dasharray="2,3"/>'
    label = f'<text x="10" y="{label_y}" fill="#5f7269" font-size="9" font-family="IBM Plex Mono" letter-spacing="1">{label_text}</text>'
    if fresh:
        marker = (
            f'<line x1="10" y1="{near}" x2="230" y2="{near}" stroke="{color}" stroke-width="1.6"/>'
            f'<line x1="230" y1="{near}" x2="262" y2="{far}" stroke="{color}" stroke-width="1.6"/>'
            f'<circle cx="262" cy="{far}" r="9" fill="{color}" opacity="0.18"/>'
            f'<circle cx="262" cy="{far}" r="5.2" fill="{color}"/>'
            f'<text x="262" y="{caption_y}" fill="{color}" font-size="9" font-family="IBM Plex Mono" text-anchor="end">now</text>'
        )
    else:
        marker = (
            f'<line x1="10" y1="{near}" x2="90" y2="{near}" stroke="{color}" stroke-width="1.4" opacity="0.5"/>'
            f'<line x1="90" y1="{near}" x2="130" y2="{far}" stroke="{color}" stroke-width="1.4" opacity="0.5"/>'
            f'<line x1="130" y1="{far}" x2="270" y2="{far}" stroke="{color}" stroke-width="1.4" opacity="0.5"/>'
            f'<circle cx="110" cy="{y}" r="3" fill="{color}" opacity="0.75"/>'
            f'<text x="110" y="{caption_y}" fill="#5f7269" font-size="9" font-family="IBM Plex Mono" text-anchor="middle">earlier</text>'
        )
    return dash + label + marker


def _ichimoku_glyph_svg(direction: str, trigger: str) -> str:
    daily_fresh, lagging_fresh = _fresh_flags_from_trigger(trigger)
    color = "#4fd1a0" if direction == "LONG" else "#e8785f"
    daily = _ichimoku_lane_svg("daily", direction, daily_fresh, color)
    lagging = _ichimoku_lane_svg("lagging", direction, lagging_fresh, color)
    return f'<svg viewBox="0 0 280 66">{daily}{lagging}</svg>'


@st.cache_data(show_spinner=False, ttl=45)
def fetch_live_prices(symbols: tuple[str, ...]) -> dict[str, float]:
    """
    Genuinely live prices via /marketfeed/ltp -- unlike the historical
    daily/intraday data used for the chart series and indicators, this
    endpoint isn't subject to Dhan's next-morning EOD publish lag. Only
    called for the small shortlist that already passed the scan.
    """
    if not symbols:
        return {}
    try:
        return _market.get_ltp(list(symbols))
    except Exception as exc:
        print(f"fetch_live_prices failed: {exc}")
        return {}


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
        pos = signals_df.index.get_loc(ts)
        prev_close = float(signals_df["Close"].iloc[pos - 1]) if pos > 0 else float(row["Close"])
        for side, flag_col in (("LONG", "LongSignal"), ("SHORT", "ShortSignal")):
            if bool(row[flag_col]):
                occurrences.append({
                    "Symbol": symbol.replace(".NS", ""),
                    "Name": getfirsttwotext(name),
                    "Lot": lot,
                    "Time": ts,
                    "Side": side,
                    "Price": round(float(row["Close"]), 2),
                    "PrevClose": round(prev_close, 2),
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

def _ic_tile_html(occ: dict, price: float, prev_close: float) -> str:
    """Built as one continuous line (no embedded newlines) -- see
    project_streamlit_html_patterns memory for why."""
    direction = occ["Side"]
    dir_class = direction.lower()
    symbol_clean = occ["Symbol"]

    change_amt = price - prev_close
    change_pct = (change_amt / prev_close * 100) if prev_close else 0.0
    if price > prev_close:
        arrow, price_class = "&#9650;", "up"
    elif price < prev_close:
        arrow, price_class = "&#9660;", "down"
    else:
        arrow, price_class = "", ""
    change_html = f'<span class="chg {price_class}">{arrow} {change_amt:+.2f} ({change_pct:+.2f}%)</span>'

    tv_url = f"https://www.tradingview.com/chart/WGKkLmP8/?symbol=NSE%3A{symbol_clean}"
    symbol_html = f'<a href="{tv_url}" target="_blank">{symbol_clean}</a>'
    price_html = f'<span class="price">&#8377;{price:,.2f}</span>'

    glyph = _ichimoku_glyph_svg(direction, occ["Trigger"])
    tag_row = f'<span class="pill">Lot {occ["Lot"]}</span><span class="pill">{occ["Trigger"]}</span>'

    cloud_color_cls = "long" if occ["FutureCloud"] == "Green" else "short"
    bar_time = occ["Time"].strftime("%d-%b %H:%M")

    body = (
        f'<div class="stat-row"><span class="k">Tenkan (Conversion Line)</span><span class="v">{occ["Tenkan"]}</span></div>'
        f'<div class="stat-row"><span class="k">Kijun (Base Line)</span><span class="v">{occ["Kijun"]}</span></div>'
        f'<div class="stat-row"><span class="k">Cloud</span><span class="v">{occ["CloudBottom"]} &ndash; {occ["CloudTop"]}</span></div>'
        f'<div class="stat-row"><span class="k">Future Cloud</span><span class="v {cloud_color_cls}">{occ["FutureCloud"]}</span></div>'
        f'<div class="stat-row"><span class="k">Bar</span><span class="v">{bar_time}</span></div>'
    )

    return (
        f'<div class="tile">'
        f'<div class="tile-top">'
        f'<div><div class="symbol">{symbol_html}</div><div class="company">{occ["Name"]}</div></div>'
        f'<span class="dir-badge {dir_class}">{direction}</span>'
        f'</div>'
        f'<div class="tag-row">{tag_row}</div>'
        f'<div class="glyph-box">{glyph}</div>'
        f'<div class="price-row">{price_html}{change_html}</div>'
        f'{body}'
        f'</div>'
    )


def _render_tiles(occurrences: list[dict], live_prices=None):
    if not occurrences:
        st.write("No signals on the latest trading bar.")
        return
    live_prices = live_prices or {}

    tiles_html = ""
    for occ in occurrences:
        price = live_prices.get(occ["Symbol"], occ["Price"])
        prev_close = occ.get("PrevClose", price)
        tiles_html += _ic_tile_html(occ, price, prev_close)

    # SVG is in these tiles -- st.markdown(unsafe_allow_html=True), not
    # st.html() (which sanitizes <svg> out). tiles_html has no embedded
    # newlines, so this is also safe from the markdown-code-block trap.
    st.markdown(f'<div class="ic-wrap"><div class="grid">{tiles_html}</div></div>', unsafe_allow_html=True)


# =====================================================
# Main Streamlit tab
# =====================================================

def run_ichimoku_tab():
    st.markdown(_CSS + _HEADER_HTML, unsafe_allow_html=True)

    st.sidebar.header("Ichimoku Settings")
    # Chart Timeframe, Lagging Line Comparison, and Max symbols moved to
    # the headline controls row below (the 3 you actually touch often) --
    # everything else here is unchanged, same sidebar, same defaults.
    with st.sidebar.expander("Advanced Ichimoku Settings"):
        tenkan_period = st.number_input("Tenkan Period", 3, 20, DEFAULT_CONFIG["tenkan_period"])
        kijun_period = st.number_input("Kijun Period", 10, 60, DEFAULT_CONFIG["kijun_period"])
        senkou_b_period = st.number_input("Senkou B Period", 20, 120, DEFAULT_CONFIG["senkou_b_period"])
        displacement = st.number_input("Displacement", 10, 60, DEFAULT_CONFIG["displacement"])

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_tf = st.selectbox("Chart Timeframe", CHART_TF_OPTIONS, index=2)
        with col2:
            lagging_mode = st.selectbox(
                "Lagging Line Comparison",
                ["true", "simple"],
                format_func=lambda x: "True-to-chart (52-bar, recommended)" if x == "true" else "Simple proxy (26-bar)",
                index=0,
            )
        with col3:
            max_symbols = st.slider("Max symbols to scan", 10, 250, 250, 10)

    refresh_col, run_col = st.columns(2)
    with refresh_col:
        if st.button(" Refresh Data Cache", use_container_width=True):
            fetch_ohlcv_for_ichimoku.clear()
            batch_scan_ichimoku.clear()
            st.session_state.pop("ichimoku_results", None)

    fo_df = process_fo_stock_list()
    if fo_df.empty:
        return
    fo_df = fo_df.iloc[:max_symbols].copy()
    ticker_list = list(fo_df["symbol"])

    data_dict = fetch_ohlcv_for_ichimoku(ticker_list, chart_tf)

    with run_col:
        run = st.button("Run Ichimoku Scan", use_container_width=True)
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

    # Persisted across tab switches -- Streamlit reruns the whole script on
    # every interaction (including changing which strategy tab is selected
    # in the sidebar), and a button's "clicked" state is only True on the
    # exact rerun it was clicked. Without storing the result in
    # session_state, navigating away and back would silently drop the
    # tiles until "Run Ichimoku Scan" was clicked again.
    results = st.session_state.get("ichimoku_results")
    if results is None:
        st.info("Click 'Run Ichimoku Scan' to scan for signals.")
        return
    longs, shorts = results

    candidate_symbols = tuple(o["Symbol"] for o in longs + shorts)
    live_prices = fetch_live_prices(candidate_symbols)

    all_signals = sorted(longs + shorts, key=lambda o: o["Time"], reverse=True)
    st.markdown(
        f'<div class="ic-wrap"><div class="summary">'
        f'<div class="step"><b>{len(fo_df):,}</b><span class="lbl">F&amp;O universe scanned</span></div>'
        f'<div class="step"><b class="long-tag">{len(longs)} Long</b>&nbsp;&middot;&nbsp;'
        f'<b class="short-tag">{len(shorts)} Short</b></div>'
        f'<div class="step"><span class="lbl">{chart_tf} chart &middot; latest bar only</span></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ic-wrap"><p class="grid-label">Signals on the latest bar</p></div>', unsafe_allow_html=True)
    _render_tiles(all_signals, live_prices)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    run_ichimoku_tab()

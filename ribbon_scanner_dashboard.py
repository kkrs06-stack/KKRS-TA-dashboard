"""
Ribbon Scanner tab: strict EMA20/50/100 ribbon crossover, confirmed by an
RSI momentum band, volume on the crossover bar, and relative strength vs
Nifty. Scans stock_universe.csv (AMFI market-cap classification, Large/
Mid/Small, NOT restricted to F&O names like the other scanners) -- run
build_stock_universe.py first to generate/refresh that file.

Same Dhan client / live-price / cache-then-rerun conventions as
pivotboss_dashboard_dhan.py, cpr_pro_dashboard.py and ichimoku_dashboard.py
so this drops into master_strategy_dashboard.py the same way they did.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv("dhan.env")

from dhan_auth import DhanTokenManager
from dhan_instruments import DhanInstrumentLookup
from dhan_market_data import DhanMarketData
from ribbon_scanner_engine import (
    DAILY_LOOKBACK_DAYS_BY_TIMEFRAME,
    DEFAULT_CONFIG,
    REJECTION_STAGES,
    RS_LOOKBACK_BARS_BY_TIMEFRAME,
    evaluate_ribbon_diagnostic,
)

UNIVERSE_PATH = "stock_universe.csv"
NIFTY_SECURITY_ID = "13"
NIFTY_EXCHANGE_SEGMENT = "IDX_I"
CAP_TIERS = ["Large Cap", "Mid Cap", "Small Cap"]
TRADINGVIEW_CHART_URL = "https://www.tradingview.com/chart/e2smZmQH/"

_token_manager = DhanTokenManager()
_lookup = DhanInstrumentLookup()
_market = DhanMarketData(token_manager=_token_manager, instrument_lookup=_lookup)


@st.cache_data(show_spinner=False, ttl=45)
def fetch_live_prices(symbols: tuple[str, ...]) -> dict[str, float]:
    """Genuinely live via /marketfeed/ltp -- see the same helper in
    pivotboss_dashboard_dhan.py for why this is kept separate from the
    historical data used to compute the signal itself."""
    if not symbols:
        return {}
    try:
        return _market.get_ltp(list(symbols))
    except Exception as exc:
        print(f"fetch_live_prices failed: {exc}")
        return {}


@st.cache_data(show_spinner=False)
def load_universe() -> pd.DataFrame:
    try:
        return pd.read_csv(UNIVERSE_PATH)
    except FileNotFoundError:
        return pd.DataFrame()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning rules as fetch_all_ohlcv_dhan in dhan_pivotboss_adapter.py."""
    if df is None or df.empty:
        return df
    df = df.dropna()
    df = df[~df.index.duplicated(keep="first")]
    df = df[df["Volume"] > 0]
    df = df[~((df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"]))]
    return df


@st.cache_data(show_spinner="Downloading history for scan universe...", ttl=3600)
def fetch_universe_history(symbols: tuple[str, ...], timeframe: str) -> dict[str, pd.DataFrame]:
    lookback_days = DAILY_LOOKBACK_DAYS_BY_TIMEFRAME[timeframe]
    from_date = date.today() - timedelta(days=lookback_days)
    to_date = date.today()
    raw = _market.get_historical_daily_batch(list(symbols), from_date, to_date)
    return {sym: cleaned for sym, df in raw.items() if (cleaned := _clean(df)) is not None and not cleaned.empty}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_nifty_history(timeframe: str) -> pd.DataFrame:
    lookback_days = DAILY_LOOKBACK_DAYS_BY_TIMEFRAME[timeframe]
    from_date = date.today() - timedelta(days=lookback_days)
    to_date = date.today()
    return _clean(_market.get_historical_daily(
        "NIFTY", from_date, to_date,
        security_id=NIFTY_SECURITY_ID, exchange_segment=NIFTY_EXCHANGE_SEGMENT,
    ))


def _scan_one(symbol, df, nifty_df, timeframe, config):
    try:
        return (symbol, *evaluate_ribbon_diagnostic(df, nifty_df, timeframe, config))
    except Exception as exc:
        print(f"{symbol}: ribbon scan error -> {exc}")
        return symbol, None, "error"


@st.cache_data(show_spinner="Running Ribbon Scanner...")
def run_scan(
    data_dict: dict[str, pd.DataFrame], nifty_df: pd.DataFrame, timeframe: str, config: dict
) -> tuple[list[dict], dict[str, int]]:
    """Returns (results, stage_counts) -- stage_counts is a breakdown of
    exactly which gate every scanned symbol dropped out at (or "passed"),
    so a suspiciously-empty scan is diagnosable from real numbers instead
    of having to guess whether the filters are just strict or something's
    actually broken."""
    results = []
    stage_counts = {stage: 0 for stage in (*REJECTION_STAGES, "passed", "error")}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(_scan_one, symbol, df, nifty_df, timeframe, config)
            for symbol, df in data_dict.items()
        ]
        for future in as_completed(futures):
            symbol, result, stage = future.result()
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            if result is not None:
                result["Symbol"] = symbol
                results.append(result)
    return results, stage_counts


# =====================================================
# LOOK & FEEL -- same visual language as the Portfolio Dashboard's
# technical-status dots (gold ring = "fresh"), applied to this strategy's
# own subject: a literal 3-line EMA ribbon glyph per tile instead of dots.
# Scoped entirely under .ribbon-scanner-wrap so it can't bleed into the
# other tabs' styling when the master dashboard reruns this tab's script.
#
# _CSS below is rendered via st.html(), which (unlike
# st.markdown(unsafe_allow_html=True)) skips markdown parsing entirely --
# but keep it comment-free regardless (same as portfolio_dashboard.py's
# _CSS), since a stray comment here once broke the whole block. Notes on
# the less obvious rules:
#   - stMarkdownContainer p max-width override: Streamlit's own markdown
#     CSS caps paragraph width for readability by default; without
#     !important, whichever stylesheet loads second silently wins and the
#     description stays narrow regardless of what this file says.
#   - stVerticalBlockBorderWrapper: targets st.container(border=True)'s
#     wrapper, for the bordered panel around the controls row. If a given
#     Streamlit version renders a different data-testid here, this rule
#     just won't match -- controls fall back to plain, not broken.
#   - stButton/stNumberInput rules: reskin Streamlit's own native widgets
#     to match this palette. Safe to target globally (not scoped under
#     .ribbon-scanner-wrap) because the master dashboard's dispatch
#     reruns the whole script and calls exactly one run_xxx_tab() per
#     run, so these rules only ever exist in the DOM
#     while Ribbon Scanner is the selected tab.
# =====================================================

_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@1,500;1,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
.ribbon-scanner-wrap{
  --ink:#0e1512; --panel:#141d19; --panel-2:#182220; --line:#26332e; --line-soft:#1d2925;
  --text:#e9ede8; --text-dim:#93a39a; --text-faint:#5f7269;
  --long:#4fd1a0; --long-dim:#2c5f4c; --short:#e8785f; --short-dim:#6b3a30; --fresh:#dcae53;
  font-family:"IBM Plex Sans",system-ui,sans-serif;
  color:var(--text);
}
.ribbon-scanner-wrap .eyebrow{
  font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--text-faint);margin:0 0 6px;
}
.ribbon-scanner-wrap h1{
  font-family:"Fraunces",Georgia,serif;font-style:italic;font-weight:500;font-size:32px;
  line-height:1.05;margin:0 0 8px;color:var(--text);
}
.ribbon-scanner-wrap .rule-line{
  font-size:13px!important;color:var(--text-dim)!important;width:100%!important;
  max-width:none!important;line-height:1.5;margin:0 0 14px;
}
.ribbon-scanner-wrap .rule-line b{color:var(--text)!important;font-weight:600;}
div[data-testid="stMarkdownContainer"] p {
  max-width:none!important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  background:#141d19!important;border:1px solid #26332e!important;
  border-radius:3px!important;padding:6px 10px!important;
}
div[data-testid="stNumberInput"] label {
  font-family:"IBM Plex Mono",monospace!important;font-size:10.5px!important;
  letter-spacing:.1em;text-transform:uppercase;color:#5f7269!important;
}
div[data-testid="stNumberInput"] input {
  font-family:"IBM Plex Sans",sans-serif!important;color:#e9ede8!important;
  background:#182220!important;border:1px solid #26332e!important;
}
.seg-label {
  font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:#5f7269;margin-bottom:8px;
}
.seg-active {
  font-family:"IBM Plex Sans",sans-serif;font-size:14px;font-weight:500;
  text-align:center;color:#e9ede8;background:#22322c;border:1px solid #dcae53;
  border-radius:2px;padding:6px 0;
}
div[data-testid="stButton"] button {
  background:#141d19!important;border:1px solid #26332e!important;
  color:#93a39a!important;font-family:"IBM Plex Sans",sans-serif!important;
}
div[data-testid="stButton"] button:hover {
  border-color:#dcae53!important;color:#e9ede8!important;
}
.ribbon-scanner-wrap .summary{
  display:flex;align-items:center;padding:14px 20px;background:var(--panel-2);
  border:1px solid var(--line-soft);border-radius:3px;margin-bottom:22px;
  font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--text-dim);
  flex-wrap:wrap;row-gap:8px;
}
.ribbon-scanner-wrap .summary .step{display:flex;align-items:baseline;gap:8px;padding:0 16px;border-right:1px solid var(--line-soft);}
.ribbon-scanner-wrap .summary .step:last-child{border-right:none;}
.ribbon-scanner-wrap .summary .step b{font-size:16px;color:var(--text);font-weight:700;}
.ribbon-scanner-wrap .summary .step .lbl{color:var(--text-faint);font-family:"IBM Plex Sans",sans-serif;font-size:16px;}
.ribbon-scanner-wrap .long-tag{color:var(--long);}
.ribbon-scanner-wrap .short-tag{color:var(--short);}
.ribbon-scanner-wrap .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px;}
.ribbon-scanner-wrap .tile{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:21px;position:relative;}
.ribbon-scanner-wrap .tile.fresh{border-color:var(--fresh);box-shadow:0 0 0 1px var(--fresh),0 0 20px -6px rgba(220,174,83,.45);}
.ribbon-scanner-wrap .tile-top{display:flex;justify-content:space-between;align-items:flex-start;}
.ribbon-scanner-wrap .symbol{font-weight:700;font-size:21px;}
.ribbon-scanner-wrap .symbol a{color:inherit;text-decoration:none;border-bottom:1px dotted var(--text-faint);}
.ribbon-scanner-wrap .symbol a:hover{border-bottom-color:var(--fresh);color:var(--fresh);}
.ribbon-scanner-wrap .company{font-size:14px;color:var(--text-faint);margin-top:2px;}
.ribbon-scanner-wrap .dir-badge{font-family:"IBM Plex Mono",monospace;font-size:13px;font-weight:700;letter-spacing:.08em;padding:4px 9px;border-radius:2px;}
.ribbon-scanner-wrap .dir-badge.long{color:var(--long);background:rgba(79,209,160,.12);}
.ribbon-scanner-wrap .dir-badge.short{color:var(--short);background:rgba(232,120,95,.12);}
.ribbon-scanner-wrap .cap-pill{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:13.5px;color:var(--text-dim);border:1px solid var(--line);border-radius:20px;padding:3px 10px;margin:12px 0 13px;}
.ribbon-scanner-wrap .ribbon-box svg{display:block;width:100%;height:62px;}
.ribbon-scanner-wrap .price-row{display:flex;align-items:baseline;gap:10px;font-family:"IBM Plex Mono",monospace;margin:10px 0 15px;font-variant-numeric:tabular-nums;flex-wrap:wrap;}
.ribbon-scanner-wrap .price{font-size:22px;font-weight:700;}
.ribbon-scanner-wrap .chg{font-size:15px;font-weight:600;}
.ribbon-scanner-wrap .chg.up{color:var(--long);}
.ribbon-scanner-wrap .chg.down{color:var(--short);}
.ribbon-scanner-wrap .stat-row{display:flex;justify-content:space-between;align-items:center;font-size:15px;padding:9px 0;border-top:1px solid var(--line-soft);font-variant-numeric:tabular-nums;gap:10px;}
.ribbon-scanner-wrap .stat-row .k{color:var(--text-faint);}
.ribbon-scanner-wrap .stat-row .v{color:var(--text);font-family:"IBM Plex Mono",monospace;font-weight:600;white-space:nowrap;}
.ribbon-scanner-wrap .stat-row .v.long{color:var(--long);}
.ribbon-scanner-wrap .stat-row .v.short{color:var(--short);}
.ribbon-scanner-wrap .rsi-gauge{width:130px;height:16px;position:relative;flex:1;}
.ribbon-scanner-wrap .rsi-track{position:absolute;top:7px;left:0;right:0;height:2px;background:var(--line);}
.ribbon-scanner-wrap .rsi-band{position:absolute;top:7px;height:2px;background:var(--text-faint);opacity:.5;}
.ribbon-scanner-wrap .rsi-mark{position:absolute;top:2px;width:2px;height:13px;background:var(--text);}
.ribbon-scanner-wrap .rsi-mark.long{background:var(--long);}
.ribbon-scanner-wrap .rsi-mark.short{background:var(--short);}
.ribbon-scanner-wrap .fresh-note{font-family:"IBM Plex Mono",monospace;font-size:13.5px;color:var(--fresh);margin-top:12px;padding-top:12px;border-top:1px solid var(--line-soft);display:flex;align-items:center;gap:7px;}
.ribbon-scanner-wrap .fresh-note .ring{width:7px;height:7px;border-radius:50%;background:var(--fresh);box-shadow:0 0 6px var(--fresh);flex-shrink:0;}
</style>
"""

_HEADER_HTML = """
<div class="ribbon-scanner-wrap">
  <p class="eyebrow">Trading Suite &middot; New Strategy</p>
  <h1>Ribbon Scanner</h1>
  <p class="rule-line">
    Scans <b>Large, Mid &amp; Small Cap</b> stocks for a strict EMA ribbon &mdash;
    <b>EMA20 &gt; EMA50 &gt; EMA100</b> for longs, reversed for shorts &mdash; crossed within
    the last 2 bars and still holding, filtered by an <b>RSI momentum band</b>,
    <b>volume confirmation</b> on the crossover bar, and <b>relative strength vs Nifty</b>.
  </p>
</div>
"""


def _ribbon_svg(ema20: list[float], ema50: list[float], ema100: list[float] | None, direction: str) -> str:
    """ema100 is None on Monthly (that leg is dropped there -- see
    ribbon_scanner_engine.py) -- draws a 2-line ribbon in that case rather
    than a fabricated third line."""
    series_list = [ema20, ema50] + ([ema100] if ema100 is not None else [])
    all_vals = [v for series in series_list for v in series]
    lo, hi = min(all_vals), max(all_vals)
    span = (hi - lo) or 1.0
    n = len(ema20)
    x_step = 240 / max(n - 1, 1)

    def _points(series: list[float]) -> str:
        return " ".join(
            f"{10 + i * x_step:.1f},{46 - ((v - lo) / span) * 40:.1f}" for i, v in enumerate(series)
        )

    c1, c2, c3 = ("#4fd1a0", "#3a9c78", "#2c5f4c") if direction == "LONG" else ("#e8785f", "#c15c46", "#6b3a30")
    last_x = 10 + (n - 1) * x_step
    last_y = 46 - ((ema20[-1] - lo) / span) * 40
    slow_line = f'<polyline points="{_points(ema100)}" fill="none" stroke="{c3}" stroke-width="1.6"/>' if ema100 is not None else ""
    return (
        f'<svg viewBox="0 0 260 48">'
        f'<polyline points="{_points(ema20)}" fill="none" stroke="{c1}" stroke-width="2.4"/>'
        f'<polyline points="{_points(ema50)}" fill="none" stroke="{c2}" stroke-width="2"/>'
        f'{slow_line}'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3.4" fill="{c1}"/>'
        f'</svg>'
    )


def _rsi_gauge_html(rsi_val: float, direction: str) -> str:
    band_lo, band_hi = DEFAULT_CONFIG["rsi_long_band"] if direction == "LONG" else DEFAULT_CONFIG["rsi_short_band"]
    marker_pos = max(0.0, min(100.0, rsi_val))
    css_class = "long" if direction == "LONG" else "short"
    return (
        f'<span class="rsi-gauge"><span class="rsi-track"></span>'
        f'<span class="rsi-band" style="left:{band_lo}%;width:{band_hi - band_lo}%;"></span>'
        f'<span class="rsi-mark {css_class}" style="left:{marker_pos}%;"></span></span>'
    )


def _missing(value) -> bool:
    return value is None or pd.isna(value)


def _tile_html(row: dict, live_price: float | None) -> str:
    symbol = row["Symbol"]
    direction = row["Direction"]
    dir_class = "long" if direction == "LONG" else "short"
    company = row.get("CompanyName") or symbol
    category = row.get("Category") or ""
    market_cap = row.get("MarketCapCr")
    cap_html = f"{category} &middot; &#8377;{market_cap:,.0f} Cr" if category and not _missing(market_cap) else (category or "")

    price = live_price if not _missing(live_price) else row.get("LastClose")
    prev_close = row.get("PrevClose")
    if _missing(price) or _missing(prev_close):
        change_html = ""
    else:
        change_amt = price - prev_close
        change_pct = (change_amt / prev_close * 100) if prev_close else 0.0
        if price > prev_close:
            arrow, price_class = "&#9650;", "up"
        elif price < prev_close:
            arrow, price_class = "&#9660;", "down"
        else:
            arrow, price_class = "", ""
        change_html = f'<span class="chg {price_class}">{arrow} {change_amt:+.2f} ({change_pct:+.2f}%)</span>'

    ribbon = _ribbon_svg(row["EMA20Series"], row["EMA50Series"], row.get("EMA100Series"), direction)
    rsi_gauge = _rsi_gauge_html(row["RSI"], direction)
    fresh = row["BarsSinceCrossover"] == 0
    rs_class = "long" if row["RelativeStrengthPct"] >= 0 else "short"

    if fresh:
        footer = '<div class="fresh-note"><span class="ring"></span>Just crossed this bar &mdash; still holding</div>'
    else:
        footer = (
            f'<div class="stat-row"><span class="k">Crossover age</span>'
            f'<span class="v">{row["BarsSinceCrossover"]} bar(s) ago</span></div>'
        )

    price_html = f'<span class="price">&#8377;{price:,.2f}</span>' if not _missing(price) else ""
    tview_url = f"{TRADINGVIEW_CHART_URL}?symbol=NSE%3A{symbol}"
    symbol_html = f'<a href="{tview_url}" target="_blank">{symbol}</a>'

    # Deliberately built as ONE line, not a nicely-indented multi-line
    # f-string: st.markdown() runs content through markdown parsing before
    # its "allow HTML" pass, and markdown treats any line indented 4+
    # spaces as a preformatted code block -- rendered as literal text, not
    # parsed HTML. A tidy multi-line triple-quoted string here (this
    # function's natural Python indentation is well over 4 spaces) hits
    # that rule the moment more than one tile's HTML gets concatenated
    # together, which is exactly what happened the first time this was
    # written that way.
    return (
        f'<div class="tile{" fresh" if fresh else ""}">'
        f'<div class="tile-top">'
        f'<div><div class="symbol">{symbol_html}</div><div class="company">{company}</div></div>'
        f'<span class="dir-badge {dir_class}">{direction}</span>'
        f'</div>'
        f'<span class="cap-pill">{cap_html}</span>'
        f'<div class="ribbon-box">{ribbon}</div>'
        f'<div class="price-row">{price_html}{change_html}</div>'
        f'<div class="stat-row"><span class="k">RSI (14)</span>{rsi_gauge}<span class="v {dir_class}">{row["RSI"]:.1f}</span></div>'
        f'<div class="stat-row"><span class="k">Volume, crossover bar</span><span class="v long">{row["VolumeRatio"]:.1f}&times; avg &#10003;</span></div>'
        f'<div class="stat-row"><span class="k">Rel. strength vs Nifty</span><span class="v {rs_class}">{row["RelativeStrengthPct"]:+.1f}%</span></div>'
        f'{footer}'
        f'</div>'
    )


def _summary_html(universe_count, filtered_count, fetched_count, matched_count, long_count, short_count, timeframe) -> str:
    # fetched_count vs filtered_count is the key diagnostic for a
    # zero-results run: if it's far below filtered_count, the scan itself
    # never even saw most symbols' data (an auth/rate-limit failure, not
    # "no signals today") -- shown in an alarm color so that's obvious at
    # a glance rather than something you have to go check logs to notice.
    fetch_class = "short-tag" if fetched_count < filtered_count * 0.9 else ""
    return f"""
    <div class="ribbon-scanner-wrap">
      <div class="summary">
        <div class="step"><b>{universe_count:,}</b><span class="lbl">AMFI universe</span></div>
        <div class="step"><b>{filtered_count:,}</b><span class="lbl">match cap-tier + floor filters</span></div>
        <div class="step"><b class="{fetch_class}">{fetched_count:,}</b><span class="lbl">had usable price history</span></div>
        <div class="step"><b>{matched_count:,}</b><span class="lbl">pass full scan</span></div>
        <div class="step"><b class="long-tag">{long_count} Long</b>&nbsp;&middot;&nbsp;<b class="short-tag">{short_count} Short</b></div>
        <div class="step"><span class="lbl">{timeframe} timeframe</span></div>
      </div>
    </div>
    """


def _segmented_control(label: str, options: list[str], session_key: str, default: str) -> str:
    """Same pattern master_strategy_dashboard.py already uses for its
    strategy-tile picker: a real st.button() for each inactive option (so
    it's genuinely clickable), custom-styled st.html() standing in for the
    active one -- full control over the active look without fighting a
    native widget's internal (checked-state) styling at all. Wrapped in
    its own bordered container with a tight column gap so the options
    read as one continuous strip (the mockup's look) rather than several
    separately-bordered boxes with visible gaps between them."""
    if session_key not in st.session_state:
        st.session_state[session_key] = default
    st.html(f'<div class="seg-label">{label}</div>')
    with st.container(border=True):
        cols = st.columns(len(options), gap="small")
        for col, option in zip(cols, options):
            if st.session_state[session_key] == option:
                with col:
                    st.html(f'<div class="seg-active">{option}</div>')
            else:
                with col:
                    if st.button(option, key=f"{session_key}_{option}", use_container_width=True):
                        st.session_state[session_key] = option
                        st.rerun()
    return st.session_state[session_key]


def _multi_toggle_chips(label: str, options: list[str], session_key: str, default: list[str]) -> list[str]:
    """Cap Tiers in the mockup are simple always-visible toggle chips with
    a dot indicator, not a dropdown multiselect -- this replicates that:
    every option stays clickable (unlike _segmented_control's active
    option, since more than one chip can be on at once), with an on/off
    dot baked into the button's own label text since individual buttons
    can't be given distinct CSS per-instance."""
    if session_key not in st.session_state:
        st.session_state[session_key] = list(default)
    st.html(f'<div class="seg-label">{label}</div>')
    with st.container(border=True):
        cols = st.columns(len(options), gap="small")
        for col, option in zip(cols, options):
            is_on = option in st.session_state[session_key]
            with col:
                dot = "●" if is_on else "○"  # filled vs hollow circle
                if st.button(f"{dot} {option}", key=f"{session_key}_{option}", use_container_width=True):
                    if is_on:
                        st.session_state[session_key].remove(option)
                    else:
                        st.session_state[session_key].append(option)
                    st.rerun()
    return st.session_state[session_key]


def run_ribbon_scanner_tab():
    st.html(_CSS + _HEADER_HTML)

    universe = load_universe()
    if universe.empty:
        st.error(
            "stock_universe.csv not found or empty. Run `python build_stock_universe.py` "
            "first to generate it from AMFI's market-cap classification."
        )
        return

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            timeframe = _segmented_control("Timeframe", ["Daily", "Weekly", "Monthly"], "ribbon_timeframe_ctrl", "Monthly")
        with col2:
            direction_filter = _segmented_control("Direction", ["Both", "Long", "Short"], "ribbon_direction_ctrl", "Both")
        with col3:
            cap_tiers = _multi_toggle_chips("Cap tiers", CAP_TIERS, "ribbon_captiers_ctrl", CAP_TIERS)
        with col4:
            min_market_cap = st.number_input("Min market cap (Rs Cr)", min_value=0, value=3000, step=500)

    # Unlike PivotBoss/CPR PRO/Ichimoku (whose whole universe IS ~250 F&O
    # names), Ribbon Scanner's universe is the full AMFI market-cap list --
    # capping this at 250 the same way silently limits every scan to the
    # top 250 by market cap and never reaches the rest.
    # This is the only sidebar-worthy setting Ribbon Scanner has, so it
    # lives next to the action buttons instead of alone in a whole sidebar.
    slider_col, refresh_col, run_col = st.columns([3, 1, 1])
    with slider_col:
        max_symbols = st.slider("Max symbols to scan", 10, len(universe), min(len(universe), 1000), 10)
    with refresh_col:
        if st.button(" Refresh Data Cache", use_container_width=True):
            fetch_universe_history.clear()
            fetch_nifty_history.clear()
            run_scan.clear()
            st.session_state.pop("ribbon_results", None)
    with run_col:
        run = st.button("Run Ribbon Scan", use_container_width=True)

    filtered_universe = (
        universe[universe["category"].isin(cap_tiers) & (universe["market_cap_cr"] >= min_market_cap)]
        .sort_values("market_cap_cr", ascending=False)
        .head(max_symbols)
    )

    if run:
        symbols = tuple(filtered_universe["symbol"])
        data_dict = fetch_universe_history(symbols, timeframe)
        nifty_df = fetch_nifty_history(timeframe)
        config = {**DEFAULT_CONFIG, "rs_lookback_bars": RS_LOOKBACK_BARS_BY_TIMEFRAME[timeframe]}
        results, stage_counts = run_scan(data_dict, nifty_df, timeframe, config)

        meta = filtered_universe.set_index("symbol")
        for r in results:
            info = meta.loc[r["Symbol"]] if r["Symbol"] in meta.index else None
            r["CompanyName"] = info["company_name"] if info is not None else r["Symbol"]
            r["Category"] = info["category"] if info is not None else ""
            r["MarketCapCr"] = float(info["market_cap_cr"]) if info is not None else None

        st.session_state["ribbon_results"] = results
        st.session_state["ribbon_stage_counts"] = stage_counts
        st.session_state["ribbon_universe_count"] = len(universe)
        st.session_state["ribbon_filtered_count"] = len(filtered_universe)
        st.session_state["ribbon_fetched_count"] = len(data_dict)
        st.session_state["ribbon_timeframe"] = timeframe

    # Persisted across tab switches -- same reasoning as pivotboss_results:
    # Streamlit reruns the whole script on every interaction, so without
    # session_state the tiles would vanish the instant you touched any
    # other control after running the scan.
    results = st.session_state.get("ribbon_results")
    if results is None:
        st.info("Click 'Run Ribbon Scan' to scan the universe.")
        return

    display_results = results
    if direction_filter != "Both":
        display_results = [r for r in results if r["Direction"].upper() == direction_filter.upper()]

    candidate_symbols = tuple(r["Symbol"] for r in display_results)
    live_prices = fetch_live_prices(candidate_symbols)

    long_count = sum(1 for r in results if r["Direction"] == "LONG")
    short_count = sum(1 for r in results if r["Direction"] == "SHORT")

    st.html(
        _summary_html(
            st.session_state.get("ribbon_universe_count", len(universe)),
            st.session_state.get("ribbon_filtered_count", len(filtered_universe)),
            st.session_state.get("ribbon_fetched_count", len(filtered_universe)),
            len(results), long_count, short_count,
            st.session_state.get("ribbon_timeframe", timeframe),
        )
    )

    fetched = st.session_state.get("ribbon_fetched_count", len(filtered_universe))
    requested = st.session_state.get("ribbon_filtered_count", len(filtered_universe))
    if fetched < requested * 0.9:
        st.warning(
            f"Only {fetched} of {requested} symbols returned usable price history this run -- "
            "0 (or very few) signals below is most likely a data-fetch problem (Dhan auth/rate-limit), "
            "not \"no stocks currently qualify.\" Check the terminal for auth/rate-limit warnings, then "
            "try 'Run Ribbon Scan' again."
        )

    stage_counts = st.session_state.get("ribbon_stage_counts")
    if stage_counts:
        stage_labels = {
            "insufficient_history": "Not enough price history",
            "no_ribbon_order": "EMA ribbon not in order",
            "no_recent_crossover": "No fresh EMA20 crossover",
            "rsi_out_of_band": "RSI outside the momentum band",
            "volume_not_confirmed": "Crossover-bar volume too low",
            "insufficient_rs_history": "Not enough history for Nifty comparison",
            "wrong_relative_strength": "Underperforming/outperforming Nifty the wrong way",
            "passed": "Passed every gate",
            "error": "Errored during evaluation",
        }
        with st.expander("Why did stocks drop out? (rejection breakdown)"):
            for stage in (*REJECTION_STAGES, "passed", "error"):
                count = stage_counts.get(stage, 0)
                if count:
                    st.write(f"**{count}** — {stage_labels.get(stage, stage)}")

    if not display_results:
        st.write("No stocks matched.")
        return

    tiles_html = "".join(_tile_html(r, live_prices.get(r["Symbol"])) for r in display_results)
    # st.html() sanitizes <svg> out of its content (a default security
    # measure against script-bearing inline SVG) -- the ribbon glyph is
    # SVG, so it silently vanished when this was switched to st.html()
    # earlier. st.markdown(unsafe_allow_html=True) doesn't sanitize SVG,
    # and this particular string has no comments in it (that was the
    # actual bug in the CSS block, now fixed there directly), so it's
    # safe to use here.
    st.markdown(f'<div class="ribbon-scanner-wrap"><div class="grid">{tiles_html}</div></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    run_ribbon_scanner_tab()
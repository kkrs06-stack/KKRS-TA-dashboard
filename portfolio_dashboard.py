"""
Institutional-grade portfolio dashboard: household summary + per-person
drill-down across the Kamlesh and Ridhi accounts.

Data sources (append-only transaction logs -- this is the source of
truth; open lots are DERIVED via FIFO, not hand-maintained):
    portfolio_kamlesh_transactions.csv, portfolio_ridhi_transactions.csv
        columns: symbol, company_name, date, action (BUY/SELL), quantity,
        rate, value. Add a new row for every buy or sell -- never edit an
        existing row's quantity by hand.
    portfolio_sector_map.csv -- symbol -> sector

Everything else (current price, daily history for technical signals /
drawdown, Nifty 50 benchmark) is pulled live from Dhan.

Rendering is custom HTML/CSS (a "ledger" visual language -- ink ground,
gold/jade/rose accents, Fraunces for display numbers, IBM Plex Mono for
tabular figures) rather than default Streamlit widgets, for information
density and readability at a glance.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd
import pytz
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv("dhan.env")

from dhan_auth import DhanTokenManager
from dhan_instruments import DhanInstrumentLookup
from dhan_market_data import DhanMarketData
from portfolio_engine import (
    NIFTY_EXCHANGE_SEGMENT,
    NIFTY_SECURITY_ID,
    compute_benchmark_xirr,
    compute_max_drawdown,
    compute_per_stock_benchmark_alpha,
    compute_portfolio_summary,
    compute_stock_metrics,
    compute_technical_signals,
    load_sector_map,
    load_transactions,
    resolve_open_lots_fifo,
    xirr,
)

IST = pytz.timezone("Asia/Kolkata")
DAILY_LOOKBACK_DAYS = 365 * 4  # comfortably covers monthly SMA20 (needs 20+ months) with margin
RECENT_POSITION_DAYS = 90  # XIRR is noisy/misleading below this holding period
TREEMAP_TOP_N = 20
TRADINGVIEW_CHART_ID = "RaPnty9s"  # same chart used across PivotBoss/CPR PRO/Ichimoku/Exhaustion

ACCOUNTS = {
    "Kamlesh": "portfolio_kamlesh_transactions.csv",
    "Ridhi": "portfolio_ridhi_transactions.csv",
}
SECTOR_MAP_PATH = "portfolio_sector_map.csv"

_token_manager = DhanTokenManager()
_lookup = DhanInstrumentLookup()
_market = DhanMarketData(token_manager=_token_manager, instrument_lookup=_lookup)


# =====================================================
# Design tokens (ported from the approved mockup)
# =====================================================

_CSS = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root{
    --ink:#12181B; --surface:#1A2226; --raised:#212B30; --line:#2B363B;
    --gold:#C9A24B; --jade:#4FAE8C; --rose:#C9636B; --amber:#D0A44C;
    --ivory:#EDE8DD; --stone:#8B9297; --stone-dim:#5B6266;
    --font-display:'Fraunces',serif; --font-sans:'IBM Plex Sans',sans-serif; --font-mono:'IBM Plex Mono',monospace;
  }
  .pl-num{font-family:var(--font-mono);font-variant-numeric:tabular-nums;}
  .pl-up{color:var(--jade);} .pl-down{color:var(--rose);}
  .pl-kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:18px;}
  .pl-kpi{background:var(--surface);padding:16px 18px;}
  .pl-kpi-label{font-family:var(--font-sans);font-size:13.8px;letter-spacing:.06em;text-transform:uppercase;color:var(--stone);margin-bottom:9px;}
  .pl-kpi-value{font-family:var(--font-display);font-size:30px;font-weight:500;color:var(--ivory);font-variant-numeric:tabular-nums;}
  .pl-kpi-delta{font-family:var(--font-mono);font-size:16.2px;margin-top:5px;}
  .pl-kpi-sub{font-family:var(--font-sans);font-size:14.4px;color:var(--stone-dim);margin-top:4px;}
  .pl-ribbon-card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin-bottom:18px;}
  .pl-ribbon-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;font-family:var(--font-sans);}
  .pl-ribbon-head h3{font-weight:500;font-size:15px;letter-spacing:.04em;text-transform:uppercase;color:var(--stone);margin:0;}
  .pl-ribbon-head span{font-size:14.4px;color:var(--stone-dim);}
  .pl-ribbon{display:flex;height:12px;border-radius:6px;overflow:hidden;margin-bottom:9px;}
  .pl-ribbon-legend{display:flex;flex-wrap:wrap;gap:12px 18px;font-family:var(--font-sans);font-size:14.4px;color:var(--stone);}
  .pl-ribbon-legend span{display:inline-flex;align-items:center;gap:6px;}
  .pl-swatch{width:8px;height:8px;border-radius:2px;display:inline-block;}
  .pl-table-card{background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:8px;}
  .pl-table-scroll{overflow-x:auto;}
  .pl-table{width:100%;border-collapse:collapse;font-size:16.2px;min-width:1040px;font-family:var(--font-sans);}
  .pl-table th{text-align:right;font-weight:500;font-size:13.1px;letter-spacing:.05em;text-transform:uppercase;color:var(--stone);padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap;background:var(--surface);}
  .pl-table th:first-child,.pl-table th:nth-child(2){text-align:left;}
  .pl-table td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap;color:var(--ivory);}
  .pl-table td:first-child,.pl-table td:nth-child(2){text-align:left;}
  .pl-table tr:last-child td{border-bottom:none;}
  .pl-table tr:hover td{background:var(--raised);}
  .pl-sym{font-weight:500;color:var(--ivory);}
  .pl-sym a{color:inherit;text-decoration:none;border-bottom:1px dotted var(--stone-dim);}
  .pl-sym a:hover{border-bottom-color:var(--gold);color:var(--gold);}
  .pl-sector{color:var(--stone);font-size:15px;}
  .pl-chip{display:inline-flex;align-items:center;gap:5px;font-size:13.8px;font-weight:500;padding:2px 8px;border-radius:20px;white-space:nowrap;}
  .pl-chip-jade{background:rgba(79,174,140,0.16);color:var(--jade);}
  .pl-chip-rose{background:rgba(201,99,107,0.16);color:var(--rose);}
  .pl-chip-amber{background:rgba(208,164,76,0.16);color:var(--amber);}
  .pl-chip-gray{background:rgba(139,146,151,0.14);color:var(--stone);}
  .pl-dot{width:6px;height:6px;border-radius:50%;display:inline-block;}
  .pl-ltcg{font-family:var(--font-mono);font-size:14.4px;color:var(--stone);}
  .pl-foot-note{font-family:var(--font-sans);color:var(--stone-dim);font-size:14.4px;margin:10px 2px 4px;line-height:1.6;}
  .pl-twin{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px;}
  .pl-acct-card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px;font-family:var(--font-sans);}
  .pl-acct-head{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:12px;}
  .pl-acct-head h3{font-family:var(--font-display);font-weight:500;font-size:21.2px;margin:0;color:var(--ivory);}
  .pl-acct-head .n{font-size:14.4px;color:var(--stone-dim);}
  .pl-acct-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
  .pl-acct-stat{border-top:1px solid var(--line);padding-top:7px;}
  .pl-acct-stat .l{font-size:12.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--stone);margin-bottom:3px;}
  .pl-acct-stat .v{font-family:var(--font-mono);font-size:20px;color:var(--ivory);font-variant-numeric:tabular-nums;}
  .pl-treemap-card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px;}
  .pl-treemap{display:flex;flex-direction:column;gap:4px;}
  .pl-tm-row{display:flex;gap:4px;}
  .pl-tm-box{border-radius:6px;padding:7px 9px;display:flex;flex-direction:column;justify-content:space-between;overflow:hidden;position:relative;}
  .pl-tm-sym{font-family:var(--font-sans);font-weight:600;font-size:15px;display:flex;align-items:center;gap:5px;}
  .pl-tm-acct{font-family:var(--font-sans);font-size:11.9px;opacity:.72;margin-top:1px;}
  .pl-tm-gain{font-family:var(--font-mono);font-size:14.4px;font-weight:500;align-self:flex-end;}
  .pl-tdot{width:6px;height:6px;border-radius:50%;flex-shrink:0;}
  .pl-tdot-jade{background:var(--jade);} .pl-tdot-amber{background:var(--amber);} .pl-tdot-rose{background:var(--rose);}
  .pl-tm-jade-3{background:#3D7A63;color:#EAF7F1;} .pl-tm-jade-2{background:#2E5B4C;color:#D6ECE3;} .pl-tm-jade-1{background:#22423A;color:#B9D6C9;}
  .pl-tm-rose-2{background:#5B3236;color:#F0D3D5;} .pl-tm-rose-1{background:#3B2528;color:#D9B3B6;}
  .pl-tm-other{background:var(--raised);color:var(--stone);border:1px dashed var(--line);}
  .pl-tm-legend{display:flex;gap:16px;margin-top:12px;font-family:var(--font-sans);font-size:13.8px;color:var(--stone);flex-wrap:wrap;}
  .pl-tm-legend span{display:inline-flex;align-items:center;gap:6px;}
</style>
"""


def _inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def _fmt_lakh(value: float) -> str:
    return f"₹{value/100000:,.2f}L"


def _missing(value) -> bool:
    """
    True for both Python None and pandas/numpy NaN. Values read back from a
    merged DataFrame column silently turn None into NaN (pandas upcasts a
    mixed None/float column to float64), so a plain `is None` check misses
    that case and lets a stray "nan" leak into the rendered text.
    """
    return value is None or pd.isna(value)


def _gain_pct_class(gain_pct: float | None) -> str:
    if _missing(gain_pct):
        return "pl-chip-gray"
    return "pl-up" if gain_pct >= 0 else "pl-down"


def _gain_tier_class(gain_pct: float | None) -> str:
    """Treemap box color tier: jade for gains, rose for losses, intensity by magnitude."""
    if _missing(gain_pct):
        return "pl-tm-other"
    if gain_pct >= 100:
        return "pl-tm-jade-3"
    if gain_pct >= 20:
        return "pl-tm-jade-2"
    if gain_pct >= 0:
        return "pl-tm-jade-1"
    if gain_pct >= -15:
        return "pl-tm-rose-1"
    return "pl-tm-rose-2"


def _dot_class(technical_badge: str) -> str:
    return {"Green": "pl-tdot-jade", "Amber": "pl-tdot-amber", "Red": "pl-tdot-rose"}.get(technical_badge, "pl-tdot-amber")


def _chip_class(technical_badge: str) -> str:
    return {"Green": "pl-chip-jade", "Amber": "pl-chip-amber", "Red": "pl-chip-rose"}.get(technical_badge, "pl-chip-gray")


# =====================================================
# Data loading (unchanged logic from earlier validation)
# =====================================================

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.dropna()
    df = df[~df.index.duplicated(keep="first")]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)
    return df


@st.cache_data(show_spinner="Fetching current prices and history (Dhan)...", ttl=3600)
def fetch_market_data(symbols: tuple[str, ...], from_date: date) -> dict:
    """
    For every symbol: resolve on NSE first, BSE fallback if needed, then
    fetch daily history (used for both current price and technical
    signals/drawdown, so only one Dhan call per symbol).
    """
    to_date = date.today()

    daily_history = {}
    current_prices = {}
    resolution_errors = {}

    for symbol in symbols:
        try:
            security_id, exchange_segment = _lookup.resolve_with_exchange(symbol)
        except KeyError as exc:
            resolution_errors[symbol] = str(exc)
            continue

        try:
            df = _market.get_historical_daily(
                symbol, from_date, to_date,
                security_id=security_id, exchange_segment=exchange_segment,
            )
            df = _clean(df)
            if df is None or df.empty:
                resolution_errors[symbol] = "No historical data returned"
                continue
            daily_history[symbol] = df
            current_prices[symbol] = float(df["Close"].iloc[-1])
        except Exception as exc:
            resolution_errors[symbol] = str(exc)

    return {
        "daily_history": daily_history,
        "current_prices": current_prices,
        "resolution_errors": resolution_errors,
    }


@st.cache_data(show_spinner="Fetching Nifty 50 benchmark history...", ttl=3600)
def fetch_nifty_history(from_date: date) -> pd.Series:
    to_date = date.today()
    df = _market.get_historical_daily(
        "NIFTY", from_date, to_date,
        security_id=NIFTY_SECURITY_ID, exchange_segment=NIFTY_EXCHANGE_SEGMENT,
    )
    df = _clean(df)
    series = df["Close"]
    series.index = series.index.tz_localize(None) if series.index.tz else series.index
    return series


def _technical_badge(sig: dict) -> str:
    """Green/Amber/Red traffic light from the three technical signals."""
    if sig.get("sma20_state_monthly") == "Insufficient Data":
        return "Insufficient Data"
    below_9 = sig.get("below_9sma_monthly")
    monthly_state = sig.get("sma20_state_monthly", "")
    weekly_state = sig.get("sma20_state_weekly", "")

    bearish_states = {"Below", "Just Crossed Below"}
    bullish_states = {"Above", "Just Crossed Back Above"}

    if monthly_state in bullish_states and weekly_state in bullish_states and not below_9:
        return "Green"
    if monthly_state in bearish_states and weekly_state in bearish_states:
        return "Red"
    return "Amber"


def _load_account(account_name: str, transactions_path: str, sector_map: dict, market_data: dict, nifty: pd.Series) -> dict:
    transactions = load_transactions(transactions_path)
    lots, realized_gains = resolve_open_lots_fifo(transactions)
    current_prices = market_data["current_prices"]
    daily_history = market_data["daily_history"]

    metrics = compute_stock_metrics(lots, current_prices, sector_map)
    summary = compute_portfolio_summary(metrics, lots)
    summary["realized_gain_all_time"] = round(realized_gains["realized_gain"].sum(), 2) if not realized_gains.empty else 0.0

    benchmark_xirr = compute_benchmark_xirr(lots, nifty)
    summary["benchmark_xirr_pct"] = round(benchmark_xirr * 100, 2) if benchmark_xirr is not None else None
    summary["alpha_pct"] = (
        round(summary["portfolio_xirr_pct"] - summary["benchmark_xirr_pct"], 2)
        if summary["portfolio_xirr_pct"] is not None and summary["benchmark_xirr_pct"] is not None
        else None
    )

    per_stock_alpha = compute_per_stock_benchmark_alpha(lots, metrics, nifty)

    technical_rows = []
    for _, row in metrics.iterrows():
        symbol = row["symbol"]
        df = daily_history.get(symbol)
        sig = compute_technical_signals(df) if df is not None else {
            "below_9sma_monthly": None, "sma20_state_monthly": "No Data", "sma20_state_weekly": "No Data",
        }
        max_dd = compute_max_drawdown(df["Close"], since=row["earliest_purchase"]) if df is not None else None
        days_held = (date.today() - row["earliest_purchase"]).days
        technical_rows.append({
            "symbol": symbol,
            **sig,
            "technical_badge": _technical_badge(sig),
            "max_drawdown_pct": max_dd,
            "xirr_provisional": days_held < RECENT_POSITION_DAYS,
            "benchmark_alpha_pct": per_stock_alpha.get(symbol),
        })
    technical_df = pd.DataFrame(technical_rows)
    metrics = metrics.merge(technical_df, on="symbol", how="left")
    metrics["account"] = account_name

    return {"lots": lots, "metrics": metrics, "summary": summary, "realized_gains": realized_gains}


# =====================================================
# Rendering: KPI band, allocation ribbon, holdings table
# =====================================================

def _render_kpi_band(items: list[dict]):
    """Each item: {label, value, delta (optional, colored), sub (optional)}."""
    cells = []
    for item in items:
        delta_html = ""
        if item.get("delta") is not None:
            cls = "pl-up" if item.get("delta_positive", True) else "pl-down"
            delta_html = f'<div class="pl-kpi-delta pl-num {cls}">{item["delta"]}</div>'
        sub_html = f'<div class="pl-kpi-sub">{item["sub"]}</div>' if item.get("sub") else ""
        value_cls = "pl-kpi-value pl-num"
        if item.get("value_positive") is True:
            value_cls += " pl-up"
        elif item.get("value_positive") is False:
            value_cls += " pl-down"
        cells.append(
            f'<div class="pl-kpi"><div class="pl-kpi-label">{item["label"]}</div>'
            f'<div class="{value_cls}">{item["value"]}</div>{delta_html}{sub_html}</div>'
        )
    st.markdown(f'<div class="pl-kpis">{"".join(cells)}</div>', unsafe_allow_html=True)


def _render_allocation_ribbon(sector_alloc_pct: dict, current_total: float):
    if not sector_alloc_pct:
        return
    ordered = sorted(sector_alloc_pct.items(), key=lambda kv: kv[1], reverse=True)
    palette = ["#C9A24B", "#4FAE8C", "#3D7A63", "#8B9297", "#5B6266", "#C9636B", "#7A3B40", "#8A733A",
               "#2B363B", "#22423A", "#5B3236", "#3B2528", "#D0A44C", "#2E5B4C"]
    segments, legend = [], []
    for i, (sector, pct) in enumerate(ordered):
        color = palette[i % len(palette)]
        segments.append(f'<div style="width:{pct}%;background:{color};"></div>')
        legend.append(f'<span><i class="pl-swatch" style="background:{color}"></i>{sector} {pct:.1f}%</span>')

    html = (
        '<div class="pl-ribbon-card"><div class="pl-ribbon-head"><h3>Sector allocation</h3>'
        f'<span>of ₹{current_total:,.0f} current value</span></div>'
        f'<div class="pl-ribbon">{"".join(segments)}</div>'
        f'<div class="pl-ribbon-legend">{"".join(legend)}</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


_TABLE_COMPONENT_HEAD = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root{
    --ink:#12181B; --surface:#1A2226; --raised:#212B30; --line:#2B363B;
    --gold:#C9A24B; --jade:#4FAE8C; --rose:#C9636B; --amber:#D0A44C;
    --ivory:#EDE8DD; --stone:#8B9297; --stone-dim:#5B6266;
    --font-sans:'IBM Plex Sans',sans-serif; --font-mono:'IBM Plex Mono',monospace;
  }
  *{box-sizing:border-box;}
  body{background:var(--ink);margin:0;font-family:var(--font-sans);}
  .pl-num{font-family:var(--font-mono);font-variant-numeric:tabular-nums;}
  .pl-up{color:var(--jade);} .pl-down{color:var(--rose);}
  .pl-table-card{background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:8px;}
  .pl-table-scroll{overflow-x:auto;}
  .pl-table{width:100%;border-collapse:collapse;font-size:16.2px;min-width:1220px;}
  .pl-table th{text-align:right;font-weight:500;font-size:13.1px;letter-spacing:.05em;text-transform:uppercase;color:var(--stone);padding:11px 14px;border-bottom:1px solid var(--line);white-space:nowrap;background:var(--surface);}
  .pl-table th:first-child,.pl-table th:nth-child(2){text-align:left;}
  .pl-table th.sortable{cursor:pointer;user-select:none;}
  .pl-table th.sortable:hover{color:var(--ivory);}
  .pl-table th .arrow{font-size:10px;margin-left:2px;color:var(--gold);}
  .pl-table td{padding:10px 14px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap;color:var(--ivory);}
  .pl-table td:first-child,.pl-table td:nth-child(2){text-align:left;}
  .pl-table tr:last-child td{border-bottom:none;}
  .pl-table tr:hover td{background:var(--raised);}
  .pl-sym{font-weight:500;color:var(--ivory);}
  .pl-sym a{color:inherit;text-decoration:none;border-bottom:1px dotted var(--stone-dim);}
  .pl-sym a:hover{border-bottom-color:var(--gold);color:var(--gold);}
  .pl-chip{display:inline-flex;align-items:center;gap:5px;font-size:13.8px;font-weight:500;padding:2px 8px;border-radius:20px;white-space:nowrap;}
  .pl-chip-jade{background:rgba(79,174,140,0.16);color:var(--jade);}
  .pl-chip-rose{background:rgba(201,99,107,0.16);color:var(--rose);}
  .pl-chip-amber{background:rgba(208,164,76,0.16);color:var(--amber);}
  .pl-chip-gray{background:rgba(139,146,151,0.14);color:var(--stone);}
  .pl-dot{width:6px;height:6px;border-radius:50%;display:inline-block;}
  .pl-tdot-jade{background:var(--jade);} .pl-tdot-amber{background:var(--amber);} .pl-tdot-rose{background:var(--rose);}
  .pl-ltcg{font-family:var(--font-mono);font-size:14.4px;color:var(--stone);}
  .pl-tf-group{display:flex;gap:10px;justify-content:flex-end;}
  .pl-tf{display:flex;flex-direction:column;align-items:center;gap:4px;padding:4px 6px;border-radius:6px;}
  .pl-tf:hover{background:var(--raised);}
  .pl-tf .lbl{font-size:12.5px;color:var(--stone-dim);letter-spacing:.03em;font-weight:500;}
  .pl-tf .pl-dot{width:11px;height:11px;}
  .pl-tf .pl-dot.fresh{box-shadow:0 0 0 3px var(--gold);}
</style>
"""


def _technical_field_dots(row: pd.Series) -> str:
    """
    Three independent readings side by side (9-SMA monthly, 20-SMA
    monthly, 20-SMA weekly) instead of one collapsed word -- a gold ring
    marks a bar that JUST crossed (fresher signal than steady above/below).
    """
    below_9 = row.get("below_9sma_monthly")
    color_9 = "pl-tdot-rose" if below_9 else ("pl-tdot-jade" if below_9 is False else "pl-tdot-amber")

    def _state_dot(state: str) -> tuple[str, bool]:
        if state in ("Above", "Just Crossed Back Above"):
            return "pl-tdot-jade", state.startswith("Just")
        if state in ("Below", "Just Crossed Below"):
            return "pl-tdot-rose", state.startswith("Just")
        return "pl-tdot-amber", False

    m_color, m_fresh = _state_dot(row.get("sma20_state_monthly", ""))
    w_color, w_fresh = _state_dot(row.get("sma20_state_weekly", ""))

    def _cell(label: str, color: str, fresh: bool, title: str) -> str:
        fresh_cls = " fresh" if fresh else ""
        return f'<div class="pl-tf" title="{title}"><span class="lbl">{label}</span><span class="pl-dot {color}{fresh_cls}"></span></div>'

    return (
        '<div class="pl-tf-group">'
        + _cell("9M", color_9, False, f"9-SMA Monthly: {'Below' if below_9 else 'Above' if below_9 is False else 'No data'}")
        + _cell("20M", m_color, m_fresh, f"20-SMA Monthly: {row.get('sma20_state_monthly', 'No data')}")
        + _cell("20W", w_color, w_fresh, f"20-SMA Weekly: {row.get('sma20_state_weekly', 'No data')}")
        + "</div>"
    )


def _render_holdings_table(metrics: pd.DataFrame):
    rows = []
    for _, r in metrics.iterrows():
        tv_url = f"https://www.tradingview.com/chart/{TRADINGVIEW_CHART_ID}/?symbol=NSE%3A{r['symbol']}"
        gain_cls = _gain_pct_class(r["gain_pct"])
        gain_txt = "N/A" if _missing(r["gain_pct"]) else f"{r['gain_pct']:+.1f}%"
        gain_sort = -999999 if _missing(r["gain_pct"]) else r["gain_pct"]

        xirr_txt = "N/A" if _missing(r["xirr_pct"]) else f"{r['xirr_pct']:.1f}%" + (" (prov.)" if r.get("xirr_provisional") else "")
        xirr_sort = -999999 if _missing(r["xirr_pct"]) else r["xirr_pct"]

        alpha = r.get("benchmark_alpha_pct")
        alpha_sort = -999999 if _missing(alpha) else alpha
        if _missing(alpha):
            alpha_chip = '<span class="pl-chip pl-chip-gray">N/A</span>'
        else:
            alpha_cls = "pl-chip-jade" if alpha >= 0 else "pl-chip-rose"
            dot_cls = "pl-tdot-jade" if alpha >= 0 else "pl-tdot-rose"
            alpha_chip = f'<span class="pl-chip {alpha_cls}"><span class="pl-dot {dot_cls}"></span>{alpha:+.1f}pp</span>'

        dd = r.get("max_drawdown_pct")
        dd_txt = "N/A" if _missing(dd) else f"{dd:.1f}%"

        entry_txt = "N/A" if _missing(r.get("entry_price")) else f"{r['entry_price']:,.2f}"
        current_price_txt = "N/A" if _missing(r.get("current_price")) else f"{r['current_price']:,.2f}"

        badge = r.get("technical_badge", "Amber")
        signal_sort = {"Green": 2, "Amber": 1, "Red": 0}.get(badge, -1)
        technical_cell = _technical_field_dots(r)

        rows.append(
            "<tr>"
            f'<td class="pl-sym" data-key="symbol" data-value="{r["symbol"]}"><a href="{tv_url}" target="_blank">{r["symbol"]}</a></td>'
            f'<td class="pl-num">{r["quantity"]:,}</td>'
            f'<td class="pl-num">{entry_txt}</td>'
            f'<td class="pl-num">{current_price_txt}</td>'
            f'<td class="pl-num">{r["invested"]:,.0f}</td>'
            f'<td class="pl-num">{r["current_value"]:,.0f}</td>'
            f'<td class="pl-num {gain_cls}" data-key="gain" data-value="{gain_sort}">{gain_txt}</td>'
            f'<td class="pl-num" data-key="xirr" data-value="{xirr_sort}">{xirr_txt}</td>'
            f'<td data-key="alpha" data-value="{alpha_sort}">{alpha_chip}</td>'
            f'<td class="pl-ltcg">{r["ltcg_pct"]:.0f}%</td>'
            f'<td class="pl-num pl-down">{dd_txt}</td>'
            f'<td data-key="signal" data-value="{signal_sort}">{technical_cell}</td>'
            "</tr>"
        )

    widths = [9, 5, 8, 8, 9, 9, 7, 8, 9, 6, 8, 14]
    labels = ["Symbol", "Qty", "Entry Price", "Current Price", "Invested", "Current",
              "Gain", "XIRR", "Vs. Nifty", "LTCG", "Drawdown", "Technical"]
    sort_keys = {"Symbol": "symbol", "Gain": "gain", "XIRR": "xirr", "Vs. Nifty": "alpha", "Technical": "signal"}
    header_cell_list = []
    for w, label in zip(widths, labels):
        if label in sort_keys:
            attrs = f'style="width:{w}%;" class="sortable" data-sort="{sort_keys[label]}"'
        else:
            attrs = f'style="width:{w}%;"'
        header_cell_list.append(f"<th {attrs}>{label}</th>")
    header_cells = "".join(header_cell_list)
    body_html = (
        f'{_TABLE_COMPONENT_HEAD}'
        f'<div class="pl-table-card"><div class="pl-table-scroll"><table class="pl-table"><thead><tr>{header_cells}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></div>'
        "<script>"
        "document.querySelectorAll('th.sortable').forEach(function(th){"
        "  th.addEventListener('click', function(){"
        "    var key = th.getAttribute('data-sort');"
        "    var table = th.closest('table');"
        "    var tbody = table.querySelector('tbody');"
        "    var thisRows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));"
        "    var newDir = th.getAttribute('data-dir') === 'asc' ? 'desc' : 'asc';"
        "    table.querySelectorAll('th.sortable').forEach(function(h){ h.removeAttribute('data-dir'); var a = h.querySelector('.arrow'); if (a) a.remove(); });"
        "    th.setAttribute('data-dir', newDir);"
        "    var arrow = document.createElement('span'); arrow.className = 'arrow'; arrow.textContent = newDir === 'asc' ? '\\u25B2' : '\\u25BC';"
        "    th.appendChild(arrow);"
        "    thisRows.sort(function(a, b){"
        "      var av = a.querySelector('td[data-key=\"' + key + '\"]').getAttribute('data-value');"
        "      var bv = b.querySelector('td[data-key=\"' + key + '\"]').getAttribute('data-value');"
        "      var an = parseFloat(av), bn = parseFloat(bv);"
        "      var cmp = (!isNaN(an) && !isNaN(bn)) ? (an - bn) : String(av).localeCompare(String(bv));"
        "      return newDir === 'asc' ? cmp : -cmp;"
        "    });"
        "    thisRows.forEach(function(r){ tbody.appendChild(r); });"
        "  });"
        "});"
        "</script>"
    )

    row_count = len(metrics)
    height = 70 + row_count * 56 + 40
    components.html(body_html, height=height, scrolling=False)

    st.markdown(
        '<p class="pl-foot-note">Technical shows the 9-SMA monthly flag plus the 20-SMA monthly &amp; weekly '
        "state independently (a gold ring marks a bar that just crossed, the freshest signal). Vs. Nifty is "
        "that stock's own XIRR minus what the same rupees, on the same dates, would have returned in Nifty "
        "50. Drawdown is the worst peak-to-trough dip since your first purchase, separate from your current "
        "gain. Click a column header to sort.</p>",
        unsafe_allow_html=True,
    )


def _render_account_view(account_data: dict, label: str):
    summary = account_data["summary"]
    metrics = account_data["metrics"]

    alpha = summary.get("alpha_pct")
    kpi_items = [
        {"label": "Invested", "value": _fmt_lakh(summary["total_invested"]), "sub": f"across {len(metrics)} positions"},
        {"label": "Current value", "value": _fmt_lakh(summary["total_current"]),
         "delta": f"↑ {summary['total_gain_pct']:+.1f}%" if summary["total_gain_pct"] is not None else None},
        {"label": "Unrealised gain", "value": _fmt_lakh(summary["total_gain"]), "value_positive": summary["total_gain"] >= 0,
         "sub": f"realised, all-time ₹{summary['realized_gain_all_time']:,.0f}"},
        {"label": "Portfolio XIRR", "value": f"{summary['portfolio_xirr_pct']:.1f}%" if summary["portfolio_xirr_pct"] is not None else "N/A",
         "sub": "annualised, money-weighted"},
        {"label": "Vs. Nifty 50", "value": f"{alpha:+.1f}pp" if alpha is not None else "N/A", "value_positive": (alpha >= 0) if alpha is not None else None,
         "sub": "alpha on same cash-flow dates"},
    ]
    st.markdown(f"### {label}")
    _render_kpi_band(kpi_items)

    if summary.get("concentration_flag"):
        st.warning(
            f"Concentration risk: {summary['top_holding_symbol']} is "
            f"{summary['top_holding_concentration_pct']:.1f}% of this portfolio (above the 15% guideline)."
        )

    _render_allocation_ribbon(summary.get("sector_allocation_pct", {}), summary["total_current"])
    _render_holdings_table(metrics)


def _render_twin_comparison(accounts: dict):
    cards = []
    for name, data in accounts.items():
        s = data["summary"]
        alpha = s.get("alpha_pct")
        alpha_html = f'<div class="v pl-up">{alpha:+.1f}pp</div>' if alpha is not None else '<div class="v" style="color:var(--stone);font-size:12px;">pending</div>'
        cards.append(
            f'<div class="pl-acct-card"><div class="pl-acct-head"><h3>{name}</h3>'
            f'<span class="n">{len(data["metrics"])} positions</span></div>'
            '<div class="pl-acct-grid">'
            f'<div class="pl-acct-stat"><div class="l">Invested</div><div class="v">{_fmt_lakh(s["total_invested"])}</div></div>'
            f'<div class="pl-acct-stat"><div class="l">Current</div><div class="v">{_fmt_lakh(s["total_current"])}</div></div>'
            f'<div class="pl-acct-stat"><div class="l">Gain</div><div class="v pl-up">{s["total_gain_pct"]:+.1f}%</div></div>'
            f'<div class="pl-acct-stat"><div class="l">XIRR</div><div class="v">{s["portfolio_xirr_pct"]:.1f}%</div></div>'
            f'<div class="pl-acct-stat"><div class="l">Vs. Nifty</div>{alpha_html}</div>'
            f'<div class="pl-acct-stat"><div class="l">Top holding</div><div class="v" style="font-size:12.5px;">{s["top_holding_symbol"]} &middot; {s["top_holding_concentration_pct"]:.1f}%</div></div>'
            '</div></div>'
        )
    st.markdown(f'<div class="pl-twin">{"".join(cards)}</div>', unsafe_allow_html=True)


def _render_household_treemap(combined_metrics: pd.DataFrame):
    ranked = combined_metrics.sort_values("current_value", ascending=False).reset_index(drop=True)
    top = ranked.iloc[:TREEMAP_TOP_N]
    rest = ranked.iloc[TREEMAP_TOP_N:]

    row_sizes = [5, 5, 6, 4]
    row_heights = [124, 104, 90, 82]
    rows_html = []
    pos = 0
    for size, height in zip(row_sizes, row_heights):
        chunk = top.iloc[pos:pos + size]
        pos += size
        if chunk.empty:
            continue
        row_total = chunk["current_value"].sum()
        boxes = []
        for _, r in chunk.iterrows():
            width_pct = (r["current_value"] / row_total * 100) if row_total > 0 else 0
            tier = _gain_tier_class(r["gain_pct"])
            dot = _dot_class(r.get("technical_badge", "Amber"))
            gain_txt = "" if _missing(r["gain_pct"]) else f"{r['gain_pct']:+.1f}%"
            boxes.append(
                f'<div class="pl-tm-box {tier}" style="width:{width_pct:.2f}%;">'
                f'<div><div class="pl-tm-sym"><span class="pl-tdot {dot}"></span>{r["symbol"]}</div>'
                f'<div class="pl-tm-acct">{r["account"]}</div></div>'
                f'<div class="pl-tm-gain">{gain_txt}</div></div>'
            )
        rows_html.append(f'<div class="pl-tm-row" style="height:{height}px;">{"".join(boxes)}</div>')

    if not rest.empty and rows_html:
        # Make room in the last row for one "N more positions" box, sized
        # relative to that row's existing total value. Shrink the row's
        # existing boxes proportionally (their widths still sum correctly
        # within the row), then insert the new box before the row's
        # closing </div>.
        other_value = rest["current_value"].sum()
        last_chunk = top.iloc[max(0, pos - row_sizes[-1]):pos]
        last_row_total = last_chunk["current_value"].sum() + other_value
        other_width = (other_value / last_row_total * 100) if last_row_total > 0 else 100
        scale = (100 - other_width) / 100

        def _shrink(match: re.Match) -> str:
            return f'style="width:{float(match.group(1)) * scale:.2f}%;'

        last_row = re.sub(r'style="width:([\d.]+)%;', _shrink, rows_html[-1])
        other_box = (
            f'<div class="pl-tm-box pl-tm-other" style="width:{other_width:.2f}%;">'
            f'<div><div class="pl-tm-sym" style="font-size:12.5px;">{len(rest)} more positions</div>'
            f'<div class="pl-tm-acct">Both accounts &middot; ₹{other_value:,.0f} combined</div></div>'
            '<div class="pl-tm-gain" style="color:var(--stone);">mixed</div></div>'
        )
        assert last_row.endswith("</div>")
        rows_html[-1] = last_row[: -len("</div>")] + other_box + "</div>"

    legend = (
        '<div class="pl-tm-legend">'
        '<span><i class="pl-swatch" style="background:#3D7A63"></i>Strong gain (100%+)</span>'
        '<span><i class="pl-swatch" style="background:#2E5B4C"></i>Moderate gain (20-100%)</span>'
        '<span><i class="pl-swatch" style="background:#22423A"></i>Flat / slight gain</span>'
        '<span><i class="pl-swatch" style="background:#3B2528"></i>Slight loss</span>'
        '<span><i class="pl-swatch" style="background:#5B3236"></i>Moderate loss (15%+)</span>'
        '<span><i class="pl-swatch" style="background:var(--raised);border:1px dashed var(--line)"></i>Grouped small positions</span>'
        "</div>"
    )
    html = (
        '<div class="pl-treemap-card"><div class="pl-ribbon-head"><h3>All holdings, by value</h3>'
        "<span>box size = current value &middot; color = gain / loss</span></div>"
        f'<div class="pl-treemap">{"".join(rows_html)}</div>{legend}</div>'
        '<p class="pl-foot-note">Top 20 positions shown individually; smaller positions are grouped so the '
        "treemap stays readable.</p>"
    )
    st.markdown(html, unsafe_allow_html=True)


# =====================================================
# Main Streamlit tab
# =====================================================

def run_portfolio_tab():
    _inject_css()
    st.title("Portfolio Dashboard")
    st.caption("Household summary + per-account drill-down. Manually re-export from the broker to refresh lot data.")

    if st.button("\U0001F504 Refresh Market Data"):
        fetch_market_data.clear()
        fetch_nifty_history.clear()
        st.session_state.pop("portfolio_results", None)
        st.session_state.pop("portfolio_nifty", None)

    if st.button("Load / Refresh Portfolio Dashboard") or "portfolio_results" in st.session_state:
        if "portfolio_results" not in st.session_state:
            sector_map = load_sector_map(SECTOR_MAP_PATH)

            all_symbols = set()
            earliest_purchase = date.today()
            for path in ACCOUNTS.values():
                txns = load_transactions(path)
                all_symbols.update(txns["symbol"].unique())
                earliest_purchase = min(earliest_purchase, txns["date"].min().date())

            # Fetch back to whichever is earlier: the fixed default lookback,
            # or 2 years before the oldest actual purchase (buffer so the
            # 20-month SMA and the Nifty benchmark both have real data to
            # work with on day one of even the oldest holding -- a fixed
            # 4-year window silently missed anything bought before ~2022).
            from_date = min(
                date.today() - timedelta(days=DAILY_LOOKBACK_DAYS),
                earliest_purchase - timedelta(days=730),
            )

            market_data = fetch_market_data(tuple(sorted(all_symbols)), from_date)
            nifty = fetch_nifty_history(from_date)

            if market_data["resolution_errors"]:
                st.warning(
                    "Could not fetch data for: " +
                    ", ".join(f"{s} ({e})" for s, e in market_data["resolution_errors"].items())
                )

            accounts = {
                name: _load_account(name, path, sector_map, market_data, nifty)
                for name, path in ACCOUNTS.items()
            }
            st.session_state["portfolio_results"] = accounts
            st.session_state["portfolio_nifty"] = nifty

        accounts = st.session_state["portfolio_results"]
        nifty = st.session_state["portfolio_nifty"]

        # Household combined summary
        total_invested = sum(a["summary"]["total_invested"] for a in accounts.values())
        total_current = sum(a["summary"]["total_current"] for a in accounts.values())
        total_gain = total_current - total_invested
        combined_sector = {}
        for a in accounts.values():
            for sector, value in a["summary"]["sector_allocation"].items():
                combined_sector[sector] = combined_sector.get(sector, 0) + value
        combined_sector_pct = {k: round(v / total_current * 100, 1) for k, v in combined_sector.items()} if total_current else {}

        all_cashflows = []
        for a in accounts.values():
            for _, row in a["lots"].iterrows():
                all_cashflows.append((row["date"].date(), -row["actual_value"]))
        all_cashflows.append((date.today(), total_current))
        household_xirr = xirr(all_cashflows)

        combined_lots = pd.concat([a["lots"] for a in accounts.values()], ignore_index=True)
        household_benchmark_xirr = compute_benchmark_xirr(combined_lots, nifty)

        combined_metrics = pd.concat([a["metrics"] for a in accounts.values()], ignore_index=True)
        top_row_df = combined_metrics.sort_values("current_value", ascending=False)
        top_row = top_row_df.iloc[0] if not top_row_df.empty else None
        household_concentration_pct = (top_row["current_value"] / total_current * 100) if top_row is not None and total_current else 0.0

        household_alpha = (
            round(household_xirr * 100 - household_benchmark_xirr * 100, 2)
            if household_xirr is not None and household_benchmark_xirr is not None else None
        )

        tab_labels = ["Household"] + list(ACCOUNTS.keys())
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            kpi_items = [
                {"label": "Invested", "value": _fmt_lakh(total_invested), "sub": f"{len(combined_metrics)} positions, {len(accounts)} accounts"},
                {"label": "Current value", "value": _fmt_lakh(total_current),
                 "delta": f"↑ {total_gain/total_invested*100:+.1f}%" if total_invested else None},
                {"label": "Unrealised gain", "value": _fmt_lakh(total_gain), "value_positive": total_gain >= 0,
                 "sub": f"realised, all-time ₹{sum(a['summary']['realized_gain_all_time'] for a in accounts.values()):,.0f}"},
                {"label": "Household XIRR", "value": f"{household_xirr*100:.1f}%" if household_xirr is not None else "N/A",
                 "sub": "annualised, money-weighted"},
                {"label": "Top position", "value": top_row["symbol"] if top_row is not None else "N/A",
                 "sub": f"{household_concentration_pct:.1f}% of household" + (" — concentration flag" if household_concentration_pct >= 15 else " — no concentration flag")},
            ]
            _render_kpi_band(kpi_items)
            if household_alpha is not None:
                st.caption(f"vs. Nifty 50 (same cash-flow dates/amounts): household XIRR {household_xirr*100:.1f}% vs. benchmark {household_benchmark_xirr*100:.1f}% — alpha {household_alpha:+.1f}pp")

            _render_twin_comparison(accounts)
            _render_household_treemap(combined_metrics)

        for i, (name, data) in enumerate(accounts.items(), start=1):
            with tabs[i]:
                _render_account_view(data, name)

    else:
        st.info("Click 'Load / Refresh Portfolio Dashboard' to fetch live prices and compute metrics.")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    run_portfolio_tab()
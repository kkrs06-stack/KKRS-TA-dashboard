import pandas as pd
import numpy as np
import ta
import yfinance as yf
import streamlit as st
import datetime
import warnings
import sys
import os
from ta.trend import ADXIndicator, MACD
from ta.volatility import AverageTrueRange

# --- Import modular dashboard functions for other tabs (already optimized) ---
from just_above_below_dashboard import just_above_below_dashboard, fetch_all_ohlcv, batch_analyze_all
from daily_weekly_dashboard import daily_weekly_dashboard
# from ss_strat_dashboard import ss_strat_dashboard, get_ssstrat_stock_data
from ss_strat_dashboard import ss_strat_dashboard, get_ssstrat_stock_data_optimized

warnings.filterwarnings("ignore")

class suppress_stdout_stderr(object):
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

st.set_page_config(layout="wide")

today_str = datetime.datetime.now().strftime("%d-%b-%Y")
STRATEGIES = [
    {"label": "RSI Strategy", "id": "rsi"},
    {"label": "SS Strat", "id": "ssstrat"},
    {"label": "Daily > Weekly", "id": "dailyweekly"},
    {"label": "Just above/below", "id": "reversal"},
    {"label": "Momentum", "id": "swing"},
    {"label": "Custom Watchlist", "id": "custom"},
]

# ----------- OPTIMIZED RSI TAB: batch + cache ----------- #
@st.cache_data(show_spinner="Batch downloading RSI data for all F&O stocks...")
def get_rsi_strategy_data_optimized():
    fo_df = pd.read_csv("fo_stock_list.csv")
    tickers = fo_df["symbol"].str.strip().str.upper().tolist()
    ticker_name_map = dict(zip(fo_df["symbol"].str.strip().str.upper(), fo_df["name"]))
    lot_size_map = dict(zip(fo_df["symbol"].str.strip().str.upper(), fo_df["lot_size"]))

    df = yf.download(
        tickers=" ".join(tickers),
        period="120d",
        interval="1d",
        group_by='ticker',
        progress=False,
        auto_adjust=True,
        threads=True
    )

    results = []
    for symbol in tickers:
        try:
            d = df[symbol] if isinstance(df.columns, pd.MultiIndex) else df
            close = d["Close"].dropna() if "Close" in d else pd.Series([])
            price = close.iloc[-1] if len(close) > 0 else None
            prev_price = close.iloc[-2] if len(close) > 1 else price
            rsi = ta.momentum.RSIIndicator(close, window=14).rsi().dropna().iloc[-1] if len(close) > 15 else None
        except Exception:
            price, prev_price, rsi = None, None, None
        results.append({
            "Stock Name": ticker_name_map.get(symbol, symbol),
            "Symbol": symbol,
            "Lot Size": lot_size_map.get(symbol, ""),
            "Latest Price": price,
            "Previous Price": prev_price,
            "Daily RSI": rsi,
        })
    df_out = pd.DataFrame(results)
    df_out["Latest Price"] = pd.to_numeric(df_out["Latest Price"], errors="coerce")
    df_out["Previous Price"] = pd.to_numeric(df_out["Previous Price"], errors="coerce")
    df_out["Daily RSI"] = pd.to_numeric(df_out["Daily RSI"], errors="coerce")
    return df_out

# ------------- CACHE CLEAR FUNCTION -------------
def clear_all_app_cache():
    get_rsi_strategy_data_optimized.clear()
    get_ssstrat_stock_data_optimized.clear()
    fetch_all_ohlcv.clear()
    batch_analyze_all.clear()
    # Add other .clear() for cached tab functions if needed

# ---- CLEAR ON FIRST LOAD ----
if "cache_cleared_initial" not in st.session_state:
    clear_all_app_cache()
    st.session_state["cache_cleared_initial"] = True

# ---- TITLE/DATELINE/CLEAR BUTTON ----
title_cols = st.columns([5, 1, 1])
with title_cols[0]:
    st.markdown(
        "<div style='font-size:2.07em;font-weight:800;color:#FFD700;padding-bottom:2px;'>Kamlesh Technical Analysis Strategies</div>",
        unsafe_allow_html=True)
with title_cols[1]:
    st.markdown(
        f"<div style='text-align:right;font-size:1.29em;color:#FFD700;font-weight:700;padding-top:8px;'>{today_str}</div>",
        unsafe_allow_html=True)
with title_cols[2]:
    if st.button("❌ Clear All Cache", key="clear_all_cache"):
        clear_all_app_cache()
        st.success("All cached data has been cleared!")
        st.rerun()

selected = st.query_params.get('selected_strategy', 'rsi')
if isinstance(selected, list):
    selected = selected[0]

button_cols = st.columns(len(STRATEGIES))
for i, strat in enumerate(STRATEGIES):
    is_active = (strat["id"] == selected)
    if is_active:
        button_cols[i].markdown(
            f"<div style='background:#FF69B4;color:#fff;border-radius:10px;padding:12px 8px;text-align:center;font-weight:700;margin-bottom:5px;border:2px solid #FFD700;font-size:1.05em;letter-spacing:1.5px;'>{strat['label']}</div>",
            unsafe_allow_html=True
        )
    else:
        if button_cols[i].button(strat["label"], key=strat["id"], use_container_width=True):
            st.query_params.selected_strategy = strat["id"]
            st.rerun()

TRADINGVIEW_LINKS = {id: "https://www.tradingview.com/chart/HHuUSOTG/" for id in [s["id"] for s in STRATEGIES]}
tv_chart_url = TRADINGVIEW_LINKS[selected]

def get_first_two(text):
    return " ".join(str(text).split()[:2])

def price_arrow_and_change(price, prev_price):
    try:
        price = float(price)
        prev_price = float(prev_price)
        change = price - prev_price
        pct = ((change) / prev_price) * 100 if prev_price else 0
        if change > 0:
            arrow, color = "↑", "#37F553"
        elif change < 0:
            arrow, color = "↓", "#FF3A3A"
        else:
            arrow, color = "", "#ECECEC"
        return arrow, color, round(change,2), round(pct,2)
    except:
        return "", "#ECECEC", "NA", "NA"

def rsi_color(val):
    try:
        v = float(val)
        if v > 60: return "#37F553"
        if v < 40: return "#FF3A3A"
        return "#FFA500"
    except: return "#ECECEC"

def rsi_colored(rsi):
    try:
        val = float(rsi)
        if val > 55:
            color = "#37F553"
        elif val < 50:
            color = "#FF3A3A"
        else:
            color = "#FFD700"
        return f"<span style='color:{color};font-weight:700;'>{val:.2f}</span>"
    except:
        return f"<span style='color:#ECECEC;'>{rsi}</span>"

def to_float(val):
    try:
        if isinstance(val, (pd.Series, np.ndarray)):
            arr = np.asarray(val)
            if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
                return float(arr.item())
            return np.nan
        else:
            return float(val)
    except:
        return np.nan

# -------------- DASHBOARD TABS --------------
if selected == "rsi":
    if st.button("🔄 Refresh RSI Data", key="refresh_rsi"):
        get_rsi_strategy_data_optimized.clear()
        st.rerun()
    df = get_rsi_strategy_data_optimized()
    search_cols = st.columns([7, 1])
    search_cols[0].markdown(
        "<div style='font-size:1.12em;font-weight: bold; color:#FFD700; padding-bottom:3px'>Search Stock Name</div>", unsafe_allow_html=True
    )
    search_cols[0].text_input(
        "", 
        value=st.session_state.get("rsi_search", ""), 
        key="rsi_search",
        placeholder="Type stock name..."
    )
    if search_cols[1].button("Reset", key="rsi_reset"):
        st.session_state["rsi_search"] = ""
    search_text = st.session_state.rsi_search.strip().lower()
    filtered_df = df if not search_text else df[df["Stock Name"].str.lower().str.contains(search_text)]

    def rsi_tile_color(val):
        try:
            v = float(val)
            if v > 55:
                return "#37F553"
            elif v < 50:
                return "#FF3A3A"
            else:
                return "#FFD700"
        except:
            return "#ECECEC"

    left_group = filtered_df[filtered_df["Daily RSI"] >= 55].reset_index(drop=True)
    right_group = filtered_df[filtered_df["Daily RSI"] < 55].reset_index(drop=True)
    max_len = max(len(left_group), len(right_group))
    rows_needed = (max_len + 3) // 4

    def card(row, rsi_green=True):
        price = row['Latest Price']
        prev = row['Previous Price']
        arrow, pcolor, change, pct = price_arrow_and_change(price, prev)
        price_str = f"{price:.2f}" if pd.notnull(price) else "NA"
        change_str = f"{change:+.2f}" if pd.notnull(price) and pd.notnull(prev) else "NA"
        pct_str = f"{pct:+.2f}%" if pd.notnull(price) and pd.notnull(prev) else "NA"
        tv_url = f"{tv_chart_url}?symbol=NSE:{row['Symbol'].replace('.NS', '')}"
        name_html = f'<a href="{tv_url}" target="_blank" style="color:#fff;text-decoration:none;"><span style="font-size:1.13em;text-align:center;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{get_first_two(row["Stock Name"])}</span></a>'
        rsi = row["Daily RSI"]
        if pd.isna(rsi):
            rsi_txt = '<span style="font-weight:bold;color:#fff;">RSI: NA</span>'
        else:
            rsi_txt = f'<span style="color:{rsi_tile_color(rsi)};font-weight:bold;">RSI: {round(rsi,2)}</span>'
        lot_line = f'{rsi_txt}&nbsp;&nbsp;<span style="font-size:0.94em;">Lot: <b>{row["Lot Size"]}</b></span>'
        return f"""
            <div style="
                background-color:#252525;
                padding:10px 8px 10px 8px;
                border-radius:12px;
                min-width:140px;
                max-width:160px;
                margin-bottom:9px;
                border:1px solid #333;
                text-align:center;
                box-shadow:1px 2px 8px #111;
                color:#ECECEC;
                ">
                {name_html}
                <span style="font-size:1.03em;color:{pcolor};font-weight:700;">
                    ₹ {price_str}
                    <span style="font-size:1.09em;">{arrow}</span>
                    <span style="color:{pcolor};margin-left:4px;font-size:0.97em">{change_str} ({pct_str})</span>
                </span>
                <br><br><br>
                {lot_line}
            </div>
        """

    for row in range(rows_needed):
        cols = st.columns(8)
        for i in range(4):
            idx = row * 4 + i
            if idx < len(left_group):
                with cols[i]:
                    st.markdown(card(left_group.iloc[idx], rsi_green=True), unsafe_allow_html=True)
        for i in range(4):
            idx = row * 4 + i
            if idx < len(right_group):
                with cols[i+4]:
                    st.markdown(card(right_group.iloc[idx], rsi_green=False), unsafe_allow_html=True)

elif selected == "ssstrat":
    ss_strat_dashboard(tv_chart_url)
elif selected == "dailyweekly":
    daily_weekly_dashboard()
elif selected == "reversal":
    just_above_below_dashboard()
elif selected not in ["rsi", "ssstrat", "dailyweekly", "reversal"]:
    st.info("This strategy dashboard is coming soon!")
else:
    st.markdown("> **Select a strategy tile above to view its details.**")

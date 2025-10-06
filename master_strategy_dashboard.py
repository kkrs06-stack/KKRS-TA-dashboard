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

# Import modular dashboard functions
from just_above_below_dashboard import just_above_below_dashboard
from daily_weekly_dashboard import daily_weekly_dashboard

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

title_cols = st.columns([5, 1])
with title_cols[0]:
    st.markdown(
        "<div style='font-size:2.07em;font-weight:800;color:#FFD700;padding-bottom:2px;'>Kamlesh Technical Analysis Strategies</div>",
        unsafe_allow_html=True)
with title_cols[1]:
    st.markdown(
        f"<div style='text-align:right;font-size:1.29em;color:#FFD700;font-weight:700;padding-top:8px;'>{today_str}</div>",
        unsafe_allow_html=True)

# ---- WORKING PINK ACTIVE TABS (IN-PAGE NAVIGATION) ----
selected = st.query_params.get('selected_strategy', 'rsi')
if isinstance(selected, list):
    selected = selected[0]

button_cols = st.columns(len(STRATEGIES))
for i, strat in enumerate(STRATEGIES):
    is_active = (strat["id"] == selected)
    
    if is_active:
        # Pink highlight for active tab
        button_cols[i].markdown(
            f"<div style='background:#FF69B4;color:#fff;border-radius:10px;padding:12px 8px;text-align:center;font-weight:700;margin-bottom:5px;border:2px solid #FFD700;font-size:1.05em;letter-spacing:1.5px;'>{strat['label']}</div>",
            unsafe_allow_html=True
        )
    else:
        # Clickable button for inactive tabs
        if button_cols[i].button(strat["label"], key=strat["id"], use_container_width=True):
            st.query_params.selected_strategy = strat["id"]
            st.rerun()

TRADINGVIEW_LINKS = {id: "https://www.tradingview.com/chart/lDI0poON/" for id in [s["id"] for s in STRATEGIES]}
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

@st.cache_data(show_spinner="Loading data for RSI Strategy...")
def get_rsi_strategy_data():
    fo_stocks = pd.read_csv("fo_stock_list.csv").to_dict(orient="records")
    results = []
    for stock in fo_stocks:
        symbol = stock["symbol"].strip().upper()
        try:
            with suppress_stdout_stderr():
                data = yf.download(
                    symbol, period="120d", interval="1d", progress=False, group_by='column'
                )
            if isinstance(data.columns, pd.MultiIndex):
                close = data[('Close', symbol)]
            else:
                close = data['Close']
            price = None
            rsi = None
            prev_price = None
            if (not data.empty) and (close.count() >= 15):
                rsi_series = ta.momentum.RSIIndicator(close, window=14).rsi()
                try:
                    last_valid_close = close.dropna().iloc[-1]
                    price = float(last_valid_close)
                except Exception:
                    price = None
                try:
                    prev_valid_close = close.dropna().iloc[-2] if close.dropna().size > 1 else price
                    prev_price = float(prev_valid_close)
                except Exception:
                    prev_price = price
                try:
                    last_valid_rsi = rsi_series.dropna().iloc[-1]
                    rsi = float(last_valid_rsi)
                except Exception:
                    rsi = None
            results.append({
                "Stock Name": stock["name"],
                "Symbol": symbol,
                "Lot Size": stock["lot_size"],
                "Latest Price": price,
                "Previous Price": prev_price,
                "Daily RSI": rsi,
            })
        except Exception:
            results.append({
                "Stock Name": stock["name"],
                "Symbol": symbol,
                "Lot Size": stock["lot_size"],
                "Latest Price": None,
                "Previous Price": None,
                "Daily RSI": None,
            })
    df = pd.DataFrame(results)
    df["Latest Price"] = pd.to_numeric(df["Latest Price"], errors="coerce")
    df["Previous Price"] = pd.to_numeric(df["Previous Price"], errors="coerce")
    df["Daily RSI"] = pd.to_numeric(df["Daily RSI"], errors="coerce")
    return df

@st.cache_data(show_spinner="Loading data for SS Strat...")
def get_ssstrat_stock_data():
    df = pd.read_csv('fo_stock_hippo.csv')
    stocks = []
    for _, row in df.iterrows():
        name = get_first_two(row["name"])
        symbol = row["symbol"]
        lot = row["lot_size"]
        try:
            with suppress_stdout_stderr():
                data = yf.download(symbol, period='180d')
            close_series = data['Close'].squeeze()
            if isinstance(close_series, pd.DataFrame): close_series = close_series.iloc[:,0]
            price = close_series.iloc[-1]
            prev_price = close_series.iloc[-2] if len(close_series) > 1 else price
            d_rsi = ta.momentum.RSIIndicator(close_series, window=14).rsi().iloc[-1]
            weekly_close = close_series.resample('W').last()
            wk_rsi = ta.momentum.RSIIndicator(weekly_close, window=14).rsi().iloc[-1]
        except Exception as e:
            price = prev_price = wk_rsi = d_rsi = "NA"
        arrow, arrow_color, change, pct = price_arrow_and_change(price, prev_price)
        stocks.append({
            "name": name, "symbol": symbol, "price": price, "prev_price": prev_price, "change": change, "pct": pct,
            "lot": lot, "wk_rsi": wk_rsi, "d_rsi": d_rsi, "arrow": arrow, "arrow_color": arrow_color
        })
    ce_sell, pe_sell, cep_sell = [], [], []
    for s in stocks:
        try:
            wk = float(s["wk_rsi"])
            if wk > 60: pe_sell.append(s)
            elif 40 <= wk <= 60: cep_sell.append(s)
            elif wk < 50: ce_sell.append(s)
        except:
            continue
    return ce_sell, pe_sell, cep_sell

# ----------- DASHBOARD TABS -----------
if selected == "rsi":
    if st.button("🔄 Refresh RSI Data", key="refresh_rsi"):
        get_rsi_strategy_data.clear()
    df = get_rsi_strategy_data()

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
    if st.button("🔄 Refresh SS Strat Data", key="refresh_ssstrat"):
        get_ssstrat_stock_data.clear()
    ce_sell, pe_sell, cep_sell = get_ssstrat_stock_data()
    section_titles = ["CE Sell", "PE Sell", "PE & CE Sell"]
    section_colors = ["#441416", "#193821", "#4B3708"]
    section_tiles = [ce_sell, pe_sell, cep_sell]
    
    def tile_html(s):
        price_str = f"{s['price']:.2f}" if isinstance(s['price'], (float, int)) and s['price'] not in ["NA", None] else s['price']
        change_str = f"{s['change']:+.2f}" if isinstance(s['change'], (float, int)) and s['change'] not in ["NA", None] else s['change']
        pct_str = f"{s['pct']:+.2f}%" if isinstance(s['pct'], (float, int)) and s['pct'] not in ["NA", None] else s['pct']
        wk_rsi_str = f"{s['wk_rsi']:.2f}" if isinstance(s['wk_rsi'], (float, int)) and s['wk_rsi'] not in ["NA", None] else s['wk_rsi']
        d_rsi_str = f"{s['d_rsi']:.2f}" if isinstance(s['d_rsi'], (float, int)) and s['d_rsi'] not in ["NA", None] else s['d_rsi']
        tv_url = f"{tv_chart_url}?symbol=NSE:{s['symbol'].replace('.NS','')}"
        return f"""
        <div style="background:#252525;border-radius:15px;width:260px;height:126px;position:relative;box-shadow:1px 2px 8px #111;margin-bottom:18px;display:flex;flex-direction:column;align-items:center;border:1px solid #333;">
          <div style="font-size:1.02em;color:#fff;font-weight:700;text-align:center;width:100%;margin-top:13px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">
            <a href="{tv_url}" target="_blank" style="color:#fff;text-decoration:none;">{s['name']}</a>
            <span style="font-size:0.93em;color:#ECECEC;">Lot: <b>{s['lot']}</b></span>
          </div>
          <div style="width:100%;text-align:center;margin-top:7px;margin-bottom:0;">
            <span style="font-size:1.07em;color:{s['arrow_color']};font-weight:700;">
                ₹ {price_str}
                <span style="font-size:1.08em;">{s['arrow']}</span>
                <span style="margin-left:6px;font-size:0.94em">{change_str} ({pct_str})</span>
            </span>
          </div>
          <div style="position:absolute;bottom:11px;left:14px;text-align:left;">
            <span style="font-size:1.07em;color:#FFD700;font-weight:600;">RSI:</span>
            <span style="font-size:1.07em;color:{rsi_color(s['wk_rsi'])};font-weight:700;">Wk:{wk_rsi_str}</span>
            <span style="font-size:1.07em;color:{rsi_color(s['d_rsi'])};font-weight:700;">D:{d_rsi_str}</span>
          </div>
          <div style="position:absolute;bottom:11px;right:15px;font-size:0.87em;color:#ECECEC;text-align:right;">
            Lot: <b>{s['lot']}</b>
          </div>
        </div>
        """

    cols = st.columns(3)
    for idx, col in enumerate(cols):
        col.markdown(
            f'''<div style="background:{section_colors[idx]};padding:11px 0 11px 0;border-radius:10px;margin-bottom:13px;text-align:center;width:99%;">
            <span style="color:#FFD700;font-size:1.13em;font-weight:650;">&#x25CF;</span> {section_titles[idx]}</div>''',
            unsafe_allow_html=True,
        )
        tiles = section_tiles[idx]
        for i in range(0, len(tiles), 2):
            row_tiles = tiles[i:i+2]
            tile_cols = col.columns(2)
            for tcol, s in zip(tile_cols, row_tiles):
                tcol.markdown(tile_html(s), unsafe_allow_html=True)
        if not tiles:
            col.write("No stocks matched.")

elif selected == "dailyweekly":
    daily_weekly_dashboard()  # Call modular function

elif selected == "reversal":
    just_above_below_dashboard()  # Call modular function

elif selected not in ["rsi", "ssstrat", "dailyweekly", "reversal"]:
    st.info("This strategy dashboard is coming soon!")
else:
    st.markdown("> **Select a strategy tile above to view its details.**")

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
    {"label": "Reversal", "id": "reversal"},
    {"label": "Swing", "id": "swing"},
    {"label": "Custom Watchlist", "id": "custom"},
]
if "selected_strategy" not in st.session_state:
    st.session_state.selected_strategy = STRATEGIES[0]["id"]

title_cols = st.columns([5, 1])
with title_cols[0]:
    st.markdown(
        "<div style='font-size:2.07em;font-weight:800;color:#FFD700;padding-bottom:2px;'>Kamlesh Technical Analysis Strategies</div>",
        unsafe_allow_html=True)
with title_cols[1]:
    st.markdown(
        f"<div style='text-align:right;font-size:1.29em;color:#FFD700;font-weight:700;padding-top:8px;'>{today_str}</div>",
        unsafe_allow_html=True)

button_row = st.columns(len(STRATEGIES), gap="small")
for i, strat in enumerate(STRATEGIES):
    if button_row[i].button(strat["label"], key=f"tile_{strat['id']}", use_container_width=True):
        st.session_state.selected_strategy = strat["id"]

selected = st.session_state.selected_strategy

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

# ---------- CACHED RSI DASHBOARD ----------
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

# ---------- CACHED SS STRAT DASHBOARD ----------
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
            "name": name, "price": price, "prev_price": prev_price, "change": change, "pct": pct,
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

# ---------- MONTHLY (DAILY > WEEKLY) STRATEGY ----------
def fetch_ohlcv(ticker):
    df_d = yf.download(ticker, period='350d', interval='1d', progress=False, auto_adjust=True)
    if isinstance(df_d.columns, pd.MultiIndex):
        df_d.columns = [col[0] for col in df_d.columns]
    df_d = df_d.dropna(subset=['High', 'Low', 'Close', 'Volume'])
    df_d = df_d[~df_d.index.duplicated(keep='first')]
    return df_d

def calc_indicators(df_d):
    adx = ADXIndicator(df_d['High'], df_d['Low'], df_d['Close'], 14)
    df_d['ADX14'] = adx.adx()
    df_d['DI+'] = adx.adx_pos()
    df_d['DI-'] = adx.adx_neg()
    df_d['SMA20'] = df_d['Close'].rolling(20).mean()
    macd = MACD(df_d['Close'])
    df_d['MACD'] = macd.macd()
    df_d['MACD_signal'] = macd.macd_signal()
    atr = AverageTrueRange(df_d['High'], df_d['Low'], df_d['Close'], 14)
    df_d['ATR14'] = atr.average_true_range()
    return df_d

def get_weekly_20_sma(df_d):
    df_w = df_d['Close'].resample('W-FRI').last().dropna().to_frame()
    today = pd.Timestamp.today().date()
    if len(df_w) == 0:
        return np.nan
    if df_w.index[-1].date() > today:
        df_w = df_w.iloc[:-1]
    df_w['SMA20W'] = df_w['Close'].rolling(20).mean()
    return to_float(df_w['SMA20W'].iloc[-1]) if len(df_w['SMA20W']) >= 1 else np.nan

def price_color_and_arrow_monthly(close, prev_close):
    try:
        close = float(close)
        prev_close = float(prev_close)
        change = close - prev_close
        pct = (change / prev_close) * 100 if prev_close else 0
        if change > 0:
            return "#18AA47", "↑", change, pct
        elif change < 0:
            return "#E53935", "↓", change, pct
        else:
            return "#ECECEC", "", change, pct
    except:
        return "#ECECEC", "", "NA", "NA"

def analyze_ticker_long(ticker, name):
    try:
        df_d = fetch_ohlcv(ticker)
        if df_d.shape[0] < 50:
            return None
        df_d = calc_indicators(df_d)
        weekly_20_sma = get_weekly_20_sma(df_d)
        if not np.isfinite(weekly_20_sma):
            return None
        last_close = to_float(df_d['Close'].iloc[-1])
        prev_close = to_float(df_d['Close'].iloc[-2])
        last_sma20 = to_float(df_d['SMA20'].iloc[-1])
        last_adx14 = to_float(df_d['ADX14'].iloc[-1])
        last_di_pos = to_float(df_d['DI+'].iloc[-1])
        last_di_neg = to_float(df_d['DI-'].iloc[-1])
        last_macd = to_float(df_d['MACD'].iloc[-1])
        last_macd_signal = to_float(df_d['MACD_signal'].iloc[-1])
        last_atr14 = to_float(df_d['ATR14'].iloc[-1])
        values = [last_close, last_sma20, prev_close, last_adx14, last_macd, last_macd_signal, last_atr14, weekly_20_sma, last_di_pos, last_di_neg]
        if not all([isinstance(x, float) and np.isfinite(x) for x in values]):
            return None
        cond_cross_sma = (last_sma20 > weekly_20_sma) and (df_d['SMA20'].iloc[-2] <= weekly_20_sma)
        if cond_cross_sma:
            return {
                'Name': name,
                'Close': round(last_close, 2),
                'PrevClose': round(prev_close, 2),
                'SMA20(D)': round(last_sma20, 2),
                'SMA20(W)': round(weekly_20_sma, 2),
                'ADX14': round(last_adx14, 2),
                'DI+': round(last_di_pos, 2),
                'DI-': round(last_di_neg, 2),
                'MACD': round(last_macd, 2),
                'MACD_signal': round(last_macd_signal, 2),
                'ATR14': round(last_atr14, 2),
            }
        return None
    except:
        return None

def analyze_ticker_short(ticker, name):
    try:
        df_d = fetch_ohlcv(ticker)
        if df_d.shape[0] < 50:
            return None
        df_d = calc_indicators(df_d)
        weekly_20_sma = get_weekly_20_sma(df_d)
        if not np.isfinite(weekly_20_sma):
            return None
        last_close = to_float(df_d['Close'].iloc[-1])
        prev_close = to_float(df_d['Close'].iloc[-2])
        last_sma20 = to_float(df_d['SMA20'].iloc[-1])
        last_adx14 = to_float(df_d['ADX14'].iloc[-1])
        last_di_pos = to_float(df_d['DI+'].iloc[-1])
        last_di_neg = to_float(df_d['DI-'].iloc[-1])
        last_macd = to_float(df_d['MACD'].iloc[-1])
        last_macd_signal = to_float(df_d['MACD_signal'].iloc[-1])
        last_atr14 = to_float(df_d['ATR14'].iloc[-1])
        values = [last_close, last_sma20, prev_close, last_adx14, last_macd, last_macd_signal, last_atr14, weekly_20_sma, last_di_pos, last_di_neg]
        if not all([isinstance(x, float) and np.isfinite(x) for x in values]):
            return None
        cond_cross_sma = (last_sma20 < weekly_20_sma) and (df_d['SMA20'].iloc[-2] >= weekly_20_sma)
        if cond_cross_sma:
            return {
                'Name': name,
                'Close': round(last_close, 2),
                'PrevClose': round(prev_close, 2),
                'SMA20(D)': round(last_sma20, 2),
                'SMA20(W)': round(weekly_20_sma, 2),
                'ADX14': round(last_adx14, 2),
                'DI+': round(last_di_pos, 2),
                'DI-': round(last_di_neg, 2),
                'MACD': round(last_macd, 2),
                'MACD_signal': round(last_macd_signal, 2),
                'ATR14': round(last_atr14, 2),
            }
        return None
    except:
        return None

def di_color(di_val, is_plus):
    try:
        di = float(di_val)
        if is_plus and di > 20:
            return "#18AA47"  # green
        elif (not is_plus) and di < 20:
            return "#E53935"  # red
        else:
            return "#ECECEC"
    except:
        return "#ECECEC"

# ... [imports and all your helper and strategy code remain unchanged] ...

def tile_html_stock_monthly(s, is_short=False):
    price_color, arrow, price_change, price_pct = price_color_and_arrow_monthly(s.get('Close'), s.get('PrevClose'))
    stock_name = s.get('Name', '')
    price_str = f"₹ {s.get('Close', '')} {arrow}"
    pct_str = f"{'+' if price_pct > 0 else ''}{round(price_pct,2)}%" if isinstance(price_pct, float) else price_pct
    price_change_str = f"{'+' if price_change > 0 else ''}{round(price_change,2)}" if isinstance(price_change, float) else price_change
    di_plus = s.get('DI+','')
    di_minus = s.get('DI-','')
    di_plus_col = di_color(di_plus, True)
    di_minus_col = di_color(di_minus, False)
    if is_short:
        sma_stuff = f"""<span style="font-size:1.02em;color:#ECECEC;">W: <b>{s.get('SMA20(W)','')}</b><br>D: <b>{s.get('SMA20(D)','')}</b></span>"""
    else:
        sma_stuff = f"""<span style="font-size:1.02em;color:#ECECEC;">D: <b>{s.get('SMA20(D)','')}</b><br>W: <b>{s.get('SMA20(W)','')}</b></span>"""
    return f"""
    <div style="background:#252525;border-radius:16px;width:340px;height:170px;position:relative;box-shadow:1px 2px 10px #111;margin-bottom:20px;display:flex;flex-direction:column;align-items:center;border:1px solid #333;">
      <div style="font-size:1.08em;color:#FFFFFF;font-weight:700;text-align:center;width:100%;margin-top:8px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">{stock_name}</div>
      <div style="width:100%;text-align:center;margin-top:2px;">
        <span style="font-size:1.22em;color:{price_color};font-weight:700;">
            {price_str}
        </span>
        <span style="font-size:1em; margin-left:10px; color:{price_color}; font-weight:700;">
            {price_change_str} ({pct_str})
        </span>
      </div>
      <div style="width:100%;display:flex;justify-content:space-between;margin-top:7px;">
        <div style="text-align:left; margin-left:22px;">
            {sma_stuff}
        </div>
        <div style="text-align:right; margin-right:22px;">
            <span style="font-size:1em; color:#ECECEC;">ATR: <b>{s.get('ATR14','')}</b></span><br>
            <span style="font-size:1em; color:#FFD700; font-weight:700;">MACD:</span>
            <span style="font-size:1em; color:#FFD700; font-weight:700;">{s.get('MACD','')}</span>
            <span style="font-size:1em; color:#FFA500; font-weight:700;">{s.get('MACD_signal','')}</span>
        </div>
      </div>
      <div style="position:absolute;bottom:6px;width:100%;text-align:center;">
        <span style="font-size:1.01em; color:{di_plus_col}; font-weight:700;">DI+ {di_plus}</span>
        <span style="font-size:1.01em; color:#FFD700; font-weight:700; margin-left:14px;">ADX {s.get('ADX14','')}</span>
        <span style="font-size:1.01em; color:{di_minus_col}; font-weight:700; margin-left:14px;">DI- {di_minus}</span>
      </div>
    </div>
    """

# ... [rest of dashboard code remains unchanged, including get_dailyweekly_data, section rendering, etc.] ...

@st.cache_data(show_spinner="Loading data for Daily > Weekly...")
def get_dailyweekly_data():
    fo_df = pd.read_csv('fo_stock_list.csv')
    long_results, short_results = [], []
    for idx, row in fo_df.iterrows():
        symbol = row['symbol']
        name = get_first_two(row['name']) if 'name' in row else symbol
        long_res = analyze_ticker_long(symbol, name)
        if long_res:
            long_results.append(long_res)
        short_res = analyze_ticker_short(symbol, name)
        if short_res:
            short_results.append(short_res)
    return long_results, short_results

if selected == "dailyweekly":
    if st.button("🔄 Refresh Daily > Weekly Data", key="refresh_dailyweekly"):
        get_dailyweekly_data.clear()
    long_results, short_results = get_dailyweekly_data()
    st.markdown("<h2 style='font-size:1.30em; text-align:center; margin-bottom:10px;color:#FFD700;'>Daily > Weekly Strategy</h2>", unsafe_allow_html=True)
    cols = st.columns(2)
    section_titles = ["Long", "Short"]
    section_colors = ["#18AA47", "#E53935"]
    section_dots = ["#80D8FF", "#FFA500"]
    section_tiles = [long_results, short_results]
    for idx, col in enumerate(cols):
        col.markdown(f'''
            <div style="background:{section_colors[idx]};padding:13px 0 13px 0;border-radius:13px;margin-bottom:12px;text-align:center;width:99%;">
            <span style="color:{section_dots[idx]};font-size:1.42em;font-weight:700;">&#x25CF;</span>
            <span style="color:#FFFFFF;font-size:1.19em;font-weight:700;letter-spacing:2px;">{section_titles[idx]}</span></div>
        ''', unsafe_allow_html=True)
        tiles = section_tiles[idx]
        for i in range(0, len(tiles), 2):
            row_tiles = tiles[i:i+2]
            tile_cols = col.columns(2)
            for tcol, s in zip(tile_cols, row_tiles):
                tcol.markdown(tile_html_stock_monthly(s, is_short=(idx==1)), unsafe_allow_html=True)
        if not tiles:
            col.write("No stocks matched.")

elif selected == "rsi":
    if st.button("🔄 Refresh RSI Data", key="refresh_rsi"):
        get_rsi_strategy_data.clear()
    df = get_rsi_strategy_data()
    def card(row, rsi_green=True):
        price = row['Latest Price']
        prev = row['Previous Price']
        arrow, pcolor, change, pct = price_arrow_and_change(price, prev)
        price_str = f"{price:.2f}" if pd.notnull(price) else "NA"
        change_str = f"{change:+.2f}" if pd.notnull(price) and pd.notnull(prev) else "NA"
        pct_str = f"{pct:+.2f}%" if pd.notnull(price) and pd.notnull(prev) else "NA"
        name_html = f'<span style="color:#fff;font-size:1.13em;text-align:center;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{get_first_two(row["Stock Name"])}</span>'
        rsi = row["Daily RSI"]
        if pd.isna(rsi):
            rsi_txt = '<span style="font-weight:bold;color:#fff;">RSI: NA</span>'
        elif rsi_green:
            rsi_txt = f'<span style="color:#37F553;font-weight:bold;">RSI: {round(rsi,2)}</span>'
        else:
            rsi_txt = f'<span style="color:#FF3A3A;font-weight:bold;">RSI: {round(rsi,2)}</span>'
        lot_line = f'<span style="font-size:0.94em;">Lot: <b>{row["Lot Size"]}</b></span>&nbsp;&nbsp;{rsi_txt}'
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
                <br><br>
                {lot_line}
            </div>
        """
    left_group = df[df["Daily RSI"] >= 55].reset_index(drop=True)
    right_group = df[df["Daily RSI"] < 55].reset_index(drop=True)
    max_len = max(len(left_group), len(right_group))
    rows_needed = (max_len + 3) // 4
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
        prev_str = f"{s['prev_price']:.2f}" if isinstance(s['prev_price'], (float, int)) and s['prev_price'] not in ["NA", None] else s['prev_price']
        change_str = f"{s['change']:+.2f}" if isinstance(s['change'], (float, int)) and s['change'] not in ["NA", None] else s['change']
        pct_str = f"{s['pct']:+.2f}%" if isinstance(s['pct'], (float, int)) and s['pct'] not in ["NA", None] else s['pct']
        wk_rsi_str = f"{s['wk_rsi']:.2f}" if isinstance(s['wk_rsi'], (float, int)) and s['wk_rsi'] not in ["NA", None] else s['wk_rsi']
        d_rsi_str = f"{s['d_rsi']:.2f}" if isinstance(s['d_rsi'], (float, int)) and s['d_rsi'] not in ["NA", None] else s['d_rsi']

        return f"""
        <div style="background:#252525;border-radius:15px;width:260px;height:180px;position:relative;box-shadow:1px 2px 8px #111;margin-bottom:18px;display:flex;flex-direction:column;align-items:center;border:1px solid #333;">
          <div style="font-size:1.02em;color:#fff;font-weight:700;text-align:center;width:100%;margin-top:13px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">{s['name']}</div>
          <div style="width:100%;text-align:center;margin-top:7px;margin-bottom:0;">
            <span style="font-size:1.07em;color:{s['arrow_color']};font-weight:700;">
                ₹ {price_str}
                <span style="font-size:1.08em;">{s['arrow']}</span>
                <span style="margin-left:6px;font-size:0.94em">{change_str} ({pct_str})</span>
            </span>
          </div>
          <div style="width:100%;text-align:center;font-size:0.97em;color:#ECECEC;margin-bottom:2px;">
            Prev Close: ₹ {prev_str}
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

elif selected not in ["rsi", "ssstrat", "dailyweekly"]:
    st.info("This strategy dashboard is coming soon!")

else:
    st.markdown("> **Select a strategy tile above to view its details.**")

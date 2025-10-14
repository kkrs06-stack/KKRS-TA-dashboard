import pandas as pd
import numpy as np
import ta
import yfinance as yf
import streamlit as st
import datetime
import warnings
import sys
import os
from ta.trend import ADXIndicator
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

def rsi_colored_custom(rsi):
    try:
        val = float(rsi)
        if val > 55:
            color = "#37F553"
        elif val > 50:
            color = "#FFD700"
        else:
            color = "#FF3A3A"
        return f"<span style='color:{color};font-weight:700;font-size:1.08em;'>{val:.2f}</span>"
    except:
        return f"<span style='color:#ECECEC;font-weight:700;'>{rsi}</span>"

def macd_tradingview(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def obv_calc(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def obv_trend_arrow(df, period):
    obv = obv_calc(df)
    if len(obv) < period + 1:
        return "<span style='font-size:1.15em;color:#FF69B4;'>–</span>"
    obv_now = obv.iloc[-1]
    obv_prev = obv.iloc[-period-1]
    flat_threshold = 0.005 * abs(obv_now)
    diff = obv_now - obv_prev
    if diff > flat_threshold:
        return "<span style='font-size:1.15em;color:#37F553;'>↑</span>"
    elif diff < -flat_threshold:
        return "<span style='font-size:1.15em;color:#FF3A3A;'>↓</span>"
    else:
        return "<span style='font-size:1.15em;color:#FF69B4;'>–</span>"

def supertrend_tradingview_wilder(df, period=10, multiplier=3.0):
    if len(df) < period:
        return None, None
    high = df['High']
    low = df['Low']
    close = df['Close']
    hl2 = (high + low) / 2
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    up = hl2 - (multiplier * atr)
    dn = hl2 + (multiplier * atr)
    final_up = up.copy()
    final_dn = dn.copy()
    trend = pd.Series(index=df.index, dtype=int)
    trend.iloc[0] = 1
    st_val = pd.Series(index=df.index, dtype=float)
    for i in range(1, len(df)):
        if close.iloc[i-1] > final_up.iloc[i-1]:
            final_up.iloc[i] = max(up.iloc[i], final_up.iloc[i-1])
        else:
            final_up.iloc[i] = up.iloc[i]
        if close.iloc[i-1] < final_dn.iloc[i-1]:
            final_dn.iloc[i] = min(dn.iloc[i], final_dn.iloc[i-1])
        else:
            final_dn.iloc[i] = dn.iloc[i]
        if trend.iloc[i-1] == -1 and close.iloc[i] > final_dn.iloc[i-1]:
            trend.iloc[i] = 1
        elif trend.iloc[i-1] == 1 and close.iloc[i] < final_up.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    for i in range(len(df)):
        st_val.iloc[i] = final_up.iloc[i] if trend.iloc[i] == 1 else final_dn.iloc[i]
    return st_val.round(2), trend.map({1: 'UP', -1: 'DOWN'})

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

def fetch_ohlcv(ticker):
    df_d = yf.download(ticker, period='500d', interval='1d', progress=False)
    if isinstance(df_d.columns, pd.MultiIndex):
        df_d.columns = [col[0] for col in df_d.columns]
    df_d = df_d.dropna(subset=['High', 'Low', 'Close', 'Volume'])
    df_d = df_d[~df_d.index.duplicated(keep='first')]
    if 'Volume' in df_d.columns:
        df_d = df_d[df_d['Volume'] > 0]
    df_d = df_d[~((df_d['Open'] == df_d['High']) & (df_d['High'] == df_d['Low']) & (df_d['Low'] == df_d['Close']))]
    return df_d

def ema50_comparison_row(ema50, sma20, section_type):
    if section_type == 'long':
        if ema50 > sma20:
            yn = "Y"
            color = "#FF3A3A"
        else:
            yn = "N"
            color = "#37F553"
    else:
        if sma20 > ema50:
            yn = "N"
            color = "#FF3A3A"
        else:
            yn = "Y"
            color = "#37F553"
    return f"<span style='font-weight:700;'>50: <b>{ema50:.2f}</b> <span style='color:{color};font-size:1.19em;'>{yn}</span></span>"

def two_percent_label(ema50, sma20, section_type):
    try:
        ema50 = float(ema50)
        sma20 = float(sma20)
        if section_type == 'long':
            pct_diff = ((ema50 - sma20) / sma20) * 100
            yn = "Y" if ema50 > sma20 else "N"
            color = "#37F553" if yn == "Y" else "#FF3A3A"
        else:
            pct_diff = ((sma20 - ema50) / ema50) * 100
            yn = "N" if sma20 > ema50 else "Y"
            color = "#FF3A3A" if yn == "N" else "#37F553"
        pct_label = f"{pct_diff:+.2f}%"
        return (f"<span style='font-weight:700;'>2%: "
                f"<span style='color:{color};'>{yn}</span> "
                f"<span style='color:#FFD700'>({pct_label})</span></span>")
    except:
        return "<span style='font-weight:700;'>2%: <span style='color:#FF3A3A;'>N</span> <span style='color:#FFD700;'>(NA)</span></span>"

def gap_ema50_label(price_now, ema50, sma20, section_type):
    try:
        price_now = float(price_now)
        ema50 = float(ema50)
        sma20 = float(sma20)
        show_gap = False
        if section_type == 'long' and ema50 > sma20:
            show_gap = True
        elif section_type == 'short' and ema50 < sma20:
            show_gap = True
        if show_gap:
            gap_pct = ((price_now - ema50) / ema50) * 100
            color = "#37F553" if gap_pct >= 0 else "#FF3A3A"
            arrow = "▲" if gap_pct >= 0 else "▼"
            return f"<span style='font-weight:700;'>Gap: <span style='color:{color};'>{arrow} {gap_pct:+.2f}%</span></span>"
        else:
            return ""
    except:
        return ""

def calc_indicators(df_d):
    adx = ADXIndicator(df_d['High'], df_d['Low'], df_d['Close'], 14)
    df_d['ADX14'] = adx.adx()
    df_d['DI+'] = adx.adx_pos()
    df_d['DI-'] = adx.adx_neg()
    df_d['SMA20'] = df_d['Close'].rolling(20).mean()
    df_d['EMA50'] = df_d['Close'].ewm(span=50).mean()
    macd_line, signal_line = macd_tradingview(df_d['Close'])
    df_d['MACD'] = macd_line
    df_d['MACD_signal'] = signal_line
    atr = AverageTrueRange(df_d['High'], df_d['Low'], df_d['Close'], 14)
    df_d['ATR14'] = atr.average_true_range()
    return df_d

def analyze_sma20_long(ticker, name, lot):
    try:
        df_d = fetch_ohlcv(ticker)
        if df_d.shape[0] < 60:
            return None
        df_d = calc_indicators(df_d)
        st_series, st_dir = supertrend_tradingview_wilder(df_d, period=10, multiplier=3.0)
        if st_series is None:
            return None
        close_now = to_float(df_d['Close'].iloc[-1])
        close_prev = to_float(df_d['Close'].iloc[-2])
        sma_now = to_float(df_d['SMA20'].iloc[-1])
        sma_prev = to_float(df_d['SMA20'].iloc[-2])
        ema_now = to_float(df_d['EMA50'].iloc[-1])
        supertrend_val = st_series.iloc[-1]
        supertrend_dir = st_dir[-1]
        cond_cross_above = (close_now > sma_now) and (close_prev <= sma_prev)
        if cond_cross_above:
            last_adx14 = to_float(df_d['ADX14'].iloc[-1])
            last_di_pos = to_float(df_d['DI+'].iloc[-1])
            last_di_neg = to_float(df_d['DI-'].iloc[-1])
            last_macd = to_float(df_d['MACD'].iloc[-1])
            last_macd_signal = to_float(df_d['MACD_signal'].iloc[-1])
            last_atr14 = to_float(df_d['ATR14'].iloc[-1])
            daily_rsi = ta.momentum.RSIIndicator(df_d['Close'], window=14).rsi().dropna().iloc[-1] if df_d['Close'].dropna().size > 15 else "NA"
            return {
                'Name': name,
                'Symbol': ticker,
                'Lot': lot,
                'D': round(sma_now, 2),
                'EMA50': round(ema_now, 2),
                'SuperTrend': round(supertrend_val, 2) if pd.notna(supertrend_val) else "NA",
                'SuperTrendDir': supertrend_dir,
                'Close': round(close_now,2),
                'PrevClose': round(close_prev,2),
                'ADX14': round(last_adx14, 2),
                'DI+': round(last_di_pos, 2),
                'DI-': round(last_di_neg, 2),
                'MACD': round(last_macd, 2),
                'MACD_signal': round(last_macd_signal, 2),
                'ATR14': round(last_atr14, 2),
                'Daily_RSI': round(daily_rsi,2) if isinstance(daily_rsi, float) else daily_rsi,
                'OBV_ARROW': obv_trend_arrow(df_d, 10),
            }
        return None
    except:
        return None

def analyze_sma20_short(ticker, name, lot):
    try:
        df_d = fetch_ohlcv(ticker)
        if df_d.shape[0] < 60:
            return None
        df_d = calc_indicators(df_d)
        st_series, st_dir = supertrend_tradingview_wilder(df_d, period=10, multiplier=3.0)
        if st_series is None:
            return None
        close_now = to_float(df_d['Close'].iloc[-1])
        close_prev = to_float(df_d['Close'].iloc[-2])
        sma_now = to_float(df_d['SMA20'].iloc[-1])
        sma_prev = to_float(df_d['SMA20'].iloc[-2])
        ema_now = to_float(df_d['EMA50'].iloc[-1])
        supertrend_val = st_series.iloc[-1]
        supertrend_dir = st_dir[-1]
        cond_cross_below = (close_now < sma_now) and (close_prev >= sma_prev)
        if cond_cross_below:
            last_adx14 = to_float(df_d['ADX14'].iloc[-1])
            last_di_pos = to_float(df_d['DI+'].iloc[-1])
            last_di_neg = to_float(df_d['DI-'].iloc[-1])
            last_macd = to_float(df_d['MACD'].iloc[-1])
            last_macd_signal = to_float(df_d['MACD_signal'].iloc[-1])
            last_atr14 = to_float(df_d['ATR14'].iloc[-1])
            daily_rsi = ta.momentum.RSIIndicator(df_d['Close'], window=14).rsi().dropna().iloc[-1] if df_d['Close'].dropna().size > 15 else "NA"
            return {
                'Name': name,
                'Symbol': ticker,
                'Lot': lot,
                'D': round(sma_now, 2),
                'EMA50': round(ema_now, 2),
                'SuperTrend': round(supertrend_val, 2) if pd.notna(supertrend_val) else "NA",
                'SuperTrendDir': supertrend_dir,
                'Close': round(close_now, 2),
                'PrevClose': round(close_prev, 2),
                'ADX14': round(last_adx14, 2),
                'DI+': round(last_di_pos, 2),
                'DI-': round(last_di_neg, 2),
                'MACD': round(last_macd, 2),
                'MACD_signal': round(last_macd_signal, 2),
                'ATR14': round(last_atr14, 2),
                'Daily_RSI': round(daily_rsi,2) if isinstance(daily_rsi, float) else daily_rsi,
                'OBV_ARROW': obv_trend_arrow(df_d, 10),
            }
        return None
    except:
        return None

@st.cache_data(show_spinner="Loading data for Just Above/Below...")
def get_sma20_crossover_data():
    fo_df = pd.read_csv('fo_stock_list.csv')
    long_results, short_results = [], []
    for idx, row in fo_df.iterrows():
        symbol = row['symbol']
        lot = row['lot_size'] if 'lot_size' in row else ''
        name = get_first_two(row['name']) if 'name' in row else symbol
        long_res = analyze_sma20_long(symbol, name, lot)
        if long_res:
            long_results.append(long_res)
        short_res = analyze_sma20_short(symbol, name, lot)
        if short_res:
            short_results.append(short_res)
    return long_results, short_results

# ------ Correct M & W logic using LAST daily close compared to last SMA20 in monthly/weekly ------
def get_above_sma_labels(ticker):
    try:
        df_d = fetch_ohlcv(ticker)
        last_close = df_d['Close'].iloc[-1]
        # Monthly
        df_m = df_d['Close'].resample('M').last().dropna().to_frame()
        df_m['SMA20'] = df_m['Close'].rolling(20).mean()
        sma20_m = df_m['SMA20'].iloc[-1] if len(df_m) >= 20 else np.nan
        if not np.isnan(sma20_m):
            m_yn = last_close > sma20_m
            m_val = f"<span style='color:{'#37F553' if m_yn else '#FF3A3A'};font-weight:bold;'>M: {'Y' if m_yn else 'N'}</span>"
        else:
            m_val = "<span style='color:#FFD700;'>M: -</span>"
        # Weekly
        df_w = df_d['Close'].resample('W-FRI').last().dropna().to_frame()
        df_w['SMA20'] = df_w['Close'].rolling(20).mean()
        sma20_w = df_w['SMA20'].iloc[-1] if len(df_w) >= 20 else np.nan
        if not np.isnan(sma20_w):
            w_yn = last_close > sma20_w
            w_val = f"<span style='color:{'#37F553' if w_yn else '#FF3A3A'};font-weight:bold;'>W: {'Y' if w_yn else 'N'}</span>"
        else:
            w_val = "<span style='color:#FFD700;'>W: -</span>"
        return f"{m_val} | {w_val}"
    except:
        return "<span style='color:#FFD700;'>M: - | W: -</span>"

def just_above_below_dashboard():
    today_str = datetime.datetime.now().strftime("%d-%b-%Y")
    st.markdown(
        "<div style='font-size:2em;font-weight:800;color:#FFD700;padding-bottom:2px;'>Just Above/Below</div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:right;font-size:1.2em;color:#FFD700;font-weight:700;padding-top:4px;'>{today_str}</div>",
        unsafe_allow_html=True)

    obv_period = st.slider('OBV Trend Lookback Period (days)', min_value=5, max_value=30, value=10, step=1)

    st.markdown("<h2 style='font-size:1.30em; text-align:center; margin-bottom:10px;color:#FFD700;'>Just Above/Below</h2>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data", key="refresh_sma20"):
        get_sma20_crossover_data.clear()

    long_results, short_results = get_sma20_crossover_data()

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
                lot = s.get('Lot',"")
                name = s.get('Name',"")
                symbol = s.get('Symbol',"")
                price = s.get('Close',"NA")
                prev = s.get('PrevClose',"NA")
                arrow, price_color, price_change, pct = price_arrow_and_change(price, prev)
                price_str = f"₹ {price}" if price != "NA" else "NA"
                change_str = f"{price_change:+.2f}" if isinstance(price_change, float) else price_change
                pct_str = f"{pct:+.2f}%" if isinstance(pct, float) else pct
                daily_rsi_val = s.get('Daily_RSI', 'NA')
                daily_rsi_html = rsi_colored_custom(daily_rsi_val)
                macd = s.get('MACD')
                macd_signal = s.get('MACD_signal')
                di_plus, di_minus = s.get('DI+'), s.get('DI-')
                d_val = s.get('D',0)
                ema50_val = s.get('EMA50',0)
                supertrend_val = s.get('SuperTrend', 'NA')
                supertrend_dir = s.get('SuperTrendDir','NA')
                obv_arrow = s.get('OBV_ARROW', '')

                section_type = 'long' if idx == 0 else 'short'
                ema_comp_html = ema50_comparison_row(ema50_val, d_val, section_type)
                macd_color = "#37F553" if macd > macd_signal else "#FF3A3A"
                gap_label = gap_ema50_label(price, ema50_val, d_val, section_type)
                mw_label = get_above_sma_labels(symbol)

                if supertrend_dir == "UP":
                    st_dir_col = "#37F553"
                elif supertrend_dir == "DOWN":
                    st_dir_col = "#FF3A3A"
                else:
                    st_dir_col = "#FFD700"
                supertrend_html = f"ST: <b>{supertrend_val}</b> <span style='color:{st_dir_col};font-weight:900;font-size:1.12em'>{supertrend_dir}</span>"

                di_plus_html = f"<span style='color:#18AA47;font-size:1em;'>DI+ {di_plus}</span>"
                di_minus_html = f"<span style='color:#E53935;font-size:1em;'>DI- {di_minus}</span>"
                if idx == 0:
                    if di_plus is not None and di_minus is not None:
                        if di_plus > di_minus:
                            di_plus_html = f"<span style='color:#18AA47;font-size:1.22em;font-weight:900;'>DI+ {di_plus}</span>"
                        if di_minus < 20 and di_plus > 20:
                            di_minus_html = f"<span style='color:#E53935;font-size:1.22em;font-weight:900;'>DI- {di_minus}</span>"
                elif idx == 1:
                    if di_plus is not None and di_minus is not None:
                        if di_minus > di_plus:
                            di_minus_html = f"<span style='color:#E53935;font-size:1.22em;font-weight:900;'>DI- {di_minus}</span>"
                        if di_minus > 20 and di_plus < 20:
                            di_minus_html = f"<span style='color:#E53935;font-size:1.22em;font-weight:900;'>DI- {di_minus}</span>"

                tview_url = f"https://www.tradingview.com/chart/lDI0poON/?symbol=NSE:{symbol.replace('.NS','')}"

                left_rows = [
                    f"D: <b>{d_val}</b>",
                    ema_comp_html,
                    supertrend_html,
                    f"RSI: {daily_rsi_html}",
                    f"OBV: {obv_arrow}",
                ]

                right_rows = [
                    f"ATR: <b>{s.get('ATR14','')}</b>",
                    f"MACD: <span style='color:{macd_color};font-weight:700;'>{macd}</span>",
                    f"Signal: <span style='color:#FFA500;font-weight:700;'>{macd_signal}</span>",
                    two_percent_label(ema50_val, d_val, section_type),
                    gap_label,
                    mw_label
                ]

                left_html = "".join([f"<div style='font-size:1.03em;color:#ECECEC;margin-bottom:2px;'>{row}</div>" for row in left_rows])
                right_html = "".join([f"<div style='font-size:1.03em;margin-bottom:2px;'>{row}</div>" for row in right_rows])

                tcol.markdown(f"""
                <div style="background:#252525;border-radius:16px;width:340px;height:300px;position:relative;box-shadow:1px 2px 10px #111;margin-bottom:20px;display:flex;flex-direction:column;align-items:center;border:1px solid #333;">
                  <div style="width:100%;text-align:center;margin-top:8px;">
                    <a href="{tview_url}" target="_blank" style="color:#fff;font-size:1.15em;font-weight:700;text-decoration:none;">
                      {name}
                    </a>
                  </div>
                  <div style="width:100%; position:relative;">
                    <div style="position:absolute;right:20px;top:-12px; font-size:0.93em;color:#ECECEC;text-align:right;">
                        Lot: <span style="font-weight:bold;">{lot}</span>
                    </div>
                  </div>
                  <div style="width:100%;text-align:center;margin-top:5px;margin-bottom:5px;">
                    <span style="font-size:1.16em;color:{price_color};font-weight:700;">
                      {price_str}
                      <span style="font-size:1.13em;">{arrow}</span>
                      <span style="color:{price_color};margin-left:6px;font-size:1em">{change_str} ({pct_str})</span>
                    </span>
                  </div>
                  <div style="display:flex;flex-direction:row;width:100%;justify-content:space-between;margin-top:8px;">
                    <div style="padding-left:16px;text-align:left;">
                        {left_html}
                    </div>
                    <div style="padding-right:16px;text-align:right;">
                        {right_html}
                    </div>
                  </div>
                  <div style="position:absolute;bottom:11px;width:100%;text-align:center;">
                    {di_plus_html}
                    &nbsp; {di_minus_html}
                    &nbsp; <span style="font-size:1em; color:#FF1493; font-weight:700;">ADX {s.get('ADX14','')}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        if not tiles:
            col.write("No stocks matched.")

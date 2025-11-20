import pandas as pd
import numpy as np
import ta
import yfinance as yf
import streamlit as st
import datetime
import warnings
from ta.trend import ADXIndicator, MACD
from ta.volatility import AverageTrueRange

warnings.filterwarnings("ignore")

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

# ----------- OPTIMIZED: Batch Download -----------
@st.cache_data(show_spinner="Batch downloading all daily/weekly OHLCV...")
def fetch_all_ohlcv(ticker_list):
    tickers_str = " ".join(ticker_list)
    df = yf.download(tickers=tickers_str, period='350d', interval='1d', group_by='ticker', auto_adjust=True, progress=False, threads=True)
    data_dict = {}
    if isinstance(df.columns, pd.MultiIndex):
        for symbol in ticker_list:
            if symbol in df:
                sdf = df[symbol].dropna().copy()
                sdf = sdf[~sdf.index.duplicated(keep='first')]
                data_dict[symbol] = sdf
    else:
        sdf = df.dropna().copy()
        sdf = sdf[~sdf.index.duplicated(keep='first')]
        data_dict[ticker_list[0]] = sdf
    return data_dict

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

def get_above_sma_labels(symbol, data_dict):
    try:
        df_d = data_dict.get(symbol)
        if df_d is None:
            return "<span style='color:#FFD700;'>M: - | W: -</span>"
        last_close = df_d['Close'].iloc[-1]
        df_m = df_d['Close'].resample('M').last().dropna().to_frame()
        df_m['SMA20'] = df_m['Close'].rolling(20).mean()
        sma20_m = df_m['SMA20'].iloc[-1] if len(df_m) >= 20 else np.nan
        if not np.isnan(sma20_m):
            m_yn = last_close > sma20_m
            m_val = f"<span style='color:{'#37F553' if m_yn else '#FF3A3A'};font-weight:bold;'>M: {'Y' if m_yn else 'N'}</span>"
        else:
            m_val = "<span style='color:#FFD700;'>M: -</span>"
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

# ----------- OPTIMIZED: Main Analysis -----------
@st.cache_data(show_spinner="Calculating daily/weekly tiles...")
def batch_analyze_all(data_dict, fo_df):
    long_results, short_results = [], []
    for idx, row in fo_df.iterrows():
        symbol = row['symbol']
        lot = row['lot_size'] if 'lot_size' in row else ''
        name = get_first_two(row['name']) if 'name' in row else symbol
        df_d = data_dict.get(symbol)
        if df_d is None or df_d.shape[0] < 50:
            continue
        df_d = calc_indicators(df_d)
        weekly_20_sma = get_weekly_20_sma(df_d)
        last_close = to_float(df_d['Close'].iloc[-1])
        prev_close = to_float(df_d['Close'].iloc[-2])
        last_sma20 = to_float(df_d['SMA20'].iloc[-1])
        last_adx14 = to_float(df_d['ADX14'].iloc[-1])
        last_di_pos = to_float(df_d['DI+'].iloc[-1])
        last_di_neg = to_float(df_d['DI-'].iloc[-1])
        last_macd = to_float(df_d['MACD'].iloc[-1])
        last_macd_signal = to_float(df_d['MACD_signal'].iloc[-1])
        last_atr14 = to_float(df_d['ATR14'].iloc[-1])
        daily_rsi = ta.momentum.RSIIndicator(df_d['Close'], window=14).rsi().dropna().iloc[-1] if df_d['Close'].dropna().size > 15 else "NA"
        if not np.isfinite(weekly_20_sma):
            continue
        # Long
        diff_pct_l = ((last_sma20 - weekly_20_sma) / weekly_20_sma) * 100 if weekly_20_sma else 0
        one_percent_bool_l = diff_pct_l >= 1
        cond_cross_sma_l = (last_sma20 > weekly_20_sma) and (df_d['SMA20'].iloc[-2] <= weekly_20_sma)
        if cond_cross_sma_l:
            long_results.append({
                'Name': name,
                'Symbol': symbol,
                'Lot': lot,
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
                '1pct': one_percent_bool_l,
                'Daily_RSI': round(daily_rsi,2) if isinstance(daily_rsi, float) else daily_rsi,
            })
        # Short
        diff_pct_s = ((weekly_20_sma - last_sma20) / weekly_20_sma) * 100 if weekly_20_sma else 0
        one_percent_bool_s = diff_pct_s >= 1
        cond_cross_sma_s = (last_sma20 < weekly_20_sma) and (df_d['SMA20'].iloc[-2] >= weekly_20_sma)
        if cond_cross_sma_s:
            short_results.append({
                'Name': name,
                'Symbol': symbol,
                'Lot': lot,
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
                '1pct': one_percent_bool_s,
                'Daily_RSI': round(daily_rsi,2) if isinstance(daily_rsi, float) else daily_rsi,
            })
    return long_results, short_results

# ----------- Optimized Tab UI -----------
def daily_weekly_dashboard():
    fo_df = pd.read_csv('fo_stock_list.csv')
    ticker_list = list(fo_df['symbol'])
    if st.button("🔄 Refresh Daily > Weekly Data", key="refresh_dailyweekly"):
        fetch_all_ohlcv.clear()
        batch_analyze_all.clear()
    data_dict = fetch_all_ohlcv(ticker_list)
    long_results, short_results = batch_analyze_all(data_dict, fo_df)
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
                price = s.get('Close',"NA")
                prev = s.get('PrevClose',"NA")
                arrow, price_color, price_change, pct = price_arrow_and_change(price, prev)
                price_str = f"₹ {price}" if price != "NA" else "NA"
                change_str = f"{price_change:+.2f}" if isinstance(price_change, float) else price_change
                pct_str = f"{pct:+.2f}%" if isinstance(pct, float) else pct
                d = s.get('SMA20(D)')
                w = s.get('SMA20(W)')
                lot = s.get('Lot',"")
                name = s.get('Name',"")
                symbol = s.get('Symbol',"")
                tv_url = f"https://www.tradingview.com/chart/lDI0poON/?symbol=NSE:{symbol.replace('.NS','')}"
                daily_rsi_val = s.get('Daily_RSI','NA')
                daily_rsi = rsi_colored(daily_rsi_val)
                mw_label = get_above_sma_labels(symbol, data_dict)
                try:
                    if d is not None and w is not None and isinstance(d, (int, float)) and isinstance(w, (int, float)):
                        if idx == 0:
                            one_pct = ((d - w) / w * 100) if w != 0 else 0
                            one_percent_y = one_pct >= 1
                        else:
                            one_pct = ((w - d) / w * 100) if w != 0 else 0
                            one_percent_y = one_pct >= 1
                    else:
                        one_percent_y = False
                except:
                    one_percent_y = False
                one_percent_value = "Y" if one_percent_y else "N"
                one_percent_color = "#18AA47" if one_percent_y else "#E53935"
                if idx == 0:
                    left_rows = [
                        f"D: <b>{d}</b>",
                        f"W: <b>{w}</b>",
                        f"<span style='color:{one_percent_color};'>1%: <b>{one_percent_value}</b></span>",
                        f"RSI: {daily_rsi}",
                    ]
                else:
                    left_rows = [
                        f"W: <b>{w}</b>",
                        f"D: <b>{d}</b>",
                        f"<span style='color:{one_percent_color};'>1%: <b>{one_percent_value}</b></span>",
                        f"RSI: {daily_rsi}",
                    ]
                right_rows = [
                    f"ATR: <b>{s.get('ATR14','')}</b>",
                    f"MACD: <span style='color:#FFD700;font-weight:700;'>{s.get('MACD','')}</span>",
                    f"Signal: <span style='color:#FFA500;font-weight:700;'>{s.get('MACD_signal','')}</span>",
                    mw_label
                ]
                tcol.markdown(f"""
                <div style="background:#252525;border-radius:16px;width:340px;height:245px;position:relative;box-shadow:1px 2px 10px #111;margin-bottom:20px;display:flex;flex-direction:column;align-items:center;border:1px solid #333;">
                  <div style="width:100%;text-align:center;margin-top:8px;">
                    <a href="{tv_url}" target="_blank" style="color:#fff;text-decoration:none;font-weight:700;font-size:1.18em;">{name}</a>
                  </div>
                  <div style="width:100%; position:relative;">
                    <div style="position:absolute;right:20px;top:-12px; font-size:0.93em;color:#ECECEC;text-align:right;">
                        Lot: <span style="font-weight:bold;">{lot}</span>
                    </div>
                  </div>
                  <div style="width:100%;text-align:center;margin-top:10px;">
                    <span style="font-size:1.16em;color:{price_color};font-weight:700;">
                      {price_str}
                      <span style="font-size:1.13em;">{arrow}</span>
                      <span style="color:{price_color};margin-left:8px;font-size:1em">{change_str} ({pct_str})</span>
                    </span>
                  </div>
                  <div style="display:flex;flex-direction:row;width:100%;justify-content:space-between;margin-top:10px;">
                    <div style="padding-left:16px;text-align:left;">
                        <div style="font-size:1.03em;color:#ECECEC;margin-bottom:2px;">{left_rows[0]}</div>
                        <div style="font-size:1.03em;color:#ECECEC;margin-bottom:2px;">{left_rows[1]}</div>
                        <div style="font-size:1.03em;margin-bottom:2px;">{left_rows[2]}</div>
                        <div style="font-size:1.03em;margin-bottom:2px;">{left_rows[3]}</div>
                    </div>
                    <div style="padding-right:16px;text-align:right;">
                        <div style="font-size:1.03em;color:#FFD700;margin-bottom:2px;">{right_rows[0]}</div>
                        <div style="font-size:1.03em;color:#FFD700;margin-bottom:2px;">{right_rows[1]}</div>
                        <div style="font-size:1.03em;color:#FFA500;margin-bottom:2px;">{right_rows[2]}</div>
                        <div style="font-size:1.03em;margin-bottom:2px;">{right_rows[3]}</div>
                    </div>
                  </div>
                  <div style="position:absolute;bottom:8px;width:100%;text-align:center;">
                    <span style="font-size:1.01em; color:#18AA47; font-weight:700;">DI+ {s.get('DI+','')}</span>
                    &nbsp; <span style="font-size:1.01em; color:#E53935; font-weight:700;">DI- {s.get('DI-','')}</span>
                    &nbsp; <span style="font-size:1.01em; color:#FF1493; font-weight:700;">ADX {s.get('ADX14','')}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        if not tiles:
            col.write("No stocks matched.")

# ---------- END OF CODE ----------

# You only need to import daily_weekly_dashboard (and optionally fetch_all_ohlcv, batch_analyze_all if you want to clear cache externally).

import pandas as pd
import numpy as np
import ta
import yfinance as yf
import streamlit as st
import datetime
import warnings
from ta.trend import ADXIndicator, MACD
from ta.volatility import AverageTrueRange
import scipy.signal

warnings.filterwarnings("ignore")


# ========== NEW: Core Enhancement Functions ==========

def calculate_support_resistance_fib(df, window=20):
    try:
        highs = df['High'].tail(window)
        lows = df['Low'].tail(window)
        swing_high = highs.max()
        swing_low = lows.min()
        week_52_high = df['High'].tail(252).max()
        week_52_low = df['Low'].tail(252).min()
        fib_range = week_52_high - week_52_low
        fib_levels = {
            'fib_0': week_52_low,
            'fib_236': week_52_low + (fib_range * 0.236),
            'fib_382': week_52_low + (fib_range * 0.382),
            'fib_500': week_52_low + (fib_range * 0.500),
            'fib_618': week_52_low + (fib_range * 0.618),
            'fib_786': week_52_low + (fib_range * 0.786),
            'fib_100': week_52_high
        }
        all_support_levels = [swing_low, week_52_low] + [fib_levels[k] for k in ['fib_236', 'fib_382', 'fib_500']]
        all_resistance_levels = [swing_high, week_52_high] + [fib_levels[k] for k in ['fib_618', 'fib_786']]
        return {
            'support_levels': sorted(all_support_levels),
            'resistance_levels': sorted(all_resistance_levels, reverse=True),
            'swing_high': swing_high,
            'swing_low': swing_low,
            'week_52_high': week_52_high,
            'week_52_low': week_52_low,
            'fib_levels': fib_levels
        }
    except:
        return None


def get_sr_status(price, sr_data, threshold=0.02):
    if sr_data is None:
        return "⚪ Unknown", None, None
    try:
        price = float(price)
        dist_to_52w_high = ((sr_data['week_52_high'] - price) / price) * 100
        dist_to_52w_low = ((price - sr_data['week_52_low']) / price) * 100
        if abs(dist_to_52w_high) < 2:
            return "⚠️ Near 52W High", sr_data['week_52_high'], dist_to_52w_high
        if abs(dist_to_52w_low) < 2:
            return "⚠️ Near 52W Low", sr_data['week_52_low'], -dist_to_52w_low
        for support in sr_data['support_levels']:
            dist = ((price - support) / support) * 100
            if abs(dist) < threshold * 100:
                return f"🟢 At Support", support, dist
        for resistance in sr_data['resistance_levels']:
            dist = ((resistance - price) / price) * 100
            if abs(dist) < threshold * 100:
                return f"🔴 At Resistance", resistance, dist
        return "🟡 Mid-range", None, None
    except:
        return "⚪ Unknown", None, None


def check_volume_status(df, lookback=20):
    try:
        current_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].tail(lookback).mean()
        ratio = current_vol / avg_vol if avg_vol > 0 else 1
        if ratio > 1.5:
            return "✓ High Vol", ratio, "#37F553"
        elif ratio < 0.8:
            return "✗ Low Vol", ratio, "#FF3A3A"
        else:
            return "→ Avg Vol", ratio, "#FFD700"
    except:
        return "⚪ Unknown", 1.0, "#ECECEC"


def detect_obv_divergence_enhanced(df, lookback=14):
    try:
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df_temp = df.copy()
        df_temp['OBV'] = obv
        price = df_temp['Close'].values
        obv_vals = df_temp['OBV'].values
        price_lows = scipy.signal.argrelextrema(price, np.less, order=lookback)[0]
        price_highs = scipy.signal.argrelextrema(price, np.greater, order=lookback)[0]
        obv_lows = scipy.signal.argrelextrema(obv_vals, np.less, order=lookback)[0]
        obv_highs = scipy.signal.argrelextrema(obv_vals, np.greater, order=lookback)[0]
        bull_div = False
        if len(price_lows) >= 2 and len(obv_lows) >= 2:
            if (price[price_lows[-1]] < price[price_lows[-2]] and obv_vals[obv_lows[-1]] > obv_vals[obv_lows[-2]]):
                bull_div = True
        bear_div = False
        if len(price_highs) >= 2 and len(obv_highs) >= 2:
            if (price[price_highs[-1]] > price[price_highs[-2]] and obv_vals[obv_highs[-1]] < obv_vals[obv_highs[-2]]):
                bear_div = True
        obv_now = obv[-1]
        obv_prev = obv[-lookback-1] if len(obv) > lookback else obv[0]
        obv_trend = "↑" if obv_now > obv_prev else "↓" if obv_now < obv_prev else "→"
        if bull_div:
            return f"{obv_trend} 🟢 Bull Div", "bull_div"
        elif bear_div:
            return f"{obv_trend} 🔴 Bear Div", "bear_div"
        else:
            return f"{obv_trend} → No Div", "no_div"
    except:
        return "⚪ Unknown", "unknown"


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


@st.cache_data(show_spinner="Batch downloading all daily/weekly OHLCV...")
def fetch_all_ohlcv(ticker_list):
    tickers_str = " ".join(ticker_list)
    df = yf.download(tickers=tickers_str, period='500d', interval='1d', group_by='ticker', auto_adjust=True, progress=False, threads=True)
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
        sr_data = calculate_support_resistance_fib(df_d, window=20)
        sr_status, sr_level, sr_dist = get_sr_status(last_close, sr_data)
        vol_status, vol_ratio, vol_color = check_volume_status(df_d, lookback=20)
        obv_div_text, obv_div_type = detect_obv_divergence_enhanced(df_d, lookback=14)
        diff_pct_l = ((last_sma20 - weekly_20_sma) / weekly_20_sma) * 100 if weekly_20_sma else 0
        one_percent_bool_l = diff_pct_l >= 1
        cond_cross_sma_l = (last_sma20 > weekly_20_sma) and (df_d['SMA20'].iloc[-2] <= weekly_20_sma)
        if cond_cross_sma_l:
            long_results.append({
                'Name': name, 'Symbol': symbol, 'Lot': lot,
                'Close': round(last_close, 2), 'PrevClose': round(prev_close, 2),
                'SMA20(D)': round(last_sma20, 2), 'SMA20(W)': round(weekly_20_sma, 2),
                'ADX14': round(last_adx14, 2), 'DI+': round(last_di_pos, 2), 'DI-': round(last_di_neg, 2),
                'MACD': round(last_macd, 2), 'MACD_signal': round(last_macd_signal, 2),
                'ATR14': round(last_atr14, 2), '1pct': one_percent_bool_l,
                'Daily_RSI': round(daily_rsi,2) if isinstance(daily_rsi, float) else daily_rsi,
                'SR_Status': sr_status, 'SR_Level': round(sr_level, 2) if sr_level else "NA",
                'SR_Dist': round(sr_dist, 2) if sr_dist else "NA",
                'Vol_Status': vol_status, 'Vol_Ratio': round(vol_ratio, 2), 'Vol_Color': vol_color,
                'OBV_Div': obv_div_text, 'OBV_Div_Type': obv_div_type,
            })
        diff_pct_s = ((weekly_20_sma - last_sma20) / weekly_20_sma) * 100 if weekly_20_sma else 0
        one_percent_bool_s = diff_pct_s >= 1
        cond_cross_sma_s = (last_sma20 < weekly_20_sma) and (df_d['SMA20'].iloc[-2] >= weekly_20_sma)
        if cond_cross_sma_s:
            short_results.append({
                'Name': name, 'Symbol': symbol, 'Lot': lot,
                'Close': round(last_close, 2), 'PrevClose': round(prev_close, 2),
                'SMA20(D)': round(last_sma20, 2), 'SMA20(W)': round(weekly_20_sma, 2),
                'ADX14': round(last_adx14, 2), 'DI+': round(last_di_pos, 2), 'DI-': round(last_di_neg, 2),
                'MACD': round(last_macd, 2), 'MACD_signal': round(last_macd_signal, 2),
                'ATR14': round(last_atr14, 2), '1pct': one_percent_bool_s,
                'Daily_RSI': round(daily_rsi,2) if isinstance(daily_rsi, float) else daily_rsi,
                'SR_Status': sr_status, 'SR_Level': round(sr_level, 2) if sr_level else "NA",
                'SR_Dist': round(sr_dist, 2) if sr_dist else "NA",
                'Vol_Status': vol_status, 'Vol_Ratio': round(vol_ratio, 2), 'Vol_Color': vol_color,
                'OBV_Div': obv_div_text, 'OBV_Div_Type': obv_div_type,
            })
    for result in (long_results + short_results):
        symbol = result['Symbol']
        try:
            df_d = data_dict.get(symbol)
            if df_d is not None and len(df_d) >= 60:
                price = float(df_d['Close'].iloc[-1])
                sma20_d = float(df_d['Close'].rolling(20).mean().iloc[-1])
                daily_ok = price > sma20_d
                weekly_ok = None
                try:
                    df_w_close = df_d['Close'].resample('W-FRI').last().dropna()
                    if len(df_w_close) >= 20:
                        sma20_w = float(df_w_close.rolling(20).mean().iloc[-1])
                        weekly_ok = price > sma20_w
                except:
                    weekly_ok = None
                monthly_ok = None
                try:
                    df_m_close = df_d['Close'].resample('M').last().dropna()
                    if len(df_m_close) >= 20:
                        sma20_m = float(df_m_close.rolling(20).mean().iloc[-1])
                        monthly_ok = price > sma20_m
                except:
                    monthly_ok = None
                result['MTF_Daily'] = "✓" if daily_ok else "✗"
                result['MTF_Weekly'] = "✓" if weekly_ok is True else ("⚪" if weekly_ok is None else "✗")
                result['MTF_Monthly'] = "✓" if monthly_ok is True else ("⚪" if monthly_ok is None else "✗")
                result['MTF_Score'] = sum([1 for x in [daily_ok, weekly_ok, monthly_ok] if x is True])
            else:
                result['MTF_Daily'] = "⚪"
                result['MTF_Weekly'] = "⚪"
                result['MTF_Monthly'] = "⚪"
                result['MTF_Score'] = 0
        except:
            result['MTF_Daily'] = "⚪"
            result['MTF_Weekly'] = "⚪"
            result['MTF_Monthly'] = "⚪"
            result['MTF_Score'] = 0
    return long_results, short_results


def daily_weekly_dashboard():
    fo_df = pd.read_csv('fo_stock_list.csv')
    ticker_list = list(fo_df['symbol'])
    if st.button("🔄 Refresh Daily > Weekly Data", key="refresh_dailyweekly"):
        fetch_all_ohlcv.clear()
        batch_analyze_all.clear()
    data_dict = fetch_all_ohlcv(ticker_list)
    long_results, short_results = batch_analyze_all(data_dict, fo_df)
    st.markdown("<h2 style='font-size:1.30em; text-align:center; margin-bottom:10px;color:#FFD700;'>Daily > Weekly - Enhanced</h2>", unsafe_allow_html=True)
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
                tv_url = f"https://www.tradingview.com/chart/HHuUSOTG/?symbol=NSE:{symbol.replace('.NS','')}"
                daily_rsi_val = s.get('Daily_RSI','NA')
                daily_rsi = rsi_colored(daily_rsi_val)
                mw_label = get_above_sma_labels(symbol, data_dict)
                one_percent_y = s.get('1pct', False)
                one_percent_value = "Y" if one_percent_y else "N"
                one_percent_color = "#18AA47" if one_percent_y else "#E53935"
                sr_status = s.get('SR_Status', '⚪ Unknown')
                sr_level = s.get('SR_Level', 'NA')
                sr_dist = s.get('SR_Dist', 'NA')
                vol_status = s.get('Vol_Status', '⚪ Unknown')
                vol_ratio = s.get('Vol_Ratio', 1.0)
                vol_color = s.get('Vol_Color', '#ECECEC')
                obv_div = s.get('OBV_Div', '⚪ Unknown')
                mtf_d = s.get('MTF_Daily', '⚪')
                mtf_w = s.get('MTF_Weekly', '⚪')
                mtf_m = s.get('MTF_Monthly', '⚪')
                mtf_score = s.get('MTF_Score', 0)
                if sr_level != "NA":
                    sr_display = f"{sr_status} ({sr_dist:+.1f}%)"
                else:
                    sr_display = sr_status
                vol_display = f'{vol_status} <span style="color:{vol_color};">({vol_ratio:.1f}x)</span>'
                mtf_display = f'<span style="font-weight:700;">MTF:</span> {mtf_d} {mtf_w} {mtf_m} <span style="color:#FFD700;">({mtf_score}/3)</span>'
                if idx == 0:
                    left_rows = [f"D: <b>{d}</b>", f"W: <b>{w}</b>", f"<span style='color:{one_percent_color};'>1%: <b>{one_percent_value}</b></span>", f"RSI: {daily_rsi}"]
                else:
                    left_rows = [f"W: <b>{w}</b>", f"D: <b>{d}</b>", f"<span style='color:{one_percent_color};'>1%: <b>{one_percent_value}</b></span>", f"RSI: {daily_rsi}"]
                right_rows = [f"ATR: <b>{s.get('ATR14','')}</b>", f"MACD: <span style='color:#FFD700;font-weight:700;'>{s.get('MACD','')}</span>", f"Signal: <span style='color:#FFA500;font-weight:700;'>{s.get('MACD_signal','')}</span>", mw_label]
                enhanced_rows = [f"<div style='font-size:0.98em;color:#ECECEC;margin-bottom:3px;'>{sr_display}</div>", f"<div style='font-size:0.98em;color:#ECECEC;margin-bottom:3px;'>{vol_display}</div>", f"<div style='font-size:0.98em;color:#ECECEC;margin-bottom:3px;'>OBV: {obv_div}</div>", f"<div style='font-size:0.98em;color:#ECECEC;margin-bottom:3px;'>{mtf_display}</div>"]
                left_html = "".join([f"<div style='font-size:1.03em;color:#ECECEC;margin-bottom:2px;'>{row}</div>" for row in left_rows])
                right_html = "".join([f"<div style='font-size:1.03em;margin-bottom:2px;'>{row}</div>" for row in right_rows])
                enhanced_html = "".join(enhanced_rows)
                tcol.markdown(f"""
                <div style="background:#252525;border-radius:16px;width:380px;height:400px;position:relative;box-shadow:1px 2px 10px #111;margin-bottom:20px;border:1px solid #333;overflow:hidden;">
                  <div style="width:100%;text-align:center;padding-top:6px;">
                    <a href="{tv_url}" target="_blank" style="color:#fff;font-size:1.35em;font-weight:700;text-decoration:none;">{name}</a>
                  </div>
                  <div style="position:absolute;right:20px;top:6px;font-size:0.93em;color:#ECECEC;">Lot: <span style="font-weight:bold;">{lot}</span></div>
                  <div style="width:100%;text-align:center;margin-top:4px;margin-bottom:4px;">
                    <span style="font-size:1.14em;color:{price_color};font-weight:700;">{price_str}<span style="font-size:1.3em;">{arrow}</span><span style="color:{price_color};margin-left:6px;font-size:0.98em">{change_str} ({pct_str})</span></span>
                  </div>
                  <div style="display:flex;flex-direction:row;width:100%;justify-content:space-between;padding:0 16px;margin-bottom:4px;">
                    <div style="text-align:left;">{left_html}</div>
                    <div style="text-align:right;">{right_html}</div>
                  </div>
                  <div style="width:90%;border-top:1px solid #444;margin:4px auto;"></div>
                  <div style="width:100%;padding:0 16px 4px 16px;">{enhanced_html}</div>
                  <div style="width:90%;border-top:1px solid #444;margin:4px auto;"></div>
                  <div style="width:100%;text-align:center;padding:6px 0 8px 0;">
                    <span style="font-size:1.03em;color:#18AA47;font-weight:700;">DI+ {s.get('DI+','')}</span>
                    &nbsp;&nbsp;
                    <span style="font-size:1.03em;color:#E53935;font-weight:700;">DI- {s.get('DI-','')}</span>
                    &nbsp;&nbsp;
                    <span style="font-size:1.03em;color:#FF1493;font-weight:700;">ADX {s.get('ADX14','')}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        if not tiles:
            col.write("No stocks matched.")


if __name__ == "__main__":
    daily_weekly_dashboard()

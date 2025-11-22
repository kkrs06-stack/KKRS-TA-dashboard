import pandas as pd
import numpy as np
import ta
import yfinance as yf
import streamlit as st
import datetime
import warnings
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange
import scipy.signal

warnings.filterwarnings("ignore")

# ========== Helper Functions ==========

def detect_rsi_divergence(df, rsi_col='RSI', lookback=14):
    price = df['Close']
    rsi = df[rsi_col]
    lows = scipy.signal.argrelextrema(price.values, np.less, order=lookback)[0]
    highs = scipy.signal.argrelextrema(price.values, np.greater, order=lookback)[0]
    if len(lows) < 2 and len(highs) < 2:
        return None
    bull = None
    bear = None
    for i in range(1, len(lows)):
        if price.iloc[lows[i]] < price.iloc[lows[i-1]] and rsi.iloc[lows[i]] > rsi.iloc[lows[i-1]]:
            bull = lows[i]
    for i in range(1, len(highs)):
        if price.iloc[highs[i]] > price.iloc[highs[i-1]] and rsi.iloc[highs[i]] < rsi.iloc[highs[i-1]]:
            bear = highs[i]
    if bull is not None and (bear is None or bull > bear):
        return "Bull"
    elif bear is not None and (bull is None or bear > bull):
        return "Bear"
    else:
        return None

def get_first_two(text):
    return " ".join(str(text).split()[:2]) if text else ""

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

def compute_stochastic(df, k_period=5, d_period=3, smooth_k=3):
    low_min = df['Low'].rolling(window=k_period, min_periods=k_period).min()
    high_max = df['High'].rolling(window=k_period, min_periods=k_period).max()
    raw_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d

def ema50_comparison_row(ema50, sma20, section_type):
    if section_type == 'long':
        yn = "Y" if ema50 > sma20 else "N"
        color = "#37F553" if yn == "Y" else "#FF3A3A"
    else:
        yn = "N" if sma20 > ema50 else "Y"
        color = "#FF3A3A" if yn == "N" else "#37F553"
    return f"<span style='font-weight:700;'>50: <b>{ema50:.2f}</b> <span style='color:{color};font-size:1.19em;'>{yn}</span></span>"

def calc_indicators(df_d):
    adx = ADXIndicator(df_d['High'], df_d['Low'], df_d['Close'], 14)
    df_d['ADX14'] = adx.adx()
    df_d['DI+'] = adx.adx_pos()
    df_d['DI-'] = adx.adx_neg()
    df_d['SMA20'] = df_d['Close'].rolling(20).mean()
    df_d['EMA50'] = df_d['Close'].ewm(span=50).mean()
    macd_line = df_d['Close'].ewm(span=12, adjust=False).mean() - df_d['Close'].ewm(span=26, adjust=False).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df_d['MACD'] = macd_line
    df_d['MACD_signal'] = signal_line
    atr = AverageTrueRange(df_d['High'], df_d['Low'], df_d['Close'], 14)
    df_d['ATR14'] = atr.average_true_range()
    return df_d

def get_above_sma_labels(symbol, data_dict):
    try:
        df_d = data_dict[symbol]
        if df_d is None or df_d.empty:
            return "<span style='color:#FFD700;'>M: - | W: -</span>"
        last_close = df_d['Close'].iloc[-1]
        df_m = df_d.resample('M').last().dropna()
        df_m['SMA20'] = df_m['Close'].rolling(20).mean()
        sma20_m = df_m['SMA20'].iloc[-1] if len(df_m) >= 20 else np.nan
        if not np.isnan(sma20_m):
            m_yn = last_close > sma20_m
            m_col = "#37F553" if m_yn else "#FF3A3A"
            m_val = f"<span style='color:{m_col};font-weight:bold;'>M: {'Y' if m_yn else 'N'}</span>"
        else:
            m_val = "<span style='color:#FFD700;'>M: -</span>"
        df_w = df_d.resample('W-FRI').last().dropna()
        df_w['SMA20'] = df_w['Close'].rolling(20).mean()
        sma20_w = df_w['SMA20'].iloc[-1] if len(df_w) >= 20 else np.nan
        if not np.isnan(sma20_w):
            w_yn = last_close > sma20_w
            w_col = "#37F553" if w_yn else "#FF3A3A"
            w_val = f"<span style='color:{w_col};font-weight:bold;'>W: {'Y' if w_yn else 'N'}</span>"
        else:
            w_val = "<span style='color:#FFD700;'>W: -</span>"
        return f"{m_val} | {w_val}"
    except:
        return "<span style='color:#FFD700;'>M: - | W: -</span>"

def calc_ichimoku(df, tenkan=9, kijun=26, senkou=52):
    high_9 = df['High'].rolling(window=tenkan).max()
    low_9 = df['Low'].rolling(window=tenkan).min()
    tenkan_sen = (high_9 + low_9) / 2
    high_26 = df['High'].rolling(window=kijun).max()
    low_26 = df['Low'].rolling(window=kijun).min()
    kijun_sen = (high_26 + low_26) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    high_52 = df['High'].rolling(window=senkou).max()
    low_52 = df['Low'].rolling(window=senkou).min()
    senkou_span_b = ((high_52 + low_52) / 2).shift(kijun)
    chikou_span = df['Close'].shift(-kijun)  # Lagging Span
    return pd.DataFrame({
        'Tenkan': tenkan_sen,
        'Kijun': kijun_sen,
        'SpanA': senkou_span_a,
        'SpanB': senkou_span_b,
        'Chikou': chikou_span,
        'Close': df['Close'],
        'High': df['High'],
        'Low': df['Low'],
    }, index=df.index)

@st.cache_data(show_spinner="Batch loading Ichimoku F&O data...")
def fetch_all_ohlcv(ticker_list):
    tickers_str = " ".join(ticker_list)
    df = yf.download(tickers=tickers_str, period='500d', interval='1d', group_by='ticker', auto_adjust=True, progress=False, threads=True)
    data_dict = {}
    if isinstance(df.columns, pd.MultiIndex):
        for symbol in ticker_list:
            if symbol in df:
                sdf = df[symbol].dropna()
                sdf = sdf[~sdf.index.duplicated(keep='first')]
                if 'Volume' in sdf.columns:
                    sdf = sdf[sdf['Volume'] > 0]
                sdf = sdf[~((sdf['Open'] == sdf['High']) & (sdf['High'] == sdf['Low']) & (sdf['Low'] == sdf['Close']))]
                data_dict[symbol] = sdf
    else:
        sdf = df.dropna()
        sdf = sdf[~sdf.index.duplicated(keep='first')]
        data_dict[ticker_list[0]] = sdf
    return data_dict

def process_fo_stock_list():
    try:
        fo_df = pd.read_csv('fo_stock_list.csv')
    except Exception:
        st.error('Could not read fo_stock_list.csv')
        return pd.DataFrame()
    return fo_df

@st.cache_data(show_spinner="Triple Alignment signals...")
def batch_analyze_ichi(data_dict, fo_df,
                       require_cloud_color,
                       require_tenkan_kijun,
                       require_chikou_cloud,
                       require_chikou_price):
    long_results, short_results = [], []
    for idx, row in fo_df.iterrows():
        symbol = row['symbol']
        lot = row.get('lot_size', '')
        name = get_first_two(row.get('name', symbol))
        df_d = data_dict.get(symbol)
        if df_d is None or df_d.shape[0] < 80:
            continue

        ichi = calc_ichimoku(df_d)
        close_now = to_float(ichi['Close'].iloc[-1])
        tenkan_now = to_float(ichi['Tenkan'].iloc[-1])
        kijun_now = to_float(ichi['Kijun'].iloc[-1])
        span_a_now = to_float(ichi['SpanA'].iloc[-1])
        span_b_now = to_float(ichi['SpanB'].iloc[-1])
        cloud_now_top = max(span_a_now, span_b_now)
        cloud_now_bot = min(span_a_now, span_b_now)
        cloud_now_dir = span_a_now > span_b_now  # green

        try:
            chikou_26 = to_float(ichi['Chikou'].iloc[-26])
            span_a_26ago = to_float(ichi['SpanA'].iloc[-26])
            span_b_26ago = to_float(ichi['SpanB'].iloc[-26])
            close_26 = to_float(ichi['Close'].iloc[-26])
        except Exception:
            continue
        cloud_26up = max(span_a_26ago, span_b_26ago)
        cloud_26down = min(span_a_26ago, span_b_26ago)

        # Build conditionals from toggles
        long_cloud_ok = cloud_now_dir if require_cloud_color else True
        short_cloud_ok = (not cloud_now_dir) if require_cloud_color else True

        long_tk_ok = (tenkan_now > kijun_now) if require_tenkan_kijun else True
        short_tk_ok = (tenkan_now < kijun_now) if require_tenkan_kijun else True

        long_chikou_cloud_ok = (chikou_26 > cloud_26up) if require_chikou_cloud else True
        short_chikou_cloud_ok = (chikou_26 < cloud_26down) if require_chikou_cloud else True

        long_chikou_price_ok = (chikou_26 > close_26) if require_chikou_price else True
        short_chikou_price_ok = (chikou_26 < close_26) if require_chikou_price else True

        long_filter = (
            (close_now > cloud_now_top)
            and long_cloud_ok
            and long_tk_ok
            and long_chikou_cloud_ok
            and long_chikou_price_ok
        )
        short_filter = (
            (close_now < cloud_now_bot)
            and short_cloud_ok
            and short_tk_ok
            and short_chikou_cloud_ok
            and short_chikou_price_ok
        )

        if not (long_filter or short_filter):
            continue

        close_prev = to_float(ichi['Close'].iloc[-2])
        df_d = calc_indicators(df_d)
        st_series, st_dir = supertrend_tradingview_wilder(df_d, period=10, multiplier=3.0)
        stoch_k, stoch_d = compute_stochastic(df_d, k_period=5, d_period=3, smooth_k=3)
        shr_k_val = round(stoch_k.iloc[-1], 2) if stoch_k is not None else "NA"
        shr_d_val = round(stoch_d.iloc[-1], 2) if stoch_d is not None else "NA"
        shr_dot = '<span style="color:#37F553;font-size:1.23em">&#x25CF;</span>' if (shr_k_val != "NA" and shr_k_val >= shr_d_val) else '<span style="color:#FF3A3A;font-size:1.23em">&#x25CF;</span>'
        if st_series is not None:
            supertrend_val = st_series.iloc[-1]
            supertrend_dir = st_dir[-1]
        else:
            supertrend_val, supertrend_dir = "NA", "NA"
        d_sma = round(to_float(df_d['SMA20'].iloc[-1]),2)
        ema50 = round(to_float(df_d['EMA50'].iloc[-1]),2)
        section_type = "long" if long_filter else "short"
        result_data = {
            'Name': name,
            'Symbol': symbol,
            'Lot': lot,
            'D': d_sma,
            'EMA50': ema50,
            'SuperTrend': round(supertrend_val,2) if pd.notna(supertrend_val) else "NA",
            'SuperTrendDir': supertrend_dir,
            'Close': round(close_now,2),
            'PrevClose': round(close_prev,2),
            'ADX14': round(to_float(df_d['ADX14'].iloc[-1]), 2),
            'DI+': round(to_float(df_d['DI+'].iloc[-1]), 2),
            'DI-': round(to_float(df_d['DI-'].iloc[-1]), 2),
            'Daily_RSI': round(to_float(ta.momentum.RSIIndicator(df_d['Close'], window=14).rsi().iloc[-1]),2),
            'OBV_ARROW': obv_trend_arrow(df_d, 10),
            'SHR_DOT': shr_dot,
            'SHR_K': shr_k_val,
            'SHR_D': shr_d_val,
            'SectionType': section_type
        }
        if long_filter:
            long_results.append(result_data)
        if short_filter:
            short_results.append(result_data)
    return long_results, short_results

def ichi_dashboard():
    today_str = datetime.datetime.now().strftime("%d-%b-%Y")
    st.markdown("<div style='font-size:2em;font-weight:800;color:#FFD700;padding-bottom:2px;'>Ichi: Triple Alignment (Toggleable)</div>",unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:right;font-size:1.2em;color:#FFD700;font-weight:700;padding-top:4px;'>{today_str}</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:1.30em; text-align:center; margin-bottom:10px;color:#FFD700;'>Ichimoku: Triple Alignment with Live Toggles</h2>", unsafe_allow_html=True)

    # --- Alignment Toggles ---
    # Place all toggles in a single row
    cb_cols = st.columns(4)
    require_cloud_color   = cb_cols[0].checkbox("Cloud Color", value=True, help="Require green (bull) or red (bear) Kumo for signal")
    require_tenkan_kijun  = cb_cols[1].checkbox("Tenkan/Kijun", value=True, help="Require Tenkan > Kijun (bull) or < (bear)")
    require_chikou_cloud  = cb_cols[2].checkbox("Chikou vs Cloud", value=True, help="Require Chikou Span above/below cloud 26 bars back")
    require_chikou_price  = cb_cols[3].checkbox("Chikou vs Price", value=True, help="Require Chikou Span above/below price 26 bars back")


    if st.button("🔄 Refresh Data", key="refresh_ichi"):
        fetch_all_ohlcv.clear()
        batch_analyze_ichi.clear()

    fo_df = process_fo_stock_list()
    if fo_df.empty:
        return
    ticker_list = list(fo_df['symbol'])
    data_dict = fetch_all_ohlcv(ticker_list)
    long_results, short_results = batch_analyze_ichi(data_dict, fo_df,
        require_cloud_color=require_cloud_color,
        require_tenkan_kijun=require_tenkan_kijun,
        require_chikou_cloud=require_chikou_cloud,
        require_chikou_price=require_chikou_price,
    )

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
                d_val = s.get('D',0)
                ema50_val = s.get('EMA50',0)
                supertrend_val = s.get('SuperTrend', 'NA')
                supertrend_dir = s.get('SuperTrendDir','NA')
                obv_arrow = s.get('OBV_ARROW', '')
                shr_html = f'<span style="font-weight:700;">SHR:</span> {s.get("SHR_DOT","")} <span style="color:#37F553;font-weight:700;">{s.get("SHR_K","")}</span> / <span style="color:#FFD700;font-weight:700;">{s.get("SHR_D","")}</span>'
                section_type = s.get('SectionType','long')
                ema_comp_html = ema50_comparison_row(ema50_val, d_val, section_type)
                mw_label = get_above_sma_labels(symbol, data_dict)
                st_dir_col = "#FFD700"
                if supertrend_dir == "UP":
                    st_dir_col = "#37F553"
                elif supertrend_dir == "DOWN":
                    st_dir_col = "#FF3A3A"
                supertrend_html = f"ST: <b>{supertrend_val}</b> <span style='color:{st_dir_col};font-weight:900;font-size:1.12em'>{supertrend_dir}</span>"
                daily_rsi_val = s.get('Daily_RSI', 'NA')
                daily_rsi_html = rsi_colored_custom(daily_rsi_val)

                adx_val = s.get('ADX14', 'NA')
                di_plus = s.get('DI+', 'NA')
                di_minus = s.get('DI-', 'NA')
                di_plus_html = f"<span style='color:#18AA47;font-size:1em;'>DI+ {di_plus}</span>"
                di_minus_html = f"<span style='color:#E53935;font-size:1em;'>DI- {di_minus}</span>"
                if section_type == "long":
                    if di_plus is not None and di_minus is not None:
                        if di_plus > di_minus:
                            di_plus_html = f"<span style='color:#18AA47;font-size:1.22em;font-weight:900;'>DI+ {di_plus}</span>"
                        if di_minus < 20 and di_plus > 20:
                            di_minus_html = f"<span style='color:#E53935;font-size:1.22em;font-weight:900;'>DI- {di_minus}</span>"
                else:
                    if di_plus is not None and di_minus is not None:
                        if di_minus > di_plus:
                            di_minus_html = f"<span style='color:#E53935;font-size:1.22em;font-weight:900;'>DI- {di_minus}</span>"
                        if di_minus > 20 and di_plus < 20:
                            di_minus_html = f"<span style='color:#E53935;font-size:1.22em;font-weight:900;'>DI- {di_minus}</span>"

                tview_url = f"https://www.tradingview.com/chart/HHuUSOTG/?symbol=NSE:{symbol.replace('.NS','')}"
                left_rows = [
                    f"D: <b>{d_val}</b>",
                    ema_comp_html,
                    supertrend_html,
                ]
                right_rows = [
                    f"RSI: {daily_rsi_html}",
                    f"OBV: {obv_arrow}",
                    shr_html,
                    mw_label,
                ]
                left_html = "".join([f"<div style='font-size:1.05em;color:#ECECEC;margin-bottom:3px;'> {row}</div>" for row in left_rows])
                right_html = "".join([f"<div style='font-size:1.05em;margin-bottom:3px;'> {row}</div>" for row in right_rows])
                tcol.markdown(f"""
                <div style="background:#252525;border-radius:16px;width:338px;height:246px;position:relative;box-shadow:1px 2px 10px #111;margin-bottom:20px;display:flex;flex-direction:column;align-items:center;border:1px solid #333;">
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
                  <div style="display:flex;flex-direction:row;width:100%;justify-content:space-between;margin-top:3px;">
                    <div style="padding-left:12px;text-align:left;width:48%">{left_html}</div>
                    <div style="padding-right:12px;text-align:right;width:48%">{right_html}</div>
                  </div>
                  <div style="position:absolute;bottom:8px;width:100%;text-align:center;">
                    {di_plus_html}
                    &nbsp; {di_minus_html}
                    &nbsp; <span style="font-size:1em; color:#FF1493; font-weight:700;">ADX {adx_val}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        if not tiles:
            col.write("No stocks matched.")

if __name__ == "__main__":
    ichi_dashboard()

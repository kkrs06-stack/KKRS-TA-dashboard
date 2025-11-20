import pandas as pd
import numpy as np
import ta
import yfinance as yf
import streamlit as st
import sys
import os
import warnings
from ta.trend import ADXIndicator
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

def rsi_color(val):
    try:
        v = float(val)
        if v > 60: return "#37F553"
        if v < 40: return "#FF3A3A"
        return "#FFA500"
    except: return "#ECECEC"

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
        return "<span style='font-size:1.15em;color:#FFD700;'>-</span>"
    obv_now = obv.iloc[-1]
    obv_prev = obv.iloc[-period-1]
    flat_threshold = 0.005 * abs(obv_now)
    diff = obv_now - obv_prev
    if diff > flat_threshold:
        return "<span style='font-size:1.15em;color:#37F553;'>↑</span>"
    elif diff < -flat_threshold:
        return "<span style='font-size:1.15em;color:#FF3A3A;'>↓</span>"
    else:
        return "<span style='font-size:1.15em;color:#FFD700;'>-</span>"

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

def compute_stochastic(df, k_period=5, d_period=3, smooth_k=3):
    low_min = df['Low'].rolling(window=k_period, min_periods=k_period).min()
    high_max = df['High'].rolling(window=k_period, min_periods=k_period).max()
    raw_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d

def get_above_sma_labels(symbol, data_dict):
    try:
        df_d = data_dict.get(symbol)
        if df_d is None:
            return "<span style='color:#FFD700;'>M: - | W: -</span>"
        last_close = df_d['Close'].iloc[-1]
        df_m = df_d['Close'].resample('M').last().dropna().to_frame()
        if len(df_m) >= 20:
            df_m['SMA20'] = df_m['Close'].rolling(20).mean()
            sma20_m = df_m['SMA20'].iloc[-1]
            m_yn = last_close > sma20_m
            m_val = f"<span style='color:{'#37F553' if m_yn else '#FF3A3A'};font-weight:bold;'>M: {'Y' if m_yn else 'N'}</span>"
        else:
            m_val = "<span style='color:#FFD700;'>M: -</span>"
        df_w = df_d['Close'].resample('W-FRI').last().dropna().to_frame()
        if len(df_w) >= 20:
            df_w['SMA20'] = df_w['Close'].rolling(20).mean()
            sma20_w = df_w['SMA20'].iloc[-1]
            w_yn = last_close > sma20_w
            w_val = f"<span style='color:{'#37F553' if w_yn else '#FF3A3A'};font-weight:bold;'>W: {'Y' if w_yn else 'N'}</span>"
        else:
            w_val = "<span style='color:#FFD700;'>W: -</span>"
        return f"{m_val} | {w_val}"
    except:
        return "<span style='color:#FFD700;'>M: - | W: -</span>"

# ----------- BATCHED & CACHED SS Strat → ALL TICKERS ----------- #

@st.cache_data(show_spinner="Batch downloading all OHLCV for SS Strat...")
def get_ssstrat_stock_data_optimized():
    df = pd.read_csv('fo_stock_hippo.csv')
    tickers = df["symbol"].tolist()

    # Download all OHLCV in one batch
    batch_df = yf.download(
        tickers=" ".join(tickers),
        period='500d', interval='1d',
        group_by='ticker', auto_adjust=True, threads=True, progress=False
    )
    # Parse into symbol->df dictionary
    data_dict = {}
    if isinstance(batch_df.columns, pd.MultiIndex):
        for symbol in tickers:
            if symbol in batch_df:
                sdf = batch_df[symbol].dropna()
                sdf = sdf[~sdf.index.duplicated(keep='first')]
                if 'Volume' in sdf.columns:
                    sdf = sdf[sdf['Volume'] > 0]
                sdf = sdf[~((sdf['Open'] == sdf['High']) & (sdf['High'] == sdf['Low']) & (sdf['Low'] == sdf['Close']))]
                data_dict[symbol] = sdf
    else:
        sdf = batch_df.dropna()
        sdf = sdf[~sdf.index.duplicated(keep='first')]
        data_dict[tickers[0]] = sdf

    stocks = []
    for _, row in df.iterrows():
        name = get_first_two(row["name"])
        symbol = row["symbol"]
        lot = row["lot_size"]
        try:
            st_df = data_dict.get(symbol)
            if st_df is None or st_df.shape[0] == 0:
                raise Exception("No data")
            price = st_df['Close'].iloc[-1]
            prev_price = st_df['Close'].iloc[-2] if len(st_df['Close']) > 1 else price
            # Supertrend
            st_val, st_dir = supertrend_tradingview_wilder(st_df, period=10, multiplier=3.0)
            supertrend_val = st_val.iloc[-1] if st_val is not None else "NA"
            supertrend_dir = st_dir.iloc[-1] if st_dir is not None else "NA"
            obv_arrow = obv_trend_arrow(st_df, 10)
            mw_label = get_above_sma_labels(symbol, data_dict)
            # Stochastic
            stoch_k, stoch_d = compute_stochastic(st_df, k_period=5, d_period=3, smooth_k=3)
            stoch_k_val = round(stoch_k.iloc[-1], 2) if stoch_k is not None else "NA"
            stoch_d_val = round(stoch_d.iloc[-1], 2) if stoch_d is not None else "NA"
            dot = '<span style="color:#37F553;font-size:1.23em">&#x25CF;</span>' if stoch_k_val >= stoch_d_val else '<span style="color:#FF3A3A;font-size:1.23em">&#x25CF;</span>'
            # RSI
            d_rsi = ta.momentum.RSIIndicator(st_df['Close'], window=14).rsi().iloc[-1]
            weekly_close = st_df['Close'].resample('W').last()
            wk_rsi = ta.momentum.RSIIndicator(weekly_close, window=14).rsi().iloc[-1]
        except Exception as e:
            price = prev_price = wk_rsi = d_rsi = stoch_k_val = stoch_d_val = "NA"
            supertrend_val = supertrend_dir = obv_arrow = mw_label = dot = "NA"
        arrow, arrow_color, change, pct = price_arrow_and_change(price, prev_price)
        stocks.append({
            "name": name, "symbol": symbol, "price": price, "prev_price": prev_price, "change": change, "pct": pct,
            "lot": lot, "wk_rsi": wk_rsi, "d_rsi": d_rsi, "arrow": arrow, "arrow_color": arrow_color,
            "supertrend_val": supertrend_val, "supertrend_dir": supertrend_dir,
            "obv_arrow": obv_arrow, "mw_label": mw_label,
            "stoch_k_val": stoch_k_val, "stoch_d_val": stoch_d_val, "stoch_dot": dot
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

# ----------------- DASHBOARD UI -------------------

def ss_strat_dashboard(tv_chart_url):
    if st.button("🔄 Refresh SS Strat Data", key="refresh_ssstrat"):
        get_ssstrat_stock_data_optimized.clear()
        st.rerun()
    ce_sell, pe_sell, cep_sell = get_ssstrat_stock_data_optimized()
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
        st_dir_col = "#37F553" if s["supertrend_dir"] == "UP" else "#FF3A3A" if s["supertrend_dir"] == "DOWN" else "#FFD700"
        return f"""
        <div style="background:#252525;border-radius:15px;width:260px;height:230px;position:relative;box-shadow:1px 2px 8px #111;margin-bottom:18px;display:flex;flex-direction:column;align-items:center;border:1px solid #333;">
          <div style="font-size:1.02em;color:#fff;font-weight:700;text-align:center;width:100%;margin-top:13px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">
            <a href="{tv_url}" target="_blank" style="color:#fff;text-decoration:none;">{s['name']}</a>
            <span style="font-size:0.93em;color:#fff;">Lot: <b>{s['lot']}</b></span>
          </div>
          <div style="width:100%;text-align:center;margin-top:7px;margin-bottom:0;">
            <span style="font-size:1.07em;color:{s['arrow_color']};font-weight:700;">
                ₹ {price_str}
                <span style="font-size:1.08em;">{s['arrow']}</span>
                <span style="margin-left:6px;font-size:0.94em">{change_str} ({pct_str})</span>
            </span>
          </div>
          <div style="margin-top:8px;width:96%;">
            <div style="display: flex; flex-direction: row; justify-content: space-between;">
              <div>
                <span style="color:#fff; font-weight:700;">ST:</span>
                <span style="font-weight:bold;">{s['supertrend_val']}</span>
                <span style="color:{st_dir_col};font-weight:900;">{s['supertrend_dir']}</span>
              </div>
              <div>
                <span style="color:#fff; font-weight:700;">OBV:</span> {s['obv_arrow']}
              </div>
            </div>
            <div style="margin-top:3px;text-align:center;">
              {s['mw_label']}
            </div>
          </div>
          <div style="margin-top:4px; font-size:1em; text-align:center;">
            <span style="color:#fff;">RSI (Wk): </span><span style="color:{rsi_color(s['wk_rsi'])};font-weight:700;">{wk_rsi_str}</span>
            &nbsp;|&nbsp;
            <span style="color:#fff;">RSI (Day): </span><span style="color:{rsi_color(s['d_rsi'])};font-weight:700;">{d_rsi_str}</span>
          </div>
          <div style="margin-top:4px; font-size:1.05em; text-align:center;">
            <span style="color:#fff; font-weight:700;">SHR:</span>
            {s['stoch_dot']}
            <span style="color:#37F553;font-weight:700;">{s['stoch_k_val']}</span>
            /
            <span style="color:#FFD700;font-weight:700;">{s['stoch_d_val']}</span>
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

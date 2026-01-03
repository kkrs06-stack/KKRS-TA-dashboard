import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import datetime
import scipy.signal
from ta.trend import ADXIndicator
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from io import StringIO

# ========== HELPER FUNCTIONS ==========

def get_first_two(text):
    """Get first two words of company name"""
    return " ".join(str(text).split()[:2]) if text else ""

def get_lot_size(symbol, fo_df):
    """Get lot size from fo_stock_list.csv"""
    try:
        row = fo_df[fo_df['symbol'] == symbol]
        if not row.empty:
            return row.iloc[0].get('lot_size', '')
        return ''
    except:
        return ''

def get_company_name(symbol, fo_df):
    """Get company name from fo_stock_list.csv"""
    try:
        row = fo_df[fo_df['symbol'] == symbol]
        if not row.empty:
            name = row.iloc[0].get('name', symbol)
            return get_first_two(name)
        return symbol.replace('.NS', '')
    except:
        return symbol.replace('.NS', '')

# ========== PRIORITY 1: VWAP CALCULATION ==========

def calculate_vwap(df):
    """
    Calculate 20-day rolling VWAP - Institutional Standard
    """
    try:
        # Use last 20 trading days
        df_period = df.tail(20).copy()

        # Typical price
        df_period['TypicalPrice'] = (df_period['High'] + df_period['Low'] + df_period['Close']) / 3

        # Volume weighted
        df_period['PV'] = df_period['TypicalPrice'] * df_period['Volume']

        # Calculate VWAP
        total_pv = df_period['PV'].sum()
        total_volume = df_period['Volume'].sum()

        if total_volume > 0:
            vwap = total_pv / total_volume
            return float(vwap)
        else:
            return None

    except:
        return None

def check_vwap_confluence(price, vwap, direction='LONG', threshold=0.01):
    """Check if price near VWAP with directional alignment"""
    try:
        if vwap is None:
            return False, "⚪ No VWAP", 0

        dist_pct = abs((price - vwap) / vwap)

        if dist_pct <= threshold:
            if direction == 'LONG' and price > vwap:
                return True, "✓ Above VWAP", ((price - vwap) / vwap) * 100
            elif direction == 'SHORT' and price < vwap:
                return True, "✓ Below VWAP", ((vwap - price) / vwap) * 100
            else:
                return False, "✗ Wrong side", dist_pct * 100
        else:
            return False, "⚪ Far from VWAP", dist_pct * 100
    except:
        return False, "⚪ Error", 0

# ========== PRIORITY 2: FII/DII DATA ==========

@st.cache_data(ttl=86400)
def fetch_fii_dii_data():
    """Fetch FII/DII data from NSE (cached for 1 day)"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

        url = 'https://www.nseindia.com/api/fiidiiTradeReact'
        session = requests.Session()
        session.get('https://www.nseindia.com', headers=headers, timeout=5)
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            fii_net = []
            dii_net = []

            for entry in data[:5]:
                if entry['category'] == 'FII/FPI':
                    fii_net.append(float(entry.get('netValue', 0)))
                elif entry['category'] == 'DII':
                    dii_net.append(float(entry.get('netValue', 0)))

            return {
                'fii_5d_net': sum(fii_net),
                'dii_5d_net': sum(dii_net),
                'total_5d_net': sum(fii_net) + sum(dii_net),
                'status': 'success'
            }
        else:
            return {'status': 'failed', 'fii_5d_net': 0, 'dii_5d_net': 0, 'total_5d_net': 0}
    except:
        return {'status': 'failed', 'fii_5d_net': 0, 'dii_5d_net': 0, 'total_5d_net': 0}

def check_fii_dii_confluence(direction='LONG'):
    """Check if FII/DII buying/selling aligns with direction"""
    try:
        fii_dii = fetch_fii_dii_data()

        if fii_dii['status'] == 'failed':
            return False, "⚪ No Data", 0

        net = fii_dii['total_5d_net']

        if direction == 'LONG' and net > 500:
            return True, "✓ Inst. Buying", net
        elif direction == 'SHORT' and net < -500:
            return True, "✓ Inst. Selling", abs(net)
        else:
            return False, "⚪ Neutral", abs(net)
    except:
        return False, "⚪ Error", 0

# ========== PRIORITY 3: OPTIONS FLOW (OPTIONAL - NO PENALTY IF UNAVAILABLE) ==========

def check_options_confluence(symbol, direction='LONG'):
    """
    Check options flow - OPTIONAL indicator
    Always returns False (no penalty) if data unavailable
    NSE restricts automated options data access
    """
    # PCR data not reliably available due to NSE restrictions
    # Return neutral - no penalty, no bonus
    return False, "⚪ PCR N/A", 0

# ========== ORIGINAL ENHANCEMENT FUNCTIONS ==========

def calculate_support_resistance_fib(df, window=20):
    """Calculate support/resistance levels using swing points and Fibonacci"""
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
    """Determine if price is at support/resistance"""
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
    """Check if current volume is above/below average"""
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
    """Detect OBV divergence using scipy peak detection"""
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

def calculate_mtf_alignment(df):
    """Calculate multi-timeframe SMA20 alignment"""
    try:
        if df is None or len(df) < 60:
            return "⚪", "⚪", "⚪", 0
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        price = float(df['Close'].iloc[-1])
        daily_ok = False
        try:
            sma20_d = float(df['Close'].rolling(20).mean().iloc[-1])
            daily_ok = price > sma20_d
        except:
            daily_ok = False
        weekly_ok = None
        try:
            df_w_close = df['Close'].resample('W-FRI').last().dropna()
            if len(df_w_close) >= 20:
                sma20_w = float(df_w_close.rolling(20).mean().iloc[-1])
                weekly_ok = price > sma20_w
        except:
            weekly_ok = None
        monthly_ok = None
        try:
            df_m_close = df['Close'].resample('M').last().dropna()
            if len(df_m_close) >= 20:
                sma20_m = float(df_m_close.rolling(20).mean().iloc[-1])
                monthly_ok = price > sma20_m
        except:
            monthly_ok = None
        mtf_d = "✓" if daily_ok else "✗"
        mtf_w = "✓" if weekly_ok is True else ("⚪" if weekly_ok is None else "✗")
        mtf_m = "✓" if monthly_ok is True else ("⚪" if monthly_ok is None else "✗")
        mtf_score = sum([1 for x in [daily_ok, weekly_ok, monthly_ok] if x is True])
        return mtf_d, mtf_w, mtf_m, mtf_score
    except:
        return "⚪", "⚪", "⚪", 0

def calculate_adx(df, period=14):
    """Calculate ADX indicator"""
    try:
        adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=period)
        adx_value = adx_indicator.adx().iloc[-1]
        return float(adx_value) if pd.notnull(adx_value) else 0
    except:
        return 0

def check_clean_path(df, current_price, direction='LONG', threshold_pct=5):
    """Check if there's a clean path to target"""
    try:
        sr_data = calculate_support_resistance_fib(df, window=20)
        if sr_data is None:
            return False
        if direction == 'LONG':
            threshold_price = current_price * (1 + threshold_pct/100)
            for resistance in sr_data['resistance_levels']:
                if current_price < resistance <= threshold_price:
                    return False
            return True
        else:
            threshold_price = current_price * (1 - threshold_pct/100)
            for support in sr_data['support_levels']:
                if threshold_price <= support < current_price:
                    return False
            return True
    except:
        return False

# ========== ENHANCED CONFLUENCE SCORING (12-POINT SYSTEM) ==========

def calculate_confluence_score(symbol, df):
    """Calculate confluence score (0-12) for a stock - PRODUCTION VERSION"""
    try:
        current_price = float(df['Close'].iloc[-1])
        sr_data = calculate_support_resistance_fib(df, window=20)
        sr_status, sr_level_val, sr_dist = get_sr_status(current_price, sr_data)
        vol_status, vol_ratio, vol_color = check_volume_status(df, lookback=20)
        obv_div_text, obv_div_type = detect_obv_divergence_enhanced(df, lookback=14)
        mtf_d, mtf_w, mtf_m, mtf_score = calculate_mtf_alignment(df)
        adx_value = calculate_adx(df, period=14)
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        direction = 'LONG' if current_price > sma_20 else 'SHORT'

        vwap = calculate_vwap(df)
        vwap_ok, vwap_status, vwap_dist = check_vwap_confluence(current_price, vwap, direction)

        fii_dii_ok, fii_dii_status, fii_dii_net = check_fii_dii_confluence(direction)

        opt_ok, opt_status, opt_pcr = check_options_confluence(symbol, direction)

        score = 0
        score_breakdown = []

        if direction == 'LONG' and '🟢 At Support' in sr_status:
            score += 2
            score_breakdown.append('+2 S/R')
        elif direction == 'SHORT' and '🔴 At Resistance' in sr_status:
            score += 2
            score_breakdown.append('+2 S/R')

        if vol_ratio > 1.5:
            score += 2
            score_breakdown.append('+2 Vol')

        if mtf_score == 3:
            score += 2
            score_breakdown.append('+2 MTF')

        if direction == 'LONG' and obv_div_type == 'bull_div':
            score += 2
            score_breakdown.append('+2 OBV')
        elif direction == 'SHORT' and obv_div_type == 'bear_div':
            score += 2
            score_breakdown.append('+2 OBV')

        if adx_value > 25:
            score += 1
            score_breakdown.append('+1 ADX')

        clean_path = check_clean_path(df, current_price, direction, threshold_pct=5)
        if clean_path:
            score += 1
            score_breakdown.append('+1 Path')

        if vwap_ok:
            score += 1
            score_breakdown.append('+1 VWAP')

        if fii_dii_ok:
            score += 1
            score_breakdown.append('+1 FII/DII')

        # PCR is optional - if it works, great; if not, no penalty
        if opt_ok:
            score += 1
            score_breakdown.append('+1 PCR')

        return {
            'Symbol': symbol,
            'Direction': direction,
            'Score': score,
            'Score_Breakdown': ' | '.join(score_breakdown) if score_breakdown else 'None',
            'Current_Price': round(current_price, 2),
            'SMA_20': round(sma_20, 2),
            'SR_Status': sr_status,
            'SR_Dist': round(sr_dist, 2) if sr_dist != "NA" and sr_dist else "NA",
            'Vol_Status': vol_status,
            'Vol_Ratio': round(vol_ratio, 2),
            'OBV_Div': obv_div_text,
            'MTF_Daily': mtf_d,
            'MTF_Weekly': mtf_w,
            'MTF_Monthly': mtf_m,
            'MTF_Score': mtf_score,
            'ADX': round(adx_value, 2),
            'Clean_Path': '✓' if clean_path else '✗',
            'VWAP': round(vwap, 2) if vwap else "NA",
            'VWAP_Status': vwap_status,
            'VWAP_Dist': round(vwap_dist, 2) if vwap_dist else "NA",
            'FII_DII_Status': fii_dii_status,
            'FII_DII_Net': round(fii_dii_net, 2) if fii_dii_net else 0,
            'Opt_Status': opt_status,
            'Opt_PCR': opt_pcr,
        }
    except:
        return None

def process_stock_confluence(symbol):
    """Process single stock for confluence scanner"""
    try:
        if not symbol.endswith('.NS'):
            symbol = symbol + ".NS"
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='2y')
        if df.empty or len(df) < 100:
            return None
        df.index = pd.to_datetime(df.index)
        return calculate_confluence_score(symbol, df)
    except:
        return None

@st.cache_data(show_spinner="Scanning for institutional-grade setups...", ttl=3600)
def get_confluence_scan_data(min_score=6):
    """Scan all F&O stocks for confluence signals"""
    fo_df = pd.read_csv('fo_stock_list.csv')
    fo_stocks = fo_df['symbol'].tolist()
    results = []
    max_workers = min(10, len(fo_stocks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(process_stock_confluence, symbol): symbol for symbol in fo_stocks}
        for future in as_completed(future_to_symbol):
            try:
                result = future.result()
                if result is not None and result['Score'] >= min_score:
                    results.append(result)
            except:
                pass
    df_results = pd.DataFrame(results)
    if len(df_results) > 0:
        df_results = df_results.sort_values('Score', ascending=False)
    return df_results

# ========== RENDER TILES ==========

def render_confluence_tile(stock_data, fo_df):
    """Render single confluence tile with institutional indicators"""
    symbol = stock_data['Symbol']
    symbol_clean = symbol.replace('.NS', '')
    name = get_company_name(symbol, fo_df)
    lot = get_lot_size(symbol, fo_df)
    price = stock_data['Current_Price']
    direction = stock_data['Direction']
    score = stock_data['Score']
    breakdown = stock_data['Score_Breakdown']
    sr_status = stock_data['SR_Status']
    sr_dist = stock_data['SR_Dist']
    vol_status = stock_data['Vol_Status']
    vol_ratio = stock_data['Vol_Ratio']
    obv_div = stock_data['OBV_Div']
    mtf_d = stock_data['MTF_Daily']
    mtf_w = stock_data['MTF_Weekly']
    mtf_m = stock_data['MTF_Monthly']
    mtf_score = stock_data['MTF_Score']
    adx = stock_data['ADX']
    clean_path = stock_data['Clean_Path']

    vwap = stock_data.get('VWAP', 'NA')
    vwap_status = stock_data.get('VWAP_Status', '⚪ No Data')
    vwap_dist = stock_data.get('VWAP_Dist', 'NA')
    fii_dii_status = stock_data.get('FII_DII_Status', '⚪ No Data')
    fii_dii_net = stock_data.get('FII_DII_Net', 0)
    opt_status = stock_data.get('Opt_Status', '⚪ PCR N/A')
    opt_pcr = stock_data.get('Opt_PCR', 0)

    # Score colors (12-point scale)
    if score >= 10:
        score_color, score_bg = "#00FF00", "#1B5E20"
    elif score >= 8:
        score_color, score_bg = "#FFD700", "#4A3300"
    else:
        score_color, score_bg = "#FFA500", "#4A2A00"

    dir_color = "#37F553" if direction == "LONG" else "#FF3A3A"
    sr_display = f"{sr_status} ({sr_dist:+.1f}%)" if sr_dist != "NA" else sr_status
    vol_display = f"{vol_status} ({vol_ratio:.1f}x)"
    obv_display = f"OBV: {obv_div}"
    mtf_display = f"MTF: {mtf_d} {mtf_w} {mtf_m} ({mtf_score}/3)"

    vwap_display = f"VWAP: {vwap_status}"
    if vwap_dist != "NA":
        vwap_display += f" ({vwap_dist:+.2f}%)"

    fii_dii_display = f"FII/DII: {fii_dii_status}"
    if fii_dii_net != 0:
        fii_dii_display += f" (₹{abs(fii_dii_net):.0f}Cr)"

    opt_display = f"PCR: {opt_status}"
    if opt_pcr != 0:
        opt_display += f" ({opt_pcr})"

    tview_url = f"https://www.tradingview.com/chart/HHuUSOTG/?symbol=NSE:{symbol_clean}"

    return f"""
    <div style="background:#252525;border-radius:14px;width:380px;height:630px;position:relative;box-shadow:2px 4px 12px #000;margin-bottom:15px;border:2px solid {score_color};overflow:hidden;">
      <div style="width:100%;text-align:center;padding-top:6px;">
        <a href="{tview_url}" target="_blank" style="color:#fff;font-size:1.35em;font-weight:700;text-decoration:none;hover:text-decoration:underline;">{name}</a>
      </div>
      <div style="position:absolute;right:16px;top:6px;font-size:1.08em;color:#ECECEC;">Lot: <b>{lot}</b></div>

      <div style="background:{score_bg};width:90%;margin:8px auto;padding:10px;border-radius:10px;border:2px solid {score_color};">
        <div style="text-align:center;">
          <span style="font-size:2.4em;color:{score_color};font-weight:900;">{score}/12</span>
          <span style="margin-left:15px;font-size:1.3em;color:{dir_color};font-weight:700;">{direction}</span>
        </div>
      </div>

      <div style="width:100%;text-align:center;margin-bottom:8px;">
        <span style="font-size:1.35em;color:#37F553;font-weight:700;">₹ {price}</span>
      </div>

      <div style="width:90%;margin:0 auto;padding:8px;background:#1a1a1a;border-radius:8px;">
        <div style="font-size:1.12em;color:#ECECEC;margin-bottom:3px;">📍 {sr_display}</div>
        <div style="font-size:1.12em;color:#ECECEC;margin-bottom:3px;">📊 {vol_display}</div>
        <div style="font-size:1.12em;color:#ECECEC;margin-bottom:3px;">📈 {obv_display}</div>
        <div style="font-size:1.12em;color:#FFD700;font-weight:700;margin-bottom:3px;">🎯 {mtf_display}</div>
        <div style="font-size:1.12em;color:#ECECEC;margin-bottom:3px;">💪 ADX: <b>{adx}</b></div>
        <div style="font-size:1.12em;color:#ECECEC;margin-bottom:3px;">🛣️ Path: <b>{clean_path}</b></div>
        <div style="font-size:1.12em;color:#00D9FF;font-weight:700;margin-bottom:3px;">💰 {vwap_display}</div>
        <div style="font-size:1.12em;color:#FF6B35;font-weight:700;margin-bottom:3px;">🏛️ {fii_dii_display}</div>
        <div style="font-size:1.12em;color:#9D4EDD;font-weight:700;">📊 {opt_display}</div>
      </div>

      <div style="width:90%;margin:8px auto;padding:8px;background:#1a1a1a;border-radius:8px;border:1px solid {score_color};">
        <div style="text-align:center;color:{score_color};font-weight:700;font-size:1.08em;margin-bottom:5px;">SCORE BREAKDOWN</div>
        <div style="font-size:1.0em;color:#ECECEC;text-align:center;word-wrap:break-word;line-height:1.4;">{breakdown}</div>
      </div>
    </div>
    """

# ========== MAIN DASHBOARD ==========

def confluence_scanner_dashboard():
    """Main confluence scanner dashboard - PRODUCTION VERSION"""
    st.markdown("<div style='font-size:2.0em;font-weight:800;color:#FFD700;padding-bottom:10px;text-align:center;'>🏛️ Institutional Confluence Scanner 🏛️</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:1.2em;color:#37F553;font-weight:700;padding:4px;'>✅ 12-Point Proven System | ✅ VWAP + FII/DII + Technical Confluence</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Institutional Scanner")
        st.markdown("**Scoring System (0-12):**")
        st.markdown("- +2: S/R Entry")
        st.markdown("- +2: High Volume")
        st.markdown("- +2: MTF Aligned")
        st.markdown("- +2: OBV Divergence")
        st.markdown("- +1: ADX > 25")
        st.markdown("- +1: Clean Path")
        st.markdown("- +1: **VWAP** 💰")
        st.markdown("- +1: **FII/DII** 🏛️")
        st.markdown("---")
        st.info("📊 PCR optional (NSE restricted)")
        st.markdown("---")
        min_score_filter = st.slider("Minimum Score", min_value=0, max_value=12, value=6, key="conf_min_score")
        direction_filter = st.selectbox("Direction Filter", ["ALL", "LONG", "SHORT"], key="conf_direction")
        st.markdown("---")
        st.success("⚡ Production v3.0")

    if st.button("🔄 Refresh Confluence Data", key="refresh_confluence"):
        get_confluence_scan_data.clear()
        fetch_fii_dii_data.clear()
        st.rerun()

    try:
        fo_df = pd.read_csv('fo_stock_list.csv')
    except FileNotFoundError:
        st.error("❌ fo_stock_list.csv not found!")
        st.stop()

    results_df = get_confluence_scan_data(min_score=min_score_filter)

    if len(results_df) == 0:
        st.warning(f"No stocks found with score >= {min_score_filter}")
        return

    if direction_filter != "ALL":
        results_df = results_df[results_df['Direction'] == direction_filter]

    # Adjusted for 12-point scale
    conservative = results_df[results_df['Score'] >= 10].to_dict('records')
    balanced = results_df[(results_df['Score'] >= 8) & (results_df['Score'] < 10)].to_dict('records')
    aggressive = results_df[(results_df['Score'] >= 6) & (results_df['Score'] < 8)].to_dict('records')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Signals", len(results_df))
    with col2:
        st.metric("🔥 Conservative", len(conservative))
    with col3:
        st.metric("⚡ Balanced", len(balanced))
    with col4:
        st.metric("⚠️ Aggressive", len(aggressive))

    st.markdown("---")

    if len(conservative) > 0:
        st.markdown(f"""
            <div style="background:#1B5E20;padding:15px 0;border-radius:13px;margin-bottom:12px;text-align:center;width:99%;">
                <span style="color:#00FF00;font-size:1.7em;font-weight:700;">🔥 CONSERVATIVE (10-12 Points) 🔥</span>
            </div>
        """, unsafe_allow_html=True)

        for i in range(0, len(conservative), 2):
            row_tiles = conservative[i:i+2]
            cols = st.columns(2)
            for col, stock in zip(cols, row_tiles):
                with col:
                    st.markdown(render_confluence_tile(stock, fo_df), unsafe_allow_html=True)

    if len(balanced) > 0:
        st.markdown(f"""
            <div style="background:#4A3300;padding:15px 0;border-radius:13px;margin-bottom:12px;text-align:center;width:99%;margin-top:20px;">
                <span style="color:#FFD700;font-size:1.7em;font-weight:700;">⚡ BALANCED (8-9 Points) ⚡</span>
            </div>
        """, unsafe_allow_html=True)

        for i in range(0, len(balanced), 2):
            row_tiles = balanced[i:i+2]
            cols = st.columns(2)
            for col, stock in zip(cols, row_tiles):
                with col:
                    st.markdown(render_confluence_tile(stock, fo_df), unsafe_allow_html=True)

    if len(aggressive) > 0:
        st.markdown(f"""
            <div style="background:#4A2A00;padding:15px 0;border-radius:13px;margin-bottom:12px;text-align:center;width:99%;margin-top:20px;">
                <span style="color:#FFA500;font-size:1.7em;font-weight:700;">⚠️ AGGRESSIVE (6-7 Points) ⚠️</span>
            </div>
        """, unsafe_allow_html=True)

        for i in range(0, len(aggressive), 2):
            row_tiles = aggressive[i:i+2]
            cols = st.columns(2)
            for col, stock in zip(cols, row_tiles):
                with col:
                    st.markdown(render_confluence_tile(stock, fo_df), unsafe_allow_html=True)

    st.markdown("---")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Institutional Confluence Results CSV",
        data=csv,
        file_name=f'institutional_confluence_{timestamp}.csv',
        mime='text/csv',
        use_container_width=True,
        key="confluence_download"
    )

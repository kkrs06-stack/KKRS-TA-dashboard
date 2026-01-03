import pandas as pd
import numpy as np
import ta
import yfinance as yf
import streamlit as st
import datetime
import warnings
import sys
import os
import scipy.signal
from ta.trend import ADXIndicator, MACD
from ta.volatility import AverageTrueRange
from concurrent.futures import ThreadPoolExecutor, as_completed

from just_above_below_dashboard import just_above_below_dashboard, fetch_all_ohlcv, batch_analyze_all
from daily_weekly_dashboard import daily_weekly_dashboard
from ss_strat_dashboard import ss_strat_dashboard, get_ssstrat_stock_data_optimized
from ichi import ichi_dashboard
from confluence_dashboard import confluence_scanner_dashboard

warnings.filterwarnings("ignore")

# ========== Helper Functions ==========

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

def get_first_two(text):
    return " ".join(str(text).split()[:2]) if text else ""

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

st.set_page_config(layout="wide", page_title="Kamlesh TA Strategies", page_icon="📊")

today_str = datetime.datetime.now().strftime("%d-%b-%Y")
STRATEGIES = [
    {"label": "RSI Strategy", "id": "rsi"},
    {"label": "SS Strat", "id": "ssstrat"},
    {"label": "Daily > Weekly", "id": "dailyweekly"},
    {"label": "Just above/below", "id": "reversal"},
    {"label": "Ichi", "id": "ichi"},
    {"label": "RKO", "id": "rko"},
    {"label": "Confluence", "id": "confluence"},  # ← ADD THIS LINE
]

# ========== ENHANCEMENT FUNCTIONS (MATCHING JUST ABOVE/BELOW FORMAT) ==========

def calculate_support_resistance_fib(df, window=20):
    """Calculate support/resistance using swing points and Fibonacci"""
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
    """Determine if price is at support/resistance - MATCHING JUST ABOVE/BELOW FORMAT"""
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
    """Check if current volume is above/below average - MATCHING FORMAT"""
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
    """Detect OBV divergence using scipy peak detection - MATCHING FORMAT"""
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
    """Calculate multi-timeframe SMA20 alignment - FIXED"""
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

# ========== CONFLUENCE SCANNER FUNCTIONS ==========

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

def calculate_confluence_score(symbol, df):
    """Calculate confluence score (0-10)"""
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
        return {
            'Symbol': symbol, 'Direction': direction, 'Score': score,
            'Score_Breakdown': ' | '.join(score_breakdown) if score_breakdown else 'None',
            'Current_Price': round(current_price, 2), 'SMA_20': round(sma_20, 2),
            'SR_Status': sr_status, 'SR_Dist': round(sr_dist, 2) if sr_dist != "NA" and sr_dist else "NA",
            'Vol_Status': vol_status, 'Vol_Ratio': round(vol_ratio, 2), 'OBV_Div': obv_div_text,
            'MTF_Daily': mtf_d, 'MTF_Weekly': mtf_w, 'MTF_Monthly': mtf_m, 'MTF_Score': mtf_score,
            'ADX': round(adx_value, 2), 'Clean_Path': '✓' if clean_path else '✗',
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

@st.cache_data(show_spinner="Scanning for high-probability setups...", ttl=3600)
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

def render_confluence_tile(stock_data, fo_df):
    """Render single confluence tile"""
    symbol = stock_data['Symbol']
    symbol_clean = symbol.replace('.NS', '')
    name = get_company_name(symbol, fo_df)
    lot = get_lot_size(symbol, fo_df)
    price = stock_data['Current_Price']
    direction = stock_data['Direction']
    score = stock_data['Score']
    breakdown = stock_data['Score_Breakdown']
    sr_status, sr_dist = stock_data['SR_Status'], stock_data['SR_Dist']
    vol_status, vol_ratio = stock_data['Vol_Status'], stock_data['Vol_Ratio']
    obv_div = stock_data['OBV_Div']
    mtf_d, mtf_w, mtf_m, mtf_score = stock_data['MTF_Daily'], stock_data['MTF_Weekly'], stock_data['MTF_Monthly'], stock_data['MTF_Score']
    adx, clean_path = stock_data['ADX'], stock_data['Clean_Path']
    if score >= 8:
        score_color, score_bg = "#00FF00", "#1B5E20"
    elif score >= 7:
        score_color, score_bg = "#FFD700", "#4A3300"
    else:
        score_color, score_bg = "#FFA500", "#4A2A00"
    dir_color = "#37F553" if direction == "LONG" else "#FF3A3A"
    sr_display = f"{sr_status} ({sr_dist:+.1f}%)" if sr_dist != "NA" else sr_status
    vol_display = f"{vol_status} ({vol_ratio:.1f}x)"
    obv_display = f"OBV: {obv_div}"
    mtf_display = f"MTF: {mtf_d} {mtf_w} {mtf_m} ({mtf_score}/3)"
    tview_url = f"https://www.tradingview.com/chart/0EhGkhFw/?symbol=NSE:{symbol_clean}"
    return f"""
    <div style="background:#252525;border-radius:14px;width:380px;height:460px;position:relative;box-shadow:2px 4px 12px #000;margin-bottom:15px;border:2px solid {score_color};overflow:hidden;">
      <div style="width:100%;text-align:center;padding-top:6px;">
        <a href="{tview_url}" target="_blank" style="color:#fff;font-size:1.35em;font-weight:700;text-decoration:none;">{name}</a>
      </div>
      <div style="position:absolute;right:16px;top:6px;font-size:1.08em;color:#ECECEC;">Lot: <b>{lot}</b></div>
      <div style="background:{score_bg};width:90%;margin:8px auto;padding:10px;border-radius:10px;border:2px solid {score_color};">
        <div style="text-align:center;">
          <span style="font-size:2.4em;color:{score_color};font-weight:900;">{score}/10</span>
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
        <div style="font-size:1.12em;color:#ECECEC;">🛣️ Path: <b>{clean_path}</b></div>
      </div>
      <div style="width:90%;margin:8px auto;padding:6px;background:#1a1a1a;border-radius:8px;border:1px solid {score_color};">
        <div style="text-align:center;color:{score_color};font-weight:700;font-size:1.08em;margin-bottom:3px;">SCORE BREAKDOWN</div>
        <div style="font-size:1.02em;color:#ECECEC;text-align:center;">{breakdown}</div>
      </div>
    </div>
    """
# ========== RENKO SCANNER CLASS WITH ENHANCEMENTS - FIXED ==========
class RenkoScanner:
    def __init__(self, config):
        self.config = config

    def download_data(self, symbol, period='2y'):
        """Download historical data from yfinance - FIXED: 2y for MTF"""
        try:
            if not symbol.endswith('.NS'):
                symbol = symbol + ".NS"
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if df.empty:
                return None
            df.index = pd.to_datetime(df.index)
            return df
        except:
            return None

    def calculate_atr(self, df, period=14):
        """Calculate ATR - vectorized"""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(window=period).mean()
        return atr

    def build_renko(self, df):
        """Build Renko bricks - optimized"""
        atr_len = self.config['atr_length']
        atr_mult = self.config['atr_multiplier']
        atr = self.calculate_atr(df, atr_len)
        brick_size = atr.iloc[-1] * atr_mult
        prices = df['Close'].values
        bricks = []
        last_brick_close = round(prices[0] / brick_size) * brick_size
        for price in prices:
            diff = price - last_brick_close
            while abs(diff) >= brick_size:
                brick_open = last_brick_close
                if diff > 0:
                    last_brick_close += brick_size
                    direction = 1
                else:
                    last_brick_close -= brick_size
                    direction = -1
                brick_high = max(brick_open, last_brick_close)
                brick_low = min(brick_open, last_brick_close)
                bricks.append({'open': brick_open, 'high': brick_high, 'low': brick_low, 'close': last_brick_close, 'direction': direction})
                diff = price - last_brick_close
        brick_df = pd.DataFrame(bricks)
        brick_df['brick_size'] = brick_size
        return brick_df

    def calculate_supertrend_on_renko(self, brick_df):
        """Calculate Supertrend - vectorized where possible"""
        period = self.config['st_period']
        multiplier = self.config['st_multiplier']
        brick_df['tr'] = np.maximum(brick_df['high'] - brick_df['low'], np.maximum(np.abs(brick_df['high'] - brick_df['close'].shift()), np.abs(brick_df['low'] - brick_df['close'].shift())))
        brick_df['atr'] = brick_df['tr'].rolling(window=period, min_periods=1).mean()
        brick_df['atr'].fillna(brick_df['brick_size'].iloc[0], inplace=True)
        brick_df['hl2'] = (brick_df['high'] + brick_df['low']) / 2
        brick_df['upper_band'] = brick_df['hl2'] + (multiplier * brick_df['atr'])
        brick_df['lower_band'] = brick_df['hl2'] - (multiplier * brick_df['atr'])
        st_trend = np.ones(len(brick_df), dtype=np.int8)
        st_value = brick_df['lower_band'].values.copy()
        upper_band = brick_df['upper_band'].values
        lower_band = brick_df['lower_band'].values
        close = brick_df['close'].values
        for i in range(1, len(brick_df)):
            if close[i-1] > lower_band[i-1] and st_trend[i-1] == 1:
                lower_band[i] = max(lower_band[i], lower_band[i-1])
            if close[i-1] < upper_band[i-1] and st_trend[i-1] == -1:
                upper_band[i] = min(upper_band[i], upper_band[i-1])
            if close[i] > upper_band[i]:
                st_trend[i] = 1
                st_value[i] = lower_band[i]
            elif close[i] < lower_band[i]:
                st_trend[i] = -1
                st_value[i] = upper_band[i]
            else:
                st_trend[i] = st_trend[i-1]
                st_value[i] = lower_band[i] if st_trend[i] == 1 else upper_band[i]
        brick_df['st_trend'] = st_trend
        brick_df['st_value'] = st_value
        brick_df['upper_band'] = upper_band
        brick_df['lower_band'] = lower_band
        return brick_df

    def calculate_rsi_on_renko(self, brick_df):
        """Calculate RSI - vectorized"""
        period = self.config['rsi_period']
        delta = brick_df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        brick_df['rsi'] = rsi
        return brick_df

    def count_consecutive_bricks(self, brick_df):
        """Count consecutive bricks - vectorized"""
        direction = brick_df['direction'].values
        brick_count = np.zeros(len(brick_df), dtype=np.int16)
        reversal_confirmed = np.zeros(len(brick_df), dtype=bool)
        for i in range(2, len(brick_df)):
            if direction[i-2] != direction[i-1] and direction[i-1] == direction[i]:
                count = 1
                for j in range(i, -1, -1):
                    if direction[j] == direction[i]:
                        count += 1
                    else:
                        break
                brick_count[i] = count
                reversal_confirmed[i] = count >= 2
        brick_df['brick_count'] = brick_count
        brick_df['reversal_confirmed'] = reversal_confirmed
        return brick_df

    def generate_signals(self, symbol, df, brick_df):
        """Generate trading signals WITH ENHANCEMENTS"""
        latest = brick_df.iloc[-1]
        prev = brick_df.iloc[-2] if len(brick_df) > 1 else latest
        sma_period = self.config['sma_period']
        current_price = df['Close'].iloc[-1]
        sma_20 = df['Close'].rolling(window=sma_period, min_periods=1).mean().iloc[-1]
        st_flipped_bullish = (latest['st_trend'] == 1) and (prev['st_trend'] == -1)
        st_flipped_bearish = (latest['st_trend'] == -1) and (prev['st_trend'] == 1)
        signal = 'NO SIGNAL'
        entry_price = latest['close']
        sl_level = latest['st_value']
        risk_pct = abs((entry_price - sl_level) / entry_price * 100)
        long_check = (latest['brick_count'] >= self.config['min_green_bricks'] and latest['direction'] == 1 and latest['reversal_confirmed'] and (st_flipped_bullish if self.config['st_flip_mandatory'] else True) and latest['st_trend'] == 1 and current_price > sma_20 and latest['rsi'] > self.config['rsi_long_threshold'])
        short_check = (latest['brick_count'] >= self.config['min_red_bricks'] and latest['direction'] == -1 and latest['reversal_confirmed'] and (st_flipped_bearish if self.config['st_flip_mandatory'] else True) and latest['st_trend'] == -1 and current_price < sma_20 and latest['rsi'] < self.config['rsi_short_threshold'])
        if long_check:
            signal = 'LONG'
        elif short_check:
            signal = 'SHORT'
        elif latest['st_trend'] == 1:
            signal = 'HOLD (Bullish)'
        elif latest['st_trend'] == -1:
            signal = 'HOLD (Bearish)'
        # ===== CALCULATE ALL ENHANCEMENTS =====
        sr_data = calculate_support_resistance_fib(df, window=20)
        sr_status, sr_level_val, sr_dist = get_sr_status(current_price, sr_data)
        vol_status, vol_ratio, vol_color = check_volume_status(df, lookback=20)
        obv_div_text, obv_div_type = detect_obv_divergence_enhanced(df, lookback=14)
        mtf_d, mtf_w, mtf_m, mtf_score = calculate_mtf_alignment(df)
        return {
            'Symbol': symbol, 'Current_Price': round(current_price, 2), 'SMA_20': round(sma_20, 2),
            'Price_vs_SMA': 'ABOVE' if current_price > sma_20 else 'BELOW',
            'ST_Trend': 'UP' if latest['st_trend'] == 1 else 'DOWN', 'ST_Level': round(sl_level, 2),
            'RSI': round(latest['rsi'], 2), 'Consecutive_Bricks': int(latest['brick_count']),
            'Signal': signal, 'Entry_Price': round(entry_price, 2), 'SL_Level': round(sl_level, 2), 'Risk_Pct': round(risk_pct, 2),
            'SR_Status': sr_status, 'SR_Level': round(sr_level_val, 2) if sr_level_val else "NA", 'SR_Dist': round(sr_dist, 2) if sr_dist else "NA",
            'Vol_Status': vol_status, 'Vol_Ratio': round(vol_ratio, 2), 'Vol_Color': vol_color,
            'OBV_Div': obv_div_text, 'OBV_Div_Type': obv_div_type,
            'MTF_Daily': mtf_d, 'MTF_Weekly': mtf_w, 'MTF_Monthly': mtf_m, 'MTF_Score': mtf_score,
        }

    def process_single_stock(self, symbol):
        """Process a single stock"""
        try:
            df = self.download_data(symbol)
            if df is None or len(df) < 100:
                return None
            brick_df = self.build_renko(df)
            if len(brick_df) < 10:
                return None
            brick_df = self.calculate_supertrend_on_renko(brick_df)
            brick_df = self.calculate_rsi_on_renko(brick_df)
            brick_df = self.count_consecutive_bricks(brick_df)
            result = self.generate_signals(symbol, df, brick_df)
            return result
        except:
            return None

    def scan_stocks(self, stock_list, progress_bar=None, status_text=None):
        """Scan multiple stocks with parallel processing"""
        results = []
        total = len(stock_list)
        completed = 0
        max_workers = min(10, len(stock_list))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(self.process_single_stock, symbol): symbol for symbol in stock_list}
            for future in as_completed(future_to_symbol):
                completed += 1
                symbol = future_to_symbol[future]
                if status_text:
                    status_text.text(f"Processing {symbol}... ({completed}/{total})")
                if progress_bar:
                    progress_bar.progress(completed / total)
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except:
                    pass
        return pd.DataFrame(results)

# ========== RENKO HELPER FUNCTIONS - FIXED ==========
def rsi_colored_custom(rsi):
    """Color RSI based on value"""
    try:
        val = float(rsi)
        if val > 55:
            color = "#37F553"
        elif val > 50:
            color = "#FFD700"
        else:
            color = "#FF3A3A"
        return f"<span style='color:{color};font-weight:700;font-size:1.06em;'>{val:.2f}</span>"
    except:
        return f"<span style='color:#ECECEC;font-weight:700;'>{rsi}</span>"

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

def render_renko_tiles(stocks, signal_type, fo_df):
    """Render stock tiles WITH ENHANCEMENTS - MATCHING JUST ABOVE/BELOW FORMAT"""
    if not stocks:
        st.write("No stocks matched.")
        return
    for i in range(0, len(stocks), 2):
        row_tiles = stocks[i:i+2]
        tile_cols = st.columns(2)
        for tcol, s in zip(tile_cols, row_tiles):
            symbol = s['Symbol']
            symbol_clean = symbol.replace('.NS', '')
            name = get_company_name(symbol, fo_df)
            lot = get_lot_size(symbol, fo_df)
            price = s['Current_Price']
            sma_20 = s['SMA_20']
            price_vs_sma = s['Price_vs_SMA']
            st_trend = s['ST_Trend']
            st_level = s['ST_Level']
            rsi = s['RSI']
            bricks = s['Consecutive_Bricks']
            entry = s['Entry_Price']
            sl = s['SL_Level']
            risk = s['Risk_Pct']

            # Enhanced fields
            sr_status = s.get('SR_Status', '⚪ Unknown')
            sr_level_val = s.get('SR_Level', 'NA')
            sr_dist = s.get('SR_Dist', 'NA')
            vol_status = s.get('Vol_Status', '⚪ Unknown')
            vol_ratio = s.get('Vol_Ratio', 1.0)
            obv_div = s.get('OBV_Div', '⚪ Unknown')
            mtf_d = s.get('MTF_Daily', '⚪')
            mtf_w = s.get('MTF_Weekly', '⚪')
            mtf_m = s.get('MTF_Monthly', '⚪')
            mtf_score = s.get('MTF_Score', 0)

            # Format displays - MATCH JUST ABOVE/BELOW EXACTLY
            if sr_dist != "NA":
                sr_display = f"{sr_status} ({sr_dist:+.1f}%)"
            else:
                sr_display = sr_status

            vol_display = f"{vol_status} ({vol_ratio:.1f}x)"
            obv_display = f"OBV: {obv_div}"
            mtf_display = f"MTF: {mtf_d} {mtf_w} {mtf_m} ({mtf_score}/3)"

            rsi_html = rsi_colored_custom(rsi)
            sma_color = "#37F553" if price_vs_sma == "ABOVE" else "#FF3A3A"
            sma_text = f"Price {price_vs_sma} SMA"
            st_color = "#37F553" if st_trend == "UP" else "#FF3A3A"
            tview_url = f"https://www.tradingview.com/chart/0EhGkhFw/?symbol=NSE:{symbol_clean}"

            left_rows = [f"D: <b>{sma_20}</b>", f"<span style='color:{sma_color};font-weight:700;'>{sma_text}</span>", f"ST: <b>{st_level}</b> <span style='color:{st_color};font-weight:900;font-size:1.08em'>{st_trend}</span>", f"RSI: {rsi_html}"]
            right_rows = [f"<span style='color:#FFA500;font-weight:700;'>Bricks: {bricks}</span>", f"<span style='color:#00FF00;font-weight:700;'>Entry: ₹{entry}</span>", f"<span style='color:#FF3A3A;font-weight:700;'>SL: ₹{sl}</span>", f"<span style='color:#FFA500;font-weight:700;'>Risk: {risk}%</span>"]

            # Enhanced section - WITH VALUES
            enhanced_rows = [
                f"<div style='font-size:0.95em;color:#ECECEC;margin-bottom:2px;'>{sr_display}</div>",
                f"<div style='font-size:0.95em;color:#ECECEC;margin-bottom:2px;'>{vol_display}</div>",
                f"<div style='font-size:0.95em;color:#ECECEC;margin-bottom:2px;'>{obv_display}</div>",
                f"<div style='font-size:0.95em;color:#FFD700;font-weight:700;margin-bottom:2px;'>{mtf_display}</div>"
            ]

            left_html = "".join([f"<div style='font-size:0.98em;color:#ECECEC;margin-bottom:2px;'>{row}</div>" for row in left_rows])
            right_html = "".join([f"<div style='font-size:0.98em;margin-bottom:2px;'>{row}</div>" for row in right_rows])
            enhanced_html = "".join(enhanced_rows)

            tcol.markdown(f"""
            <div style="background:#252525;border-radius:14px;width:380px;height:400px;position:relative;box-shadow:1px 2px 8px #111;margin-bottom:15px;border:1px solid #333;overflow:hidden;">
              <div style="width:100%;text-align:center;padding-top:6px;">
                <a href="{tview_url}" target="_blank" style="color:#fff;font-size:1.08em;font-weight:700;text-decoration:none;">{name}</a>
              </div>
              <div style="position:absolute;right:16px;top:6px;font-size:1.08em;color:#ECECEC;">Lot: <span style="font-weight:bold;">{lot}</span></div>
              <div style="width:100%;text-align:center;margin-top:4px;margin-bottom:4px;">
                <span style="font-size:1.10em;color:#37F553;font-weight:700;">₹ {price}</span>
              </div>
              <div style="display:flex;flex-direction:row;width:100%;justify-content:space-between;padding:0 14px;margin-bottom:4px;">
                <div style="text-align:left;">{left_html}</div>
                <div style="text-align:right;">{right_html}</div>
              </div>
              <div style="width:90%;border-top:2px solid #FFD700;margin:8px auto;"></div>
              <div style="width:100%;padding:0 14px 6px 14px;background:#1a1a1a;margin:0 auto;">
                <div style="text-align:center;color:#FFD700;font-weight:700;font-size:1.12em;margin-bottom:4px;">🔥 ENHANCEMENTS 🔥</div>
                {enhanced_html}
              </div>
            </div>
            """, unsafe_allow_html=True)

@st.cache_data(show_spinner="Scanning Renko signals...")
def get_renko_scan_data(config_tuple):
    """Cached Renko scan"""
    config = {'atr_length': config_tuple[0], 'atr_multiplier': config_tuple[1], 'atr_timeframe': config_tuple[2], 'st_period': config_tuple[3], 'st_multiplier': config_tuple[4], 'rsi_period': config_tuple[5], 'sma_period': config_tuple[6], 'min_green_bricks': config_tuple[7], 'min_red_bricks': config_tuple[8], 'rsi_long_threshold': config_tuple[9], 'rsi_short_threshold': config_tuple[10], 'st_flip_mandatory': config_tuple[11]}
    fo_df = pd.read_csv('fo_stock_list.csv')
    fo_stocks = fo_df['symbol'].tolist()
    scanner = RenkoScanner(config)
    results_df = scanner.scan_stocks(fo_stocks)
    return results_df

@st.cache_data(show_spinner="Fast RSI scan for all F&O stocks...")
def get_rsi_strategy_data_optimized():
    fo_df = pd.read_csv("fo_stock_list.csv")
    tickers = fo_df["symbol"].str.strip().str.upper().tolist()
    ticker_name_map = dict(zip(fo_df["symbol"].str.strip().str.upper(), fo_df["name"]))
    lot_size_map = dict(zip(fo_df["symbol"].str.strip().str.upper(), fo_df["lot_size"]))
    df = yf.download(tickers=" ".join(tickers), period="20d", interval="1d", group_by='ticker', progress=False, auto_adjust=True, threads=True)
    results = []
    for symbol in tickers:
        try:
            d = df[symbol] if isinstance(df.columns, pd.MultiIndex) else df
            close = d["Close"].dropna() if "Close" in d else pd.Series([])
            price = close.iloc[-1] if len(close) > 0 else None
            prev_price = close.iloc[-2] if len(close) > 1 else price
            rsi = ta.momentum.RSIIndicator(close, window=14).rsi().dropna().iloc[-1] if len(close) > 14 else None
        except Exception:
            price, prev_price, rsi = None, None, None
        results.append({"Stock Name": ticker_name_map.get(symbol, symbol), "Symbol": symbol, "Lot Size": lot_size_map.get(symbol, ""), "Latest Price": price, "Previous Price": prev_price, "Daily RSI": rsi})
    df_out = pd.DataFrame(results)
    df_out["Latest Price"] = pd.to_numeric(df_out["Latest Price"], errors="coerce")
    df_out["Previous Price"] = pd.to_numeric(df_out["Previous Price"], errors="coerce")
    df_out["Daily RSI"] = pd.to_numeric(df_out["Daily RSI"], errors="coerce")
    return df_out

def clear_all_app_cache():
    """Single function to clear all caches"""
    get_rsi_strategy_data_optimized.clear()
    get_renko_scan_data.clear()
    get_confluence_scan_data.clear()  # ← ADD THIS LINE
    try:
        get_ssstrat_stock_data_optimized.clear()
    except:
        pass
    try:
        fetch_all_ohlcv.clear()
        batch_analyze_all.clear()
    except:
        pass
    try:
        from ichi import fetch_all_ohlcv as ichi_fetch_ohlcv, batch_analyze_ichi
        ichi_fetch_ohlcv.clear()
        batch_analyze_ichi.clear()
    except:
        pass

if "cache_cleared_initial" not in st.session_state:
    clear_all_app_cache()
    st.session_state["cache_cleared_initial"] = True

title_cols = st.columns([5, 1, 1])
with title_cols[0]:
    st.markdown("<div style='font-size:2.07em;font-weight:800;color:#FFD700;padding-bottom:2px;'>Kamlesh Technical Analysis Strategies</div>", unsafe_allow_html=True)
with title_cols[1]:
    st.markdown(f"<div style='text-align:right;font-size:1.29em;color:#FFD700;font-weight:700;padding-top:8px;'>{today_str}</div>", unsafe_allow_html=True)
with title_cols[2]:
    if st.button("❌ Clear Cache", key="clear_all_cache", use_container_width=True):
        clear_all_app_cache()
        st.success("✅ Cache cleared!")
        st.rerun()

selected = st.query_params.get('selected_strategy', 'rsi')
if isinstance(selected, list):
    selected = selected[0]

button_cols = st.columns(len(STRATEGIES))
for i, strat in enumerate(STRATEGIES):
    is_active = (strat["id"] == selected)
    if is_active:
        button_cols[i].markdown(f"<div style='background:#FF69B4;color:#fff;border-radius:10px;padding:12px 8px;text-align:center;font-weight:700;margin-bottom:5px;border:2px solid #FFD700;font-size:1.05em;letter-spacing:1.5px;'>{strat['label']}</div>", unsafe_allow_html=True)
    else:
        if button_cols[i].button(strat["label"], key=strat["id"], use_container_width=True):
            st.query_params.selected_strategy = strat["id"]
            st.rerun()

TRADINGVIEW_LINKS = {id: "https://www.tradingview.com/chart/HHuUSOTG/" for id in [s["id"] for s in STRATEGIES]}
tv_chart_url = TRADINGVIEW_LINKS.get(selected, "https://www.tradingview.com/chart/HHuUSOTG/")

if selected == "rsi":
    if st.button("🔄 Refresh RSI Data", key="refresh_rsi"):
        get_rsi_strategy_data_optimized.clear()
        st.rerun()
    df = get_rsi_strategy_data_optimized()
    filter_cols = st.columns(4)
    require_rsi_above_55 = filter_cols[0].checkbox("RSI ≥ 55", value=True)
    require_rsi_above_60 = filter_cols[1].checkbox("RSI ≥ 60", value=False)
    require_rsi_below_45 = filter_cols[2].checkbox("RSI ≤ 45", value=False)
    require_rsi_not_nan = filter_cols[3].checkbox("Valid RSI Only", value=True)
    search_cols = st.columns([7, 1])
    if search_cols[1].button("Reset", key="rsi_reset"):
        st.session_state["rsi_search"] = ""
        st.rerun()
    search_cols[0].markdown("<div style='font-size:1.12em;font-weight: bold; color:#FFD700; padding-bottom:3px'>Search Stock Name</div>", unsafe_allow_html=True)
    search_cols[0].text_input("", value=st.session_state.get("rsi_search", ""), key="rsi_search", placeholder="Type stock name...")
    search_text = st.session_state.get("rsi_search", "").strip().lower()
    filtered_df = df.copy()
    if require_rsi_not_nan:
        filtered_df = filtered_df[filtered_df["Daily RSI"].notnull()]
    if require_rsi_above_55:
        filtered_df = filtered_df[filtered_df["Daily RSI"] >= 55]
    if require_rsi_above_60:
        filtered_df = filtered_df[filtered_df["Daily RSI"] >= 60]
    if require_rsi_below_45:
        filtered_df = filtered_df[filtered_df["Daily RSI"] <= 45]
    if search_text:
        filtered_df = filtered_df[filtered_df["Stock Name"].str.lower().str.contains(search_text, na=False)]
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
    def card(row):
        price = row['Latest Price']
        prev = row['Previous Price']
        arrow, pcolor, change, pct = price_arrow_and_change(price, prev)
        price_str = f"{price:.2f}" if pd.notnull(price) else "NA"
        change_str = f"{change:+.2f}" if pd.notnull(price) and pd.notnull(prev) else "NA"
        pct_str = f"{pct:+.2f}%" if pd.notnull(price) and pd.notnull(prev) else "NA"
        tv_url = f"{tv_chart_url}?symbol=NSE:{row['Symbol'].replace('.NS', '')}"
        name_html = f'<a href="{tv_url}" target="_blank" style="color:#fff;text-decoration:none;"><span style="font-size:1.13em;text-align:center;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{get_first_two(row["Stock Name"])}</span></a>'
        rsi = row["Daily RSI"]
        rsi_txt = f'<span style="color:{rsi_tile_color(rsi)};font-weight:bold;">RSI: {round(rsi,2) if pd.notnull(rsi) else "NA"}</span>'
        lot_line = f'{rsi_txt}&nbsp;&nbsp;<span style="font-size:0.94em;">Lot: <b>{row["Lot Size"]}</b></span>'
        return f"""
            <div style="background-color:#252525;padding:10px 8px 10px 8px;border-radius:12px;min-width:140px;max-width:160px;margin-bottom:9px;border:1px solid #333;text-align:center;box-shadow:1px 2px 8px #111;color:#ECECEC;">
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
    num_cards = len(filtered_df)
    cols_per_row = 8
    rows_needed = (num_cards + cols_per_row - 1) // cols_per_row
    for row_idx in range(rows_needed):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            data_idx = row_idx * cols_per_row + col_idx
            if data_idx < num_cards:
                with cols[col_idx]:
                    st.markdown(card(filtered_df.iloc[data_idx]), unsafe_allow_html=True)
elif selected == "ssstrat":
    ss_strat_dashboard(tv_chart_url)
elif selected == "dailyweekly":
    daily_weekly_dashboard()
elif selected == "reversal":
    just_above_below_dashboard()
elif selected == "ichi":
    ichi_dashboard()
elif selected == "confluence":
    confluence_scanner_dashboard()
elif selected == "rko":
    st.markdown("<div style='font-size:2.0em;font-weight:800;color:#FFD700;padding-bottom:10px;text-align:center;'>🔥 RKO - Enhanced Renko Scanner 🔥</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:1.2em;color:#37F553;font-weight:700;padding:4px;'>✅ S/R Detection | ✅ Volume Analysis | ✅ OBV Divergence | ✅ MTF Alignment</div>", unsafe_allow_html=True)
    with st.sidebar:
        st.header("⚙️ RKO Configuration")
        st.subheader("Renko Settings")
        atr_length = st.number_input("ATR Length", min_value=5, max_value=30, value=14, key="rko_atr_len")
        atr_multiplier = st.number_input("ATR Multiplier", min_value=0.5, max_value=3.0, value=1.2, step=0.1, key="rko_atr_mult")
        st.subheader("Supertrend Settings")
        st_period = st.number_input("ST ATR Period", min_value=5, max_value=20, value=10, key="rko_st_period")
        st_multiplier = st.number_input("ST Multiplier", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="rko_st_mult")
        st.subheader("Signal Conditions")
        rsi_period = st.number_input("RSI Period", min_value=5, max_value=30, value=14, key="rko_rsi_period")
        sma_period = st.number_input("SMA Period", min_value=10, max_value=50, value=20, key="rko_sma_period")
        min_green_bricks = st.number_input("Min Green Bricks", min_value=1, max_value=5, value=2, key="rko_green")
        min_red_bricks = st.number_input("Min Red Bricks", min_value=1, max_value=5, value=2, key="rko_red")
        rsi_long_threshold = st.number_input("RSI Long Threshold", min_value=45, max_value=70, value=55, key="rko_rsi_long")
        rsi_short_threshold = st.number_input("RSI Short Threshold", min_value=30, max_value=55, value=45, key="rko_rsi_short")
        st_flip_mandatory = st.checkbox("ST Flip Mandatory", value=True, key="rko_st_flip")
        st.markdown("---")
        st.info("⚡ Enhanced v2.0 - FIXED: period='2y' + Values Display")
    if st.button("🔄 Refresh RKO Data", key="refresh_rko"):
        get_renko_scan_data.clear()
        st.rerun()
    try:
        fo_df = pd.read_csv('fo_stock_list.csv')
    except FileNotFoundError:
        st.error("❌ fo_stock_list.csv not found!")
        st.stop()
    config_tuple = (atr_length, atr_multiplier, 'daily', st_period, st_multiplier, rsi_period, sma_period, min_green_bricks, min_red_bricks, rsi_long_threshold, rsi_short_threshold, st_flip_mandatory)
    results_df = get_renko_scan_data(config_tuple)
    if len(results_df) == 0:
        st.warning("No results found")
    else:
        long_signals = results_df[results_df['Signal'] == 'LONG'].to_dict('records')
        short_signals = results_df[results_df['Signal'] == 'SHORT'].to_dict('records')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Scanned", len(results_df))
        with col2:
            st.metric("LONG Signals", len(long_signals))
        with col3:
            st.metric("SHORT Signals", len(short_signals))
        st.markdown("---")
        cols = st.columns(2)
        section_titles = ["Long", "Short"]
        section_colors = ["#18AA47", "#E53935"]
        section_dots = ["#80D8FF", "#FFA500"]
        section_tiles = [long_signals, short_signals]
        for idx, col in enumerate(cols):
            col.markdown(f"""<div style="background:{section_colors[idx]};padding:13px 0 13px 0;border-radius:13px;margin-bottom:12px;text-align:center;width:99%;"><span style="color:{section_dots[idx]};font-size:1.42em;font-weight:700;">&#x25CF;</span><span style="color:#FFFFFF;font-size:1.19em;font-weight:700;letter-spacing:2px;">{section_titles[idx]}</span></div>""", unsafe_allow_html=True)
            with col:
                render_renko_tiles(section_tiles[idx], section_titles[idx], fo_df)
        st.markdown("---")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        csv = results_df.to_csv(index=False)
        st.download_button(label="📥 Download Full Results CSV", data=csv, file_name=f'renko_scan_enhanced_{timestamp}.csv', mime='text/csv', use_container_width=True, key="rko_download")
else:
    st.markdown("> **Select a strategy tile above to view its details.**")

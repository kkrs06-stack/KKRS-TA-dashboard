import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

class RenkoScanner:
    def __init__(self, config):
        self.config = config
        
    def download_data(self, symbol, period='6mo'):
        """Download historical data from yfinance"""
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
        
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
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
                
                bricks.append({
                    'open': brick_open,
                    'high': brick_high,
                    'low': brick_low,
                    'close': last_brick_close,
                    'direction': direction
                })
                
                diff = price - last_brick_close
        
        brick_df = pd.DataFrame(bricks)
        brick_df['brick_size'] = brick_size
        
        return brick_df
    
    def calculate_supertrend_on_renko(self, brick_df):
        """Calculate Supertrend - vectorized where possible"""
        period = self.config['st_period']
        multiplier = self.config['st_multiplier']
        
        brick_df['tr'] = np.maximum(
            brick_df['high'] - brick_df['low'],
            np.maximum(
                np.abs(brick_df['high'] - brick_df['close'].shift()),
                np.abs(brick_df['low'] - brick_df['close'].shift())
            )
        )
        
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
        """Generate trading signals - optimized"""
        
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
        
        long_check = (
            latest['brick_count'] >= self.config['min_green_bricks'] and
            latest['direction'] == 1 and
            latest['reversal_confirmed'] and
            (st_flipped_bullish if self.config['st_flip_mandatory'] else True) and
            latest['st_trend'] == 1 and
            current_price > sma_20 and
            latest['rsi'] > self.config['rsi_long_threshold']
        )
        
        short_check = (
            latest['brick_count'] >= self.config['min_red_bricks'] and
            latest['direction'] == -1 and
            latest['reversal_confirmed'] and
            (st_flipped_bearish if self.config['st_flip_mandatory'] else True) and
            latest['st_trend'] == -1 and
            current_price < sma_20 and
            latest['rsi'] < self.config['rsi_short_threshold']
        )
        
        if long_check:
            signal = 'LONG'
        elif short_check:
            signal = 'SHORT'
        elif latest['st_trend'] == 1:
            signal = 'HOLD (Bullish)'
        elif latest['st_trend'] == -1:
            signal = 'HOLD (Bearish)'
        
        return {
            'Symbol': symbol,
            'Current_Price': round(current_price, 2),
            'SMA_20': round(sma_20, 2),
            'Price_vs_SMA': 'ABOVE' if current_price > sma_20 else 'BELOW',
            'ST_Trend': 'UP' if latest['st_trend'] == 1 else 'DOWN',
            'ST_Level': round(sl_level, 2),
            'RSI': round(latest['rsi'], 2),
            'Consecutive_Bricks': int(latest['brick_count']),
            'Signal': signal,
            'Entry_Price': round(entry_price, 2),
            'SL_Level': round(sl_level, 2),
            'Risk_Pct': round(risk_pct, 2)
        }
    
    def process_single_stock(self, symbol):
        """Process a single stock - used for parallel execution"""
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
            future_to_symbol = {executor.submit(self.process_single_stock, symbol): symbol 
                               for symbol in stock_list}
            
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


# ==================== HELPER FUNCTIONS ====================

def get_first_two(text):
    """Get first two words of company name"""
    return " ".join(str(text).split()[:2])


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


# ==================== STREAMLIT APP ====================

def render_renko_tiles(stocks, signal_type, fo_df):
    """Render stock tiles in Just Above/Below exact format"""
    if not stocks:
        st.write("No stocks matched.")
        return
    
    # Create tiles in groups of 2
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
            
            # RSI colored HTML
            rsi_html = rsi_colored_custom(rsi)
            
            # Price vs SMA color
            sma_color = "#37F553" if price_vs_sma == "ABOVE" else "#FF3A3A"
            sma_text = f"Price {price_vs_sma} SMA"
            
            # SuperTrend color
            if st_trend == "UP":
                st_color = "#37F553"
            else:
                st_color = "#FF3A3A"
            
            # TradingView URL
            tview_url = f"https://www.tradingview.com/chart/0EhGkhFw/?symbol=NSE:{symbol_clean}"
            
            # Left column data
            left_rows = [
                f"D: <b>{sma_20}</b>",
                f"<span style='color:{sma_color};font-weight:700;'>{sma_text}</span>",
                f"ST: <b>{st_level}</b> <span style='color:{st_color};font-weight:900;font-size:1.08em'>{st_trend}</span>",
                f"RSI: {rsi_html}",
                f"<span style='color:#FFA500;font-weight:700;'>Bricks: {bricks}</span>",
            ]
            
            # Right column data
            right_rows = [
                f"<span style='color:#00FF00;font-weight:700;'>Entry: ₹{entry}</span>",
                f"<span style='color:#FF3A3A;font-weight:700;'>SL: ₹{sl}</span>",
                f"<span style='color:#FFA500;font-weight:700;'>Risk: {risk}%</span>",
            ]
            
            left_html = "".join([f"<div style='font-size:0.98em;color:#ECECEC;margin-bottom:1px;'>{row}</div>" for row in left_rows])
            right_html = "".join([f"<div style='font-size:0.98em;margin-bottom:1px;'>{row}</div>" for row in right_rows])
            
            # Render tile - 300x245
            tcol.markdown(f"""
            <div style="background:#252525;border-radius:14px;width:300px;height:245px;position:relative;box-shadow:1px 2px 8px #111;margin-bottom:15px;display:flex;flex-direction:column;align-items:center;border:1px solid #333;">
              <div style="width:100%;text-align:center;margin-top:5px;">
                <a href="{tview_url}" target="_blank" style="color:#fff;font-size:1.08em;font-weight:700;text-decoration:none;">
                  {name}
                </a>
              </div>
              <div style="width:100%; position:relative;">
                <div style="position:absolute;right:16px;top:-9px; font-size:0.88em;color:#ECECEC;text-align:right;">
                    Lot: <span style="font-weight:bold;">{lot}</span>
                </div>
              </div>
              <div style="width:100%;text-align:center;margin-top:2px;margin-bottom:2px;">
                <span style="font-size:1.10em;color:#37F553;font-weight:700;">
                  ₹ {price}
                </span>
              </div>
              <div style="display:flex;flex-direction:row;width:100%;justify-content:space-between;margin-top:4px;">
                <div style="padding-left:14px;text-align:left;">
                    {left_html}
                </div>
                <div style="padding-right:14px;text-align:right;">
                    {right_html}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Renko Scanner", page_icon="📊", layout="wide")
    
    today_str = datetime.now().strftime("%d-%b-%Y")
    st.markdown(
        "<div style='font-size:2em;font-weight:800;color:#FFD700;padding-bottom:2px;'>Renko Scanner</div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:right;font-size:1.2em;color:#FFD700;font-weight:700;padding-top:4px;'>{today_str}</div>",
        unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Renko Settings")
        atr_length = st.number_input("ATR Length", min_value=5, max_value=30, value=14)
        atr_multiplier = st.number_input("ATR Multiplier", min_value=0.5, max_value=3.0, value=1.2, step=0.1)
        
        st.subheader("Supertrend Settings")
        st_period = st.number_input("ST ATR Period", min_value=5, max_value=20, value=10)
        st_multiplier = st.number_input("ST Multiplier", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        
        st.subheader("Signal Conditions")
        rsi_period = st.number_input("RSI Period", min_value=5, max_value=30, value=14)
        sma_period = st.number_input("SMA Period", min_value=10, max_value=50, value=20)
        min_green_bricks = st.number_input("Min Green Bricks", min_value=1, max_value=5, value=2)
        min_red_bricks = st.number_input("Min Red Bricks", min_value=1, max_value=5, value=2)
        rsi_long_threshold = st.number_input("RSI Long Threshold", min_value=45, max_value=70, value=55)
        rsi_short_threshold = st.number_input("RSI Short Threshold", min_value=30, max_value=55, value=45)
        st_flip_mandatory = st.checkbox("ST Flip Mandatory", value=True)
        
        st.markdown("---")
        st.info("⚡ Parallel processing enabled")
    
    # Load CSV
    try:
        fo_df = pd.read_csv('fo_stock_list.csv')
        fo_stocks = fo_df['symbol'].tolist()
    except FileNotFoundError:
        st.error("❌ fo_stock_list.csv not found!")
        return
    except KeyError:
        st.error("❌ 'symbol' column not found in CSV!")
        return
    
    # Refresh and Scan buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col_btn2:
        scan_button = st.button("🚀 Start Scan", type="primary", use_container_width=True)
    
    if scan_button:
        start_time = datetime.now()
        
        config = {
            'atr_length': atr_length,
            'atr_multiplier': atr_multiplier,
            'atr_timeframe': 'daily',
            'st_period': st_period,
            'st_multiplier': st_multiplier,
            'rsi_period': rsi_period,
            'sma_period': sma_period,
            'min_green_bricks': min_green_bricks,
            'min_red_bricks': min_red_bricks,
            'rsi_long_threshold': rsi_long_threshold,
            'rsi_short_threshold': rsi_short_threshold,
            'st_flip_mandatory': st_flip_mandatory
        }
        
        scanner = RenkoScanner(config)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run scan
        results_df = scanner.scan_stocks(fo_stocks, progress_bar, status_text)
        
        end_time = datetime.now()
        scan_duration = (end_time - start_time).total_seconds()
        
        progress_bar.empty()
        status_text.empty()
        
        if len(results_df) == 0:
            st.warning("No results found")
            return
        
        # Filter signals
        long_signals = results_df[results_df['Signal'] == 'LONG'].to_dict('records')
        short_signals = results_df[results_df['Signal'] == 'SHORT'].to_dict('records')
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Scanned", len(results_df))
        with col2:
            st.metric("LONG Signals", len(long_signals))
        with col3:
            st.metric("SHORT Signals", len(short_signals))
        
        st.markdown(f"<div style='text-align:center;color:#FFD700;font-size:1.1em;'>⏱️ Scan completed in {scan_duration:.1f}s</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Create two columns for Long and Short
        cols = st.columns(2)
        section_titles = ["Long", "Short"]
        section_colors = ["#18AA47", "#E53935"]
        section_dots = ["#80D8FF", "#FFA500"]
        section_tiles = [long_signals, short_signals]
        
        for idx, col in enumerate(cols):
            col.markdown(f'''
                <div style="background:{section_colors[idx]};padding:13px 0 13px 0;border-radius:13px;margin-bottom:12px;text-align:center;width:99%;">
                <span style="color:{section_dots[idx]};font-size:1.42em;font-weight:700;">&#x25CF;</span>
                <span style="color:#FFFFFF;font-size:1.19em;font-weight:700;letter-spacing:2px;">{section_titles[idx]}</span></div>
            ''', unsafe_allow_html=True)
            
            with col:
                render_renko_tiles(section_tiles[idx], section_titles[idx], fo_df)
        
        # Download button
        st.markdown("---")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results CSV",
            data=csv,
            file_name=f'renko_scan_{timestamp}.csv',
            mime='text/csv',
            use_container_width=True
        )

if __name__ == "__main__":
    main()

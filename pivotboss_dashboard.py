"""
PIVOTBOSS SCANNER v2.0 - PARALLEL EOD/INTRADAY ENGINE
- Uses batched yfinance download for all symbols
- Parallel per-symbol PivotBoss scan using ThreadPoolExecutor
- Supports multiple chart timeframes (1h, 4h, 1D, 1W, 1M)
- Keeps original PivotBoss UI (LONG/SHORT tiles + enhancements)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import scipy.signal
import warnings
#import exchange_calendars as ec
import pytz

#NSE_CAL = ec.get_calendar("NSE")
IST = pytz.timezone("Asia/Kolkata")

warnings.filterwarnings("ignore")


# =====================================================
# ENHANCEMENT FUNCTIONS (RENKO/PIVOTBOSS STYLE)
# =====================================================

def calculatesupportresistancefib(df, window=20):
    try:
        highs = df["High"].tail(window)
        lows = df["Low"].tail(window)

        swinghigh = highs.max()
        swinglow = lows.min()

        week52high = df["High"].tail(252).max()
        week52low = df["Low"].tail(252).min()

        fibrange = week52high - week52low
        fiblevels = {
            "fib0": week52low,
            "fib236": week52low + fibrange * 0.236,
            "fib382": week52low + fibrange * 0.382,
            "fib500": week52low + fibrange * 0.500,
            "fib618": week52low + fibrange * 0.618,
            "fib786": week52low + fibrange * 0.786,
            "fib100": week52high,
        }

        allsupportlevels = [swinglow, week52low] + [
            fiblevels["fib236"],
            fiblevels["fib382"],
            fiblevels["fib500"],
        ]
        allresistancelevels = [swinghigh, week52high] + [
            fiblevels["fib618"],
            fiblevels["fib786"],
        ]

        return {
            "supportlevels": sorted(allsupportlevels),
            "resistancelevels": sorted(allresistancelevels, reverse=True),
            "swinghigh": swinghigh,
            "swinglow": swinglow,
            "week52high": week52high,
            "week52low": week52low,
            "fiblevels": fiblevels,
        }
    except:
        return None


def getsrstatus(price, srdata, threshold=0.02):
    if srdata is None:
        return "Unknown", None, None
    try:
        price = float(price)
        distto52whigh = (srdata["week52high"] - price) / price * 100
        distto52wlow = (price - srdata["week52low"]) / price * 100

        if abs(distto52whigh) <= 2:
            return "Near 52W High", srdata["week52high"], distto52whigh
        if abs(distto52wlow) <= 2:
            return "Near 52W Low", srdata["week52low"], -distto52wlow

        for support in srdata["supportlevels"]:
            dist = (price - support) / support * 100
            if abs(dist) <= threshold * 100:
                return "At Support", support, dist

        for resistance in srdata["resistancelevels"]:
            dist = (resistance - price) / price * 100
            if abs(dist) <= threshold * 100:
                return "At Resistance", resistance, dist

        return "Mid-range", None, None
    except:
        return "Unknown", None, None


def checkvolumestatus(df, lookback=20):
    try:
        currentvol = df["Volume"].iloc[-1]
        avgvol = df["Volume"].tail(lookback).mean()
        ratio = currentvol / avgvol if avgvol > 0 else 1.0

        if ratio >= 1.5:
            return "High Vol", ratio, "#37F553"
        elif ratio <= 0.8:
            return "Low Vol", ratio, "#FF3A3A"
        else:
            return "Avg Vol", ratio, "#FFD700"
    except:
        return "Unknown", 1.0, "#ECECEC"


def detectobvdivergenceenhanced(df, lookback=14):
    try:
        obv = [0]
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
                obv.append(obv[-1] + df["Volume"].iloc[i])
            elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
                obv.append(obv[-1] - df["Volume"].iloc[i])
            else:
                obv.append(obv[-1])

        price = df["Close"].values
        obvvals = np.array(obv)

        pricelows = scipy.signal.argrelextrema(price, np.less, order=lookback)[0]
        pricehighs = scipy.signal.argrelextrema(price, np.greater, order=lookback)[0]
        obvlows = scipy.signal.argrelextrema(obvvals, np.less, order=lookback)[0]
        obvhighs = scipy.signal.argrelextrema(obvvals, np.greater, order=lookback)[0]

        bulldiv = False
        if len(pricelows) >= 2 and len(obvlows) >= 2:
            if (
                price[pricelows[-1]] < price[pricelows[-2]]
                and obvvals[obvlows[-1]] > obvvals[obvlows[-2]]
            ):
                bulldiv = True

        beardiv = False
        if len(pricehighs) >= 2 and len(obvhighs) >= 2:
            if (
                price[pricehighs[-1]] > price[pricehighs[-2]]
                and obvvals[obvhighs[-1]] < obvvals[obvhighs[-2]]
            ):
                beardiv = True

        obvnow = obv[-1]
        obvprev = obv[-lookback - 1] if len(obv) > lookback else obv[0]
        obvtrend = "Bull" if obvnow > obvprev else "Bear" if obvnow < obvprev else "Flat"

        if bulldiv:
            return f"{obvtrend} Bull Div", "bullish"
        elif beardiv:
            return f"{obvtrend} Bear Div", "bearish"
        else:
            return f"{obvtrend} No Div", "none"
    except:
        return "Unknown", "unknown"


def calculatemtfalignment(df):
    try:
        if df is None or len(df) < 60:
            return "", "", "", 0

        price = float(df["Close"].iloc[-1])

        try:
            sma20d = float(df["Close"].rolling(20).mean().iloc[-1])
            dailyok = price > sma20d
        except:
            dailyok = False

        weeklyok = None
        try:
            dfw = df["Close"].resample("W-FRI").last().dropna()
            if len(dfw) >= 20:
                sma20w = float(dfw.rolling(20).mean().iloc[-1])
                weeklyok = price > sma20w
        except:
            weeklyok = None

        monthlyok = None
        try:
            dfm = df["Close"].resample("M").last().dropna()
            if len(dfm) >= 20:
                sma20m = float(dfm.rolling(20).mean().iloc[-1])
                monthlyok = price > sma20m
        except:
            monthlyok = None

        mtf_d = "↑" if dailyok else "↓"
        mtf_w = "↑" if weeklyok is True else "↓" if weeklyok is False else "-"
        mtf_m = "↑" if monthlyok is True else "↓" if monthlyok is False else "-"
        mtfscore = sum(1 for x in [dailyok, weeklyok, monthlyok] if x is True)

        return mtf_d, mtf_w, mtf_m, mtfscore
    except:
        return "", "", "", 0


def calculate_supertrend_ohlc(df, period=10, multiplier=3.0):
    """
    Candle-based Supertrend, adapted from your Renko version.
    Returns df with columns: 'ST_Trend' (1/-1), 'ST_Value'.
    """
    df = df.copy()
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=1).mean()
    if len(atr) >= period:
        atr.iloc[:period] = atr.iloc[period - 1]

    hl2 = (high + low) / 2.0
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    st_trend = np.ones(len(df), dtype=np.int8)
    st_value = lowerband.copy()

    for i in range(1, len(df)):
        if close.iloc[i - 1] <= upperband.iloc[i - 1] and st_trend[i - 1] == 1:
            upperband.iloc[i] = min(upperband.iloc[i], upperband.iloc[i - 1])
        if close.iloc[i - 1] >= lowerband.iloc[i - 1] and st_trend[i - 1] == -1:
            lowerband.iloc[i] = max(lowerband.iloc[i], lowerband.iloc[i - 1])

        if close.iloc[i] > upperband.iloc[i]:
            st_trend[i] = 1
            st_value.iloc[i] = lowerband.iloc[i]
        elif close.iloc[i] < lowerband.iloc[i]:
            st_trend[i] = -1
            st_value.iloc[i] = upperband.iloc[i]
        else:
            st_trend[i] = st_trend[i - 1]
            st_value.iloc[i] = lowerband.iloc[i] if st_trend[i] == 1 else upperband.iloc[i]

    df["ST_Trend"] = st_trend
    df["ST_Value"] = st_value
    return df

# =====================================================
# PIVOTBOSS VWAP ENGINE
# =====================================================

class PivotBossEngine:
    def __init__(self, config):
        self.config = config
        self.signal_type = config.get("signal_type", "Range")

    def _get_vwap_source(self, df):
        src_opt = self.config.get("vwap_source", "Close") or "Close"

        if src_opt == "Close" or src_opt.startswith("Close"):
            return df["Close"]
        if src_opt.startswith("HL2"):
            return (df["High"] + df["Low"]) / 2.0
        if src_opt.startswith("HLC3"):
            return (df["High"] + df["Low"] + df["Close"]) / 3.0
        if src_opt == "OHLC4":
            return (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0

        return df["Close"]

    def _make_anchor_key(self, idx):
        tf = self.config.get("anchor_tf", "1 Day")
        mult = max(int(self.config.get("anchor_mult", 1)), 1)

        if tf == "1 Day":
            days = pd.Index(idx.date)
            base, _ = pd.factorize(days)
            return base // mult

        if tf == "1 Week":
            weeks = idx.to_period("W-MON").view("int64")
            return weeks // mult

        if tf == "1 Month":
            months = idx.to_period("M").view("int64")
            base = months // mult
        elif tf == "3 Months":
            quarters = idx.to_period("Q").view("int64")
            base = quarters // mult
        elif tf == "6 Months":
            halfyear = idx.to_period("Q").view("int64") // 2
            base = halfyear // mult
        elif tf == "9 Months":
            three_q = idx.to_period("Q").view("int64") // 3
            base = three_q // mult
        elif tf == "12 Months":
            years = idx.to_period("A").view("int64")
            base = years // mult
        else:
            days = pd.Index(idx.date)
            base, _ = pd.factorize(days)
            base = base // mult

        return base

    def calculate_pivotboss_bands(self, df):
        df = df.copy()

        src = self._get_vwap_source(df)
        groups = self._make_anchor_key(df.index)

        src = src.copy()
        if src.name is None:
            src.name = "src"

        vol = df["Volume"]
        pv = src * vol

        cum_vol = vol.groupby(groups).cumsum()
        cum_pv = pv.groupby(groups).cumsum()
        df["VWAP"] = cum_pv / cum_vol

        length = int(self.config.get("sd_periods", 300))
        k = float(self.config.get("std_devs", 1.0))

        spread = (df["VWAP"] - df["Close"]).abs()
        df["stdev"] = spread.rolling(length, min_periods=max(25, length // 5)).std()

        df["R1"] = df["VWAP"] + k * df["stdev"]
        df["R2"] = df["VWAP"] + 2 * k * df["stdev"]
        df["R3"] = df["VWAP"] + 3 * k * df["stdev"]
        df["R4"] = df["VWAP"] + 4 * k * df["stdev"]
        df["S1"] = df["VWAP"] - k * df["stdev"]
        df["S2"] = df["VWAP"] - 2 * k * df["stdev"]
        df["S3"] = df["VWAP"] - 3 * k * df["stdev"]
        df["S4"] = df["VWAP"] - 4 * k * df["stdev"]

        return df

    def detect_reclaims(self, df):
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest

        tol_pct = float(self.config.get("band_tolerance_pct", 0.0))
        tol = tol_pct / 100.0

        o = latest["Open"]
        h = latest["High"]
        l = latest["Low"]
        c = latest["Close"]

        o1 = prev["Open"]
        c1 = prev["Close"]

        S3 = latest["S3"]
        R3 = latest["R3"]

        is_green = c > o
        is_red   = c < o

        # ---------- LONG: S3 reclaim + green ----------
        # Define S3 zone for "wash through"
        s3_low  = S3 * (1 - tol)
        s3_high = S3 * (1 + tol)

        # Same-bar: low must go below/into S3 zone, close must finish ABOVE S3, candle green
        cond_s3_same = (
            (l <= s3_high) and   # traded into/below S3 zone
            (c > S3) and         # close clearly above S3 line
            is_green
        )

        # 2-bar: prior close and current open below/into S3 zone, current close above S3, candle green
        cond_s3_prev = (
            (c1 <= s3_high) and
            (o  <= s3_high) and
            (c  > S3) and
            is_green
        )

        buy_from_s3 = cond_s3_same or cond_s3_prev

        # ---------- SHORT: R3 reclaim + red ----------
        # Define R3 zone for "wash above"
        r3_low  = R3 * (1 - tol)
        r3_high = R3 * (1 + tol)

        # Same-bar: high must go above/into R3 zone, close must finish BELOW R3, candle red
        cond_r3_same = (
            (h >= r3_low) and    # actually tags R3 zone
            (c < r3_high) and         # close clearly below R3 line
            is_red
        )

        # 2-bar: prior close and current open above/into R3 zone, current close below R3, candle red
        cond_r3_prev = (
            (c1 >= r3_low) and
            (o  >= r3_low) and
            (c  < r3_high) and
            is_red
        )

        sell_from_r3 = cond_r3_same or cond_r3_prev

        # Range mode: only S3 / R3
        range_buy_base = buy_from_s3
        range_sell_base = sell_from_r3

        # Trending mode: symmetric mapping
        trending_buy_base  = sell_from_r3
        trending_sell_base = buy_from_s3

        if self.signal_type == "Trending":
            base_long = trending_buy_base
            base_short = trending_sell_base
        else:
            base_long = range_buy_base
            base_short = range_sell_base

        # DEBUG: only when we actually have a signal
        #if base_long or base_short:
        #    debug_rows = df.iloc[-3:].copy()
        #    debug_rows["S3"] = debug_rows["S3"].round(2)
        #    debug_rows["R3"] = debug_rows["R3"].round(2)
        #    print("=== PVB RECLAIM DEBUG (FILTERED) ===")
        #    print(debug_rows[["Open", "High", "Low", "Close", "S3", "R3"]])
        #    print(
        #        "buy_from_s3:", buy_from_s3,
        #        "| sell_from_r3:", sell_from_r3,
        #        "| base_long:", base_long,
        #        "| base_short:", base_short,
        #    )

        return base_long, base_short

    def volume_confirmation(self, df):
        bars_back = self.config.get("bars_back", 50)
        vol_mult = self.config.get("vol_multiplier", 1.2)

        df["avg_vol"] = df["Volume"].rolling(bars_back).mean()
        df["vol_confirm"] = df["Volume"] > (df["avg_vol"] * vol_mult)
        df["delta"] = (df["Close"] - df["Open"]) * df["Volume"]
        df["buy_delta"] = df["delta"] > 0
        df["sell_delta"] = df["delta"] < 0

        return (
            df["vol_confirm"].iloc[-1],
            df["buy_delta"].iloc[-1],
            df["sell_delta"].iloc[-1],
        )

    def detect_vwap_regime(self, df, lookback=30, slope_threshold_bp=0.5):
        """
        Classify VWAP as 'Trending' or 'Range' based on recent slope and band touches.
        slope_threshold_bp is in basis points per bar (0.5 = 0.5% over 100 bars).
        """
        if "VWAP" not in df.columns or len(df) < lookback + 5:
            return "Unknown"

        sub = df.tail(lookback).copy()
        vwap = sub["VWAP"].values
        idx = np.arange(len(vwap))

        if len(vwap) < 5:
            return "Unknown"

        slope, _ = np.polyfit(idx, vwap, 1)
        mid = vwap.mean()
        slope_pct_per_bar = (slope / mid) * 100 if mid != 0 else 0

        touches_upper = (sub["High"] >= sub["R2"]).sum()
        touches_lower = (sub["Low"] <= sub["S2"]).sum()
        touch_ratio = (touches_upper + touches_lower) / len(sub)

        strong_slope = abs(slope_pct_per_bar) >= slope_threshold_bp / 100.0
        strong_trend_bands = touch_ratio >= 0.3

        if strong_slope or strong_trend_bands:
            return "Trending"
        else:
            return "Range"

    def generatesignals(self, symbol, df):
        try:
            if df is None or len(df) < 40:
                print(
                    f"{symbol}: skipped in generatesignals, len(df)={0 if df is None else len(df)}"
                )
                return None

            df = self.calculate_pivotboss_bands(df)

            df = df.dropna(subset=["VWAP"]).copy()
            if df.empty:
                print(f"{symbol}: all VWAP NaN after bands calc")
                return None

            df = df.ffill()

            st_period = int(self.config.get("st_period", 10))
            st_mult = float(self.config.get("st_multiplier", 3.0))
            df = calculate_supertrend_ohlc(df, period=st_period, multiplier=st_mult)

            latest = df.iloc[-1]
            if pd.isna(latest.get("S3")) or pd.isna(latest.get("R3")):
                print(f"{symbol}: latest S3/R3 NaN")
                return None

            base_long, base_short = self.detect_reclaims(df)
            vol_ok, buy_delta, sell_delta = self.volume_confirmation(df)

            current_price = float(latest["Close"])
            vwap = float(latest["VWAP"])
            s3_level = float(latest["S3"])
            r3_level = float(latest["R3"])

            st_trend_raw = int(latest["ST_Trend"])
            st_level = float(latest["ST_Value"])
            st_trend_label = "UP" if st_trend_raw == 1 else "DOWN"

            confirmed_long = base_long and (vol_ok or buy_delta)
            confirmed_short = base_short and (vol_ok or sell_delta)

            if confirmed_long:
                signal = "LONG"
            elif confirmed_short:
                signal = "SHORT"
            else:
                signal = "HOLD Bullish" if current_price >= vwap else "HOLD Bearish"

            if (signal == "LONG" and confirmed_long) or (
                signal == "SHORT" and confirmed_short
            ):
                strength = "Strong"
            elif (signal == "LONG" and base_long) or (signal == "SHORT" and base_short):
                strength = "Early"
            else:
                strength = ""

            entry_price = current_price
            sl_long = s3_level
            sl_short = r3_level

            if signal == "LONG":
                risk_pct = abs(entry_price - sl_long) / entry_price * 100
            elif signal == "SHORT":
                risk_pct = abs(entry_price - sl_short) / entry_price * 100
            else:
                risk_pct = 0.0

            srdata = calculatesupportresistancefib(df, window=20)
            sr_status, sr_level, sr_dist = getsrstatus(current_price, srdata)
            vol_status, vol_ratio, vol_color = checkvolumestatus(df, lookback=20)
            obv_div, obv_type = detectobvdivergenceenhanced(df, lookback=14)
            mtf_d, mtf_w, mtf_m, mtf_score = calculatemtfalignment(df)

            vwap_regime = self.detect_vwap_regime(
                df, lookback=30, slope_threshold_bp=0.5
            )

            rsi_val = ta.momentum.rsi(df["Close"], window=14).iloc[-1]
            try:
                adx_series = ta.trend.adx(
                    high=df["High"], low=df["Low"], close=df["Close"], window=14
                )
                adx_val = float(adx_series.iloc[-1])
            except Exception:
                adx_val = np.nan

            try:
                di_pos = ta.trend.adx_pos(
                    high=df["High"], low=df["Low"], close=df["Close"], window=14
                ).iloc[-1]
                di_neg = ta.trend.adx_neg(
                    high=df["High"], low=df["Low"], close=df["Close"], window=14
                ).iloc[-1]
            except Exception:
                di_pos, di_neg = np.nan, np.nan

            # DEBUG: inspect POWERGRID behaviour
            #if symbol.startswith("POWERGRID"):
            #    print(
            #        "POWERGRID DEBUG:",
            #        "R3", latest["R3"],
            #        "High", latest["High"],
            #        "Close", latest["Close"],
            #        "base_short", base_short,
            #        "vol_ok", vol_ok,
            #        "sell_delta", sell_delta,
            #        "signal", signal,
            #    )

            return {
                "Symbol": symbol.replace(".NS", ""),
                "CurrentPrice": round(current_price, 2),
                "VWAP": round(vwap, 2),
                "PricevsVWAP": "ABOVE" if current_price > vwap else "BELOW",
                "VWAPRegime": vwap_regime,
                "STTrend": st_trend_label,
                "STLevel": round(st_level, 2),
                "RSI": round(rsi_val, 2),
                "ADX": round(adx_val, 2) if not np.isnan(adx_val) else "NA",
                "DIP": round(di_pos, 2) if not np.isnan(di_pos) else "NA",
                "DIN": round(di_neg, 2) if not np.isnan(di_neg) else "NA",
                "ConsecutiveBricks": "N/A (Bands)",
                "Signal": signal,
                "SignalStrength": strength,
                "EntryPrice": round(entry_price, 2),
                "SLLevel": round(sl_long if signal == "LONG" else sl_short, 2),
                "RiskPct": round(risk_pct, 2),
                "SRStatus": sr_status,
                "SRLevel": round(sr_level, 2) if sr_level else "NA",
                "SRDist": round(sr_dist, 2) if sr_dist else "NA",
                "VolStatus": vol_status,
                "VolRatio": round(vol_ratio, 2),
                "VolColor": vol_color,
                "OBVDiv": obv_div,
                "OBVDivType": obv_type,
                "MTFDaily": mtf_d,
                "MTFWeekly": mtf_w,
                "MTFMonthly": mtf_m,
                "MTFScore": mtf_score,
            }
        except Exception as e:
            print(f"{symbol}: ERROR in generatesignals -> {e}")
            return None

# =====================================================
# BATCH DATA ENGINE (DAILY OR 1-HOUR)
# =====================================================

@st.cache_data(show_spinner="Downloading all symbol data ...")
def fetch_all_ohlcv(ticker_list, base_interval):
    """
    base_interval: '1d' for EOD, '60m' for intraday base bars.
    """
    tickers_str = " ".join(ticker_list)

    if base_interval == "1d":
        period = "500d"
    else:
        period = "60d"

    df = yf.download(
        tickers=tickers_str,
        period=period,
        interval=base_interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    data_dict = {}
    if isinstance(df.columns, pd.MultiIndex):
        for symbol in ticker_list:
            if symbol in df:
                sdf = df[symbol].dropna()
                sdf = sdf[~sdf.index.duplicated(keep="first")]
                if "Volume" in sdf.columns:
                    sdf = sdf[sdf["Volume"] > 0]
                sdf = sdf[
                    ~(
                        (sdf["Open"] == sdf["High"])
                        & (sdf["High"] == sdf["Low"])
                        & (sdf["Low"] == sdf["Close"])
                    )
                ]
                if sdf.index.tz is None:
                    sdf.index = sdf.index.tz_localize("UTC")
                sdf.index = sdf.index.tz_convert(IST)
                data_dict[symbol] = sdf
    else:
        sdf = df.dropna()
        sdf = sdf[~sdf.index.duplicated(keep="first")]
        data_dict[ticker_list[0]] = sdf
    return data_dict


def process_fo_stock_list():
    try:
        fo_df = pd.read_csv("fo_stock_list.csv")
        if "lot_size" in fo_df.columns and "lotsize" not in fo_df.columns:
            fo_df = fo_df.rename(columns={"lot_size": "lotsize"})
        return fo_df
    except Exception:
        st.error("Could not read fo_stock_list.csv")
        return pd.DataFrame()


def getfirsttwotext(text):
    return " ".join(str(text).split()[:2])


def getlotsize(symbol, fodf):
    try:
        row = fodf[fodf["symbol"] == symbol]
        if not row.empty:
            return row.iloc[0].get("lotsize", "")
    except:
        pass
    return ""


def getcompanyname(symbol, fodf):
    try:
        row = fodf[fodf["symbol"] == symbol]
        if not row.empty:
            name = row.iloc[0].get("name", symbol)
            return getfirsttwotext(name)
    except:
        pass
    return symbol.replace(".NS", "")

# =====================================================
# PARALLEL PIVOTBOSS PROCESSING
# =====================================================

def resample_to_chart_tf(df, chart_tf, base_interval):
    """
    Resample base 60m or 1d data to requested chart timeframe,
    using IST. Only apply 09:15–15:30 session filter when using intraday base data.
    """
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}

    # Ensure IST timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.tz_convert(IST)

    # If base is intraday (60m), restrict to NSE hours; if daily, keep all
    if base_interval == "60m":
        session_df = df.between_time("09:15", "15:30")
    else:
        session_df = df

    if chart_tf == "1h":
        # Base is already 60m from yfinance
        return session_df

    if chart_tf == "4h":
        return session_df.resample("240min").agg(agg).dropna()

    if chart_tf == "1D":
        return session_df.resample("1D").agg(agg).dropna()

    if chart_tf == "1W":
        return session_df.resample("W-FRI").agg(agg).dropna()

    if chart_tf == "1M":
        return session_df.resample("M").agg(agg).dropna()

    return df

def process_single_pivotboss(args):
    symbol, lot, name, data_dict, config = args
    df = data_dict.get(symbol)
    if df is None or len(df) < 40:
        print(f"{symbol}: data_dict df is too short ({0 if df is None else len(df)})")
        return None

    chart_tf = config.get("chart_tf", "1D")
    base_interval = config.get("base_interval", "1d")
    df = resample_to_chart_tf(df, chart_tf, base_interval)
    if df is None or len(df) < 40:
        print(f"{symbol}: df too short after chart_tf={chart_tf} resample ({len(df)})")
        return None

    engine = PivotBossEngine(config)
    res = engine.generatesignals(symbol, df)
    if res is None:
        return None
    res["Lot"] = lot
    res["Name"] = getfirsttwotext(name)
    return res


@st.cache_data(show_spinner="Running PivotBoss scan in parallel...")
def batch_scan_pivotboss(data_dict, fo_df, config):
    args_list = []
    for _, row in fo_df.iterrows():
        symbol = row["symbol"]
        lot = row.get("lotsize", "")
        name = row.get("name", symbol)
        args_list.append((symbol, lot, name, data_dict, config))

    results = []
    if not args_list:
        return [], []

    max_workers = min(10, len(args_list))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {
            executor.submit(process_single_pivotboss, args): args for args in args_list
        }
        for future in as_completed(future_to_args):
            try:
                r = future.result()
                if r is not None:
                    results.append(r)
            except:
                pass

    long_res = [r for r in results if r["Signal"] == "LONG"]
    short_res = [r for r in results if r["Signal"] == "SHORT"]
    return long_res, short_res

# =====================================================
# TILE RENDERING
# =====================================================

def rsicolored(rsi):
    try:
        val = float(rsi)
        if val >= 55:
            color = "#37F553"
        elif val >= 50:
            color = "#FFD700"
        else:
            color = "#FF3A3A"
        return f'<span style="color:{color};font-weight:700;font-size:1.06em">{val:.2f}</span>'
    except:
        return '<span style="#ECECEC;font-weight:700">rsi</span>'


def rendertiles(stocks, signaltype, fodf):
    if not stocks:
        st.write("No stocks matched.")
        return

    for i in range(0, len(stocks), 2):
        row_tiles = stocks[i : i + 2]
        tile_cols = st.columns(2)

        for tcol, s in zip(tile_cols, row_tiles):
            symbol_clean = s["Symbol"]
            name = getcompanyname(symbol_clean, fodf)
            lot = s.get("Lot", getlotsize(symbol_clean, fodf))

            price = s["CurrentPrice"]
            vwap = s["VWAP"]
            pricevsvwap = s["PricevsVWAP"]
            sttrend = s["STTrend"]
            stlevel = s["STLevel"]
            rsi = s["RSI"]
            adx = s.get("ADX", "NA")
            dip = s.get("DIP", "NA")
            din = s.get("DIN", "NA")
            bricks = s["ConsecutiveBricks"]
            entry = s["EntryPrice"]
            sl = s["SLLevel"]
            risk = s["RiskPct"]
            signal = s.get("Signal", "")
            strength = s.get("SignalStrength", "")

            if strength:
                signal_label = f"{signal} ({strength})"
            else:
                signal_label = signal

            vwap_regime = s.get("VWAPRegime", "Unknown")

            srstatus = s.get("SRStatus", "Unknown")
            srdist = s.get("SRDist", "NA")
            volstatus = s.get("VolStatus", "Unknown")
            volratio = s.get("VolRatio", 1.0)
            obvdiv = s.get("OBVDiv", "Unknown")
            mtfd = s.get("MTFDaily", "")
            mtfw = s.get("MTFWeekly", "")
            mtfm = s.get("MTFMonthly", "")
            mtfscore = s.get("MTFScore", 0)

            smacolor = "#37F553" if pricevsvwap == "ABOVE" else "#FF3A3A"
            stcolor = "#37F553" if sttrend == "UP" else "#FF3A3A"

            rsihtml = rsicolored(rsi)
            smatext = f"Price {pricevsvwap} VWAP"

            leftrows = [
                f"📊 {vwap}",
                f'<span style="color:{smacolor};font-weight:700">{smatext}</span>',
                f"ST {stlevel} <span style=\"color:{stcolor};font-weight:900;font-size:1.08em\">{sttrend}</span>",
                f"RSI {rsihtml}",
            ]

            rightrows = [
                "<span style='color:#FFA500;font-weight:700'>Bricks {}</span>".format(bricks),
                "<span style='color:#00FF00;font-weight:700'>Entry {}</span>".format(entry),
                "<span style='color:#FF3A3A;font-weight:700'>SL {}</span>".format(sl),
                "<span style='color:#FFA500;font-weight:700'>Risk {}%</span>".format(risk),
            ]

            if srdist != "NA" and isinstance(srdist, (int, float)):
                srdisplay = f"{srstatus} {srdist:.1f}%"
            else:
                srdisplay = srstatus

            voldisplay = f"{volstatus} {volratio:.1f}x"
            obvdisplay = f"OBV {obvdiv}"
            mtfdisplay = f"MTF {mtfd}{mtfw}{mtfm} {mtfscore}/3"

            enhancedrows = [
                f'<div style="font-size:0.95em;color:#ECECEC;margin-bottom:2px">{srdisplay}</div>',
                f'<div style="font-size:0.95em;color:#ECECEC;margin-bottom:2px">{voldisplay}</div>',
                f'<div style="font-size:0.95em;color:#ECECEC;margin-bottom:2px">{obvdisplay}</div>',
                f'<div style="font-size:0.95em;color:#FFD700;font-weight:700;margin-bottom:2px">{mtfdisplay}</div>',
            ]

            lefthtml = "\n".join(
                [
                    f'<div style="font-size:0.98em;color:#ECECEC;margin-bottom:2px">{row}</div>'
                    for row in leftrows
                ]
            )
            righthtml = "\n".join(
                [f'<div style="font-size:0.98em;margin-bottom:2px">{row}</div>' for row in rightrows]
            )
            enhancedhtml = "\n".join(enhancedrows)

            tviewurl = f"https://www.tradingview.com/chart/RaPnty9s/?symbol=NSE%3A{symbol_clean}"

            if signal.startswith("LONG"):
                sig_bg = "#18AA47"
            elif signal.startswith("SHORT"):
                sig_bg = "#E53935"
            else:
                sig_bg = "#555555"

            di_plus_html = f"<span style='color:#18AA47;font-size:1em;'>DI+ {dip}</span>"
            di_minus_html = f"<span style='color:#E53935;font-size:1em;'>DI- {din}</span>"
            adx_html = f"<span style='color:#FF1493;font-size:1em;font-weight:700;'>ADX {adx}</span>"

            card_html = f"""
            <div style="background:#252525;border-radius:14px;width:380px;height:460px;position:relative;
                        box-shadow:1px 2px 8px #111;margin-bottom:15px;border:1px solid #333;overflow:hidden">
                <div style="width:100%;text-align:center;padding-top:6px">
                    <a href="{tviewurl}" target="_blank"
                       style="color:#fff;font-size:1.08em;font-weight:700;text-decoration:none">{name}</a>
                </div>
                <div style="position:absolute;left:14px;top:6px;font-size:0.82em;
                            background:{sig_bg};color:#fff;padding:2px 8px;border-radius:10px;font-weight:700">
                    {signal_label}
                </div>
                <div style="position:absolute;left:14px;top:30px;font-size:0.8em;color:#FFD700;">
                    VWAP: {vwap_regime}
                </div>
                <div style="position:absolute;right:16px;top:6px;font-size:0.88em;color:#ECECEC">
                    Lot <span style="font-weight:bold">{lot}</span>
                </div>
                <div style="width:100%;text-align:center;margin-top:18px;margin-bottom:4px">
                    <span style="font-size:1.10em;color:#37F553;font-weight:700">₹{price}</span>
                </div>
                <div style="display:flex;flex-direction:row;width:100%;justify-content:space-between;
                            padding:0 14px;margin-bottom:4px">
                    <div style="text-align:left">{lefthtml}</div>
                    <div style="text-align:right">{righthtml}</div>
                </div>
                <div style="width:90%;border-top:2px solid #FFD700;margin:8px auto"></div>
                <div style="width:100%;padding:0 14px 6px 14px;background:#1a1a1a;margin:0 auto">
                    <div style="text-align:center;color:#FFD700;font-weight:700;font-size:0.92em;margin-bottom:4px">
                        🔍 ENHANCEMENTS
                    </div>
                    {enhancedhtml}
                </div>
                <div style="width:90%;border-top:1px solid #444;margin:6px auto"></div>
                <div style="width:100%;text-align:center;padding:6px 0 10px 0;">
                    <span style="font-size:1.02em;">{di_plus_html}</span>
                    &nbsp;&nbsp;
                    <span style="font-size:1.02em;">{di_minus_html}</span>
                    &nbsp;&nbsp;
                    <span style="font-size:1.02em;">{adx_html}</span>
                </div>
            </div>
            """
            tcol.markdown(card_html, unsafe_allow_html=True)

# =====================================================
# MAIN STREAMLIT APP
# =====================================================

def run_pivotboss_tab():
    st.title("PVB Signal")

    st.sidebar.header("PivotBoss VWAP Settings")

    anchor_tf = st.sidebar.selectbox(
        "Anchor Timeframe (VWAP Period)",
        ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "9 Months", "12 Months"],
        index=0,
    )

    anchor_mult = st.sidebar.number_input(
        "Anchor Timeframe Multiplier (N Period)",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
    )

    st.sidebar.markdown(
        "<span class='chart-tf-label'>Chart Timeframe (bars)</span>",
        unsafe_allow_html=True,
    )
    chart_tf = st.sidebar.selectbox(
        "",
        ["1h", "4h", "1D", "1W", "1M"],
        index=2,
    )

    vwap_source_opt = st.sidebar.selectbox(
        "VWAP Source",
        ["Close", "HL2 (High+Low)/2", "HLC3 (High+Low+Close)/3", "OHLC4"],
        index=0,
    )

    st.sidebar.header("Bands / Signals")
    std_devs = st.sidebar.number_input("StdDev Multiplier", 0.5, 3.0, 1.0, 0.1)
    sd_periods = st.sidebar.number_input("Band Length", 50, 500, 300, 50)
    vol_multiplier = st.sidebar.number_input("Vol Multiplier", 1.0, 2.0, 1.2, 0.1)
    enable_r2s2 = st.sidebar.checkbox("Enable R2/S2 Signals", value=False)
    enable_r1s1 = st.sidebar.checkbox("Enable R1/S1 Signals", value=False)
    signal_type = st.sidebar.selectbox("Signal Type", ["Range", "Trending"], index=0)

    band_tolerance_pct = st.sidebar.slider(
        "Band Tolerance (%) for Range",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.25,
        help="Tolerance around R/S bands when checking RANGE reclaims",
    )

    st.sidebar.header("Supertrend Settings")
    st_period = st.sidebar.number_input(
        "ST ATR Period", min_value=5, max_value=30, value=10, step=1
    )
    st_multiplier = st.sidebar.number_input(
        "ST Multiplier", min_value=1.0, max_value=5.0, value=3.0, step=0.1
    )

    max_symbols = st.sidebar.slider("Max symbols to scan", 10, 200, 50, 10)

    if st.button("🔄 Refresh Data Cache"):
        fetch_all_ohlcv.clear()
        batch_scan_pivotboss.clear()

    fo_df = process_fo_stock_list()
    if fo_df.empty:
        return

    fo_df = fo_df.iloc[:max_symbols].copy()
    ticker_list = list(fo_df["symbol"])

    if chart_tf in ["1h", "4h"]:
        base_interval = "60m"
    else:
        base_interval = "1d"

    data_dict = fetch_all_ohlcv(ticker_list, base_interval)

    run = st.button("Run PivotBoss Scan")
    if not run:
        return

    # FROM HERE ON, WE ARE INSIDE THE 'run' PATH
    config = {
        "signal_type": signal_type,
        "std_devs": std_devs,
        "sd_periods": sd_periods,
        "vol_multiplier": vol_multiplier,
        "enable_r2s2": enable_r2s2,
        "enable_r1s1": enable_r1s1,
        "bars_back": 50,
        "anchor_tf": anchor_tf,
        "anchor_mult": anchor_mult,
        "vwap_source": vwap_source_opt,
        "chart_tf": chart_tf,
        "st_period": st_period,
        "st_multiplier": st_multiplier,
        "base_interval": base_interval,
        "band_tolerance_pct": band_tolerance_pct,
    }


    longsignals, shortsignals = batch_scan_pivotboss(data_dict, fo_df, config)

    total = len(longsignals) + len(shortsignals)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Signals", total)
    with col2:
        st.metric("LONG Signals", len(longsignals))
    with col3:
        st.metric("SHORT Signals", len(shortsignals))

    st.markdown("---")
    cols = st.columns(2)
    for idx, (title, tiles) in enumerate(zip(["LONG", "SHORT"], [longsignals, shortsignals])):
        with cols[idx]:
            st.markdown(
                f"<div style='background:{'#18AA47' if idx == 0 else '#E53935'};"
                "padding:13px 0;border-radius:13px;margin-bottom:12px;text-align:center;width:99%'>"
                f"<span style='color:{'#80D8FF' if idx == 0 else '#FFA500'};"
                "font-size:1.42em;font-weight:700'>●</span>"
                f"<span style='color:#FFF;font-size:1.19em;font-weight:700;letter-spacing:2px'>{title}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            rendertiles(tiles, title, fo_df)


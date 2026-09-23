"""
PIVOTBOSS SCANNER v2.0 - PARALLEL EOD/INTRADAY ENGINE
- Uses Dhan for all symbol data (via dhan_pivotboss_adapter)
- Parallel per-symbol PivotBoss scan using ThreadPoolExecutor
- Supports multiple chart timeframes (1h, 4h, 1D, 1W, 1M)
- Keeps original PivotBoss UI (LONG/SHORT tiles + enhancements)
"""

import streamlit as st
import pandas as pd
import numpy as np
import ta
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import scipy.signal
import warnings
#import exchange_calendars as ec
import pytz

from dotenv import load_dotenv
load_dotenv("dhan.env")

from dhan_pivotboss_adapter import fetch_all_ohlcv_dhan
from dhan_auth import DhanTokenManager
from dhan_instruments import DhanInstrumentLookup
from dhan_market_data import DhanMarketData

st.set_page_config(layout="wide")

#NSE_CAL = ec.get_calendar("NSE")
IST = pytz.timezone("Asia/Kolkata")

warnings.filterwarnings("ignore")

_token_manager = DhanTokenManager()
_lookup = DhanInstrumentLookup()
_market = DhanMarketData(token_manager=_token_manager, instrument_lookup=_lookup)


# =====================================================
# LOOK & FEEL -- same dark "ledger" palette as Ribbon Scanner/Portfolio
# Dashboard (this exact mockup's own hex values), applied to PivotBoss's
# own subject: a VWAP band ladder (S3-S2-S1-VWAP-R1-R2-R3) replaces the
# EMA-ribbon glyph, since PivotBoss's signal is band-based, not EMA-based.
#
# _CSS below is rendered via st.markdown(unsafe_allow_html=True), not
# st.html() -- it (and the tiles) contain inline SVG, which st.html()
# silently strips. Kept deliberately comment-free and any HTML that gets
# concatenated across many tiles is built as single-line strings -- both
# are load-bearing: a stray comment or a 4-space-indented multi-line
# f-string broke this exact rendering path building Ribbon Scanner. See
# the project_streamlit_html_patterns memory for the full writeup.
#
# No logic changes anywhere in this file -- only new fields threaded
# through for display (S1-S3/R1-R3 levels, IsFresh) and this rendering
# layer. Every existing signal-computation branch is untouched.
# =====================================================

_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@1,500;1,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
.pb-wrap{
  --ink:#0e1512; --panel:#141d19; --panel-2:#182220; --line:#26332e; --line-soft:#1d2925;
  --text:#e9ede8; --text-dim:#93a39a; --text-faint:#5f7269;
  --long:#4fd1a0; --long-dim:#2c5f4c; --short:#e8785f; --short-dim:#6b3a30;
  --fresh:#dcae53; --hold:#7d93c2;
  font-family:"IBM Plex Sans",system-ui,sans-serif;
  color:var(--text);
}
.pb-wrap .eyebrow{
  font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--text-faint);margin:0 0 6px;
}
.pb-wrap h1{
  font-family:"Fraunces",Georgia,serif;font-style:italic;font-weight:500;font-size:32px;
  line-height:1.05;margin:0 0 8px;color:var(--text);
}
.pb-wrap .rule-line{
  font-size:16.3px;color:var(--text-dim);width:100%;max-width:none;line-height:1.5;margin:0 0 14px;
}
.pb-wrap .rule-line b{color:var(--text);font-weight:600;}
div[data-testid="stMarkdownContainer"] p{max-width:none!important;}
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:#141d19!important;border:1px solid #26332e!important;
  border-radius:3px!important;padding:6px 10px!important;
}
div[data-testid="stButton"] button{
  background:#141d19!important;border:1px solid #26332e!important;
  color:#93a39a!important;font-family:"IBM Plex Sans",sans-serif!important;
}
div[data-testid="stButton"] button:hover{border-color:#dcae53!important;color:#e9ede8!important;}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label{
  font-family:"IBM Plex Mono",monospace!important;font-size:13.1px!important;
  letter-spacing:.1em;text-transform:uppercase;color:#5f7269!important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div{
  background:#182220!important;border-color:#26332e!important;color:#e9ede8!important;
  font-family:"IBM Plex Sans",sans-serif!important;font-size:13.1px!important;
}
div[data-testid="stSelectbox"] svg{fill:#93a39a!important;}
div[data-testid="stNumberInput"] input{
  background:#182220!important;border:1px solid #26332e!important;
  color:#e9ede8!important;font-family:"IBM Plex Mono",monospace!important;font-size:13.1px!important;
}
div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"],
div[data-testid="stSlider"] div[data-testid="stThumbValue"]{
  font-size:13.1px!important;
}
.pb-wrap .summary{display:flex;align-items:center;padding:14px 20px;background:var(--panel-2);border:1px solid var(--line-soft);border-radius:3px;margin-bottom:22px;font-family:"IBM Plex Mono",monospace;font-size:20px;color:var(--text-dim);flex-wrap:wrap;row-gap:8px;}
.pb-wrap .summary .step{display:flex;align-items:baseline;gap:8px;padding:0 16px;border-right:1px solid var(--line-soft);}
.pb-wrap .summary .step:last-child{border-right:none;}
.pb-wrap .summary .step b{font-size:20px;color:var(--text);font-weight:700;}
.pb-wrap .summary .step .lbl{color:var(--text-faint);font-family:"IBM Plex Sans",sans-serif;font-size:20px;}
.pb-wrap .long-tag{color:var(--long);}
.pb-wrap .short-tag{color:var(--short);}
.pb-wrap .hold-tag{color:var(--hold);}
.pb-wrap .grid-label{font-family:"IBM Plex Mono",monospace;font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--text-faint);margin:0 0 14px;}
.pb-wrap .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px;}
.pb-wrap .tile{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:18px;position:relative;}
.pb-wrap .tile.fresh{border-color:var(--fresh);box-shadow:0 0 0 1px var(--fresh),0 0 20px -6px rgba(220,174,83,.45);}
.pb-wrap .tile-top{display:flex;justify-content:space-between;align-items:flex-start;}
.pb-wrap .symbol{font-weight:700;font-size:21px;}
.pb-wrap .symbol a{color:inherit;text-decoration:none;border-bottom:1px dotted var(--text-faint);}
.pb-wrap .symbol a:hover{border-bottom-color:var(--fresh);color:var(--fresh);}
.pb-wrap .company{font-size:14px;color:var(--text-faint);margin-top:2px;}
.pb-wrap .dir-badge{font-family:"IBM Plex Mono",monospace;font-size:13px;font-weight:700;letter-spacing:.06em;padding:4px 9px;border-radius:2px;white-space:nowrap;}
.pb-wrap .dir-badge.long{color:var(--long);background:rgba(79,209,160,.12);}
.pb-wrap .dir-badge.short{color:var(--short);background:rgba(232,120,95,.12);}
.pb-wrap .dir-badge.hold{color:var(--hold);background:rgba(125,147,194,.14);}
.pb-wrap .tag-row{display:flex;gap:8px;margin:10px 0 11px;flex-wrap:wrap;}
.pb-wrap .pill{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:16px;color:var(--text-dim);border:1px solid var(--line);border-radius:20px;padding:3px 10px;}
.pb-wrap .ladder-box svg{display:block;width:100%;height:74px;}
.pb-wrap .price-row{display:flex;align-items:baseline;gap:10px;font-family:"IBM Plex Mono",monospace;margin:8px 0 12px;font-variant-numeric:tabular-nums;flex-wrap:wrap;}
.pb-wrap .price{font-size:22px;font-weight:700;}
.pb-wrap .chg{font-size:18px;font-weight:600;}
.pb-wrap .chg.up{color:var(--long);}
.pb-wrap .chg.down{color:var(--short);}
.pb-wrap .stat-row{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;font-size:17px;padding:7px 0;border-top:1px solid var(--line-soft);font-variant-numeric:tabular-nums;gap:10px;}
.pb-wrap .stat-row .k{color:var(--text-faint);white-space:nowrap;}
.pb-wrap .stat-row .v{color:var(--text);font-family:"IBM Plex Mono",monospace;font-weight:600;white-space:nowrap;}
.pb-wrap .stat-row .v.long{color:var(--long);}
.pb-wrap .stat-row .v.short{color:var(--short);}
.pb-wrap .rsi-gauge{width:130px;height:16px;position:relative;flex:1;}
.pb-wrap .rsi-track{position:absolute;top:7px;left:0;right:0;height:2px;background:var(--line);}
.pb-wrap .rsi-mark{position:absolute;top:2px;width:2px;height:13px;background:var(--text);}
.pb-wrap .mtf-dots{display:flex;gap:8px;align-items:center;}
.pb-wrap .mtf-dot{display:flex;flex-direction:column;align-items:center;gap:3px;}
.pb-wrap .mtf-dot .lbl{font-size:11px;color:var(--text-faint);font-family:"IBM Plex Mono",monospace;}
.pb-wrap .mtf-dot .dot{width:9px;height:9px;border-radius:50%;}
.pb-wrap .mtf-dot .dot.up{background:var(--long);}
.pb-wrap .mtf-dot .dot.down{background:var(--short);}
.pb-wrap .mtf-dot .dot.flat{background:var(--text-faint);}
.pb-wrap .trade-plan{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;margin-top:10px;padding-top:10px;border-top:1px solid var(--line-soft);}
.pb-wrap .trade-plan div{text-align:center;}
.pb-wrap .trade-plan .tp-label{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--text-faint);font-family:"IBM Plex Mono",monospace;margin-bottom:4px;}
.pb-wrap .trade-plan .tp-value{font-family:"IBM Plex Mono",monospace;font-size:16px;font-weight:600;}
.pb-wrap .fresh-note{font-family:"IBM Plex Mono",monospace;font-size:13.5px;color:var(--fresh);margin-top:10px;padding-top:10px;border-top:1px solid var(--line-soft);display:flex;align-items:center;gap:7px;}
.pb-wrap .fresh-note .ring{width:7px;height:7px;border-radius:50%;background:var(--fresh);box-shadow:0 0 6px var(--fresh);flex-shrink:0;}
</style>
"""

_HEADER_HTML = """
<div class="pb-wrap">
  <p class="eyebrow">Trading Suite</p>
  <h1>PivotBoss VWAP</h1>
  <p class="rule-line">
    Anchored-VWAP bands (&plusmn;1/2/3 std. dev.) with Supertrend confirmation.
    A <b>LONG</b> reclaims S3 on a green candle, a <b>SHORT</b> rejects R3 on a red one &mdash;
    each confirmed by volume delta, RSI/ADX momentum, OBV divergence and D/W/M alignment.
  </p>
</div>
"""


_LADDER_TICKS_X = (10, 53, 96, 140, 184, 227, 270)


def _vwap_ladder_svg(s3, s2, s1, vwap, r1, r2, r3, price, direction) -> str:
    """S3-S2-S1-VWAP-R1-R2-R3 ladder with a price marker and a shaded
    tolerance zone at whichever end the signal's own reclaim/rejection
    happens at -- LONG's zone sits at S3 (band_tolerance_pct governs how
    wide a "reclaim" counts around it), SHORT's at R3. HOLD gets a marker
    with no zone, since neither band condition fired."""
    color = {"LONG": "#4fd1a0", "SHORT": "#e8785f"}.get(direction, "#7d93c2")
    lo, hi = min(s3, r3), max(s3, r3)
    span = (hi - lo) or 1.0
    frac = max(0.0, min(1.0, (price - lo) / span))
    marker_x = 10 + frac * 260
    zone = ""
    if direction == "LONG":
        zone = '<rect x="1" y="27" width="18" height="16" fill="#4fd1a0" opacity="0.18"/>'
    elif direction == "SHORT":
        zone = '<rect x="261" y="27" width="18" height="16" fill="#e8785f" opacity="0.18"/>'
    tick_circles = "".join(f'<circle cx="{x}" cy="35" r="2.6"/>' for x in _LADDER_TICKS_X)
    return (
        f'<svg viewBox="0 0 280 66">'
        f'<line x1="10" y1="35" x2="270" y2="35" stroke="#26332e" stroke-width="2"/>'
        f'{zone}'
        f'<line x1="140" y1="19" x2="140" y2="52" stroke="#93a39a" stroke-width="1.6"/>'
        f'<text x="140" y="14" fill="#93a39a" font-size="10.5" font-family="IBM Plex Mono" text-anchor="middle">VWAP {vwap:,.1f}</text>'
        f'{tick_circles}'
        f'<g fill="#5f7269" font-size="10.5" font-family="IBM Plex Mono" text-anchor="middle">'
        f'<text x="10" y="59">S3</text><text x="96" y="59">S1</text><text x="184" y="59">R1</text><text x="270" y="59">R3</text>'
        f'</g>'
        f'<circle cx="{marker_x:.1f}" cy="35" r="5.9" fill="{color}"/>'
        f'</svg>'
    )


def _pb_rsi_gauge_html(rsi_val) -> str:
    """Not a banded gauge like Ribbon Scanner's -- PivotBoss doesn't gate
    on an RSI range, just colors by the same >=55/>=50/else thresholds
    rsicolored() already used for the old tile's plain text."""
    try:
        val = float(rsi_val)
    except (TypeError, ValueError):
        return '<span class="rsi-gauge"><span class="rsi-track"></span></span>'
    color = "#4fd1a0" if val >= 55 else ("#dcae53" if val >= 50 else "#e8785f")
    marker_pos = max(0.0, min(100.0, val))
    return f'<span class="rsi-gauge"><span class="rsi-track"></span><span class="rsi-mark" style="left:{marker_pos}%;background:{color};"></span></span>'


def _mtf_dots_html(mtf_fast, mtf_d, mtf_w) -> str:
    def _cls(arrow):
        return "up" if arrow == "↑" else ("down" if arrow == "↓" else "flat")
    dots = (
        f'<span class="mtf-dot"><span class="dot {_cls(mtf_fast)}"></span><span class="lbl">9</span></span>'
        f'<span class="mtf-dot"><span class="dot {_cls(mtf_d)}"></span><span class="lbl">D</span></span>'
        f'<span class="mtf-dot"><span class="dot {_cls(mtf_w)}"></span><span class="lbl">W</span></span>'
    )
    return f'<span class="mtf-dots">{dots}</span>'


@st.cache_data(show_spinner=False, ttl=45)
def fetch_live_prices(symbols: tuple[str, ...]) -> dict[str, float]:
    """
    Genuinely live prices via /marketfeed/ltp -- unlike the historical
    daily/intraday data used for the chart series and indicators, this
    endpoint isn't subject to Dhan's next-morning EOD publish lag. Only
    called for the small shortlist that already passed the scan, not the
    whole F&O universe. Falls back to an empty dict on any failure so a
    live-price hiccup degrades to "no color" rather than breaking the scan.
    """
    if not symbols:
        return {}
    try:
        return _market.get_ltp(list(symbols))
    except Exception as exc:
        print(f"fetch_live_prices failed: {exc}")
        return {}


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
    """Sized for a swing hold of <=4 weeks (per Kamlesh, 2026-09-23): the old
    Monthly leg needed 20 months of history to compute (fetch only covers
    ~13), so it was permanently stuck at "-". Replaced with a 9-day EMA fast
    leg; Weekly's own SMA shortened from 20 weeks (~5 months) to 8 weeks
    (~2 months) so all three legs are actually relevant to the hold period."""
    try:
        if df is None or len(df) < 60:
            return "", "", "", 0

        price = float(df["Close"].iloc[-1])

        try:
            ema9 = float(df["Close"].ewm(span=9, adjust=False).mean().iloc[-1])
            fastok = price > ema9
        except:
            fastok = False

        try:
            sma20d = float(df["Close"].rolling(20).mean().iloc[-1])
            dailyok = price > sma20d
        except:
            dailyok = False

        weeklyok = None
        try:
            dfw = df["Close"].resample("W-FRI").last().dropna()
            if len(dfw) >= 8:
                sma8w = float(dfw.rolling(8).mean().iloc[-1])
                weeklyok = price > sma8w
        except:
            weeklyok = None

        mtf_fast = "↑" if fastok else "↓"
        mtf_d = "↑" if dailyok else "↓"
        mtf_w = "↑" if weeklyok is True else "↓" if weeklyok is False else "-"
        mtfscore = sum(1 for x in [fastok, dailyok, weeklyok] if x is True)

        return mtf_fast, mtf_d, mtf_w, mtfscore
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
            # In Trending mode the long/short base conditions are the
            # symmetric-mapped R3/S3 ones (see above) -- freshness must
            # track whichever condition actually drives that direction,
            # not always cond_s3_same/cond_r3_same.
            long_fresh = cond_r3_same
            short_fresh = cond_s3_same
        else:
            base_long = range_buy_base
            base_short = range_sell_base
            long_fresh = cond_s3_same
            short_fresh = cond_r3_same

        # long_fresh/short_fresh are purely additional information for the
        # tile UI (same-bar reclaim vs a 2-bar-old one) -- they don't
        # change base_long/base_short's values at all, just expose which
        # sub-condition fired for display.
        return base_long, base_short, long_fresh, short_fresh

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

            base_long, base_short, long_fresh, short_fresh = self.detect_reclaims(df)
            vol_ok, buy_delta, sell_delta = self.volume_confirmation(df)

            current_price = float(latest["Close"])
            prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_price
            vwap = float(latest["VWAP"])
            s1_level = float(latest["S1"])
            s2_level = float(latest["S2"])
            s3_level = float(latest["S3"])
            r1_level = float(latest["R1"])
            r2_level = float(latest["R2"])
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
            mtf_fast, mtf_d, mtf_w, mtf_score = calculatemtfalignment(df)

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

            is_fresh = strength == "Strong" and (
                (signal == "LONG" and long_fresh) or (signal == "SHORT" and short_fresh)
            )

            return {
                "Symbol": symbol.replace(".NS", ""),
                "CurrentPrice": round(current_price, 2),
                "PrevClose": round(prev_close, 2),
                "VWAP": round(vwap, 2),
                "S1Level": round(s1_level, 2),
                "S2Level": round(s2_level, 2),
                "S3Level": round(s3_level, 2),
                "R1Level": round(r1_level, 2),
                "R2Level": round(r2_level, 2),
                "R3Level": round(r3_level, 2),
                "IsFresh": is_fresh,
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
                "MTFFast": mtf_fast,
                "MTFDaily": mtf_d,
                "MTFWeekly": mtf_w,
                "MTFScore": mtf_score,
            }
        except Exception as e:
            print(f"{symbol}: ERROR in generatesignals -> {e}")
            return None

# =====================================================
# STOCK LIST HELPERS
# =====================================================

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
    using IST. Only apply 09:15-15:30 session filter when using intraday base data.
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
        # Base is already 60m from Dhan
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
        return [], [], 0

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
    # total_scanned (LONG + SHORT + HOLD, i.e. every symbol that returned
    # a valid signal at all) is purely additive info for the summary bar
    # -- long_res/short_res's contents are exactly what they were before.
    return long_res, short_res, len(results)

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


def _pb_tile_html(s: dict, name: str, lot, price: float, prev_close: float) -> str:
    """Built as one continuous line (no embedded newlines) rather than
    readable multi-line source -- concatenating many nicely-indented
    tiles is exactly what triggered markdown's "4+ spaces = code block"
    rule building Ribbon Scanner. See project_streamlit_html_patterns."""
    symbol_clean = s["Symbol"]
    signal = s.get("Signal", "")
    strength = s.get("SignalStrength", "")
    direction = "LONG" if signal == "LONG" else ("SHORT" if signal == "SHORT" else "HOLD")
    dir_class = direction.lower()

    if direction in ("LONG", "SHORT"):
        badge_text = f'{direction} &middot; {strength.upper()}' if strength else direction
    else:
        badge_text = f'HOLD &middot; {"BULLISH" if "Bullish" in signal else "BEARISH"}'

    change_amt = price - prev_close
    change_pct = (change_amt / prev_close * 100) if prev_close else 0.0
    if price > prev_close:
        arrow, price_class = "&#9650;", "up"
    elif price < prev_close:
        arrow, price_class = "&#9660;", "down"
    else:
        arrow, price_class = "", ""
    change_html = f'<span class="chg {price_class}">{arrow} {change_amt:+.2f} ({change_pct:+.2f}%)</span>'

    vwap_regime = s.get("VWAPRegime", "Unknown")
    sttrend = s.get("STTrend", "")
    tag_row = f'<span class="pill">Lot {lot}</span><span class="pill">VWAP: {vwap_regime}</span><span class="pill">ST {sttrend}</span>'

    ladder = _vwap_ladder_svg(
        s.get("S3Level", price), s.get("S2Level", price), s.get("S1Level", price), s.get("VWAP", price),
        s.get("R1Level", price), s.get("R2Level", price), s.get("R3Level", price), price, direction,
    )
    rsi_gauge = _pb_rsi_gauge_html(s.get("RSI"))
    adx, dip, din = s.get("ADX", "NA"), s.get("DIP", "NA"), s.get("DIN", "NA")

    tv_url = f"https://www.tradingview.com/chart/RaPnty9s/?symbol=NSE%3A{symbol_clean}"
    symbol_html = f'<a href="{tv_url}" target="_blank">{symbol_clean}</a>'
    price_html = f'<span class="price">&#8377;{price:,.2f}</span>'

    fresh = bool(s.get("IsFresh")) and direction in ("LONG", "SHORT")
    tile_class = "tile fresh" if fresh else "tile"
    mtf_dots = _mtf_dots_html(s.get("MTFFast", ""), s.get("MTFDaily", ""), s.get("MTFWeekly", ""))
    mtfscore = s.get("MTFScore", 0)

    if direction in ("LONG", "SHORT"):
        volstatus, volratio = s.get("VolStatus", "Unknown"), s.get("VolRatio", 1.0)
        obvdiv, obvtype = s.get("OBVDiv", "Unknown"), s.get("OBVDivType", "")
        obv_cls = "long" if obvtype == "bull_div" else ("short" if obvtype == "bear_div" else "")
        entry, sl, risk = s.get("EntryPrice", price), s.get("SLLevel", price), s.get("RiskPct", 0.0)
        stop_color = "var(--short)" if direction == "LONG" else "var(--long)"
        if fresh:
            reclaim_word = "S3 reclaim" if direction == "LONG" else "R3 rejection"
            footer = f'<div class="fresh-note"><span class="ring"></span>{reclaim_word} confirmed this bar</div>'
        else:
            footer = '<div class="stat-row"><span class="k">Signal age</span><span class="v">2 bars ago</span></div>'
        body = (
            f'<div class="stat-row"><span class="k">RSI (14)</span>{rsi_gauge}<span class="v {dir_class}">{s.get("RSI","NA")}</span></div>'
            f'<div class="stat-row"><span class="k">ADX / DI+ / DI-</span><span class="v">{adx}&nbsp;<span class="v long">{dip}</span>&nbsp;<span class="v short">{din}</span></span></div>'
            f'<div class="stat-row"><span class="k">Volume / OBV</span><span><span class="v long">{volstatus} {volratio:.1f}&times;</span>&nbsp;&middot;&nbsp;<span class="v {obv_cls}">{obvdiv}</span></span></div>'
            f'<div class="stat-row"><span class="k">9 / D / W alignment</span>{mtf_dots}<span class="v">{mtfscore}/3</span></div>'
            f'<div class="trade-plan">'
            f'<div><div class="tp-label">Entry</div><div class="tp-value">{entry:,.2f}</div></div>'
            f'<div><div class="tp-label">Stop</div><div class="tp-value" style="color:{stop_color};">{sl:,.2f}</div></div>'
            f'<div><div class="tp-label">Risk</div><div class="tp-value">{risk:.1f}%</div></div>'
            f'</div>'
            f'{footer}'
        )
    else:
        pricevsvwap = s.get("PricevsVWAP", "ABOVE")
        status_text = "Above VWAP, no band reclaim yet" if pricevsvwap == "ABOVE" else "Below VWAP, no band reclaim yet"
        body = (
            f'<div class="stat-row"><span class="k">RSI (14)</span>{rsi_gauge}<span class="v">{s.get("RSI","NA")}</span></div>'
            f'<div class="stat-row"><span class="k">ADX / DI+ / DI-</span><span class="v">{adx}&nbsp;<span class="v long">{dip}</span>&nbsp;<span class="v short">{din}</span></span></div>'
            f'<div class="stat-row"><span class="k">9 / D / W alignment</span>{mtf_dots}<span class="v">{mtfscore}/3</span></div>'
            f'<div class="stat-row"><span class="k">Status</span><span class="v">{status_text}</span></div>'
        )

    return (
        f'<div class="{tile_class}">'
        f'<div class="tile-top">'
        f'<div><div class="symbol">{symbol_html}</div><div class="company">{name}</div></div>'
        f'<span class="dir-badge {dir_class}">{badge_text}</span>'
        f'</div>'
        f'<div class="tag-row">{tag_row}</div>'
        f'<div class="ladder-box">{ladder}</div>'
        f'<div class="price-row">{price_html}{change_html}</div>'
        f'{body}'
        f'</div>'
    )


def rendertiles(stocks, signaltype, fodf, live_prices=None):
    if not stocks:
        st.write("No stocks matched.")
        return
    live_prices = live_prices or {}

    tiles_html = ""
    for s in stocks:
        symbol_clean = s["Symbol"]
        name = getcompanyname(symbol_clean, fodf)
        lot = s.get("Lot", getlotsize(symbol_clean, fodf))
        price = live_prices.get(symbol_clean, s["CurrentPrice"])
        prev_close = s.get("PrevClose", price)
        tiles_html += _pb_tile_html(s, name, lot, price, prev_close)

    # SVG is in these tiles -- st.markdown(unsafe_allow_html=True), not
    # st.html() (which sanitizes <svg> out). tiles_html has no embedded
    # newlines, so this is also safe from the markdown-code-block trap.
    st.markdown(f'<div class="pb-wrap"><div class="grid">{tiles_html}</div></div>', unsafe_allow_html=True)

# =====================================================
# MAIN STREAMLIT APP
# =====================================================

def run_pivotboss_tab():
    st.markdown(_CSS + _HEADER_HTML, unsafe_allow_html=True)

    st.sidebar.header("PivotBoss VWAP Settings")
    # Anchor Timeframe, Chart Timeframe, Band Tolerance, and Max symbols
    # moved to the headline controls row below (the 4 you actually touch
    # often) -- everything else here is unchanged, same sidebar, same
    # defaults, same order.
    anchor_mult = st.sidebar.number_input(
        "Anchor Timeframe Multiplier (N Period)",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
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

    st.sidebar.header("Supertrend Settings")
    st_period = st.sidebar.number_input(
        "ST ATR Period", min_value=5, max_value=30, value=10, step=1
    )
    st_multiplier = st.sidebar.number_input(
        "ST Multiplier", min_value=1.0, max_value=5.0, value=3.0, step=0.1
    )

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            anchor_tf = st.selectbox(
                "Anchor Timeframe (VWAP Period)",
                ["3 Months", "1 Day", "1 Week", "1 Month", "6 Months", "9 Months", "12 Months"],
                index=0,
            )
        with col2:
            band_tolerance_pct = st.slider(
                "Band Tolerance (%) for Range", min_value=0.0, max_value=3.0, value=0.5, step=0.25,
                help="Tolerance around R/S bands when checking RANGE reclaims",
            )
        with col3:
            max_symbols = st.slider("Max symbols to scan", 10, 250, 250, 10)
        with col4:
            chart_tf = st.selectbox("Chart Timeframe", ["1h", "4h", "1D", "1W", "1M"], index=2)

    refresh_col, run_col = st.columns(2)
    with refresh_col:
        if st.button(" Refresh Data Cache", use_container_width=True):
            fetch_all_ohlcv_dhan.clear()
            batch_scan_pivotboss.clear()
            st.session_state.pop("pivotboss_results", None)

    fo_df = process_fo_stock_list()
    if fo_df.empty:
        return

    fo_df = fo_df.iloc[:max_symbols].copy()
    ticker_list = list(fo_df["symbol"])

    if chart_tf in ["1h", "4h"]:
        base_interval = "60m"
    else:
        base_interval = "1d"

    data_dict = fetch_all_ohlcv_dhan(ticker_list, base_interval)

    with run_col:
        run = st.button("Run PivotBoss Scan", use_container_width=True)
    if run:
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
        st.session_state["pivotboss_results"] = batch_scan_pivotboss(data_dict, fo_df, config)
        st.session_state["pivotboss_signal_type"] = signal_type
        st.session_state["pivotboss_chart_tf"] = chart_tf

    # Persisted across tab switches -- Streamlit reruns the whole script on
    # every interaction (including changing which strategy tab is selected
    # in the sidebar), and a button's "clicked" state is only True on the
    # exact rerun it was clicked. Without storing the result in
    # session_state, navigating away and back would silently drop the
    # tiles until "Run PivotBoss Scan" was clicked again.
    results = st.session_state.get("pivotboss_results")
    if results is None:
        st.info("Click 'Run PivotBoss Scan' to scan for signals.")
        return
    longsignals, shortsignals, total_scanned = results
    hold_count = max(total_scanned - len(longsignals) - len(shortsignals), 0)

    candidate_symbols = tuple(s["Symbol"] for s in longsignals + shortsignals)
    live_prices = fetch_live_prices(candidate_symbols)

    all_signals = longsignals + shortsignals
    st.markdown(
        f'<div class="pb-wrap"><div class="summary">'
        f'<div class="step"><b>{len(fo_df):,}</b><span class="lbl">F&amp;O universe scanned</span></div>'
        f'<div class="step"><b class="long-tag">{len(longsignals)} Long</b>&nbsp;&middot;&nbsp;'
        f'<b class="short-tag">{len(shortsignals)} Short</b>&nbsp;&middot;&nbsp;'
        f'<b class="hold-tag">{hold_count} Hold</b></div>'
        f'<div class="step"><span class="lbl">{st.session_state.get("pivotboss_chart_tf","1D")} chart &middot; '
        f'{st.session_state.get("pivotboss_signal_type","Range")} mode</span></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="pb-wrap"><p class="grid-label">Confirmed signals</p></div>', unsafe_allow_html=True)
    rendertiles(all_signals, "ALL", fo_df, live_prices)

if __name__ == "__main__":
    run_pivotboss_tab()

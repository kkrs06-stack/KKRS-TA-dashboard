"""
Ribbon Scanner signal engine.

Flags a stock LONG when its EMA ribbon is in strict bullish order RIGHT NOW,
price crossed above EMA20 within the last 2 bars ("current or previous"
period), RSI(14) sits in a momentum band (not just above a floor -- also
capped, so it excludes already-overbought moves), the crossover bar itself
saw above-average volume, and the stock is outperforming Nifty over the
same lookback. SHORT is the mirror image.

The ribbon is EMA20/50/100 on Daily and Weekly. On MONTHLY it's EMA20/50
only -- EMA100 there would need ~100 months (8+ years) of history just to
populate, which made a Monthly scan pull a decade of daily OHLCV per
symbol across the whole universe. Dropped by explicit choice once that
cost became visible in practice, not because EMA100 stopped mattering.

Deliberately excludes EMA200 everywhere (dropped per the original spec) --
the 20/50(/100) ribbon alone already captures "short/medium/long template"
the way the original Chartink-style scan intended.

All conditions are required together, matching the strictness style
already used in exhaustion_engine.py's multi-condition gate -- a false
positive here is a stock flagged as freshly trending that isn't.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_CONFIG = {
    "crossover_lookback_bars": 2,   # "just crossed... in the current [bar] or previous [bar]"
    "rsi_period": 14,
    "rsi_long_band": (55.0, 75.0),
    "rsi_short_band": (25.0, 45.0),
    "volume_multiplier": 1.2,
    "rs_lookback_bars": 6,           # overridden per-timeframe by the caller -- see below
    "sparkline_bars": 6,
}

# "M" -> "ME" fallback needed because pandas 2.2+ deprecated the "M" alias
# (removed in 3.0); older installs only understand "M". Same approach as
# portfolio_engine.py's _resample, kept independent so this file has no
# import-time dependency on portfolio_engine.
TIMEFRAME_RESAMPLE_RULE = {"Daily": None, "Weekly": "W-FRI", "Monthly": "M"}

# (fast, mid[, slow]) -- Monthly drops the 100-period leg. See module
# docstring for why.
RIBBON_EMA_PERIODS_BY_TIMEFRAME = {
    "Daily": (20, 50, 100),
    "Weekly": (20, 50, 100),
    "Monthly": (20, 50),
}

# A "20-bar" volume average means something very different on Monthly
# (20 months) than Daily (20 sessions) -- a 20-month trailing average is
# not a convention anyone actually uses for a volume-confirmation check.
# Scaled per timeframe to stay a sensible, comparably "recent" window.
VOLUME_LOOKBACK_BARS_BY_TIMEFRAME = {"Daily": 20, "Weekly": 10, "Monthly": 6}

# A "6-bar" relative-strength lookback means something very different on
# Monthly (6 months) vs Daily (6 days) -- these give each timeframe a
# comparable, genuinely medium-term window instead of reusing one raw bar
# count across all three.
RS_LOOKBACK_BARS_BY_TIMEFRAME = {"Daily": 63, "Weekly": 12, "Monthly": 6}

# Fetch windows sized to the longest EMA actually used at that timeframe
# (see RIBBON_EMA_PERIODS_BY_TIMEFRAME) plus that timeframe's volume
# lookback and a small margin -- not padded further than that, since the
# whole point of dropping Monthly's EMA100 was to cut this down.
DAILY_LOOKBACK_DAYS_BY_TIMEFRAME = {"Daily": 400, "Weekly": 1100, "Monthly": 2200}

_OHLCV_AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def _resample(daily_df: pd.DataFrame, rule: str) -> pd.DataFrame:
    try:
        return daily_df.resample("ME" if rule == "M" else rule).agg(_OHLCV_AGG).dropna()
    except ValueError:
        return daily_df.resample(rule).agg(_OHLCV_AGG).dropna()


def to_timeframe(daily_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """daily_df -> bars at the requested timeframe. "Daily" is a passthrough."""
    rule = TIMEFRAME_RESAMPLE_RULE[timeframe]
    return daily_df if rule is None else _resample(daily_df, rule)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Same plain-rolling-mean RSI as exhaustion_engine.py, for consistency
    across the codebase's scanners rather than introducing Wilder smoothing
    in just this one file."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# Every rejection point in the gate, in the order they're checked -- used
# by evaluate_ribbon_diagnostic() to report exactly where a symbol dropped
# out, instead of everything below "0 pass full scan" being a black box.
REJECTION_STAGES = (
    "insufficient_history",
    "no_ribbon_order",
    "no_recent_crossover",
    "rsi_out_of_band",
    "volume_not_confirmed",
    "insufficient_rs_history",
    "wrong_relative_strength",
)


def evaluate_ribbon_diagnostic(
    daily_df: pd.DataFrame,
    nifty_daily_df: pd.DataFrame,
    timeframe: str,
    config: dict | None = None,
) -> tuple[dict | None, str]:
    """
    Same evaluation as evaluate_ribbon(), but always returns
    (result_or_None, stage) where `stage` is "passed" on a signal, or one
    of REJECTION_STAGES naming exactly which gate the symbol failed --
    for diagnosing a suspiciously-empty scan (is it genuinely 0 qualifying
    stocks, or are all/most of them dying at the same specific gate,
    which would point at a bug in that gate rather than market reality).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if daily_df is None or len(daily_df) < 3:
        return None, "insufficient_history"

    ema_periods = RIBBON_EMA_PERIODS_BY_TIMEFRAME[timeframe]
    has_slow = len(ema_periods) == 3
    vol_lookback = VOLUME_LOOKBACK_BARS_BY_TIMEFRAME[timeframe]
    min_bars_needed = max(ema_periods) + vol_lookback + cfg["crossover_lookback_bars"] + 5

    tf_df = to_timeframe(daily_df, timeframe)
    if len(tf_df) < min_bars_needed:
        return None, "insufficient_history"

    close, volume = tf_df["Close"], tf_df["Volume"]
    ema_fast = close.ewm(span=ema_periods[0], adjust=False).mean()
    ema_mid = close.ewm(span=ema_periods[1], adjust=False).mean()
    ema_slow = close.ewm(span=ema_periods[2], adjust=False).mean() if has_slow else None
    rsi = _rsi(close, cfg["rsi_period"])

    if has_slow:
        long_ribbon_now = bool((ema_fast.iloc[-1] > ema_mid.iloc[-1]) and (ema_mid.iloc[-1] > ema_slow.iloc[-1]))
        short_ribbon_now = bool((ema_fast.iloc[-1] < ema_mid.iloc[-1]) and (ema_mid.iloc[-1] < ema_slow.iloc[-1]))
    else:
        long_ribbon_now = bool(ema_fast.iloc[-1] > ema_mid.iloc[-1])
        short_ribbon_now = bool(ema_fast.iloc[-1] < ema_mid.iloc[-1])

    if not long_ribbon_now and not short_ribbon_now:
        return None, "no_ribbon_order"
    direction = "LONG" if long_ribbon_now else "SHORT"

    # Crossover: price closed on the correct side of EMA20 for the first
    # time within the lookback window (2 bars = "current or previous").
    crossed_above = (close > ema_fast) & (close.shift(1) <= ema_fast.shift(1))
    crossed_below = (close < ema_fast) & (close.shift(1) >= ema_fast.shift(1))
    cross_series = crossed_above if direction == "LONG" else crossed_below

    lookback = cfg["crossover_lookback_bars"]
    recent_cross = list(cross_series.tail(lookback))
    cross_hits = [i for i, hit in enumerate(recent_cross) if hit]
    if not cross_hits:
        return None, "no_recent_crossover"
    bars_since_crossover = (lookback - 1) - max(cross_hits)  # 0 = this bar, 1 = previous bar
    crossover_idx = len(tf_df) - 1 - bars_since_crossover

    rsi_now = rsi.iloc[-1]
    if pd.isna(rsi_now):
        return None, "rsi_out_of_band"
    rsi_lo, rsi_hi = cfg["rsi_long_band"] if direction == "LONG" else cfg["rsi_short_band"]
    if not (rsi_lo <= rsi_now <= rsi_hi):
        return None, "rsi_out_of_band"

    if crossover_idx < vol_lookback:
        return None, "volume_not_confirmed"
    avg_vol = volume.iloc[crossover_idx - vol_lookback:crossover_idx].mean()
    crossover_vol = volume.iloc[crossover_idx]
    if pd.isna(avg_vol) or avg_vol <= 0:
        return None, "volume_not_confirmed"
    vol_ratio = crossover_vol / avg_vol
    if vol_ratio < cfg["volume_multiplier"]:
        return None, "volume_not_confirmed"

    rs_bars = RS_LOOKBACK_BARS_BY_TIMEFRAME.get(timeframe, cfg["rs_lookback_bars"])
    nifty_tf_df = to_timeframe(nifty_daily_df, timeframe)
    if len(tf_df) <= rs_bars or len(nifty_tf_df) <= rs_bars:
        return None, "insufficient_rs_history"
    stock_return_pct = (close.iloc[-1] / close.iloc[-1 - rs_bars] - 1) * 100
    nifty_close = nifty_tf_df["Close"]
    nifty_return_pct = (nifty_close.iloc[-1] / nifty_close.iloc[-1 - rs_bars] - 1) * 100
    relative_strength_pct = stock_return_pct - nifty_return_pct
    if direction == "LONG" and relative_strength_pct <= 0:
        return None, "wrong_relative_strength"
    if direction == "SHORT" and relative_strength_pct >= 0:
        return None, "wrong_relative_strength"

    spark = cfg["sparkline_bars"]
    daily_close = daily_df["Close"]

    result = {
        "Direction": direction,
        "Timeframe": timeframe,
        "EMA20": round(float(ema_fast.iloc[-1]), 2),
        "EMA50": round(float(ema_mid.iloc[-1]), 2),
        "EMA20Series": [round(float(v), 2) for v in ema_fast.tail(spark)],
        "EMA50Series": [round(float(v), 2) for v in ema_mid.tail(spark)],
        "RSI": round(float(rsi_now), 2),
        "VolumeRatio": round(float(vol_ratio), 2),
        "VolumeConfirmed": True,
        "RelativeStrengthPct": round(float(relative_strength_pct), 2),
        "BarsSinceCrossover": int(bars_since_crossover),
        "LastClose": round(float(daily_close.iloc[-1]), 2),
        "PrevClose": round(float(daily_close.iloc[-2]), 2) if len(daily_close) >= 2 else round(float(daily_close.iloc[-1]), 2),
    }
    if has_slow:
        result["EMA100"] = round(float(ema_slow.iloc[-1]), 2)
        result["EMA100Series"] = [round(float(v), 2) for v in ema_slow.tail(spark)]
    return result, "passed"


def evaluate_ribbon(
    daily_df: pd.DataFrame,
    nifty_daily_df: pd.DataFrame,
    timeframe: str,
    config: dict | None = None,
) -> dict | None:
    """
    daily_df / nifty_daily_df: OHLCV DataFrames of DAILY bars (already
    cleaned -- same contract as fetch_all_ohlcv_dhan's output), indexed by
    date. Resampling to the requested timeframe happens inside this
    function so callers always fetch/cache at daily granularity.

    Returns a result dict if every condition holds, else None. The result
    only carries "EMA100"/"EMA100Series" keys on Daily/Weekly -- Monthly's
    dict omits them entirely (not None-filled), so callers/renderers can
    tell "no third leg at this timeframe" apart from "missing data".

    Thin wrapper over evaluate_ribbon_diagnostic() -- see that function if
    you need to know WHY a symbol didn't pass, not just that it didn't.
    """
    result, _stage = evaluate_ribbon_diagnostic(daily_df, nifty_daily_df, timeframe, config)
    return result
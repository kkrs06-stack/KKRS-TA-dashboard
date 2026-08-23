"""
Parabolic Exhaustion signal engine.

Detects stocks that have made a large, ACCELERATING move over the last
~20-24 sessions and are showing the first signs of stalling -- candidates
for selling OTM options against the move (calls after an up-move,
puts after a down-move), betting on consolidation/mean-reversion rather
than continuation.

Five conditions, all required together (deliberately strict -- a false
positive here means selling options into a move that's still running):
  1. Magnitude: % change over the lookback window exceeds a threshold.
  2. Acceleration: the second half of the move is bigger than the first
     half (steepening, not a steady grind).
  3. Stretched from its own 9-session mean.
  4. Stretched from its own 20-session mean (both required -- confirms
     extension on both a short and medium-term basis).
  5. RSI(14) at an extreme.
  6. Same-bar exhaustion signature: today still pushes to a fresh
     lookback-window extreme (so it's still extended, RSI still elevated)
     but CLOSES weak relative to its own day's range -- a same-day
     rejection candle, not "the peak was a few days ago." (An earlier
     version required the peak to be 1-3 bars in the past, but that
     directly contradicts "still extended/overbought today" -- by the
     time price has pulled back for a few days, RSI and stretch-from-SMA
     have already decayed too. Evaluating everything on the same bar
     avoids that self-contradiction.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TICK_SIZE = 0.05

DEFAULT_CONFIG = {
    "lookback_sessions": 22,       # 20-24 range
    "min_move_pct": 8.0,
    "sma_short_period": 9,
    "sma_long_period": 20,
    "stretch_short_pct": 4.5,
    "stretch_long_pct": 9.0,
    "acceleration_tolerance": 0.75,  # second half must be >= 75% of first half's move (allows some deceleration, not strict steepening)
    "rsi_period": 14,
    "rsi_overbought": 70.0,
    "rsi_oversold": 30.0,
    "weak_close_threshold": 0.35,  # close in bottom/top 35% of the day's range
    # Both OFF by default -- empirically, Stretch9 fades during genuinely
    # sustained multi-week exhaustion (a 9-day average catches up to price
    # almost as fast as price rises), and the two-half Accelerating check
    # is brittle around the exact halfway boundary. Still computed and
    # shown as context either way; just not required to gate a signal
    # unless explicitly turned on.
    "require_stretch_short": False,
    "require_acceleration": False,
}


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _bars_since_extreme(series: pd.Series, window: int, mode: str) -> pd.Series:
    """For each row, how many bars ago (0 = today) the max/min occurred
    within the trailing `window` bars."""
    def _calc(arr):
        idx = np.argmax(arr) if mode == "max" else np.argmin(arr)
        return len(arr) - 1 - idx
    return series.rolling(window).apply(_calc, raw=True)


def compute_exhaustion_signals(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    df: DataFrame with Open/High/Low/Close/Volume, indexed by bar timestamp.
    Returns df with all intermediate columns plus BearishExhaustion (up-move,
    sell-calls candidate) and BullishExhaustion (down-move, sell-puts
    candidate) boolean columns.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    df = df.copy()

    close, high, low = df["Close"], df["High"], df["Low"]
    n = cfg["lookback_sessions"]
    half = n // 2

    pct_change_full = (close - close.shift(n)) / close.shift(n) * 100
    pct_change_first_half = (close.shift(half) - close.shift(n)) / close.shift(n) * 100
    pct_change_second_half = (close - close.shift(half)) / close.shift(half) * 100

    sma_short = close.rolling(cfg["sma_short_period"]).mean()
    sma_long = close.rolling(cfg["sma_long_period"]).mean()
    stretch_short_pct = (close - sma_short) / sma_short * 100
    stretch_long_pct = (close - sma_long) / sma_long * 100

    rsi = _rsi(close, cfg["rsi_period"])

    bars_since_high = _bars_since_extreme(high, n, "max")
    bars_since_low = _bars_since_extreme(low, n, "min")

    df["PctChangeFull"] = pct_change_full
    df["StretchShortPct"] = stretch_short_pct
    df["StretchLongPct"] = stretch_long_pct
    df["RSI"] = rsi
    df["BarsSinceHigh"] = bars_since_high
    df["BarsSinceLow"] = bars_since_low

    # "Accelerating" allows some deceleration (tolerance < 1.0) rather than
    # requiring strict day-to-day steepening -- a rigid two-half split is
    # sensitive to exactly where the halfway boundary falls, which made
    # this flip noisily day-to-day in testing even during genuine
    # multi-week parabolic moves.
    tol = cfg["acceleration_tolerance"]
    accelerating_up = (pct_change_first_half > 0) & (pct_change_second_half > 0) & (pct_change_second_half >= pct_change_first_half * tol)
    accelerating_down = (pct_change_first_half < 0) & (pct_change_second_half < 0) & (pct_change_second_half <= pct_change_first_half * tol)
    df["Accelerating"] = accelerating_up | accelerating_down  # informational either way

    bar_range = (high - low).clip(lower=TICK_SIZE)
    close_position = (close - low) / bar_range  # 0 = at low, 1 = at high
    df["ClosePosition"] = close_position

    is_new_high_today = bars_since_high == 0
    is_new_low_today = bars_since_low == 0
    weak_close = close_position <= cfg["weak_close_threshold"]
    strong_close = close_position >= (1 - cfg["weak_close_threshold"])

    bearish_exhaustion_bar = is_new_high_today & weak_close
    bullish_exhaustion_bar = is_new_low_today & strong_close

    # Both gated by config flags -- OFF by default (see DEFAULT_CONFIG note).
    # When off, the condition is a pass-through (always True) rather than
    # being removed from the code, so it can be re-enabled without editing
    # this function.
    true_series = pd.Series(True, index=df.index)
    stretch_short_up_gate = (stretch_short_pct >= cfg["stretch_short_pct"]) if cfg["require_stretch_short"] else true_series
    stretch_short_down_gate = (stretch_short_pct <= -cfg["stretch_short_pct"]) if cfg["require_stretch_short"] else true_series
    accel_up_gate = accelerating_up if cfg["require_acceleration"] else true_series
    accel_down_gate = accelerating_down if cfg["require_acceleration"] else true_series

    df["BearishExhaustion"] = (
        (pct_change_full >= cfg["min_move_pct"])
        & accel_up_gate
        & stretch_short_up_gate
        & (stretch_long_pct >= cfg["stretch_long_pct"])
        & (rsi >= cfg["rsi_overbought"])
        & bearish_exhaustion_bar
    ).fillna(False)

    df["BullishExhaustion"] = (
        (pct_change_full <= -cfg["min_move_pct"])
        & accel_down_gate
        & stretch_short_down_gate
        & (stretch_long_pct <= -cfg["stretch_long_pct"])
        & (rsi <= cfg["rsi_oversold"])
        & bullish_exhaustion_bar
    ).fillna(False)

    return df
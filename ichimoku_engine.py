"""
Ichimoku Cloud long/short signal engine.

Unlike CPR, Ichimoku runs directly on the chart's own timeframe -- there's
no separate "anchor timeframe" concept. Tenkan/Kijun/Span calculations all
use the chart's own bars throughout.

Key subtlety (see conversation notes -- this is the single most commonly
misimplemented part of Ichimoku): Senkou Span A/B are computed from
TODAY's data but PLOTTED `displacement` bars in the FUTURE. So the cloud
actually visible at today's chart position is the raw Span A/B value
computed `displacement` bars ago -- not today's raw value (that raw value
is the "future cloud", not yet drawn).

The Chikou (lagging) span is today's close, plotted `displacement` bars
BACKWARD. The cloud "behind" it, visually, is therefore the raw Span A/B
value from `displacement * 2` bars ago (one displacement to reach the
Chikou's plotted position, another for that position's own displayed
cloud). config["lagging_compare_mode"] lets you choose:
  - "true"   (default, recommended): the geometrically accurate 2x-shift
             comparison -- matches what you'd actually see on a real chart.
  - "simple": a 1x-shift proxy some retail scanners use instead. Faster to
             reason about, but doesn't correspond to anything literally
             drawn on the chart.

Signal logic (LONG; SHORT is the exact mirror):
  Always required:
    1. Future cloud is green (Span A > Span B, raw/unshifted).
    5. Tenkan > Kijun.
  Plus at least one of three mutually exclusive breakout scenarios:
    2. Daily price AND lagging line both cross above their respective
       clouds fresh, on the same bar (a "double cross").
    3. Daily price is already above its cloud (crossed earlier); lagging
       line is the one crossing fresh today.
    4. Lagging line is already above its cloud (crossed earlier); daily
       price is the one crossing fresh today.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CONFIG = {
    "tenkan_period": 9,
    "kijun_period": 26,
    "senkou_b_period": 52,
    "displacement": 26,
    "lagging_compare_mode": "true",  # "true" (2x shift, matches real chart) | "simple" (1x shift proxy)
}


def compute_ichimoku_signals(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    df: DataFrame with Open/High/Low/Close/Volume, indexed by bar timestamp,
        already at whatever chart timeframe you want Ichimoku computed on.
    Returns df with Ichimoku lines, cloud levels, LongSignal/ShortSignal,
    and a human-readable TriggerType column added.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    df = df.copy()

    tenkan_p = cfg["tenkan_period"]
    kijun_p = cfg["kijun_period"]
    senkou_b_p = cfg["senkou_b_period"]
    disp = cfg["displacement"]

    high, low, close = df["High"], df["Low"], df["Close"]

    tenkan = (high.rolling(tenkan_p).max() + low.rolling(tenkan_p).min()) / 2
    kijun = (high.rolling(kijun_p).max() + low.rolling(kijun_p).min()) / 2

    # Raw (unshifted) Span A/B computed from today's data -- the "future
    # cloud" that will be plotted `disp` bars ahead, not yet on the chart.
    span_a_raw = (tenkan + kijun) / 2
    span_b_raw = (high.rolling(senkou_b_p).max() + low.rolling(senkou_b_p).min()) / 2
    raw_cloud_top = pd.concat([span_a_raw, span_b_raw], axis=1).max(axis=1)
    raw_cloud_bottom = pd.concat([span_a_raw, span_b_raw], axis=1).min(axis=1)

    df["Tenkan"] = tenkan
    df["Kijun"] = kijun
    df["FutureSpanA"] = span_a_raw
    df["FutureSpanB"] = span_b_raw
    df["FutureCloudGreen"] = span_a_raw > span_b_raw

    # The cloud actually visible at today's chart position = raw cloud
    # computed `disp` bars ago.
    cloud_top = raw_cloud_top.shift(disp)
    cloud_bottom = raw_cloud_bottom.shift(disp)
    df["CloudTop"] = cloud_top
    df["CloudBottom"] = cloud_bottom

    prev_close = close.shift(1)
    prev_cloud_top = cloud_top.shift(1)
    prev_cloud_bottom = cloud_bottom.shift(1)

    daily_cross_above = (close > cloud_top) & (prev_close <= prev_cloud_top)
    daily_cross_below = (close < cloud_bottom) & (prev_close >= prev_cloud_bottom)
    daily_already_above = (close > cloud_top) & ~daily_cross_above
    daily_already_below = (close < cloud_bottom) & ~daily_cross_below

    df["DailyCrossAbove"] = daily_cross_above
    df["DailyCrossBelow"] = daily_cross_below
    df["DailyAlreadyAbove"] = daily_already_above
    df["DailyAlreadyBelow"] = daily_already_below

    # Lagging (Chikou) reference cloud.
    lag_shift = disp * 2 if cfg["lagging_compare_mode"] == "true" else disp
    cloud_top_lag = raw_cloud_top.shift(lag_shift)
    cloud_bottom_lag = raw_cloud_bottom.shift(lag_shift)
    prev_cloud_top_lag = cloud_top_lag.shift(1)
    prev_cloud_bottom_lag = cloud_bottom_lag.shift(1)

    lagging_cross_above = (close > cloud_top_lag) & (prev_close <= prev_cloud_top_lag)
    lagging_cross_below = (close < cloud_bottom_lag) & (prev_close >= prev_cloud_bottom_lag)
    lagging_already_above = (close > cloud_top_lag) & ~lagging_cross_above
    lagging_already_below = (close < cloud_bottom_lag) & ~lagging_cross_below

    df["LaggingCrossAbove"] = lagging_cross_above
    df["LaggingCrossBelow"] = lagging_cross_below
    df["LaggingAlreadyAbove"] = lagging_already_above
    df["LaggingAlreadyBelow"] = lagging_already_below

    df["TenkanAboveKijun"] = tenkan > kijun
    df["TenkanBelowKijun"] = tenkan < kijun

    # Three mutually exclusive breakout scenarios, long side:
    cond2_long = daily_cross_above & lagging_cross_above    # double cross, same bar
    cond3_long = daily_already_above & lagging_cross_above  # daily established, lagging confirms
    cond4_long = lagging_already_above & daily_cross_above  # lagging established, daily confirms

    cond2_short = daily_cross_below & lagging_cross_below
    cond3_short = daily_already_below & lagging_cross_below
    cond4_short = lagging_already_below & daily_cross_below

    # Guard against insufficient warmup history: rolling windows (Senkou B
    # needs `senkou_b_period` bars, the lagging reference needs `lag_shift`
    # bars of prior cloud history) are genuinely undefined (NaN) near the
    # start of the fetched data. Pandas comparisons against NaN silently
    # resolve to False rather than staying undefined -- which would let
    # e.g. FutureCloudGreen read as "red" (satisfying a SHORT) when the
    # true state is "not yet computable", not actually red. Explicitly
    # exclude any row where the core components aren't fully formed yet.
    has_sufficient_history = (
        tenkan.notna() & kijun.notna() & span_a_raw.notna() & span_b_raw.notna()
        & cloud_top.notna() & cloud_bottom.notna()
        & cloud_top_lag.notna() & cloud_bottom_lag.notna()
    )

    df["LongSignal"] = (
        df["FutureCloudGreen"] & df["TenkanAboveKijun"] & (cond2_long | cond3_long | cond4_long)
        & has_sufficient_history
    ).fillna(False)
    df["ShortSignal"] = (
        (~df["FutureCloudGreen"]) & df["TenkanBelowKijun"] & (cond2_short | cond3_short | cond4_short)
        & has_sufficient_history
    ).fillna(False)

    df["TriggerType"] = np.select(
        [cond2_long | cond2_short, cond3_long | cond3_short, cond4_long | cond4_short],
        ["Double Cross (Daily+Lagging)", "Daily Established, Lagging Confirms", "Lagging Established, Daily Confirms"],
        default="",
    )

    return df
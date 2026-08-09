"""
PivotBoss CPR PRO - vectorized pandas port of the ThinkScript indicator.

This mirrors the original script's structure and variable names as closely
as possible (section headers match), so it can be audited line-by-line
against the ThinkScript source. Two structural differences from the TOS
version, both purely mechanical (same math, cleaner implementation):

1. TOS's `rec`-based shift registers (w0..w19, overlapStreak) exist only
   because ThinkScript has no native "resample then rolling-mean" concept.
   Here, CPR levels/width/overlap are computed once per CPR PERIOD (via
   groupby + shift + rolling), then broadcast onto every chart-timeframe
   bar within that period. Mathematically identical, no shift registers
   needed.
2. `close(period=X)[1]` (TOS's "prior period's close") is implemented as
   a period-level shift; `close[1]` (TOS's "prior chart bar's close") is
   a bar-level shift. Both appear in the original script and are kept
   distinct here.

Config keys and defaults match the user's actual confirmed indicator
settings (verified against screenshots), not just the ThinkScript's
hardcoded defaults -- though for this indicator the two are identical
except CPR/MTF timeframe selection.

KNOWN GAPS vs. the real TradingView PivotBoss CPR PRO indicator
(this ThinkScript is itself an AI-assisted "v1.5 BETA" port of the true
Pine Script, per its own header -- these were found by spot-checking
RELIANCE bars against the live TradingView indicator on 2026-08-09/10;
no access to the true Pine Script source was available to resolve them):

1. A narrow/"Inside Value" CPR that price has already broken cleanly
   above/below has no matching path in the buyOpportunityPermission /
   sellOpportunityPermission OR-chain (compressionCoilOpportunity
   specifically requires price to still be INSIDE the CPR). Verified
   against the ThinkScript text itself, not just this port -- confirmed
   missed an A+ Buy the real indicator showed (RELIANCE, 2026-07-17 09:15,
   OHLC matched TradingView almost exactly).
2. At least one bar (RELIANCE, 2026-06-24 15:00, OHLC exact match against
   TradingView) scored 6/10 here -- below the aPlusScoreThreshold of 8 --
   suggesting the real indicator's scoring awards points this port doesn't,
   for a "CPR migrated to Lower Value, price now resting inside it, choppy"
   context.
3. At least one bar (RELIANCE, 2026-07-15 15:00, OHLC ~matched) scored a
   maximal 10/10 and fired A+ Sell confidently in this port, but showed no
   signal on the real indicator -- root cause not isolated; likely the
   prior day's H/L/C (which the whole day's CPR/MID is built from) differs
   between Dhan and TradingView's feed, but this wasn't confirmed.

Net effect: treat this scanner as a solid early-screening approximation,
not a 1:1 mirror of the real TradingView indicator. Both false negatives
(gap 1) and possible false positives (gap 3) are known to occur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TICK_SIZE = 0.05  # NSE equity minimum tick, used only as a safety floor for divisions.

DEFAULT_CONFIG = {
    "cpr_tf": "1 Day",
    "cpr_mult": 1,
    "mtf_tf": "1 Week",
    "mtf_mult": 1,
    "mtfMode": "Soft",  # Off | Soft | Strict

    "cprWidthLookback": 10,
    "narrowCPRThreshold": 0.65,
    "wideCPRThreshold": 1.15,
    "veryWideCPRThreshold": 1.50,
    "trendEMALength": 50,
    "flatEMAThresholdPct": 0.03,

    "enableOpportunityEngine": True,
    "weakStateShiftMaxPct": 10.0,
    "moderateStateShiftMaxPct": 20.0,
    "strongStateShiftMaxPct": 40.0,
    "multiOverlapThreshold": 2,
    "compressionCoilThreshold": 3,
    "allowStateBreakSignalsThroughStandDown": True,
    "prioritizeMajorMIDTrendBreaks": True,
    "allowResponsiveSignalsInWideCPR": True,
    "enableWideCPRReversionOverride": True,

    "minimumSignalScore": 6,
    "aPlusScoreThreshold": 8,
    "blockSignalsInChop": True,
    "chopScoreThreshold": 3,
    "overlapGoNoGoThresholdPct": 50.0,

    "enableBalancedRangeFilter": True,
    "balancedCloseThresholdPct": 10.0,
    "signalBarAvgRangeLength": 10,
    "signalRangeMultiplier": 1.25,

    "rejectionCloseThresholdPct": 35.0,
    "proLookback": 50,
    "volumeMultiplier": 1.2,
}


# =====================================================
# Period-key construction (mirrors PivotBoss's anchor-key logic)
# =====================================================

def _make_period_key(index: pd.DatetimeIndex, period: str, mult: int = 1) -> pd.Series:
    mult = max(int(mult), 1)

    if period == "1 Day":
        days = pd.Index(index.date)
        codes, _ = pd.factorize(days, sort=True)
        return pd.Series(codes // mult, index=index)

    # Strip tz before .to_period() -- IST has no DST, so this is a no-op
    # numerically, it just silences pandas's "will drop timezone" warning.
    naive_index = index.tz_localize(None) if index.tz is not None else index

    if period == "1 Week":
        weeks = naive_index.to_period("W-MON")
        codes, _ = pd.factorize(weeks, sort=True)
        return pd.Series(codes // mult, index=index)

    if period == "1 Month":
        months = naive_index.to_period("M")
        codes, _ = pd.factorize(months, sort=True)
        return pd.Series(codes // mult, index=index)

    if period == "3 Months":
        q = naive_index.to_period("Q")
        codes, _ = pd.factorize(q, sort=True)
        return pd.Series(codes // mult, index=index)

    if period == "6 Months":
        q = naive_index.to_period("Q")
        codes, _ = pd.factorize(q, sort=True)
        return pd.Series((codes // 2) // mult, index=index)

    if period == "9 Months":
        q = naive_index.to_period("Q")
        codes, _ = pd.factorize(q, sort=True)
        return pd.Series((codes // 3) // mult, index=index)

    if period == "12 Months":
        years = naive_index.to_period("A")
        codes, _ = pd.factorize(years, sort=True)
        return pd.Series(codes // mult, index=index)

    days = pd.Index(index.date)
    codes, _ = pd.factorize(days, sort=True)
    return pd.Series(codes // mult, index=index)


def _aggregate_period(df: pd.DataFrame, period_key: pd.Series) -> pd.DataFrame:
    grouped = df.groupby(period_key)
    agg = pd.DataFrame({
        "Open": grouped["Open"].first(),
        "High": grouped["High"].max(),
        "Low": grouped["Low"].min(),
        "Close": grouped["Close"].last(),
        "Volume": grouped["Volume"].sum(),
    })
    return agg.sort_index()


# =====================================================
# Period-level CPR math, Market State, Width, Overlap
# (mirrors "CPR Math" / "Period Change" / "Market State Engine" /
#  "Overlap Logic" sections of the ThinkScript)
# =====================================================

def _compute_period_level(period_frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    pf = period_frame.copy()

    h1 = pf["High"].shift(1)
    l1 = pf["Low"].shift(1)
    c1 = pf["Close"].shift(1)
    h2 = pf["High"].shift(2)
    l2 = pf["Low"].shift(2)
    c2 = pf["Close"].shift(2)
    pf["h1"], pf["l1"], pf["c1"] = h1, l1, c1

    PP = (h1 + l1 + c1) / 3
    MID = (h1 + l1) / 2
    PR = (PP - MID) + PP
    CPRHigh = np.maximum(MID, PR)
    CPRLow = np.minimum(MID, PR)
    CPRWidth = CPRHigh - CPRLow
    pf["PP"], pf["MID"], pf["PR"] = PP, MID, PR
    pf["CPRHigh"], pf["CPRLow"], pf["CPRWidth"] = CPRHigh, CPRLow, CPRWidth

    PriorPP = (h2 + l2 + c2) / 3
    PriorMID = (h2 + l2) / 2
    PriorPR = (PriorPP - PriorMID) + PriorPP
    PriorCPRHigh = np.maximum(PriorMID, PriorPR)
    PriorCPRLow = np.minimum(PriorMID, PriorPR)
    pf["PriorCPRHigh"], pf["PriorCPRLow"] = PriorCPRHigh, PriorCPRLow

    higherCPR = CPRLow > PriorCPRHigh
    lowerCPR = CPRHigh < PriorCPRLow
    insideCPR = (CPRHigh < PriorCPRHigh) & (CPRLow > PriorCPRLow)
    outsideCPR = (CPRHigh > PriorCPRHigh) & (CPRLow < PriorCPRLow)
    overlapCPR = ~(higherCPR | lowerCPR | insideCPR | outsideCPR)
    pf["higherCPR"], pf["lowerCPR"] = higherCPR, lowerCPR
    pf["insideCPR"], pf["outsideCPR"], pf["overlapCPR"] = insideCPR, outsideCPR, overlapCPR

    prevRange = (h1 - l1).clip(lower=TICK_SIZE)

    priorCPRCenter = (PriorCPRHigh + PriorCPRLow) / 2
    currCPRCenter = (CPRHigh + CPRLow) / 2
    centerShift = currCPRCenter - priorCPRCenter
    centerShiftPct = centerShift.abs() / prevRange * 100

    gapShiftUpRaw = CPRLow - PriorCPRHigh
    gapShiftDownRaw = PriorCPRLow - CPRHigh
    gapShiftUp = pd.Series(np.where(higherCPR, gapShiftUpRaw.clip(lower=0), 0.0), index=pf.index)
    gapShiftDown = pd.Series(np.where(lowerCPR, gapShiftDownRaw.clip(lower=0), 0.0), index=pf.index)
    gapShift = pd.Series(np.where(higherCPR, gapShiftUp, np.where(lowerCPR, gapShiftDown, 0.0)), index=pf.index)
    gapShiftPct = gapShift / prevRange * 100

    stateShiftPct = pd.concat([centerShiftPct.fillna(0), gapShiftPct.fillna(0)], axis=1).max(axis=1)
    pf["stateShiftPct"] = stateShiftPct
    pf["stateShiftUp"] = centerShift > 0
    pf["stateShiftDown"] = centerShift < 0

    weak = (stateShiftPct > 0) & (stateShiftPct < config["weakStateShiftMaxPct"])
    moderate = (stateShiftPct >= config["weakStateShiftMaxPct"]) & (stateShiftPct < config["moderateStateShiftMaxPct"])
    strong = (stateShiftPct >= config["moderateStateShiftMaxPct"]) & (stateShiftPct < config["strongStateShiftMaxPct"])
    extreme = stateShiftPct >= config["strongStateShiftMaxPct"]
    pf["weakStateShift"], pf["moderateStateShift"] = weak, moderate
    pf["strongStateShift"], pf["extremeStateShift"] = strong, extreme
    pf["directionalStateShift"] = moderate | strong | extreme

    lookback = config["cprWidthLookback"]
    window = 5 if lookback <= 5 else (10 if lookback <= 10 else 20)
    avgCPRWidth = CPRWidth.rolling(window, min_periods=1).mean()
    relativeWidth = (CPRWidth / avgCPRWidth).replace([np.inf, -np.inf], np.nan)
    pf["relativeWidth"] = relativeWidth

    isNarrowCPR = relativeWidth < config["narrowCPRThreshold"]
    isVeryWideCPR = relativeWidth >= config["veryWideCPRThreshold"]
    isWideCPR = (relativeWidth >= config["wideCPRThreshold"]) & (~isVeryWideCPR)
    isNormalCPR = (~isNarrowCPR) & (~isWideCPR) & (~isVeryWideCPR)
    pf["isNarrowCPR"], pf["isWideCPR"] = isNarrowCPR, isWideCPR
    pf["isVeryWideCPR"], pf["isNormalCPR"] = isVeryWideCPR, isNormalCPR

    cprOverlapTop = np.minimum(CPRHigh, PriorCPRHigh)
    cprOverlapBot = np.maximum(CPRLow, PriorCPRLow)
    cprOverlapAmt = (cprOverlapTop - cprOverlapBot).clip(lower=0)
    currCPRWidthSafe = (CPRHigh - CPRLow).clip(lower=TICK_SIZE)
    overlapPct = cprOverlapAmt / currCPRWidthSafe * 100
    pf["overlapPct"] = overlapPct

    overlapAllowed = overlapCPR & (overlapPct < config["overlapGoNoGoThresholdPct"])
    overlapHighRisk = overlapCPR & (overlapPct >= config["overlapGoNoGoThresholdPct"])
    overlappingHigher = overlapCPR & (CPRLow > PriorCPRLow) & (CPRHigh > PriorCPRHigh)
    overlappingLower = overlapCPR & (CPRLow < PriorCPRLow) & (CPRHigh < PriorCPRHigh)
    pf["overlapAllowed"], pf["overlapHighRisk"] = overlapAllowed, overlapHighRisk
    pf["overlappingHigher"], pf["overlappingLower"] = overlappingHigher, overlappingLower

    overlap_filled = overlapCPR.fillna(False)
    reset_grp = (~overlap_filled).cumsum()
    overlapStreak = overlap_filled.groupby(reset_grp).cumsum()
    overlapStreak = overlapStreak.where(overlapCPR.notna())
    pf["overlapStreak"] = overlapStreak
    pf["multiOverlap"] = overlapStreak >= config["multiOverlapThreshold"]
    pf["compressionCoil"] = overlapStreak >= config["compressionCoilThreshold"]

    prevMidpoint = (h1 + l1) / 2
    prevCloseDistancePct = (c1 - prevMidpoint).abs() / prevRange * 100
    priorCloseBalanced = prevCloseDistancePct <= config["balancedCloseThresholdPct"]
    if config["enableBalancedRangeFilter"]:
        pf["balancedExtremeRange"] = priorCloseBalanced
    else:
        pf["balancedExtremeRange"] = pd.Series(False, index=pf.index)

    return pf


PERIOD_COLUMNS_TO_BROADCAST = [
    "PP", "MID", "PR", "CPRHigh", "CPRLow", "CPRWidth", "PriorCPRHigh", "PriorCPRLow",
    "higherCPR", "lowerCPR", "insideCPR", "outsideCPR", "overlapCPR",
    "stateShiftPct", "stateShiftUp", "stateShiftDown",
    "weakStateShift", "moderateStateShift", "strongStateShift", "extremeStateShift", "directionalStateShift",
    "relativeWidth", "isNarrowCPR", "isWideCPR", "isVeryWideCPR", "isNormalCPR",
    "overlapPct", "overlapAllowed", "overlapHighRisk", "overlappingHigher", "overlappingLower",
    "overlapStreak", "multiOverlap", "compressionCoil",
    "balancedExtremeRange", "c1", "h1", "l1",
]


# =====================================================
# MTF CPR (separate timeline, separate period key)
# =====================================================

def _compute_mtf(chart_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    mtf_key = _make_period_key(chart_df.index, config["mtf_tf"], config["mtf_mult"])
    mtf_period = _aggregate_period(chart_df, mtf_key)

    mtfH = mtf_period["High"].shift(1)
    mtfL = mtf_period["Low"].shift(1)
    mtfC = mtf_period["Close"].shift(1)
    mtfPP = (mtfH + mtfL + mtfC) / 3
    mtfMID = (mtfH + mtfL) / 2
    mtfPR = (mtfPP - mtfMID) + mtfPP
    mtf_period["mtfCPRHigh"] = np.maximum(mtfMID, mtfPR)
    mtf_period["mtfCPRLow"] = np.minimum(mtfMID, mtfPR)

    out = chart_df[[]].copy()
    out["_mtf_key"] = mtf_key
    out = out.merge(mtf_period[["mtfCPRHigh", "mtfCPRLow"]], left_on="_mtf_key", right_index=True, how="left")
    return out[["mtfCPRHigh", "mtfCPRLow"]]


# =====================================================
# Main entry point
# =====================================================

def compute_cpr_signals(chart_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    chart_df: DataFrame with Open/High/Low/Close/Volume, IST-indexed,
              at whatever chart timeframe you're scanning on.
    config:   see DEFAULT_CONFIG for all keys; merged over the defaults.

    Returns chart_df with every intermediate + final signal column added,
    including aPlusBuy / aPlusSell (boolean) and dashboard text fields.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    df = chart_df.copy()

    # --- Period-level CPR / Market State / Width / Overlap ---
    period_key = _make_period_key(df.index, cfg["cpr_tf"], cfg["cpr_mult"])
    period_frame = _aggregate_period(df, period_key)
    period_frame = _compute_period_level(period_frame, cfg)

    df["_cpr_period_key"] = period_key
    df = df.merge(
        period_frame[PERIOD_COLUMNS_TO_BROADCAST],
        left_on="_cpr_period_key", right_index=True, how="left",
    )
    df.drop(columns=["_cpr_period_key"], inplace=True)
    df = df.copy()  # defragment after the big merge

    # --- MTF ---
    mtf_cols = _compute_mtf(chart_df, cfg)
    df["mtfCPRHigh"] = mtf_cols["mtfCPRHigh"]
    df["mtfCPRLow"] = mtf_cols["mtfCPRLow"]
    df["mtfBull"] = df["Close"] > df["mtfCPRHigh"]
    df["mtfBear"] = df["Close"] < df["mtfCPRLow"]

    # --- Bar-level: EMA trend, price-vs-CPR, volume, chop ---
    ema = df["Close"].ewm(span=cfg["trendEMALength"], adjust=False).mean()
    df["emaTrend"] = ema
    df["emaRising"] = ema > ema.shift(1)
    df["emaFalling"] = ema < ema.shift(1)
    df["emaFlat"] = (ema - ema.shift(1)).abs() / df["Close"] * 100 <= cfg["flatEMAThresholdPct"]

    df["priceAboveCPR"] = df["Close"] > df["CPRHigh"]
    df["priceBelowCPR"] = df["Close"] < df["CPRLow"]
    df["priceInsideCPR"] = (df["Close"] <= df["CPRHigh"]) & (df["Close"] >= df["CPRLow"])

    df["avgVol"] = df["Volume"].rolling(cfg["proLookback"], min_periods=1).mean()
    df["lowVol"] = df["Volume"] < df["avgVol"]

    df["chopScore"] = (
        df["priceInsideCPR"].astype(int) + df["overlapCPR"].fillna(False).astype(int) +
        df["emaFlat"].astype(int) + df["lowVol"].astype(int) + (~df["isNarrowCPR"].fillna(False)).astype(int)
    )
    df["chopZone"] = df["chopScore"] >= cfg["chopScoreThreshold"]

    df["bullState"] = df["priceAboveCPR"] & df["emaRising"]
    df["bearState"] = df["priceBelowCPR"] & df["emaFalling"]
    df["balanceCompression"] = df["isNarrowCPR"] & df["priceInsideCPR"]
    df["bullCompression"] = df["isNarrowCPR"] & df["priceAboveCPR"]
    df["bearCompression"] = df["isNarrowCPR"] & df["priceBelowCPR"]

    # --- Balanced Midpoint Range Lock Filter (bar-level parts) ---
    df["avgBarRange"] = (df["High"] - df["Low"]).rolling(cfg["signalBarAvgRangeLength"], min_periods=1).mean()
    df["signalBarExpansion"] = (df["High"] - df["Low"]) >= df["avgBarRange"] * cfg["signalRangeMultiplier"]
    df["barTouchesCPR"] = (df["High"] >= df["CPRLow"]) & (df["Low"] <= df["CPRHigh"])

    df["balancedRangeBuyAllowed"] = (~df["balancedExtremeRange"]) | (
        df["signalBarExpansion"] & df["barTouchesCPR"] & (df["Close"] > df["CPRHigh"])
    )
    df["balancedRangeSellAllowed"] = (~df["balancedExtremeRange"]) | (
        df["signalBarExpansion"] & df["barTouchesCPR"] & (df["Close"] < df["CPRLow"])
    )

    df = df.copy()  # defragment

    # --- Opportunity Engine ---
    df["bullishExpansionOpportunity"] = (
        df["higherCPR"] & df["directionalStateShift"] & df["isNarrowCPR"] & df["priceAboveCPR"] & df["emaRising"]
    )
    df["bearishExpansionOpportunity"] = (
        df["lowerCPR"] & df["directionalStateShift"] & df["isNarrowCPR"] & df["priceBelowCPR"] & df["emaFalling"]
    )
    df["bullishTrendOpportunity"] = df["higherCPR"] & df["priceAboveCPR"] & df["emaRising"]
    df["bearishTrendOpportunity"] = df["lowerCPR"] & df["priceBelowCPR"] & df["emaFalling"]

    if cfg["allowResponsiveSignalsInWideCPR"]:
        df["responsiveOpportunity"] = (
            (df["isWideCPR"] | df["isVeryWideCPR"]) & df["priceInsideCPR"] &
            (~df["strongStateShift"]) & (~df["extremeStateShift"])
        )
    else:
        df["responsiveOpportunity"] = pd.Series(False, index=df.index)

    df["compressionCoilOpportunity"] = (
        df["isNarrowCPR"] & (df["insideCPR"] | df["multiOverlap"] | df["compressionCoil"]) & df["priceInsideCPR"]
    )
    df["rotationalBalanceOpportunity"] = df["overlapCPR"] & (~df["compressionCoil"]) & (~df["overlapHighRisk"])

    df = df.copy()  # defragment

    # --- Signal-pattern bar-level primitives ---
    df["barMid"] = (df["High"] + df["Low"]) / 2
    df["closeUpperHalf"] = df["Close"] > df["barMid"]
    df["closeLowerHalf"] = df["Close"] < df["barMid"]

    barRange = (df["High"] - df["Low"]).clip(lower=TICK_SIZE)
    df["closePosition"] = (df["Close"] - df["Low"]) / barRange
    df["closeTop25"] = df["closePosition"] >= 0.75
    df["closeBottom25"] = df["closePosition"] <= 0.25

    df["prevCloseAboveCPR"] = df["c1"] > df["CPRHigh"]
    df["prevCloseBelowCPR"] = df["c1"] < df["CPRLow"]

    df["buyRejectionClose"] = df["closePosition"] >= 1.0 - cfg["rejectionCloseThresholdPct"] / 100.0
    df["sellRejectionClose"] = df["closePosition"] <= cfg["rejectionCloseThresholdPct"] / 100.0

    def _buy_at_level(level_col, rej_col):
        level = df[level_col]
        return level.notna() & (df["Low"] < level) & (df["Close"] > level) & df[rej_col]

    def _sell_at_level(level_col, rej_col):
        level = df[level_col]
        return level.notna() & (df["High"] > level) & (df["Close"] < level) & df[rej_col]

    df["buyPR"] = _buy_at_level("PR", "buyRejectionClose")
    df["buyPP"] = _buy_at_level("PP", "buyRejectionClose")
    df["buyMID"] = _buy_at_level("MID", "buyRejectionClose")
    df["sellPR"] = _sell_at_level("PR", "sellRejectionClose")
    df["sellPP"] = _sell_at_level("PP", "sellRejectionClose")
    df["sellMID"] = _sell_at_level("MID", "sellRejectionClose")
    df["sellCPRLow"] = _sell_at_level("CPRLow", "sellRejectionClose")

    mid_valid = df["MID"].notna()
    o, c, l, h, mid, bmid = df["Open"], df["Close"], df["Low"], df["High"], df["MID"], df["barMid"]
    df["buyThruMID"] = mid_valid & (
        ((o < mid) & (c > mid) & df["closeUpperHalf"]) |
        ((o > mid) & (l < mid) & (c > mid) & df["closeUpperHalf"] & (c > o))
    )
    df["sellThruMID"] = mid_valid & (
        ((o > mid) & (c < mid) & df["closeLowerHalf"]) |
        ((o < mid) & (h > mid) & (c < bmid) & (c < o))
    )

    df["higherBullCase"] = df["higherCPR"] & df["prevCloseAboveCPR"]
    df["lowerBearCase"] = df["lowerCPR"] & df["prevCloseBelowCPR"]
    df["higherFailedCase"] = df["higherCPR"] & df["prevCloseBelowCPR"]
    df["lowerFailedCase"] = df["lowerCPR"] & df["prevCloseAboveCPR"]

    prev_close_bar = df["Close"].shift(1)
    df["insideBreakoutBuy"] = (
        df["insideCPR"] & (prev_close_bar <= df["CPRHigh"]) & (df["Close"] > df["CPRHigh"]) &
        (df["barMid"] > df["CPRHigh"]) & (df["Close"] > df["Open"]) & df["closeTop25"]
    )
    df["insideBreakoutSell"] = (
        df["insideCPR"] & (prev_close_bar >= df["CPRLow"]) & (df["Close"] < df["CPRLow"]) &
        (df["barMid"] < df["CPRLow"]) & (df["Close"] < df["Open"]) & df["closeBottom25"]
    )

    df["outsideBuyCase"] = df["outsideCPR"] & df["prevCloseAboveCPR"] & (df["buyPR"] | df["buyThruMID"])
    df["outsideSellCase"] = df["outsideCPR"] & df["prevCloseBelowCPR"] & (df["sellPR"] | df["sellThruMID"])

    df["overlapBuy"] = df["overlappingHigher"] & df["overlapAllowed"] & df["prevCloseAboveCPR"] & df["buyPR"]
    df["overlapSell"] = (
        df["overlappingLower"] & df["overlapAllowed"] & df["prevCloseBelowCPR"] &
        (df["High"] > df["CPRLow"]) & (df["Close"] < df["CPRLow"]) & df["closeBottom25"] & (df["Close"] < df["Open"])
    )
    df["overlapBreakoutBuy"] = (
        df["overlapHighRisk"] & (~df["chopZone"]) & (prev_close_bar <= df["CPRHigh"]) & (df["Close"] > df["CPRHigh"]) &
        (df["barMid"] > df["CPRHigh"]) & (df["Close"] > df["Open"]) & df["closeTop25"]
    )
    df["overlapBreakoutSell"] = (
        df["overlapHighRisk"] & (~df["chopZone"]) & (prev_close_bar >= df["CPRLow"]) & (df["Close"] < df["CPRLow"]) &
        (df["barMid"] < df["CPRLow"]) & (df["Close"] < df["Open"]) & df["closeBottom25"]
    )
    df["overlapSignalBlock"] = df["overlapCPR"] & ~(
        df["overlapBuy"] | df["overlapSell"] | df["overlapBreakoutBuy"] | df["overlapBreakoutSell"]
    )

    df["standDownOpportunity"] = (
        df["chopZone"] | df["overlapHighRisk"] | df["compressionCoil"] |
        (df["balancedExtremeRange"] & ~df["signalBarExpansion"])
    )

    df["bullishDirectionalConviction"] = (
        df["higherCPR"] | df["bullishExpansionOpportunity"] | df["bullishTrendOpportunity"] |
        df["bullState"] | df["prevCloseAboveCPR"]
    )
    df["bearishDirectionalConviction"] = (
        df["lowerCPR"] | df["bearishExpansionOpportunity"] | df["bearishTrendOpportunity"] |
        df["bearState"] | df["prevCloseBelowCPR"]
    )

    if cfg["prioritizeMajorMIDTrendBreaks"]:
        df["majorBearishMIDBreak"] = (
            df["bullishDirectionalConviction"] & mid_valid & (df["High"] > mid) & (df["Close"] < mid) &
            (df["Close"] < df["Open"]) & df["closeLowerHalf"] & df["closeBottom25"] & df["signalBarExpansion"]
        )
        df["majorBullishMIDBreak"] = (
            df["bearishDirectionalConviction"] & mid_valid & (df["Low"] < mid) & (df["Close"] > mid) &
            (df["Close"] > df["Open"]) & df["closeUpperHalf"] & df["closeTop25"] & df["signalBarExpansion"]
        )
    else:
        df["majorBearishMIDBreak"] = pd.Series(False, index=df.index)
        df["majorBullishMIDBreak"] = pd.Series(False, index=df.index)

    if cfg["enableWideCPRReversionOverride"]:
        wide_reversion_env = df["isVeryWideCPR"]
        df["wideReversionSell"] = wide_reversion_env & (df["High"] > df["CPRHigh"]) & (df["Close"] < df["CPRHigh"]) & df["sellRejectionClose"]
        df["wideReversionBuy"] = wide_reversion_env & (df["Low"] < df["CPRLow"]) & (df["Close"] > df["CPRLow"]) & df["buyRejectionClose"]
    else:
        df["wideReversionSell"] = pd.Series(False, index=df.index)
        df["wideReversionBuy"] = pd.Series(False, index=df.index)

    if cfg["allowStateBreakSignalsThroughStandDown"]:
        df["bullishStateBreakSignal"] = df["signalBarExpansion"] & (
            df["majorBullishMIDBreak"] | ((df["Close"] > df["CPRHigh"]) & df["closeTop25"] & (df["Close"] > df["Open"]))
        )
        df["bearishStateBreakSignal"] = df["signalBarExpansion"] & (
            df["majorBearishMIDBreak"] | ((df["Close"] < df["CPRLow"]) & df["closeBottom25"] & (df["Close"] < df["Open"]))
        )
    else:
        df["bullishStateBreakSignal"] = pd.Series(False, index=df.index)
        df["bearishStateBreakSignal"] = pd.Series(False, index=df.index)

    df["responsiveBuySignal"] = df["buyMID"] | df["buyPP"] | df["buyPR"] | df["buyThruMID"]
    df["responsiveSellSignal"] = df["sellMID"] | df["sellPP"] | df["sellPR"] | df["sellThruMID"] | df["sellCPRLow"]

    df = df.copy()  # defragment

    # --- Opportunity permission gates ---
    if cfg["enableOpportunityEngine"]:
        df["buyOpportunityPermission"] = (
            df["majorBullishMIDBreak"] | df["wideReversionBuy"] |
            df["bullishExpansionOpportunity"] | df["bullishTrendOpportunity"] |
            df["outsideBuyCase"] | df["overlapBuy"] | df["overlapBreakoutBuy"] |
            (df["compressionCoilOpportunity"] & df["bullishStateBreakSignal"]) |
            (df["rotationalBalanceOpportunity"] & (df["overlapBuy"] | df["bullishStateBreakSignal"])) |
            (df["responsiveOpportunity"] & df["responsiveBuySignal"]) |
            (df["standDownOpportunity"] & df["bullishStateBreakSignal"]) |
            (df["lowerFailedCase"] & df["bullishStateBreakSignal"])
        )
        df["sellOpportunityPermission"] = (
            df["majorBearishMIDBreak"] | df["wideReversionSell"] |
            df["bearishExpansionOpportunity"] | df["bearishTrendOpportunity"] |
            df["outsideSellCase"] | df["overlapSell"] | df["overlapBreakoutSell"] |
            (df["compressionCoilOpportunity"] & df["bearishStateBreakSignal"]) |
            (df["rotationalBalanceOpportunity"] & (df["overlapSell"] | df["bearishStateBreakSignal"])) |
            (df["responsiveOpportunity"] & df["responsiveSellSignal"]) |
            (df["standDownOpportunity"] & df["bearishStateBreakSignal"]) |
            (df["higherFailedCase"] & df["bearishStateBreakSignal"])
        )
    else:
        df["buyOpportunityPermission"] = pd.Series(True, index=df.index)
        df["sellOpportunityPermission"] = pd.Series(True, index=df.index)

    # --- Core rule triggers ---
    df["ruleBuy"] = (~df["overlapSignalBlock"]) & (
        ((df["higherBullCase"] | df["lowerFailedCase"]) & (df["buyPR"] | df["buyPP"] | df["buyMID"])) |
        (df["lowerBearCase"] & df["buyThruMID"]) |
        df["wideReversionBuy"] | df["majorBullishMIDBreak"] | df["insideBreakoutBuy"] |
        df["outsideBuyCase"] | df["overlapBuy"] | df["overlapBreakoutBuy"]
    )
    df["ruleSell"] = (~df["overlapSignalBlock"]) & (
        (df["lowerBearCase"] & (df["sellMID"] | df["sellCPRLow"] | df["sellPP"])) |
        (df["higherFailedCase"] & (df["sellMID"] | df["sellPR"] | df["sellPP"])) |
        df["majorBearishMIDBreak"] | df["insideBreakoutSell"] |
        df["outsideSellCase"] | df["overlapSell"] | df["overlapBreakoutSell"]
    )

    df["trendBuy"] = df["bullState"] & df["ruleBuy"]
    df["trendSell"] = df["bearState"] & df["ruleSell"]
    df["trendBreakBuy"] = df["majorBullishMIDBreak"]
    df["trendBreakSell"] = df["majorBearishMIDBreak"]

    df["baseBuy"] = df["majorBullishMIDBreak"] | df["wideReversionBuy"] | (df["ruleBuy"] & df["buyOpportunityPermission"])
    df["baseSell"] = df["majorBearishMIDBreak"] | df["wideReversionSell"] | (df["ruleSell"] & df["sellOpportunityPermission"])

    df = df.copy()  # defragment

    # --- Confirmations + Score ---
    df["volConfirm"] = df["Volume"] > df["avgVol"] * cfg["volumeMultiplier"]
    volDeltaProxy = (df["Close"] - df["Open"]) * df["Volume"]
    df["buyDeltaConfirm"] = volDeltaProxy > 0
    df["sellDeltaConfirm"] = volDeltaProxy < 0

    mtf_mode = cfg["mtfMode"]
    mtf_bull_score = df["mtfBull"] if mtf_mode == "Soft" else pd.Series(False, index=df.index)
    mtf_bear_score = df["mtfBear"] if mtf_mode == "Soft" else pd.Series(False, index=df.index)

    df["buyScore"] = (
        df["priceAboveCPR"].astype(int) + df["emaRising"].astype(int) + df["higherCPR"].fillna(False).astype(int) +
        df["isNarrowCPR"].fillna(False).astype(int) + df["volConfirm"].astype(int) + df["buyDeltaConfirm"].astype(int) +
        df["closeUpperHalf"].astype(int) + mtf_bull_score.astype(int) + (~df["chopZone"]).astype(int) +
        (df["trendBuy"] | df["trendBreakBuy"]).astype(int)
    )
    df["sellScore"] = (
        df["priceBelowCPR"].astype(int) + df["emaFalling"].astype(int) + df["lowerCPR"].fillna(False).astype(int) +
        df["isNarrowCPR"].fillna(False).astype(int) + df["volConfirm"].astype(int) + df["sellDeltaConfirm"].astype(int) +
        df["closeLowerHalf"].astype(int) + mtf_bear_score.astype(int) + (~df["chopZone"]).astype(int) +
        (df["trendSell"] | df["trendBreakSell"]).astype(int)
    )

    if mtf_mode == "Strict":
        mtfBuyAllowed = df["mtfBull"]
        mtfSellAllowed = df["mtfBear"]
    else:
        mtfBuyAllowed = pd.Series(True, index=df.index)
        mtfSellAllowed = pd.Series(True, index=df.index)

    chopAllowed = (~df["chopZone"]) if cfg["blockSignalsInChop"] else pd.Series(True, index=df.index)

    df["buyQualifiedRaw"] = df["majorBullishMIDBreak"] | df["wideReversionBuy"] | (
        df["baseBuy"] & (df["buyScore"] >= cfg["minimumSignalScore"]) & mtfBuyAllowed & chopAllowed & df["balancedRangeBuyAllowed"]
    )
    df["sellQualifiedRaw"] = df["majorBearishMIDBreak"] | df["wideReversionSell"] | (
        df["baseSell"] & (df["sellScore"] >= cfg["minimumSignalScore"]) & mtfSellAllowed & chopAllowed & df["balancedRangeSellAllowed"]
    )

    # oneSignalPerDirectionPerPeriod is confirmed OFF -- no per-period dedup applied.
    df["earlyBuy"] = df["buyQualifiedRaw"].fillna(False)
    df["earlySell"] = df["sellQualifiedRaw"].fillna(False)

    df["confirmedBuy"] = df["earlyBuy"] & df["volConfirm"] & df["buyDeltaConfirm"]
    df["confirmedSell"] = df["earlySell"] & df["volConfirm"] & df["sellDeltaConfirm"]

    df["confirmedTrendBuy"] = df["confirmedBuy"] & df["trendBuy"]
    df["confirmedTrendSell"] = df["confirmedSell"] & df["trendSell"]
    df["confirmedBreakBuy"] = df["trendBreakBuy"] & df["earlyBuy"]
    df["confirmedBreakSell"] = df["trendBreakSell"] & df["earlySell"]

    df["aPlusBuy"] = (df["confirmedBuy"] & (df["buyScore"] >= cfg["aPlusScoreThreshold"])).fillna(False)
    df["aPlusSell"] = (df["confirmedSell"] & (df["sellScore"] >= cfg["aPlusScoreThreshold"])).fillna(False)

    df = df.copy()  # defragment before adding the dashboard text columns
    _add_dashboard_labels(df)

    return df


# =====================================================
# Dashboard label text (mirrors the AddLabel section)
# =====================================================

def _add_dashboard_labels(df: pd.DataFrame) -> None:
    state = np.select(
        [
            df["balancedExtremeRange"] & ~df["signalBarExpansion"],
            df["balancedExtremeRange"] & df["signalBarExpansion"],
            df["chopZone"],
            df["balanceCompression"],
            df["bullCompression"],
            df["bearCompression"],
            df["bullState"],
            df["bearState"],
            df["priceAboveCPR"],
            df["priceBelowCPR"],
        ],
        [
            "Range Lock", "Expansion Unlock", "Chop / Stand Down", "Balance Compression",
            "Bullish Compression", "Bearish Compression", "Bullish Acceptance",
            "Bearish Acceptance", "Above Value", "Below Value",
        ],
        default="Inside Value",
    )
    df["State"] = state

    width = np.select(
        [df["isNarrowCPR"], df["isVeryWideCPR"], df["isWideCPR"]],
        ["Narrow", "Very Wide", "Wide"],
        default="Normal",
    )
    df["Width"] = [f"{w} ({r:.2f})" if pd.notna(r) else w for w, r in zip(width, df["relativeWidth"])]

    relation = np.select(
        [df["higherCPR"], df["lowerCPR"], df["insideCPR"], df["outsideCPR"]],
        ["Higher Value", "Lower Value", "Inside Value", "Outside Value"],
        default="Overlapping Value",
    )
    df["Relation"] = relation

    shift_dir = np.where(df["stateShiftUp"], "Up", np.where(df["stateShiftDown"], "Down", "-"))
    df["StateShift"] = [f"{p:.1f}% {d}" if pd.notna(p) else "- %" for p, d in zip(df["stateShiftPct"], shift_dir)]

    shift_class = np.select(
        [df["weakStateShift"], df["moderateStateShift"], df["strongStateShift"], df["extremeStateShift"]],
        ["Weak", "Moderate", "Strong", "Extreme"],
        default="None",
    )
    df["Shift"] = shift_class

    opportunity = np.select(
        [
            df["isVeryWideCPR"], df["isWideCPR"],
            df["bullishExpansionOpportunity"], df["bearishExpansionOpportunity"],
            df["bullishTrendOpportunity"], df["bearishTrendOpportunity"],
            df["compressionCoilOpportunity"], df["standDownOpportunity"],
        ],
        [
            "Very Wide Reversion", "Wide Reversion Watch", "Bullish Expansion", "Bearish Expansion",
            "Bullish Trend", "Bearish Trend", "Compression Coil", "Chop / Stand Down",
        ],
        default=np.where(df["priceAboveCPR"], "Above Value", np.where(df["priceBelowCPR"], "Below Value", "Inside Value")),
    )
    df["Opportunity"] = opportunity

    energy = np.select(
        [
            df["standDownOpportunity"], df["isVeryWideCPR"], df["isWideCPR"],
            df["isNarrowCPR"] & df["directionalStateShift"], df["directionalStateShift"], df["priceInsideCPR"],
        ],
        ["Chop", "Exhaustion", "Responsive", "Expansion", "Trend", "Balance"],
        default="Neutral",
    )
    df["Energy"] = energy

    conviction_score = (
        (df["higherCPR"] | df["lowerCPR"]).astype(int) + df["directionalStateShift"].astype(int) +
        (df["strongStateShift"] | df["extremeStateShift"]).astype(int) +
        ((df["mtfBull"] & df["stateShiftUp"]) | (df["mtfBear"] & df["stateShiftDown"])).astype(int) +
        (df["isNarrowCPR"] | df["isNormalCPR"]).astype(int) -
        df["isWideCPR"].astype(int) - 2 * df["isVeryWideCPR"].astype(int) -
        df["chopZone"].astype(int) - df["overlapHighRisk"].astype(int)
    )
    df["ConvictionScore"] = conviction_score
    df["Conviction"] = np.select(
        [conviction_score >= 4, conviction_score >= 3, conviction_score >= 2],
        ["Extreme", "High", "Moderate"],
        default="Low",
    )

    df["MTF"] = np.where(df["mtfBull"], "Bullish", np.where(df["mtfBear"], "Bearish", "Neutral"))
    df["Chop"] = np.where(df["chopZone"], "Blocked", "Clear")
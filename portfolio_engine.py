"""
Portfolio analytics engine: per-lot XIRR, tax-lot status, concentration,
sector allocation, Nifty-equivalent benchmark comparison, and monthly/
weekly SMA9/SMA20 technical signals.

Data model: one row per BUY LOT (a stock bought on multiple dates has
multiple rows, same symbol). This is the only way to compute an honest
money-weighted return (XIRR) when a stock was bought in tranches.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import brentq

LTCG_HOLDING_DAYS = 365
NIFTY_SECURITY_ID = "13"
NIFTY_EXCHANGE_SEGMENT = "IDX_I"


def load_lots(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["actual_rate"] = df["actual_rate"].astype(float)
    df["actual_value"] = df["actual_value"].astype(float)
    df["quantity"] = df["quantity"].astype(int)
    return df


def load_sector_map(csv_path: str) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    return dict(zip(df["symbol"], df["sector"]))


def load_transactions(csv_path: str) -> pd.DataFrame:
    """
    Append-only transaction log: symbol, company_name, date, action
    (BUY/SELL), quantity, rate, value. This is the source of truth going
    forward -- open lots are DERIVED from it via resolve_open_lots_fifo(),
    not maintained by hand.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["action"] = df["action"].str.upper()
    df["quantity"] = df["quantity"].astype(int)
    df["rate"] = df["rate"].astype(float)
    df["value"] = df["value"].astype(float)
    return df


def resolve_open_lots_fifo(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Nets BUY/SELL transactions per symbol using FIFO (sell against the
    oldest open lot first -- matches how Indian STCG/LTCG holding periods
    are actually determined for tax purposes, not an arbitrary choice).

    Returns:
      open_lots: same column shape as load_lots()'s output (symbol,
        company_name, date, quantity, actual_rate, actual_value) --
        drop-in compatible with compute_stock_metrics(), so nothing
        downstream needs to change.
      realized_gains: one row per SELL-to-BUY(s) match, with the realized
        gain and whether that specific match was long- or short-term.
    """
    open_lot_rows = []
    realized_rows = []

    for symbol, group in transactions.groupby("symbol"):
        group = group.sort_values("date")
        company_name = group["company_name"].iloc[0]
        queue: list[dict] = []  # open lots for this symbol, oldest first

        for _, txn in group.iterrows():
            if txn["action"] == "BUY":
                queue.append({"date": txn["date"], "quantity": txn["quantity"], "rate": txn["rate"]})
            elif txn["action"] == "SELL":
                remaining_to_sell = txn["quantity"]
                while remaining_to_sell > 0 and queue:
                    lot = queue[0]
                    matched_qty = min(remaining_to_sell, lot["quantity"])
                    holding_days = (txn["date"] - lot["date"]).days
                    realized_rows.append({
                        "symbol": symbol,
                        "company_name": company_name,
                        "buy_date": lot["date"].date(),
                        "sell_date": txn["date"].date(),
                        "quantity": matched_qty,
                        "buy_rate": lot["rate"],
                        "sell_rate": txn["rate"],
                        "realized_gain": round(matched_qty * (txn["rate"] - lot["rate"]), 2),
                        "holding_days": holding_days,
                        "tax_status": "LTCG" if holding_days >= LTCG_HOLDING_DAYS else "STCG",
                    })
                    lot["quantity"] -= matched_qty
                    remaining_to_sell -= matched_qty
                    if lot["quantity"] == 0:
                        queue.pop(0)
                if remaining_to_sell > 0:
                    raise ValueError(
                        f"{symbol}: SELL on {txn['date'].date()} for {txn['quantity']} shares "
                        f"exceeds open quantity at that point (short by {remaining_to_sell}). "
                        "Check the transaction log for a missing BUY or a data entry mistake."
                    )
            else:
                raise ValueError(
                    f"{symbol}: unknown action '{txn['action']}' on {txn['date'].date()} -- must be BUY or SELL."
                )

        for lot in queue:
            open_lot_rows.append({
                "symbol": symbol,
                "company_name": company_name,
                "date": lot["date"],
                "quantity": lot["quantity"],
                "actual_rate": lot["rate"],
                "actual_value": round(lot["quantity"] * lot["rate"], 2),
            })

    open_lots = pd.DataFrame(open_lot_rows)
    realized_gains = pd.DataFrame(realized_rows)
    return open_lots, realized_gains


# ---------------------------------------------------------------------------
# XIRR
# ---------------------------------------------------------------------------

def xirr(cashflows: list[tuple[date, float]], guess_bounds: tuple[float, float] = (-0.999, 50.0)) -> float | None:
    """
    Money-weighted annualized return for a set of dated cash flows.
    Convention: purchases are NEGATIVE (money out), the final valuation is
    POSITIVE (money "in" if liquidated today). Handles irregular dates and
    multiple lots naturally -- a zero-value cash flow (e.g. a bonus/split
    share credit with actual_value=0) contributes nothing to the equation
    and doesn't need special-casing.

    Returns None if no sign change exists (e.g. all cash flows are zero or
    same-signed) -- brentq needs a genuine root-bracketing interval.
    """
    if len(cashflows) < 2:
        return None
    t0 = min(d for d, _ in cashflows)

    def npv(rate: float) -> float:
        total = 0.0
        for d, amount in cashflows:
            years = (d - t0).days / 365.0
            total += amount / ((1 + rate) ** years)
        return total

    lo, hi = guess_bounds
    try:
        npv_lo, npv_hi = npv(lo), npv(hi)
        if npv_lo * npv_hi > 0:
            return None  # no sign change in range -- can't bracket a root
        return brentq(npv, lo, hi, maxiter=200)
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# Per-stock metrics
# ---------------------------------------------------------------------------

def compute_stock_metrics(
    lots: pd.DataFrame, current_prices: dict[str, float], sector_map: dict[str, str],
    as_of: date | None = None,
) -> pd.DataFrame:
    """
    One row per symbol: total qty, invested, current value, gain, XIRR,
    LTCG%, sector, and each lot's own age/tax status for drill-down.
    """
    as_of = as_of or date.today()
    rows = []
    for symbol, group in lots.groupby("symbol"):
        price = current_prices.get(symbol)
        if price is None:
            continue

        total_qty = int(group["quantity"].sum())
        total_invested = float(group["actual_value"].sum())
        total_current = total_qty * price
        total_gain = total_current - total_invested
        gain_pct = (total_gain / total_invested * 100) if total_invested > 0 else None

        cashflows = [(row["date"].date(), -row["actual_value"]) for _, row in group.iterrows()]
        cashflows.append((as_of, total_current))
        stock_xirr = xirr(cashflows)

        ltcg_value = 0.0
        for _, row in group.iterrows():
            held_days = (as_of - row["date"].date()).days
            lot_current_value = row["quantity"] * price
            if held_days >= LTCG_HOLDING_DAYS:
                ltcg_value += lot_current_value
        ltcg_pct = (ltcg_value / total_current * 100) if total_current > 0 else 0.0

        rows.append({
            "symbol": symbol,
            "company_name": group["company_name"].iloc[0],
            "sector": sector_map.get(symbol, "Unknown"),
            "quantity": total_qty,
            "entry_price": round(total_invested / total_qty, 2) if total_qty else None,
            "current_price": round(price, 2),
            "invested": round(total_invested, 2),
            "current_value": round(total_current, 2),
            "gain": round(total_gain, 2),
            "gain_pct": round(gain_pct, 2) if gain_pct is not None else None,
            "xirr_pct": round(stock_xirr * 100, 2) if stock_xirr is not None else None,
            "ltcg_pct": round(ltcg_pct, 1),
            "num_lots": len(group),
            "earliest_purchase": group["date"].min().date(),
        })

    return pd.DataFrame(rows).sort_values("current_value", ascending=False).reset_index(drop=True)


def compute_portfolio_summary(stock_metrics: pd.DataFrame, lots: pd.DataFrame, as_of: date | None = None) -> dict:
    as_of = as_of or date.today()
    total_invested = stock_metrics["invested"].sum()
    total_current = stock_metrics["current_value"].sum()
    total_gain = total_current - total_invested

    all_cashflows = [(row["date"].date(), -row["actual_value"]) for _, row in lots.iterrows()]
    all_cashflows.append((as_of, total_current))
    portfolio_xirr = xirr(all_cashflows)

    top_holding = stock_metrics.iloc[0] if not stock_metrics.empty else None
    concentration_pct = (top_holding["current_value"] / total_current * 100) if total_current > 0 and top_holding is not None else 0.0

    sector_alloc = (
        stock_metrics.groupby("sector")["current_value"].sum().sort_values(ascending=False)
    )
    sector_alloc_pct = (sector_alloc / total_current * 100).round(1) if total_current > 0 else sector_alloc

    return {
        "total_invested": round(total_invested, 2),
        "total_current": round(total_current, 2),
        "total_gain": round(total_gain, 2),
        "total_gain_pct": round(total_gain / total_invested * 100, 2) if total_invested > 0 else None,
        "portfolio_xirr_pct": round(portfolio_xirr * 100, 2) if portfolio_xirr is not None else None,
        "top_holding_symbol": top_holding["symbol"] if top_holding is not None else None,
        "top_holding_concentration_pct": round(concentration_pct, 1),
        "concentration_flag": concentration_pct >= 15.0,
        "sector_allocation": sector_alloc.round(2).to_dict(),
        "sector_allocation_pct": sector_alloc_pct.to_dict(),
    }


# ---------------------------------------------------------------------------
# Benchmark comparison (Nifty-equivalent XIRR using the SAME cash flow
# dates/amounts as the real portfolio -- an apples-to-apples "what if you'd
# bought Nifty instead, on the same days, with the same money" comparison)
# ---------------------------------------------------------------------------

def compute_benchmark_xirr(lots: pd.DataFrame, nifty_daily: pd.Series, as_of: date | None = None) -> float | None:
    """
    nifty_daily: pd.Series of Nifty 50 Close, indexed by date (tz-naive or
    tz-aware, must cover from the earliest lot date through `as_of`).
    """
    as_of = as_of or date.today()
    nifty_daily = nifty_daily.copy()
    if hasattr(nifty_daily.index, "tz") and nifty_daily.index.tz is not None:
        nifty_daily.index = nifty_daily.index.tz_localize(None)

    def nifty_price_on_or_before(d: date) -> float | None:
        eligible = nifty_daily[nifty_daily.index.date <= d]
        return float(eligible.iloc[-1]) if not eligible.empty else None

    nifty_today = nifty_price_on_or_before(as_of)
    if nifty_today is None:
        return None

    cashflows = []
    total_units = 0.0
    for _, row in lots.iterrows():
        lot_date = row["date"].date()
        nifty_price = nifty_price_on_or_before(lot_date)
        if nifty_price is None or row["actual_value"] <= 0:
            continue
        units = row["actual_value"] / nifty_price
        total_units += units
        cashflows.append((lot_date, -row["actual_value"]))

    if not cashflows:
        return None
    cashflows.append((as_of, total_units * nifty_today))
    return xirr(cashflows)


def compute_per_stock_benchmark_alpha(
    lots: pd.DataFrame, stock_metrics: pd.DataFrame, nifty_daily: pd.Series, as_of: date | None = None,
) -> dict[str, float | None]:
    """
    For each symbol: that stock's own XIRR minus what the same rupees, on
    the same purchase dates, would have earned in Nifty 50 instead. This
    is the per-stock version of compute_benchmark_xirr -- same method,
    just run on one symbol's lots at a time so each stock gets its own
    "did picking this over the index actually pay off" answer.
    """
    alpha_by_symbol = {}
    xirr_by_symbol = dict(zip(stock_metrics["symbol"], stock_metrics["xirr_pct"]))

    for symbol, group in lots.groupby("symbol"):
        stock_xirr_pct = xirr_by_symbol.get(symbol)
        if stock_xirr_pct is None or pd.isna(stock_xirr_pct):
            alpha_by_symbol[symbol] = None
            continue
        benchmark_xirr = compute_benchmark_xirr(group, nifty_daily, as_of=as_of)
        if benchmark_xirr is None or pd.isna(benchmark_xirr):
            alpha_by_symbol[symbol] = None
        else:
            alpha_by_symbol[symbol] = round(stock_xirr_pct - benchmark_xirr * 100, 2)

    return alpha_by_symbol


# ---------------------------------------------------------------------------
# Max drawdown (since first purchase date, using the actual daily Close
# history -- not just the cost-basis-vs-current snapshot)
# ---------------------------------------------------------------------------

def compute_max_drawdown(daily_close: pd.Series, since: date) -> float | None:
    series = daily_close[daily_close.index.date >= since] if hasattr(daily_close.index, "date") else daily_close
    if series.empty:
        return None
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return round(float(drawdown.min()) * 100, 2)


# ---------------------------------------------------------------------------
# Technical signals: monthly 9-SMA flag + monthly/weekly 20-SMA crossover
# state, reusing the same resample-then-SMA approach as pivotboss_dashboard_dhan.py
# ---------------------------------------------------------------------------

def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    pandas 2.2+ renamed the month-end resample alias from "M" to "ME" (and
    will remove "M" entirely in pandas 3.0); older pandas only understands
    "M". Try the modern alias first, fall back for older installs, so this
    works regardless of which pandas version is actually installed.
    """
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    if rule == "M":
        try:
            return df.resample("ME").agg(agg).dropna()
        except ValueError:
            return df.resample("M").agg(agg).dropna()
    return df.resample(rule).agg(agg).dropna()


def _sma_state(close: pd.Series, period: int) -> str:
    """Returns one of: 'Just Crossed Below', 'Below', 'Just Crossed Back Above', 'Above'."""
    sma = close.rolling(period).mean()
    if len(close) < period + 2 or sma.iloc[-1] != sma.iloc[-1]:  # NaN check
        return "Insufficient Data"

    prev_close, curr_close = close.iloc[-2], close.iloc[-1]
    prev_sma, curr_sma = sma.iloc[-2], sma.iloc[-1]

    was_above = prev_close >= prev_sma
    is_above = curr_close >= curr_sma

    if was_above and not is_above:
        return "Just Crossed Below"
    if not was_above and is_above:
        return "Just Crossed Back Above"
    return "Above" if is_above else "Below"


def compute_technical_signals(daily_df: pd.DataFrame, sma_short: int = 9, sma_long: int = 20) -> dict:
    """
    daily_df: OHLCV DataFrame, daily bars, datetime index.
    Returns a dict with the monthly 9-SMA flag and the monthly + weekly
    20-SMA crossover states.
    """
    monthly = _resample(daily_df, "M")
    weekly = _resample(daily_df, "W-FRI")

    below_9sma_monthly = None
    if len(monthly) >= sma_short + 1:
        sma9_monthly = monthly["Close"].rolling(sma_short).mean()
        if sma9_monthly.iloc[-1] == sma9_monthly.iloc[-1]:
            below_9sma_monthly = bool(monthly["Close"].iloc[-1] < sma9_monthly.iloc[-1])

    return {
        "below_9sma_monthly": below_9sma_monthly,
        "sma20_state_monthly": _sma_state(monthly["Close"], sma_long) if len(monthly) >= sma_long + 2 else "Insufficient Data",
        "sma20_state_weekly": _sma_state(weekly["Close"], sma_long) if len(weekly) >= sma_long + 2 else "Insufficient Data",
    }
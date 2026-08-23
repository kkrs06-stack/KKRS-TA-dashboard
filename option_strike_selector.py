"""
OI-concentration-based OTM strike selection from Dhan's option chain data.

Picks the highest-OI strike within an OTM % band from spot -- treating
heavy OI as both a liquidity signal and a "wall" (resistance for calls,
support for puts). Surfaces delta as a secondary sanity check, not as the
primary selection criterion.
"""

from __future__ import annotations

DEFAULT_OTM_MIN_PCT = 3.0
DEFAULT_OTM_MAX_PCT = 8.0

# Delta sanity-check bounds: outside this range, flag it, even though OI
# concentration still picked this strike.
DELTA_HIGH_FLAG = 0.35   # closer to the money than an OTM pick should be
DELTA_LOW_FLAG = 0.05    # so far OTM the premium is likely negligible


def select_otm_strike(
    chain_data: dict,
    side: str,
    otm_min_pct: float = DEFAULT_OTM_MIN_PCT,
    otm_max_pct: float = DEFAULT_OTM_MAX_PCT,
) -> dict | None:
    """
    chain_data: the raw `data` dict from DhanMarketData.get_option_chain()
                (has "last_price" and "oc" keys).
    side: "CALL" (bearish exhaustion, sell calls above spot) or
          "PUT" (bullish exhaustion, sell puts below spot).

    Returns the recommended strike's info, or None if nothing qualifies
    (e.g. no strikes in the band have any real open interest).
    """
    spot = chain_data.get("last_price")
    oc = chain_data.get("oc", {})
    if not spot or not oc:
        return None

    candidates = []
    for strike_key, row in oc.items():
        try:
            strike = float(strike_key)
        except (TypeError, ValueError):
            continue
        pct_away = (strike - spot) / spot * 100

        if side == "CALL":
            if not (otm_min_pct <= pct_away <= otm_max_pct):
                continue
            leg = row.get("ce", {})
        elif side == "PUT":
            if not (-otm_max_pct <= pct_away <= -otm_min_pct):
                continue
            leg = row.get("pe", {})
        else:
            raise ValueError(f"side must be 'CALL' or 'PUT', got {side!r}")

        oi = leg.get("oi", 0) or 0
        if oi <= 0:
            continue  # no real interest at this strike, skip

        candidates.append({
            "strike": strike,
            "oi": oi,
            "previous_oi": leg.get("previous_oi", 0) or 0,
            "last_price": leg.get("last_price", 0) or 0,
            "implied_volatility": leg.get("implied_volatility", 0) or 0,
            "delta": (leg.get("greeks") or {}).get("delta", 0) or 0,
            "security_id": leg.get("security_id"),
            "pct_away_from_spot": pct_away,
        })

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c["oi"])
    best["oi_change"] = best["oi"] - best["previous_oi"]
    best["oi_trend"] = (
        "Building" if best["oi_change"] > 0
        else ("Unwinding" if best["oi_change"] < 0 else "Flat")
    )
    best["wall_label"] = "Resistance" if side == "CALL" else "Support"

    abs_delta = abs(best["delta"])
    if abs_delta > DELTA_HIGH_FLAG:
        best["delta_flag"] = f"Closer to money than expected (|delta|={abs_delta:.2f})"
    elif abs_delta < DELTA_LOW_FLAG and abs_delta > 0:
        best["delta_flag"] = f"Very far OTM, premium likely thin (|delta|={abs_delta:.2f})"
    else:
        best["delta_flag"] = None

    return best




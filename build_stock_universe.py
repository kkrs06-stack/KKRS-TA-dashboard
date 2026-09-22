"""
Builds stock_universe.csv for the Ribbon Scanner strategy: AMFI's official
semi-annual Large/Mid/Small Cap classification (published under SEBI's
circular dated Oct 6, 2017), filtered to a minimum market cap and matched
against Dhan's instrument master to get a tradeable symbol + confirm it's
actually resolvable via the API.

Matching is done by ISIN, not by parsing "which of AMFI's several symbol
columns is the NSE one" out of a PDF -- Dhan's own instrument master carries
ISIN too, so joining on that is far more reliable than text-matching a
ticker straight out of unstructured PDF text.

This is intentionally NOT restricted to F&O-eligible names (unlike
fo_stock_list.csv) -- Ribbon Scanner scans the full market-cap universe,
so keep the two files separate; don't merge them.

AMFI republishes this only twice a year (Jan and Jul, based on the
preceding 6 months' average market cap) -- there's no need to run this on
every dashboard load. Run manually when you want a refresh:

    python build_stock_universe.py
    python build_stock_universe.py --min-market-cap 5000
"""

from __future__ import annotations

import argparse
import io
import re
from datetime import date

import pandas as pd
import requests
from pypdf import PdfReader

from dhan_instruments import DhanInstrumentLookup

# AMFI has used two naming conventions over time -- try the current
# (shorter) one first, then the older descriptive one, for each candidate
# period. If AMFI changes the naming again, this loop will start failing
# for every period and raise the RuntimeError below with a pointer to the
# live page to check by hand.
AMFI_URL_TEMPLATES = [
    "https://www.amfiindia.com/Themes/Theme1/downloads/AverageMarketCapitalization{d}{mon}{yyyy}.pdf",
    "https://www.amfiindia.com/Themes/Theme1/downloads/AverageMarketCapitalizationoflistedcompaniesduringthesixmonthsended{d}{mon}{yyyy}.pdf",
]
MONTH_ABBR = {6: "Jun", 12: "Dec"}
DEFAULT_MIN_MARKET_CAP_CR = 3000.0
OUTPUT_PATH = "stock_universe.csv"

# Anchors on the ISIN (a fixed, reliable 12-char pattern) and the trailing
# category label -- both are structurally guaranteed by AMFI's format,
# unlike the variable number of BSE/NSE/MSEI symbol+market-cap column pairs
# that precede them (present only for exchanges a company is actually
# listed on). The captured market-cap figure is always the one immediately
# before the category label, i.e. "Average of All Exchanges" -- true
# regardless of how many symbol/cap pairs came before it.
ROW_PATTERN = re.compile(
    r"^\s*(\d+)\s+(.+?)\s+([A-Z]{2}[A-Z0-9]{9}\d)\s+.*?([\d,]+\.\d{2})\s+(Large Cap|Mid Cap|Small Cap)\s*$"
)


def _candidate_periods(count: int = 6) -> list[tuple[int, int, int]]:
    """Most recent AMFI publish periods first: (day, month, year) for the
    30 Jun / 31 Dec cutoffs, walking backward from today."""
    today = date.today()
    month, year = (6, today.year) if today.month < 7 else (12, today.year)
    periods = []
    for _ in range(count):
        day = 30 if month == 6 else 31
        periods.append((day, month, year))
        month, year = (12, year - 1) if month == 6 else (6, year)
    return periods


def _fetch_latest_amfi_pdf() -> tuple[bytes, str]:
    for day, month, year in _candidate_periods():
        mon = MONTH_ABBR[month]
        for template in AMFI_URL_TEMPLATES:
            url = template.format(d=day, mon=mon, yyyy=year)
            try:
                resp = requests.get(url, timeout=30)
            except requests.RequestException:
                continue
            if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                print(f"Found AMFI classification file: {url}")
                return resp.content, url
    raise RuntimeError(
        "Could not find a current AMFI market-cap classification PDF at any "
        "expected URL. AMFI may have changed its naming convention -- check "
        "https://www.amfiindia.com/research-information/other-data by hand "
        "and update AMFI_URL_TEMPLATES in build_stock_universe.py."
    )


def _parse_amfi_pdf(pdf_bytes: bytes) -> pd.DataFrame:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    rows, unparsed = [], 0
    for page in reader.pages:
        for line in page.extract_text().splitlines():
            match = ROW_PATTERN.match(line.strip())
            if not match:
                if line.strip()[:1].isdigit():
                    unparsed += 1
                continue
            sr_no, company_name, isin, market_cap_str, category = match.groups()
            rows.append({
                "sr_no": int(sr_no),
                "company_name": company_name.strip(),
                "isin": isin,
                "market_cap_cr": float(market_cap_str.replace(",", "")),
                "category": category,
            })
    if unparsed:
        print(f"Note: {unparsed} row(s) could not be parsed (likely a wrapped company name) and were skipped.")
    return pd.DataFrame(rows)


def _load_dhan_isin_maps() -> tuple[dict[str, str], dict[str, str]]:
    """
    ISIN -> (symbol, exchange), NSE preferred over BSE -- same preference
    order as DhanInstrumentLookup.resolve_with_exchange(). Reads Dhan's
    cached instrument master CSV directly rather than extending
    DhanInstrumentLookup itself, so this one-off universe-building script
    can't affect the shared lookup that every live strategy dashboard
    depends on.
    """
    lookup = DhanInstrumentLookup()  # triggers/reuses the existing 24h cache
    df = pd.read_csv(lookup.cache_path, low_memory=False)
    equity = df[df["SEGMENT"].astype(str).str.upper() == "E"]

    isin_to_symbol: dict[str, str] = {}
    isin_to_exchange: dict[str, str] = {}
    for exch in ("NSE", "BSE"):  # NSE first so it wins on ISINs listed on both
        exch_rows = equity[equity["EXCH_ID"].astype(str).str.upper() == exch]
        for isin, symbol in zip(exch_rows["ISIN"], exch_rows["UNDERLYING_SYMBOL"]):
            isin = str(isin).strip()
            symbol = str(symbol).strip().upper()
            if isin and isin != "nan" and symbol and symbol != "nan" and isin not in isin_to_symbol:
                isin_to_symbol[isin] = symbol
                isin_to_exchange[isin] = exch
    return isin_to_symbol, isin_to_exchange


def build_universe(min_market_cap_cr: float = DEFAULT_MIN_MARKET_CAP_CR) -> pd.DataFrame:
    pdf_bytes, source_url = _fetch_latest_amfi_pdf()
    amfi_df = _parse_amfi_pdf(pdf_bytes)
    print(f"Parsed {len(amfi_df)} companies from AMFI's classification.")

    filtered = amfi_df[amfi_df["market_cap_cr"] >= min_market_cap_cr].copy()
    print(f"{len(filtered)} companies pass the >= Rs {min_market_cap_cr:,.0f} Cr floor.")

    isin_to_symbol, isin_to_exchange = _load_dhan_isin_maps()
    filtered["symbol"] = filtered["isin"].map(isin_to_symbol)
    filtered["exchange"] = filtered["isin"].map(isin_to_exchange)

    matched = filtered.dropna(subset=["symbol"]).copy()
    dropped = len(filtered) - len(matched)
    if dropped:
        print(f"{dropped} companies had no Dhan-tradeable symbol for their ISIN and were dropped.")

    matched = matched.sort_values("market_cap_cr", ascending=False).reset_index(drop=True)
    result = matched[["symbol", "company_name", "isin", "exchange", "market_cap_cr", "category"]]
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(result)} rows to {OUTPUT_PATH}")
    print(f"Source: {source_url}")
    print(result["category"].value_counts().to_string())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--min-market-cap", type=float, default=DEFAULT_MIN_MARKET_CAP_CR,
        help="Minimum average market cap in Rs Crore (default: 3000)",
    )
    args = parser.parse_args()
    build_universe(args.min_market_cap)
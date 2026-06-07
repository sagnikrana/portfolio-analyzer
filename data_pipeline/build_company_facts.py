"""Fetch SEC XBRL company facts for every ticker the app cares about.

The original data pipeline scoped the SEC company-facts pull to whatever tickers
appeared in a single transactions file at build time, which left coverage at just
9 names. This script rebuilds ``company_facts.csv`` for a much wider universe:

  * every instrument held across one or more transactions CSVs, plus
  * (optionally) the buy-candidate universe.

It reproduces the exact schema and ``classify_fact_tag`` canonical mapping used by
the notebooks, so the output is a drop-in replacement for the existing files.

Usage
-----
    python data_pipeline/build_company_facts.py                 # holdings only
    python data_pipeline/build_company_facts.py --with-candidates
    python data_pipeline/build_company_facts.py --tickers AAPL MSFT NVDA
    python data_pipeline/build_company_facts.py --limit 50      # cap for a test run

SEC's fair-access policy requires a descriptive User-Agent and ~10 req/sec max.
Set SEC_USER_AGENT in your .env (e.g. "your-name your-email@example.com").
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional; environment variables still work.
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_SEC_DIR = DATA_DIR / "external" / "sec"
STRUCTURED_DIR = DATA_DIR / "processed" / "structured"

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Default holdings sources — every Instrument in these files is included.
DEFAULT_TRANSACTION_FILES = [
    RAW_DIR / "mantis_invest.csv",
    RAW_DIR / "fake_mantis_invest.csv",
]
BUY_CANDIDATE_UNIVERSE_PATH = RAW_DIR / "buy_candidate_universe.csv"

# Polite crawl delay between SEC requests (well under the 10 req/sec ceiling).
REQUEST_DELAY_SECONDS = 0.2

# The app only ever reads these fields from company_facts.csv (via
# diagnosis._latest_metric_value by canonical_metric, and the EBITDA estimator by
# raw fact_tag). Keeping only these trims the file to ~19% of the raw size, which
# matters a lot once the universe grows past a few hundred tickers — the app loads
# the whole CSV into memory at startup. Use --full to keep every fact tag.
RELEVANT_CANONICAL_METRICS = {
    "revenue", "net_income", "cash_and_equivalents", "total_assets",
    "liability_related", "debt_related", "stockholders_equity", "operating_cash_flow",
}
RELEVANT_FACT_TAGS = {
    # revenue
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues", "SalesRevenueNet",
    # operating income
    "OperatingIncomeLoss",
    # depreciation & amortization
    "DepreciationDepletionAndAmortization",
    "DepreciationAmortizationAndAccretionNet",
    "DepreciationAndAmortization",
    # debt
    "LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations",
    "LongTermDebtCurrent", "DebtCurrent",
}


def filter_to_relevant(facts_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the facts the app actually consumes (see notes above)."""
    if facts_df.empty:
        return facts_df
    mask = (
        facts_df["canonical_metric"].isin(RELEVANT_CANONICAL_METRICS)
        | facts_df["fact_tag"].isin(RELEVANT_FACT_TAGS)
    )
    return facts_df[mask].reset_index(drop=True)

load_dotenv(ROOT / ".env")

SEC_HEADERS = {
    "User-Agent": os.getenv(
        "SEC_USER_AGENT", "portfolio-analyzer research contact@example.com"
    ),
    "Accept-Encoding": "gzip, deflate",
}


# ── HTTP helpers (mirrors the notebook, with retry/backoff) ─────────────────

def _read_response_bytes(response) -> bytes:
    payload = response.read()
    encoding = (response.headers.get("Content-Encoding") or "").lower()
    if "gzip" in encoding:
        payload = gzip.decompress(payload)
    return payload


def fetch_url(url: str, *, retries: int = 4, base_delay: float = 1.5) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        request = Request(url, headers=SEC_HEADERS)
        try:
            with urlopen(request) as response:
                return _read_response_bytes(response)
        except HTTPError as error:
            last_error = error
            # 404 = no facts for this CIK; surface it to the caller immediately.
            if error.code == 404:
                raise
            if error.code != 429 or attempt == retries - 1:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else base_delay * (2 ** attempt)
            print(f"  rate limited; sleeping {delay:.1f}s "
                  f"(retry {attempt + 2}/{retries})")
            time.sleep(delay)
    raise last_error if last_error else RuntimeError("request failed")


def fetch_json(url: str) -> dict | list:
    return json.loads(fetch_url(url).decode("utf-8"))


# ── Canonical fact-tag classifier (verbatim from the data pipeline) ─────────

def classify_fact_tag(fact_tag: str) -> dict[str, str]:
    tag = (fact_tag or "").strip()
    lower = tag.lower()

    exact_map = {
        "revenues": ("income_statement", "revenue", "revenue", "core", "use_now"),
        "salesrevenuenet": ("income_statement", "revenue", "revenue", "core", "use_now"),
        "costofrevenue": ("income_statement", "costs", "cost_of_revenue", "core", "use_now"),
        "costofgoodsold": ("income_statement", "costs", "cost_of_goods_sold", "core", "use_now"),
        "grossprofit": ("income_statement", "profitability", "gross_profit", "core", "use_now"),
        "researchanddevelopmentexpense": ("income_statement", "expenses", "research_and_development", "secondary", "use_now"),
        "sellinggeneralandadministrativeexpense": ("income_statement", "expenses", "selling_general_admin", "secondary", "use_now"),
        "operatingexpenses": ("income_statement", "expenses", "operating_expenses", "secondary", "use_now"),
        "operatingincomeloss": ("income_statement", "profitability", "operating_income", "core", "use_now"),
        "netincomeloss": ("income_statement", "profitability", "net_income", "core", "use_now"),
        "earningspersharebasic": ("per_share", "eps", "eps_basic", "core", "use_now"),
        "earningspersharediluted": ("per_share", "eps", "eps_diluted", "core", "use_now"),
        "assets": ("balance_sheet", "assets", "total_assets", "core", "use_now"),
        "assetscurrent": ("balance_sheet", "assets", "current_assets", "core", "use_now"),
        "liabilities": ("balance_sheet", "liabilities", "total_liabilities", "core", "use_now"),
        "liabilitiescurrent": ("balance_sheet", "liabilities", "current_liabilities", "core", "use_now"),
        "stockholdersequity": ("balance_sheet", "equity", "stockholders_equity", "core", "use_now"),
        "stockholdersequityincludingportionattributabletononcontrollinginterest": ("balance_sheet", "equity", "stockholders_equity_incl_nci", "secondary", "use_now"),
        "cashandcashequivalentsatcarryingvalue": ("balance_sheet", "liquidity", "cash_and_equivalents", "core", "use_now"),
        "longtermdebtnoncurrent": ("balance_sheet", "debt", "long_term_debt", "core", "use_now"),
        "longtermdebtandcapitalleaseobligations": ("balance_sheet", "debt", "long_term_debt_and_capital_leases", "secondary", "use_now"),
        "shorttermborrowings": ("balance_sheet", "debt", "short_term_borrowings", "secondary", "use_now"),
        "commonstocksharesoutstanding": ("share_metrics", "share_count", "shares_outstanding", "core", "use_now"),
        "weightedaveragenumberofsharesoutstandingbasic": ("share_metrics", "share_count", "weighted_avg_shares_basic", "core", "use_now"),
        "weightedaveragenumberofdilutedsharesoutstanding": ("share_metrics", "share_count", "weighted_avg_shares_diluted", "core", "use_now"),
        "netcashprovidedbyusedinoperatingactivities": ("cash_flow", "cash_generation", "operating_cash_flow", "core", "use_now"),
        "paymentstoacquirepropertyplantandequipment": ("cash_flow", "capital_spending", "capital_expenditures", "core", "use_now"),
        "depreciationdepletionandamortization": ("cash_flow", "non_cash_items", "depreciation_amortization", "secondary", "use_now"),
    }
    if lower in exact_map:
        statement_group, fact_group, canonical_metric, priority_tier, review_action = exact_map[lower]
        return {
            "statement_group": statement_group,
            "fact_group": fact_group,
            "canonical_metric": canonical_metric,
            "priority_tier": priority_tier,
            "review_action": review_action,
            "tag_reason": "exact_match",
        }

    pattern_rules = [
        (r"revenue|sales", ("income_statement", "revenue", "revenue_related", "core", "use_now", "pattern_revenue")),
        (r"grossprofit|operatingincome|netincome|profit", ("income_statement", "profitability", "profitability_related", "core", "use_now", "pattern_profitability")),
        (r"expense|costof|marketing|advertising|fulfillment|restructuring", ("income_statement", "expenses", "expense_related", "secondary", "use_now", "pattern_expense")),
        (r"cash|cashequivalent|marketablesecurit", ("balance_sheet", "liquidity", "cash_or_investments", "core", "use_now", "pattern_cash")),
        (r"asset", ("balance_sheet", "assets", "asset_related", "secondary", "use_now", "pattern_assets")),
        (r"liabilit|payable|accrued", ("balance_sheet", "liabilities", "liability_related", "secondary", "use_now", "pattern_liabilities")),
        (r"debt|borrowing|notespayable|creditfacility", ("balance_sheet", "debt", "debt_related", "core", "use_now", "pattern_debt")),
        (r"equity|stockholder|retainedearnings|additionalpaidincapital|accumulatedothercomprehensive", ("balance_sheet", "equity", "equity_related", "secondary", "use_now", "pattern_equity")),
        (r"shares|stock|eps|earningspershare|sharebasedcompensation", ("share_metrics", "share_metrics", "share_related", "secondary", "use_now", "pattern_shares")),
        (r"dividend|repurchase", ("capital_allocation", "capital_returns", "capital_return_related", "secondary", "use_now", "pattern_capital_returns")),
        (r"netcash|cashprovided|cashused|capitalexpenditure|propertyplantandequipment", ("cash_flow", "cash_flow", "cash_flow_related", "core", "use_now", "pattern_cash_flow")),
        (r"tax|deferredtax", ("tax", "taxes", "tax_related", "secondary", "use_later", "pattern_tax")),
        (r"lease|rightofuse", ("operating_details", "leases", "lease_related", "long_tail", "use_later", "pattern_lease")),
        (r"fairvalue|derivative|hedg", ("risk_and_exposure", "market_exposure", "fair_value_or_derivative", "long_tail", "use_later", "pattern_exposure")),
        (r"goodwill|intangible", ("balance_sheet", "intangibles", "intangibles_related", "secondary", "use_now", "pattern_intangibles")),
        (r"inventory|receivable|payable", ("working_capital", "working_capital", "working_capital_related", "secondary", "use_now", "pattern_working_capital")),
        (r"segment|geograph|customer|supplier", ("operating_details", "business_mix", "segment_or_customer_related", "long_tail", "use_later", "pattern_business_mix")),
    ]
    for pattern, values in pattern_rules:
        if re.search(pattern, lower):
            statement_group, fact_group, canonical_metric, priority_tier, review_action, tag_reason = values
            return {
                "statement_group": statement_group,
                "fact_group": fact_group,
                "canonical_metric": canonical_metric,
                "priority_tier": priority_tier,
                "review_action": review_action,
                "tag_reason": tag_reason,
            }

    return {
        "statement_group": "other",
        "fact_group": "long_tail",
        "canonical_metric": "unmapped_long_tail",
        "priority_tier": "long_tail",
        "review_action": "review_later",
        "tag_reason": "fallback_unmapped",
    }


# ── Ticker universe & CIK resolution ────────────────────────────────────────

def _normalize_ticker(value: str) -> str:
    # SEC uses dashes for share classes (e.g. BRK-B), Robinhood exports use dots.
    return str(value).strip().upper().replace(".", "-")


def collect_holdings_tickers(transaction_files: list[Path]) -> set[str]:
    tickers: set[str] = set()
    for path in transaction_files:
        if not path.exists():
            print(f"  (skip, not found) {path.name}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "Instrument" not in df.columns:
            continue
        found = {
            _normalize_ticker(v)
            for v in df["Instrument"].dropna().tolist()
            if str(v).strip()
        }
        print(f"  {path.name}: {len(found)} instruments")
        tickers |= found
    return tickers


def collect_candidate_tickers() -> set[str]:
    if not BUY_CANDIDATE_UNIVERSE_PATH.exists():
        print(f"  (skip, not found) {BUY_CANDIDATE_UNIVERSE_PATH.name}")
        return set()
    df = pd.read_csv(BUY_CANDIDATE_UNIVERSE_PATH)
    col = next((c for c in ("ticker", "Ticker", "symbol", "Symbol") if c in df.columns), None)
    if col is None:
        print("  (skip) no ticker column in candidate universe")
        return set()
    found = {_normalize_ticker(v) for v in df[col].dropna().tolist() if str(v).strip()}
    print(f"  buy_candidate_universe.csv: {len(found)} tickers")
    return found


def resolve_ciks(tickers: set[str]) -> pd.DataFrame:
    """Map tickers to zero-padded CIKs using SEC's official ticker table."""
    payload = fetch_json(COMPANY_TICKERS_URL)
    table = pd.DataFrame(payload).T
    table["ticker"] = table["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    table["cik"] = table["cik_str"].astype(str).str.zfill(10)
    table = table.rename(columns={"title": "company_name"})
    matched = table[table["ticker"].isin(tickers)][["cik", "ticker", "company_name"]].copy()
    matched = matched.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    missing = sorted(tickers - set(matched["ticker"]))
    if missing:
        print(f"  no SEC CIK for {len(missing)} tickers (ETFs/foreign/ADRs are expected): "
              f"{', '.join(missing[:25])}{' ...' if len(missing) > 25 else ''}")
    return matched


# ── Fact fetching & flattening (mirrors notebook cell 10) ───────────────────

def fetch_company_facts(company_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict] = []
    status_rows: list[dict] = []
    total = len(company_master)
    for i, row in enumerate(company_master.itertuples(index=False), start=1):
        url = COMPANY_FACTS_URL.format(cik=row.cik)
        try:
            facts = fetch_json(url)
            us_gaap = facts.get("facts", {}).get("us-gaap", {})
            status_rows.append({
                "ticker": row.ticker, "company_name": row.company_name, "cik": row.cik,
                "facts_status": "available", "fact_tag_count": len(us_gaap),
            })
            for fact_tag, payload in us_gaap.items():
                for unit_name, observations in payload.get("units", {}).items():
                    for obs in observations:
                        detail_rows.append({
                            "ticker": row.ticker,
                            "company_name": row.company_name,
                            "cik": row.cik,
                            "fact_tag": fact_tag,
                            "unit": unit_name,
                            "value": obs.get("val"),
                            "filed_date": obs.get("filed"),
                            "fiscal_year": obs.get("fy"),
                            "fiscal_period": obs.get("fp"),
                            "frame": obs.get("frame"),
                            "form": obs.get("form"),
                            "fact_start": obs.get("start"),
                            "fact_end": obs.get("end"),
                        })
            print(f"[{i}/{total}] {row.ticker}: {len(us_gaap)} tags")
        except HTTPError as error:
            status_rows.append({
                "ticker": row.ticker, "company_name": row.company_name, "cik": row.cik,
                "facts_status": f"http_{error.code}", "fact_tag_count": 0,
            })
            print(f"[{i}/{total}] {row.ticker}: no facts (HTTP {error.code})")
        except Exception as error:  # network hiccup — keep going, report at end
            status_rows.append({
                "ticker": row.ticker, "company_name": row.company_name, "cik": row.cik,
                "facts_status": f"error_{type(error).__name__}", "fact_tag_count": 0,
            })
            print(f"[{i}/{total}] {row.ticker}: error {error}")
        time.sleep(REQUEST_DELAY_SECONDS)

    facts_df = pd.DataFrame(detail_rows)
    if not facts_df.empty:
        meta = facts_df["fact_tag"].apply(classify_fact_tag).apply(pd.Series)
        facts_df = pd.concat([facts_df, meta], axis=1)
    return facts_df, pd.DataFrame(status_rows)


# ── Entry point ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild SEC company_facts.csv for a wider ticker universe.")
    parser.add_argument("--with-candidates", action="store_true",
                        help="also include the buy-candidate universe")
    parser.add_argument("--tickers", nargs="+", metavar="SYM",
                        help="explicit ticker list (overrides holdings/candidate discovery)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap number of tickers (useful for a test run)")
    parser.add_argument("--full", action="store_true",
                        help="keep every SEC fact tag instead of only the fields the app uses "
                             "(much larger file; default keeps the app-relevant subset)")
    args = parser.parse_args()

    print("Resolving ticker universe...")
    if args.tickers:
        tickers = {_normalize_ticker(t) for t in args.tickers}
    else:
        tickers = collect_holdings_tickers(DEFAULT_TRANSACTION_FILES)
        if args.with_candidates:
            tickers |= collect_candidate_tickers()
    if not tickers:
        print("No tickers found. Provide --tickers or check the transaction files.")
        return 1
    print(f"Universe: {len(tickers)} distinct tickers")

    print("\nResolving CIKs from SEC...")
    company_master = resolve_ciks(tickers)
    if args.limit:
        company_master = company_master.head(args.limit)
    if company_master.empty:
        print("No CIKs resolved; nothing to fetch.")
        return 1
    print(f"Fetching facts for {len(company_master)} companies "
          f"(User-Agent: {SEC_HEADERS['User-Agent']})\n")

    facts_df, status_df = fetch_company_facts(company_master)
    if facts_df.empty:
        print("\nNo facts fetched. Aborting without overwriting existing files.")
        return 1

    raw_row_count = len(facts_df)
    if not args.full:
        facts_df = filter_to_relevant(facts_df)
        print(f"\nFiltered to app-relevant facts: {len(facts_df):,} of {raw_row_count:,} rows "
              f"({len(facts_df) / raw_row_count * 100:.0f}%). Use --full to keep everything.")

    EXTERNAL_SEC_DIR.mkdir(parents=True, exist_ok=True)
    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    company_master.to_csv(EXTERNAL_SEC_DIR / "company_master.csv", index=False)
    status_df.to_csv(EXTERNAL_SEC_DIR / "company_facts_review.csv", index=False)
    facts_df.to_csv(EXTERNAL_SEC_DIR / "company_facts_full.csv", index=False)
    facts_df.to_csv(STRUCTURED_DIR / "company_facts.csv", index=False)

    covered = sorted(facts_df["ticker"].unique())
    print(f"\nDone. {len(facts_df):,} fact rows across {len(covered)} tickers.")
    print(f"Wrote: {STRUCTURED_DIR / 'company_facts.csv'}")
    print(f"       {EXTERNAL_SEC_DIR / 'company_facts_full.csv'}")
    failed = status_df[status_df["facts_status"] != "available"]
    if not failed.empty:
        print(f"{len(failed)} tickers returned no facts: "
              f"{', '.join(failed['ticker'].tolist()[:30])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

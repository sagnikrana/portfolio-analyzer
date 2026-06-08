"""Point-in-time (PIT) S&P 500 membership — Phase 1, free & survivorship-aware.

Backtests must ask "who was investable AS OF the cutoff date," not "who is in the
index today." Using today's members for past dates silently deletes the losers
(names later dropped/delisted) and massively inflates any concentrated strategy —
it cut a momentum backtest's apparent net alpha from +50% (today's members) to
+10% (PIT members). ~80% of that "edge" was survivorship bias.

Data: historical S&P 500 constituents (ticker, start_date, end_date) from the
public fja05680/sp500 dataset, stored at data/external/sp500_membership.csv.

Residual gap (Phase 2): yfinance lacks prices for some delisted/renamed names, so
a pick that later died still can't be fully measured. A paid delisted-inclusive
source (Sharadar/Polygon/Norgate) would close that.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

MEMBERSHIP_PATH = Path(__file__).resolve().parents[1] / "data" / "external" / "sp500_membership.csv"


@lru_cache(maxsize=1)
def load_membership() -> pd.DataFrame:
    df = pd.read_csv(MEMBERSHIP_PATH)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")  # NaT = still a member
    return df


def members_as_of(as_of_date) -> list[str]:
    """S&P 500 tickers that were members on `as_of_date` (PIT, survivorship-free)."""
    d = pd.Timestamp(as_of_date)
    df = load_membership()
    m = df[(df["start_date"] <= d) & (df["end_date"].isna() | (df["end_date"] >= d))]
    return sorted(set(m["ticker"].astype(str)))


if __name__ == "__main__":
    for d in ("2022-01-01", "2024-01-01", "2026-01-01"):
        print(f"{d}: {len(members_as_of(d))} S&P 500 members")

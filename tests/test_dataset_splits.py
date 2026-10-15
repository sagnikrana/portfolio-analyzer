"""Synthetic-dataset stock splits.

A split stock keeps its pre-split share count unless the generator emits the SPL
share addition. Combined with yfinance's split-adjusted prices in analysis, a
missing split made a big winner (e.g. AVGO, +271%) look like a ~60% loss and get
wrongly flagged to trim. Pin the split history so it can't silently regress.
"""

from datetime import date

import pytest

from generate_fake_invest_dataset import SECURITIES_BY_TICKER


# Ticker -> at least these (effective date, ratio) splits must be modeled.
REQUIRED_SPLITS = {
    "AAPL": (date(2020, 8, 31), 4.0),
    "GOOGL": (date(2022, 7, 18), 20.0),
    "AMZN": (date(2022, 6, 6), 20.0),
    "AVGO": (date(2024, 7, 15), 10.0),
    "NVDA": (date(2024, 6, 10), 10.0),
    "TSLA": (date(2022, 8, 25), 3.0),
}


@pytest.mark.parametrize("ticker, required", REQUIRED_SPLITS.items())
def test_split_history_present(ticker, required):
    sec = SECURITIES_BY_TICKER[ticker]
    assert sec.splits, f"{ticker} is a split stock but has no splits modeled"
    assert required in sec.splits, f"{ticker} missing split {required}; has {sec.splits}"


def test_nvda_has_both_splits():
    # NVDA split twice over the window (4:1 in 2021, 10:1 in 2024).
    nvda = SECURITIES_BY_TICKER["NVDA"]
    assert (date(2021, 7, 20), 4.0) in nvda.splits
    assert (date(2024, 6, 10), 10.0) in nvda.splits


def test_non_split_names_have_none():
    # Sanity: a name that never split carries no split history.
    assert SECURITIES_BY_TICKER["META"].splits == ()

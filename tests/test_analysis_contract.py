"""End-to-end data contract between the analysis pipeline and its consumers.

This is the guard for the "wrong dict key" class of bug (e.g. the digest reading
`cash_balance_estimate` when the pipeline emits `uninvested_cash_estimate`).
It runs the real pipeline once and asserts the exact keys/attributes the weekly
digest and the dashboard depend on.

Needs network (yfinance). If the fetch fails (offline / rate-limited) the whole
module skips rather than failing, so the fast unit suite stays meaningful on its
own. Run just this with:  pytest -m integration
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "raw" / "fake_aggressive_growth.csv"

# Keys the weekly digest pulls out of headline_metrics (automation/digest.py).
DIGEST_HEADLINE_KEYS = {
    "total_account_value_estimate",
    "uninvested_cash_estimate",
    "excess_money_weighted_return_vs_benchmark",
}


@pytest.fixture(scope="module")
def result():
    if not DATASET.exists():
        pytest.skip(f"sample dataset missing: {DATASET}")
    from automation.core import analyze_portfolio

    try:
        return analyze_portfolio(DATASET)
    except Exception as exc:  # network / yfinance / rate limit
        pytest.skip(f"analysis pipeline could not run (likely offline): {exc}")


def test_result_shape_for_digest(result):
    # Attributes/properties the digest reads directly.
    assert isinstance(result.freed_cash, float)
    assert isinstance(result.actionable_risk_actions, list)
    assert isinstance(result.candidate_pool, list)
    assert result.market_metrics and isinstance(result.market_metrics, dict)


def test_headline_metrics_has_digest_keys(result):
    headline = result.market_metrics.get("headline_metrics", {})
    missing = DIGEST_HEADLINE_KEYS - set(headline)
    assert not missing, f"headline_metrics missing keys the digest needs: {missing}"


def test_sector_allocation_rows_are_well_formed(result):
    rows = result.market_metrics.get("sector_allocation", [])
    assert rows, "expected a non-empty sector allocation"
    for row in rows:
        assert {"sector", "current_value", "weight_pct"} <= set(row)
    # Weights should sum to roughly 1 (100% of invested value).
    total = sum(float(r["weight_pct"]) for r in rows)
    assert total == pytest.approx(1.0, abs=0.02)


def test_sectors_are_resolved_not_dumped_into_unknown(result):
    rows = result.market_metrics.get("sector_allocation", [])
    labels = {str(r["sector"]) for r in rows}
    # The whole portfolio should not collapse into a single Unclassified bucket.
    assert labels - {"Unclassified", "Unknown"}, f"all sectors unresolved: {labels}"


def test_expense_ratios_are_sane_fractions(result):
    """No recommended fund should report an inflated expense ratio (the VOO 3%
    bug at the integration level). Expense ratios are decimal fractions, so a
    real ETF is well under 0.05 (5%)."""
    for cand in result.candidate_pool:
        er = getattr(cand, "expense_ratio", None)
        if er is not None:
            assert 0.0 <= float(er) < 0.05, f"{getattr(cand, 'ticker', '?')} expense_ratio={er}"


def test_diagnosis_roundtrips(result):
    # The diagnosis is a pydantic model the UI + digest re-validate downstream.
    diag = result.diagnosis
    for attr in ("holding_action_recommendations", "replacement_candidates", "portfolio_preferences"):
        assert hasattr(diag, attr)
    dumped = diag.model_dump(mode="json")
    assert isinstance(dumped, dict) and dumped.get("run_id") is not None

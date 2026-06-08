"""Headless analysis core shared by the Gradio app and the weekly agent.

`analyze_portfolio(csv_path)` runs the same pipeline the dashboard runs
(load → lots → summary → market data → enriched metrics → risk diagnosis →
buy candidates) and returns a plain `AnalysisResult`, with no Gradio coupling.
The weekly job calls this; the app keeps its own UI wiring.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

# Make the portfolio_analyzer package importable when run as a standalone script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio_analyzer.app import (  # noqa: E402
    BENCHMARK_SYMBOL,
    PROJECTION_YEARS,
    build_lot_analytics,
    build_market_enriched_metrics,
    fetch_market_data,
    load_transactions,
    save_diagnosis_artifacts,
    summarize_portfolio,
)
from portfolio_analyzer.diagnosis import (  # noqa: E402
    PortfolioRiskDiagnosis,
    ReplacementCandidate,
    portfolio_risk_diagnosis_from_saved_artifacts,
    replacement_candidates_from_user_preferences,
)

# Pull a wide candidate pool so the digest can enforce a blend (e.g. 3 ETF / 7
# stock) and still have room after filtering.
DEFAULT_BUY_POOL_LIMIT = 40


@dataclass
class AnalysisResult:
    """Everything the weekly digest needs, decoupled from the UI."""

    as_of: date
    transactions: Any
    portfolio_summary: dict[str, Any]
    market_metrics: dict[str, Any]
    market_data: dict[str, Any]
    diagnosis: PortfolioRiskDiagnosis
    candidate_pool: list[ReplacementCandidate]

    @property
    def actionable_risk_actions(self) -> list[Any]:
        """Sell/trim recommendations that are actually actionable, with $ to move."""
        items = getattr(self.diagnosis, "holding_action_recommendations", []) or []
        return [
            item for item in items
            if getattr(item, "is_actionable", False) and float(getattr(item, "value_to_sell", 0) or 0) > 0
        ]

    @property
    def freed_cash(self) -> float:
        """Cash freed by following the risk actions (no uninvested cash added)."""
        return float(sum(float(item.value_to_sell or 0.0) for item in self.actionable_risk_actions))


def analyze_portfolio(
    csv_path: str | Path,
    *,
    risk_profile: int = 60,
    buy_pool_limit: int = DEFAULT_BUY_POOL_LIMIT,
    dataset_source: str = "weekly agent",
) -> AnalysisResult:
    """Run the full diagnosis pipeline on a transactions CSV and return results.

    Mirrors `_build_backtest_diagnosis` but for the *current* date (full file,
    latest prices). Rebuilds the buy candidates with a wide pool that allows both
    ETFs and individual stocks so the digest can compose a deliberate blend.
    """
    csv_path = Path(csv_path)
    transactions = load_transactions(csv_path)
    if transactions is None or transactions.empty:
        raise ValueError(f"No valid transactions found in {csv_path}")

    lot_data = build_lot_analytics(transactions)
    portfolio_summary = summarize_portfolio(transactions, lot_data)
    traded_symbols = sorted(
        set(transactions.loc[transactions["Instrument"].ne(""), "Instrument"].tolist())
    )
    market_data = fetch_market_data(
        traded_symbols=traded_symbols,
        benchmark_symbol=BENCHMARK_SYMBOL,
        start_date=portfolio_summary["date_range"]["start"],
    )
    market_metrics = build_market_enriched_metrics(
        df=transactions,
        portfolio_summary=portfolio_summary,
        lot_data=lot_data,
        market_data=market_data,
        benchmark_symbol=BENCHMARK_SYMBOL,
        projection_years=PROJECTION_YEARS,
        stated_risk_score=risk_profile,
    )
    diagnosis_paths = save_diagnosis_artifacts(
        csv_path=csv_path,
        dataset_source=dataset_source,
        risk_profile=risk_profile,
        transactions=transactions,
        portfolio_summary=portfolio_summary,
        market_metrics=market_metrics,
    )
    diagnosis = portfolio_risk_diagnosis_from_saved_artifacts(Path(diagnosis_paths["latest_dir"]))

    # Rebuild candidates as a wide pool allowing both vehicle types, so the digest
    # can pick a deliberate ETF + individual-stock blend from real options.
    candidate_pool: list[ReplacementCandidate] = []
    base_prefs = getattr(diagnosis, "portfolio_preferences", None)
    if base_prefs is not None:
        update = {
            "buy_idea_limit": buy_pool_limit,
            "allow_etfs": True,
            "allow_single_stocks": True,
            "include_existing_holdings": False,
        }
        pool_prefs = (
            base_prefs.model_copy(update=update)
            if hasattr(base_prefs, "model_copy")
            else base_prefs.copy(update=update)
        )
        candidate_pool = replacement_candidates_from_user_preferences(
            diagnosis=diagnosis, preferences=pool_prefs
        )
        diagnosis.portfolio_preferences = pool_prefs
    if not candidate_pool:
        candidate_pool = list(getattr(diagnosis, "replacement_candidates", []) or [])

    return AnalysisResult(
        as_of=date.today(),
        transactions=transactions,
        portfolio_summary=portfolio_summary,
        market_metrics=market_metrics,
        market_data=market_data,
        diagnosis=diagnosis,
        candidate_pool=candidate_pool,
    )


if __name__ == "__main__":  # quick manual smoke test
    import argparse

    ap = argparse.ArgumentParser(description="Run the headless portfolio analysis core.")
    ap.add_argument("csv", help="path to a transactions CSV")
    ap.add_argument("--risk", type=int, default=60)
    args = ap.parse_args()

    result = analyze_portfolio(args.csv, risk_profile=args.risk)
    print(f"as_of: {result.as_of}")
    print(f"risk actions (actionable): {len(result.actionable_risk_actions)}")
    print(f"freed cash to redeploy: ${result.freed_cash:,.2f}")
    print(f"candidate pool size: {len(result.candidate_pool)}")
    etfs = [c for c in result.candidate_pool if (c.asset_type or '').upper() == 'ETF']
    stocks = [c for c in result.candidate_pool if (c.asset_type or '').upper() != 'ETF']
    print(f"  pool ETFs: {len(etfs)} | pool stocks: {len(stocks)}")

"""Validation harness — the honest scorecard for the recommendation engine.

The single rule it enforces: a recommendation must beat a cheap S&P 500 index
fund, FAIRLY, before we trust it. "Fairly" means:

  1. No look-ahead — candidates are scored as of each cutoff (the engine's
     `as_of_returns` path), never with present-day data.
  2. Real costs — a round-trip transaction cost is charged on the moved sleeve.
  3. Benchmarked — every window is compared to the S&P 500 over the same window.
  4. Fixed forward horizon — each cutoff is evaluated over the SAME horizon
     (default 12 months), so windows are comparable instead of one overlapping blob.

Output: a per-window table + a PASS/FAIL verdict on whether the buy picks beat
the index net of costs. Run this before shipping any recommendation change.

    python -m automation.validation                 # default 2022Q1..2025Q2, 12mo
    python -m automation.validation --horizon 6 --cost-bps 10
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio_analyzer.app import (  # noqa: E402
    BACKTEST_BUY_IDEA_COUNT,
    BENCHMARK_SYMBOL,
    BUY_CANDIDATE_UNIVERSE_PATH,
    _backtest_candidate_weights,
    _build_backtest_diagnosis,
    _candidate_returns_as_of,
    _price_at_or_after,
    _price_at_or_before,
    fetch_market_data,
    load_buy_candidate_universe,
    load_transactions,
)
from portfolio_analyzer.diagnosis import replacement_candidates_from_user_preferences  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "mantis_invest.csv"


def _eligible_universe_symbols() -> dict[str, str]:
    entries = load_buy_candidate_universe(BUY_CANDIDATE_UNIVERSE_PATH)
    return {
        e.ticker: (getattr(e, "market_data_symbol", None) or e.ticker)
        for e in entries
        if getattr(e, "eligible_for_buy_engine", True)
    }


def evaluate_cutoff(
    csv_path: Path,
    transactions: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_end: pd.Timestamp,
    universe_symbols: dict[str, str],
    cost_bps: float,
) -> dict | None:
    """Return the picks' vs-S&P result over [cutoff, horizon_end], net of costs.

    Picks are scored AS OF the cutoff (no look-ahead) and held to horizon_end.
    """
    cut_tx = transactions.loc[transactions["Activity Date"] <= cutoff].copy()
    if cut_tx.empty:
        return None
    try:
        diag, _mm, _cmd, _ps = _build_backtest_diagnosis(
            csv_path=csv_path, dataset_source="validation",
            cutoff_transactions=cut_tx, risk_profile=60, cutoff_date=cutoff,
        )
    except Exception:
        return None
    base_prefs = getattr(diag, "portfolio_preferences", None)
    if base_prefs is None:
        return None
    prefs = base_prefs.model_copy(update={"buy_idea_limit": BACKTEST_BUY_IDEA_COUNT})
    as_of_returns = _candidate_returns_as_of(universe_symbols, cutoff)
    if not as_of_returns:
        return None
    cands = replacement_candidates_from_user_preferences(
        diagnosis=diag, preferences=prefs, as_of_returns=as_of_returns
    )
    weights = _backtest_candidate_weights(cands or [], limit=BACKTEST_BUY_IDEA_COUNT)
    if not weights:
        return None

    tickers = [c.ticker for c, _ in weights]
    md = fetch_market_data(traded_symbols=tickers, benchmark_symbol=BENCHMARK_SYMBOL,
                           start_date=cutoff.strftime("%Y-%m-%d"))
    close = md.get("ticker_close", pd.DataFrame())
    bench = md.get("benchmark_close", pd.Series(dtype=float))
    if close is None or close.empty or bench is None or bench.empty:
        return None
    close = close.copy(); close.index = pd.to_datetime(close.index); close = close.sort_index().ffill()
    bench = bench.copy(); bench.index = pd.to_datetime(bench.index); bench = bench.sort_index().ffill()

    start_val = end_val = 0.0
    for c, w in weights:
        if c.ticker not in close.columns:
            continue
        _, sp = _price_at_or_after(close[c.ticker], cutoff)
        _, ep = _price_at_or_before(close[c.ticker], horizon_end)
        if not sp or not ep or sp <= 0:
            continue
        start_val += float(w); end_val += float(w) * (ep / sp)
    if start_val <= 0:
        return None
    pick_ret = end_val / start_val - 1.0

    _, bs = _price_at_or_after(bench, cutoff)
    _, be = _price_at_or_before(bench, horizon_end)
    if not bs or not be or bs <= 0:
        return None
    sp_ret = be / bs - 1.0

    cost = 2.0 * (cost_bps / 10000.0)  # round trip (buy + eventual sell)
    net_alpha = pick_ret - sp_ret - cost
    return {
        "cutoff": cutoff.date(), "horizon_end": horizon_end.date(),
        "n_picks": len(weights), "pick_return": pick_ret, "sp_return": sp_ret,
        "cost": cost, "net_alpha": net_alpha, "beat_index": net_alpha > 0,
    }


def run_scorecard(
    csv_path: Path = DEFAULT_CSV,
    start: str = "2022-01-01",
    end: str = "2025-06-30",
    freq: str = "QS",
    horizon_months: int = 12,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    transactions = load_transactions(csv_path).sort_values("Activity Date")
    universe_symbols = _eligible_universe_symbols()
    today = pd.Timestamp.today().normalize()
    rows = []
    for cutoff in pd.date_range(start, end, freq=freq):
        cutoff = pd.Timestamp(cutoff).normalize()
        horizon_end = cutoff + pd.DateOffset(months=horizon_months)
        if horizon_end > today:
            continue  # not enough forward data yet — don't evaluate a partial window
        r = evaluate_cutoff(csv_path, transactions, cutoff, horizon_end, universe_symbols, cost_bps)
        if r is None:
            print(f"{cutoff.date()}: skipped (no actionable picks / data)")
            continue
        rows.append(r)
        print(f"{r['cutoff']} +{horizon_months}mo  picks {r['pick_return']:+.1%}  "
              f"S&P {r['sp_return']:+.1%}  net alpha {r['net_alpha']:+.1%}  "
              f"{'BEAT' if r['beat_index'] else 'LAGGED'}")

    df = pd.DataFrame(rows)
    print("\n================= SCORECARD =================")
    print(f"windows evaluated: {len(df)}  | horizon {horizon_months}mo | round-trip cost {2*cost_bps:.0f}bps")
    if len(df):
        beat = df["beat_index"].mean()
        avg_alpha = df["net_alpha"].mean()
        med_alpha = df["net_alpha"].median()
        print(f"picks beat the index (net of costs): {beat:.0%} of windows")
        print(f"average net alpha vs S&P: {avg_alpha:+.1%}   (median {med_alpha:+.1%})")
        print(f"avg picks {df['pick_return'].mean():+.1%}  vs  avg S&P {df['sp_return'].mean():+.1%}")
        verdict = "PASS" if (avg_alpha > 0 and beat >= 0.5) else "FAIL"
        print(f"\nVERDICT: {verdict} — "
              + ("the buy picks beat a cheap index fund net of costs."
                 if verdict == "PASS"
                 else "the buy picks do NOT beat a cheap index fund net of costs; "
                      "do not present them as market-beating."))
        print("CAVEAT: overlapping windows + small single-portfolio sample, so this is "
              "DIRECTIONAL, not statistically significant. Check the per-window table for "
              "regime dependence (a value/quality tilt can beat in down years and lag in "
              "megacap-led bull years).")
    out = REPO_ROOT / "automation" / "state" / "scorecard.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved {out}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Honest validation scorecard for the buy recommendations.")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--freq", default="QS", help="pandas date_range freq (QS=quarterly, MS=monthly)")
    ap.add_argument("--horizon", type=int, default=12, help="forward horizon in months")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="one-way transaction cost in bps")
    args = ap.parse_args()
    run_scorecard(Path(args.csv), args.start, args.end, args.freq, args.horizon, args.cost_bps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

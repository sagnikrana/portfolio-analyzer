"""Weekly digest builder: turns an AnalysisResult into the email's content.

Composition:
  - Risk Actions: the actionable sell/trim recommendations (freed cash, no
    uninvested cash).
  - 10 buy suggestions: a deliberate 3-ETF / 7-stock blend chosen from the
    candidate pool, sized by redeploying the freed risk-action cash.
  - A blend-vs-S&P growth chart (rebased to 100), rendered to PNG.
  - An "explain & prioritize" summary written by a LOCAL Ollama agent
    (see risk_digest_agent) — no API key, data stays on the machine; falls back
    to a deterministic template only if Ollama is unreachable.
  - A week-over-week diff (what's new / dropped) via a local snapshot.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio_analyzer.app import BENCHMARK_SYMBOL, fetch_market_data  # noqa: E402

from automation.core import AnalysisResult  # noqa: E402

STATE_DIR = REPO_ROOT / "automation" / "state"
STATE_FILE = STATE_DIR / "last_digest.json"

N_ETF = 3
N_STOCK = 7
CHART_LOOKBACK_YEARS = 3

# Local Ollama agent — no API key, data never leaves the machine. Override the
# model with OLLAMA_MODEL (e.g. "llama3.3:latest" for higher quality).
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
OLLAMA_TIMEOUT_S = int(os.environ.get("OLLAMA_TIMEOUT_S", "300"))


# ── Data shapes ───────────────────────────────────────────────────────────────
@dataclass
class Pick:
    ticker: str
    name: str
    asset_type: str
    fit_score: float
    why: str
    allocation: float = 0.0
    weight: float = 0.0
    ret_1y: float | None = None
    ret_3y: float | None = None


@dataclass
class RiskAction:
    ticker: str
    label: str
    value_to_sell: float
    rel_vs_benchmark: float | None
    reason: str


@dataclass
class Digest:
    as_of: date
    freed_cash: float
    risk_actions: list[RiskAction]
    picks: list[Pick]
    chart_png: bytes | None
    summary: str
    diff: dict[str, Any] = field(default_factory=dict)


# ── Selection + sizing ────────────────────────────────────────────────────────
def _is_etf(candidate: Any) -> bool:
    return (getattr(candidate, "asset_type", "") or "").strip().upper() == "ETF"


def select_blend(pool: list[Any], n_etf: int = N_ETF, n_stock: int = N_STOCK) -> list[Any]:
    """Pick a deliberate ETF + individual-stock blend, best fit_score first.

    If one bucket is short, backfill from the other so we always return up to
    n_etf + n_stock names.
    """
    by_fit = sorted(pool, key=lambda c: float(getattr(c, "fit_score", 0) or 0), reverse=True)
    etfs = [c for c in by_fit if _is_etf(c)]
    stocks = [c for c in by_fit if not _is_etf(c)]
    chosen = etfs[:n_etf] + stocks[:n_stock]
    target = n_etf + n_stock
    if len(chosen) < target:
        chosen_ids = {id(c) for c in chosen}
        for c in by_fit:
            if id(c) not in chosen_ids:
                chosen.append(c)
                if len(chosen) >= target:
                    break
    return chosen[:target]


def _allocate(candidates: list[Any], freed_cash: float) -> list[Pick]:
    """Weight by fit_score; redeploy freed_cash across the blend (0 if none)."""
    scores = [max(float(getattr(c, "fit_score", 0) or 0), 0.0) for c in candidates]
    total = sum(scores) or float(len(candidates) or 1)
    picks: list[Pick] = []
    for c, s in zip(candidates, scores):
        w = (s / total) if total else (1.0 / len(candidates))
        picks.append(
            Pick(
                ticker=c.ticker,
                name=getattr(c, "security_name", c.ticker),
                asset_type=("ETF" if _is_etf(c) else "Stock"),
                fit_score=float(getattr(c, "fit_score", 0) or 0),
                why=getattr(c, "why_it_fits", "") or "",
                weight=w,
                allocation=round(freed_cash * w, 2),
                ret_1y=getattr(c, "relative_1y_return_pct", None),
                ret_3y=getattr(c, "relative_3y_return_pct", None),
            )
        )
    return picks


# ── Blend vs S&P growth chart ─────────────────────────────────────────────────
def build_blend_chart(picks: list[Pick], lookback_years: int = CHART_LOOKBACK_YEARS) -> bytes | None:
    """Rebased-to-100 growth of the weighted blend vs the S&P 500, as PNG bytes."""
    tickers = [p.ticker for p in picks]
    weights = {p.ticker: (p.weight or 0.0) for p in picks}
    if sum(weights.values()) <= 0:  # $0 week — show equal-weight illustration
        weights = {t: 1.0 / len(tickers) for t in tickers}

    start = (date.today() - timedelta(days=365 * lookback_years + 10)).strftime("%Y-%m-%d")
    try:
        md = fetch_market_data(traded_symbols=tickers, benchmark_symbol=BENCHMARK_SYMBOL, start_date=start)
    except Exception:
        return None
    closes = md.get("ticker_close", pd.DataFrame())
    bench = md.get("benchmark_close", pd.Series(dtype=float))
    if closes is None or closes.empty or bench is None or bench.empty:
        return None
    closes = closes.copy()
    closes.index = pd.to_datetime(closes.index)
    bench = bench.copy()
    bench.index = pd.to_datetime(bench.index)

    avail = [t for t in tickers if t in closes.columns]
    if not avail:
        return None
    sub = closes[avail].sort_index().ffill().dropna(how="all")
    sub = sub.dropna()
    if sub.empty:
        return None

    # Renormalize weights to available tickers; build weighted growth index.
    wsum = sum(weights[t] for t in avail) or 1.0
    blend = sum((sub[t] / sub[t].iloc[0]) * (weights[t] / wsum) for t in avail) * 100.0

    b = bench.sort_index().reindex(sub.index).ffill().dropna()
    common = blend.index.intersection(b.index)
    blend, b = blend.loc[common], b.loc[common]
    if len(common) < 2:
        return None
    b_index = (b / b.iloc[0]) * 100.0
    x = [d.strftime("%Y-%m-%d") for d in common]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=blend.tolist(), name="Suggested blend",
                             line=dict(color="#2563eb", width=2.5)))
    fig.add_trace(go.Scatter(x=x, y=b_index.tolist(), name="S&P 500",
                             line=dict(color="#94a3b8", width=2, dash="dash")))
    fig.update_layout(
        title={"text": "Suggested blend vs S&P 500 (growth, 100 = start)", "x": 0.5, "xanchor": "center"},
        xaxis_title="Date", yaxis_title="Growth since start (100 = start)",
        template="plotly_white", height=420, width=900,
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
    )
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None


# ── Week-over-week diff ───────────────────────────────────────────────────────
def compute_diff(risk_tickers: list[str], pick_tickers: list[str]) -> dict[str, Any]:
    prev = {}
    if STATE_FILE.exists():
        try:
            prev = json.loads(STATE_FILE.read_text())
        except Exception:
            prev = {}
    prev_risk = set(prev.get("risk_tickers", []))
    prev_picks = set(prev.get("pick_tickers", []))
    diff = {
        "first_run": not bool(prev),
        "new_risk": sorted(set(risk_tickers) - prev_risk),
        "cleared_risk": sorted(prev_risk - set(risk_tickers)),
        "new_picks": sorted(set(pick_tickers) - prev_picks),
        "dropped_picks": sorted(prev_picks - set(pick_tickers)),
        "prev_date": prev.get("as_of"),
    }
    return diff


def _save_state(as_of: date, risk_tickers: list[str], pick_tickers: list[str]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({
        "as_of": as_of.isoformat(),
        "risk_tickers": risk_tickers,
        "pick_tickers": pick_tickers,
    }, indent=2))


# ── The local Ollama "explain & prioritize" agent ─────────────────────────────
def _ollama_chat(system: str, user: str) -> str | None:
    """Call the local Ollama chat API. Returns text, or None on any failure."""
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {"temperature": 0.3},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("message", {}).get("content") or "").strip() or None
    except Exception:
        return None


def risk_digest_agent(
    *, as_of: date, freed_cash: float, risk_actions: list[RiskAction],
    picks: list[Pick], diff: dict[str, Any], headline: dict[str, Any],
) -> str:
    """Generate the prioritized plain-English weekly summary with a LOCAL Ollama
    model. Falls back to a deterministic template if Ollama is unreachable."""
    payload = {
        "as_of": as_of.isoformat(),
        "freed_cash_to_redeploy": round(freed_cash, 2),
        "uninvested_cash_used": False,
        "risk_actions": [
            {"ticker": r.ticker, "action": r.label, "dollars": round(r.value_to_sell, 2),
             "vs_sp500": r.rel_vs_benchmark, "reason": r.reason} for r in risk_actions
        ],
        "buy_blend": [
            {"ticker": p.ticker, "type": p.asset_type, "allocation": p.allocation,
             "fit": round(p.fit_score, 1), "rel_1y_vs_sp": p.ret_1y, "why": p.why} for p in picks
        ],
        "week_over_week": diff,
        "portfolio_headline": headline,
    }
    system = (
        "You are a portfolio-review assistant writing a concise WEEKLY email digest "
        "for the account owner. This is educational portfolio review, NOT investment, "
        "tax, or trading advice. Be direct and prioritized: lead with what CHANGED "
        "since last week (if week_over_week.first_run is true, say it's the first "
        "baseline digest instead of inventing changes), then the single most "
        "important risk action and why, then a one-line read on the suggested buy "
        "blend. "
        "CRITICAL: do NOT invent, round, or recompute any dollar amounts or "
        "percentages — the email shows exact figures in a table beneath your text. "
        "Refer to tickers and direction (trim/redeploy), not fabricated numbers. "
        "You may state the total freed cash only if you copy it exactly from the data. "
        "Output PLAIN TEXT only: no markdown, no bold/asterisks, no headers, no bullet "
        "characters. 150 words max. End with one short line: 'Educational review only — "
        "not investment advice.'"
    )
    user = (
        "Here is this week's structured analysis as JSON. Write the digest summary.\n\n"
        + json.dumps(payload, indent=2, default=str)
    )
    text = _ollama_chat(system, user)
    if text:
        return text
    # Ollama not running / model not pulled — keep the email working.
    return _template_summary(freed_cash, risk_actions, picks, diff)


def _template_summary(freed_cash, risk_actions, picks, diff) -> str:
    lines = []
    if diff.get("first_run"):
        lines.append("First weekly digest — baseline below.")
    else:
        chg = []
        if diff.get("new_risk"):
            chg.append("new risk flags: " + ", ".join(diff["new_risk"]))
        if diff.get("cleared_risk"):
            chg.append("cleared: " + ", ".join(diff["cleared_risk"]))
        if diff.get("new_picks"):
            chg.append("new picks: " + ", ".join(diff["new_picks"]))
        lines.append("Changes since last week — " + ("; ".join(chg) if chg else "none."))
    if risk_actions:
        top = max(risk_actions, key=lambda r: r.value_to_sell)
        lines.append(f"Top risk action: {top.label} {top.ticker} (~${top.value_to_sell:,.0f}). "
                     f"Freed cash to redeploy: ${freed_cash:,.0f} (no uninvested cash used).")
    else:
        lines.append("No actionable risk trims this week; no cash freed to redeploy.")
    if picks:
        names = ", ".join(f"{p.ticker}(${p.allocation:,.0f})" for p in picks[:5])
        lines.append(f"Suggested blend (3 ETF / 7 stock), top sizing: {names} …")
    lines.append("Educational portfolio review only — not investment advice.")
    return "\n".join(lines)


# ── Orchestration ─────────────────────────────────────────────────────────────
def build_digest(result: AnalysisResult, *, persist_state: bool = True) -> Digest:
    risk_actions = [
        RiskAction(
            ticker=r.ticker,
            label=getattr(r, "recommendation_label", "Trim"),
            value_to_sell=float(r.value_to_sell or 0.0),
            rel_vs_benchmark=getattr(r, "relative_performance_vs_benchmark", None),
            reason=getattr(r, "recommendation_summary", "") or "",
        )
        for r in result.actionable_risk_actions
    ]
    freed_cash = result.freed_cash
    blend = select_blend(result.candidate_pool)
    picks = _allocate(blend, freed_cash)

    risk_tickers = [r.ticker for r in risk_actions]
    pick_tickers = [p.ticker for p in picks]
    diff = compute_diff(risk_tickers, pick_tickers)

    headline = (result.market_metrics or {}).get("headline_metrics", {})
    headline_slim = {
        k: headline.get(k) for k in (
            "total_account_value_estimate", "cash_balance_estimate",
            "relative_performance_vs_benchmark", "analysis_end",
        ) if k in headline
    }

    chart_png = build_blend_chart(picks)
    summary = risk_digest_agent(
        as_of=result.as_of, freed_cash=freed_cash, risk_actions=risk_actions,
        picks=picks, diff=diff, headline=headline_slim,
    )

    if persist_state:
        _save_state(result.as_of, risk_tickers, pick_tickers)

    return Digest(
        as_of=result.as_of, freed_cash=freed_cash, risk_actions=risk_actions,
        picks=picks, chart_png=chart_png, summary=summary, diff=diff,
    )


if __name__ == "__main__":
    import argparse
    from automation.core import analyze_portfolio

    ap = argparse.ArgumentParser(description="Build the weekly digest from a CSV.")
    ap.add_argument("csv")
    ap.add_argument("--no-state", action="store_true", help="don't update the diff snapshot")
    args = ap.parse_args()

    res = analyze_portfolio(args.csv)
    dig = build_digest(res, persist_state=not args.no_state)
    print("as_of:", dig.as_of, "| freed cash:", f"${dig.freed_cash:,.2f}")
    print("risk actions:", len(dig.risk_actions))
    print("picks:", ", ".join(f"{p.ticker}[{p.asset_type}/${p.allocation:,.0f}]" for p in dig.picks))
    print("chart png bytes:", len(dig.chart_png) if dig.chart_png else "none")
    print("diff:", dig.diff)
    print("\n--- SUMMARY ---\n" + dig.summary)

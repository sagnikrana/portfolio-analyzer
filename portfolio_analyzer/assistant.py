"""Dashboard AI assistant — local Ollama, grounded in the latest analysis.

Two features wired into the Gradio app:
  - generate_plain_summary(): a plain-English, prioritized narrative of the
    current portfolio (button in the Next Steps tab).
  - chat_about_portfolio(message, history): a conversational agent that answers
    questions using the latest saved diagnosis as context ("Ask Your Portfolio").

Both run entirely on the local Ollama model (no API key) and degrade gracefully
if Ollama isn't running or no analysis has been run yet.
"""

from __future__ import annotations

from typing import Any, Optional

from portfolio_analyzer.diagnosis import portfolio_risk_diagnosis_from_saved_artifacts
from portfolio_analyzer.local_llm import OLLAMA_MODEL, ollama_available, ollama_chat

NO_ANALYSIS_MSG = (
    "I don't have an analysis to look at yet. Upload your CSV (or pick the bundled "
    "dataset) and click **Run Analysis** first, then ask me again."
)
NO_OLLAMA_MSG = (
    "The local AI (Ollama) isn't reachable. Start it with `ollama serve` and make "
    f"sure a model is pulled (current: `{OLLAMA_MODEL}`), then try again."
)


def _load_latest_diagnosis() -> Optional[Any]:
    """Load the most recent diagnosis the dashboard saved during Run Analysis."""
    try:
        from portfolio_analyzer.app import DIAGNOSIS_DIR  # lazy: avoid import cycle
        return portfolio_risk_diagnosis_from_saved_artifacts(DIAGNOSIS_DIR)
    except Exception:
        return None


def build_portfolio_context(diagnosis: Any, *, max_holdings: int = 12, max_buys: int = 10) -> str:
    """Compact, model-friendly text snapshot of the current portfolio analysis."""
    g = lambda obj, name, default=None: getattr(obj, name, default)
    lines: list[str] = []
    lines.append(
        f"Analysis window: {g(diagnosis, 'analysis_start')} to {g(diagnosis, 'analysis_end')} "
        f"(benchmark {g(diagnosis, 'benchmark_symbol')})."
    )
    lines.append(
        f"Observed risk: {g(diagnosis, 'observed_risk_score')} ({g(diagnosis, 'observed_risk_band')}); "
        f"stated risk: {g(diagnosis, 'stated_risk_score')} ({g(diagnosis, 'stated_risk_band')}); "
        f"alignment: {g(diagnosis, 'alignment')}."
    )
    summary = g(diagnosis, "diagnostic_summary")
    if summary:
        lines.append(f"Diagnostic summary: {summary}")

    holdings = g(diagnosis, "current_holdings", []) or []
    holdings = sorted(holdings, key=lambda h: float(g(h, "current_weight", 0) or 0), reverse=True)
    if holdings:
        lines.append("\nTop holdings (ticker, weight, value):")
        for h in holdings[:max_holdings]:
            w = g(h, "current_weight")
            v = g(h, "current_value")
            wt = f"{float(w) * 100:.1f}%" if isinstance(w, (int, float)) else "?"
            vt = f"${float(v):,.0f}" if isinstance(v, (int, float)) else "?"
            lines.append(f"  - {g(h, 'ticker')}: {wt}, {vt}")

    actions = [a for a in (g(diagnosis, "holding_action_recommendations", []) or [])
               if g(a, "is_actionable", False) and float(g(a, "value_to_sell", 0) or 0) > 0]
    if actions:
        lines.append("\nRisk actions (sell/trim recommendations):")
        for a in actions:
            lines.append(
                f"  - {g(a, 'recommendation_label', 'Trim')} {g(a, 'ticker')} "
                f"(~${float(g(a, 'value_to_sell', 0) or 0):,.0f}): {g(a, 'recommendation_summary', '')}"
            )
    else:
        lines.append("\nRisk actions: none actionable right now.")

    buys = g(diagnosis, "replacement_candidates", []) or []
    if buys:
        lines.append("\nBuy ideas (ticker, type, fit, why):")
        for c in buys[:max_buys]:
            lines.append(
                f"  - {g(c, 'ticker')} [{g(c, 'asset_type')}] fit {float(g(c, 'fit_score', 0) or 0):.0f}: "
                f"{g(c, 'why_it_fits', '')}"
            )
    return "\n".join(lines)


_DISCLAIMER = "Educational portfolio review only — not investment, tax, or trading advice."


def generate_plain_summary() -> str:
    """Plain-English, prioritized narrative of the current portfolio (markdown)."""
    diagnosis = _load_latest_diagnosis()
    if diagnosis is None:
        return NO_ANALYSIS_MSG
    if not ollama_available():
        return NO_OLLAMA_MSG
    context = build_portfolio_context(diagnosis)
    system = (
        "You are a portfolio-review assistant. Explain the user's portfolio in clear, "
        "prioritized plain English for a non-expert: 1) overall risk and whether it "
        "matches their stated risk, 2) the most important risk actions and why, "
        "3) the gist of the buy ideas. Be specific with tickers and dollars FROM THE "
        "CONTEXT ONLY — never invent numbers. Keep it under 220 words. End with one "
        "short line: '" + _DISCLAIMER + "'"
    )
    user = "Here is the current analysis:\n\n" + context + "\n\nWrite the summary."
    text = ollama_chat(system, user)
    return text or (
        "The local AI returned nothing this time — please try again. "
        "(Model: " + OLLAMA_MODEL + ")"
    )


def chat_about_portfolio(message: str, history: list) -> tuple[str, list]:
    """Conversational agent grounded in the latest analysis.

    `history` is Gradio Chatbot format: a list of [user, assistant] pairs.
    Returns ("", updated_history) so the input box clears.
    """
    message = (message or "").strip()
    if not message:
        return "", history
    history = list(history or [])

    diagnosis = _load_latest_diagnosis()
    if diagnosis is None:
        return "", history + [[message, NO_ANALYSIS_MSG]]
    if not ollama_available():
        return "", history + [[message, NO_OLLAMA_MSG]]

    context = build_portfolio_context(diagnosis)
    system = (
        "You are a helpful assistant answering questions about THIS user's portfolio. "
        "Use only the analysis context below; if something isn't in it, say you don't "
        "have that data rather than guessing. Never invent numbers. Be concise and "
        "concrete. This is educational portfolio review, not investment advice.\n\n"
        "=== PORTFOLIO ANALYSIS CONTEXT ===\n" + context
    )
    oll_history: list[dict] = []
    for turn in history:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            u, b = turn
            if u:
                oll_history.append({"role": "user", "content": str(u)})
            if b:
                oll_history.append({"role": "assistant", "content": str(b)})

    answer = ollama_chat(system, message, history=oll_history)
    if not answer:
        answer = "Sorry — the local AI didn't respond that time. Please try again."
    return "", history + [[message, answer]]

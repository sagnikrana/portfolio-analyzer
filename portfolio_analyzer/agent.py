"""Local, tool-using portfolio agent (conversational analyst + what-if sims).

Design: the buy/sell decisions stay in the deterministic engine. This agent only
*explains* and *simulates*, and it can only answer by calling tools that read the
already-computed diagnosis — so it cannot invent numbers, prices, or
recommendations. Runs on the local Ollama model (no API key; data stays local).

The tools operate on the in-session diagnosis (current_holdings, gaps, risk
actions, buy ideas), so every number the agent quotes is real and auditable.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable

from .local_llm import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_TIMEOUT_S

MAX_TOOL_STEPS = 6


# ── helpers over the diagnosis ────────────────────────────────────────────────
def _holdings(diagnosis) -> list[dict[str, Any]]:
    rows = []
    for h in getattr(diagnosis, "current_holdings", []) or []:
        try:
            val = float(getattr(h, "current_value", 0) or 0)
        except (TypeError, ValueError):
            val = 0.0
        if val <= 0:
            continue
        rows.append({
            "ticker": getattr(h, "ticker", ""),
            "name": getattr(h, "security_name", "") or "",
            "sector": getattr(h, "sector", "") or "Unknown",
            "value": round(val, 2),
            "weight_pct": round(float(getattr(h, "current_weight", 0) or 0) * 100, 2),
            "excess_vs_sp500_pct": (
                round(float(getattr(h, "excess_return_vs_benchmark", 0) or 0) * 100, 1)
                if getattr(h, "excess_return_vs_benchmark", None) is not None else None
            ),
        })
    return rows


def _concentration_from_values(values_by_ticker: dict[str, float], sector_by_ticker: dict[str, str]) -> dict[str, Any]:
    total = sum(v for v in values_by_ticker.values() if v > 0)
    if total <= 0:
        return {"error": "no invested value"}
    weights = {t: v / total for t, v in values_by_ticker.items() if v > 0}
    ordered = sorted(weights.items(), key=lambda kv: -kv[1])
    hhi = sum(w * w for w in weights.values())
    sector_val: dict[str, float] = {}
    for t, v in values_by_ticker.items():
        if v > 0:
            sector_val[sector_by_ticker.get(t, "Unknown")] = sector_val.get(sector_by_ticker.get(t, "Unknown"), 0.0) + v
    sector_pct = {s: round(v / total * 100, 1) for s, v in sorted(sector_val.items(), key=lambda kv: -kv[1])}
    return {
        "invested_value": round(total, 2),
        "largest_position": {"ticker": ordered[0][0], "weight_pct": round(ordered[0][1] * 100, 1)} if ordered else None,
        "top_5_weight_pct": round(sum(w for _, w in ordered[:5]) * 100, 1),
        "effective_holdings": round(1 / hhi, 1) if hhi else None,
        "positions": len(weights),
        "sector_mix_pct": sector_pct,
    }


# ── tools (each returns a JSON-serializable dict grounded in the diagnosis) ────
def tool_portfolio_overview(diagnosis) -> dict[str, Any]:
    h = _holdings(diagnosis)
    vbt = {r["ticker"]: r["value"] for r in h}
    sbt = {r["ticker"]: r["sector"] for r in h}
    return _concentration_from_values(vbt, sbt)


def tool_list_holdings(diagnosis, top_n: int = 10) -> dict[str, Any]:
    h = sorted(_holdings(diagnosis), key=lambda r: -r["value"])
    return {"holdings": h[: max(1, min(int(top_n or 10), 50))], "total_positions": len(h)}


def tool_concentration(diagnosis) -> dict[str, Any]:
    return tool_portfolio_overview(diagnosis)


def tool_risk_actions(diagnosis) -> dict[str, Any]:
    acts = [a for a in (getattr(diagnosis, "holding_action_recommendations", []) or [])
            if getattr(a, "is_actionable", False) and float(getattr(a, "value_to_sell", 0) or 0) > 0]
    return {"actions": [{
        "ticker": a.ticker,
        "action": getattr(a, "recommendation_label", ""),
        "reduce_pct": round(float(getattr(a, "position_reduction_pct", 0) or 0) * 100, 0),
        "value_to_sell": round(float(getattr(a, "value_to_sell", 0) or 0), 2),
        "reason": getattr(a, "recommendation_summary", "") or "",
    } for a in acts], "count": len(acts)}


def tool_buy_ideas(diagnosis) -> dict[str, Any]:
    rc = getattr(diagnosis, "replacement_candidates", []) or []
    return {"buy_ideas": [{
        "ticker": c.ticker,
        "name": getattr(c, "security_name", ""),
        "fills_gap": getattr(c, "linked_gap_label", ""),
        "fit": getattr(c, "fit_band", None) or getattr(c, "fit_score", None),
        "why": getattr(c, "why_it_fits", ""),
    } for c in rc[:20]], "count": len(rc)}


def tool_portfolio_gaps(diagnosis) -> dict[str, Any]:
    gaps = getattr(diagnosis, "portfolio_gaps", []) or []
    return {"gaps": [{
        "key": g.gap_key, "label": g.label,
        "what_is_missing": getattr(g, "what_is_missing", ""),
        "severity": getattr(g, "severity_band", ""),
    } for g in gaps]}


def tool_whatif_move(diagnosis, amount_usd: float, from_ticker: str, to_ticker: str) -> dict[str, Any]:
    """Simulate moving `amount_usd` from one holding (or CASH) into another
    (existing holding, a new ticker, or CASH) and report the concentration shift."""
    h = _holdings(diagnosis)
    vbt = {r["ticker"]: r["value"] for r in h}
    sbt = {r["ticker"]: r["sector"] for r in h}
    try:
        amount = float(amount_usd)
    except (TypeError, ValueError):
        return {"error": "amount_usd must be a number"}
    if amount <= 0:
        return {"error": "amount_usd must be positive"}
    frm = str(from_ticker or "").strip().upper()
    to = str(to_ticker or "").strip().upper()
    before = _concentration_from_values(vbt, sbt)

    new = dict(vbt)
    if frm not in ("CASH", ""):
        if frm not in new:
            return {"error": f"{frm} is not a current holding", "available": sorted(new)[:30]}
        if amount > new[frm]:
            return {"error": f"{frm} only has ${new[frm]:,.0f}; cannot move ${amount:,.0f}"}
        new[frm] -= amount
        if new[frm] <= 0:
            new.pop(frm, None)
    if to not in ("CASH", ""):
        new[to] = new.get(to, 0.0) + amount
        sbt.setdefault(to, sbt.get(to, "Unknown"))
    after = _concentration_from_values(new, sbt)

    def _delta(k):
        try:
            return round(after.get(k, 0) - before.get(k, 0), 1)
        except TypeError:
            return None
    return {
        "move": f"${amount:,.0f} from {frm or 'CASH'} to {to or 'CASH'}",
        "before": {"largest_position": before.get("largest_position"), "top_5_weight_pct": before.get("top_5_weight_pct"),
                   "effective_holdings": before.get("effective_holdings"), "sector_mix_pct": before.get("sector_mix_pct")},
        "after": {"largest_position": after.get("largest_position"), "top_5_weight_pct": after.get("top_5_weight_pct"),
                  "effective_holdings": after.get("effective_holdings"), "sector_mix_pct": after.get("sector_mix_pct")},
        "change": {"top_5_weight_pct": _delta("top_5_weight_pct"), "effective_holdings": _delta("effective_holdings")},
    }


def _buy_candidates_module():
    try:
        from . import buy_candidates as bc
    except ImportError:  # pragma: no cover
        import buy_candidates as bc  # type: ignore
    return bc


def tool_recent_news(diagnosis, ticker: str) -> dict[str, Any]:
    """Recent news headlines (with source) for a ticker — for grounded research
    with citations. Live via yfinance, SEQUENTIAL (never concurrent)."""
    t = str(ticker or "").strip().upper()
    if not t:
        return {"error": "ticker required"}
    try:
        headlines = _buy_candidates_module().fetch_candidate_news_signals(t, limit=4)
    except Exception:
        headlines = []
    return {"ticker": t, "headlines": headlines or [], "note": "Cite these headline sources; do not invent news."}


def tool_fundamentals(diagnosis, ticker: str) -> dict[str, Any]:
    """A few fundamental data points for a ticker (valuation/profitability),
    sourced from the cached market metadata."""
    t = str(ticker or "").strip().upper()
    if not t:
        return {"error": "ticker required"}
    try:
        meta = _buy_candidates_module().fetch_candidate_market_metadata(t) or {}
    except Exception:
        meta = {}
    keep = ("sector", "industry", "trailing_pe", "forward_pe", "enterprise_to_ebitda",
            "ebitda_margin", "market_cap", "beta", "expense_ratio", "dividend_yield")
    out = {k: meta.get(k) for k in keep if meta.get(k) is not None}
    return {"ticker": t, "fundamentals": out or "no fundamental data available"}


TOOL_FUNCS: dict[str, Callable] = {
    "get_portfolio_overview": tool_portfolio_overview,
    "list_holdings": tool_list_holdings,
    "get_concentration": tool_concentration,
    "get_risk_actions": tool_risk_actions,
    "get_buy_ideas": tool_buy_ideas,
    "get_portfolio_gaps": tool_portfolio_gaps,
    "simulate_whatif_move": tool_whatif_move,
    "get_recent_news": tool_recent_news,
    "get_fundamentals": tool_fundamentals,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "get_portfolio_overview",
        "description": "Invested value, largest position, top-5 weight, effective # of holdings, and sector mix.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "list_holdings",
        "description": "List current holdings (ticker, value, weight, sector) sorted by value.",
        "parameters": {"type": "object", "properties": {"top_n": {"type": "integer", "description": "How many (default 10)"}}}}},
    {"type": "function", "function": {"name": "get_concentration",
        "description": "Concentration metrics: largest position, top-5 weight, effective holdings, sector mix.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_risk_actions",
        "description": "The deterministic engine's actionable sell/trim recommendations and dollar amounts.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_buy_ideas",
        "description": "Current buy ideas and which portfolio gap each one fills.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_portfolio_gaps",
        "description": "The portfolio gaps (what's missing) the engine identified from the user's risks.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "simulate_whatif_move",
        "description": "Simulate moving a dollar amount from one holding (or CASH) into another holding/new ticker/CASH; returns the before/after concentration and sector shift.",
        "parameters": {"type": "object", "properties": {
            "amount_usd": {"type": "number", "description": "Dollars to move"},
            "from_ticker": {"type": "string", "description": "Source holding ticker, or 'CASH'"},
            "to_ticker": {"type": "string", "description": "Destination ticker, or 'CASH'"},
        }, "required": ["amount_usd", "from_ticker", "to_ticker"]}}},
    {"type": "function", "function": {"name": "get_recent_news",
        "description": "Recent news headlines (with their source) for a ticker — use for research and CITE the sources.",
        "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}},
    {"type": "function", "function": {"name": "get_fundamentals",
        "description": "Valuation/profitability fundamentals for a ticker (P/E, EV/EBITDA, margins, etc.).",
        "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}},
]

SYSTEM_PROMPT = (
    "You are a portfolio analyst assistant for an educational portfolio-review app. "
    "Answer ONLY using the tools provided — never invent numbers, prices, holdings, or recommendations. "
    "The buy/sell/trim decisions come from a deterministic rule engine; you explain and simulate them, you do not change them. "
    "Call tools to get real figures, then answer in clear plain English with the actual numbers. "
    "For 'what if' questions, use simulate_whatif_move. "
    "To research a stock, call get_recent_news and get_fundamentals, then summarize the story and "
    "CITE the news source names you used (do not invent news or fundamentals). "
    "If a question can't be answered from the tools, say so. "
    "Keep answers concise. End with: 'Educational only, not investment advice.'"
)


def _ollama_chat_raw(messages: list[dict], tools: list[dict] | None = None, fmt: str | None = None) -> dict:
    payload = {"model": OLLAMA_MODEL, "stream": False, "keep_alive": OLLAMA_KEEP_ALIVE,
               "options": {"temperature": 0.2}, "messages": messages}
    if tools:
        payload["tools"] = tools
    if fmt:
        payload["format"] = fmt
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_json(text: str) -> Any:
    """Parse a JSON object from the model's text (tolerant of stray prose)."""
    import re
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
    return {}


def critique_buy_ideas(diagnosis) -> list[dict[str, Any]]:
    """Reflection / LLM-as-judge: audit the engine's buy ideas against the
    portfolio's gaps and concentration, flagging weak ones. Advisory only — it
    does NOT change the deterministic recommendations. Grounded in real numbers.
    """
    ideas = tool_buy_ideas(diagnosis).get("buy_ideas", [])
    if not ideas:
        return []
    payload = {
        "buy_ideas": ideas,
        "concentration": tool_concentration(diagnosis),
        "gaps": tool_portfolio_gaps(diagnosis).get("gaps", []),
    }
    system = (
        "You are a skeptical investment-committee reviewer auditing an engine's buy ideas. "
        "Use ONLY the supplied JSON (buy_ideas, the portfolio's concentration, and its gaps). "
        "For each idea give a verdict: 'solid' (clearly fills a real gap without worsening "
        "concentration), 'caution' (helps but with a caveat), or 'weak' (doesn't really fill its "
        "claimed gap, or piles into an already-heavy sector/position). Cite the actual numbers in a "
        "one-sentence concern. Do not invent data. "
        'Return JSON only: {"reviews":[{"ticker":"..","verdict":"solid|caution|weak","concern":".."}]}'
    )
    try:
        data = _ollama_chat_raw(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, default=str)}],
            fmt="json",
        )
        parsed = _extract_json(data.get("message", {}).get("content", ""))
        reviews = parsed.get("reviews", []) if isinstance(parsed, dict) else []
    except Exception:
        reviews = []
    by_ticker = {str(r.get("ticker", "")).upper(): r for r in reviews if isinstance(r, dict)}
    out: list[dict[str, Any]] = []
    for idea in ideas:
        r = by_ticker.get(str(idea.get("ticker", "")).upper(), {})
        verdict = str(r.get("verdict", "")).strip().lower()
        out.append({
            "ticker": idea.get("ticker", ""),
            "name": idea.get("name", ""),
            "gap": idea.get("fills_gap", ""),
            "verdict": verdict if verdict in ("solid", "caution", "weak") else "unrated",
            "concern": str(r.get("concern", "")).strip(),
        })
    return out


def run_portfolio_agent(message: str, history: list[dict], diagnosis) -> str:
    """Tool-using agent loop. `history` is prior [{role, content}] turns."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": message})

    try:
        for _ in range(MAX_TOOL_STEPS):
            data = _ollama_chat_raw(messages, tools=TOOL_SCHEMAS)
            msg = data.get("message", {}) or {}
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return (msg.get("content") or "").strip() or "I couldn't form an answer — try rephrasing."
            messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls})
            for call in tool_calls:
                fn = (call.get("function") or {})
                name = fn.get("name", "")
                args = fn.get("arguments", {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                func = TOOL_FUNCS.get(name)
                result = func(diagnosis, **args) if func else {"error": f"unknown tool {name}"}
                messages.append({"role": "tool", "name": name, "content": json.dumps(result, default=str)})
        # Ran out of steps — make a final call without tools to summarize.
        data = _ollama_chat_raw(messages)
        return (data.get("message", {}).get("content") or "").strip() or "I needed too many steps; please narrow the question."
    except Exception as exc:  # noqa: BLE001
        return f"The local AI is unavailable right now ({type(exc).__name__}). Make sure Ollama is running."

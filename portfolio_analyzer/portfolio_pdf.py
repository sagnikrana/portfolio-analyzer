"""Build a Portfolio Summary PDF (overview, risk actions, monthly performance,
execution summary, buy ideas) for ad-hoc emailing from the dashboard.

Pure-Python (reportlab + plotly/kaleido) so it works locally and on Hugging Face
without system libraries. Pixel-perfect screenshots of the live web UI aren't
feasible server-side, so each section is reproduced from the same underlying data
the dashboard renders, and the real benchmark chart is embedded as an image.

Layout notes: pages are LANDSCAPE (the monthly/execution tables are wide), and
every cell is wrapped in a Paragraph so long text wraps inside its column instead
of overflowing into the next one. Column widths are auto-sized to content.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = colors.HexColor("#0f172a")
SLATE = colors.HexColor("#475569")
BLUE = colors.HexColor("#2563eb")
LIGHT = colors.HexColor("#eef2f8")

# Landscape letter with 0.5" margins -> ~10in usable width.
PAGE_SIZE = landscape(letter)
MARGIN = 0.5 * inch
USABLE_WIDTH = PAGE_SIZE[0] - 2 * MARGIN

_CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=7.5, leading=9.5, textColor=NAVY)
_CELL_R = ParagraphStyle("cellR", parent=_CELL, alignment=TA_RIGHT)
_HEAD = ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=7.5, leading=9.5, textColor=colors.white)
_HEAD_R = ParagraphStyle("headR", parent=_HEAD, alignment=TA_RIGHT)


def _money(v: Any) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _pct(v: Any) -> str:
    try:
        return f"{float(v):.2f}%"
    except (TypeError, ValueError):
        return "—"


def _looks_numeric(text: str) -> bool:
    t = str(text).strip().lstrip("$(-").rstrip("%)")
    return bool(t) and t.replace(",", "").replace(".", "").isdigit()


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("PA_H1", parent=ss["Heading1"], textColor=NAVY, fontSize=20, spaceAfter=4))
    ss.add(ParagraphStyle("PA_H2", parent=ss["Heading2"], textColor=BLUE, fontSize=13, spaceBefore=14, spaceAfter=6))
    ss.add(ParagraphStyle("PA_Body", parent=ss["BodyText"], textColor=SLATE, fontSize=9, leading=12))
    ss.add(ParagraphStyle("PA_Small", parent=ss["BodyText"], textColor=SLATE, fontSize=7.5, leading=10))
    return ss


def _benchmark_png(timeseries: list[dict[str, Any]]) -> bytes | None:
    """Render the portfolio-vs-S&P line chart to a PNG via plotly/kaleido."""
    if not timeseries:
        return None
    try:
        import plotly.graph_objects as go

        df = pd.DataFrame(timeseries)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        fig = go.Figure()
        if "account_value" in df:
            fig.add_trace(go.Scatter(x=df["date"], y=df["account_value"],
                                     name="Invested Portfolio", line=dict(color="#2563eb", width=2)))
        if "benchmark_value" in df:
            fig.add_trace(go.Scatter(x=df["date"], y=df["benchmark_value"],
                                     name="Trade-Matched S&P 500", line=dict(color="#f59e0b", width=2)))
        fig.update_layout(
            title="Portfolio vs S&P 500 Over Your Investing Horizon",
            template="plotly_white", height=360, width=900,
            margin=dict(l=55, r=20, t=50, b=40),
            legend=dict(orientation="h", y=1.08, x=0),
            yaxis_title="Value ($)", xaxis_title="Date",
        )
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None


def _auto_widths(rows: list[list[str]], usable: float) -> list[float]:
    """Size columns by max content length (capped so one long column can't starve
    the rest — the Paragraph wrapping handles overflow)."""
    ncol = max((len(r) for r in rows), default=1)
    maxlen = [1.0] * ncol
    for r in rows:
        for i, c in enumerate(r):
            maxlen[i] = max(maxlen[i], min(len(str(c)), 28))
    total = sum(maxlen) or 1.0
    return [usable * m / total for m in maxlen]


def _table(rows: list[list[str]], usable: float = USABLE_WIDTH, *, header: bool = True) -> Table:
    """Build a Table where every cell is a wrapped Paragraph (no overflow)."""
    col_widths = _auto_widths(rows, usable)

    def cellify(r_idx: int, c_idx: int, value: str):
        text = "" if value is None else str(value)
        numeric = _looks_numeric(text)
        if header and r_idx == 0:
            return Paragraph(text, _HEAD_R if numeric else _HEAD)
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(safe, _CELL_R if numeric else _CELL)

    data = [[cellify(ri, ci, c) for ci, c in enumerate(r)] for ri, r in enumerate(rows)]
    t = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbe2ea")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    t.setStyle(TableStyle(style))
    return t


def _overview_metrics(market_metrics: dict, portfolio_summary: dict) -> list[list[str]]:
    h = market_metrics.get("headline_metrics", {}) or {}
    rs = market_metrics.get("risk_score")
    if isinstance(rs, dict):
        score, band = rs.get("score"), rs.get("band")
        risk_str = f"{score}/100 · {band}" if score is not None else (band or "—")
    else:
        risk_str = str(rs) if rs is not None else "—"
    excess = h.get("excess_money_weighted_return_vs_benchmark")
    pairs = [
        ("Analysis window", f"{h.get('analysis_start', '?')} → {h.get('analysis_end', '?')}"),
        ("Invested value", _money(h.get("current_portfolio_value"))),
        ("Total portfolio value", _money(h.get("total_account_value_estimate"))),
        ("Cash in hand", _money(h.get("uninvested_cash_estimate"))),
        ("Realized P&L", _money(h.get("total_realized_pnl"))),
        ("Unrealized P&L", _money(h.get("total_unrealized_pnl"))),
        ("Observed risk", risk_str),
        ("Vs S&P 500 (excess)", _pct(excess * 100) if excess is not None else "—"),
    ]
    rows = [["Metric", "Value", "Metric", "Value"]]
    for i in range(0, len(pairs), 2):
        left = pairs[i]
        right = pairs[i + 1] if i + 1 < len(pairs) else ("", "")
        rows.append([left[0], str(left[1]), right[0], str(right[1])])
    return rows


def _risk_action_rows(diagnosis) -> tuple[list[str], list[list[str]]]:
    items = getattr(diagnosis, "holding_action_recommendations", []) or []
    actions = [a for a in items
               if getattr(a, "is_actionable", False) and float(getattr(a, "value_to_sell", 0) or 0) > 0]
    total_freed = sum(float(getattr(a, "value_to_sell", 0) or 0) for a in actions)
    names = ", ".join(getattr(a, "ticker", "") for a in actions) or "None"
    summary = [
        f"Actionable names: {len(actions)} ({names})",
        f"Capital freed: {_money(total_freed)}",
    ]
    rows = [["Ticker", "Action", "Reduce %", "Value to sell", "Sector"]]
    for a in actions:
        rows.append([
            getattr(a, "ticker", ""),
            getattr(a, "recommendation_label", ""),
            f"{float(getattr(a, 'position_reduction_pct', 0) or 0) * 100:.0f}%",
            _money(getattr(a, "value_to_sell", 0)),
            getattr(a, "sector", "") or "—",
        ])
    if len(rows) == 1:
        rows.append(["—", "No actionable trims", "—", "—", "—"])
    return summary, rows


# Short headers so the 9-column monthly table stays readable.
_MONTHLY_HEADERS = {
    "Beginning Balance": "Beginning",
    "Deposits / Withdrawals": "Deposits/Wd",
    "Investment Amount": "Invested",
    "Market Gain / Loss": "Mkt Gain/Loss",
    "Income Returns": "Income",
    "Personal Investment Returns": "Personal Ret",
    "Cumulative Returns": "Cumulative",
    "Ending Balance": "Ending",
}


def _monthly_rows(frame: pd.DataFrame, max_rows: int = 18) -> list[list[str]]:
    if frame is None or frame.empty:
        return [["No monthly performance available"]]
    cols = list(frame.columns)
    header = [_MONTHLY_HEADERS.get(c, c) for c in cols]
    rows = [header]
    for _, r in frame.head(max_rows).iterrows():
        rows.append([str(r["Month"]) if "Month" in cols else ""] +
                    [_money(r[c]) for c in cols if c != "Month"])
    return rows


def _execution_rows(frame: pd.DataFrame) -> list[list[str]]:
    if frame is None or frame.empty:
        return [["No execution steps available"]]
    cols = list(frame.columns)
    rows = [cols]
    for _, r in frame.iterrows():
        rows.append([str(r[c]) for c in cols])
    return rows


def _buy_idea_rows(candidates: list[dict]) -> list[list[str]]:
    rows = [["Ticker", "Name", "Fills gap", "Fit"]]
    for c in candidates or []:
        rows.append([
            str(c.get("ticker", "")),
            str(c.get("security_name", "")),
            str(c.get("linked_gap_label", "")),
            str(c.get("fit_band", c.get("fit_score", ""))),
        ])
    if len(rows) == 1:
        rows.append(["—", "No buy ideas generated", "—", "—"])
    return rows


def build_portfolio_summary_pdf(
    *,
    diagnosis,
    market_metrics: dict,
    portfolio_summary: dict,
    monthly_frame: pd.DataFrame,
    execution_frame: pd.DataFrame,
    buy_candidates: list[dict],
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    ss = _styles()
    doc = SimpleDocTemplate(
        str(out_path), pagesize=PAGE_SIZE,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
        title="Portfolio Summary",
    )
    flow: list[Any] = []
    flow.append(Paragraph("Portfolio Summary", ss["PA_H1"]))
    flow.append(Paragraph(f"Generated {datetime.now():%b %d, %Y %H:%M}", ss["PA_Small"]))
    flow.append(Spacer(1, 6))

    # 1. Overview
    flow.append(Paragraph("1. Overview", ss["PA_H2"]))
    flow.append(_table(_overview_metrics(market_metrics, portfolio_summary), USABLE_WIDTH))
    png = _benchmark_png(market_metrics.get("timeseries", []))
    if png:
        flow.append(Spacer(1, 8))
        flow.append(RLImage(io.BytesIO(png), width=9.0 * inch, height=3.6 * inch))

    # 2. Risk Action Summary
    flow.append(Paragraph("2. Risk Action Summary", ss["PA_H2"]))
    ra_summary, ra_rows = _risk_action_rows(diagnosis)
    for line in ra_summary:
        flow.append(Paragraph(line, ss["PA_Body"]))
    flow.append(Spacer(1, 4))
    flow.append(_table(ra_rows, USABLE_WIDTH))

    # 3. Monthly Performance (whole portfolio)
    flow.append(Paragraph("3. Monthly Performance (whole portfolio)", ss["PA_H2"]))
    flow.append(_table(_monthly_rows(monthly_frame), USABLE_WIDTH))

    # 4. Execution Summary
    flow.append(Paragraph("4. Execution Summary", ss["PA_H2"]))
    flow.append(_table(_execution_rows(execution_frame), USABLE_WIDTH))

    # 5. Buy Ideas (current selection)
    flow.append(Paragraph("5. Buy Ideas (current selection)", ss["PA_H2"]))
    flow.append(_table(_buy_idea_rows(buy_candidates), USABLE_WIDTH))

    flow.append(Spacer(1, 14))
    flow.append(Paragraph(
        "Educational portfolio review only. Not investment, tax, or trading advice.",
        ss["PA_Small"]))

    doc.build(flow)
    return out_path

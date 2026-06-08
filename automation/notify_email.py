"""Email delivery for the weekly digest: HTML body with the chart inline.

Config via environment (e.g. an .env you source before the job runs):
    SMTP_HOST   (default smtp.gmail.com)
    SMTP_PORT   (default 587, STARTTLS)
    SMTP_USER   your sending address / username
    SMTP_PASS   app password (for Gmail, create an App Password)
    EMAIL_FROM  (default = SMTP_USER)
    EMAIL_TO    (default = rana.sagnik05@gmail.com)

Use send_digest(digest, dry_run=True) to write the rendered HTML to /tmp instead
of sending — handy for testing without SMTP credentials.
"""

from __future__ import annotations

import os
import smtplib
import sys
from datetime import date
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import make_msgid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from automation.digest import Digest, Pick, RiskAction  # noqa: E402

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", SMTP_USER)
EMAIL_TO = os.environ.get("EMAIL_TO", "rana.sagnik05@gmail.com")


def _money(v: float | None) -> str:
    return "—" if v is None else f"${v:,.2f}"


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}%" if abs(v) < 5 else f"{v:.1f}%"


def _risk_rows(actions: list[RiskAction]) -> str:
    if not actions:
        return ("<tr><td colspan='4' style='padding:10px;color:#64748b'>"
                "No actionable risk trims this week.</td></tr>")
    out = []
    for r in actions:
        out.append(
            "<tr>"
            f"<td style='padding:8px 10px;font-weight:700'>{r.ticker}</td>"
            f"<td style='padding:8px 10px'>{r.label}</td>"
            f"<td style='padding:8px 10px;text-align:right'>{_money(r.value_to_sell)}</td>"
            f"<td style='padding:8px 10px;color:#475569'>{r.reason[:140]}</td>"
            "</tr>"
        )
    return "".join(out)


def _pick_rows(picks: list[Pick]) -> str:
    out = []
    for p in picks:
        badge_bg = "#dbeafe" if p.asset_type == "ETF" else "#dcfce7"
        badge_fg = "#1e40af" if p.asset_type == "ETF" else "#166534"
        out.append(
            "<tr>"
            f"<td style='padding:8px 10px;font-weight:700'>{p.ticker}</td>"
            f"<td style='padding:8px 10px'><span style='background:{badge_bg};color:{badge_fg};"
            f"padding:2px 8px;border-radius:10px;font-size:12px;font-weight:700'>{p.asset_type}</span></td>"
            f"<td style='padding:8px 10px'>{p.name[:34]}</td>"
            f"<td style='padding:8px 10px;text-align:right;font-weight:700'>{_money(p.allocation)}</td>"
            f"<td style='padding:8px 10px;text-align:right'>{p.fit_score:.0f}</td>"
            f"<td style='padding:8px 10px;color:#475569'>{(p.why or '')[:120]}</td>"
            "</tr>"
        )
    return "".join(out)


def _strip_md(text: str) -> str:
    """Remove stray markdown a local model may emit despite plain-text instructions."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # **bold**
    text = re.sub(r"__(.+?)__", r"\1", text)         # __bold__
    text = re.sub(r"(?m)^\s*#+\s*", "", text)        # # headers
    text = re.sub(r"(?m)^\s*[-*]\s+", "• ", text)    # bullet markers
    return text.strip()


def build_email_html(digest: Digest, chart_cid: str | None) -> str:
    summary_html = _strip_md(digest.summary).replace("\n", "<br>")
    chart_block = (
        f"<img src='cid:{chart_cid}' alt='Suggested blend vs S&P 500' "
        "style='width:100%;max-width:760px;border:1px solid #e2e8f0;border-radius:12px'/>"
        if chart_cid else
        "<div style='color:#64748b'>(chart unavailable this week)</div>"
    )
    th = "padding:8px 10px;text-align:left;font-size:12px;color:#475569;border-bottom:2px solid #e2e8f0"
    th_r = th + ";text-align:right"
    return f"""\
<div style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#0f172a;max-width:800px;margin:0 auto">
  <div style="padding:20px 22px;background:linear-gradient(180deg,#ffffff,#f8fafc);border:1px solid #e2e8f0;border-radius:18px">
    <div style="font-size:22px;font-weight:900">Weekly Portfolio Review — {digest.as_of:%b %d, %Y}</div>
    <div style="margin-top:4px;color:#475569;font-size:14px">
      Freed risk-action cash to redeploy: <b>{_money(digest.freed_cash)}</b> · no uninvested cash used
    </div>
  </div>

  <div style="margin-top:16px;padding:16px 18px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;line-height:1.55;font-size:14px">
    <div style="font-weight:800;margin-bottom:6px">Summary</div>
    {summary_html}
  </div>

  <div style="margin-top:18px;font-size:16px;font-weight:800">Risk Actions</div>
  <table style="width:100%;border-collapse:collapse;margin-top:6px;font-size:13px">
    <tr><th style="{th}">Ticker</th><th style="{th}">Action</th><th style="{th_r}">$ to move</th><th style="{th}">Why</th></tr>
    {_risk_rows(digest.risk_actions)}
  </table>

  <div style="margin-top:22px;font-size:16px;font-weight:800">Suggested Buys — 3 ETF / 7 stock blend</div>
  <div style="color:#64748b;font-size:12px;margin-bottom:6px">Sized by redeploying the freed risk-action cash above.</div>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <tr><th style="{th}">Ticker</th><th style="{th}">Type</th><th style="{th}">Name</th>
        <th style="{th_r}">Allocation</th><th style="{th_r}">Fit</th><th style="{th}">Why it fits</th></tr>
    {_pick_rows(digest.picks)}
  </table>

  <div style="margin-top:22px;font-size:16px;font-weight:800">Performance — suggested blend vs S&P 500</div>
  <div style="margin-top:8px">{chart_block}</div>

  <div style="margin-top:22px;padding:12px 14px;background:#fef2f2;border:1px solid #fecaca;border-radius:12px;color:#991b1b;font-size:12px">
    Educational portfolio review only. Not investment, tax, or trading advice.
  </div>
</div>"""


def build_message(digest: Digest) -> MIMEMultipart:
    msg = MIMEMultipart("related")
    msg["Subject"] = (
        f"Weekly Portfolio Review — {digest.as_of:%b %d} · "
        f"{len(digest.risk_actions)} risk action(s), {_money(digest.freed_cash)} to redeploy"
    )
    msg["From"] = EMAIL_FROM or EMAIL_TO
    msg["To"] = EMAIL_TO

    chart_cid = None
    if digest.chart_png:
        chart_cid = make_msgid(domain="portfolio.local")[1:-1]  # strip <>
    html = build_email_html(digest, chart_cid)
    msg.attach(MIMEText(html, "html"))
    if digest.chart_png and chart_cid:
        img = MIMEImage(digest.chart_png, _subtype="png")
        img.add_header("Content-ID", f"<{chart_cid}>")
        img.add_header("Content-Disposition", "inline", filename="blend_vs_sp500.png")
        msg.attach(img)
    return msg


def send_digest(digest: Digest, *, dry_run: bool = False) -> str:
    """Send the digest email. With dry_run=True, write the HTML to /tmp instead."""
    msg = build_message(digest)
    if dry_run:
        out = Path("/tmp") / f"weekly_digest_{digest.as_of:%Y%m%d}.html"
        # Reconstruct standalone HTML with an embedded data-URI chart for preview.
        import base64
        html = build_email_html(digest, None)
        if digest.chart_png:
            b64 = base64.b64encode(digest.chart_png).decode()
            html = html.replace(
                "(chart unavailable this week)",
                f"<img src='data:image/png;base64,{b64}' style='width:100%;max-width:760px'/>",
            )
        out.write_text(html)
        print(f"[dry-run] wrote {out}")
        return str(out)

    if not (SMTP_USER and SMTP_PASS):
        raise RuntimeError(
            "SMTP_USER and SMTP_PASS must be set to send email. "
            "For Gmail, create an App Password and export SMTP_USER / SMTP_PASS."
        )
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(EMAIL_FROM or SMTP_USER, [EMAIL_TO], msg.as_string())
    print(f"Sent weekly digest to {EMAIL_TO}")
    return EMAIL_TO


if __name__ == "__main__":
    import argparse
    from automation.core import analyze_portfolio
    from automation.digest import build_digest

    ap = argparse.ArgumentParser(description="Render/send the weekly digest email.")
    ap.add_argument("csv")
    ap.add_argument("--send", action="store_true", help="actually send (otherwise dry-run to /tmp)")
    args = ap.parse_args()

    res = analyze_portfolio(args.csv)
    dig = build_digest(res, persist_state=False)
    send_digest(dig, dry_run=not args.send)

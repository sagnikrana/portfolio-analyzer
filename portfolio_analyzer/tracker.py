"""Recommendation tracking — saves snapshots, upserts transactions, runs attribution."""
from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

SNAPSHOT_WINDOW_DAYS = 14

_ACTION_SELL_CODES = {"sell", "trim"}
_TRANS_BUY = "Buy"
_TRANS_SELL = "Sell"


# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS users (
    username   TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id       TEXT PRIMARY KEY,
    username          TEXT NOT NULL,
    snapshot_date     TEXT NOT NULL,
    expiry_date       TEXT NOT NULL,
    portfolio_value   REAL,
    FOREIGN KEY(username) REFERENCES users(username)
);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id  TEXT PRIMARY KEY,
    snapshot_id        TEXT NOT NULL,
    ticker             TEXT NOT NULL,
    action             TEXT NOT NULL,
    recommended_value  REAL,
    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id      TEXT PRIMARY KEY,
    username            TEXT NOT NULL,
    date                TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    action              TEXT NOT NULL,
    quantity            REAL,
    price               REAL,
    total_value         REAL,
    attribution         TEXT DEFAULT 'unprocessed',
    matched_snapshot_id TEXT,
    FOREIGN KEY(username) REFERENCES users(username)
);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(_DDL)
    con.commit()
    con.close()


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


# ── User ──────────────────────────────────────────────────────────────────────

def ensure_user(db_path: Path, username: str) -> None:
    con = _connect(db_path)
    con.execute(
        "INSERT OR IGNORE INTO users(username, created_at) VALUES(?, ?)",
        (username, datetime.now().date().isoformat()),
    )
    con.commit()
    con.close()


# ── Snapshot ──────────────────────────────────────────────────────────────────

def _extract_recommendations(diagnosis: Any) -> list[dict]:
    """Pull sell/trim and buy recs from a PortfolioRiskDiagnosis."""
    recs: list[dict] = []

    # Sell / Trim recommendations
    for rec in getattr(diagnosis, "holding_action_recommendations", []):
        reduction = float(getattr(rec, "position_reduction_pct", 0) or 0)
        if reduction <= 0:
            continue
        action = "sell" if reduction >= 0.999 else "trim"
        recs.append({
            "ticker": rec.ticker,
            "action": action,
            "recommended_value": float(getattr(rec, "value_to_sell", 0) or 0),
        })

    # Buy recommendations (from next_steps actions)
    next_steps = getattr(diagnosis, "portfolio_next_steps", None)
    if next_steps:
        for action in getattr(next_steps, "actions", []):
            if getattr(action, "action_type", "") not in ("Buy", "Add"):
                continue
            ticker = getattr(action, "ticker", "")
            if not ticker:
                continue
            # amount_text examples:
            #   "Suggested starting slice: $19,152.00 (34.1% of buy budget)"
            #   "$5,000.00 (13.21 shares)"
            amount_text = getattr(action, "amount_text", "") or ""
            value = 0.0
            m = re.search(r"\$([\d,]+(?:\.\d+)?)", amount_text)
            if m:
                try:
                    value = float(m.group(1).replace(",", ""))
                except ValueError:
                    pass
            recs.append({"ticker": ticker, "action": "buy", "recommended_value": value})

    return recs


def save_snapshot(db_path: Path, username: str, diagnosis: Any) -> str:
    """Persist one analysis snapshot. Returns the new snapshot_id."""
    ensure_user(db_path, username)

    snapshot_id = str(uuid.uuid4())
    snapshot_date = datetime.now().date().isoformat()
    expiry_date = (datetime.now().date() + timedelta(days=SNAPSHOT_WINDOW_DAYS)).isoformat()

    preferences = getattr(diagnosis, "portfolio_preferences", None)
    portfolio_value = float(getattr(preferences, "current_total_portfolio_value", 0) or 0)

    recs = _extract_recommendations(diagnosis)

    con = _connect(db_path)
    con.execute(
        "INSERT INTO snapshots(snapshot_id, username, snapshot_date, expiry_date, portfolio_value) VALUES(?,?,?,?,?)",
        (snapshot_id, username, snapshot_date, expiry_date, portfolio_value),
    )
    for rec in recs:
        con.execute(
            "INSERT INTO recommendations(recommendation_id, snapshot_id, ticker, action, recommended_value) VALUES(?,?,?,?,?)",
            (str(uuid.uuid4()), snapshot_id, rec["ticker"], rec["action"], rec["recommended_value"]),
        )
    con.commit()
    con.close()
    return snapshot_id


# ── Transactions ──────────────────────────────────────────────────────────────

def upsert_transactions(db_path: Path, username: str, transactions_df: pd.DataFrame) -> int:
    """Insert new Buy/Sell transactions from a CSV DataFrame. Returns count inserted."""
    ensure_user(db_path, username)

    df = transactions_df[transactions_df["Trans Code"].isin([_TRANS_BUY, _TRANS_SELL])].copy()
    df = df[df["Instrument"].str.strip() != ""]

    con = _connect(db_path)
    inserted = 0
    for _, row in df.iterrows():
        ticker = str(row["Instrument"]).strip()
        action = str(row["Trans Code"]).strip()
        date = pd.Timestamp(row["Activity Date"]).date().isoformat()
        quantity = float(row.get("Quantity_num", 0) or 0)
        price = float(row.get("Price_num", 0) or 0)
        total_value = abs(float(row.get("Amount_num", 0) or 0))

        # Natural key: username + date + ticker + action + quantity to avoid re-inserting
        existing = con.execute(
            "SELECT 1 FROM transactions WHERE username=? AND date=? AND ticker=? AND action=? AND quantity=?",
            (username, date, ticker, action, quantity),
        ).fetchone()
        if existing:
            continue

        con.execute(
            "INSERT INTO transactions(transaction_id, username, date, ticker, action, quantity, price, total_value) VALUES(?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), username, date, ticker, action, quantity, price, total_value),
        )
        inserted += 1

    con.commit()
    con.close()
    return inserted


# ── Attribution ───────────────────────────────────────────────────────────────

def run_attribution(db_path: Path, username: str) -> None:
    """Match unprocessed transactions to recommendation windows."""
    con = _connect(db_path)

    unprocessed = con.execute(
        "SELECT * FROM transactions WHERE username=? AND attribution='unprocessed'",
        (username,),
    ).fetchall()

    snapshots = con.execute(
        "SELECT snapshot_id, snapshot_date, expiry_date FROM snapshots WHERE username=?",
        (username,),
    ).fetchall()

    for tx in unprocessed:
        tx_date = tx["date"]
        tx_ticker = tx["ticker"]
        tx_action = tx["action"]  # "Buy" or "Sell"

        matched_snapshot_id = None

        for snap in snapshots:
            if not (snap["snapshot_date"] <= tx_date <= snap["expiry_date"]):
                continue

            # Check if any recommendation in this snapshot matches
            expected_action = "buy" if tx_action == _TRANS_BUY else None  # sell or trim both map to Sell in CSV
            if tx_action == _TRANS_SELL:
                recs = con.execute(
                    "SELECT 1 FROM recommendations WHERE snapshot_id=? AND ticker=? AND action IN ('sell','trim')",
                    (snap["snapshot_id"], tx_ticker),
                ).fetchone()
            else:
                recs = con.execute(
                    "SELECT 1 FROM recommendations WHERE snapshot_id=? AND ticker=? AND action='buy'",
                    (snap["snapshot_id"], tx_ticker),
                ).fetchone()

            if recs:
                matched_snapshot_id = snap["snapshot_id"]
                break

        attribution = "recommendation_driven" if matched_snapshot_id else "own_selection"
        con.execute(
            "UPDATE transactions SET attribution=?, matched_snapshot_id=? WHERE transaction_id=?",
            (attribution, matched_snapshot_id, tx["transaction_id"]),
        )

    con.commit()
    con.close()


# ── History HTML ──────────────────────────────────────────────────────────────

def build_history_html(db_path: Path, username: str) -> str:
    con = _connect(db_path)

    snapshots = con.execute(
        "SELECT * FROM snapshots WHERE username=? ORDER BY snapshot_date DESC",
        (username,),
    ).fetchall()

    if not snapshots:
        con.close()
        return _empty_state_html(username)

    today = datetime.now().date().isoformat()
    rows_html = ""

    for snap in snapshots:
        sid = snap["snapshot_id"]
        snap_date = snap["snapshot_date"]
        expiry = snap["expiry_date"]
        portfolio_value = snap["portfolio_value"] or 0.0

        is_active = snap_date <= today <= expiry
        window_label = (
            f'<span style="color:#16a34a;font-weight:700">Active</span> — expires {expiry}'
            if is_active
            else f'<span style="color:#64748b">Closed {expiry}</span>'
        )

        recs = con.execute(
            "SELECT ticker, action, recommended_value FROM recommendations WHERE snapshot_id=? ORDER BY action, ticker",
            (sid,),
        ).fetchall()

        sell_recs = [r for r in recs if r["action"] in ("sell", "trim")]
        buy_recs = [r for r in recs if r["action"] == "buy"]

        txs = con.execute(
            "SELECT * FROM transactions WHERE username=? AND matched_snapshot_id=? ORDER BY date",
            (username, sid),
        ).fetchall()
        rec_driven_txs = [t for t in txs if t["attribution"] == "recommendation_driven"]

        own_txs = con.execute(
            """SELECT * FROM transactions WHERE username=?
               AND date BETWEEN ? AND ?
               AND (matched_snapshot_id IS NULL OR matched_snapshot_id != ?)
               AND attribution = 'own_selection'
               ORDER BY date""",
            (username, snap_date, expiry, sid),
        ).fetchall()

        rec_rows = ""
        for r in sell_recs:
            action_badge = (
                '<span style="background:#fef2f2;color:#dc2626;padding:1px 7px;border-radius:4px;font-size:11px;font-weight:700">SELL</span>'
                if r["action"] == "sell"
                else '<span style="background:#fff7ed;color:#ea580c;padding:1px 7px;border-radius:4px;font-size:11px;font-weight:700">TRIM</span>'
            )
            value_str = f"~${r['recommended_value']:,.0f}" if r["recommended_value"] else ""
            rec_rows += f"<tr><td style='padding:4px 8px'>{action_badge}</td><td style='padding:4px 8px;font-weight:600'>{r['ticker']}</td><td style='padding:4px 8px;color:#64748b'>{value_str}</td></tr>"

        for r in buy_recs:
            action_badge = '<span style="background:#f0fdf4;color:#16a34a;padding:1px 7px;border-radius:4px;font-size:11px;font-weight:700">BUY</span>'
            value_str = f"~${r['recommended_value']:,.0f}" if r["recommended_value"] else ""
            rec_rows += f"<tr><td style='padding:4px 8px'>{action_badge}</td><td style='padding:4px 8px;font-weight:600'>{r['ticker']}</td><td style='padding:4px 8px;color:#64748b'>{value_str}</td></tr>"

        matched_rows = ""
        for t in rec_driven_txs:
            act = t["action"]
            badge_style = "background:#f0fdf4;color:#16a34a" if act == "Buy" else "background:#fef2f2;color:#dc2626"
            matched_rows += (
                f"<tr>"
                f"<td style='padding:3px 8px'>{t['date']}</td>"
                f"<td style='padding:3px 8px;font-weight:600'>{t['ticker']}</td>"
                f"<td style='padding:3px 8px'><span style='{badge_style};padding:1px 6px;border-radius:4px;font-size:11px;font-weight:700'>{act.upper()}</span></td>"
                f"<td style='padding:3px 8px;color:#64748b'>{t['quantity']:.4f} @ ${t['price']:.2f}</td>"
                f"<td style='padding:3px 8px'>✓ matched</td>"
                f"</tr>"
            )

        own_rows = ""
        for t in own_txs:
            act = t["action"]
            badge_style = "background:#f0fdf4;color:#16a34a" if act == "Buy" else "background:#fef2f2;color:#dc2626"
            own_rows += (
                f"<tr>"
                f"<td style='padding:3px 8px'>{t['date']}</td>"
                f"<td style='padding:3px 8px;font-weight:600'>{t['ticker']}</td>"
                f"<td style='padding:3px 8px'><span style='{badge_style};padding:1px 6px;border-radius:4px;font-size:11px;font-weight:700'>{act.upper()}</span></td>"
                f"<td style='padding:3px 8px;color:#64748b'>{t['quantity']:.4f} @ ${t['price']:.2f}</td>"
                f"<td style='padding:3px 8px;color:#94a3b8'>own pick</td>"
                f"</tr>"
            )

        adherence_count = len(rec_driven_txs)
        total_recs = len(sell_recs) + len(buy_recs)
        adherence_pct = (adherence_count / total_recs * 100) if total_recs > 0 else 0

        rows_html += f"""
<div style="border:1px solid #e2e8f0;border-radius:12px;margin-bottom:18px;overflow:hidden;background:#ffffff">
  <div style="background:#f8fafc;padding:14px 18px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #e2e8f0">
    <div>
      <span style="font-size:15px;font-weight:700;color:#0f172a">Snapshot — {snap_date}</span>
      <span style="margin-left:14px;font-size:13px;color:#64748b">{window_label}</span>
    </div>
    <div style="text-align:right">
      <span style="font-size:13px;color:#64748b">Portfolio at snapshot: </span>
      <span style="font-size:13px;font-weight:700;color:#0f172a">${portfolio_value:,.0f}</span>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0">
    <div style="padding:14px 18px;border-right:1px solid #f1f5f9">
      <div style="font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">
        Recommendations ({total_recs})
      </div>
      {"<table style='width:100%;border-collapse:collapse'>" + rec_rows + "</table>" if rec_rows else "<div style='color:#94a3b8;font-size:13px'>No actionable recommendations in this snapshot</div>"}
    </div>

    <div style="padding:14px 18px">
      <div style="font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">
        Adherence — {adherence_count} of {total_recs} followed ({adherence_pct:.0f}%)
      </div>
      {"<table style='width:100%;border-collapse:collapse;font-size:13px'>" + matched_rows + own_rows + "</table>" if (matched_rows or own_rows) else "<div style='color:#94a3b8;font-size:13px'>No trades recorded in this window yet</div>"}
    </div>
  </div>
</div>
"""

    con.close()

    return f"""
<div style="padding:0 2px">
  <div style="font-size:18px;font-weight:800;color:#0f172a;margin-bottom:4px">Recommendation History — {username}</div>
  <div style="font-size:13px;color:#64748b;margin-bottom:18px">
    Snapshots are valid for {SNAPSHOT_WINDOW_DAYS} days. Trades within the window are matched automatically when you save a new snapshot.
  </div>
  {rows_html}
</div>
"""


def _empty_state_html(username: str) -> str:
    return f"""
<div style="padding:32px;text-align:center;color:#64748b;border:1.5px dashed #cbd5e1;border-radius:12px;margin-top:8px">
  <div style="font-size:22px;margin-bottom:8px">📋</div>
  <div style="font-size:15px;font-weight:600;color:#0f172a;margin-bottom:6px">No snapshots yet for "{username}"</div>
  <div style="font-size:13px">Run an analysis, then click <strong>Save Snapshot</strong> to start tracking your recommendation adherence.</div>
</div>
"""

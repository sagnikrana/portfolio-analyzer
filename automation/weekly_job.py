"""Weekly orchestration for the portfolio agent.

Because a Robinhood report can take minutes to many HOURS to build, generation
and delivery are decoupled into two scheduled steps driven by a tiny state
machine (automation/state/job_state.json):

  generate   (weekly)  -> request a fresh activity report; mark pending=True
  process    (hourly)  -> while pending: download-if-ready; when ready, build the
                          digest, email it once, mark pending=False

This means `process` does nothing 99% of the time (cheap), polls only after a
generate, and never double-emails the same report (guarded by a content hash).

Manual:
  python -m automation.weekly_job test data/raw/mantis_invest.csv   # dry-run email
  python -m automation.weekly_job generate
  python -m automation.weekly_job process            # dry-run unless --send
  python -m automation.weekly_job process --send
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from automation.core import analyze_portfolio  # noqa: E402
from automation.digest import build_digest  # noqa: E402
from automation.ingest.robinhood_scraper import download_report, generate_report  # noqa: E402
from automation.notify_email import send_digest  # noqa: E402

STATE_DIR = REPO_ROOT / "automation" / "state"
JOB_STATE = STATE_DIR / "job_state.json"


def _log(msg: str) -> None:
    print(f"[weekly-job {datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def _load_state() -> dict:
    if JOB_STATE.exists():
        try:
            return json.loads(JOB_STATE.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    JOB_STATE.write_text(json.dumps(state, indent=2))


def _csv_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_universe() -> bool:
    """Rebuild the buy-candidate universe from the latest index constituents.

    Re-scrapes S&P 500 / Nasdaq-100 / Dow 30 membership (plus curated ETFs),
    rewrites data/raw/buy_candidate_universe.csv, and refreshes the enriched
    present-day returns the recommender reads. Wrapped so a Wikipedia/Yahoo
    hiccup logs and continues rather than blocking report generation — a slightly
    stale universe is better than a skipped week.
    """
    _log("Refreshing buy universe (S&P 500 / Nasdaq-100 / Dow 30 constituents) ...")
    try:
        # Lazy import: pulls in yfinance/pandas, so keep it out of module load.
        from data_pipeline.build_buy_candidate_universe import enrich_buy_candidate_universe

        frame = enrich_buy_candidate_universe()
        state = _load_state()
        state["last_universe_refresh_at"] = datetime.now().isoformat()
        state["last_universe_size"] = int(len(frame))
        _save_state(state)
        _log(f"Universe refreshed: {len(frame)} candidates from the latest index membership.")
        return True
    except Exception as exc:  # noqa: BLE001 - never let a refresh failure skip the week
        _log(f"Universe refresh FAILED ({exc!r}); continuing with the existing universe.")
        return False


def cmd_generate() -> int:
    """Refresh the buy universe, then request a fresh activity report."""
    _refresh_universe()
    _log("Requesting a fresh Robinhood activity report ...")
    ok = generate_report(headless=True)  # session persists; headless is fine
    state = _load_state()
    if ok:
        state["pending_generation"] = True
        state["last_generated_at"] = datetime.now().isoformat()
        _save_state(state)
        _log("Generation requested; pipeline is now pending download.")
        return 0
    _log("Generation request FAILED (session may need re-auth — run an interactive "
         "generate once: python -m automation.ingest.robinhood_scraper --mode generate).")
    return 1


def _process_csv(csv_path: Path, *, send: bool) -> None:
    _log(f"Analyzing {csv_path.name} ...")
    result = analyze_portfolio(csv_path)
    digest = build_digest(result)  # persists the week-over-week snapshot
    send_digest(digest, dry_run=not send)
    _log(f"Digest delivered ({'sent' if send else 'dry-run'}): "
         f"{len(digest.risk_actions)} risk action(s), {len(digest.picks)} picks, "
         f"${digest.freed_cash:,.0f} to redeploy.")


def cmd_process(*, send: bool) -> int:
    """If a report is pending and ready, download → digest → email (once)."""
    state = _load_state()
    if not state.get("pending_generation"):
        _log("Nothing pending — no report to process.")
        return 0
    _log("Pending report — checking if it's ready to download ...")
    csv_path = download_report(headless=True)
    if csv_path is None:
        _log("Report not ready yet (will retry next hour).")
        return 0
    h = _csv_hash(csv_path)
    if h == state.get("last_emailed_hash"):
        _log("Downloaded report is identical to the last one emailed — skipping.")
        state["pending_generation"] = False
        _save_state(state)
        return 0
    _process_csv(csv_path, send=send)
    state["pending_generation"] = False
    state["last_emailed_hash"] = h
    state["last_emailed_at"] = datetime.now().isoformat()
    state["last_csv"] = str(csv_path)
    _save_state(state)
    return 0


def cmd_test(csv: str, *, send: bool) -> int:
    """Run the full digest+email pipeline on a given CSV (dry-run by default)."""
    _process_csv(Path(csv), send=send)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Weekly portfolio agent orchestration.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate", help="refresh universe + request a fresh Robinhood report (weekly)")
    sub.add_parser("refresh-universe", help="rebuild buy universe from latest index constituents")
    p_proc = sub.add_parser("process", help="download-if-ready, then digest+email (hourly)")
    p_proc.add_argument("--send", action="store_true", help="actually send (default dry-run)")
    p_test = sub.add_parser("test", help="run digest+email on a given CSV")
    p_test.add_argument("csv")
    p_test.add_argument("--send", action="store_true", help="actually send (default dry-run)")
    args = ap.parse_args()

    if args.cmd == "generate":
        return cmd_generate()
    if args.cmd == "refresh-universe":
        return 0 if _refresh_universe() else 1
    if args.cmd == "process":
        return cmd_process(send=args.send)
    if args.cmd == "test":
        return cmd_test(args.csv, send=args.send)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

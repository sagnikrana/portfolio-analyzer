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
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from automation.core import analyze_portfolio  # noqa: E402
from automation.digest import build_digest  # noqa: E402
from automation.ingest.robinhood_scraper import download_report, generate_report  # noqa: E402
from automation.notify_email import send_alert, send_digest  # noqa: E402

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


def _alert(subject: str, body: str, *, key: str, throttle_hours: int = 12) -> None:
    """Email an operational alert, de-duplicated by `key` within a throttle window.

    The hourly `process` job could otherwise send the same failure 24x/day, so a
    given alert key is sent at most once per `throttle_hours`. Never raises —
    alerting must not itself crash the job (it only logs on failure).
    """
    try:
        state = _load_state()
        alerts = state.get("alerts") or {}
        now = datetime.now()
        last = alerts.get(key)
        if last:
            try:
                if (now - datetime.fromisoformat(last)).total_seconds() < throttle_hours * 3600:
                    _log(f"Alert '{key}' throttled (last sent {last}); not re-sending.")
                    return
            except ValueError:
                pass  # unparsable timestamp — treat as no prior alert
        send_alert(f"[Portfolio Analyzer] {subject}", body)
        alerts[key] = now.isoformat()
        state["alerts"] = alerts
        _save_state(state)
        _log(f"Alert emailed: {subject}")
    except Exception as exc:  # noqa: BLE001 - alerting must never break the job
        _log(f"Alert send FAILED ({exc!r}); continuing.")


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


def _refresh_macro() -> bool:
    """Refresh FRED macro series (Fed Funds, CPI, unemployment, 10Y/2Y) weekly."""
    _log("Refreshing macro series from FRED ...")
    try:
        from data_pipeline.build_macro_series import build_macro_series

        frame = build_macro_series()
        state = _load_state()
        state["last_macro_refresh_at"] = datetime.now().isoformat()
        _save_state(state)
        _log(f"Macro refreshed: {len(frame)} rows across {frame['series_id'].nunique()} series.")
        return True
    except Exception as exc:  # noqa: BLE001
        _log(f"Macro refresh FAILED ({exc!r}); continuing with existing macro data.")
        return False


def _refresh_news() -> bool:
    """Refresh news_metadata + document_corpus from GDELT (holdings) weekly."""
    _log("Refreshing news/narrative data from GDELT ...")
    try:
        from data_pipeline.build_news_corpus import build_news_corpus

        summary = build_news_corpus()
        if summary.get("skipped"):
            _log("News refresh skipped (GDELT rate-limited/down); kept existing news.")
            return False
        state = _load_state()
        state["last_news_refresh_at"] = datetime.now().isoformat()
        _save_state(state)
        _log(f"News refreshed: {summary['news_rows']} articles across {summary['tickers']} tickers.")
        return True
    except Exception as exc:  # noqa: BLE001
        _log(f"News refresh FAILED ({exc!r}); continuing with existing news.")
        return False


def _refresh_fundamentals() -> bool:
    """Rebuild SEC company facts (monthly — fundamentals only move quarterly).

    Heavy (hundreds of SEC requests), so run as an isolated subprocess and only
    on the first weekly run of each month.
    """
    import subprocess

    _log("Refreshing SEC company facts (monthly) ...")
    try:
        result = subprocess.run(
            # --with-candidates so the buy-candidate universe's fundamentals (which
            # feed buy recommendations) refresh too, not just current holdings.
            [sys.executable, "-m", "data_pipeline.build_company_facts", "--with-candidates"],
            cwd=str(REPO_ROOT),
            timeout=2 * 60 * 60,  # generous; SEC pull over holdings + candidate universe
        )
        if result.returncode == 0:
            state = _load_state()
            state["last_fundamentals_refresh_at"] = datetime.now().isoformat()
            _save_state(state)
            _log("SEC company facts refreshed.")
            return True
        _log(f"SEC company-facts rebuild exited {result.returncode}; kept existing facts.")
        return False
    except Exception as exc:  # noqa: BLE001
        _log(f"SEC company-facts refresh FAILED ({exc!r}); kept existing facts.")
        return False


def _refresh_external_data() -> None:
    """Refresh all data that backs the recommendations before the weekly report.

    Weekly: universe (index membership), macro (FRED), news (GDELT).
    Monthly (first weekly run of the month): SEC fundamentals.
    Each refresher self-contains its failures so one bad source never blocks the
    rest or the report request.
    """
    results = {
        "buy universe": _refresh_universe(),
        "macro (FRED)": _refresh_macro(),
    }
    # News is intentionally excluded from alerting: GDELT rate-limit "skips" are
    # routine and a week of slightly stale news is low-impact.
    _refresh_news()
    if datetime.now().day <= 7:  # first weekly run of the month
        results["SEC fundamentals"] = _refresh_fundamentals()

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        failed_list = "\n".join(f"  - {name}" for name in failed)
        _alert(
            f"Weekly data refresh degraded: {', '.join(failed)}",
            "The weekly job could not refresh the following data source(s):\n"
            f"{failed_list}\n\n"
            "The job continued using the last-known-good data for those sources, so "
            "recommendations may be slightly stale. Details in /tmp/pa_generate.log.",
            key="external-data",
            throttle_hours=20,
        )


def cmd_generate() -> int:
    """Refresh all supporting data, then request a fresh activity report."""
    _refresh_external_data()
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
    _alert(
        "Robinhood session needs re-auth — no report requested this week",
        "The weekly job could NOT request a fresh Robinhood activity report because "
        "the saved login session has expired.\n\n"
        "Impact: no new report was requested, so you will NOT get a digest this week "
        "until you re-authenticate.\n\n"
        "Fix (run once on the Mac — a browser window opens for login + 2FA):\n"
        "  cd /Users/sagnikrana/Documents/GitHub/portfolio-analyzer\n"
        "  .venv/bin/python -m automation.ingest.robinhood_scraper --mode generate\n",
        key="rh-session",
        throttle_hours=20,
    )
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
    try:
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
    except Exception as exc:  # noqa: BLE001
        _log(f"Process step FAILED ({exc!r}).")
        _alert(
            "Weekly report processing failed",
            "The hourly process step failed while downloading, analyzing, or emailing "
            f"the pending report:\n\n  {exc!r}\n\n"
            "The report is still marked pending and the job will retry next hour. "
            "See /tmp/pa_process.log for the full traceback.",
            key="process-error",
            throttle_hours=12,
        )
        return 1


def cmd_test(csv: str, *, send: bool) -> int:
    """Run the full digest+email pipeline on a given CSV (dry-run by default)."""
    _process_csv(Path(csv), send=send)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Weekly portfolio agent orchestration.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate", help="refresh all supporting data + request a fresh Robinhood report (weekly)")
    sub.add_parser("refresh-universe", help="rebuild buy universe from latest index constituents")
    sub.add_parser("refresh-macro", help="refresh FRED macro series")
    sub.add_parser("refresh-news", help="refresh GDELT news / narrative corpus")
    sub.add_parser("refresh-fundamentals", help="rebuild SEC company facts (heavy)")
    sub.add_parser("refresh-data", help="run all supporting-data refreshers (weekly + monthly gate)")
    p_proc = sub.add_parser("process", help="download-if-ready, then digest+email (hourly)")
    p_proc.add_argument("--send", action="store_true", help="actually send (default dry-run)")
    p_test = sub.add_parser("test", help="run digest+email on a given CSV")
    p_test.add_argument("csv")
    p_test.add_argument("--send", action="store_true", help="actually send (default dry-run)")
    args = ap.parse_args()

    try:
        if args.cmd == "generate":
            return cmd_generate()
        if args.cmd == "refresh-universe":
            return 0 if _refresh_universe() else 1
        if args.cmd == "refresh-macro":
            return 0 if _refresh_macro() else 1
        if args.cmd == "refresh-news":
            return 0 if _refresh_news() else 1
        if args.cmd == "refresh-fundamentals":
            return 0 if _refresh_fundamentals() else 1
        if args.cmd == "refresh-data":
            _refresh_external_data()
            return 0
        if args.cmd == "process":
            return cmd_process(send=args.send)
        if args.cmd == "test":
            return cmd_test(args.csv, send=args.send)
        return 2
    except Exception as exc:  # noqa: BLE001 - safety net: never fail silently
        _log(f"Uncaught error in '{args.cmd}' ({exc!r}).")
        _alert(
            f"Weekly job crashed in '{args.cmd}'",
            f"The weekly job hit an uncaught error running '{args.cmd}':\n\n"
            f"{traceback.format_exc()}",
            key=f"job-crash-{args.cmd}",
            throttle_hours=6,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

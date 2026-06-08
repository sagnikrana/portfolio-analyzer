"""Robinhood activity-report scraper (Playwright, attaches to your logged-in Chrome).

This is the *ingestion adapter* for the weekly automation: it drives Robinhood's
"Activity reports" page to generate and download the transaction CSV that the
portfolio analyzer consumes. It is deliberately isolated from the app so that
when Robinhood changes their UI (they will), only this file needs touching.

WHY PLAYWRIGHT, NOT SELENIUM
    Playwright is already installed in the venv (no chromedriver to manage) and
    auto-waits on dynamic elements, which Robinhood's React UI needs. It attaches
    to your *already-logged-in* Chrome over the DevTools protocol (CDP), so we
    reuse your real session and never store your password or MFA secret.

HOW TO RUN
    1. Fully quit Chrome (Cmd+Q — not just close the window).
    2. Relaunch Chrome with remote debugging on, using your normal profile so you
       stay logged in:

        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
            --remote-debugging-port=9222

    3. Confirm you're still logged into robinhood.com in that Chrome.
    4. Run:

        .venv/bin/python automation/ingest/robinhood_scraper.py

    Add --debug to dump page HTML + screenshots to /tmp when a step can't find an
    element (useful for tightening selectors against the live DOM).

IMPORTANT
    - Scraping Robinhood is against their Terms of Service and the page DOM shifts
      without notice; treat this as best-effort personal automation that will need
      occasional selector maintenance.
    - The downloaded CSV contains real financial data. Keep this repo private and
      gitignore real exports (see the note at the bottom of this file).
    - Robinhood's date field is not text-editable, so we open the calendar and
      click the "previous month" arrow until we reach the target month, exactly
      as you described.
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from playwright.sync_api import (
    Locator,
    Page,
    TimeoutError as PWTimeoutError,
    sync_playwright,
)

# ── Configuration (matches the screenshot; override via CLI) ──────────────────
REPORTS_URL = "https://robinhood.com/account/reports-statements/activity-reports"
ACCOUNT_NAME = "Sagnik Invest"
START_DATE = date(2020, 8, 8)
END_DATE = date.today()  # Robinhood's form already defaults End date to today.

REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_DIR = REPO_ROOT / "data" / "raw"

# Dedicated, non-default Chrome profile that Playwright launches and controls.
# Using a non-default profile is required: Chrome 136+ refuses remote debugging on
# the default profile. The session you log in with persists here for future runs.
BOT_PROFILE_DIR = Path.home() / ".portfolio-analyzer" / "chrome-bot"

# How long to wait for the report to finish generating ("Pending" → "Download CSV").
REPORT_READY_TIMEOUT_S = 12 * 60
REPORT_POLL_EVERY_S = 15

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

DEBUG = False


def log(msg: str) -> None:
    print(f"[rh-scraper] {msg}", flush=True)


def _dump_debug(page: Page, tag: str) -> None:
    """Save a screenshot + HTML so we can refine selectors against the live page."""
    if not DEBUG:
        return
    try:
        png = Path("/tmp") / f"rh_{tag}.png"
        html = Path("/tmp") / f"rh_{tag}.html"
        page.screenshot(path=str(png), full_page=True)
        html.write_text(page.content())
        log(f"  [debug] wrote {png} and {html}")
    except Exception as exc:  # pragma: no cover - debug aid only
        log(f"  [debug] could not dump page: {exc}")


def _first_visible(page: Page, selectors: list[str], timeout_ms: int = 8000) -> Locator | None:
    """Return the first selector that resolves to a visible element, else None.

    Robinhood ships obfuscated class names, so we try several semantic selectors
    (role, accessible name, visible text) and use whichever the live DOM exposes.
    """
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if loc.count() > 0 and loc.is_visible():
                    return loc
            except Exception:
                continue
        page.wait_for_timeout(250)
    return None


def _rh_date_label(d: date) -> str:
    """Format a date the way Robinhood displays it, e.g. 'Aug 8, 2020'."""
    return f"{d.strftime('%b')} {d.day}, {d.year}"


def _ordinal(n: int) -> str:
    """1 -> '1st', 8 -> '8th', 22 -> '22nd' (matches react-datepicker aria-labels)."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ── Page steps ───────────────────────────────────────────────────────────────
LOGIN_WAIT_TIMEOUT_S = 5 * 60


def _is_login_screen(page: Page) -> bool:
    return "login" in page.url or bool(
        _first_visible(page, ["text=Log in", "text=Sign in", "input[name=username]"], 1200)
    )


def _ensure_logged_in(page: Page, wait_for_login: bool = True) -> bool:
    """Return True if logged in. If a login is needed and wait_for_login is True,
    poll until the user logs in (one-time, interactive). In unattended/headless
    modes pass wait_for_login=False to fail fast so the caller can flag re-auth."""
    page.goto(REPORTS_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(2500)
    if not _is_login_screen(page):
        log("Already logged in — reached the Activity reports page.")
        return True
    if not wait_for_login:
        log("Session is not logged in — re-auth needed (run the generate step "
            "interactively once to refresh the saved session).")
        return False
    log("Robinhood needs a login. Please log in (and complete MFA) in the Chrome "
        f"window that just opened. Waiting up to {LOGIN_WAIT_TIMEOUT_S // 60} minutes ...")
    deadline = time.time() + LOGIN_WAIT_TIMEOUT_S
    while time.time() < deadline:
        page.wait_for_timeout(3000)
        if not _is_login_screen(page):
            log("Login detected — continuing.")
            page.goto(REPORTS_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(2500)
            if not _is_login_screen(page):
                return True
    log("Timed out waiting for login.")
    _dump_debug(page, "login")
    return False


def _select_account(page: Page, account_name: str) -> None:
    # Native <select> case.
    try:
        sel = page.locator("select").first
        if sel.count() > 0 and sel.is_visible():
            sel.select_option(label=account_name)
            log(f"Selected account (native select): {account_name}")
            return
    except Exception:
        pass
    # Custom dropdown: open it, click the option by text.
    trigger = _first_visible(page, [
        f"button:has-text('{account_name}')",
        "label:has-text('Account') ~ * button",
        "text=Account >> xpath=following::button[1]",
    ], 4000)
    if trigger is None:
        log(f"Could not find the Account control; assuming '{account_name}' is already selected.")
        return
    try:
        current = (trigger.inner_text() or "").strip()
        if account_name.lower() in current.lower():
            log(f"Account already set to: {account_name}")
            return
        trigger.click()
        page.wait_for_timeout(500)
        opt = _first_visible(page, [f"text='{account_name}'", f"li:has-text('{account_name}')"], 3000)
        if opt:
            opt.click()
            log(f"Selected account: {account_name}")
        else:
            log(f"Opened account dropdown but couldn't find option '{account_name}'.")
            _dump_debug(page, "account")
    except Exception as exc:
        log(f"Account selection hit an issue ({exc}); continuing with current value.")


def _open_date_field(page: Page, field_label: str) -> bool:
    """Click a date field ('Start date' / 'End date') to open its calendar popup."""
    field = _first_visible(page, [
        f"text={field_label} >> xpath=following::input[1]",
        f"text={field_label} >> xpath=following::button[1]",
        f"label:has-text('{field_label}') ~ *",
        f"text={field_label}",
    ], 6000)
    if field is None:
        log(f"Could not locate the '{field_label}' field.")
        _dump_debug(page, "datefield")
        return False
    field.click()
    page.wait_for_timeout(700)
    return True


def _read_calendar_header(page: Page) -> tuple[int, int] | None:
    """Return (year, month_index) currently shown in the open calendar, or None.

    Robinhood uses react-datepicker; the visible header is
    `.react-datepicker__month-title-custom` with text like "January 2026".
    """
    header = _first_visible(page, [
        ".react-datepicker__month-title-custom",
        ".react-datepicker__current-month",
        ".react-datepicker__header",
    ], 4000)
    if header is None:
        return None
    try:
        text = (header.inner_text() or "").strip()
    except Exception:
        return None
    for i, name in enumerate(MONTHS):
        if name in text:
            for token in text.replace(",", " ").split():
                if token.isdigit() and len(token) == 4:
                    return int(token), i
    return None


def _calendar_prev_button(page: Page) -> Locator | None:
    return _first_visible(page, ["button[aria-label='Decrease month']"], 3000)


def _calendar_next_button(page: Page) -> Locator | None:
    return _first_visible(page, ["button[aria-label='Increase month']"], 3000)


def _set_date(page: Page, field_label: str, target: date) -> bool:
    """Open a date field and navigate the calendar to `target`, then click the day."""
    log(f"Setting {field_label} to {_rh_date_label(target)} ...")
    if not _open_date_field(page, field_label):
        return False

    for _ in range(240):  # generous cap (20 years of month-stepping)
        header = _read_calendar_header(page)
        if header is None:
            log(f"  Could not read the calendar header for {field_label}.")
            _dump_debug(page, "calendar")
            return False
        cur_year, cur_month = header
        target_month = target.month - 1
        if cur_year == target.year and cur_month == target_month:
            break
        going_back = (cur_year, cur_month) > (target.year, target_month)
        btn = _calendar_prev_button(page) if going_back else _calendar_next_button(page)
        if btn is None:
            log(f"  Could not find the {'<' if going_back else '>'} month-navigation arrow.")
            _dump_debug(page, "calendar_nav")
            return False
        btn.click()
        page.wait_for_timeout(180)
    else:
        log(f"  Gave up navigating the calendar to {target}.")
        return False

    # Click the target day. The aria-label ("Choose Thursday, August 8th, 2020")
    # encodes the exact date, so it uniquely targets the right cell and never an
    # adjacent-month day. Fall back to the zero-padded day class for the current
    # month only (react-datepicker marks adjacent-month days --outside-month).
    aria = f"{MONTHS[target.month - 1]} {_ordinal(target.day)}, {target.year}"
    day_cell = _first_visible(page, [
        f"[role=option][aria-label*='{aria}']",
        f".react-datepicker__day--{target.day:03d}:not(.react-datepicker__day--outside-month)",
    ], 4000)
    if day_cell is None:
        log(f"  Could not find day {target.day} ({aria}) in the calendar.")
        _dump_debug(page, "calendar_day")
        return False
    day_cell.click()
    page.wait_for_timeout(500)
    log(f"  {field_label} set to {_rh_date_label(target)}.")
    return True


def _click_generate(page: Page) -> bool:
    btn = _first_visible(page, [
        "button:has-text('Generate report')",
        "text=Generate report",
    ], 6000)
    if btn is None:
        log("Could not find the 'Generate report' button.")
        _dump_debug(page, "generate")
        return False
    btn.click()
    log("Clicked 'Generate report'.")
    # Dismiss the "We're preparing your report" confirmation if it appears.
    done = _first_visible(page, ["button:has-text('Done')"], 8000)
    if done:
        done.click()
        log("Acknowledged the 'preparing your report' dialog.")
    return True


def _report_row(page: Page, start: date, end: date) -> Locator | None:
    """Find the 'Your reports' row matching our date range."""
    start_lbl = _rh_date_label(start)
    end_lbl = _rh_date_label(end)
    # Row text looks like "Aug 8, 2020 – Jun 8, 2026". Match on both endpoints.
    candidates = [
        f"text=/{start_lbl}.*{end_lbl}/",
        f"*:has-text('{start_lbl}'):has-text('{end_lbl}')",
    ]
    return _first_visible(page, candidates, 4000)


def _do_download(page: Page, row: Locator) -> Path | None:
    """Download the CSV from a ready report row into DOWNLOAD_DIR."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOWNLOAD_DIR / f"robinhood_activity_{date.today():%Y%m%d}.csv"
    dl_link = row.locator("text=Download CSV").first
    try:
        with page.expect_download(timeout=60000) as dl:
            dl_link.click()
        dl.value.save_as(str(out_path))
        log(f"Saved: {out_path}")
        return out_path
    except PWTimeoutError:
        log("Clicked Download but no download event fired.")
        return None


@contextmanager
def _browser_page(headless: bool):
    """Launch the dedicated-profile Chrome and yield a page; always clean up."""
    with sync_playwright() as pw:
        BOT_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        context = pw.chromium.launch_persistent_context(
            user_data_dir=str(BOT_PROFILE_DIR),
            channel="chrome",
            headless=headless,
            accept_downloads=True,
            args=["--no-first-run", "--no-default-browser-check"],
        )
        page = context.pages[0] if context.pages else context.new_page()
        try:
            yield page
        finally:
            try:
                context.close()
            except Exception:
                pass


def generate_report(
    *,
    account_name: str = ACCOUNT_NAME,
    start: date = START_DATE,
    end: date = END_DATE,
    headless: bool = False,
) -> bool:
    """Trigger report generation, then return immediately (do NOT wait).

    Robinhood can take minutes to many hours to build a long report, so we never
    block on it. The report persists in 'Your reports' and is fetched later by
    download_report(). Headed by default so a one-time login can be completed.
    """
    with _browser_page(headless) as page:
        try:
            if not _ensure_logged_in(page, wait_for_login=not headless):
                return False
            _select_account(page, account_name)
            if not _set_date(page, "Start date", start):
                return False
            if end == date.today():
                log("End date already defaults to today — leaving it as-is.")
            elif not _set_date(page, "End date", end):
                return False
            ok = _click_generate(page)
            if ok:
                log("Report generation requested. It will appear in 'Your reports' "
                    "when ready (can take minutes to hours).")
            return ok
        except Exception as exc:
            log(f"generate_report error: {exc}")
            _dump_debug(page, "error")
            return False


def download_report(
    *,
    start: date = START_DATE,
    end: date = END_DATE,
    headless: bool = True,
) -> Path | None:
    """Download the report for the date range if it's READY. Returns the saved
    path, or None if it isn't ready yet / not found / re-auth is needed.

    Headless by default — intended to be run periodically (e.g. hourly) by the
    weekly job until the report is ready."""
    with _browser_page(headless) as page:
        try:
            if not _ensure_logged_in(page, wait_for_login=False):
                return None
            page.goto(REPORTS_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(2500)
            row = _report_row(page, start, end)
            if row is None:
                log(f"No report row for {_rh_date_label(start)} – {_rh_date_label(end)} yet.")
                return None
            try:
                row_text = row.inner_text()
            except Exception:
                row_text = ""
            if "Download CSV" not in row_text:
                log("Report exists but is still pending.")
                return None
            log("Report is ready — downloading CSV ...")
            return _do_download(page, row)
        except Exception as exc:
            log(f"download_report error: {exc}")
            _dump_debug(page, "error")
            return None


def fetch_transactions(
    *,
    account_name: str = ACCOUNT_NAME,
    start: date = START_DATE,
    end: date = END_DATE,
    timeout_min: int = 30,
) -> Path | None:
    """Convenience: generate, then poll-download in one process (for manual use).

    Not recommended for the weekly job since reports can take hours — use
    generate_report() then periodic download_report() instead.
    """
    if not generate_report(account_name=account_name, start=start, end=end):
        return None
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        path = download_report(start=start, end=end, headless=True)
        if path is not None:
            return path
        time.sleep(REPORT_POLL_EVERY_S)
    log(f"Report not ready within {timeout_min} min — run the download step later.")
    return None


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> int:
    global DEBUG
    ap = argparse.ArgumentParser(description="Generate / download a Robinhood activity report CSV.")
    ap.add_argument("--mode", choices=["generate", "download", "full"], default="full",
                    help="generate: request only; download: fetch if ready; full: generate then poll")
    ap.add_argument("--account", default=ACCOUNT_NAME)
    ap.add_argument("--start", type=_parse_date, default=START_DATE, help="YYYY-MM-DD")
    ap.add_argument("--end", type=_parse_date, default=END_DATE, help="YYYY-MM-DD")
    ap.add_argument("--timeout-min", type=int, default=30, help="full-mode poll budget (minutes)")
    ap.add_argument("--debug", action="store_true", help="dump screenshots/HTML to /tmp on misses")
    args = ap.parse_args()
    DEBUG = args.debug

    if args.mode == "generate":
        ok = generate_report(account_name=args.account, start=args.start, end=args.end)
        log("DONE — generation requested." if ok else "FAILED — could not request generation.")
        return 0 if ok else 1

    if args.mode == "download":
        out = download_report(start=args.start, end=args.end)
        if out is None:
            log("NOT READY — no CSV downloaded yet (retry later).")
            return 4  # distinct code so a poller knows to retry
        log(f"DONE — {out}")
        return 0

    out = fetch_transactions(
        account_name=args.account, start=args.start, end=args.end, timeout_min=args.timeout_min
    )
    if out is None:
        log("FAILED — no CSV downloaded.")
        return 1
    log(f"DONE — {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# NOTE: add real exports to .gitignore so financial data never gets committed, e.g.:
#   data/raw/robinhood_activity_*.csv

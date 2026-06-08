# Weekly Portfolio Agent

A local, self-hosted automation layer on top of the portfolio analyzer. Every
week it fetches your Robinhood activity, runs the same risk diagnosis the
dashboard runs, and emails you a prioritized digest: **Risk Actions** + a
**10-name buy blend (3 ETF / 7 stock)** funded by redeploying the freed
risk-action cash (no uninvested cash), with a **blend-vs-S&P performance chart**
and a summary written by a **local Ollama agent** (no API key, data stays on the
machine).

## Pipeline

```
launchd
  generate (weekly)  → ingest/robinhood_scraper.py : request the activity report
  process  (hourly)  → robinhood_scraper download   : fetch CSV when ready
                     → core.py        : analyze_portfolio(csv) -> AnalysisResult
                     → digest.py      : risk actions + 3/7 blend + chart + Ollama summary + diff
                     → notify_email.py: HTML email with the chart inline
```

Generation and delivery are **decoupled** because a Robinhood report can take
minutes to many hours. A tiny state machine (`automation/state/job_state.json`)
makes `process` cheap: it does nothing unless a report is pending, polls only
after a `generate`, and never double-emails the same report (content-hash guard).

## Modules

| File | Role |
|------|------|
| `core.py` | Headless `analyze_portfolio(csv) -> AnalysisResult` (shared with the app) |
| `digest.py` | Builds risk actions, the 3-ETF/7-stock blend, allocations, the chart, the Ollama summary, and the week-over-week diff |
| `notify_email.py` | Renders the HTML email and sends via SMTP (`dry_run` writes to `/tmp`) |
| `weekly_job.py` | Orchestrates `generate` / `process`; manual `test` runs the chain on any CSV |
| `ingest/robinhood_scraper.py` | Playwright scraper (`--mode generate|download|full`) |
| `launchd/` | `run.sh` + two plists + `secrets.env.example` |

## One-time setup

1. **Log in once** so the scraper's dedicated Chrome profile has a session:
   ```bash
   .venv/bin/python automation/ingest/robinhood_scraper.py --mode generate
   ```
   (Log into Robinhood + MFA in the window that opens. The session persists.)

2. **Secrets** — copy and fill in:
   ```bash
   mkdir -p ~/.portfolio-analyzer
   cp automation/launchd/secrets.env.example ~/.portfolio-analyzer/secrets.env
   # edit it: SMTP_USER / SMTP_PASS (Gmail App Password), EMAIL_TO, OLLAMA_MODEL
   ```

3. **Ollama** running with a model pulled (`ollama list`). Default `llama3.1:latest`;
   set `OLLAMA_MODEL=llama3.3:latest` for higher quality.

4. **Install the schedule**:
   ```bash
   cp automation/launchd/com.portfolioanalyzer.*.plist ~/Library/LaunchAgents/
   launchctl load ~/Library/LaunchAgents/com.portfolioanalyzer.generate.plist
   launchctl load ~/Library/LaunchAgents/com.portfolioanalyzer.process.plist
   ```

## Manual use / testing

```bash
# Full chain on any CSV, dry-run (writes HTML to /tmp, no email):
.venv/bin/python -m automation.weekly_job test data/raw/mantis_invest.csv

# Real run pieces:
.venv/bin/python -m automation.weekly_job generate          # request report
.venv/bin/python -m automation.weekly_job process --send    # download+email if ready
```

## Notes

- Scraping Robinhood is against their ToS and the page DOM changes; the scraper
  is isolated so only that file needs maintenance when it breaks.
- Real exports (`data/raw/robinhood_activity_*.csv`), `automation/state/`, and
  `secrets.env` are gitignored — financial data and credentials never get committed.
- Educational portfolio review only — not investment advice.

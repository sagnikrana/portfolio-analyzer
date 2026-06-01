---
name: technical-state
description: Current codebase state, recent fixes, known open issues, and what to tackle first
metadata:
  type: project
---

## Recently completed (from Codex)
- Volatility score now uses robust recent weekly-value-based method (matches the chart)
- Drawdown and market sensitivity aligned to same recent evidence path
- Backtesting controls changed from fragile checkboxes to `gr.Dropdown` (No/Yes)
- Quality mega-cap guardrails added to prevent over-trimming MSFT/GOOGL/AAPL
- Non-core laggard escalation implemented
- Tiny low-impact trade suppression added
- Buy ranking gives more credit to multi-window S&P outperformance

## Known open issues (priority order)
1. **Backtesting controls may still look washed out / unreadable in live app** — user was not confident the dropdown fixes landed correctly. Verify first.
2. **Run Analysis red error pills with mantis_invest** — recurring issue, output contract mismatches or stale UI state. Needs end-to-end test.
3. **Buy Preferences / Buy Ideas freezing** — heavy browser payload. Needs load testing.
4. **Backtesting chart scaling** — revised multiple times, should be sanity-checked on several dates.
5. **Stale frontend state** — sometimes requires hard refresh. Root cause not fully resolved.

## What to do first when picking up
1. Start the app and Run Analysis with mantis_invest — verify no red errors
2. Go to Backtesting tab — verify controls are readable and usable
3. Run a backtest on a few dates — verify results make sense
4. Only then add new features

## Architecture mental model
- `app.py`: Gradio app — tabs, callbacks, plotting, state, backtesting, upload handling, UI CSS
- `diagnosis.py`: risk scoring, sell/trim logic, buy idea scoring, recommendation rules, backtest-relevant decision logic
- `buy_candidates.py`: buy candidate universe (S&P 500, Nasdaq 100, Dow 30, curated ETFs), enrichment pipeline

## Upload handling
- Gradio sometimes passes file object, sometimes string path — app handles both
- Upload path stored in state to avoid fragile re-processing on click

## BACKTEST_BUY_IDEA_COUNT = 15 (tests all top 15 buy ideas)

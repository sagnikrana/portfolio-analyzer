---
name: project-context
description: Portfolio analyzer app — architecture, tabs, data, and product vision
metadata:
  type: project
---

**Repo:** /Users/sagnikrana/Documents/GitHub/portfolio-analyzer  
**Run command:** `cd repo && source .venv/bin/activate && python -m portfolio_analyzer`  
**App URL:** http://127.0.0.1:7863

## Core files
- `portfolio_analyzer/app.py` (~10,600 lines) — Gradio UI, callbacks, plotting, backtesting tab
- `portfolio_analyzer/diagnosis.py` (~5,100 lines) — deterministic risk scoring, sell/trim logic, buy scoring, recommendation rules
- `portfolio_analyzer/buy_candidates.py` (~955 lines) — buy candidate universe loading and enrichment

## Data
- `data/raw/mantis_invest.csv` — real sample dataset (user's Robinhood history)
- `data/raw/fake_mantis_invest.csv` — synthetic dataset
- `data/raw/buy_candidate_universe.csv` — buy candidate list
- `data/external/` — FMP company profiles, FRED macro, GDELT news, SEC facts

## Tabs (in order)
Overview → Risk Diagnosis → Risk Actions → Portfolio Gaps → Buy Preferences → Buy Ideas → Portfolio Rebalancing Plan → Backtesting → Next Steps

Risk Guide and Holdings are hidden (inside `gr.Group(visible=False)`).

## Core philosophy (agreed repeatedly)
- Math, rankings, thresholds, and portfolio decisions stay **deterministic in Python**
- LLMs explain, simplify, summarize, connect external evidence
- Agents come later, after deterministic core and backtesting are solid

## Product vision
- Weekly monitoring system
- User accounts with recommendation snapshots
- Actual path vs app-followed counterfactual ("cost of ignoring the app")
- Mobile-style notifications eventually

## Risk model (three-layer)
- Stated Risk Appetite: user input 0-100
- Risk Capacity: age, income, liquidity, horizon
- Observed Risk Behavior: derived from transaction history

## Backtesting design
- User picks cutoff date → app rebuilds portfolio as of that date
- Generates sell/trim actions + top 15 buy ideas as of cutoff
- Simulates counterfactual: what if user had followed the app?
- Compares actual path vs app-followed path through today
- Controls: backtest date (dropdown), use uninvested cash (No/Yes dropdown), include soft signals (No/Yes dropdown)

## Why: build for long-term retail investors who don't know if their portfolio matches their risk tolerance or where to allocate monthly budget. Not a stock oracle — a portfolio intelligence and planning assistant.

# Tests

Focused regression suite covering the bug classes this project has actually hit.

## Running

```bash
# fast, offline unit suite (default — ~1s)
.venv/bin/python -m pytest

# the end-to-end contract test (runs the real pipeline, needs network, ~3-4 min)
.venv/bin/python -m pytest -m integration

# everything
.venv/bin/python -m pytest -m ""
```

## Layout

| File | Guards against |
|------|----------------|
| `test_finance_formatters.py` | Unit-scaling bugs — the VOO **0.03% vs 3.00%** expense-ratio inflation, plus all the number parse/format helpers. |
| `test_sector_resolution.py` | Sectors collapsing into one bucket / `Unclassified`. Locks the deterministic override map, its canonical names, and that failed lookups are **not** cached. |
| `test_monthly_performance.py` | The monthly statement identity (`Ending = Beginning + Deposits + Market Gain + Income`) and the column set (dropped "Personal Investment Returns", kept Cumulative). |
| `test_preferences.py` | The "ETFs only" vehicle-selector mapping and the preference fields the buy engine reads by name. |
| `test_analysis_contract.py` *(integration)* | The **wrong-dict-key** class — asserts the pipeline emits exactly the keys/attributes the weekly digest and dashboard consume. |

The integration test skips itself (rather than failing) when the market data
can't be fetched, so the fast suite is meaningful offline.

"""Shared test configuration.

The suite is deliberately split into two speeds:

* Fast, deterministic unit tests (no network) that lock down the pure functions
  where real bugs have bitten us — unit scaling (expense ratios), sector-label
  resolution, the monthly-performance math, and the buy-preference mapping.
* One `integration` test that runs the full analysis pipeline and checks the
  data *contract* the weekly digest and the UI depend on (the "wrong dict key"
  class of bug). It needs network (yfinance) and skips itself if that fails, so
  the core suite stays green offline.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# yfinance/urllib3 on macOS LibreSSL is noisy; keep test output readable.
warnings.filterwarnings("ignore")

# Make the project importable when pytest is invoked from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

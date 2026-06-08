"""Refresh the macro series the diagnosis engine reads (FRED, no API key needed).

The app's rate-regime logic (_build_macro_context / portfolio-gap rate flags)
reads data/processed/structured/macro_series.csv with columns:

    date, value, series_id, title

and looks up the latest observation per series_id (FEDFUNDS, CPIAUCSL, UNRATE,
DGS10, DGS2). FRED exposes a key-free CSV download per series, so this rebuild
stays simple and dependency-light:

    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>

Run:  python data_pipeline/build_macro_series.py
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_PATH = ROOT / "data" / "processed" / "structured" / "macro_series.csv"
# Bound the range: FRED's CSV endpoint 504s on full daily history (DGS10/DGS2 go
# back to 1962). A 2015 start keeps the request fast while leaving ample history
# for the trailing CPI-YoY and yield-curve calcs (which only need ~1-2 years).
MACRO_START_DATE = "2015-01-01"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=" + MACRO_START_DATE

# series_id -> human title (must match what the app already displays).
SERIES: dict[str, str] = {
    "FEDFUNDS": "Fed Funds Rate",
    "CPIAUCSL": "CPI",
    "UNRATE": "Unemployment Rate",
    "DGS10": "10Y Treasury",
    "DGS2": "2Y Treasury",
}

# FRED resets connections for non-browser User-Agents, so present as a browser.
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio-analyzer macro refresh)"}


def _get_with_retry(url: str, *, retries: int = 4, base_delay: float = 1.5) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=45)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # transient resets / 5xx
            last_exc = exc
            time.sleep(base_delay * (attempt + 1))
    raise last_exc if last_exc else RuntimeError(f"failed to fetch {url}")


def _fetch_series(series_id: str) -> pd.DataFrame:
    """Download one FRED series as a long-format frame: date, value, series_id, title."""
    url = FRED_CSV_URL.format(series_id=series_id)
    resp = _get_with_retry(url)
    raw = pd.read_csv(io.StringIO(resp.text))
    # FRED returns two columns: a date column (observation_date or DATE) + the series.
    date_col = raw.columns[0]
    value_col = next((c for c in raw.columns[1:] if c.upper() == series_id.upper()), raw.columns[1])
    out = raw[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"})
    out["value"] = pd.to_numeric(out["value"], errors="coerce")  # FRED uses "." for gaps
    out = out.dropna(subset=["value"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date"])
    out["series_id"] = series_id
    out["title"] = SERIES[series_id]
    return out


def _load_existing() -> pd.DataFrame:
    """Existing macro_series.csv (for fallback) or an empty frame."""
    if OUTPUT_PATH.exists():
        try:
            return pd.read_csv(OUTPUT_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "value", "series_id", "title"])


def build_macro_series() -> pd.DataFrame:
    existing = _load_existing()
    frames: list[pd.DataFrame] = []
    fetched: set[str] = set()
    for series_id in SERIES:
        try:
            frame = _fetch_series(series_id)
            frames.append(frame)
            fetched.add(series_id)
            latest = frame["date"].max() if not frame.empty else "?"
            print(f"  {series_id}: {len(frame)} obs (latest {latest})")
        except Exception as exc:  # one bad series shouldn't sink the rest
            print(f"  {series_id}: FAILED ({exc!r})")

    # Never DROP a series on a transient failure: keep its prior rows as fallback.
    for series_id in SERIES:
        if series_id not in fetched and not existing.empty:
            kept = existing[existing["series_id"] == series_id]
            if not kept.empty:
                frames.append(kept[["date", "value", "series_id", "title"]])
                print(f"  {series_id}: kept {len(kept)} prior rows (fetch failed).")

    if not frames:
        raise RuntimeError("No FRED series fetched and no existing data; not writing.")
    combined = pd.concat(frames, ignore_index=True)[["date", "value", "series_id", "title"]]
    combined = combined.drop_duplicates(subset=["series_id", "date"], keep="first")
    combined = combined.sort_values(["series_id", "date"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    return combined


if __name__ == "__main__":
    print("Refreshing macro series from FRED ...")
    df = build_macro_series()
    print(f"Wrote {len(df)} rows across {df['series_id'].nunique()} series to {OUTPUT_PATH}")

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio_analyzer.buy_candidates import (
    build_known_universe_datasets,
    load_buy_candidate_universe,
)

RAW_UNIVERSE_PATH = ROOT / "data" / "raw" / "buy_candidate_universe.csv"
STRUCTURED_DIR = ROOT / "data" / "processed" / "structured"
RAW_OUTPUT_PATH = STRUCTURED_DIR / "buy_candidate_universe.csv"
ENRICHED_OUTPUT_PATH = STRUCTURED_DIR / "buy_candidate_universe_enriched.csv"
BENCHMARK_SYMBOL = "^GSPC"


def _safe_float(value: Any) -> float | None:
    """Convert loosely typed finance values to floats when possible."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _window_relative_return(
    history: pd.Series,
    benchmark_history: pd.Series,
    trading_days: int,
) -> tuple[float | None, float | None, float | None]:
    """Compute trailing stock, benchmark, and relative returns for one window."""
    joined = pd.concat(
        [
            pd.to_numeric(history, errors="coerce").rename("stock"),
            pd.to_numeric(benchmark_history, errors="coerce").rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if len(joined) < trading_days + 1:
        return None, None, None

    window = joined.tail(trading_days + 1)
    stock_return = (window["stock"].iloc[-1] / window["stock"].iloc[0]) - 1.0
    benchmark_return = (window["benchmark"].iloc[-1] / window["benchmark"].iloc[0]) - 1.0
    return float(stock_return), float(benchmark_return), float(stock_return - benchmark_return)


def _annualized_volatility(history: pd.Series, trading_days: int = 252) -> float | None:
    """Compute annualized daily-return volatility over the trailing year."""
    series = pd.to_numeric(history, errors="coerce").dropna()
    if len(series) < trading_days + 1:
        return None
    returns = series.pct_change().dropna().tail(trading_days)
    if returns.empty:
        return None
    return float(returns.std() * (252 ** 0.5))


def _download_close_frame(symbols: list[str], batch_size: int = 50) -> pd.DataFrame:
    """Download adjusted-close history in batches for a larger universe.

    The candidate universe is now big enough that one giant Yahoo request is
    more fragile than it needs to be. Batching keeps the offline build steadier
    while still remaining much faster than any app-load fetch path.
    """
    close_frames: list[pd.DataFrame] = []
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        price_history = yf.download(
            tickers=batch,
            period="6y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if isinstance(price_history.columns, pd.MultiIndex):
            close_batch = price_history.get("Close", pd.DataFrame()).copy()
        else:
            close_batch = price_history.rename(columns={"Close": batch[0] if len(batch) == 1 else "Close"})
        if isinstance(close_batch, pd.Series):
            close_batch = close_batch.to_frame(name=batch[0])
        close_frames.append(close_batch)

    if not close_frames:
        return pd.DataFrame()
    combined = pd.concat(close_frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined.sort_index()


def enrich_buy_candidate_universe() -> pd.DataFrame:
    """Build the enriched buy-candidate universe from the curated seed file.

    The pipeline intentionally stays lightweight:
    - refresh the saved known-universe datasets first
    - read the saved candidate list
    - fetch market history and profile details from Yahoo Finance
    - compute simple comparable features for the future buy engine
    - write both the normalized curated file and the enriched output
    """
    build_known_universe_datasets()
    entries = load_buy_candidate_universe(RAW_UNIVERSE_PATH)
    if not entries:
        raise FileNotFoundError(f"No buy-candidate universe found at {RAW_UNIVERSE_PATH}")

    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)

    raw_frame = pd.DataFrame([item.model_dump() for item in entries])
    raw_frame.to_csv(RAW_OUTPUT_PATH, index=False)

    symbols = sorted({item.market_data_symbol for item in entries} | {BENCHMARK_SYMBOL})
    close_frame = _download_close_frame(symbols)

    benchmark_history = pd.to_numeric(close_frame.get(BENCHMARK_SYMBOL), errors="coerce")
    enriched_rows: list[dict[str, Any]] = []

    for item in entries:
        history = pd.to_numeric(close_frame.get(item.market_data_symbol), errors="coerce")

        stock_1y, benchmark_1y, relative_1y = _window_relative_return(history, benchmark_history, 252)
        stock_3y, benchmark_3y, relative_3y = _window_relative_return(history, benchmark_history, 252 * 3)
        stock_5y, benchmark_5y, relative_5y = _window_relative_return(history, benchmark_history, 252 * 5)
        annualized_volatility = _annualized_volatility(history)

        current_price = None
        if history is not None and not pd.isna(history.dropna().iloc[-1] if not history.dropna().empty else None):
            current_price = float(history.dropna().iloc[-1])

        enriched_rows.append(
            {
                **item.model_dump(),
                "current_price": current_price,
                "market_cap": None,
                "beta": None,
                "industry": "",
                "country": item.region,
                "currency": "USD",
                "annualized_volatility_1y": annualized_volatility,
                "stock_1y_return_pct": stock_1y,
                "benchmark_1y_return_pct": benchmark_1y,
                "relative_1y_return_pct": relative_1y,
                "stock_3y_return_pct": stock_3y,
                "benchmark_3y_return_pct": benchmark_3y,
                "relative_3y_return_pct": relative_3y,
                "stock_5y_return_pct": stock_5y,
                "benchmark_5y_return_pct": benchmark_5y,
                "relative_5y_return_pct": relative_5y,
            }
        )

    enriched = pd.DataFrame(enriched_rows)
    enriched.to_csv(ENRICHED_OUTPUT_PATH, index=False)
    return enriched


if __name__ == "__main__":
    frame = enrich_buy_candidate_universe()
    print(f"Wrote {RAW_OUTPUT_PATH}")
    print(f"Wrote {ENRICHED_OUTPUT_PATH}")
    print(frame.head(10).to_string(index=False))

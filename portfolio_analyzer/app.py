from __future__ import annotations

import json
import math
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

try:
    from portfolio_analyzer.diagnosis import PortfolioRiskDiagnosis, portfolio_risk_diagnosis_from_saved_artifacts
except ModuleNotFoundError:
    from diagnosis import PortfolioRiskDiagnosis, portfolio_risk_diagnosis_from_saved_artifacts


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"
DIAGNOSIS_DIR = PROCESSED_DIR / "diagnosis"
DIAGNOSIS_RUNS_DIR = INTERIM_DIR / "diagnosis_runs"
DEFAULT_MODEL_NAME = "gemma:2b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
BENCHMARK_SYMBOL = "^GSPC"
PROJECTION_YEARS = 18
SHORT_HOLD_TARGET_DAYS = round(365.25 * 1.5)
ROLLING_DRAWDOWN_WINDOW_DAYS = round(365.25 / 2)
ROLLING_DRAWDOWN_MEMORY_WEIGHT = 0.25
ROLLING_DRAWDOWN_HALF_LIFE_DAYS = round(365.25)
BENCHMARK_RELATIVE_RISK_MAX_RATIO = 2.0
RELATIVE_DOWNSIDE_CAPTURE_MAX_RATIO = 1.15
RELATIVE_MARKET_SENSITIVITY_MAX_RATIO = 1.75
VERY_HIGH_CONCERN_SCORE = 80.0
RECENT_RISK_CHART_DAYS = round(365.25 * 1.5)
RISK_CHART_START_DATE = "2024-01-01"
WEEKLY_FLOW_DOMINANCE_LIMIT = 0.25
MAX_STABLE_WEEKLY_RETURN = 0.75


@dataclass
class PositionLot:
    quantity: float
    unit_cost: float
    buy_date: datetime


def parse_money(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    text = str(value).strip().replace("$", "").replace(",", "")
    if not text:
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    try:
        amount = float(text)
    except ValueError:
        return 0.0
    return -amount if negative else amount


def parse_quantity(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else 0.0


def load_transactions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    try:
        df["Activity Date"] = pd.to_datetime(df["Activity Date"], format="%m/%d/%y", errors="coerce")
    except Exception:
        df["Activity Date"] = pd.to_datetime(df["Activity Date"], errors="coerce")
    df = df.dropna(subset=["Activity Date"]).copy()
    df["Amount_num"] = df.get("Amount", 0).apply(parse_money)
    df["Price_num"] = df.get("Price", 0).apply(parse_money)
    df["Quantity_num"] = df.get("Quantity", 0).apply(parse_quantity)
    df["Instrument"] = df["Instrument"].fillna("").astype(str).str.strip()
    df["Trans Code"] = df["Trans Code"].fillna("").astype(str).str.strip()
    return df.sort_values("Activity Date").reset_index(drop=True)


def holding_period_bucket(days: float) -> str:
    if days <= 30:
        return "<=30d"
    if days < 365:
        return "31-364d"
    return ">=365d"


def weighted_average_date(dates: list[datetime], weights: list[float]) -> str | None:
    if not dates or not weights or sum(weights) <= 0:
        return None
    weighted_ordinal = sum(date.toordinal() * weight for date, weight in zip(dates, weights)) / sum(weights)
    return datetime.fromordinal(int(round(weighted_ordinal))).date().isoformat()


def weighted_median(values: list[float], weights: list[float]) -> float | None:
    if not values or not weights or len(values) != len(weights):
        return None
    pairs = sorted((float(value), float(weight)) for value, weight in zip(values, weights) if weight > 0)
    if not pairs:
        return None
    total_weight = sum(weight for _, weight in pairs)
    midpoint = total_weight / 2
    running_weight = 0.0
    for value, weight in pairs:
        running_weight += weight
        if running_weight >= midpoint:
            return value
    return pairs[-1][0]


def max_drawdown_for_series(series: pd.Series) -> float | None:
    clean_series = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if clean_series.empty or len(clean_series) < 2:
        return None
    drawdown = (clean_series / clean_series.cummax()) - 1
    return abs(float(drawdown.min()))


def score_relative_to_benchmark(
    ratio: float | None,
    *,
    max_ratio: float = BENCHMARK_RELATIVE_RISK_MAX_RATIO,
) -> float:
    if ratio is None or max_ratio <= 1:
        return 0.0
    # A ratio at or below 1 means the portfolio was no riskier than the S&P 500 on
    # that metric. Risk only ramps up once the portfolio becomes more extreme than
    # the benchmark, and reaches the cap at the chosen max ratio.
    return clip01((ratio - 1.0) / (max_ratio - 1.0))


def recency_weighted_rolling_drawdown(
    series: pd.Series,
    *,
    window_days: int = ROLLING_DRAWDOWN_WINDOW_DAYS,
    memory_weight: float = ROLLING_DRAWDOWN_MEMORY_WEIGHT,
    half_life_days: int = ROLLING_DRAWDOWN_HALF_LIFE_DAYS,
) -> dict[str, float | int | None]:
    clean_series = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if clean_series.empty or len(clean_series) < 2:
        return {
            "full_history_max_drawdown": None,
            "weighted_rolling_drawdown": None,
            "blended_drawdown": None,
            "window_days": window_days,
            "memory_weight": memory_weight,
            "recency_weight_half_life_days": half_life_days,
        }

    if not isinstance(clean_series.index, pd.DatetimeIndex):
        clean_series.index = pd.to_datetime(clean_series.index)
    clean_series = clean_series.sort_index()
    latest_date = clean_series.index[-1]
    window_delta = pd.Timedelta(days=window_days)

    rolling_drawdowns: list[float] = []
    recency_weights: list[float] = []
    for end_date in clean_series.index:
        # Use overlapping calendar windows so the metric reacts to evolving downside behavior
        # instead of being locked to arbitrary Jan-Jun / Jul-Dec buckets.
        window_slice = clean_series.loc[end_date - window_delta : end_date]
        if len(window_slice) < 2:
            continue
        window_drawdown = max_drawdown_for_series(window_slice)
        if window_drawdown is None:
            continue
        # Exponential decay keeps the weighting explainable: a window roughly one year older
        # carries about half the influence of the newest window.
        age_days = max((latest_date - end_date).days, 0)
        recency_weight = math.exp(-math.log(2) * age_days / max(half_life_days, 1))
        rolling_drawdowns.append(window_drawdown)
        recency_weights.append(recency_weight)

    full_history_drawdown = max_drawdown_for_series(clean_series)
    weighted_rolling_drawdown = None
    if rolling_drawdowns and sum(recency_weights) > 0:
        weighted_rolling_drawdown = float(
            sum(drawdown * weight for drawdown, weight in zip(rolling_drawdowns, recency_weights))
            / sum(recency_weights)
        )

    if weighted_rolling_drawdown is None:
        blended_drawdown = full_history_drawdown
    elif full_history_drawdown is None:
        blended_drawdown = weighted_rolling_drawdown
    else:
        blended_drawdown = (
            (1 - memory_weight) * weighted_rolling_drawdown
            + memory_weight * full_history_drawdown
        )

    return {
        "full_history_max_drawdown": round(full_history_drawdown, 4) if full_history_drawdown is not None else None,
        "weighted_rolling_drawdown": round(weighted_rolling_drawdown, 4)
        if weighted_rolling_drawdown is not None
        else None,
        "blended_drawdown": round(blended_drawdown, 4) if blended_drawdown is not None else None,
        "window_days": window_days,
        "memory_weight": memory_weight,
        "recency_weight_half_life_days": half_life_days,
    }


def recency_weighted_rolling_downside_capture(
    returns_frame: pd.DataFrame,
    *,
    window_days: int = ROLLING_DRAWDOWN_WINDOW_DAYS,
    half_life_days: int = ROLLING_DRAWDOWN_HALF_LIFE_DAYS,
) -> dict[str, float | int | None]:
    if returns_frame.empty:
        return {
            "weighted_downside_capture": None,
            "window_days": window_days,
            "recency_weight_half_life_days": half_life_days,
        }

    clean_returns = returns_frame.copy()
    clean_returns.index = pd.to_datetime(clean_returns.index)
    clean_returns = clean_returns.sort_index().dropna()
    if len(clean_returns) < 2 or "portfolio" not in clean_returns.columns or "benchmark" not in clean_returns.columns:
        return {
            "weighted_downside_capture": None,
            "window_days": window_days,
            "recency_weight_half_life_days": half_life_days,
        }

    latest_date = clean_returns.index[-1]
    window_delta = pd.Timedelta(days=window_days)
    rolling_ratios: list[float] = []
    recency_weights: list[float] = []

    for end_date in clean_returns.index:
        window_slice = clean_returns.loc[end_date - window_delta : end_date]
        if len(window_slice) < 2:
            continue
        downside_slice = window_slice[window_slice["benchmark"] < 0]
        if downside_slice.empty:
            continue
        benchmark_down_mean = float(downside_slice["benchmark"].mean())
        if benchmark_down_mean == 0:
            continue
        ratio = float(downside_slice["portfolio"].mean() / benchmark_down_mean)
        age_days = max((latest_date - end_date).days, 0)
        recency_weight = math.exp(-math.log(2) * age_days / max(half_life_days, 1))
        rolling_ratios.append(ratio)
        recency_weights.append(recency_weight)

    weighted_downside_capture = None
    if rolling_ratios and sum(recency_weights) > 0:
        weighted_downside_capture = float(
            sum(ratio * weight for ratio, weight in zip(rolling_ratios, recency_weights))
            / sum(recency_weights)
        )

    return {
        "weighted_downside_capture": round(weighted_downside_capture, 4)
        if weighted_downside_capture is not None
        else None,
        "window_days": window_days,
        "recency_weight_half_life_days": half_life_days,
    }


def recency_weighted_rolling_market_sensitivity(
    returns_frame: pd.DataFrame,
    *,
    window_days: int = ROLLING_DRAWDOWN_WINDOW_DAYS,
    half_life_days: int = ROLLING_DRAWDOWN_HALF_LIFE_DAYS,
) -> dict[str, float | int | None]:
    if returns_frame.empty:
        return {
            "weighted_market_sensitivity": None,
            "window_days": window_days,
            "recency_weight_half_life_days": half_life_days,
        }

    clean_returns = returns_frame.copy()
    clean_returns.index = pd.to_datetime(clean_returns.index)
    clean_returns = clean_returns.sort_index().dropna()
    if len(clean_returns) < 2 or "portfolio" not in clean_returns.columns or "benchmark" not in clean_returns.columns:
        return {
            "weighted_market_sensitivity": None,
            "window_days": window_days,
            "recency_weight_half_life_days": half_life_days,
        }

    latest_date = clean_returns.index[-1]
    window_delta = pd.Timedelta(days=window_days)
    rolling_betas: list[float] = []
    recency_weights: list[float] = []

    for end_date in clean_returns.index:
        window_slice = clean_returns.loc[end_date - window_delta : end_date]
        if len(window_slice) < 5:
            continue
        benchmark_var = float(window_slice["benchmark"].var())
        if benchmark_var <= 0:
            continue
        beta = float(window_slice["portfolio"].cov(window_slice["benchmark"]) / benchmark_var)
        age_days = max((latest_date - end_date).days, 0)
        recency_weight = math.exp(-math.log(2) * age_days / max(half_life_days, 1))
        rolling_betas.append(beta)
        recency_weights.append(recency_weight)

    weighted_market_sensitivity = None
    if rolling_betas and sum(recency_weights) > 0:
        weighted_market_sensitivity = float(
            sum(beta * weight for beta, weight in zip(rolling_betas, recency_weights))
            / sum(recency_weights)
        )

    return {
        "weighted_market_sensitivity": round(weighted_market_sensitivity, 4)
        if weighted_market_sensitivity is not None
        else None,
        "window_days": window_days,
        "recency_weight_half_life_days": half_life_days,
    }


def build_lot_analytics(df: pd.DataFrame) -> dict[str, Any]:
    lots: dict[str, deque[PositionLot]] = defaultdict(deque)
    realized_pnl: dict[str, float] = defaultdict(float)
    closed_lots: list[dict[str, Any]] = []
    share_credit_codes = {"REC"}

    for _, row in df.iterrows():
        symbol = row["Instrument"]
        code = row["Trans Code"]
        if not symbol:
            continue

        trade_date = row["Activity Date"].to_pydatetime()
        quantity = float(row["Quantity_num"])
        amount = float(row["Amount_num"])

        if code == "Buy" and quantity > 0:
            lots[symbol].append(
                PositionLot(
                    quantity=quantity,
                    unit_cost=(-amount / quantity) if quantity else 0.0,
                    buy_date=trade_date,
                )
            )
        elif code == "SPL" and quantity > 0:
            total_qty = sum(lot.quantity for lot in lots[symbol])
            if total_qty > 0:
                factor = (total_qty + quantity) / total_qty
                adjusted = deque()
                for lot in lots[symbol]:
                    adjusted.append(
                        PositionLot(
                            quantity=lot.quantity * factor,
                            unit_cost=lot.unit_cost / factor,
                            buy_date=lot.buy_date,
                        )
                    )
                lots[symbol] = adjusted
        elif code in share_credit_codes and quantity > 0:
            lots[symbol].append(PositionLot(quantity=quantity, unit_cost=0.0, buy_date=trade_date))
        elif code == "Sell" and quantity > 0:
            remaining = quantity
            sell_unit_price = amount / quantity if quantity else 0.0

            while remaining > 1e-9 and lots[symbol]:
                lot = lots[symbol][0]
                take = min(remaining, lot.quantity)
                cost_basis = take * lot.unit_cost
                proceeds = take * sell_unit_price
                pnl = proceeds - cost_basis
                realized_pnl[symbol] += pnl
                closed_lots.append(
                    {
                        "ticker": symbol,
                        "quantity": round(take, 6),
                        "buy_date": lot.buy_date.date().isoformat(),
                        "sell_date": trade_date.date().isoformat(),
                        "hold_days": (trade_date - lot.buy_date).days,
                        "unit_cost": round(lot.unit_cost, 6),
                        "sell_unit_price": round(sell_unit_price, 6),
                        "cost_basis": round(cost_basis, 2),
                        "proceeds": round(proceeds, 2),
                        "realized_pnl": round(pnl, 2),
                    }
                )
                lot.quantity -= take
                remaining -= take
                if lot.quantity <= 1e-9:
                    lots[symbol].popleft()

    current_positions = []
    open_lots: list[dict[str, Any]] = []
    for symbol, queue in lots.items():
        total_qty = sum(lot.quantity for lot in queue)
        if total_qty <= 1e-9:
            continue
        total_cost = sum(lot.quantity * lot.unit_cost for lot in queue)
        buy_dates = [lot.buy_date for lot in queue]
        weights = [lot.quantity for lot in queue]
        current_positions.append(
            {
                "ticker": symbol,
                "quantity": round(total_qty, 6),
                "cost_basis_total": round(total_cost, 2),
                "avg_cost": round(total_cost / total_qty, 6) if total_qty else 0.0,
                "first_buy_date": min(buy_dates).date().isoformat(),
                "weighted_avg_buy_date": weighted_average_date(buy_dates, weights),
            }
        )
        for lot in queue:
            open_lots.append(
                {
                    "ticker": symbol,
                    "quantity": round(lot.quantity, 6),
                    "buy_date": lot.buy_date.date().isoformat(),
                    "unit_cost": round(lot.unit_cost, 6),
                    "cost_basis": round(lot.quantity * lot.unit_cost, 2),
                }
            )

    current_positions.sort(key=lambda item: item["cost_basis_total"], reverse=True)
    return {
        "current_positions": current_positions,
        "open_lots": open_lots,
        "closed_lots": closed_lots,
        "realized_pnl_by_ticker": {
            ticker: round(float(pnl), 2)
            for ticker, pnl in sorted(realized_pnl.items(), key=lambda item: item[1], reverse=True)
        },
    }


def summarize_portfolio(df: pd.DataFrame, lot_data: dict[str, Any]) -> dict[str, Any]:
    buy_df = df[df["Trans Code"] == "Buy"]
    sell_df = df[df["Trans Code"] == "Sell"]
    ach_df = df[df["Trans Code"] == "ACH"]
    dividend_df = df[df["Trans Code"] == "CDIV"]
    interest_df = df[df["Trans Code"] == "INT"]
    as_of_date = pd.Timestamp.today().normalize()

    holding_periods = [lot["hold_days"] for lot in lot_data["closed_lots"]]
    holding_counter = Counter(holding_period_bucket(days) for days in holding_periods)
    holding_duration_values: list[float] = []
    holding_duration_weights: list[float] = []
    for lot in lot_data["closed_lots"]:
        cost_basis = float(lot.get("cost_basis") or 0.0)
        hold_days = float(lot.get("hold_days") or 0.0)
        if cost_basis > 0:
            holding_duration_values.append(hold_days)
            holding_duration_weights.append(cost_basis)
    for lot in lot_data.get("open_lots", []):
        cost_basis = float(lot.get("cost_basis") or 0.0)
        if cost_basis <= 0:
            continue
        buy_date = pd.Timestamp(lot["buy_date"]).normalize()
        holding_duration_values.append(float((as_of_date - buy_date).days))
        holding_duration_weights.append(cost_basis)

    years = max(((df["Activity Date"].max() - df["Activity Date"].min()).days / 365.25), 0.01)

    buys_by_ticker = buy_df.groupby("Instrument")["Amount_num"].sum().abs().sort_values(ascending=False)
    sells_by_ticker = sell_df.groupby("Instrument")["Amount_num"].sum().sort_values(ascending=False)

    return {
        "date_range": {
            "start": df["Activity Date"].min().date().isoformat(),
            "end": df["Activity Date"].max().date().isoformat(),
            "years": round(years, 2),
        },
        "transaction_counts": {
            "rows": int(len(df)),
            "buys": int(len(buy_df)),
            "sells": int(len(sell_df)),
            "distinct_symbols_traded": int(df.loc[df["Instrument"].ne(""), "Instrument"].nunique()),
        },
        "cash_flows": {
            "net_deposits": round(float(ach_df["Amount_num"].sum()), 2),
            "gross_buys": round(float(buy_df["Amount_num"].sum() * -1), 2),
            "gross_sells": round(float(sell_df["Amount_num"].sum()), 2),
            "dividends": round(float(dividend_df["Amount_num"].sum()), 2),
            "interest": round(float(interest_df["Amount_num"].sum()), 2),
            "cash_balance_estimate": round(float(df["Amount_num"].sum()), 2),
        },
        "behavioral_metrics": {
            "sell_to_buy_ratio": round(
                float(sell_df["Amount_num"].sum()) / max(float(buy_df["Amount_num"].sum() * -1), 1.0),
                3,
            ),
            "annualized_turnover_proxy": round(
                (float(sell_df["Amount_num"].sum()) / max(float(buy_df["Amount_num"].sum() * -1), 1.0))
                / years,
                3,
            ),
            "holding_period_buckets": dict(holding_counter),
            # Weight holding duration by deployed capital so small speculative trades do not
            # dominate the behavior signal for much larger long-term positions.
            "capital_weighted_median_holding_period_days": round(
                float(weighted_median(holding_duration_values, holding_duration_weights)),
                1,
            )
            if holding_duration_values
            else None,
            "median_holding_period_days": round(float(pd.Series(holding_periods).median()), 1)
            if holding_periods
            else None,
        },
        "current_positions": lot_data["current_positions"],
        "current_positions_top_15": lot_data["current_positions"][:15],
        "top_deployed_capital": [
            {
                "ticker": ticker,
                "gross_buy_amount": round(float(amount * -1), 2),
                "gross_sell_amount": round(float(sells_by_ticker.get(ticker, 0.0)), 2),
                "realized_pnl": round(float(lot_data["realized_pnl_by_ticker"].get(ticker, 0.0)), 2),
            }
            for ticker, amount in buys_by_ticker.head(15).items()
        ],
        "realized_pnl_by_ticker": [
            {"ticker": ticker, "realized_pnl": pnl}
            for ticker, pnl in lot_data["realized_pnl_by_ticker"].items()
        ],
    }


def extract_close_frame(downloaded: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame(columns=tickers)

    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" in downloaded.columns.get_level_values(0):
            close = downloaded["Close"].copy()
        else:
            close = downloaded.xs("Close", axis=1, level=-1).copy()
    else:
        close = downloaded[["Close"]].copy()
        close.columns = tickers[:1]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    close = close.apply(pd.to_numeric, errors="coerce")
    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def normalize_ticker_for_yahoo(symbol: str) -> str:
    return str(symbol).strip().replace(".", "-")


@lru_cache(maxsize=512)
def fetch_yahoo_sector_label(yahoo_symbol: str) -> str:
    try:
        info = yf.Ticker(yahoo_symbol).info or {}
    except Exception:
        return "Unclassified"
    sector = info.get("sectorDisp") or info.get("sector")
    if sector:
        return str(sector)
    quote_type = str(info.get("quoteType") or "").upper()
    if quote_type in {"ETF", "MUTUALFUND", "INDEX", "FUND"}:
        return "ETF / Fund"
    return "Unclassified"


def fetch_market_data(traded_symbols: list[str], benchmark_symbol: str, start_date: str) -> dict[str, Any]:
    tickers = sorted(set(symbol for symbol in traded_symbols if symbol))
    yahoo_map = {symbol: normalize_ticker_for_yahoo(symbol) for symbol in tickers}
    yahoo_tickers = [yahoo_map[symbol] for symbol in tickers]
    history_start = (pd.Timestamp(start_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    long_horizon_start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=5) - pd.Timedelta(days=15)).strftime("%Y-%m-%d")

    if yahoo_tickers:
        ticker_history_raw = yf.download(
            tickers=yahoo_tickers,
            start=history_start,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        ticker_close = extract_close_frame(ticker_history_raw, yahoo_tickers)
        ticker_close = ticker_close.rename(
            columns={yahoo_symbol: original for original, yahoo_symbol in yahoo_map.items()}
        )

        long_horizon_history_raw = yf.download(
            tickers=yahoo_tickers,
            start=long_horizon_start,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        long_horizon_ticker_close = extract_close_frame(long_horizon_history_raw, yahoo_tickers)
        long_horizon_ticker_close = long_horizon_ticker_close.rename(
            columns={yahoo_symbol: original for original, yahoo_symbol in yahoo_map.items()}
        )
    else:
        ticker_close = pd.DataFrame()
        long_horizon_ticker_close = pd.DataFrame()

    benchmark_raw = yf.download(
        tickers=benchmark_symbol,
        start=history_start,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    benchmark_close = extract_close_frame(benchmark_raw, [benchmark_symbol]).iloc[:, 0].dropna()
    benchmark_close.name = benchmark_symbol

    benchmark_long_raw = yf.download(
        tickers=benchmark_symbol,
        start=long_horizon_start,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    benchmark_long_close = extract_close_frame(benchmark_long_raw, [benchmark_symbol]).iloc[:, 0].dropna()
    benchmark_long_close.name = benchmark_symbol

    latest_prices = ticker_close.ffill().iloc[-1].dropna().to_dict() if not ticker_close.empty else {}
    latest_benchmark_price = float(benchmark_close.iloc[-1]) if not benchmark_close.empty else None
    sector_by_ticker = {symbol: fetch_yahoo_sector_label(yahoo_symbol) for symbol, yahoo_symbol in yahoo_map.items()}

    return {
        "ticker_close": ticker_close,
        "latest_prices": latest_prices,
        "benchmark_close": benchmark_close,
        "long_horizon_ticker_close": long_horizon_ticker_close,
        "long_horizon_benchmark_close": benchmark_long_close,
        "latest_benchmark_price": latest_benchmark_price,
        "sector_by_ticker": sector_by_ticker,
    }


def next_market_date(index: pd.Index, dt: Any) -> pd.Timestamp:
    ts = pd.Timestamp(dt).normalize()
    pos = index.searchsorted(ts)
    if pos >= len(index):
        pos = len(index) - 1
    return index[pos]


def price_return_from_date(price_series: pd.Series, start_date: Any) -> float | None:
    if price_series.empty:
        return None
    start_ts = next_market_date(price_series.index, start_date)
    start_price = float(price_series.loc[start_ts])
    end_price = float(price_series.iloc[-1])
    if start_price <= 0:
        return None
    return (end_price / start_price) - 1


def price_return_between_dates(price_series: pd.Series, start_date: Any, end_date: Any) -> float | None:
    if price_series.empty:
        return None
    start_ts = next_market_date(price_series.index, start_date)
    end_ts = next_market_date(price_series.index, end_date)
    start_price = float(price_series.loc[start_ts])
    end_price = float(price_series.loc[end_ts])
    if start_price <= 0:
        return None
    return (end_price / start_price) - 1


def trailing_price_return(price_series: pd.Series, years: int) -> float | None:
    if price_series.empty:
        return None
    end_ts = pd.Timestamp(price_series.index.max()).normalize()
    start_target = end_ts - pd.DateOffset(years=years)
    if pd.Timestamp(price_series.index.min()).normalize() > start_target:
        return None
    return price_return_between_dates(price_series, start_target, end_ts)


def build_daily_share_matrix(df: pd.DataFrame, market_index: pd.Index, tickers: list[str]) -> pd.DataFrame:
    events: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        symbol = row["Instrument"]
        code = row["Trans Code"]
        qty = float(row["Quantity_num"])
        if not symbol or symbol not in tickers or qty <= 0:
            continue

        delta = 0.0
        if code == "Buy":
            delta = qty
        elif code == "Sell":
            delta = -qty
        elif code == "SPL":
            delta = qty
        elif code == "REC":
            delta = qty

        if delta == 0.0:
            continue

        events.append(
            {"date": next_market_date(market_index, row["Activity Date"]), "ticker": symbol, "delta": delta}
        )

    if not events:
        return pd.DataFrame(index=market_index, columns=tickers).fillna(0.0)

    event_df = pd.DataFrame(events)
    share_changes = event_df.pivot_table(index="date", columns="ticker", values="delta", aggfunc="sum")
    # Fill sparse event-day holes before the cumulative sum so untouched holdings carry
    # forward cleanly instead of disappearing for one day in the valuation chart.
    share_changes = share_changes.reindex(market_index, fill_value=0.0).fillna(0.0)
    share_matrix = share_changes.cumsum()
    return share_matrix.reindex(columns=tickers, fill_value=0.0)


def build_cash_balance_series(df: pd.DataFrame, market_index: pd.Index) -> pd.Series:
    cash_events = [
        {"date": next_market_date(market_index, row["Activity Date"]), "amount": float(row["Amount_num"])}
        for _, row in df.iterrows()
    ]
    cash_df = pd.DataFrame(cash_events)
    cash_changes = cash_df.groupby("date")["amount"].sum().reindex(market_index, fill_value=0.0)
    return cash_changes.cumsum()


def build_trade_matched_benchmark_series(df: pd.DataFrame, benchmark_close: pd.Series) -> tuple[pd.Series, float]:
    trade_flows = df[df["Trans Code"].isin(["Buy", "Sell"])].copy()
    share_events = pd.Series(0.0, index=benchmark_close.index)

    for _, row in trade_flows.iterrows():
        amount = float(row["Amount_num"])
        if amount == 0:
            continue
        trade_date = next_market_date(benchmark_close.index, row["Activity Date"])
        benchmark_price = float(benchmark_close.loc[trade_date])
        if benchmark_price <= 0:
            continue
        if row["Trans Code"] == "Buy":
            share_events.loc[trade_date] += (-amount) / benchmark_price
        elif row["Trans Code"] == "Sell":
            share_events.loc[trade_date] -= amount / benchmark_price

    shares = share_events.cumsum()
    benchmark_value_series = shares * benchmark_close
    invested_net_cash = -float(trade_flows["Amount_num"].sum())
    return benchmark_value_series, invested_net_cash


def build_trade_flow_series(df: pd.DataFrame, market_index: pd.Index) -> pd.Series:
    trade_events = [
        # Buying moves cash into the invested sleeve; selling moves it back out.
        {"date": next_market_date(market_index, row["Activity Date"]), "flow": -float(row["Amount_num"])}
        for _, row in df[df["Trans Code"].isin(["Buy", "Sell"])].iterrows()
    ]
    if not trade_events:
        return pd.Series(0.0, index=market_index)
    trade_df = pd.DataFrame(trade_events)
    return trade_df.groupby("date")["flow"].sum().reindex(market_index, fill_value=0.0).astype(float)


def build_flow_adjusted_performance_index(
    value_series: pd.Series,
    flow_series: pd.Series,
    *,
    min_start_value: float = 1000.0,
) -> pd.Series:
    clean_values = pd.to_numeric(value_series, errors="coerce").astype(float)
    clean_flows = pd.to_numeric(flow_series, errors="coerce").reindex(clean_values.index, fill_value=0.0).astype(float)
    if clean_values.empty:
        return pd.Series(dtype=float)

    returns = pd.Series(0.0, index=clean_values.index, dtype=float)
    started = False
    for idx in range(1, len(clean_values)):
        prev_value = float(clean_values.iloc[idx - 1])
        current_value = float(clean_values.iloc[idx])
        trade_flow = float(clean_flows.iloc[idx])
        if not started:
            if prev_value >= min_start_value:
                started = True
            else:
                continue
        if prev_value <= 0:
            continue
        returns.iloc[idx] = (current_value - prev_value - trade_flow) / prev_value

    wealth_index = (1 + returns).cumprod()
    wealth_index.iloc[0] = 1.0
    start_candidates = clean_values[clean_values >= min_start_value]
    if not start_candidates.empty:
        first_start = start_candidates.index[0]
        wealth_index = wealth_index.loc[first_start:]
        wealth_index = wealth_index / wealth_index.iloc[0]
    return wealth_index.astype(float)


def annualized_return(final_value: float, initial_value: float, years: float) -> float | None:
    if years <= 0 or initial_value <= 0 or final_value <= 0:
        return None
    return (final_value / initial_value) ** (1 / years) - 1


def build_holding_performance_context(
    open_positions_df: pd.DataFrame,
    market_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build trailing 1Y/3Y/5Y return context for current holdings.

    The recommendation layer needs a broader lens than "since you bought it".
    This dataset answers whether a holding that has lagged since purchase has
    also lagged the S&P 500 across broader trailing windows.
    """
    ticker_close = market_data.get("long_horizon_ticker_close", pd.DataFrame())
    benchmark_close = market_data.get("long_horizon_benchmark_close", pd.Series(dtype=float))
    if open_positions_df.empty or benchmark_close.empty:
        return []

    benchmark_window_returns = {
        years: trailing_price_return(benchmark_close, years)
        for years in (1, 3, 5)
    }
    as_of_date = pd.Timestamp(benchmark_close.index.max()).date().isoformat()
    rows: list[dict[str, Any]] = []
    for _, row in open_positions_df.iterrows():
        ticker = row.get("ticker")
        if not ticker or ticker not in ticker_close.columns:
            continue
        series = ticker_close[ticker].dropna()
        if series.empty:
            continue
        context_row: dict[str, Any] = {"ticker": ticker, "as_of_date": as_of_date}
        for years in (1, 3, 5):
            stock_return = trailing_price_return(series, years)
            benchmark_return = benchmark_window_returns.get(years)
            context_row[f"stock_{years}y_return_pct"] = stock_return
            context_row[f"benchmark_{years}y_return_pct"] = benchmark_return
            context_row[f"relative_{years}y_return_pct"] = (
                stock_return - benchmark_return
                if stock_return is not None and benchmark_return is not None
                else None
            )
        rows.append(context_row)
    return rows


def xnpv(rate: float, cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    if not cashflows:
        return 0.0
    start_date = min(dt for dt, _ in cashflows)
    return sum(amount / ((1 + rate) ** (((dt - start_date).days) / 365.25)) for dt, amount in cashflows)


def xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float | None:
    if len(cashflows) < 2:
        return None
    amounts = [amount for _, amount in cashflows]
    if not (any(amount < 0 for amount in amounts) and any(amount > 0 for amount in amounts)):
        return None

    low, high = -0.9999, 1.0
    f_low = xnpv(low, cashflows)
    f_high = xnpv(high, cashflows)

    expand_count = 0
    while f_low * f_high > 0 and expand_count < 50:
        high *= 2
        f_high = xnpv(high, cashflows)
        expand_count += 1
    if f_low * f_high > 0:
        return None

    for _ in range(200):
        mid = (low + high) / 2
        f_mid = xnpv(mid, cashflows)
        if abs(f_mid) < 1e-7:
            return mid
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return (low + high) / 2


def aggregate_cashflows(cashflows: list[tuple[pd.Timestamp, float]]) -> list[tuple[pd.Timestamp, float]]:
    if not cashflows:
        return []
    series = defaultdict(float)
    for dt, amount in cashflows:
        series[pd.Timestamp(dt).normalize()] += float(amount)
    return sorted(series.items(), key=lambda item: item[0])


def clip01(value: float | None) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def risk_band(score: float) -> str:
    if score <= 20:
        return "Very Conservative"
    if score <= 40:
        return "Conservative"
    if score <= 60:
        return "Moderate"
    if score <= 80:
        return "Aggressive"
    return "Very Aggressive"


def confidence_band(score: float) -> str:
    if score < 0.35:
        return "Low"
    if score < 0.7:
        return "Medium"
    return "High"


def compute_observed_risk_score(
    *,
    stated_risk_score: int,
    max_position_weight: float,
    top_5_weight: float,
    effective_holdings: float | None,
    meaningful_holdings_count: int,
    annualized_turnover: float | None,
    capital_weighted_holding_days: float | None,
    equity_exposure: float | None,
    relative_drawdown_to_benchmark: float | None,
    relative_downside_capture_to_benchmark: float | None,
    relative_volatility_to_benchmark: float | None,
    relative_market_sensitivity_to_benchmark: float | None,
    years_of_history: float,
    closed_lot_count: int,
) -> dict[str, Any]:
    # Score effective holdings relative to the number of meaningful positions rather than
    # against a universal threshold. This keeps the metric explainable and portfolio-size aware.
    effective_holdings_value = effective_holdings or 0.0
    meaningful_holdings_value = max(int(meaningful_holdings_count or 0), 1)
    effective_holdings_ratio = min(effective_holdings_value, meaningful_holdings_value) / meaningful_holdings_value

    concentration_components = {
        "single_position_weight": clip01((max_position_weight or 0.0) / 0.22),
        "top_5_weight": clip01((top_5_weight or 0.0) / 0.65),
        "effective_holdings": clip01(1 - effective_holdings_ratio),
    }
    market_components = {
        # Volatility and drawdown are benchmark-relative so the portfolio is only
        # penalized for being riskier than the S&P 500 over the same horizon.
        "relative_volatility_to_benchmark": score_relative_to_benchmark(relative_volatility_to_benchmark),
        "relative_drawdown_to_benchmark": score_relative_to_benchmark(relative_drawdown_to_benchmark),
        "relative_downside_capture_to_benchmark": score_relative_to_benchmark(
            relative_downside_capture_to_benchmark,
            max_ratio=RELATIVE_DOWNSIDE_CAPTURE_MAX_RATIO,
        ),
        "relative_market_sensitivity_to_benchmark": score_relative_to_benchmark(
            relative_market_sensitivity_to_benchmark,
            max_ratio=RELATIVE_MARKET_SENSITIVITY_MAX_RATIO,
        ),
        "equity_exposure": clip01((equity_exposure or 0.0) / 1.0),
    }
    behavior_components = {
        "turnover": clip01((annualized_turnover or 0.0) / 0.50),
        # Use an 18-month target so this behavior score reflects how long the investor's
        # actual dollars are held, not just how many tiny trades were flipped quickly.
        "short_holding_period": clip01(
            (SHORT_HOLD_TARGET_DAYS - (capital_weighted_holding_days or SHORT_HOLD_TARGET_DAYS))
            / SHORT_HOLD_TARGET_DAYS
        ),
    }
    dimension_scores = {
        "concentration_risk": sum(concentration_components.values()) / len(concentration_components),
        "market_risk": sum(market_components.values()) / len(market_components),
        "behavioral_risk": sum(behavior_components.values()) / len(behavior_components),
    }
    dimension_weights = {
        "concentration_risk": 0.4,
        "market_risk": 0.4,
        "behavioral_risk": 0.2,
    }
    weighted_score = sum(dimension_scores[name] * dimension_weights[name] for name in dimension_scores)
    observed_risk_score = round(weighted_score * 100, 1)
    difference = round(observed_risk_score - stated_risk_score, 1)
    alignment_score = round(max(0.0, 100 - abs(difference) * 2), 1)

    if abs(difference) < 8:
        alignment = "Aligned"
    elif difference >= 8:
        alignment = "Observed portfolio risk is higher than stated risk"
    else:
        alignment = "Observed portfolio risk is lower than stated risk"

    confidence_score = round(
        min(1.0, (years_of_history / 3.0) * 0.5 + (closed_lot_count / 25.0) * 0.5),
        3,
    )
    component_raw_values = {
        "concentration::single_position_weight": max_position_weight,
        "concentration::top_5_weight": top_5_weight,
        "concentration::effective_holdings": effective_holdings_value if effective_holdings is not None else None,
        "market::relative_volatility_to_benchmark": relative_volatility_to_benchmark,
        "market::relative_drawdown_to_benchmark": relative_drawdown_to_benchmark,
        "market::relative_downside_capture_to_benchmark": relative_downside_capture_to_benchmark,
        "market::relative_market_sensitivity_to_benchmark": relative_market_sensitivity_to_benchmark,
        "market::equity_exposure": equity_exposure,
        "behavior::turnover": annualized_turnover,
        "behavior::short_holding_period": capital_weighted_holding_days,
    }
    return {
        "score": observed_risk_score,
        "band": risk_band(observed_risk_score),
        "stated_score": stated_risk_score,
        "stated_band": risk_band(stated_risk_score),
        "difference_vs_stated": difference,
        "alignment": alignment,
        "alignment_score": alignment_score,
        "confidence_score": confidence_score,
        "confidence_band": confidence_band(confidence_score),
        "effective_holdings_value": round(effective_holdings_value, 2) if effective_holdings is not None else None,
        "meaningful_holdings_count": meaningful_holdings_value,
        "short_holding_period_target_days": SHORT_HOLD_TARGET_DAYS,
        "dimension_scores": {k: round(v * 100, 1) for k, v in dimension_scores.items()},
        "component_scores": {
            **{f"concentration::{k}": round(v * 100, 1) for k, v in concentration_components.items()},
            **{f"market::{k}": round(v * 100, 1) for k, v in market_components.items()},
            **{f"behavior::{k}": round(v * 100, 1) for k, v in behavior_components.items()},
        },
        "component_raw_values": {
            key: round(float(value), 4) if value is not None and not pd.isna(value) else None
            for key, value in component_raw_values.items()
        },
    }


def build_market_enriched_metrics(
    df: pd.DataFrame,
    portfolio_summary: dict[str, Any],
    lot_data: dict[str, Any],
    market_data: dict[str, Any],
    benchmark_symbol: str,
    projection_years: int,
    stated_risk_score: int,
) -> dict[str, Any]:
    ticker_close = market_data["ticker_close"]
    latest_prices = market_data["latest_prices"]
    benchmark_close = market_data["benchmark_close"]
    sector_by_ticker = market_data.get("sector_by_ticker", {})

    if benchmark_close.empty:
        raise RuntimeError(f"No benchmark history returned for {benchmark_symbol}")

    open_positions_df = pd.DataFrame(lot_data["current_positions"]).copy()
    if open_positions_df.empty:
        open_positions_df = pd.DataFrame(
            columns=[
                "ticker",
                "quantity",
                "cost_basis_total",
                "avg_cost",
                "first_buy_date",
                "weighted_avg_buy_date",
            ]
        )

    open_positions_df["latest_price"] = open_positions_df["ticker"].map(latest_prices)
    open_positions_df["sector"] = open_positions_df["ticker"].map(sector_by_ticker).fillna("Unclassified")
    open_positions_df["current_value"] = open_positions_df["quantity"] * open_positions_df["latest_price"]
    open_positions_df["unrealized_pnl"] = open_positions_df["current_value"] - open_positions_df["cost_basis_total"]
    open_positions_df["unrealized_return_pct"] = (
        open_positions_df["unrealized_pnl"] / open_positions_df["cost_basis_total"]
    )
    open_positions_df["benchmark_return_since_buy"] = open_positions_df["weighted_avg_buy_date"].apply(
        lambda d: price_return_from_date(benchmark_close, d) if pd.notna(d) else None
    )
    open_positions_df["excess_return_vs_benchmark"] = (
        open_positions_df["unrealized_return_pct"] - open_positions_df["benchmark_return_since_buy"]
    )
    holding_performance_context = build_holding_performance_context(open_positions_df, market_data)

    current_portfolio_value = float(open_positions_df["current_value"].fillna(0.0).sum())
    total_unrealized_pnl = float(open_positions_df["unrealized_pnl"].fillna(0.0).sum())
    total_realized_pnl = float(sum(lot_data["realized_pnl_by_ticker"].values()))
    raw_cash_balance_estimate = float(df["Amount_num"].sum())
    uninvested_cash_estimate = max(raw_cash_balance_estimate, 0.0)
    total_account_value_estimate = current_portfolio_value + raw_cash_balance_estimate
    net_deposits = float(df.loc[df["Trans Code"] == "ACH", "Amount_num"].sum())
    total_profit_estimate = total_account_value_estimate - net_deposits
    total_return_estimate = (total_profit_estimate / net_deposits) if net_deposits else None

    invested_net_cash_estimate = -float(df.loc[df["Trans Code"].isin(["Buy", "Sell"]), "Amount_num"].sum())
    invested_only_profit_estimate = current_portfolio_value - invested_net_cash_estimate
    invested_only_return_estimate = (
        invested_only_profit_estimate / invested_net_cash_estimate if invested_net_cash_estimate else None
    )

    if current_portfolio_value > 0:
        open_positions_df["current_weight"] = open_positions_df["current_value"] / current_portfolio_value
    else:
        open_positions_df["current_weight"] = 0.0

    sector_allocation = pd.DataFrame(columns=["sector", "current_value", "weight_pct", "excess_return_vs_benchmark"])
    if not open_positions_df.empty:
        sector_rows = open_positions_df[["sector", "current_value", "excess_return_vs_benchmark"]].copy()
        sector_rows["weighted_excess_value"] = (
            sector_rows["current_value"].fillna(0.0) * sector_rows["excess_return_vs_benchmark"].fillna(0.0)
        )
        sector_allocation = (
            sector_rows.groupby("sector", dropna=False)[["current_value", "weighted_excess_value"]]
            .sum()
            .reset_index()
        )
        sector_allocation["weight_pct"] = (
            sector_allocation["current_value"] / current_portfolio_value if current_portfolio_value > 0 else 0.0
        )
        sector_allocation["excess_return_vs_benchmark"] = sector_allocation.apply(
            lambda row: row["weighted_excess_value"] / row["current_value"] if row["current_value"] > 0 else None,
            axis=1,
        )
        sector_allocation = sector_allocation.sort_values("current_value", ascending=False).reset_index(drop=True)

    best_sector_row = None
    valid_sector_alpha = sector_allocation.dropna(subset=["excess_return_vs_benchmark"])
    if not valid_sector_alpha.empty:
        best_sector_row = valid_sector_alpha.sort_values("excess_return_vs_benchmark", ascending=False).iloc[0]

    hhi = float((open_positions_df["current_weight"].fillna(0.0) ** 2).sum())
    max_position_weight = float(open_positions_df["current_weight"].max()) if not open_positions_df.empty else 0.0
    effective_holdings = (1.0 / hhi) if hhi > 0 else None
    # Treat positions at or above 1% weight as meaningful so tiny dust positions do not
    # artificially inflate the diversification baseline used by the explainable risk score.
    meaningful_holdings_count = int((open_positions_df["current_weight"].fillna(0.0) >= 0.01).sum())
    top_5_weight = (
        float(open_positions_df.nlargest(5, "current_value")["current_weight"].sum())
        if not open_positions_df.empty
        else 0.0
    )
    concentration_adjusted_return_proxy = (
        invested_only_return_estimate * (1 - hhi) if invested_only_return_estimate is not None else None
    )

    benchmark_value_series, benchmark_invested_net_cash = build_trade_matched_benchmark_series(df, benchmark_close)
    benchmark_current_value = float(benchmark_value_series.iloc[-1]) if not benchmark_value_series.empty else 0.0
    benchmark_return_estimate = (
        (benchmark_current_value - benchmark_invested_net_cash) / benchmark_invested_net_cash
        if benchmark_invested_net_cash
        else None
    )
    excess_return_vs_benchmark = (
        invested_only_return_estimate - benchmark_return_estimate
        if invested_only_return_estimate is not None and benchmark_return_estimate is not None
        else None
    )

    market_index = benchmark_close.index
    tracked_tickers = sorted(ticker_close.columns.tolist()) if not ticker_close.empty else []
    price_matrix = ticker_close.reindex(market_index).ffill().reindex(columns=tracked_tickers)
    price_matrix = price_matrix.apply(pd.to_numeric, errors="coerce").astype(float)
    share_matrix = build_daily_share_matrix(df, market_index, tracked_tickers)
    share_matrix = share_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    cash_balance_series = pd.to_numeric(build_cash_balance_series(df, market_index), errors="coerce").fillna(0.0)
    benchmark_value_series = pd.to_numeric(benchmark_value_series, errors="coerce").fillna(0.0)
    portfolio_equity_series = (share_matrix * price_matrix).sum(axis=1).astype(float)
    account_value_series = (portfolio_equity_series + cash_balance_series).astype(float)

    active_mask = portfolio_equity_series.ne(0) | benchmark_value_series.ne(0)
    if active_mask.any():
        first_active = active_mask[active_mask].index[0]
        portfolio_equity_series = portfolio_equity_series.loc[first_active:]
        account_value_series = account_value_series.loc[first_active:]
        benchmark_value_series = benchmark_value_series.loc[first_active:]
    trade_flow_series = build_trade_flow_series(df, portfolio_equity_series.index)

    # Build flow-adjusted wealth indices so market-risk metrics reflect market movement rather
    # than capital deployment between cash and invested positions.
    portfolio_volatility_floor = max(1000.0, current_portfolio_value * 0.02) if current_portfolio_value > 0 else 1000.0
    benchmark_volatility_floor = (
        max(1000.0, benchmark_current_value * 0.02) if benchmark_current_value > 0 else 1000.0
    )
    portfolio_performance_index = build_flow_adjusted_performance_index(
        portfolio_equity_series,
        trade_flow_series,
        min_start_value=portfolio_volatility_floor,
    )
    benchmark_performance_index = build_flow_adjusted_performance_index(
        benchmark_value_series,
        trade_flow_series,
        min_start_value=benchmark_volatility_floor,
    )
    aligned_performance = pd.concat(
        [
            portfolio_performance_index.rename("portfolio"),
            benchmark_performance_index.rename("benchmark"),
        ],
        axis=1,
    ).dropna()
    performance_returns = (
        aligned_performance.pct_change(fill_method=None).replace([math.inf, -math.inf], float("nan")).dropna()
    )
    portfolio_returns = performance_returns["portfolio"] if not performance_returns.empty else pd.Series(dtype=float)
    benchmark_returns = performance_returns["benchmark"] if not performance_returns.empty else pd.Series(dtype=float)
    # Volatility now uses the more robust weekly value-based return method that also powers
    # the evidence chart, so the score and the proof view tell the same story.
    volatility_weekly_returns = build_stable_weekly_value_return_frame(
        pd.DataFrame(
            {
                "date": portfolio_equity_series.index,
                "portfolio_invested_value": portfolio_equity_series.values,
                "benchmark_value": benchmark_value_series.reindex(portfolio_equity_series.index).values,
                "trade_flow": trade_flow_series.reindex(portfolio_equity_series.index).values,
            }
        ).to_dict(orient="records")
    )
    rolling_volatility_frame = build_rolling_relative_volatility_frame(volatility_weekly_returns)
    recent_rolling_volatility_frame = trim_to_recent_window(rolling_volatility_frame, "date", RECENT_RISK_CHART_DAYS)
    aligned_returns = performance_returns
    portfolio_drawdown = (aligned_performance["portfolio"] / aligned_performance["portfolio"].cummax()) - 1
    benchmark_drawdown = (aligned_performance["benchmark"] / aligned_performance["benchmark"].cummax()) - 1
    rolling_drawdown_frame = build_rolling_relative_drawdown_frame(volatility_weekly_returns)
    recent_rolling_drawdown_frame = trim_to_recent_window(rolling_drawdown_frame, "date", RECENT_RISK_CHART_DAYS)
    rolling_market_sensitivity_frame = build_rolling_relative_market_sensitivity_frame(volatility_weekly_returns)
    recent_rolling_market_sensitivity_frame = trim_to_recent_window(
        rolling_market_sensitivity_frame, "date", RECENT_RISK_CHART_DAYS
    )
    # Blend overlapping 6-month drawdowns with recency weights so old crises matter less
    # than the portfolio's more recent downside behavior.
    drawdown_profile = recency_weighted_rolling_drawdown(aligned_performance["portfolio"])
    benchmark_drawdown_profile = recency_weighted_rolling_drawdown(aligned_performance["benchmark"])
    if not recent_rolling_volatility_frame.empty:
        latest_volatility_row = recent_rolling_volatility_frame.iloc[-1]
        annualized_volatility = float(latest_volatility_row["portfolio_volatility"])
        benchmark_annualized_volatility = float(latest_volatility_row["benchmark_volatility"])
        relative_volatility_to_benchmark = float(latest_volatility_row["ratio"])
    else:
        annualized_volatility = None
        benchmark_annualized_volatility = None
        relative_volatility_to_benchmark = None
    if not recent_rolling_drawdown_frame.empty:
        latest_drawdown_row = recent_rolling_drawdown_frame.iloc[-1]
        relative_drawdown_to_benchmark = float(latest_drawdown_row["ratio"])
    else:
        relative_drawdown_to_benchmark = None
    downside_capture_profile = recency_weighted_rolling_downside_capture(aligned_returns)
    relative_downside_capture_to_benchmark = downside_capture_profile["weighted_downside_capture"]
    market_sensitivity_profile = recency_weighted_rolling_market_sensitivity(aligned_returns)
    if not recent_rolling_market_sensitivity_frame.empty:
        latest_market_sensitivity_row = recent_rolling_market_sensitivity_frame.iloc[-1]
        relative_market_sensitivity_to_benchmark = float(latest_market_sensitivity_row["ratio"])
    else:
        relative_market_sensitivity_to_benchmark = None
    tracking_error = None
    info_ratio_proxy = None
    if len(aligned_returns) > 5 and aligned_returns["benchmark"].var() > 0:
        active_return_series = aligned_returns["portfolio"] - aligned_returns["benchmark"]
        tracking_error = (
            float(active_return_series.std() * math.sqrt(252)) if len(active_return_series) > 2 else None
        )
        if tracking_error and tracking_error > 0:
            info_ratio_proxy = float((active_return_series.mean() * 252) / tracking_error)

    worst_benchmark_date = benchmark_drawdown.idxmin() if not benchmark_drawdown.empty else None
    portfolio_drawdown_on_worst_benchmark_day = (
        float(portfolio_drawdown.loc[worst_benchmark_date])
        if worst_benchmark_date is not None and worst_benchmark_date in portfolio_drawdown.index
        else None
    )

    years = float(portfolio_summary["date_range"]["years"])
    portfolio_cagr = annualized_return(total_account_value_estimate, net_deposits, years)
    invested_only_cagr = annualized_return(current_portfolio_value, invested_net_cash_estimate, years)
    benchmark_cagr = annualized_return(benchmark_current_value, benchmark_invested_net_cash, years)
    capital_weighted_holding_days = portfolio_summary["behavioral_metrics"].get(
        "capital_weighted_median_holding_period_days"
    )
    annualized_turnover = portfolio_summary["behavioral_metrics"].get("annualized_turnover_proxy")
    equity_exposure = current_portfolio_value / total_account_value_estimate if total_account_value_estimate > 0 else 0.0

    today = pd.Timestamp.today().normalize()
    account_cashflows = aggregate_cashflows(
        [(row["Activity Date"], -float(row["Amount_num"])) for _, row in df[df["Trans Code"] == "ACH"].iterrows()]
        + [(today, total_account_value_estimate)]
    )
    invested_cashflows = aggregate_cashflows(
        [(row["Activity Date"], float(row["Amount_num"])) for _, row in df[df["Trans Code"].isin(["Buy", "Sell"])].iterrows()]
        + [(today, current_portfolio_value)]
    )
    benchmark_cashflows = aggregate_cashflows(
        [(row["Activity Date"], float(row["Amount_num"])) for _, row in df[df["Trans Code"].isin(["Buy", "Sell"])].iterrows()]
        + [(today, benchmark_current_value)]
    )

    account_xirr = xirr(account_cashflows)
    invested_only_xirr = xirr(invested_cashflows)
    benchmark_xirr = xirr(benchmark_cashflows)
    excess_xirr_vs_benchmark = (
        invested_only_xirr - benchmark_xirr
        if invested_only_xirr is not None and benchmark_xirr is not None
        else None
    )

    risk_score = compute_observed_risk_score(
        stated_risk_score=stated_risk_score,
        max_position_weight=max_position_weight,
        top_5_weight=top_5_weight,
        effective_holdings=effective_holdings,
        meaningful_holdings_count=meaningful_holdings_count,
        annualized_turnover=annualized_turnover,
        capital_weighted_holding_days=capital_weighted_holding_days,
        equity_exposure=equity_exposure,
        relative_drawdown_to_benchmark=relative_drawdown_to_benchmark,
        relative_downside_capture_to_benchmark=relative_downside_capture_to_benchmark,
        relative_volatility_to_benchmark=relative_volatility_to_benchmark,
        relative_market_sensitivity_to_benchmark=relative_market_sensitivity_to_benchmark,
        years_of_history=years,
        closed_lot_count=len(lot_data["closed_lots"]),
    )

    realized_map = lot_data["realized_pnl_by_ticker"].copy()
    unrealized_map = {
        row["ticker"]: float(row["unrealized_pnl"])
        for _, row in open_positions_df.iterrows()
        if pd.notna(row["unrealized_pnl"])
    }
    combined_map = defaultdict(float)
    for ticker, pnl in realized_map.items():
        combined_map[ticker] += float(pnl)
    for ticker, pnl in unrealized_map.items():
        combined_map[ticker] += float(pnl)

    total_combined_pnl = sum(combined_map.values())
    attribution_rows = []
    for ticker, pnl in sorted(combined_map.items(), key=lambda item: item[1], reverse=True):
        attribution_rows.append(
            {
                "ticker": ticker,
                "realized_pnl": round(float(realized_map.get(ticker, 0.0)), 2),
                "unrealized_pnl": round(float(unrealized_map.get(ticker, 0.0)), 2),
                "combined_pnl": round(float(pnl), 2),
                "pnl_contribution_pct": round((float(pnl) / total_combined_pnl) * 100, 2)
                if total_combined_pnl
                else None,
            }
        )

    closed_lots_df = pd.DataFrame(lot_data["closed_lots"]).copy()
    sold_too_early = pd.DataFrame(columns=["ticker", "missed_upside", "realized_pnl", "if_held_pnl"])
    selection_alpha = pd.DataFrame(columns=["ticker", "alpha_pnl"])
    if not closed_lots_df.empty:
        closed_lots_df["current_price"] = closed_lots_df["ticker"].map(latest_prices)
        closed_lots_df["if_held_pnl"] = closed_lots_df["quantity"] * (
            closed_lots_df["current_price"] - closed_lots_df["unit_cost"]
        )
        closed_lots_df["missed_upside"] = closed_lots_df["if_held_pnl"] - closed_lots_df["realized_pnl"]
        closed_lots_df["realized_return_pct"] = closed_lots_df["realized_pnl"] / closed_lots_df["cost_basis"]
        closed_lots_df["benchmark_return_during_hold"] = closed_lots_df.apply(
            lambda row: price_return_between_dates(benchmark_close, row["buy_date"], row["sell_date"]),
            axis=1,
        )
        closed_lots_df["excess_return_vs_benchmark"] = (
            closed_lots_df["realized_return_pct"] - closed_lots_df["benchmark_return_during_hold"]
        )
        closed_lots_df["alpha_pnl"] = closed_lots_df["cost_basis"] * closed_lots_df["excess_return_vs_benchmark"]
        sold_too_early = (
            closed_lots_df.groupby("ticker", dropna=False)[["missed_upside", "realized_pnl", "if_held_pnl"]]
            .sum()
            .sort_values("missed_upside", ascending=False)
            .reset_index()
        )
        selection_alpha = (
            closed_lots_df.groupby("ticker", dropna=False)["alpha_pnl"].sum().reset_index()
        )

    open_alpha = open_positions_df[["ticker", "cost_basis_total", "excess_return_vs_benchmark"]].copy()
    if not open_alpha.empty:
        open_alpha["alpha_pnl"] = open_alpha["cost_basis_total"] * open_alpha["excess_return_vs_benchmark"]
        selection_alpha = pd.concat(
            [selection_alpha, open_alpha[["ticker", "alpha_pnl"]]], ignore_index=True
        )
    if not selection_alpha.empty:
        selection_alpha = (
            selection_alpha.groupby("ticker", dropna=False)["alpha_pnl"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

    conservative_rate = benchmark_xirr if benchmark_xirr is not None else benchmark_cagr or 0.05
    base_rate = benchmark_xirr if benchmark_xirr is not None else benchmark_cagr or 0.08
    optimistic_rate = max(
        invested_only_xirr if invested_only_xirr is not None else invested_only_cagr or 0.1,
        (benchmark_xirr if benchmark_xirr is not None else benchmark_cagr or 0.08) * 1.2,
    )
    projection_rates = {
        "conservative": max(0.03, conservative_rate * 0.75),
        "base": max(0.05, base_rate),
        "optimistic": max(0.07, optimistic_rate),
    }
    projection_rows = []
    for year in range(0, projection_years + 1):
        row = {"year": year}
        for name, rate in projection_rates.items():
            row[name] = round(total_account_value_estimate * ((1 + rate) ** year), 2)
        projection_rows.append(row)
    projection_df = pd.DataFrame(projection_rows)

    # Attribute 2025 volatility to the holdings that actually drove the portfolio's
    # day-to-day variance so users can connect the chart behavior back to real names.
    volatility_driver_rows: list[dict[str, Any]] = []
    asset_returns = price_matrix.pct_change(fill_method=None).replace([math.inf, -math.inf], float("nan"))
    value_matrix = (share_matrix * price_matrix).astype(float)
    total_value_series = value_matrix.sum(axis=1).replace(0.0, pd.NA)
    lagged_weights = value_matrix.shift(1).div(total_value_series.shift(1), axis=0)
    lagged_weights = lagged_weights.apply(pd.to_numeric, errors="coerce")
    lagged_weights = lagged_weights.where(np.isfinite(lagged_weights.to_numpy(dtype=float)), np.nan).fillna(0.0)
    contribution_matrix = (lagged_weights * asset_returns).fillna(0.0)
    contribution_period = contribution_matrix.loc["2025-01-01":"2025-10-31"].dropna(how="all")
    if not contribution_period.empty:
        portfolio_return_period = contribution_period.sum(axis=1).dropna()
        contribution_period = contribution_period.loc[portfolio_return_period.index]
        portfolio_variance = float(portfolio_return_period.var()) if len(portfolio_return_period) > 1 else 0.0
        if portfolio_variance > 0:
            for ticker in contribution_period.columns:
                ticker_contribution = contribution_period[ticker]
                covariance_to_portfolio = ticker_contribution.cov(portfolio_return_period)
                if pd.isna(covariance_to_portfolio):
                    continue
                ticker_returns = asset_returns.loc[contribution_period.index, ticker]
                avg_weight = float(lagged_weights.loc[contribution_period.index, ticker].mean())
                volatility_driver_rows.append(
                    {
                        "ticker": ticker,
                        "avg_weight_pct": round(avg_weight, 4),
                        "asset_volatility_pct": round(float(ticker_returns.std() * math.sqrt(252)), 4)
                        if len(ticker_returns.dropna()) > 1
                        else None,
                        "variance_contribution_pct": round(float(covariance_to_portfolio / portfolio_variance), 4),
                    }
                )
            volatility_driver_rows = sorted(
                [row for row in volatility_driver_rows if row["avg_weight_pct"] > 0],
                key=lambda row: row["variance_contribution_pct"],
                reverse=True,
            )[:12]

    headline_metrics = {
        "benchmark_symbol": benchmark_symbol,
        "benchmark_method": "Trade-matched S&P 500 comparison excluding uninvested cash, with money-weighted return metrics",
        "analysis_start": portfolio_summary["date_range"]["start"],
        "analysis_end": portfolio_summary["date_range"]["end"],
        "analysis_years": round(years, 2),
        "current_portfolio_value": round(current_portfolio_value, 2),
        "uninvested_cash_estimate": round(uninvested_cash_estimate, 2),
        "raw_cash_balance_estimate": round(raw_cash_balance_estimate, 2),
        "total_account_value_estimate": round(total_account_value_estimate, 2),
        "invested_net_cash_estimate": round(invested_net_cash_estimate, 2),
        "benchmark_invested_net_cash": round(benchmark_invested_net_cash, 2),
        "benchmark_current_value": round(benchmark_current_value, 2),
        "total_realized_pnl": round(total_realized_pnl, 2),
        "total_unrealized_pnl": round(total_unrealized_pnl, 2),
        "total_profit_estimate": round(total_profit_estimate, 2),
        "total_return_estimate_including_cash": round(total_return_estimate, 4) if total_return_estimate is not None else None,
        "invested_only_profit_estimate": round(invested_only_profit_estimate, 2),
        "invested_only_return_estimate": round(invested_only_return_estimate, 4) if invested_only_return_estimate is not None else None,
        "benchmark_return_estimate": round(benchmark_return_estimate, 4) if benchmark_return_estimate is not None else None,
        "excess_return_vs_benchmark": round(excess_return_vs_benchmark, 4) if excess_return_vs_benchmark is not None else None,
        "account_money_weighted_return": round(account_xirr, 4) if account_xirr is not None else None,
        "invested_only_money_weighted_return": round(invested_only_xirr, 4) if invested_only_xirr is not None else None,
        "benchmark_money_weighted_return": round(benchmark_xirr, 4) if benchmark_xirr is not None else None,
        "excess_money_weighted_return_vs_benchmark": round(excess_xirr_vs_benchmark, 4)
        if excess_xirr_vs_benchmark is not None
        else None,
        "portfolio_cagr_including_cash": round(portfolio_cagr, 4) if portfolio_cagr is not None else None,
        "invested_only_cagr": round(invested_only_cagr, 4) if invested_only_cagr is not None else None,
        "benchmark_cagr": round(benchmark_cagr, 4) if benchmark_cagr is not None else None,
        "annualized_volatility": round(annualized_volatility, 4) if annualized_volatility is not None else None,
        "benchmark_annualized_volatility": round(benchmark_annualized_volatility, 4)
        if benchmark_annualized_volatility is not None
        else None,
        "relative_volatility_to_benchmark": round(relative_volatility_to_benchmark, 4)
        if relative_volatility_to_benchmark is not None
        else None,
        "portfolio_volatility_floor_value": round(portfolio_volatility_floor, 2),
        "benchmark_volatility_floor_value": round(benchmark_volatility_floor, 2),
        "relative_market_sensitivity_to_benchmark": round(relative_market_sensitivity_to_benchmark, 4)
        if relative_market_sensitivity_to_benchmark is not None
        else None,
        "relative_market_sensitivity_window_days": market_sensitivity_profile["window_days"],
        "relative_market_sensitivity_recency_weight_half_life_days": market_sensitivity_profile[
            "recency_weight_half_life_days"
        ],
        "tracking_error": round(tracking_error, 4) if tracking_error is not None else None,
        "information_ratio_proxy": round(info_ratio_proxy, 4) if info_ratio_proxy is not None else None,
        "concentration_hhi": round(hhi, 4),
        "max_position_weight": round(max_position_weight, 4),
        "effective_holdings": round(effective_holdings, 2) if effective_holdings is not None else None,
        "top_5_weight": round(top_5_weight, 4),
        "concentration_adjusted_return_proxy": round(concentration_adjusted_return_proxy, 4)
        if concentration_adjusted_return_proxy is not None
        else None,
        "drawdown_window_days": drawdown_profile["window_days"],
        "drawdown_recency_weight_half_life_days": drawdown_profile["recency_weight_half_life_days"],
        "drawdown_memory_weight": round(float(drawdown_profile["memory_weight"]), 4),
        "weighted_rolling_relative_portfolio_drawdown_reference": drawdown_profile["weighted_rolling_drawdown"],
        "blended_relative_portfolio_drawdown_reference": drawdown_profile["blended_drawdown"],
        "weighted_rolling_relative_benchmark_drawdown_reference": benchmark_drawdown_profile["weighted_rolling_drawdown"],
        "blended_relative_benchmark_drawdown_reference": benchmark_drawdown_profile["blended_drawdown"],
        "relative_drawdown_to_benchmark": round(relative_drawdown_to_benchmark, 4)
        if relative_drawdown_to_benchmark is not None
        else None,
        "relative_downside_capture_to_benchmark": round(relative_downside_capture_to_benchmark, 4)
        if relative_downside_capture_to_benchmark is not None
        else None,
        "relative_downside_capture_window_days": downside_capture_profile["window_days"],
        "relative_downside_capture_recency_weight_half_life_days": downside_capture_profile[
            "recency_weight_half_life_days"
        ],
        "max_portfolio_drawdown": round(float(portfolio_drawdown.min()), 4) if not portfolio_drawdown.empty else None,
        "max_benchmark_drawdown": round(float(benchmark_drawdown.min()), 4) if not benchmark_drawdown.empty else None,
        "portfolio_drawdown_on_worst_benchmark_day": round(portfolio_drawdown_on_worst_benchmark_day, 4)
        if portfolio_drawdown_on_worst_benchmark_day is not None
        else None,
    }
    series = pd.DataFrame(
        {
            "date": portfolio_equity_series.index,
            "portfolio_invested_value": portfolio_equity_series.values,
            "account_value": account_value_series.reindex(portfolio_equity_series.index).values,
            "benchmark_value": benchmark_value_series.reindex(portfolio_equity_series.index).values,
            "trade_flow": trade_flow_series.reindex(portfolio_equity_series.index).values,
            "portfolio_performance_index": aligned_performance["portfolio"].reindex(portfolio_equity_series.index).values,
            "benchmark_performance_index": aligned_performance["benchmark"].reindex(portfolio_equity_series.index).values,
            "portfolio_drawdown": portfolio_drawdown.reindex(portfolio_equity_series.index).values,
            "benchmark_drawdown": benchmark_drawdown.reindex(portfolio_equity_series.index).values,
        }
    )
    return {
        "headline_metrics": headline_metrics,
        "risk_score": risk_score,
        "open_positions": open_positions_df.sort_values("current_value", ascending=False).round(4).to_dict(orient="records"),
        "holding_performance_context": holding_performance_context,
        "sector_allocation": sector_allocation[["sector", "current_value", "weight_pct", "excess_return_vs_benchmark"]]
        .round(4)
        .to_dict(orient="records"),
        "best_sector_vs_benchmark": (
            {
                "sector": str(best_sector_row["sector"]),
                "current_value": round(float(best_sector_row["current_value"]), 2),
                "weight_pct": round(float(best_sector_row["weight_pct"]), 4),
                "excess_return_vs_benchmark": round(float(best_sector_row["excess_return_vs_benchmark"]), 4),
            }
            if best_sector_row is not None
            else None
        ),
        "performance_attribution": attribution_rows[:15],
        "top_volatility_drivers_2025": volatility_driver_rows,
        "sold_too_early": sold_too_early.head(15).round(4).to_dict(orient="records") if not sold_too_early.empty else [],
        "selection_alpha": selection_alpha.head(15).round(4).to_dict(orient="records") if not selection_alpha.empty else [],
        "projection_scenarios_no_new_contributions": {
            "years": projection_years,
            "annual_return_assumptions": {k: round(v, 4) for k, v in projection_rates.items()},
            "future_values": projection_rows[-1],
            "table": projection_df.round(2).to_dict(orient="records"),
        },
        "timeseries": series.round(4).to_dict(orient="records"),
    }


def ollama_tags(base_url: str = OLLAMA_BASE_URL) -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url}/api/tags", method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_available(model_name: str, base_url: str = OLLAMA_BASE_URL) -> bool:
    try:
        payload = ollama_tags(base_url)
    except Exception:
        return False
    names = {item.get("name") for item in payload.get("models", [])}
    return model_name in names


def pull_model(model_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["ollama", "pull", model_name], check=True, text=True, capture_output=True)


def build_messages(analysis_payload: dict[str, Any], risk_profile: int) -> list[dict[str, str]]:
    system_prompt = (
        "You are an investment portfolio analysis assistant. "
        "Use only the supplied structured portfolio data, latest prices, benchmark metrics, and risk scoring outputs. "
        "Do not invent prices, news, or fundamentals. "
        "Do not claim certainty about future returns. "
        "Provide educational portfolio analysis rather than guaranteed financial advice. "
        "Return concise markdown with these sections: "
        "1. Investor Profile "
        "2. Portfolio Snapshot "
        "3. Actual Portfolio Risk Score "
        "4. Performance vs S&P 500 "
        "5. Which Holdings Drove Returns "
        "6. Were Winners Sold Too Early? "
        "7. Important Caveats"
    )
    user_prompt = (
        f"User input:\n- Risk profile: {risk_profile}/100\n\n"
        "Analysis payload:\n"
        f"{json.dumps(make_json_safe(analysis_payload), indent=2)}\n\n"
        "Analyze this investor and provide:\n"
        "- the actual observed portfolio risk score, confidence level, and what drove it\n"
        "- whether the observed risk is aligned with the stated risk profile\n"
        "- current portfolio value, uninvested cash, and unrealized P&L context\n"
        "- how returns over the investor's own time horizon compare with the S&P 500 benchmark\n"
        "- use the money-weighted return comparison when discussing benchmark outperformance\n"
        "- which holdings truly drove performance\n"
        "- whether the results look like stock selection skill or broad market tailwind\n"
        "- any signs that winners were sold too early\n"
        "- clear caveats about what can and cannot be concluded from this data"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(make_json_safe(payload), indent=2)}\n", encoding="utf-8")


def write_csv_file(path: Path, data: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame = frame.apply(lambda column: column.map(make_json_safe)) if not frame.empty else frame
    frame.to_csv(path, index=False)


def build_diagnosis_metric_frames(market_metrics: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    explanations = metric_explanations()
    risk = market_metrics["risk_score"]
    raw_metrics = risk.get("raw_metrics", {})
    component_scores = risk.get("component_scores", {})

    metric_rows: list[dict[str, Any]] = []
    for metric_key in metric_navigation_order():
        info = explanations.get(metric_key, {})
        metric_rows.append(
            {
                "metric_key": metric_key,
                "group": info.get("group"),
                "label": info.get("label"),
                "raw_value": raw_metrics.get(metric_key),
                "score": component_scores.get(metric_key),
                "score_readout": score_readout(component_scores.get(metric_key)),
                "meaning": info.get("meaning"),
                "bigger_picture": info.get("bigger_picture"),
            }
        )

    dimension_rows = [
        {
            "dimension": dimension,
            "score": score,
            "score_readout": score_readout(score),
        }
        for dimension, score in risk.get("dimension_scores", {}).items()
    ]

    return pd.DataFrame(metric_rows), pd.DataFrame(dimension_rows)


def save_diagnosis_artifacts(
    *,
    csv_path: Path,
    dataset_source: str,
    risk_profile: int,
    transactions: pd.DataFrame,
    portfolio_summary: dict[str, Any],
    market_metrics: dict[str, Any],
) -> dict[str, str]:
    generated_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_dir = DIAGNOSIS_DIR
    run_dir = DIAGNOSIS_RUNS_DIR / run_id
    metric_frame, dimension_frame = build_diagnosis_metric_frames(market_metrics)

    bundle_manifest = {
        "generated_at": generated_at,
        "run_id": run_id,
        "dataset_source": dataset_source,
        "input_csv_path": str(csv_path),
        "stated_risk_profile": risk_profile,
        "analysis_start": market_metrics["headline_metrics"]["analysis_start"],
        "analysis_end": market_metrics["headline_metrics"]["analysis_end"],
        "benchmark_symbol": market_metrics["headline_metrics"]["benchmark_symbol"],
        "files": [
            "transactions.csv",
            "open_positions.csv",
            "holding_performance_context.csv",
            "sector_allocation.csv",
            "timeseries.csv",
            "risk_metric_scores.csv",
            "risk_dimension_scores.csv",
            "top_volatility_drivers_2025.csv",
            "performance_attribution.csv",
            "portfolio_summary.json",
            "headline_metrics.json",
            "risk_score.json",
            "diagnosis_manifest.json",
        ],
    }

    file_writes: list[tuple[str, Any]] = [
        ("transactions.csv", transactions),
        ("open_positions.csv", market_metrics["open_positions"]),
        ("holding_performance_context.csv", market_metrics.get("holding_performance_context", [])),
        ("sector_allocation.csv", market_metrics["sector_allocation"]),
        ("timeseries.csv", market_metrics["timeseries"]),
        ("risk_metric_scores.csv", metric_frame),
        ("risk_dimension_scores.csv", dimension_frame),
        ("top_volatility_drivers_2025.csv", market_metrics.get("top_volatility_drivers_2025", [])),
        ("performance_attribution.csv", market_metrics.get("performance_attribution", [])),
        ("portfolio_summary.json", portfolio_summary),
        ("headline_metrics.json", market_metrics["headline_metrics"]),
        ("risk_score.json", market_metrics["risk_score"]),
        ("diagnosis_manifest.json", bundle_manifest),
    ]

    for target_dir in (latest_dir, run_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        for file_name, payload in file_writes:
            path = target_dir / file_name
            if file_name.endswith(".json"):
                write_json_file(path, payload)
            else:
                write_csv_file(path, payload)

    return {
        "latest_dir": str(latest_dir),
        "run_dir": str(run_dir),
    }


def call_ollama(
    model: str,
    messages: list[dict[str, str]],
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.2,
    num_predict: int = 1400,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        details = raw
        try:
            parsed = json.loads(raw) if raw else {}
            details = parsed.get("error") or parsed.get("message") or raw
        except Exception:
            pass
        raise RuntimeError(
            f"Ollama /api/chat failed with HTTP {exc.code}: {details}. "
            "If this mentions context length or memory, reduce the generated payload or num_predict."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to Ollama at {base_url}. Ensure `ollama serve` is running.") from exc
    message = body.get("message", {})
    content = message.get("content")
    if not content:
        raise RuntimeError(f"Unexpected Ollama response: {body}")
    return content.strip()


def pct_text(value: float | None) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value * 100:.2f}%"


def money_text(value: float | None) -> str:
    return "N/A" if value is None or pd.isna(value) else f"${value:,.2f}"


def number_text(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def percent_display(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{decimals}f}%"


def parse_display_number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("$", "").replace(",", "").replace("%", "")
    if not text or text == "N/A":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_display_dataframe(
    df: pd.DataFrame,
    *,
    currency_columns: list[str] | None = None,
    percent_columns: list[str] | None = None,
    number_columns: dict[str, int] | None = None,
) -> pd.DataFrame:
    # Keep all analytics numeric internally and only format values at the UI boundary.
    formatted = df.copy()
    for column in currency_columns or []:
        if column in formatted.columns:
            formatted[column] = formatted[column].apply(money_text)
    for column in percent_columns or []:
        if column in formatted.columns:
            formatted[column] = formatted[column].apply(percent_display)
    for column, decimals in (number_columns or {}).items():
        if column in formatted.columns:
            formatted[column] = formatted[column].apply(lambda value, d=decimals: number_text(value, d))
    return formatted


def metric_card(label: str, value: str, subtitle: str = "") -> str:
    subtitle_md = f"<div style='color:#cbd5e1;font-size:12px'>{subtitle}</div>" if subtitle else ""
    return (
        "<div class='metric-card' style='padding:18px 18px;border:1px solid rgba(148,163,184,.18);"
        "border-radius:18px;background:linear-gradient(180deg, rgba(30,41,59,.98), rgba(15,23,42,.94));"
        "min-height:120px;display:flex;flex-direction:column;justify-content:space-between;overflow:hidden;"
        "box-shadow:0 16px 40px rgba(2,6,23,.24)'>"
        f"<div class='metric-card-label' style='font-size:12px;color:#93c5fd;text-transform:uppercase;letter-spacing:.08em;line-height:1.25'>{label}</div>"
        f"<div class='metric-card-value' style='font-size:28px;font-weight:700;margin-top:8px;line-height:1.15;word-break:break-word;color:#f8fafc'>{value}</div>"
        f"{subtitle_md}</div>"
    )


def dataframe_from_records(records: list[dict[str, Any]], columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if columns:
        for column in columns:
            if column not in df.columns:
                df[column] = None
        df = df[columns]
    return df


def score_readout(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "Not available"
    value = float(score)
    if value < 20:
        return "Low concern"
    if value < 40:
        return "Mild concern"
    if value < 60:
        return "Moderate concern"
    if value < 80:
        return "High concern"
    return "Very high concern"


def metric_explanations() -> dict[str, dict[str, str]]:
    return {
        "concentration::single_position_weight": {
            "group": "Concentration",
            "anchor": "risk-guide-single-position-weight",
            "label": "Largest position size",
            "meaning": "Checks whether one holding is large enough to dominate the portfolio.",
            "bigger_picture": "A high score means one stock can strongly drive the whole account.",
            "question": "How dependent is the portfolio on one stock?",
            "why_it_matters": "Even a great company can create fragile portfolio risk if it grows too large. The bigger one position gets, the more your whole result depends on that one name.",
            "score_reading": "A low score means no single stock dominates. A high score means one stock can meaningfully move the whole portfolio.",
            "low_means": "No one position has enough weight to dominate outcomes on its own.",
            "high_means": "One holding is large enough that a bad move there can materially hurt the full account.",
        },
        "concentration::top_5_weight": {
            "group": "Concentration",
            "anchor": "risk-guide-top-five-weight",
            "label": "Top 5 holdings dominance",
            "meaning": "Checks whether just a few names are carrying most of the portfolio.",
            "bigger_picture": "A high score means your results depend heavily on a small cluster of stocks.",
            "question": "Are a few holdings driving most of the portfolio?",
            "why_it_matters": "A portfolio can look diversified by ticker count but still be heavily concentrated if most of the money sits in the top few names.",
            "score_reading": "A low score means capital is spread more broadly. A high score means a small group of names is carrying too much of the portfolio.",
            "low_means": "Your top five holdings do not control most of the portfolio.",
            "high_means": "A handful of stocks are doing most of the work, so diversification is thinner than it first appears.",
        },
        "concentration::effective_holdings": {
            "group": "Concentration",
            "anchor": "risk-guide-effective-holdings",
            "label": "True diversification",
            "meaning": "Looks past ticker count and asks how many holdings really matter after concentration is considered.",
            "bigger_picture": "A high score means the portfolio is less diversified than it appears at first glance.",
            "question": "How many holdings really matter once concentration is taken into account?",
            "why_it_matters": "Owning many tickers is not the same as being well diversified. If the money is unevenly spread, the portfolio can behave like it has far fewer real positions.",
            "score_reading": "A low score means the portfolio behaves like it has many meaningful positions. A high score means it behaves like only a smaller set truly matters.",
            "low_means": "Diversification is broad relative to the number of meaningful positions you actually hold.",
            "high_means": "The portfolio acts more concentrated than its ticker count suggests.",
        },
        "behavior::turnover": {
            "group": "Behavior",
            "anchor": "risk-guide-turnover",
            "label": "Trading churn",
            "meaning": "Measures how much of the portfolio you rotate through selling over time.",
            "bigger_picture": "A high score means your style is more active and decision-heavy than buy-and-hold.",
            "question": "How much of the portfolio are you trading instead of holding?",
            "why_it_matters": "Higher turnover usually means more decisions, more timing risk, and a more active style. It does not automatically mean bad investing, but it does mean behavior matters more.",
            "score_reading": "A low score means your trading style looks calmer and more buy-and-hold. A high score means you are rotating capital more actively.",
            "low_means": "You are mostly accumulating and holding positions rather than constantly cycling through them.",
            "high_means": "You are trading more actively, which adds behavioral risk even if the holdings themselves are reasonable.",
        },
        "behavior::short_holding_period": {
            "group": "Behavior",
            "anchor": "risk-guide-short-holding-period",
            "label": "How long your dollars stay invested",
            "meaning": "Measures whether larger investments are being held long enough to look like long-term investing.",
            "bigger_picture": "A high score means bigger dollars are being cycled out faster than a long-term profile would suggest.",
            "question": "How long do your invested dollars usually stay in positions?",
            "why_it_matters": "This metric gives more weight to larger investments than tiny trades, so it reflects your real capital behavior instead of being distorted by small speculative flips.",
            "score_reading": "A low score means your bigger dollars are staying invested for longer. A high score means meaningful capital is being turned over sooner.",
            "low_means": "Your larger investments are generally held long enough to look patient and long-term.",
            "high_means": "Your larger investments are being rotated out faster than an 18-month long-term benchmark would suggest.",
        },
        "market::relative_volatility_to_benchmark": {
            "group": "Market",
            "anchor": "risk-guide-relative-volatility",
            "label": "Volatility vs S&P 500",
            "meaning": "Checks whether your portfolio swings around more than the S&P 500 over comparable periods.",
            "bigger_picture": "A high score means your ride has been rougher than the market's ride.",
            "question": "Has the portfolio been swinging around more than the market?",
            "why_it_matters": "This keeps the score fair. If the whole market was volatile, the portfolio is only penalized for being more volatile than the S&P 500, not for simply living through a rough market.",
            "score_reading": "A low score means your volatility has been close to, or below, the S&P 500. A high score means the ride has been rougher than the market over comparable periods.",
            "low_means": "Your recent 6-month rolling volatility has looked broadly market-like or calmer.",
            "high_means": "Your portfolio has been moving around materially more than the S&P 500 over the same kinds of periods.",
        },
        "market::relative_drawdown_to_benchmark": {
            "group": "Market",
            "anchor": "risk-guide-relative-drawdown",
            "label": "Downside depth vs S&P 500",
            "meaning": "Checks whether your portfolio's bad stretches have been deeper than the S&P 500's.",
            "bigger_picture": "A high score means your losses in rough periods have been worse than a simple market portfolio.",
            "question": "When the portfolio goes through a bad stretch, is the drop deeper than the S&P 500's?",
            "why_it_matters": "Investors usually feel drawdowns more than volatility. This metric compares your downside depth to the market so the score reflects pain beyond what a simple S&P 500 investor experienced.",
            "score_reading": "A low score means recent rolling drawdowns have been similar to, or better than, the benchmark. A high score means your downside has been materially worse than the market's.",
            "low_means": "Your drawdown depth has been in line with or better than the S&P 500.",
            "high_means": "Your bad periods have been deeper than what a simple market portfolio would have gone through.",
        },
        "market::relative_downside_capture_to_benchmark": {
            "group": "Market",
            "anchor": "risk-guide-relative-downside-capture",
            "label": "Bad-day behavior vs S&P 500",
            "meaning": "Checks how your portfolio tends to behave on market down days compared with the S&P 500.",
            "bigger_picture": "A high score means your portfolio tends to lose more than the market when the market is already under stress.",
            "question": "On bad market days, does the portfolio hold up better than the S&P 500, about the same, or worse?",
            "why_it_matters": "This focuses on the days investors care about most: the down days. It tells you whether your portfolio tends to amplify market pain when the market is already red.",
            "score_reading": "A low score means your portfolio has usually held up at least as well as the S&P 500 on bad days. A high score means it tends to lose more than the benchmark during stress.",
            "low_means": "Your portfolio has been relatively defensive, or at least not worse than the S&P 500, on down-market days.",
            "high_means": "Your portfolio tends to lose more than the S&P 500 when the market is already falling.",
        },
        "market::relative_market_sensitivity_to_benchmark": {
            "group": "Market",
            "anchor": "risk-guide-relative-market-sensitivity",
            "label": "Market sensitivity vs S&P 500",
            "meaning": "Checks how strongly your portfolio tends to move when the S&P 500 moves.",
            "bigger_picture": "A high score means your portfolio has been amplifying broad market moves rather than moving in line with them.",
            "question": "When the S&P 500 moves, does your portfolio usually move about the same, less, or more?",
            "why_it_matters": "Some portfolios do not just move with the market, they amplify it. This metric captures how strongly your portfolio reacts to broad market moves over recent 6-month windows.",
            "score_reading": "A low score means your market sensitivity is close to, or below, the S&P 500. A high score means the portfolio has been amplifying market moves.",
            "low_means": "Your portfolio has behaved roughly like the market or a bit more defensively.",
            "high_means": "Your portfolio has tended to move more than the S&P 500 when the market moves.",
        },
        "market::equity_exposure": {
            "group": "Market",
            "anchor": "risk-guide-equity-exposure",
            "label": "How fully invested you are",
            "meaning": "Checks how much of your account is currently in the market rather than sitting in cash.",
            "bigger_picture": "A high score means more of your account is directly exposed to market gains and losses right now.",
            "question": "How much of the account is actually exposed to market risk right now?",
            "why_it_matters": "Two investors can own the same stocks but have very different immediate market risk if one is fully invested and the other is holding a meaningful cash buffer.",
            "score_reading": "A low score means more of the account is sitting in cash. A high score means most of the account is riding market gains and losses right now.",
            "low_means": "You have more cash cushion and less immediate market exposure.",
            "high_means": "Most of the account is deployed into the market, so more of your capital participates in market swings.",
        },
    }


def metric_group_order() -> list[tuple[str, list[str], str]]:
    return [
        (
            "Concentration",
            [
                "concentration::single_position_weight",
                "concentration::top_5_weight",
                "concentration::effective_holdings",
            ],
            "These metrics ask whether a small number of holdings can drive too much of the portfolio.",
        ),
        (
            "Behavior",
            [
                "behavior::turnover",
                "behavior::short_holding_period",
            ],
            "These metrics ask whether your real investing behavior looks patient and steady or more reactive and active.",
        ),
        (
            "Market",
            [
                "market::relative_volatility_to_benchmark",
                "market::relative_drawdown_to_benchmark",
                "market::relative_downside_capture_to_benchmark",
                "market::relative_market_sensitivity_to_benchmark",
                "market::equity_exposure",
            ],
            "These metrics ask how your portfolio has behaved relative to the S&P 500 and how much of your money is exposed to that behavior.",
        ),
    ]


def build_risk_guide_sections_html(
    risk: dict[str, Any],
    *,
    anchor_builder: Callable[[str], str],
    include_group_toc: bool,
) -> str:
    explanations = metric_explanations()
    component_scores = risk["component_scores"]

    def section_card(metric_key: str) -> str:
        info = explanations[metric_key]
        score = float(component_scores.get(metric_key, 0.0))
        anchor = anchor_builder(metric_key)
        return (
            f"<section id='{anchor}' style='padding:18px;border:1px solid rgba(148,163,184,.18);"
            "border-radius:18px;background:rgba(15,23,42,.34);scroll-margin-top:16px'>"
            f"<div style='font-size:12px;color:#93c5fd;text-transform:uppercase;letter-spacing:.08em'>{info['group']}</div>"
            f"<div style='font-size:22px;font-weight:700;color:#f8fafc;margin-top:6px'>{info['label']}</div>"
            f"<div style='font-size:13px;color:#93c5fd;margin-top:6px'>Current score: {score:.1f}/100 · {score_readout(score)}</div>"
            f"<div style='font-size:15px;color:#e2e8f0;margin-top:14px'><strong>What this metric is asking:</strong> {info['question']}</div>"
            f"<div style='font-size:14px;color:#cbd5e1;margin-top:12px'><strong>Plain-English meaning:</strong> {info['meaning']}</div>"
            f"<div style='font-size:14px;color:#cbd5e1;margin-top:10px'><strong>Why it matters:</strong> {info['why_it_matters']}</div>"
            f"<div style='font-size:14px;color:#cbd5e1;margin-top:10px'><strong>How to read the score:</strong> {info['score_reading']}</div>"
            f"<div style='font-size:14px;color:#cbd5e1;margin-top:10px'><strong>Low score usually means:</strong> {info['low_means']}</div>"
            f"<div style='font-size:14px;color:#cbd5e1;margin-top:10px'><strong>High score usually means:</strong> {info['high_means']}</div>"
            f"<div style='font-size:14px;color:#e2e8f0;margin-top:12px'><strong>Bigger picture:</strong> {info['bigger_picture']}</div>"
            "</section>"
        )

    group_sections = []
    for group_name, metric_keys, subtitle in metric_group_order():
        toc_links = ""
        if include_group_toc:
            toc_links = "".join(
                f"<a href='#{anchor_builder(key)}' "
                "style='display:inline-block;margin:0 8px 8px 0;padding:6px 10px;border-radius:999px;"
                "background:rgba(59,130,246,.12);color:#bfdbfe;text-decoration:none;font-size:12px'>"
                f"{explanations[key]['label']}</a>"
                for key in metric_keys
            )
        section_cards = "".join(section_card(key) for key in metric_keys)
        group_sections.append(
            "<div style='display:grid;gap:14px'>"
            f"<div style='padding:16px;border:1px solid rgba(148,163,184,.16);border-radius:16px;background:rgba(15,23,42,.28)'>"
            f"<div style='font-size:12px;color:#93c5fd;text-transform:uppercase;letter-spacing:.08em'>{group_name} Risk</div>"
            f"<div style='font-size:14px;color:#cbd5e1;margin-top:8px'>{subtitle}</div>"
            f"<div style='margin-top:12px'>{toc_links}</div>"
            "</div>"
            f"{section_cards}"
            "</div>"
        )
    return "".join(group_sections)


def build_risk_guide_html(risk: dict[str, Any], focused_metric: str | None = None) -> str:
    dimension_scores = risk["dimension_scores"]
    guide_sections = build_risk_guide_sections_html(
        risk,
        anchor_builder=lambda metric_key: metric_explanations()[metric_key]["anchor"],
        include_group_toc=True,
    )
    focused_section = ""
    if focused_metric in metric_explanations():
        info = metric_explanations()[focused_metric]
        score = float(risk["component_scores"].get(focused_metric, 0.0))
        focused_section = (
            "<div style='padding:18px;border:1px solid rgba(96,165,250,.35);border-radius:18px;"
            "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.96))'>"
            "<div style='font-size:12px;color:#60a5fa;text-transform:uppercase;letter-spacing:.08em'>Selected Metric</div>"
            f"<div style='font-size:24px;font-weight:700;color:#f8fafc;margin-top:8px'>{info['label']}</div>"
            f"<div style='font-size:13px;color:#93c5fd;margin-top:6px'>Current score: {score:.1f}/100 · {score_readout(score)}</div>"
            f"<div style='font-size:14px;color:#cbd5e1;margin-top:10px'>{info['meaning']}</div>"
            f"<div style='font-size:14px;color:#e2e8f0;margin-top:10px'><strong>Bigger picture:</strong> {info['bigger_picture']}</div>"
            "</div>"
        )

    return (
        "<div id='risk-guide-top' style='display:grid;gap:16px'>"
        "<div style='padding:18px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
        "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.96))'>"
        "<div style='font-size:28px;font-weight:700;color:#f8fafc'>Risk Guide</div>"
        "<div style='font-size:15px;color:#cbd5e1;margin-top:10px'>This guide explains what each risk metric is trying to measure in plain English. The goal is not just to show scores, but to help the user understand what those scores are saying about the portfolio.</div>"
        "<div style='font-size:14px;color:#cbd5e1;margin-top:12px'><strong>How the overall score works:</strong> concentration risk is 40% of the final score, market risk is 40%, and behavior risk is 20%.</div>"
        "</div>"
    ) + focused_section + guide_sections + "</div>"


def build_risk_explainer_html(risk: dict[str, Any]) -> str:
    dimension_scores = risk["dimension_scores"]

    def render_dimension_card(title: str, score: float, subtitle: str) -> str:
        return (
            "<div style='padding:14px 16px;border:1px solid rgba(148,163,184,.16);"
            "border-radius:14px;background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.92));"
            "box-shadow:0 8px 24px rgba(2,6,23,.18)'>"
            f"<div style='font-size:12px;color:#93c5fd;text-transform:uppercase;letter-spacing:.08em'>{title}</div>"
            f"<div style='font-size:28px;font-weight:700;color:#f8fafc;margin-top:8px'>{score:.1f}</div>"
            f"<div style='font-size:12px;color:#cbd5e1;margin-top:6px'>{subtitle}</div>"
            "</div>"
        )

    observed_vs_stated = (
        "about in line with your stated risk"
        if abs(risk["difference_vs_stated"]) < 8
        else "higher than your stated risk"
        if risk["difference_vs_stated"] > 0
        else "lower than your stated risk"
    )
    observed_subtitle = f"{risk['band']} · {observed_vs_stated}"
    market_subtitle = f"Relative to S&P 500 · Confidence: {risk['confidence_band']}"

    return (
        "<div style='display:grid;gap:16px'>"
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px'>"
        f"{render_dimension_card('Observed Risk', risk['score'], observed_subtitle)}"
        f"{render_dimension_card('Concentration', dimension_scores['concentration_risk'], 'How much a few holdings can dominate outcomes')}"
        f"{render_dimension_card('Behavior', dimension_scores['behavioral_risk'], 'How patient or active your investing style looks')}"
        f"{render_dimension_card('Market', dimension_scores['market_risk'], market_subtitle)}"
        "</div>"
        "<div style='font-size:12px;color:#60a5fa'>Each metric card below summarizes what the score is trying to say. Detailed explanations live in the Risk Guide section below.</div>"
        "</div>"
    )


def metric_navigation_order() -> list[str]:
    ordered: list[str] = []
    for _group_name, metric_keys, _subtitle in metric_group_order():
        ordered.extend(metric_keys)
    return ordered


def build_metric_card_values(risk: dict[str, Any]) -> list[str]:
    explanations = metric_explanations()
    component_scores = risk["component_scores"]
    values: list[str] = []
    for metric_key in metric_navigation_order():
        info = explanations[metric_key]
        score = float(component_scores.get(metric_key, 0.0))
        values.append(
            "<div class='risk-metric-card' style='padding:12px 14px;border:1px solid rgba(148,163,184,.16);"
            "border-radius:12px;background:rgba(15,23,42,.38);min-height:132px'>"
            f"<div style='font-size:13px;color:#e2e8f0;font-weight:600'>{info['label']}</div>"
            f"<div style='font-size:12px;color:#94a3b8;margin-top:4px'>{info['bigger_picture']}</div>"
            f"<div style='font-size:12px;color:#93c5fd;margin-top:8px'>Score: {score:.1f}/100 · {score_readout(score)}</div>"
            "<div style='font-size:11px;color:#7dd3fc;margin-top:10px'>See matching explanation in Risk Guide</div>"
            "</div>"
        )
    return values


def metric_value_display(metric_key: str, raw_value: Any) -> str:
    value = parse_display_number(raw_value)
    if value is None:
        return str(raw_value) if raw_value not in (None, "") else "N/A"
    percent_metric_keys = {
        "concentration::single_position_weight",
        "concentration::top_5_weight",
        "behavior::turnover",
    }
    if metric_key in percent_metric_keys:
        return percent_display(value)
    if metric_key == "behavior::short_holding_period":
        return f"{value:.0f} days"
    if metric_key.startswith("market::relative_"):
        return f"{value:.2f}x"
    if metric_key == "market::equity_exposure":
        return percent_display(value)
    if metric_key == "concentration::effective_holdings":
        return number_text(value, 2)
    return number_text(value, 2)


def build_diagnosis_summary_html(diagnosis: PortfolioRiskDiagnosis) -> str:
    top_concern_text = ", ".join(concern.label for concern in diagnosis.top_concerns[:3]) or "No major concerns yet"
    return (
        "<div style='padding:18px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
        "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.96))'>"
        "<div style='font-size:22px;font-weight:700;color:#f8fafc'>Diagnosis Snapshot and Alignment</div>"
        "<div style='font-size:15px;color:#cbd5e1;margin-top:10px'>"
        f"{diagnosis.diagnostic_summary}"
        "</div>"
        "<div style='font-size:14px;color:#e2e8f0;margin-top:14px'>"
        f"After looking at the holding drivers and sector pressure above, the portfolio currently reads as "
        f"<strong>{diagnosis.alignment.lower()}</strong>. Observed risk is <strong>{diagnosis.observed_risk_score:.1f}/100</strong> "
        f"({diagnosis.observed_risk_band}) versus a stated risk of <strong>{diagnosis.stated_risk_score:.1f}/100</strong> "
        f"({diagnosis.stated_risk_band})."
        "</div>"
        "<div style='font-size:13px;color:#93c5fd;margin-top:12px'>"
        f"Most important concerns right now: {top_concern_text}. Confidence in this diagnosis is <strong>{diagnosis.confidence_band}</strong>."
        "</div>"
        "</div>"
    )


def plot_diagnosis_concerns(diagnosis: PortfolioRiskDiagnosis) -> go.Figure:
    concerns = diagnosis.top_concerns
    if not concerns:
        fig = go.Figure()
        fig.add_annotation(
            text="No ranked diagnosis concerns are available yet.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15, "color": "#e2e8f0"},
        )
        fig.update_layout(
            title="Top Risk Categories",
            height=320,
            margin={"l": 24, "r": 24, "t": 56, "b": 24},
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.55)",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    labels = [concern.label for concern in concerns][::-1]
    base_scores = [concern.base_severity_score for concern in concerns][::-1]
    adjustment_scores = [concern.external_adjustment_score for concern in concerns][::-1]
    total_scores = [concern.severity_score for concern in concerns][::-1]
    adjustment_text = [
        "<br>".join(concern.adjustment_reasons[:2]) if concern.adjustment_reasons else "No external adjustment"
        for concern in concerns
    ][::-1]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=base_scores,
            y=labels,
            orientation="h",
            name="Base severity",
            marker={"color": "#3B82F6"},
            hovertemplate=(
                "%{y}<br>Base severity: %{x:.1f}/100"
                "<br>This comes from the measured portfolio risk layer."
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=adjustment_scores,
            y=labels,
            orientation="h",
            name="External reinforcement",
            marker={"color": "#22C55E"},
            customdata=adjustment_text,
            hovertemplate=(
                "%{y}<br>External adjustment: %{x:.1f}"
                "<br>%{customdata}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=total_scores,
            y=labels,
            mode="text",
            text=[f"{score:.1f}/100" for score in total_scores],
            textposition="middle right",
            textfont={"color": "#f8fafc", "size": 12},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="Top Risk Categories (Ranked)",
        height=420,
        margin={"l": 40, "r": 24, "t": 60, "b": 36},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.55)",
        barmode="stack",
        hoverlabel=dark_hoverlabel(),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    fig.update_xaxes(title_text="Severity score", gridcolor="rgba(148,163,184,0.18)", range=[0, 110])
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.08)")
    return fig


def plot_portfolio_gaps(diagnosis: PortfolioRiskDiagnosis) -> go.Figure:
    """Render portfolio gaps as a compact severity chart."""
    gaps = diagnosis.portfolio_gaps
    if not gaps:
        fig = go.Figure()
        fig.add_annotation(
            text="No portfolio gaps are available yet.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15, "color": "#e2e8f0"},
        )
        fig.update_layout(
            title="Portfolio Gaps",
            height=340,
            margin={"l": 24, "r": 24, "t": 56, "b": 24},
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.55)",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    labels = [gap.label for gap in gaps][::-1]
    scores = [gap.severity_score for gap in gaps][::-1]
    custom = [
        (
            gap.what_is_missing,
            " | ".join(gap.what_would_change[:2]),
            gap.suggested_vehicle_tilt,
        )
        for gap in gaps
    ][::-1]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker={"color": "#38bdf8", "line": {"color": "#0ea5e9", "width": 1}},
            text=[f"{score:.0f}/100" for score in scores],
            textposition="outside",
            customdata=custom,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Severity: %{x:.1f}/100<br>"
                "Missing: %{customdata[0]}<br>"
                "What a future add should do: %{customdata[1]}<br>"
                "Default tilt: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="What The Portfolio Still Needs Most",
        height=max(340, 90 + 66 * len(labels)),
        margin={"l": 40, "r": 24, "t": 56, "b": 36},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.55)",
        hoverlabel=dark_hoverlabel(),
        xaxis={
            "title": "Gap severity",
            "range": [0, 100],
            "gridcolor": "rgba(148,163,184,0.14)",
            "zeroline": False,
        },
        yaxis={
            "tickfont": {"size": 12, "color": "#e2e8f0"},
            "gridcolor": "rgba(148,163,184,0.08)",
        },
        font={"color": "#e2e8f0"},
    )
    return fig


def compact_text_snippet(text: Any, limit: int = 120) -> str:
    snippet = str(text or "").replace("\n", " ").replace("\r", " ").strip()
    snippet = " ".join(snippet.split())
    return snippet[:limit]


def html_entity_clean(text: Any) -> str:
    return (
        str(text or "")
        .replace("&#160;", " ")
        .replace("&nbsp;", " ")
        .replace("  ", " ")
        .strip()
    )


def summarize_filing_signal(title: str | None, snippet: Any) -> str:
    """Translate raw filing cues into plain-English findings for the dashboard.

    The SEC snippets we currently store are often section headers rather than
    prose. Instead of showing users `Item 1A. Risk Factors`, we convert those
    markers into an explanation of what that means for the diagnosis.
    """
    cleaned = compact_text_snippet(html_entity_clean(snippet), 220).lower()
    report_kind = "quarterly report" if str(title or "").strip() == "10-Q filing" else "annual report"

    has_risk_factors = "risk factors" in cleaned or "item 1a" in cleaned
    has_cybersecurity = "cybersecurity" in cleaned or "item 1c" in cleaned
    has_unresolved_comments = "unresolved staff comments" in cleaned or "item 1b" in cleaned

    findings: list[str] = []
    if has_risk_factors:
        findings.append(
            f"the latest {report_kind} still points investors to material business risks"
        )
    if has_cybersecurity:
        findings.append(
            f"the latest {report_kind} also explicitly calls out cybersecurity disclosures"
        )
    if has_unresolved_comments:
        findings.append(
            f"the latest {report_kind} still includes a section on unresolved regulator questions or comments"
        )

    if findings:
        sentence = "; ".join(findings)
        return sentence[0].upper() + sentence[1:] + "."

    if cleaned:
        readable = compact_text_snippet(html_entity_clean(snippet), 160)
        return f"The latest {report_kind} contained sections highlighted as: {readable}."

    return f"The latest {report_kind} was reviewed as part of the diagnosis."


def humanize_reason_label(label: str | None) -> str:
    mapping = {
        "Concentration pressure": "Big position size",
        "Meaningful position size": "Still a meaningful position",
        "Volatility contributor": "Large contributor to portfolio swings",
        "Noticeable volatility contributor": "Adds to portfolio swings",
        "Lagging benchmark": "Trailing the market",
        "High beta": "Moves more than the market",
        "Negative earnings": "Profits are under pressure",
        "Balance-sheet stretch": "Balance sheet looks stretched",
        "Narrative or event risk": "Recent company developments to watch",
        "Restrictive-rate sensitivity": "Can be more sensitive when rates stay high",
        "Sector crowding": "A big share of the portfolio is in this sector",
        "Sector benchmark lag": "This sector has trailed the market",
    }
    return mapping.get(str(label or "").strip(), str(label or ""))


def humanize_concern_label(label: str | None) -> str:
    mapping = {
        "Concentration risk": "This position is large enough to noticeably move the whole portfolio",
        "Market-relative risk": "This holding has been moving more sharply than a steadier market-like portfolio",
        "Behavioral risk": "Trading behavior around this holding may be adding extra risk",
        "Sector crowding": "A large share of the portfolio is stacked into the same sector",
        "Company-specific risk": "There are company-specific issues that deserve closer attention",
        "Macro sensitivity": "This holding can be more exposed when rates stay high or the economy tightens",
    }
    return mapping.get(str(label or "").strip(), str(label or ""))


def humanize_macro_flag(flag: str) -> str:
    mapping = {
        "rates still restrictive": "interest rates are still relatively high",
        "inflation still sticky": "inflation is still proving hard to bring down",
        "labor market still stable": "the job market still looks fairly stable",
        "yield curve positively sloped": "long-term yields are back above short-term yields",
        "yield curve still inverted": "short-term yields are still above long-term yields",
    }
    return mapping.get(str(flag or "").strip(), str(flag or ""))


def render_bold_markers(text: Any) -> str:
    """Convert **double-asterisk** emphasis into HTML bold text."""
    raw = str(text or "")
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", raw)


def short_concern_badge(label: str | None) -> str:
    """Shorter concern label for compact dashboard chips."""
    mapping = {
        "Concentration risk": "Large-position risk",
        "Market-relative risk": "Sharper-than-market moves",
        "Behavioral risk": "Trading-pattern risk",
        "Sector crowding": "Sector crowding",
        "Company-specific risk": "Company-specific issues",
        "Macro sensitivity": "Rate / macro sensitivity",
    }
    return mapping.get(str(label or "").strip(), str(label or "Portfolio review"))


def build_action_summary_for_users(action_need: Any, contribution: Any) -> str:
    """Rewrite the action-layer summary into plain user language for the dashboard.

    The diagnosis objects keep precise internal labels because they are useful
    for debugging and notebook review. The dashboard needs a friendlier bridge
    from those labels to plain-English portfolio meaning.
    """
    action_label = getattr(action_need, "action_label", "Needs review")
    ticker = getattr(action_need, "ticker", "This holding")
    concern_label = getattr(action_need, "linked_primary_concern", None)
    concern_explanation = humanize_concern_label(concern_label)
    score = getattr(contribution, "overall_contribution_score", None)
    pressure = getattr(action_need, "action_pressure_score", None)

    if action_label == "Reduce exposure":
        prefix = f"{ticker} looks large or influential enough that cutting it back would directly lower portfolio risk."
    elif action_label == "Trim and monitor":
        prefix = f"{ticker} is strong enough as a risk driver that trimming it is worth considering, even if this is not an urgent sell call."
    elif action_label == "Monitor closely":
        prefix = f"{ticker} does not yet look like a clear trim, but it is important enough to keep on the watchlist."
    else:
        prefix = f"{ticker} shows up in the diagnosis, but it does not yet look like a holding that needs action."

    score_clause = ""
    if score is not None and pressure is not None:
        score_clause = (
            f" It contributes meaningfully to the current diagnosis ({float(score):.1f}/100 contribution score) "
            f"and is currently at {float(pressure):.1f}/100 action pressure."
        )
    return prefix + " The main reason is simple: " + concern_explanation.lower() + "." + score_clause


def humanize_action_reasoning_point(text: Any) -> str:
    """Make recommendation reasoning read naturally in the Risk Actions tab."""
    point = str(text or "").strip()
    if not point:
        return ""
    point = point.replace("trade-matched S&P 500", "S&P 500 over the same period")
    point = point.replace("diagnosis pressure", "extra portfolio risk pressure")
    point = point.replace("trimming easier to defend", "a trim more reasonable")
    point = point.replace("It has also trailed the S&P 500 in", "It also underperformed the S&P 500 in")
    point = point.replace("At the same time, it has still beaten the S&P 500 in", "At the same time, it still beat the S&P 500 in")
    return point


def humanize_evidence_point(evidence: Any, ticker: str | None = None) -> str:
    text = str(evidence or "").strip()
    ticker_label = ticker or "This holding"
    if not text:
        return ""
    if text == "10-Q filing":
        return ""
    if text == "10-K filing":
        return ""
    if text.startswith("Macro flag: "):
        return humanize_macro_flag(text.replace("Macro flag: ", "", 1)).capitalize() + "."
    if text.startswith("Beta: "):
        value = text.replace("Beta: ", "", 1)
        return f"{ticker_label} has been moving about {value}x as much as the market."
    if text.startswith("Current weight: "):
        value = text.replace("Current weight: ", "", 1)
        return f"{ticker_label} currently makes up {value} of the portfolio."
    if text.startswith("Variance contribution: "):
        value = text.replace("Variance contribution: ", "", 1)
        return f"{ticker_label} drove about {value} of recent portfolio swings."
    if text.startswith("Excess return vs benchmark: "):
        value = text.replace("Excess return vs benchmark: ", "", 1)
        return f"Since purchase, {ticker_label} has trailed the trade-matched S&P 500 by {value}."
    if text.startswith("Latest filed fundamentals: "):
        value = text.replace("Latest filed fundamentals: ", "", 1)
        return f"The latest filed financial snapshot used here is dated {value}."
    if text.startswith("Holding return since weighted-average buy date: "):
        value = text.replace("Holding return since weighted-average buy date: ", "", 1)
        return f"Since the average buy date, {ticker_label} returned {value}."
    if text.startswith("Benchmark return over the same period: "):
        value = text.replace("Benchmark return over the same period: ", "", 1)
        return f"Over that same period, the S&P 500 returned {value}."
    if text.startswith("Relative performance vs benchmark: "):
        value = text.replace("Relative performance vs benchmark: ", "", 1)
        return f"Relative to the S&P 500, {ticker_label} was {value}."
    if text.startswith("1Y relative vs S&P 500: "):
        value = text.replace("1Y relative vs S&P 500: ", "", 1)
        return f"Over the last year, {ticker_label} was {value} versus the S&P 500."
    if text.startswith("3Y relative vs S&P 500: "):
        value = text.replace("3Y relative vs S&P 500: ", "", 1)
        return f"Over the last 3 years, {ticker_label} was {value} versus the S&P 500."
    if text.startswith("5Y relative vs S&P 500: "):
        value = text.replace("5Y relative vs S&P 500: ", "", 1)
        return f"Over the last 5 years, {ticker_label} was {value} versus the S&P 500."
    return text


def build_narrative_driver_detail(narrative_items: list[Any]) -> str | None:
    filing_item = next((item for item in narrative_items if item.source_type == "sec_filing"), None)
    news_item = next((item for item in narrative_items if item.source_type == "news_article"), None)
    details: list[str] = []
    if filing_item:
        filing_title = filing_item.title or "latest company filing"
        filing_finding = summarize_filing_signal(filing_title, filing_item.snippet)
        details.append(filing_finding)
    if news_item:
        news_title = news_item.title or "recent company news"
        news_snippet = compact_text_snippet(news_item.snippet, 120)
        details.append(f"recent company news also included “{news_title}”")
        if news_snippet:
            details.append(f"what stood out in that news: {news_snippet}")
    if not details:
        return None
    return "; ".join(details)


def build_reason_code_lookup(driver: Any) -> dict[str, Any]:
    return {
        str(getattr(reason_code, "code", "")): reason_code
        for reason_code in getattr(driver, "reason_codes", []) or []
        if getattr(reason_code, "code", None)
    }


def build_driver_secondary_labels(driver: Any) -> list[str]:
    code_lookup = build_reason_code_lookup(driver)
    labels: list[str] = []
    for code in getattr(driver, "secondary_reason_codes", []) or []:
        reason_code = code_lookup.get(code)
        label = reason_code.label if reason_code else str(code).replace("_", " ").title()
        labels.append(humanize_reason_label(label))
    return labels


def build_contribution_spillover_labels(contribution: Any) -> list[str]:
    labels: list[str] = []
    for item in getattr(contribution, "concern_contributions", [])[1:3]:
        label = getattr(item, "concern_label", None)
        if label:
            labels.append(str(label))
    return labels


def build_driver_evidence_points(diagnosis: PortfolioRiskDiagnosis, driver: Any) -> list[str]:
    fundamentals_by_ticker = {item.ticker: item for item in diagnosis.holding_fundamentals}
    narrative_by_ticker: dict[str, list[Any]] = {}
    for item in diagnosis.narrative_evidence:
        narrative_by_ticker.setdefault(item.ticker, []).append(item)

    evidence_points: list[str] = []
    reason_codes = list(getattr(driver, "reason_codes", []) or [])
    for reason_code in reason_codes[:3]:
        for evidence in getattr(reason_code, "evidence", [])[:2]:
            humanized = humanize_evidence_point(evidence, driver.ticker)
            if humanized and humanized not in evidence_points:
                evidence_points.append(humanized)

    fundamentals = fundamentals_by_ticker.get(driver.ticker)
    if fundamentals:
        beta_evidence = humanize_evidence_point(f"Beta: {fundamentals.beta:.2f}", driver.ticker) if fundamentals.beta is not None else None
        if beta_evidence and beta_evidence not in evidence_points:
            evidence_points.append(beta_evidence)
        if fundamentals.latest_filed_date:
            evidence_points.append(humanize_evidence_point(f"Latest filed fundamentals: {fundamentals.latest_filed_date}", driver.ticker))

    narrative_items = narrative_by_ticker.get(driver.ticker, [])
    narrative_detail = build_narrative_driver_detail(narrative_items)
    if narrative_detail and narrative_detail not in evidence_points:
        evidence_points.append(narrative_detail)

    for evidence in getattr(driver, "evidence_summary", [])[:3]:
        humanized = humanize_evidence_point(evidence, driver.ticker)
        if humanized and humanized not in evidence_points:
            evidence_points.append(humanized)
    return evidence_points[:4]


def build_holding_driver_explanation(diagnosis: PortfolioRiskDiagnosis, driver: Any) -> str:
    fundamentals_by_ticker = {item.ticker: item for item in diagnosis.holding_fundamentals}
    narrative_by_ticker: dict[str, list[Any]] = {}
    for item in diagnosis.narrative_evidence:
        narrative_by_ticker.setdefault(item.ticker, []).append(item)

    explanation_parts: list[str] = []
    current_weight = driver.current_weight or 0.0
    variance_contribution_pct = driver.variance_contribution_pct or 0.0
    excess_return = driver.excess_return_vs_benchmark
    top_weight = max((item.current_weight or 0.0) for item in diagnosis.top_holding_drivers) if diagnosis.top_holding_drivers else 0.0

    if current_weight >= 0.15:
        explanation_parts.append(
            f"it is a large enough position to materially move the whole portfolio on its own ({percent_display(current_weight)} of capital)"
        )
    elif current_weight >= 0.05:
        explanation_parts.append(
            f"it is still a meaningful position size at {percent_display(current_weight)} of capital"
        )

    if variance_contribution_pct >= 0.08:
        explanation_parts.append(
            f"it contributed heavily to recent portfolio swings ({percent_display(variance_contribution_pct)} of tracked variance)"
        )
    elif variance_contribution_pct >= 0.03:
        explanation_parts.append(
            f"it was a noticeable contributor to recent volatility ({percent_display(variance_contribution_pct)} of tracked variance)"
        )

    if excess_return is not None and excess_return < 0:
        explanation_parts.append(
            f"it has lagged the trade-matched S&P 500 since purchase by {percent_display(excess_return)}"
        )

    fundamentals = fundamentals_by_ticker.get(driver.ticker)
    if fundamentals:
        if fundamentals.beta is not None and fundamentals.beta > 1.2:
            explanation_parts.append(f"it has been moving more than the market lately ({number_text(fundamentals.beta, 2)}x market sensitivity)")
        if "latest net income negative" in fundamentals.signals:
            explanation_parts.append("its latest financial snapshot shows profits under pressure")
        if "liabilities are a large share of assets" in fundamentals.signals:
            explanation_parts.append("its balance sheet looks more stretched than the steadier names in the portfolio")

    narrative_items = narrative_by_ticker.get(driver.ticker, [])
    if narrative_items:
        narrative_detail = build_narrative_driver_detail(narrative_items)
        if narrative_detail:
            explanation_parts.append(narrative_detail)

    if not explanation_parts:
        if current_weight >= top_weight * 0.75 and top_weight > 0:
            explanation_parts.append("it matters because it is one of the portfolio’s largest remaining positions")
        else:
            explanation_parts.append("it currently matters because it shows up in the portfolio’s most influential holding set")

    return "This holding is notable because " + "; ".join(explanation_parts) + "."


def build_sector_driver_explanation(driver: Any) -> str:
    explanation_parts: list[str] = []
    if (driver.weight_pct or 0.0) >= 0.25:
        explanation_parts.append(
            f"the sector is carrying a large share of capital at {percent_display(driver.weight_pct)}"
        )
    if driver.excess_return_vs_benchmark is not None and driver.excess_return_vs_benchmark < 0:
        explanation_parts.append(
            f"the sector has lagged the trade-matched S&P 500 by {percent_display(driver.excess_return_vs_benchmark)}"
        )
    if not explanation_parts:
        explanation_parts.append("the sector is one of the bigger exposures in the current portfolio")
    return "This sector matters because " + "; ".join(explanation_parts) + "."


def build_diagnosis_driver_html(diagnosis: PortfolioRiskDiagnosis) -> str:
    driver_lookup = {driver.ticker: driver for driver in diagnosis.top_holding_drivers}
    action_lookup = {item.ticker: item for item in getattr(diagnosis, "holding_action_needs", []) or []}
    holding_items = ""
    contribution_items = diagnosis.holding_risk_contributions or []
    for contribution in contribution_items:
        driver = driver_lookup.get(contribution.ticker)
        if driver is None:
            continue
        action_need = action_lookup.get(contribution.ticker)
        spillover_labels = build_contribution_spillover_labels(contribution)
        evidence_points = build_driver_evidence_points(diagnosis, driver)
        evidence_html = "".join(
            f"<li style='margin-bottom:6px'>{point}</li>"
            for point in evidence_points
        ) or "<li>No extra evidence attached yet.</li>"
        spillover_html = "".join(
            (
                "<span style='padding:6px 10px;border-radius:999px;"
                "background:rgba(59,130,246,.10);border:1px solid rgba(96,165,250,.18);"
                "color:#dbeafe;font-size:12px;font-weight:600'>"
                f"{label}"
                "</span>"
            )
            for label in spillover_labels
        ) or (
            "<span style='padding:6px 10px;border-radius:999px;background:rgba(148,163,184,.08);"
            "border:1px solid rgba(148,163,184,.14);color:#cbd5e1;font-size:12px'>No major spillover concerns</span>"
        )
        confidence_band = contribution.contribution_confidence_band
        confidence_tone = "#34d399" if confidence_band == "High" else "#fbbf24" if confidence_band == "Medium" else "#f87171"
        score_tone = "#60a5fa" if contribution.overall_contribution_score < 40 else "#fbbf24" if contribution.overall_contribution_score < 70 else "#f87171"
        action_label = getattr(action_need, "action_label", "No action read yet")
        action_urgency = getattr(action_need, "action_urgency", "Needs review")
        action_pressure_score = getattr(action_need, "action_pressure_score", None)
        action_summary = (
            build_action_summary_for_users(action_need, contribution)
            if action_need is not None
            else "The action layer has not produced a stable read for this holding yet."
        )
        action_reason = getattr(action_need, "primary_action_reason", None) or "The system has not identified a dominant action reason yet."
        action_support = ", ".join(
            humanize_concern_label(label)
            for label in getattr(action_need, "supporting_concerns", [])[:2]
        ) or "No major secondary pressure noted"
        if action_label == "Reduce exposure":
            action_tone = "#f87171"
            action_bg = "rgba(127,29,29,.22)"
        elif action_label == "Trim and monitor":
            action_tone = "#fb923c"
            action_bg = "rgba(124,45,18,.22)"
        elif action_label == "Monitor closely":
            action_tone = "#fbbf24"
            action_bg = "rgba(120,53,15,.18)"
        else:
            action_tone = "#34d399"
            action_bg = "rgba(6,78,59,.18)"
        action_pressure_html = (
            f"<div style='font-size:13px;color:#cbd5e1;margin-top:10px'>"
            f"Action pressure <strong style='color:{action_tone}'>{action_pressure_score:.1f}/100</strong> · {action_urgency}"
            "</div>"
            if action_pressure_score is not None
            else ""
        )
        holding_items += (
            "<div style='padding:18px 20px;border:1px solid rgba(148,163,184,.14);border-radius:18px;"
            "background:rgba(15,23,42,.34);margin-bottom:14px'>"
            "<div style='display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap'>"
            "<div>"
            f"<div style='font-size:20px;font-weight:800;color:#f8fafc'>{contribution.ticker}"
            + (f"<span style='font-size:15px;font-weight:500;color:#cbd5e1;margin-left:10px'>({driver.sector})</span>" if driver.sector else "")
            + "</div>"
            "<div style='display:flex;flex-wrap:wrap;gap:10px 18px;margin-top:10px;font-size:14px;color:#93c5fd'>"
            f"<span>Weight <strong style='color:#f8fafc'>{percent_display(driver.current_weight)}</strong></span>"
            f"<span>Vs benchmark <strong style='color:#f8fafc'>{percent_display(driver.excess_return_vs_benchmark)}</strong></span>"
            f"<span>Variance contribution <strong style='color:#f8fafc'>{percent_display(driver.variance_contribution_pct)}</strong></span>"
            "</div>"
            "</div>"
            "<div style='display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end'>"
            f"<div style='padding:8px 12px;border-radius:999px;background:rgba(15,23,42,.72);border:1px solid rgba(148,163,184,.16);color:{score_tone};font-size:12px;font-weight:700'>"
            f"Contribution {contribution.overall_contribution_score:.1f}/100"
            "</div>"
            f"<div style='padding:8px 12px;border-radius:999px;background:{action_bg};border:1px solid rgba(148,163,184,.16);color:{action_tone};font-size:12px;font-weight:700'>"
            f"{action_label}"
            "</div>"
            f"<div style='padding:8px 12px;border-radius:999px;background:rgba(15,23,42,.72);border:1px solid rgba(148,163,184,.16);color:{confidence_tone};font-size:12px;font-weight:700'>"
            f"{confidence_band} confidence"
            "</div>"
            "</div>"
            "</div>"
            "<div style='display:grid;grid-template-columns:minmax(260px,1.2fr) minmax(240px,.95fr) minmax(240px,1fr);gap:16px;margin-top:16px'>"
            "<div style='padding:14px 16px;border-radius:14px;background:rgba(30,41,59,.46);border:1px solid rgba(148,163,184,.10)'>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700'>Main portfolio risk this holding is feeding</div>"
            f"<div style='font-size:18px;font-weight:800;color:#f8fafc;margin-top:8px'>{contribution.primary_concern_label}</div>"
            f"<div style='font-size:13px;line-height:1.5;color:#cbd5e1;margin-top:8px'>{humanize_concern_label(contribution.primary_concern_label)}</div>"
            f"<div style='font-size:15px;line-height:1.6;color:#dbe4f0;margin-top:10px'>{contribution.primary_concern_summary}</div>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700;margin-top:16px'>Other risks this holding also spills into</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:10px'>{spillover_html}</div>"
            "</div>"
            "<div style='padding:14px 16px;border-radius:14px;background:rgba(30,41,59,.40);border:1px solid rgba(148,163,184,.10)'>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700'>What level of action this currently deserves</div>"
            f"<div style='font-size:18px;font-weight:800;color:{action_tone};margin-top:8px'>{action_label}</div>"
            f"{action_pressure_html}"
            f"<div style='font-size:15px;line-height:1.6;color:#dbe4f0;margin-top:10px'>{action_summary}</div>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700;margin-top:16px'>Why this action label was chosen</div>"
            f"<div style='font-size:14px;line-height:1.6;color:#e2e8f0;margin-top:8px'>{action_reason.capitalize()}.</div>"
            f"<div style='font-size:13px;line-height:1.5;color:#cbd5e1;margin-top:10px'>Secondary pressure still comes from: <strong style='color:#f8fafc'>{action_support}</strong>.</div>"
            "</div>"
            "<div style='padding:14px 16px;border-radius:14px;background:rgba(30,41,59,.36);border:1px solid rgba(148,163,184,.10)'>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700'>What changed in the portfolio that led to this</div>"
            "<ul style='margin:12px 0 0 18px;color:#e2e8f0;line-height:1.55'>"
            f"{evidence_html}"
            "</ul>"
            "</div>"
            "</div>"
            "</div>"
        )

    if not holding_items:
        holding_items = "<div style='color:#cbd5e1'>No holding-level drivers were identified yet.</div>"

    sector_items = "".join(
        (
            "<div style='padding:14px 16px;border:1px solid rgba(148,163,184,.14);border-radius:14px;"
            "background:rgba(15,23,42,.28);margin-bottom:10px'>"
            f"<div style='display:flex;justify-content:space-between;gap:10px;align-items:flex-start;flex-wrap:wrap'>"
            f"<div style='font-size:16px;font-weight:700;color:#f8fafc'>{driver.sector}</div>"
            f"<div style='font-size:12px;color:#93c5fd'>{driver.driver_confidence_band} confidence</div>"
            "</div>"
            f"<div style='font-size:14px;color:#93c5fd;margin-top:8px'>Weight <strong style='color:#f8fafc'>{percent_display(driver.weight_pct)}</strong> · "
            f"Vs benchmark <strong style='color:#f8fafc'>{percent_display(driver.excess_return_vs_benchmark)}</strong></div>"
            f"<div style='margin-top:10px;font-size:14px;line-height:1.5;color:#e2e8f0'><strong>{humanize_reason_label(driver.primary_reason_label or 'Main sector reason')}:</strong> {driver.primary_reason_summary or build_sector_driver_explanation(driver)}</div>"
            "</div>"
        )
        for driver in diagnosis.top_sector_drivers
    ) or "<div style='color:#cbd5e1'>No sector-level drivers were identified yet.</div>"
    return (
        "<div style='display:grid;gap:16px'>"
        "<div style='padding:20px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
        "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.94))'>"
        "<div style='font-size:20px;font-weight:800;color:#f8fafc'>Top Holding Drivers</div>"
        "<div style='font-size:14px;color:#93c5fd;margin-top:6px'>For each holding, start with the main portfolio risk it is feeding, then look at the current action read and the evidence that made the diagnosis flag it.</div>"
        f"<div style='margin-top:14px'>{holding_items}</div>"
        "</div>"
        "<div style='padding:18px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
        "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.94))'>"
        "<div style='font-size:18px;font-weight:700;color:#f8fafc'>Sector Drivers</div>"
        "<div style='font-size:13px;color:#93c5fd;margin-top:6px'>These are the sector-level patterns that make the portfolio feel crowded or fragile.</div>"
        f"<div style='margin-top:14px'>{sector_items}</div>"
        "</div>"
        "</div>"
    )


def build_supporting_metrics_frame(diagnosis: PortfolioRiskDiagnosis) -> pd.DataFrame:
    evidence_keys = {key for concern in diagnosis.top_concerns for key in concern.evidence_metric_keys}
    rows = []
    for metric in diagnosis.supporting_metrics:
        if metric.metric_key not in evidence_keys:
            continue
        rows.append(
            {
                "Group": metric.group,
                "Metric": metric.label,
                "Raw value": metric_value_display(metric.metric_key, metric.raw_value),
                "Score": f"{metric.score:.1f}/100" if metric.score is not None else "N/A",
                "Readout": metric.score_readout or "Not available",
                "Why it matters": metric.bigger_picture or metric.meaning or "",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["Group", "Metric", "Raw value", "Score", "Readout", "Why it matters"])
    return frame.sort_values(["Group", "Score"], ascending=[True, False]).reset_index(drop=True)


def build_holding_fundamentals_frame(diagnosis: PortfolioRiskDiagnosis) -> pd.DataFrame:
    rows = [
        {
            "Ticker": item.ticker,
            "Company": item.company_name,
            "Sector": item.sector,
            "Beta": number_text(item.beta, 2),
            "Market cap": money_text(item.market_cap),
            "Revenue": money_text(item.revenue),
            "Net income": money_text(item.net_income),
            "Cash": money_text(item.cash_and_equivalents),
            "Operating cash flow": money_text(item.operating_cash_flow),
            "Signals": "; ".join(item.signals),
            "Latest filed date": item.latest_filed_date or "N/A",
        }
        for item in diagnosis.holding_fundamentals
    ]
    if not rows:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Company",
                "Sector",
                "Beta",
                "Market cap",
                "Revenue",
                "Net income",
                "Cash",
                "Operating cash flow",
                "Signals",
                "Latest filed date",
            ]
        )
    return pd.DataFrame(rows)


def build_narrative_evidence_frame(diagnosis: PortfolioRiskDiagnosis) -> pd.DataFrame:
    rows = [
        {
            "Ticker": item.ticker,
            "Source": "SEC filing" if item.source_type == "sec_filing" else "News article",
            "Date": item.document_date or "N/A",
            "Title": item.title or "Untitled",
            "Snippet": item.snippet,
            "URL": item.url or "",
        }
        for item in diagnosis.narrative_evidence
    ]
    if not rows:
        return pd.DataFrame(columns=["Ticker", "Source", "Date", "Title", "Snippet", "URL"])
    return pd.DataFrame(rows)


def build_macro_context_html(diagnosis: PortfolioRiskDiagnosis) -> str:
    macro = diagnosis.macro_context
    if macro is None:
        return (
            "<div style='padding:18px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
            "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.94))'>"
            "<div style='font-size:18px;font-weight:700;color:#f8fafc'>Macro Context</div>"
            "<div style='font-size:14px;color:#cbd5e1;margin-top:10px'>Macro regime data is not available yet.</div>"
            "</div>"
        )
    regime_flags = "".join(
        f"<li style='margin-bottom:8px'>{humanize_macro_flag(flag).capitalize()}</li>"
        for flag in macro.regime_flags
    ) or "<li>No regime flags were raised.</li>"
    macro_cards = (
        metric_card("Fed Funds", f"{macro.fed_funds_rate:.2f}%" if macro.fed_funds_rate is not None else "N/A", "Policy rate")
        + metric_card("Inflation", percent_display(macro.inflation_yoy), "Year-over-year CPI")
        + metric_card("Unemployment", f"{macro.unemployment_rate:.1f}%" if macro.unemployment_rate is not None else "N/A", "Labor market")
        + metric_card("10Y-2Y Spread", f"{macro.yield_curve_spread:.2f}%" if macro.yield_curve_spread is not None else "N/A", "Yield curve slope")
    )
    return (
        "<div style='display:grid;gap:16px'>"
        "<div style='padding:18px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
        "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.94))'>"
        "<div style='font-size:18px;font-weight:700;color:#f8fafc'>Macro Regime Context</div>"
        f"<div style='font-size:14px;color:#cbd5e1;margin-top:10px'>{macro.summary}</div>"
        f"<div style='font-size:12px;color:#93c5fd;margin-top:8px'>As of {macro.as_of_date or 'N/A'}</div>"
        "<div style='margin-top:14px'>"
        f"<ul style='color:#e2e8f0;padding-left:18px'>{regime_flags}</ul>"
        "</div>"
        "</div>"
        f"<div class='metric-strip'>{macro_cards}</div>"
        "</div>"
    )


def build_data_coverage_frame(diagnosis: PortfolioRiskDiagnosis) -> pd.DataFrame:
    coverage = (
        diagnosis.data_coverage.model_dump()
        if hasattr(diagnosis.data_coverage, "model_dump")
        else diagnosis.data_coverage.dict()
    )
    rows = [
        {
            "Source": key.replace("_", " ").replace(" available", "").title(),
            "Available": "Yes" if value else "No",
        }
        for key, value in coverage.items()
    ]
    return pd.DataFrame(rows)


def build_confidence_completeness_html(diagnosis: PortfolioRiskDiagnosis) -> str:
    gaps = "".join(
        f"<li style='margin-bottom:8px'>{gap}</li>"
        for gap in diagnosis.evidence_gaps
    ) or "<li>No major evidence gaps were recorded.</li>"
    coverage = (
        diagnosis.data_coverage.model_dump()
        if hasattr(diagnosis.data_coverage, "model_dump")
        else diagnosis.data_coverage.dict()
    )
    coverage_count = sum(1 for value in coverage.values() if value)
    total_count = len(coverage)
    return (
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px'>"
        "<div style='padding:18px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
        "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.94))'>"
        "<div style='font-size:18px;font-weight:700;color:#f8fafc'>Confidence and Completeness</div>"
        f"<div style='font-size:15px;color:#cbd5e1;margin-top:10px'>Overall diagnosis confidence is <strong>{diagnosis.confidence_band}</strong>.</div>"
        f"<div style='font-size:14px;color:#93c5fd;margin-top:10px'>Coverage check: {coverage_count}/{total_count} evidence sources are currently available.</div>"
        "<div style='font-size:14px;color:#cbd5e1;margin-top:12px'>The system should stay explicit about uncertainty, especially when external evidence is thin or incomplete.</div>"
        "</div>"
        "<div style='padding:18px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
        "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.94))'>"
        "<div style='font-size:18px;font-weight:700;color:#f8fafc'>Evidence Gaps</div>"
        f"<ul style='margin-top:14px;color:#e2e8f0;padding-left:18px'>{gaps}</ul>"
        "</div>"
        "</div>"
    )


def performance_window_chip(label: str, relative_value: float | None) -> str:
    """Render one trailing-window comparison chip for the Risk Actions tab."""
    if relative_value is None:
        tone = "#cbd5e1"
        background = "rgba(51,65,85,.55)"
        text = "Not enough history"
    elif relative_value >= 0:
        tone = "#34d399"
        background = "rgba(6,78,59,.22)"
        text = f"Beat S&P 500 by {percent_display(relative_value)}"
    else:
        tone = "#f87171"
        background = "rgba(127,29,29,.22)"
        text = f"Lagged S&P 500 by {percent_display(abs(relative_value))}"
    return (
        "<div style='padding:10px 12px;border-radius:14px;"
        f"background:{background};border:1px solid rgba(148,163,184,.14)'>"
        f"<div style='font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700'>{label}</div>"
        f"<div style='font-size:13px;color:{tone};font-weight:700;margin-top:6px'>{text}</div>"
        "</div>"
    )


def build_risk_actions_frame(diagnosis: PortfolioRiskDiagnosis) -> pd.DataFrame:
    """Create a review table for the new Risk Actions tab."""
    rows = []
    for item in diagnosis.holding_action_recommendations:
        rows.append(
            {
                "Ticker": item.ticker,
                "Action": item.recommendation_label,
                "Actionable": "Yes" if item.is_actionable else "No",
                "Current weight": percent_display(item.current_weight),
                "Current value": money_text(item.current_value),
                "Sell / trim value": money_text(item.value_to_sell),
                "Shares to sell": number_text(item.shares_to_sell, 4),
                "Target weight": percent_display(item.target_weight_after_action),
                "Weight reduction": percent_display(item.projected_weight_reduction_pct_points),
                "Variance reduction": percent_display(item.projected_variance_reduction_pct_points),
                "Lag-exposure reduction": percent_display(item.projected_relative_drag_reduction_pct_points),
                "Since-buy vs S&P 500": percent_display(item.relative_performance_vs_benchmark),
                "1Y vs S&P 500": percent_display(item.relative_1y_return_pct),
                "3Y vs S&P 500": percent_display(item.relative_3y_return_pct),
                "5Y vs S&P 500": percent_display(item.relative_5y_return_pct),
                "Confidence": item.confidence_band,
                "Explicit sell modifiers": " | ".join(item.explicit_sell_modifiers),
                "Modifier score": item.modifier_score,
                "What changed": item.what_changed,
                "Why it matters": item.why_it_matters,
                "Why this amount": item.amount_rationale,
                "Why": item.recommendation_summary,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Action",
                "Actionable",
                "Current weight",
                "Current value",
                "Sell / trim value",
                "Shares to sell",
                "Target weight",
                "Weight reduction",
                "Variance reduction",
                "Lag-exposure reduction",
                "Since-buy vs S&P 500",
                "1Y vs S&P 500",
                "3Y vs S&P 500",
                "5Y vs S&P 500",
                "Confidence",
                "Explicit sell modifiers",
                "Modifier score",
                "What changed",
                "Why it matters",
                "Why this amount",
                "Why",
            ]
        )
    action_order = {"Sell all shares": 0, "Reduce by 50%": 1, "Trim 35%": 2, "Trim 25%": 3, "Trim 20%": 4, "Trim 15%": 5, "Trim 10%": 6, "Hold for now": 7}
    frame = pd.DataFrame(rows)
    frame["_sort"] = frame["Action"].map(action_order).fillna(99)
    return frame.sort_values(["_sort", "Ticker"]).drop(columns=["_sort"]).reset_index(drop=True)


def build_risk_actions_html(diagnosis: PortfolioRiskDiagnosis) -> str:
    """Render simple sell/trim guidance backed by relative performance windows."""
    actionable = [item for item in diagnosis.holding_action_recommendations if item.is_actionable]
    held = [item for item in diagnosis.holding_action_recommendations if not item.is_actionable][:4]
    action_impact = diagnosis.portfolio_action_impact

    if not actionable:
        return (
            "<div style='padding:20px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
            "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.94))'>"
            "<div style='font-size:20px;font-weight:800;color:#f8fafc'>Risk Actions</div>"
            "<div style='font-size:15px;color:#cbd5e1;margin-top:10px'>No holdings currently meet the stricter sell / trim rule. The current logic only recommends action when a holding has actually been underperforming the S&P 500 over a meaningful period.</div>"
            "</div>"
        )

    impact_section = ""
    if action_impact is not None:
        impact_bullets_html = "".join(
            f"<li style='margin-bottom:7px'>{render_bold_markers(note)}</li>"
            for note in action_impact.impact_bullets[:5]
        ) or "<li>No portfolio-level impact preview was available.</li>"
        impact_cards = (
            metric_card("Actionable names", str(action_impact.actionable_count), ", ".join(action_impact.actionable_tickers[:4]) or "None")
            + metric_card("Capital freed", money_text(action_impact.total_value_to_sell), "Estimated value sold or trimmed")
            + metric_card("Weight reduction", percent_display(action_impact.total_weight_reduction_pct_points), "Combined weight reduction")
            + metric_card("Variance relief", percent_display(action_impact.total_variance_reduction_pct_points), "Recent variance contribution removed")
            + metric_card("Lag relief", percent_display(action_impact.total_relative_drag_reduction_pct_points), "Weighted benchmark-lag exposure reduced")
            + metric_card(
                "Largest position",
                percent_display(action_impact.projected_largest_position_pct),
                f"From {percent_display(action_impact.current_largest_position_pct)}",
            )
        )
        impact_section = (
            "<div style='padding:18px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
            "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.94));margin-bottom:18px'>"
            "<div style='font-size:20px;font-weight:800;color:#f8fafc'>If You Follow The Current Action Set</div>"
            f"<div style='font-size:14px;line-height:1.6;color:#cbd5e1;margin-top:10px'>{render_bold_markers(action_impact.impact_summary)}</div>"
            f"<div class='metric-strip' style='margin-top:14px'>{impact_cards}</div>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700;margin-top:16px'>What likely improves at the portfolio level</div>"
            f"<ul style='margin:12px 0 0 18px;color:#e2e8f0;line-height:1.55'>{impact_bullets_html}</ul>"
            "</div>"
        )

    cards = ""
    for item in actionable:
        if item.recommendation_label == "Sell all shares":
            action_line = (
                f"Sell the full position: about <strong>{number_text(item.shares_to_sell, 4)}</strong> shares, "
                f"or roughly <strong>{money_text(item.value_to_sell)}</strong>."
            )
        else:
            action_line = (
                f"{item.recommendation_label} now: sell about <strong>{number_text(item.shares_to_sell, 4)}</strong> shares "
                f"(roughly <strong>{money_text(item.value_to_sell)}</strong>) and bring the weight from "
                f"<strong>{percent_display(item.current_weight)}</strong> down to about "
                f"<strong>{percent_display(item.target_weight_after_action)}</strong>."
            )

        reasoning = "".join(
            f"<li style='margin-bottom:7px'>{humanize_action_reasoning_point(point)}</li>"
            for point in item.reasoning_points[:4]
        ) or "<li>No extra reasoning was attached.</li>"

        evidence = "".join(
            f"<li style='margin-bottom:7px'>{humanize_evidence_point(point, item.ticker)}</li>"
            for point in item.supporting_evidence[:5]
        ) or "<li>No supporting evidence was attached.</li>"
        modifiers = "".join(
            f"<li style='margin-bottom:7px'>{render_bold_markers(note)}</li>"
            for note in item.explicit_sell_modifiers[:4]
        ) or "<li>No extra filing, news, or macro modifiers were strong enough to change the recommendation.</li>"
        impacts = "".join(
            f"<li style='margin-bottom:7px'>{render_bold_markers(note)}</li>"
            for note in item.portfolio_impact_bullets[:4]
        ) or "<li>No meaningful portfolio impact was estimated.</li>"
        guardrails = "".join(
            f"<li style='margin-bottom:7px'>{note}</li>"
            for note in item.guardrail_notes[:3]
        ) or "<li>No major guardrails were triggered.</li>"

        cards += (
            "<div style='padding:18px 20px;border:1px solid rgba(148,163,184,.14);border-radius:18px;"
            "background:rgba(15,23,42,.34);margin-bottom:14px'>"
            "<div style='display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap'>"
            "<div>"
            f"<div style='font-size:22px;font-weight:800;color:#f8fafc'>{item.ticker}"
            + (f"<span style='font-size:15px;font-weight:500;color:#cbd5e1;margin-left:10px'>({item.sector})</span>" if item.sector else "")
            + "</div>"
            f"<div style='font-size:14px;color:#93c5fd;margin-top:8px'>Confidence: <strong style='color:#f8fafc'>{item.confidence_band}</strong> · Since-buy window: <strong style='color:#f8fafc'>{percent_display(item.relative_performance_vs_benchmark)}</strong> versus the S&P 500</div>"
            "</div>"
            "<div style='display:flex;gap:8px;flex-wrap:wrap'>"
            f"<div style='padding:8px 12px;border-radius:999px;background:rgba(127,29,29,.22);border:1px solid rgba(248,113,113,.22);color:#fca5a5;font-size:12px;font-weight:700'>{item.recommendation_label}</div>"
            f"<div style='padding:8px 12px;border-radius:999px;background:rgba(15,23,42,.72);border:1px solid rgba(148,163,184,.16);color:#93c5fd;font-size:12px;font-weight:700'>{short_concern_badge(item.linked_primary_concern)}</div>"
            "</div>"
            "</div>"
            "<div style='font-size:16px;line-height:1.6;color:#e2e8f0;margin-top:14px'>"
            f"{action_line}"
            "</div>"
            f"<div style='font-size:14px;line-height:1.6;color:#cbd5e1;margin-top:10px'>{render_bold_markers(item.recommendation_summary)}</div>"
            "<div style='display:grid;grid-template-columns:minmax(280px,1.2fr) minmax(250px,1fr);gap:16px;margin-top:16px'>"
            "<div style='padding:14px 16px;border-radius:14px;background:rgba(30,41,59,.42);border:1px solid rgba(148,163,184,.10)'>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700'>What changed</div>"
            f"<div style='font-size:14px;line-height:1.6;color:#e2e8f0;margin-top:10px'>{render_bold_markers(item.what_changed)}</div>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700;margin-top:16px'>Why that matters</div>"
            f"<div style='font-size:14px;line-height:1.6;color:#e2e8f0;margin-top:10px'>{render_bold_markers(item.why_it_matters)}</div>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700;margin-top:16px'>Why this amount</div>"
            f"<div style='font-size:14px;line-height:1.6;color:#e2e8f0;margin-top:10px'>{render_bold_markers(item.amount_rationale)}</div>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700;margin-top:16px'>Extra reasoning</div>"
            f"<ul style='margin:12px 0 0 18px;color:#e2e8f0;line-height:1.55'>{reasoning}</ul>"
            "</div>"
            "<div style='padding:14px 16px;border-radius:14px;background:rgba(30,41,59,.36);border:1px solid rgba(148,163,184,.10)'>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700'>Recent performance vs the S&P 500</div>"
            f"<div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:12px'>{performance_window_chip('1Y', item.relative_1y_return_pct)}{performance_window_chip('3Y', item.relative_3y_return_pct)}{performance_window_chip('5Y', item.relative_5y_return_pct)}</div>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700;margin-top:16px'>What likely improves if you do this</div>"
            f"<div style='font-size:14px;line-height:1.6;color:#e2e8f0;margin-top:10px'>{render_bold_markers(item.portfolio_impact_summary)}</div>"
            f"<ul style='margin:12px 0 0 18px;color:#e2e8f0;line-height:1.55'>{impacts}</ul>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700;margin-top:16px'>Extra market and company signals that added pressure</div>"
            f"<ul style='margin:12px 0 0 18px;color:#e2e8f0;line-height:1.55'>{modifiers}</ul>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700;margin-top:16px'>Evidence used</div>"
            f"<ul style='margin:12px 0 0 18px;color:#e2e8f0;line-height:1.55'>{evidence}</ul>"
            "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700;margin-top:16px'>What kept this from being more aggressive</div>"
            f"<ul style='margin:12px 0 0 18px;color:#e2e8f0;line-height:1.55'>{guardrails}</ul>"
            "</div>"
            "</div>"
            "</div>"
        )

    held_section = ""
    if held:
        held_cards = "".join(
            "<div style='padding:12px 14px;border-radius:14px;border:1px solid rgba(148,163,184,.12);background:rgba(15,23,42,.26)'>"
            f"<div style='font-size:16px;font-weight:700;color:#f8fafc'>{item.ticker} <span style='font-size:12px;color:#34d399;font-weight:700'>{item.recommendation_label}</span></div>"
            f"<div style='font-size:13px;line-height:1.55;color:#cbd5e1;margin-top:8px'>{render_bold_markers(item.recommendation_summary)}</div>"
            "</div>"
            for item in held
        )
        held_section = (
            "<div style='padding:18px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
            "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.94));margin-top:18px'>"
            "<div style='font-size:18px;font-weight:700;color:#f8fafc'>Names Not Flagged for Sale Right Now</div>"
            "<div style='font-size:13px;color:#93c5fd;margin-top:6px'>This is here to show the rule is not blindly selling every higher-risk name. Some holdings still look acceptable once their broader performance versus the S&P 500 is considered.</div>"
            f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:14px'>{held_cards}</div>"
            "</div>"
        )

    return (
        "<div style='display:grid;gap:16px'>"
        "<div style='padding:20px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
        "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.94))'>"
        "<div style='font-size:22px;font-weight:800;color:#f8fafc'>Risk Actions</div>"
        "<div style='font-size:14px;color:#93c5fd;margin-top:8px'>This tab is narrower than a full rebalance engine. It focuses on one question: which holdings now have enough underperformance versus the S&P 500, combined with portfolio risk and recent company or market caution signals, to justify trimming or selling.</div>"
        f"<div style='margin-top:16px'>{impact_section}{cards}</div>"
        "</div>"
        f"{held_section}"
        "</div>"
    )


def build_portfolio_gap_frame(diagnosis: PortfolioRiskDiagnosis) -> pd.DataFrame:
    """Create a quick-read table for the Portfolio Gaps tab."""
    rows: list[dict[str, Any]] = []
    for gap in diagnosis.portfolio_gaps:
        rows.append(
            {
                "Gap": gap.label,
                "Severity": f"{gap.severity_score:.1f}/100",
                "Band": gap.severity_band,
                "Why this matters now": gap.why_this_gap_exists,
                "What a future add should do": " | ".join(gap.what_would_change[:2]),
                "Default tilt": gap.suggested_vehicle_tilt,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "Gap",
                "Severity",
                "Band",
                "Why this matters now",
                "What a future add should do",
                "Default tilt",
            ]
        )
    frame = pd.DataFrame(rows)
    frame["_sort"] = frame["Severity"].str.extract(r"([0-9.]+)").astype(float)
    return frame.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)


def build_portfolio_preferences_frame(diagnosis: PortfolioRiskDiagnosis) -> pd.DataFrame:
    """Render the inferred PortfolioPreferences object as a review table."""
    preferences = diagnosis.portfolio_preferences
    if preferences is None:
        return pd.DataFrame(columns=["Preference", "Current setting"])

    rows = [
        {"Preference": "Available cash now", "Current setting": money_text(preferences.available_cash_now)},
        {
            "Preference": "Available cash if current actions are followed",
            "Current setting": money_text(preferences.available_cash_if_actions_followed),
        },
        {"Preference": "Reinvest freed cash", "Current setting": preferences.reinvestment_preference_label},
        {
            "Preference": "Stated risk tolerance",
            "Current setting": f"{preferences.stated_risk_score:.0f}/100 ({preferences.stated_risk_band})",
        },
        {
            "Preference": "Suggested max size for a new add",
            "Current setting": percent_display(preferences.suggested_max_new_position_pct),
        },
        {"Preference": "ETFs allowed", "Current setting": "Yes" if preferences.allow_etfs else "No"},
        {"Preference": "Single stocks allowed", "Current setting": "Yes" if preferences.allow_single_stocks else "No"},
        {"Preference": "Default vehicle tilt", "Current setting": preferences.vehicle_preference_label},
        {
            "Preference": "Sector preferences",
            "Current setting": ", ".join(preferences.sector_preferences) if preferences.sector_preferences else "Not set yet",
        },
        {
            "Preference": "Inferred sector avoidances",
            "Current setting": ", ".join(preferences.inferred_sector_avoidances) if preferences.inferred_sector_avoidances else "None inferred",
        },
        {
            "Preference": "Unresolved user decisions",
            "Current setting": " | ".join(preferences.unresolved_preferences),
        },
        {
            "Preference": "Assumptions being used right now",
            "Current setting": " | ".join(preferences.assumption_notes),
        },
    ]
    return pd.DataFrame(rows)


def build_portfolio_gaps_html(diagnosis: PortfolioRiskDiagnosis) -> str:
    """Render a scan-first summary for the Portfolio Gaps tab."""
    preferences = diagnosis.portfolio_preferences
    if not diagnosis.portfolio_gaps:
        return (
            "<div style='padding:20px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
            "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.94))'>"
            "<div style='font-size:22px;font-weight:800;color:#f8fafc'>Portfolio Gaps</div>"
            "<div style='font-size:15px;color:#cbd5e1;margin-top:10px'>No portfolio gaps were surfaced from the current action set.</div>"
            "</div>"
        )

    top_gap = diagnosis.portfolio_gaps[0]
    cash_value = preferences.available_cash_if_actions_followed if preferences is not None else 0.0
    max_new_position = preferences.suggested_max_new_position_pct if preferences is not None else None
    sector_avoidance = (
        ", ".join(preferences.inferred_sector_avoidances)
        if preferences is not None and preferences.inferred_sector_avoidances
        else "None inferred"
    )
    quick_cards = (
        metric_card("Top portfolio need", top_gap.label, top_gap.severity_band)
        + metric_card("Capital available", money_text(cash_value), "If current actions are followed")
        + metric_card("Max new add", percent_display(max_new_position), "Temporary cap for a new position")
        + metric_card("Avoid stacking into", sector_avoidance, "Current crowded area")
    )

    top_gap_bullets = "".join(
        f"<li style='margin-bottom:7px'>{render_bold_markers(item)}</li>"
        for item in top_gap.what_would_change[:3]
    )

    preference_section = ""
    if preferences is not None:
        unresolved = "".join(
            f"<li style='margin-bottom:7px'>{render_bold_markers(item)}</li>"
            for item in preferences.unresolved_preferences[:4]
        )
        preference_section = (
            "<div style='padding:18px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
            "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.94));margin-top:18px'>"
            "<div style='font-size:18px;font-weight:800;color:#f8fafc'>Before buy ideas, we still need</div>"
            f"<ul style='margin:12px 0 0 18px;color:#e2e8f0;line-height:1.55'>{unresolved}</ul>"
            "</div>"
        )

    return (
        "<div style='display:flex;flex-direction:column;gap:16px'>"
        "<div style='padding:20px;border:1px solid rgba(148,163,184,.16);border-radius:18px;"
        "background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.94))'>"
        "<div style='font-size:22px;font-weight:800;color:#f8fafc'>Portfolio Gaps</div>"
        "<div style='font-size:15px;color:#cbd5e1;margin-top:10px'>Why buy anything at all? This view shows what the portfolio still needs after the current trims, before we suggest any names.</div>"
        f"<div class='metric-strip' style='margin-top:14px'>{quick_cards}</div>"
        "<div style='padding:14px 16px;border-radius:14px;background:rgba(30,41,59,.42);border:1px solid rgba(148,163,184,.10);margin-top:16px'>"
        f"<div style='font-size:14px;color:#93c5fd;font-weight:700'>Fast read: the biggest need right now is <span style='color:#f8fafc'>{top_gap.label}</span>.</div>"
        f"<div style='font-size:14px;line-height:1.6;color:#cbd5e1;margin-top:8px'>{render_bold_markers(top_gap.why_this_gap_exists)}</div>"
        "<div style='font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#93c5fd;font-weight:700;margin-top:14px'>If a future add solves this well</div>"
        f"<ul style='margin:10px 0 0 18px;color:#e2e8f0;line-height:1.55'>{top_gap_bullets}</ul>"
        "</div>"
        + preference_section
        + "</div>"
    )


def build_single_ticker_timeseries_records(
    ticker: str,
    ticker_close: pd.DataFrame,
    benchmark_close: pd.Series,
) -> list[dict[str, Any]]:
    if ticker_close.empty or ticker not in ticker_close.columns or benchmark_close.empty:
        return []
    frame = pd.concat(
        [
            pd.to_numeric(ticker_close[ticker], errors="coerce").rename("ticker_value"),
            pd.to_numeric(benchmark_close, errors="coerce").rename("benchmark_raw_value"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if frame.empty:
        return []
    base_ticker = float(frame["ticker_value"].iloc[0])
    base_benchmark = float(frame["benchmark_raw_value"].iloc[0])
    if base_ticker <= 0 or base_benchmark <= 0:
        return []
    frame.index.name = "date"
    scale = 10000.0
    frame["portfolio_invested_value"] = (frame["ticker_value"] / base_ticker) * scale
    frame["benchmark_value"] = (frame["benchmark_raw_value"] / base_benchmark) * scale
    frame["trade_flow"] = 0.0
    frame["account_value"] = frame["portfolio_invested_value"]
    frame["portfolio_performance_index"] = frame["portfolio_invested_value"] / scale
    frame["benchmark_performance_index"] = frame["benchmark_value"] / scale
    return (
        frame.reset_index()
        [
            [
                "date",
                "portfolio_invested_value",
                "account_value",
                "benchmark_value",
                "trade_flow",
                "portfolio_performance_index",
                "benchmark_performance_index",
            ]
        ]
        .round(6)
        .to_dict(orient="records")
    )


def build_diagnosis_risk_chart_payload(
    diagnosis: PortfolioRiskDiagnosis,
    market_metrics: dict[str, Any],
    market_data: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    payload: dict[str, Any] = {"Portfolio": market_metrics["timeseries"]}
    ticker_close = market_data.get("ticker_close", pd.DataFrame())
    benchmark_close = market_data.get("benchmark_close", pd.Series(dtype=float))
    ranked_tickers = [driver.ticker for driver in diagnosis.top_holding_drivers]
    remaining_tickers = [
        row["ticker"]
        for row in sorted(
            market_metrics.get("open_positions", []),
            key=lambda item: float(item.get("current_weight") or 0.0),
            reverse=True,
        )
        if row["ticker"] not in ranked_tickers
    ]
    ordered_tickers = ranked_tickers + remaining_tickers
    available_choices = ["Portfolio"]
    for ticker in ordered_tickers:
        if ticker in payload:
            continue
        records = build_single_ticker_timeseries_records(ticker, ticker_close, benchmark_close)
        if not records:
            continue
        payload[ticker] = records
        available_choices.append(ticker)
    return payload, available_choices


def plot_diagnosis_volatility_lens(timeseries_records: list[dict[str, Any]], subject_label: str) -> go.Figure:
    fig = plot_recent_volatility_comparison(timeseries_records)
    fig.update_layout(title=f"Volatility of {subject_label} vs S&P 500")
    return fig


def plot_diagnosis_drawdown_lens(timeseries_records: list[dict[str, Any]], subject_label: str) -> go.Figure:
    fig = plot_drawdowns(timeseries_records)
    fig.update_layout(title=f"Downside Depth of {subject_label} vs S&P 500")
    return fig


def plot_diagnosis_market_sensitivity_lens(timeseries_records: list[dict[str, Any]], subject_label: str) -> go.Figure:
    frame = rolling_relative_market_sensitivity_frame(timeseries_records)
    if frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Recent market sensitivity evidence is unavailable for {subject_label}.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15, "color": "#e2e8f0"},
        )
        fig.update_layout(
            title=f"Market Sensitivity of {subject_label} vs S&P 500",
            height=520,
            margin={"l": 24, "r": 24, "t": 56, "b": 24},
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.55)",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    x_values = frame["date"].dt.strftime("%Y-%m-%d").tolist()
    y_values = frame["ratio"].tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            line={"width": 3, "color": "#A855F7"},
            hovertemplate=(
                "%{x}<br>Market sensitivity ratio: %{y:.2f}x"
                "<br>What this means: how strongly this holding moves when the S&P 500 moves"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="#10B981")
    fig.add_hline(y=RELATIVE_MARKET_SENSITIVITY_MAX_RATIO, line_dash="dot", line_color="#F59E0B")
    if x_values:
        fig.add_annotation(
            x=x_values[-1],
            y=float(y_values[-1]),
            text=f"Latest: {y_values[-1]:.2f}x",
            showarrow=True,
            arrowhead=2,
            ax=42,
            ay=-26,
            bgcolor="rgba(15,23,42,0.92)",
            bordercolor="#A855F7",
            font={"color": "#f8fafc", "size": 11},
        )
    fig.update_layout(
        title=f"Market Sensitivity of {subject_label} vs S&P 500",
        height=520,
        margin={"l": 24, "r": 24, "t": 56, "b": 24},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.55)",
        hovermode="x unified",
        hoverlabel=dark_hoverlabel(),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(
        title_text="x Benchmark",
        gridcolor="rgba(148,163,184,0.18)",
        range=padded_axis_range(y_values, baseline_values=[1.0, RELATIVE_MARKET_SENSITIVITY_MAX_RATIO], min_padding=0.08),
    )
    return fig


def update_diagnosis_stock_plots(selection: str, chart_payload: dict[str, Any]) -> tuple[go.Figure, go.Figure, go.Figure]:
    key = selection if selection in chart_payload else "Portfolio"
    subject_label = "Portfolio" if key == "Portfolio" else key
    timeseries_records = chart_payload.get(key, chart_payload.get("Portfolio", []))
    return (
        plot_diagnosis_volatility_lens(timeseries_records, subject_label),
        plot_diagnosis_drawdown_lens(timeseries_records, subject_label),
        plot_diagnosis_market_sensitivity_lens(timeseries_records, subject_label),
    )


def open_metric_guide(metric_key: str, risk: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    if risk is None:
        raise gr.Error("Run the analysis first so the guide can open the right metric.")
    return gr.Tabs.update(selected="risk-guide"), build_risk_guide_html(risk, focused_metric=metric_key)


def format_display_tables(market_metrics: dict[str, Any]) -> dict[str, pd.DataFrame]:
    holdings = dataframe_from_records(
        market_metrics["open_positions"],
        [
            "ticker",
            "quantity",
            "avg_cost",
            "latest_price",
            "cost_basis_total",
            "current_value",
            "unrealized_pnl",
            "unrealized_return_pct",
            "current_weight",
            "excess_return_vs_benchmark",
        ],
    )
    holdings = format_display_dataframe(
        holdings,
        currency_columns=["avg_cost", "latest_price", "cost_basis_total", "current_value", "unrealized_pnl"],
        percent_columns=["unrealized_return_pct", "current_weight", "excess_return_vs_benchmark"],
        number_columns={"quantity": 4},
    )
    attribution = dataframe_from_records(
        market_metrics["performance_attribution"],
        ["ticker", "realized_pnl", "unrealized_pnl", "combined_pnl", "pnl_contribution_pct"],
    )
    attribution = format_display_dataframe(
        attribution,
        currency_columns=["realized_pnl", "unrealized_pnl", "combined_pnl"],
        percent_columns=["pnl_contribution_pct"],
    )
    sold = dataframe_from_records(
        market_metrics["sold_too_early"],
        ["ticker", "missed_upside", "realized_pnl", "if_held_pnl"],
    )
    sold = format_display_dataframe(
        sold,
        currency_columns=["missed_upside", "realized_pnl", "if_held_pnl"],
    )
    selection_alpha = dataframe_from_records(market_metrics["selection_alpha"], ["ticker", "alpha_pnl"])
    selection_alpha = format_display_dataframe(
        selection_alpha,
        currency_columns=["alpha_pnl"],
    )
    volatility_drivers = dataframe_from_records(
        market_metrics.get("top_volatility_drivers_2025", []),
        ["ticker", "avg_weight_pct", "asset_volatility_pct", "variance_contribution_pct"],
    )
    volatility_drivers = format_display_dataframe(
        volatility_drivers,
        percent_columns=["avg_weight_pct", "asset_volatility_pct", "variance_contribution_pct"],
    )
    explanations = metric_explanations()
    risk_metric_order = [
        "concentration::single_position_weight",
        "concentration::top_5_weight",
        "concentration::effective_holdings",
        "behavior::turnover",
        "behavior::short_holding_period",
        "market::relative_volatility_to_benchmark",
        "market::relative_drawdown_to_benchmark",
        "market::relative_downside_capture_to_benchmark",
        "market::relative_market_sensitivity_to_benchmark",
        "market::equity_exposure",
    ]
    # Keep the table short and scannable; the richer narrative lives in the cards above.
    risk_components = pd.DataFrame(
        [
            {
                "group": explanations[key]["group"],
                "metric": explanations[key]["label"],
                "score": market_metrics["risk_score"]["component_scores"].get(key),
            }
            for key in risk_metric_order
            if key in market_metrics["risk_score"]["component_scores"]
        ]
    )
    risk_components = format_display_dataframe(risk_components, number_columns={"score": 1})
    projection = dataframe_from_records(
        market_metrics["projection_scenarios_no_new_contributions"]["table"],
        ["year", "conservative", "base", "optimistic"],
    )
    projection = format_display_dataframe(
        projection,
        currency_columns=["conservative", "base", "optimistic"],
        number_columns={"year": 0},
    )
    return {
        "holdings": holdings,
        "attribution": attribution,
        "sold": sold,
        "selection_alpha": selection_alpha,
        "volatility_drivers": volatility_drivers,
        "risk_components": risk_components,
        "projection": projection,
    }


def plot_equity_curves(timeseries_records: list[dict[str, Any]]) -> go.Figure:
    series = pd.DataFrame(timeseries_records)
    fig = go.Figure()
    if not series.empty:
        series["date"] = pd.to_datetime(series["date"])
        x_values = series["date"].dt.strftime("%Y-%m-%d").tolist()
        portfolio_values = pd.to_numeric(series["portfolio_invested_value"], errors="coerce").fillna(0.0).tolist()
        benchmark_values = pd.to_numeric(series["benchmark_value"], errors="coerce").fillna(0.0).tolist()
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=portfolio_values,
                mode="lines",
                name="Invested Portfolio",
                line={"width": 3, "color": "#3B82F6"},
                hovertemplate="%{x|%Y-%m-%d}<br>Portfolio: $%{y:,.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=benchmark_values,
                mode="lines",
                name="Trade-Matched S&P 500",
                line={"width": 3, "color": "#F59E0B"},
                hovertemplate="%{x|%Y-%m-%d}<br>S&P 500: $%{y:,.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Portfolio vs S&P 500 Over Your Investing Horizon",
        height=430,
        margin={"l": 24, "r": 24, "t": 56, "b": 24},
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode="x unified",
        template="plotly_dark",
        title_font={"size": 24},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.55)",
        hoverlabel=dark_hoverlabel(),
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True, gridcolor="rgba(148,163,184,0.18)")
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.18)")
    return fig


def plot_drawdowns(timeseries_records: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    frame = rolling_relative_drawdown_frame(timeseries_records)
    if not frame.empty:
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        x_values = frame["date"].dt.strftime("%Y-%m-%d").tolist()
        # The rolling drawdown engine stores drawdown depth as positive magnitudes.
        # Flip them negative for charting so the visual reads like a normal drawdown curve.
        portfolio_drawdown = (-pd.to_numeric(frame["portfolio_drawdown"], errors="coerce").fillna(0.0) * 100).tolist()
        benchmark_drawdown = (-pd.to_numeric(frame["benchmark_drawdown"], errors="coerce").fillna(0.0) * 100).tolist()
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=portfolio_drawdown,
                mode="lines",
                name="Portfolio Drawdown",
                line={"width": 2.5, "color": "#EF4444"},
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.18)",
                hovertemplate="%{x|%Y-%m-%d}<br>Portfolio DD: %{y:.2f}%<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=benchmark_drawdown,
                mode="lines",
                name="S&P 500 Drawdown",
                line={"width": 2.5, "color": "#A855F7"},
                fill="tozeroy",
                fillcolor="rgba(168,85,247,0.14)",
                hovertemplate="%{x|%Y-%m-%d}<br>S&P DD: %{y:.2f}%<extra></extra>",
            )
        )
        if x_values:
            fig.add_annotation(
                x=x_values[-1],
                y=portfolio_drawdown[-1],
                text=f"Latest: {portfolio_drawdown[-1]:.1f}%",
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=-24,
                bgcolor="rgba(15,23,42,0.92)",
                bordercolor="#EF4444",
                font={"color": "#f8fafc", "size": 11},
            )
            fig.add_annotation(
                x=x_values[-1],
                y=benchmark_drawdown[-1],
                text=f"S&P: {benchmark_drawdown[-1]:.1f}%",
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=24,
                bgcolor="rgba(15,23,42,0.92)",
                bordercolor="#A855F7",
                font={"color": "#f8fafc", "size": 11},
            )
            y_range = padded_axis_range(portfolio_drawdown + benchmark_drawdown, baseline_values=[0.0], min_padding=2.5)
            if y_range is not None:
                fig.update_yaxes(range=y_range)
    else:
        fig.add_annotation(
            text="Recent rolling drawdown evidence is unavailable for the last 18 months.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15, "color": "#e2e8f0"},
        )
    fig.update_layout(
        title="Recent 6-Month Rolling Drawdown Depth vs S&P 500",
        height=520,
        margin={"l": 24, "r": 24, "t": 56, "b": 24},
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        template="plotly_dark",
        title_font={"size": 24},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.55)",
        hoverlabel=dark_hoverlabel() ,
    )
    fig.update_yaxes(ticksuffix="%", gridcolor="rgba(148,163,184,0.18)")
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.18)")
    return fig


def plot_sector_allocation(sector_rows: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    frame = pd.DataFrame(sector_rows)
    if not frame.empty:
        frame = frame.copy()
        frame["current_value"] = pd.to_numeric(frame["current_value"], errors="coerce").fillna(0.0)
        frame["weight_pct"] = pd.to_numeric(frame["weight_pct"], errors="coerce").fillna(0.0) * 100
        frame = frame[frame["current_value"] > 0].sort_values("current_value", ascending=False)
        fig.add_trace(
            go.Pie(
                labels=frame["sector"].tolist(),
                values=frame["current_value"].tolist(),
                hole=0.46,
                textinfo="label+percent",
                textposition="inside",
                hovertemplate=(
                    "%{label}<br>Current value: $%{value:,.2f}"
                    "<br>Weight: %{percent}"
                    "<extra></extra>"
                ),
                marker={"line": {"color": "rgba(15,23,42,0.95)", "width": 2}},
            )
        )
    else:
        fig.add_annotation(
            text="Sector allocation is unavailable for the current portfolio.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15, "color": "#e2e8f0"},
        )
    fig.update_layout(
        title="Current Sector Allocation",
        height=430,
        margin={"l": 24, "r": 24, "t": 56, "b": 24},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.55)",
        showlegend=True,
        legend={"orientation": "v", "x": 1.02, "xanchor": "left", "y": 0.5, "yanchor": "middle"},
        hoverlabel=dark_hoverlabel(),
    )
    return fig


def plot_projection(projection_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not projection_df.empty:
        # The table shown in the UI is currency-formatted, so the plot parser strips display symbols as needed.
        years = pd.Series(projection_df["year"]).apply(parse_display_number).fillna(0).astype(int).tolist()
        conservative = pd.Series(projection_df["conservative"]).apply(parse_display_number).fillna(0.0).tolist()
        base = pd.Series(projection_df["base"]).apply(parse_display_number).fillna(0.0).tolist()
        optimistic = pd.Series(projection_df["optimistic"]).apply(parse_display_number).fillna(0.0).tolist()
        fig.add_trace(
            go.Scatter(
                x=years,
                y=conservative,
                mode="lines",
                name="Conservative",
                line={"width": 3, "color": "#10B981"},
                hovertemplate="Year %{x}<br>Conservative: $%{y:,.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=years,
                y=base,
                mode="lines",
                name="Base",
                line={"width": 3, "color": "#3B82F6"},
                hovertemplate="Year %{x}<br>Base: $%{y:,.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=years,
                y=optimistic,
                mode="lines",
                name="Optimistic",
                line={"width": 3, "color": "#F97316"},
                hovertemplate="Year %{x}<br>Optimistic: $%{y:,.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="18-Year Projection Without New Contributions",
        height=400,
        margin={"l": 24, "r": 24, "t": 56, "b": 24},
        xaxis_title="Years From Today",
        yaxis_title="Projected Value ($)",
        hovermode="x unified",
        template="plotly_dark",
        title_font={"size": 24},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.55)",
        hoverlabel=dark_hoverlabel(),
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True, gridcolor="rgba(148,163,184,0.18)")
    fig.update_xaxes(dtick=1, gridcolor="rgba(148,163,184,0.18)")
    return fig


def build_market_evidence_series(timeseries_records: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    series = pd.DataFrame(timeseries_records)
    if series.empty:
        return {"aligned_performance": pd.DataFrame(), "aligned_returns": pd.DataFrame()}

    series["date"] = pd.to_datetime(series["date"])
    aligned_performance = (
        series[["date", "portfolio_performance_index", "benchmark_performance_index"]]
        .rename(
            columns={
                "portfolio_performance_index": "portfolio",
                "benchmark_performance_index": "benchmark",
            }
        )
        .dropna()
        .set_index("date")
        .sort_index()
    )
    aligned_returns = (
        aligned_performance.pct_change(fill_method=None).replace([math.inf, -math.inf], float("nan")).dropna()
        if not aligned_performance.empty
        else pd.DataFrame()
    )
    return {"aligned_performance": aligned_performance, "aligned_returns": aligned_returns}


def modified_dietz_return(end_value: float, begin_value: float, flow_value: float) -> float | None:
    if pd.isna(begin_value) or begin_value <= 0:
        return None
    denominator = begin_value + (0.5 * flow_value)
    if denominator <= 0:
        return None
    return float((end_value - begin_value - flow_value) / denominator)


def build_stable_weekly_value_return_frame(timeseries_records: list[dict[str, Any]]) -> pd.DataFrame:
    series = pd.DataFrame(timeseries_records)
    if series.empty:
        return pd.DataFrame(columns=["date", "portfolio_ret", "benchmark_ret"])

    required_columns = {"date", "portfolio_invested_value", "benchmark_value", "trade_flow"}
    if not required_columns.issubset(series.columns):
        return pd.DataFrame(columns=["date", "portfolio_ret", "benchmark_ret"])

    series["date"] = pd.to_datetime(series["date"])
    series = series.set_index("date").sort_index()
    weekly = (
        pd.DataFrame(
            {
                "portfolio_value": pd.to_numeric(series["portfolio_invested_value"], errors="coerce"),
                "benchmark_value": pd.to_numeric(series["benchmark_value"], errors="coerce"),
                "trade_flow": pd.to_numeric(series["trade_flow"], errors="coerce").fillna(0.0),
            }
        )
        .resample("W-FRI")
        .agg({"portfolio_value": "last", "benchmark_value": "last", "trade_flow": "sum"})
        .dropna()
    )
    if weekly.empty:
        return pd.DataFrame(columns=["date", "portfolio_ret", "benchmark_ret"])

    weekly["portfolio_begin"] = weekly["portfolio_value"].shift(1)
    weekly["benchmark_begin"] = weekly["benchmark_value"].shift(1)
    weekly["portfolio_ret"] = [
        modified_dietz_return(end_value, begin_value, flow_value)
        for end_value, begin_value, flow_value in zip(
            weekly["portfolio_value"],
            weekly["portfolio_begin"],
            weekly["trade_flow"],
        )
    ]
    weekly["benchmark_ret"] = [
        modified_dietz_return(end_value, begin_value, flow_value)
        for end_value, begin_value, flow_value in zip(
            weekly["benchmark_value"],
            weekly["benchmark_begin"],
            weekly["trade_flow"],
        )
    ]

    current_portfolio_value = float(pd.to_numeric(series["portfolio_invested_value"], errors="coerce").dropna().iloc[-1])
    current_benchmark_value = float(pd.to_numeric(series["benchmark_value"], errors="coerce").dropna().iloc[-1])
    portfolio_floor = max(1000.0, current_portfolio_value * 0.02)
    benchmark_floor = max(1000.0, current_benchmark_value * 0.02)

    weekly["flow_ratio"] = (
        weekly["trade_flow"].abs() / weekly["portfolio_begin"].abs().replace(0.0, pd.NA)
    )
    # Treat extreme one-week jumps as unstable reconstruction periods rather than real
    # market behavior. This keeps the evidence and score focused on economically believable
    # weekly moves instead of weeks where the sleeve math is clearly breaking down.
    stable_mask = (
        (weekly["portfolio_value"] >= portfolio_floor)
        & (weekly["portfolio_begin"] >= portfolio_floor)
        & (weekly["benchmark_value"] >= benchmark_floor)
        & (weekly["benchmark_begin"] >= benchmark_floor)
        & (weekly["flow_ratio"].fillna(0.0) <= WEEKLY_FLOW_DOMINANCE_LIMIT)
        & (weekly["portfolio_ret"].abs() <= MAX_STABLE_WEEKLY_RETURN)
    )
    weekly = weekly.loc[stable_mask].copy()
    weekly = (
        weekly.replace([math.inf, -math.inf], float("nan"))
        .dropna(subset=["portfolio_ret", "benchmark_ret"])
        .reset_index()
        .rename(columns={"index": "date"})
    )
    return weekly[["date", "portfolio_ret", "benchmark_ret"]]


def build_rolling_relative_volatility_frame(weekly_returns: pd.DataFrame) -> pd.DataFrame:
    if weekly_returns.empty:
        return pd.DataFrame(columns=["date", "ratio", "portfolio_volatility", "benchmark_volatility"])

    rows: list[dict[str, Any]] = []
    window_delta = pd.Timedelta(days=ROLLING_DRAWDOWN_WINDOW_DAYS)
    weekly_returns = weekly_returns.copy()
    weekly_returns["date"] = pd.to_datetime(weekly_returns["date"])
    weekly_returns = weekly_returns.set_index("date").sort_index()
    for end_date in weekly_returns.index:
        window_slice = weekly_returns.loc[end_date - window_delta : end_date]
        if len(window_slice) < 8:
            continue
        benchmark_vol = float(window_slice["benchmark_ret"].std() * math.sqrt(52))
        if benchmark_vol <= 0:
            continue
        portfolio_vol = float(window_slice["portfolio_ret"].std() * math.sqrt(52))
        rows.append(
            {
                "date": end_date,
                "ratio": portfolio_vol / benchmark_vol,
                "portfolio_volatility": portfolio_vol,
                "benchmark_volatility": benchmark_vol,
            }
        )
    return pd.DataFrame(rows, columns=["date", "ratio", "portfolio_volatility", "benchmark_volatility"])


def build_rolling_relative_drawdown_frame(weekly_returns: pd.DataFrame) -> pd.DataFrame:
    if weekly_returns.empty:
        return pd.DataFrame(columns=["date", "ratio", "portfolio_drawdown", "benchmark_drawdown"])

    rows: list[dict[str, Any]] = []
    window_delta = pd.Timedelta(days=ROLLING_DRAWDOWN_WINDOW_DAYS)
    clean_returns = weekly_returns.copy()
    clean_returns["date"] = pd.to_datetime(clean_returns["date"])
    clean_returns = clean_returns.set_index("date").sort_index().dropna()
    wealth_frame = pd.DataFrame(
        {
            "portfolio": (1 + clean_returns["portfolio_ret"]).cumprod(),
            "benchmark": (1 + clean_returns["benchmark_ret"]).cumprod(),
        }
    ).dropna()
    for end_date in wealth_frame.index:
        window_slice = wealth_frame.loc[end_date - window_delta : end_date]
        if len(window_slice) < 2:
            continue
        portfolio_dd = max_drawdown_for_series(window_slice["portfolio"])
        benchmark_dd = max_drawdown_for_series(window_slice["benchmark"])
        if portfolio_dd is None or benchmark_dd is None or benchmark_dd <= 0:
            continue
        rows.append(
            {
                "date": end_date,
                "ratio": portfolio_dd / benchmark_dd,
                "portfolio_drawdown": portfolio_dd,
                "benchmark_drawdown": benchmark_dd,
            }
        )
    return pd.DataFrame(rows, columns=["date", "ratio", "portfolio_drawdown", "benchmark_drawdown"])


def build_rolling_relative_market_sensitivity_frame(aligned_returns: pd.DataFrame) -> pd.DataFrame:
    if aligned_returns.empty:
        return pd.DataFrame(columns=["date", "ratio"])

    rows: list[dict[str, Any]] = []
    window_delta = pd.Timedelta(days=ROLLING_DRAWDOWN_WINDOW_DAYS)
    clean_returns = aligned_returns.copy()
    if "date" in clean_returns.columns:
        clean_returns["date"] = pd.to_datetime(clean_returns["date"])
        clean_returns = clean_returns.set_index("date")
    clean_returns.index = pd.to_datetime(clean_returns.index)
    clean_returns = clean_returns.sort_index().dropna()
    if "portfolio_ret" in clean_returns.columns and "benchmark_ret" in clean_returns.columns:
        clean_returns = clean_returns.rename(columns={"portfolio_ret": "portfolio", "benchmark_ret": "benchmark"})
    for end_date in clean_returns.index:
        window_slice = clean_returns.loc[end_date - window_delta : end_date]
        if len(window_slice) < 5:
            continue
        benchmark_var = float(window_slice["benchmark"].var())
        if benchmark_var <= 0:
            continue
        rows.append(
            {
                "date": end_date,
                "ratio": float(window_slice["portfolio"].cov(window_slice["benchmark"]) / benchmark_var),
            }
        )
    return pd.DataFrame(rows, columns=["date", "ratio"])


def trim_to_recent_window(
    df: pd.DataFrame,
    date_column: str,
    lookback_days: int,
    *,
    fallback_to_full: bool = True,
) -> pd.DataFrame:
    if df.empty or date_column not in df.columns:
        return df
    latest_date = pd.to_datetime(df[date_column]).max()
    if pd.isna(latest_date):
        return df
    cutoff = latest_date - pd.Timedelta(days=lookback_days)
    trimmed = df[pd.to_datetime(df[date_column]) >= cutoff].copy()
    if not trimmed.empty:
        return trimmed
    return df.copy() if fallback_to_full else df.iloc[0:0].copy()


def trim_to_start_date(
    df: pd.DataFrame,
    date_column: str,
    start_date: str,
    *,
    fallback_to_full: bool = True,
) -> pd.DataFrame:
    if df.empty or date_column not in df.columns:
        return df
    start_timestamp = pd.Timestamp(start_date)
    trimmed = df[pd.to_datetime(df[date_column]) >= start_timestamp].copy()
    if not trimmed.empty:
        return trimmed
    return df.copy() if fallback_to_full else df.iloc[0:0].copy()


def trim_index_to_recent_window(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    latest_date = df.index.max()
    if pd.isna(latest_date):
        return df
    cutoff = latest_date - pd.Timedelta(days=lookback_days)
    trimmed = df.loc[df.index >= cutoff].copy()
    return trimmed if not trimmed.empty else df.copy()


def padded_axis_range(
    values: list[float],
    *,
    baseline_values: list[float] | None = None,
    min_padding: float = 0.15,
) -> list[float] | None:
    clean_values = [float(value) for value in values if value is not None and not pd.isna(value)]
    if baseline_values:
        clean_values.extend(float(value) for value in baseline_values if value is not None and not pd.isna(value))
    if not clean_values:
        return None
    min_value = min(clean_values)
    max_value = max(clean_values)
    spread = max_value - min_value
    padding = max(spread * 0.15, min_padding)
    lower = min_value - padding
    upper = max_value + padding
    if lower == upper:
        lower -= min_padding
        upper += min_padding
    if min_value >= 0:
        lower = max(0.0, lower)
    return [lower, upper]


def add_latest_annotation(fig: go.Figure, x_value: str, y_value: float, text: str, color: str, *, row: int, col: int) -> None:
    fig.add_annotation(
        x=x_value,
        y=y_value,
        text=text,
        showarrow=True,
        arrowhead=2,
        ax=42,
        ay=-26,
        bgcolor="rgba(15,23,42,0.92)",
        bordercolor=color,
        font={"color": "#f8fafc", "size": 11},
        row=row,
        col=col,
    )


def dark_hoverlabel() -> dict[str, Any]:
    return {
        "bgcolor": "rgba(15,23,42,0.96)",
        "bordercolor": "#475569",
        "font": {"color": "#f8fafc", "size": 13},
    }


def rolling_relative_volatility_frame(timeseries_records: list[dict[str, Any]]) -> pd.DataFrame:
    weekly_returns = build_stable_weekly_value_return_frame(timeseries_records)
    if weekly_returns.empty:
        return pd.DataFrame(columns=["date", "ratio"])
    frame = build_rolling_relative_volatility_frame(weekly_returns)
    return trim_to_start_date(frame, "date", RISK_CHART_START_DATE, fallback_to_full=False)


def rolling_relative_drawdown_frame(timeseries_records: list[dict[str, Any]]) -> pd.DataFrame:
    weekly_returns = build_stable_weekly_value_return_frame(timeseries_records)
    if weekly_returns.empty:
        return pd.DataFrame(columns=["date", "ratio", "portfolio_drawdown", "benchmark_drawdown"])
    frame = build_rolling_relative_drawdown_frame(weekly_returns)
    return trim_to_start_date(frame, "date", RISK_CHART_START_DATE, fallback_to_full=False)


def rolling_relative_downside_capture_frame(timeseries_records: list[dict[str, Any]]) -> pd.DataFrame:
    aligned_returns = build_market_evidence_series(timeseries_records)["aligned_returns"]
    if aligned_returns.empty:
        return pd.DataFrame(columns=["date", "ratio"])

    rows: list[dict[str, Any]] = []
    window_delta = pd.Timedelta(days=ROLLING_DRAWDOWN_WINDOW_DAYS)
    for end_date in aligned_returns.index:
        window_slice = aligned_returns.loc[end_date - window_delta : end_date]
        downside_slice = window_slice[window_slice["benchmark"] < 0]
        if len(downside_slice) < 3:
            continue
        benchmark_down_mean = float(downside_slice["benchmark"].mean())
        if benchmark_down_mean == 0:
            continue
        rows.append({"date": end_date, "ratio": float(downside_slice["portfolio"].mean() / benchmark_down_mean)})
    frame = pd.DataFrame(rows, columns=["date", "ratio"])
    return trim_to_start_date(frame, "date", RISK_CHART_START_DATE, fallback_to_full=False)


def rolling_relative_market_sensitivity_frame(timeseries_records: list[dict[str, Any]]) -> pd.DataFrame:
    weekly_returns = build_stable_weekly_value_return_frame(timeseries_records)
    if weekly_returns.empty:
        return pd.DataFrame(columns=["date", "ratio"])
    frame = build_rolling_relative_market_sensitivity_frame(weekly_returns)
    return trim_to_start_date(frame, "date", RISK_CHART_START_DATE, fallback_to_full=False)


def plot_recent_volatility_comparison(timeseries_records: list[dict[str, Any]]) -> go.Figure:
    frame = rolling_relative_volatility_frame(timeseries_records)
    if frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Recent volatility evidence is unavailable for the last 18 months.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15, "color": "#e2e8f0"},
        )
        fig.update_layout(
            title="Recent 6-Month Rolling Volatility vs S&P 500",
            height=520,
            margin={"l": 24, "r": 24, "t": 60, "b": 24},
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.55)",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    x_values = frame["date"].dt.strftime("%Y-%m-%d").tolist()
    portfolio_values = (frame["portfolio_volatility"] * 100).tolist()
    benchmark_values = (frame["benchmark_volatility"] * 100).tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=portfolio_values,
            mode="lines",
            name="Portfolio",
            line={"width": 3, "color": "#3B82F6"},
            hovertemplate=(
                "%{x}<br>Portfolio volatility: %{y:.1f}%"
                "<br>What this means: how much the portfolio has been swinging around"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=benchmark_values,
            mode="lines",
            name="S&P 500",
            line={"width": 3, "color": "#F59E0B"},
            hovertemplate=(
                "%{x}<br>S&P 500 volatility: %{y:.1f}%"
                "<br>What this means: the market's own recent swing level"
                "<extra></extra>"
            ),
        )
    )
    fig.add_annotation(
        x=x_values[-1],
        y=float(portfolio_values[-1]),
        text=f"Portfolio: {portfolio_values[-1]:.1f}%",
        showarrow=True,
        arrowhead=2,
        ax=42,
        ay=-28,
        bgcolor="rgba(15,23,42,0.92)",
        bordercolor="#3B82F6",
        font={"color": "#f8fafc", "size": 11},
    )
    fig.add_annotation(
        x=x_values[-1],
        y=float(benchmark_values[-1]),
        text=f"S&P 500: {benchmark_values[-1]:.1f}%",
        showarrow=True,
        arrowhead=2,
        ax=42,
        ay=28,
        bgcolor="rgba(15,23,42,0.92)",
        bordercolor="#F59E0B",
        font={"color": "#f8fafc", "size": 11},
    )
    fig.update_layout(
        title="Volatility vs S&P 500",
        height=520,
        margin={"l": 40, "r": 24, "t": 60, "b": 36},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.55)",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
        hoverlabel=dark_hoverlabel(),
    )
    fig.update_xaxes(title_text="Date", gridcolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(
        title_text="Volatility (%)",
        gridcolor="rgba(148,163,184,0.18)",
        range=padded_axis_range(
            portfolio_values + benchmark_values,
            min_padding=1.5,
        ),
    )
    return fig


def plot_risk_evidence(market_metrics: dict[str, Any], portfolio_summary: dict[str, Any]) -> go.Figure:
    risk = market_metrics["risk_score"]
    scores = risk["component_scores"]
    raw_values = risk["component_raw_values"]
    evidence_specs: list[dict[str, Any]] = []

    concentration_keys = {
        "concentration::single_position_weight",
        "concentration::top_5_weight",
        "concentration::effective_holdings",
    }
    if any(scores.get(key, 0.0) >= VERY_HIGH_CONCERN_SCORE for key in concentration_keys):
        evidence_specs.append({"kind": "concentration", "title": "Concentration Evidence"})
    if scores.get("behavior::turnover", 0.0) >= VERY_HIGH_CONCERN_SCORE:
        evidence_specs.append({"kind": "turnover", "title": "Trading churn"})
    if scores.get("behavior::short_holding_period", 0.0) >= VERY_HIGH_CONCERN_SCORE:
        evidence_specs.append({"kind": "holding_period", "title": "How long your dollars stay invested"})
    if scores.get("market::relative_volatility_to_benchmark", 0.0) >= VERY_HIGH_CONCERN_SCORE:
        evidence_specs.append({"kind": "relative_volatility", "title": "Volatility vs S&P 500"})
    if scores.get("market::relative_drawdown_to_benchmark", 0.0) >= VERY_HIGH_CONCERN_SCORE:
        evidence_specs.append({"kind": "relative_drawdown", "title": "Downside depth vs S&P 500"})
    if scores.get("market::relative_downside_capture_to_benchmark", 0.0) >= VERY_HIGH_CONCERN_SCORE:
        evidence_specs.append({"kind": "relative_downside_capture", "title": "Bad-day behavior vs S&P 500"})
    if scores.get("market::relative_market_sensitivity_to_benchmark", 0.0) >= VERY_HIGH_CONCERN_SCORE:
        evidence_specs.append({"kind": "relative_market_sensitivity", "title": "Market sensitivity vs S&P 500"})
    if scores.get("market::equity_exposure", 0.0) >= VERY_HIGH_CONCERN_SCORE:
        evidence_specs.append({"kind": "equity_exposure", "title": "How fully invested you are"})

    if not evidence_specs:
        fig = go.Figure()
        fig.add_annotation(
            text="No metric is currently in the highest-scoring range, so no extra proof charts are needed right now.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15, "color": "#e2e8f0"},
        )
        fig.update_layout(
            title="Evidence Behind Top Risk Signals",
            height=220,
            margin={"l": 24, "r": 24, "t": 56, "b": 24},
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.55)",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    fig = make_subplots(
        rows=len(evidence_specs),
        cols=1,
        subplot_titles=[item["title"] for item in evidence_specs],
        vertical_spacing=0.08,
    )

    for row_idx, item in enumerate(evidence_specs, start=1):
        kind = item["kind"]
        if kind == "concentration":
            weights_df = pd.DataFrame(market_metrics["open_positions"])
            weights_df = weights_df.sort_values("current_weight", ascending=True).tail(8)
            weight_values = (weights_df["current_weight"].fillna(0.0) * 100).tolist()
            largest_weight = float(raw_values.get("concentration::single_position_weight") or 0.0) * 100
            largest_score = float(scores.get("concentration::single_position_weight", 0.0))
            top_5_weight = float(raw_values.get("concentration::top_5_weight") or 0.0) * 100
            top_5_score = float(scores.get("concentration::top_5_weight", 0.0))
            effective_holdings = float(raw_values.get("concentration::effective_holdings") or 0.0)
            effective_holdings_score = float(scores.get("concentration::effective_holdings", 0.0))
            fig.add_trace(
                go.Bar(
                    x=weight_values,
                    y=weights_df["ticker"].tolist(),
                    orientation="h",
                    marker={"color": "#3B82F6"},
                    name="Weight",
                    text=[f"{value:.1f}%" for value in weight_values],
                    textposition="outside",
                    hovertemplate=(
                        "%{y}: %{x:.1f}% of portfolio"
                        "<br>What this means: how much one holding can influence the account"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
            if not weights_df.empty:
                fig.add_annotation(
                    x=weight_values[-1],
                    y=weights_df["ticker"].iloc[-1],
                    text=f"Score {largest_score:.1f}/100 at {largest_weight:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    ax=70,
                    ay=-28,
                    bgcolor="rgba(15,23,42,0.92)",
                    bordercolor="#3B82F6",
                    font={"color": "#f8fafc", "size": 11},
                    row=row_idx,
                    col=1,
                )
            summary_text = (
                f"Largest position: {largest_weight:.1f}% | score {largest_score:.1f}/100"
                "<br>"
                f"Top 5 holdings: {top_5_weight:.1f}% | score {top_5_score:.1f}/100"
                "<br>"
                f"Effective holdings: {effective_holdings:.1f} | score {effective_holdings_score:.1f}/100"
            )
            fig.add_annotation(
                x=1.02,
                y=0.85,
                xref=f"x domain{'' if row_idx == 1 else row_idx}",
                yref=f"y domain{'' if row_idx == 1 else row_idx}",
                text=summary_text,
                showarrow=False,
                align="left",
                bgcolor="rgba(15,23,42,0.92)",
                bordercolor="#475569",
                borderwidth=1,
                font={"color": "#e2e8f0", "size": 12},
                row=row_idx,
                col=1,
            )
            fig.update_xaxes(title_text="Current Weight (%)", row=row_idx, col=1)
            fig.update_xaxes(
                range=padded_axis_range(weight_values, min_padding=2.0),
                row=row_idx,
                col=1,
            )
        elif kind == "turnover":
            actual_turnover = float(raw_values.get("behavior::turnover") or 0.0)
            fig.add_trace(
                go.Bar(
                    x=[actual_turnover],
                    y=["Annualized turnover"],
                    orientation="h",
                    marker={"color": "#EF4444"},
                    hovertemplate=(
                        "Turnover: %{x:.2f}"
                        "<br>What this means: how much of the portfolio you are rotating through trades"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="#F59E0B", row=row_idx, col=1)
            fig.add_annotation(
                x=0.5,
                y=1,
                xref=f"x{'' if row_idx == 1 else row_idx}",
                yref=f"paper",
                text="High-risk line",
                showarrow=False,
                font={"size": 11, "color": "#F59E0B"},
            )
            fig.update_xaxes(title_text="Turnover Ratio", row=row_idx, col=1)
            fig.update_xaxes(range=padded_axis_range([actual_turnover], baseline_values=[0.5], min_padding=0.08), row=row_idx, col=1)
        elif kind == "holding_period":
            holding_days = float(raw_values.get("behavior::short_holding_period") or 0.0)
            fig.add_trace(
                go.Bar(
                    x=[holding_days],
                    y=["Capital-weighted holding period"],
                    orientation="h",
                    marker={"color": "#8B5CF6"},
                    hovertemplate=(
                        "Holding days: %{x:.0f}"
                        "<br>What this means: how long your invested dollars usually stay in positions"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
            fig.add_vline(x=SHORT_HOLD_TARGET_DAYS, line_dash="dash", line_color="#10B981", row=row_idx, col=1)
            fig.add_annotation(
                x=SHORT_HOLD_TARGET_DAYS,
                y=1,
                xref=f"x{'' if row_idx == 1 else row_idx}",
                yref=f"paper",
                text="18-month target",
                showarrow=False,
                font={"size": 11, "color": "#10B981"},
            )
            fig.update_xaxes(title_text="Days Held", row=row_idx, col=1)
            fig.update_xaxes(range=padded_axis_range([holding_days], baseline_values=[SHORT_HOLD_TARGET_DAYS], min_padding=25.0), row=row_idx, col=1)
        elif kind == "relative_volatility":
            frame = rolling_relative_volatility_frame(market_metrics["timeseries"])
            x_values = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d").tolist()
            y_values = frame["ratio"].tolist()
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    line={"width": 3, "color": "#3B82F6"},
                    hovertemplate=(
                        "%{x}<br>Volatility ratio: %{y:.2f}x"
                        "<br>What this means: how much rougher the ride has been than the S&P 500"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="#10B981", row=row_idx, col=1)
            fig.add_hline(y=2.0, line_dash="dot", line_color="#F59E0B", row=row_idx, col=1)
            if x_values:
                add_latest_annotation(fig, x_values[-1], float(y_values[-1]), f"Latest: {y_values[-1]:.2f}x", "#3B82F6", row=row_idx, col=1)
                fig.add_annotation(
                    x=x_values[0],
                    y=1.0,
                    text="Same as S&P",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="rgba(15,23,42,0.88)",
                    bordercolor="#10B981",
                    font={"size": 11, "color": "#f8fafc"},
                    row=row_idx,
                    col=1,
                )
                fig.add_annotation(
                    x=x_values[0],
                    y=2.0,
                    text="High-risk line",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="rgba(15,23,42,0.88)",
                    bordercolor="#F59E0B",
                    font={"size": 11, "color": "#f8fafc"},
                    row=row_idx,
                    col=1,
                )
            fig.update_yaxes(title_text="x Benchmark", row=row_idx, col=1)
            fig.update_yaxes(range=padded_axis_range(y_values, baseline_values=[1.0, 2.0], min_padding=0.2), row=row_idx, col=1)
        elif kind == "relative_drawdown":
            frame = rolling_relative_drawdown_frame(market_metrics["timeseries"])
            x_values = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d").tolist()
            y_values = frame["ratio"].tolist()
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    line={"width": 3, "color": "#EF4444"},
                    hovertemplate=(
                        "%{x}<br>Downside depth ratio: %{y:.2f}x"
                        "<br>What this means: how much deeper bad stretches have been than the S&P 500"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="#10B981", row=row_idx, col=1)
            fig.add_hline(y=2.0, line_dash="dot", line_color="#F59E0B", row=row_idx, col=1)
            if x_values:
                add_latest_annotation(fig, x_values[-1], float(y_values[-1]), f"Latest: {y_values[-1]:.2f}x", "#EF4444", row=row_idx, col=1)
            fig.update_yaxes(title_text="x Benchmark", row=row_idx, col=1)
            fig.update_yaxes(range=padded_axis_range(y_values, baseline_values=[1.0, 2.0], min_padding=0.2), row=row_idx, col=1)
        elif kind == "relative_downside_capture":
            frame = rolling_relative_downside_capture_frame(market_metrics["timeseries"])
            x_values = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d").tolist()
            y_values = frame["ratio"].tolist()
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    line={"width": 3, "color": "#F97316"},
                    hovertemplate=(
                        "%{x}<br>Bad-day behavior ratio: %{y:.2f}x"
                        "<br>What this means: whether the portfolio loses more than the S&P 500 on down days"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="#10B981", row=row_idx, col=1)
            fig.add_hline(y=RELATIVE_DOWNSIDE_CAPTURE_MAX_RATIO, line_dash="dot", line_color="#F59E0B", row=row_idx, col=1)
            if x_values:
                add_latest_annotation(fig, x_values[-1], float(y_values[-1]), f"Latest: {y_values[-1]:.2f}x", "#F97316", row=row_idx, col=1)
            fig.update_yaxes(title_text="x Benchmark", row=row_idx, col=1)
            fig.update_yaxes(
                range=padded_axis_range(y_values, baseline_values=[1.0, RELATIVE_DOWNSIDE_CAPTURE_MAX_RATIO], min_padding=0.08),
                row=row_idx,
                col=1,
            )
        elif kind == "relative_market_sensitivity":
            frame = rolling_relative_market_sensitivity_frame(market_metrics["timeseries"])
            x_values = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d").tolist()
            y_values = frame["ratio"].tolist()
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    line={"width": 3, "color": "#A855F7"},
                    hovertemplate=(
                        "%{x}<br>Market sensitivity ratio: %{y:.2f}x"
                        "<br>What this means: how strongly the portfolio reacts when the market moves"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="#10B981", row=row_idx, col=1)
            fig.add_hline(
                y=RELATIVE_MARKET_SENSITIVITY_MAX_RATIO,
                line_dash="dot",
                line_color="#F59E0B",
                row=row_idx,
                col=1,
            )
            if x_values:
                add_latest_annotation(fig, x_values[-1], float(y_values[-1]), f"Latest: {y_values[-1]:.2f}x", "#A855F7", row=row_idx, col=1)
            fig.update_yaxes(title_text="x Benchmark", row=row_idx, col=1)
            fig.update_yaxes(
                range=padded_axis_range(
                    y_values,
                    baseline_values=[1.0, RELATIVE_MARKET_SENSITIVITY_MAX_RATIO],
                    min_padding=0.08,
                ),
                row=row_idx,
                col=1,
            )
        elif kind == "equity_exposure":
            headline = market_metrics["headline_metrics"]
            invested_value = float(headline.get("current_portfolio_value") or 0.0)
            cash_value = float(headline.get("uninvested_cash_estimate") or 0.0)
            fig.add_trace(
                go.Bar(
                    x=[invested_value],
                    y=["Account mix"],
                    orientation="h",
                    marker={"color": "#3B82F6"},
                    name="Invested",
                    hovertemplate="Invested: $%{x:,.2f}<extra></extra>",
                    showlegend=(row_idx == 1),
                ),
                row=row_idx,
                col=1,
            )
            fig.add_trace(
                go.Bar(
                    x=[cash_value],
                    y=["Account mix"],
                    orientation="h",
                    marker={"color": "#10B981"},
                    name="Cash",
                    hovertemplate="Cash: $%{x:,.2f}<extra></extra>",
                    showlegend=(row_idx == 1),
                ),
                row=row_idx,
                col=1,
            )
            fig.update_xaxes(title_text="Dollar Value", row=row_idx, col=1)
            fig.update_xaxes(range=padded_axis_range([invested_value, cash_value], baseline_values=[invested_value + cash_value], min_padding=5000.0), row=row_idx, col=1)

    fig.update_layout(
        title="Recent Evidence Behind Top Risk Signals",
        height=max(520, 420 * len(evidence_specs)),
        margin={"l": 40, "r": 24, "t": 72, "b": 24},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.55)",
        hovermode="x unified",
        barmode="stack",
        hoverlabel=dark_hoverlabel(),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.18)")
    return fig


def build_overview_markdown(portfolio_summary: dict[str, Any], market_metrics: dict[str, Any]) -> str:
    headline = market_metrics["headline_metrics"]
    risk = market_metrics["risk_score"]
    behavior = portfolio_summary["behavioral_metrics"]
    return f"""
### Time Horizon
- Analysis window: `{headline["analysis_start"]}` to `{headline["analysis_end"]}` ({headline["analysis_years"]:.2f} years)
- Distinct symbols traded: `{portfolio_summary["transaction_counts"]["distinct_symbols_traded"]}`
- Buys vs sells: `{portfolio_summary["transaction_counts"]["buys"]}` buys, `{portfolio_summary["transaction_counts"]["sells"]}` sells

### Portfolio Health
- Observed risk score: `{risk["score"]}/100` (`{risk["band"]}`)
- Alignment to stated risk: `{risk["alignment"]}` with alignment score `{risk["alignment_score"]}`
- Confidence in observed risk read: `{risk["confidence_band"]}` (`{risk["confidence_score"]}`)
- Capital-weighted median holding period: `{behavior["capital_weighted_median_holding_period_days"]}` days
- Annualized turnover proxy: `{behavior["annualized_turnover_proxy"]}`

### Current Snapshot
- Invested portfolio value: `{money_text(headline["current_portfolio_value"])}`
- Uninvested cash estimate: `{money_text(headline["uninvested_cash_estimate"])}`
- Total account value estimate: `{money_text(headline["total_account_value_estimate"])}`
- Realized P&L: `{money_text(headline["total_realized_pnl"])}`
- Unrealized P&L: `{money_text(headline["total_unrealized_pnl"])}`
"""


def build_benchmark_markdown(market_metrics: dict[str, Any]) -> str:
    headline = market_metrics["headline_metrics"]
    return f"""
### Time-Horizon Benchmark Comparison
- Benchmark: `{headline["benchmark_symbol"]}` using `{headline["benchmark_method"]}`
- Invested-only return over your investing horizon: `{pct_text(headline["invested_only_return_estimate"])}`
- Trade-matched S&P 500 return over the same horizon: `{pct_text(headline["benchmark_return_estimate"])}`
- Excess return vs S&P 500: `{pct_text(headline["excess_return_vs_benchmark"])}`

### Annualized View
- Invested-only CAGR: `{pct_text(headline["invested_only_cagr"])}`
- S&P 500 CAGR on the same trade-matched flows: `{pct_text(headline["benchmark_cagr"])}`
- Invested-only money-weighted return: `{pct_text(headline["invested_only_money_weighted_return"])}`
- Benchmark money-weighted return: `{pct_text(headline["benchmark_money_weighted_return"])}`
- Excess money-weighted return: `{pct_text(headline["excess_money_weighted_return_vs_benchmark"])}`

### Risk-Relative Metrics
- Annualized volatility: `{pct_text(headline["annualized_volatility"])}`
- S&P 500 volatility: `{pct_text(headline["benchmark_annualized_volatility"])}`
- Relative market sensitivity to S&P 500: `{headline["relative_market_sensitivity_to_benchmark"]}`
- Tracking error: `{pct_text(headline["tracking_error"])}`
- Information ratio proxy: `{headline["information_ratio_proxy"]}`
"""


def build_sector_overview_markdown(market_metrics: dict[str, Any]) -> str:
    best_sector = market_metrics.get("best_sector_vs_benchmark")
    sector_rows = market_metrics.get("sector_allocation", [])
    if not sector_rows:
        return "### Sector View\nSector allocation is unavailable for the current portfolio."
    if not best_sector:
        return "### Sector View\nSector allocation is available below, but benchmark-relative sector performance could not be estimated yet."
    return (
        "### Sector View\n"
        f"- Best current sector vs trade-matched S&P 500: `{best_sector['sector']}`\n"
        f"- Weighted excess return vs benchmark: `{pct_text(best_sector['excess_return_vs_benchmark'])}`\n"
        f"- Current portfolio weight: `{pct_text(best_sector['weight_pct'])}`\n"
    )


def generate_fallback_insight(portfolio_summary: dict[str, Any], market_metrics: dict[str, Any]) -> str:
    headline = market_metrics["headline_metrics"]
    risk = market_metrics["risk_score"]
    top_drivers = market_metrics["performance_attribution"][:3]
    winners = ", ".join(item["ticker"] for item in top_drivers if item.get("combined_pnl", 0) > 0) or "none yet"
    laggards = ", ".join(
        item["ticker"] for item in market_metrics["performance_attribution"][-3:] if item.get("combined_pnl", 0) < 0
    ) or "none yet"
    return f"""
## Investor Profile
Observed portfolio risk is **{risk["score"]}/100 ({risk["band"]})**, while the stated risk was **{risk["stated_score"]}/100**. Current alignment reads as **{risk["alignment"]}** with **{risk["confidence_band"]}** confidence.

## Portfolio Snapshot
Across **{headline["analysis_years"]:.2f} years**, the account built an estimated **{money_text(headline["current_portfolio_value"])}** invested portfolio plus **{money_text(headline["uninvested_cash_estimate"])}** in idle cash. Unrealized P&L is **{money_text(headline["total_unrealized_pnl"])}** and realized P&L is **{money_text(headline["total_realized_pnl"])}**.

## Performance vs S&P 500
Over the same investing horizon, invested-only return is **{pct_text(headline["invested_only_return_estimate"])}** versus **{pct_text(headline["benchmark_return_estimate"])}** for the trade-matched S&P 500 benchmark. Money-weighted excess return is **{pct_text(headline["excess_money_weighted_return_vs_benchmark"])}**.

## Which Holdings Drove Returns
Top positive contributors so far: **{winners}**. Main laggards: **{laggards}**.

## Important Caveats
This is a holdings-and-cash-flow based analysis. It does not infer taxes, outside assets, options exposure, or future fundamentals. The benchmark comparison excludes idle cash on purpose so it measures deployed capital versus a fully invested market path.
"""


def build_analysis_payload(portfolio_summary: dict[str, Any], market_metrics: dict[str, Any]) -> dict[str, Any]:
    return {"portfolio_summary": portfolio_summary, "market_metrics": market_metrics}


def resolve_ollama_analysis(
    payload: dict[str, Any],
    risk_profile: int,
    model_name: str,
    use_ollama: bool,
) -> str:
    if not use_ollama:
        return generate_fallback_insight(payload["portfolio_summary"], payload["market_metrics"])
    if not ollama_available(model_name):
        return (
            f"Ollama model `{model_name}` is not available locally. "
            "Run `ollama serve` and `ollama pull <model>` or disable the AI summary toggle.\n\n"
            + generate_fallback_insight(payload["portfolio_summary"], payload["market_metrics"])
        )
    messages = build_messages(payload, risk_profile)
    try:
        return call_ollama(model=model_name, messages=messages)
    except Exception as exc:
        return f"Could not generate Ollama summary: {exc}\n\n" + generate_fallback_insight(
            payload["portfolio_summary"], payload["market_metrics"]
        )


def run_analysis(file_obj: Any, risk_profile: int, dataset_source: str) -> tuple[Any, ...]:
    started_at = time.perf_counter()
    if dataset_source == "Use bundled fake dataset":
        csv_path = REPO_ROOT / "data" / "raw" / "fake_mantis_invest.csv"
    else:
        if file_obj is None:
            raise gr.Error("Upload a Robinhood CSV export first or switch to the bundled fake dataset.")
        csv_path = Path(file_obj.name)

    transactions = load_transactions(csv_path)
    if transactions.empty:
        raise gr.Error("No valid transaction rows were found in the selected file.")

    lot_data = build_lot_analytics(transactions)
    portfolio_summary = summarize_portfolio(transactions, lot_data)
    traded_symbols = sorted(set(transactions.loc[transactions["Instrument"].ne(""), "Instrument"].tolist()))
    market_data = fetch_market_data(
        traded_symbols=traded_symbols,
        benchmark_symbol=BENCHMARK_SYMBOL,
        start_date=portfolio_summary["date_range"]["start"],
    )
    market_metrics = build_market_enriched_metrics(
        df=transactions,
        portfolio_summary=portfolio_summary,
        lot_data=lot_data,
        market_data=market_data,
        benchmark_symbol=BENCHMARK_SYMBOL,
        projection_years=PROJECTION_YEARS,
        stated_risk_score=risk_profile,
    )
    diagnosis_paths = save_diagnosis_artifacts(
        csv_path=csv_path,
        dataset_source=dataset_source,
        risk_profile=risk_profile,
        transactions=transactions,
        portfolio_summary=portfolio_summary,
        market_metrics=market_metrics,
    )
    diagnosis = portfolio_risk_diagnosis_from_saved_artifacts(Path(diagnosis_paths["latest_dir"]))
    tables = format_display_tables(market_metrics)
    sector_overview_md = build_sector_overview_markdown(market_metrics)
    headline = market_metrics["headline_metrics"]
    risk = market_metrics["risk_score"]
    risk_guide_html = build_risk_guide_html(risk)
    metric_card_values = build_metric_card_values(risk)
    diagnosis_summary_html = build_diagnosis_summary_html(diagnosis)
    diagnosis_concern_fig = plot_diagnosis_concerns(diagnosis)
    diagnosis_driver_html = build_diagnosis_driver_html(diagnosis)
    diagnosis_supporting_metrics_df = build_supporting_metrics_frame(diagnosis)
    diagnosis_holding_fundamentals_df = build_holding_fundamentals_frame(diagnosis)
    diagnosis_narrative_evidence_df = build_narrative_evidence_frame(diagnosis)
    diagnosis_macro_html = build_macro_context_html(diagnosis)
    diagnosis_coverage_df = build_data_coverage_frame(diagnosis)
    diagnosis_confidence_html = build_confidence_completeness_html(diagnosis)
    diagnosis_chart_payload, diagnosis_stock_choices = build_diagnosis_risk_chart_payload(
        diagnosis,
        market_metrics,
        market_data,
    )
    risk_actions_html = build_risk_actions_html(diagnosis)
    risk_actions_df = build_risk_actions_frame(diagnosis)
    portfolio_gaps_html = build_portfolio_gaps_html(diagnosis)
    portfolio_gap_fig = plot_portfolio_gaps(diagnosis)
    portfolio_gaps_df = build_portfolio_gap_frame(diagnosis)
    portfolio_preferences_df = build_portfolio_preferences_frame(diagnosis)
    eq_fig = plot_equity_curves(market_metrics["timeseries"])
    sector_fig = plot_sector_allocation(market_metrics.get("sector_allocation", []))
    dd_fig = plot_drawdowns(market_metrics["timeseries"])
    recent_volatility_fig = plot_recent_volatility_comparison(market_metrics["timeseries"])
    market_sensitivity_fig = plot_diagnosis_market_sensitivity_lens(market_metrics["timeseries"], "Portfolio")
    risk_evidence_fig = plot_risk_evidence(market_metrics, portfolio_summary)
    elapsed_seconds = time.perf_counter() - started_at

    summary_cards_inner = (
        metric_card("Analysis Window", f'{headline["analysis_years"]:.2f}y', f'{headline["analysis_start"]} to {headline["analysis_end"]}')
        + metric_card("Invested Value", money_text(headline["current_portfolio_value"]), "Current invested sleeve")
        + metric_card("Total Portfolio Value", money_text(headline["total_account_value_estimate"]), "Holdings plus cash estimate")
        + metric_card("Cash in Hand", money_text(headline["uninvested_cash_estimate"]), "Estimated uninvested cash")
        + metric_card("Realized P&L", money_text(headline["total_realized_pnl"]), "Closed positions and cash events")
        + metric_card("Unrealized P&L", money_text(headline["total_unrealized_pnl"]), "Open positions only")
        + metric_card("Observed Risk", f'{risk["score"]}/100', f'{risk["band"]} | {risk["alignment"]}')
        + metric_card("Vs S&P 500", pct_text(headline["excess_money_weighted_return_vs_benchmark"]), "Excess money-weighted return")
        + metric_card("Prep Time", f"{elapsed_seconds:.2f}s", "Upload to dashboard-ready")
    )
    summary_cards = f"<div class='metric-strip'>{summary_cards_inner}</div>"

    risk_md = build_risk_explainer_html(risk)

    return (
        summary_cards,
        sector_overview_md,
        risk_md,
        risk_guide_html,
        risk,
        tables["holdings"],
        tables["attribution"],
        tables["volatility_drivers"],
        eq_fig,
        sector_fig,
        dd_fig,
        recent_volatility_fig,
        risk_evidence_fig,
        diagnosis_summary_html,
        diagnosis_concern_fig,
        diagnosis_driver_html,
        diagnosis_chart_payload,
        gr.update(choices=diagnosis_stock_choices, value="Portfolio"),
        risk_actions_html,
        risk_actions_df,
        portfolio_gaps_html,
        portfolio_gap_fig,
        portfolio_gaps_df,
        portfolio_preferences_df,
        diagnosis_supporting_metrics_df,
        diagnosis_holding_fundamentals_df,
        diagnosis_narrative_evidence_df,
        recent_volatility_fig,
        diagnosis_macro_html,
        diagnosis_coverage_df,
        diagnosis_confidence_html,
        sector_fig,
        dd_fig,
        market_sensitivity_fig,
        risk_evidence_fig,
        *metric_card_values,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Portfolio Analyzer Dashboard",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css="""
        .gradio-container {
            background:
                radial-gradient(circle at top left, rgba(59,130,246,.12), transparent 28%),
                radial-gradient(circle at top right, rgba(14,165,233,.10), transparent 26%),
                linear-gradient(180deg, #0b1220 0%, #111827 100%);
            max-width: 100vw !important;
            padding-left: 18px !important;
            padding-right: 18px !important;
        }
        .app-shell {width: 100%; max-width: 100% !important; margin: 0; align-items: start;}
        .control-rail {
            padding: 18px;
            border: 1px solid rgba(148,163,184,.16);
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(15,23,42,.92), rgba(15,23,42,.78));
            box-shadow: 0 20px 48px rgba(2,6,23,.24);
            backdrop-filter: blur(14px);
        }
        .content-rail {
            width: 100%;
            padding: 8px 0 0;
        }
        .metric-strip {display:grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 14px; width: 100%; align-items: stretch;}
        .metric-card {
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 44px rgba(2,6,23,.32);
            border-color: rgba(96,165,250,.28) !important;
        }
        button[role="tab"] {
            border-radius: 14px 14px 0 0 !important;
            padding: 10px 16px !important;
            font-weight: 600 !important;
            letter-spacing: .01em;
        }
        button[role="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, rgba(59,130,246,.18), rgba(59,130,246,.08)) !important;
            color: #f8fafc !important;
            border-color: rgba(96,165,250,.28) !important;
        }
        .control-rail .gr-button-primary {
            min-height: 48px !important;
            border-radius: 14px !important;
            box-shadow: 0 12px 30px rgba(37,99,235,.28);
        }
        .control-rail .gr-box,
        .control-rail .gr-form,
        .control-rail .gradio-file,
        .control-rail .gradio-slider,
        .control-rail .gradio-radio,
        .control-rail .gradio-markdown {
            border-radius: 14px !important;
        }
        .risk-guide-link button {
            min-height: 28px !important;
            padding: 4px 10px !important;
            border-radius: 999px !important;
            font-size: 11px !important;
            line-height: 1.1 !important;
            width: auto !important;
            min-width: 0 !important;
        }
        .summary-cards-wrap {margin-bottom: 8px;}
        .risk-guide-link button {
            opacity: .94;
        }
        @media (max-width: 1200px) {
            .metric-strip {grid-template-columns: repeat(2, minmax(220px, 1fr));}
            .control-rail {padding: 16px;}
        }
        @media (max-width: 820px) {
            .metric-strip {grid-template-columns: 1fr;}
            .control-rail {padding: 14px;}
            .gradio-container {
                padding-left: 10px !important;
                padding-right: 10px !important;
            }
        }
        """,
    ) as demo:
        gr.Markdown(
            """
            # Portfolio Analyzer Dashboard
            Upload your Robinhood CSV, compare your portfolio to the S&P 500 over your actual investing horizon,
            and inspect observed portfolio risk in a more explainable way.
            """
        )

        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(scale=1, min_width=280, elem_classes=["control-rail"]):
                upload = gr.File(label="Robinhood CSV", file_types=[".csv"])
                dataset_source = gr.Radio(
                    choices=["Upload my CSV", "Use bundled fake dataset"],
                    value="Upload my CSV",
                    label="Dataset Source",
                    info="Use the fake dataset if you want to explore the dashboard without a Robinhood export.",
                )
                risk_profile = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=60,
                    step=1,
                    label="Stated Risk Profile (0-100)",
                )
                analyze_btn = gr.Button("Run Analysis", variant="primary")
                gr.Markdown(
                    """
                    **Notes**
                    - Benchmark uses `^GSPC`
                    - Benchmark comparison excludes idle cash
                    - Timeframe stats are shown using your actual investing horizon
                    - A bundled fake dataset is available for demo purposes
                    """
                )

            with gr.Column(scale=3, elem_classes=["content-rail"]):
                risk_state = gr.State(value=None)
                diagnosis_chart_state = gr.State(value={})
                cards = gr.HTML(label="Summary Cards", elem_classes=["summary-cards-wrap"])
                with gr.Tabs(selected="overview") as main_tabs:
                    with gr.Tab("Overview", id="overview"):
                        sector_overview_md = gr.Markdown()
                        equity_plot = gr.Plot(label="Benchmark")
                        sector_plot = gr.Plot(label="Sector Allocation")
                    with gr.Tab("Risk", id="risk"):
                        risk_md = gr.HTML()
                        metric_card_components: list[gr.HTML] = []
                        metric_buttons: list[gr.Button] = []
                        for group_name, metric_keys, _subtitle in metric_group_order():
                            gr.Markdown(f"**{group_name}**")
                            with gr.Row():
                                for metric_key in metric_keys:
                                    with gr.Column():
                                        card = gr.HTML()
                                        button = gr.Button(
                                            "Show in Risk Guide",
                                            variant="secondary",
                                            size="sm",
                                            elem_classes=["risk-guide-link"],
                                        )
                                        metric_card_components.append(card)
                                        metric_buttons.append(button)
                        volatility_drivers_df = gr.Dataframe(label="Top Drivers of 2025 Volatility", interactive=False)
                        recent_volatility_plot = gr.Plot(label="Volatility vs S&P 500")
                        risk_evidence_plot = gr.Plot(label="Evidence Behind Top Risk Signals")
                        drawdown_plot = gr.Plot(label="Downside Depth vs S&P 500")
                    with gr.Tab("Risk Diagnosis", id="risk-diagnosis"):
                        diagnosis_driver_md = gr.HTML()
                        with gr.Row(equal_height=True):
                            diagnosis_concern_plot = gr.Plot(label="Top Risk Categories")
                            diagnosis_sector_plot = gr.Plot(label="Sector Crowding View")
                        diagnosis_stock_filter = gr.Dropdown(
                            label="Risk Lens",
                            choices=["Portfolio"],
                            value="Portfolio",
                            interactive=True,
                            info="View these risk charts for the full portfolio or a single stock. Stocks are ordered by diagnosis risk.",
                        )
                        with gr.Row(equal_height=True):
                            diagnosis_recent_volatility_plot = gr.Plot(label="Volatility vs S&P 500")
                            diagnosis_market_sensitivity_plot = gr.Plot(label="Market Sensitivity vs S&P 500")
                        with gr.Row(equal_height=True):
                            diagnosis_drawdown_plot = gr.Plot(label="Downside Depth vs S&P 500")
                        diagnosis_macro_md = gr.HTML()
                        with gr.Accordion("Detailed Evidence", open=False):
                            diagnosis_risk_evidence_plot = gr.Plot(label="Evidence Behind Top Risk Signals")
                            diagnosis_supporting_metrics_df = gr.Dataframe(
                                label="Supporting Evidence",
                                interactive=False,
                                wrap=True,
                            )
                            diagnosis_holding_fundamentals_df = gr.Dataframe(
                                label="Holding Fundamentals",
                                interactive=False,
                                wrap=True,
                            )
                            diagnosis_narrative_evidence_df = gr.Dataframe(
                                label="Filing and News Evidence",
                                interactive=False,
                                wrap=True,
                            )
                            diagnosis_coverage_df = gr.Dataframe(
                                label="Source Coverage",
                                interactive=False,
                            )
                            diagnosis_confidence_md = gr.HTML()
                        with gr.Accordion("Diagnosis Snapshot and Alignment", open=False):
                            diagnosis_summary_md = gr.HTML()
                    with gr.Tab("Risk Actions", id="risk-actions"):
                        risk_actions_md = gr.HTML()
                        risk_actions_df = gr.Dataframe(
                            label="Recommendation Detail",
                            interactive=False,
                            wrap=True,
                        )
                    with gr.Tab("Portfolio Gaps", id="portfolio-gaps"):
                        portfolio_gaps_md = gr.HTML()
                        portfolio_gap_plot = gr.Plot(label="What The Portfolio Still Needs Most")
                        portfolio_gaps_df = gr.Dataframe(
                            label="Gap Quick Read",
                            interactive=False,
                            wrap=True,
                        )
                        with gr.Accordion("Preferences and Constraints", open=False):
                            portfolio_preferences_df = gr.Dataframe(
                                label="Preferences and Constraints",
                                interactive=False,
                                wrap=True,
                            )
                    with gr.Tab("Risk Guide", id="risk-guide"):
                        risk_guide_md = gr.HTML()
                    with gr.Tab("Holdings", id="holdings"):
                        holdings_df = gr.Dataframe(label="Open Holdings", interactive=False)
                        attribution_df = gr.Dataframe(label="Performance Attribution", interactive=False)

        analyze_btn.click(
            fn=run_analysis,
            inputs=[upload, risk_profile, dataset_source],
            outputs=[
                cards,
                sector_overview_md,
                risk_md,
                risk_guide_md,
                risk_state,
                holdings_df,
                attribution_df,
                volatility_drivers_df,
                equity_plot,
                sector_plot,
                drawdown_plot,
                recent_volatility_plot,
                risk_evidence_plot,
                diagnosis_summary_md,
                diagnosis_concern_plot,
                diagnosis_driver_md,
                diagnosis_chart_state,
                diagnosis_stock_filter,
                risk_actions_md,
                risk_actions_df,
                portfolio_gaps_md,
                portfolio_gap_plot,
                portfolio_gaps_df,
                portfolio_preferences_df,
                diagnosis_supporting_metrics_df,
                diagnosis_holding_fundamentals_df,
                diagnosis_narrative_evidence_df,
                diagnosis_recent_volatility_plot,
                diagnosis_macro_md,
                diagnosis_coverage_df,
                diagnosis_confidence_md,
                diagnosis_sector_plot,
                diagnosis_drawdown_plot,
                diagnosis_market_sensitivity_plot,
                diagnosis_risk_evidence_plot,
                *metric_card_components,
            ],
        )

        diagnosis_stock_filter.change(
            fn=update_diagnosis_stock_plots,
            inputs=[diagnosis_stock_filter, diagnosis_chart_state],
            outputs=[
                diagnosis_recent_volatility_plot,
                diagnosis_drawdown_plot,
                diagnosis_market_sensitivity_plot,
            ],
        )

        for metric_key, button in zip(metric_navigation_order(), metric_buttons):
            button.click(
                fn=lambda risk, key=metric_key: open_metric_guide(key, risk),
                inputs=[risk_state],
                outputs=[main_tabs, risk_guide_md],
            )

    return demo


def launch_app() -> None:
    app = build_app()
    # Default to local launch so the dashboard reliably binds to 7861 without
    # depending on Gradio's share tunnel port allocation.
    app.launch(server_name="127.0.0.1", server_port=7862, share=True)


if __name__ == "__main__":
    launch_app()

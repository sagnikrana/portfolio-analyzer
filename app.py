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
from pathlib import Path
from typing import Any
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf


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


def fetch_market_data(traded_symbols: list[str], benchmark_symbol: str, start_date: str) -> dict[str, Any]:
    tickers = sorted(set(symbol for symbol in traded_symbols if symbol))
    yahoo_map = {symbol: normalize_ticker_for_yahoo(symbol) for symbol in tickers}
    yahoo_tickers = [yahoo_map[symbol] for symbol in tickers]
    history_start = (pd.Timestamp(start_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

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
    else:
        ticker_close = pd.DataFrame()

    benchmark_raw = yf.download(
        tickers=benchmark_symbol,
        start=history_start,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    benchmark_close = extract_close_frame(benchmark_raw, [benchmark_symbol]).iloc[:, 0].dropna()
    benchmark_close.name = benchmark_symbol

    latest_prices = ticker_close.ffill().iloc[-1].dropna().to_dict() if not ticker_close.empty else {}
    latest_benchmark_price = float(benchmark_close.iloc[-1]) if not benchmark_close.empty else None

    return {
        "ticker_close": ticker_close,
        "latest_prices": latest_prices,
        "benchmark_close": benchmark_close,
        "latest_benchmark_price": latest_benchmark_price,
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


def build_daily_share_matrix(df: pd.DataFrame, market_index: pd.Index, tickers: list[str]) -> pd.DataFrame:
    share_credit_codes = {"REC"}
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
    # Fill non-trade days with zero share delta before the cumulative sum. Otherwise pandas
    # leaves NaNs for untouched tickers on existing rows, which can make holdings appear to
    # disappear between trade dates and corrupt every downstream time-series KPI.
    share_changes = share_changes.reindex(market_index).fillna(0.0)
    share_matrix = share_changes.cumsum().ffill().fillna(0.0)
    return share_matrix.reindex(columns=tickers, fill_value=0.0).fillna(0.0)


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
        "relative_market_sensitivity_to_benchmark": clip01(
            (relative_market_sensitivity_to_benchmark or 0.0) / 1.25
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
    # Use weekly flow-adjusted returns for volatility so the score reflects broad portfolio
    # behavior rather than noisy day-level artifacts from sparse prices or tiny sleeves.
    portfolio_volatility_series = aligned_performance["portfolio"].resample("W-FRI").last()
    benchmark_volatility_series = aligned_performance["benchmark"].resample("W-FRI").last()
    volatility_aligned_returns = pd.concat(
        [
            portfolio_volatility_series.pct_change(fill_method=None).rename("portfolio"),
            benchmark_volatility_series.pct_change(fill_method=None).rename("benchmark"),
        ],
        axis=1,
    ).replace([math.inf, -math.inf], float("nan")).dropna()
    aligned_returns = performance_returns
    portfolio_drawdown = (aligned_performance["portfolio"] / aligned_performance["portfolio"].cummax()) - 1
    benchmark_drawdown = (aligned_performance["benchmark"] / aligned_performance["benchmark"].cummax()) - 1
    # Blend overlapping 6-month drawdowns with recency weights so old crises matter less
    # than the portfolio's more recent downside behavior.
    drawdown_profile = recency_weighted_rolling_drawdown(aligned_performance["portfolio"])
    benchmark_drawdown_profile = recency_weighted_rolling_drawdown(aligned_performance["benchmark"])
    annualized_volatility = (
        float(volatility_aligned_returns["portfolio"].std() * math.sqrt(52))
        if len(volatility_aligned_returns) > 2
        else None
    )
    benchmark_annualized_volatility = (
        float(volatility_aligned_returns["benchmark"].std() * math.sqrt(52))
        if len(volatility_aligned_returns) > 2
        else None
    )
    relative_volatility_to_benchmark = (
        float(annualized_volatility / benchmark_annualized_volatility)
        if annualized_volatility is not None
        and benchmark_annualized_volatility is not None
        and benchmark_annualized_volatility > 0
        else None
    )
    relative_drawdown_to_benchmark = (
        float(drawdown_profile["blended_drawdown"] / benchmark_drawdown_profile["blended_drawdown"])
        if drawdown_profile["blended_drawdown"] is not None
        and benchmark_drawdown_profile["blended_drawdown"] is not None
        and benchmark_drawdown_profile["blended_drawdown"] > 0
        else None
    )
    downside_capture_profile = recency_weighted_rolling_downside_capture(aligned_returns)
    relative_downside_capture_to_benchmark = downside_capture_profile["weighted_downside_capture"]
    market_sensitivity_profile = recency_weighted_rolling_market_sensitivity(aligned_returns)
    relative_market_sensitivity_to_benchmark = market_sensitivity_profile["weighted_market_sensitivity"]
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
            "portfolio_drawdown": portfolio_drawdown.reindex(portfolio_equity_series.index).values,
            "benchmark_drawdown": benchmark_drawdown.reindex(portfolio_equity_series.index).values,
        }
    )
    return {
        "headline_metrics": headline_metrics,
        "risk_score": risk_score,
        "open_positions": open_positions_df.sort_values("current_value", ascending=False).round(4).to_dict(orient="records"),
        "performance_attribution": attribution_rows[:15],
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
        "<div class='metric-card' style='padding:16px 18px;border:1px solid rgba(148,163,184,.22);"
        "border-radius:16px;background:linear-gradient(180deg, rgba(30,41,59,.96), rgba(15,23,42,.96));"
        "min-height:120px;display:flex;flex-direction:column;justify-content:space-between;overflow:hidden;"
        "box-shadow:0 8px 24px rgba(2,6,23,.22)'>"
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
    risk_components = pd.DataFrame(
        [
            {"metric": key, "score": value}
            for key, value in market_metrics["risk_score"]["component_scores"].items()
        ]
    ).sort_values("score", ascending=False)
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
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True, gridcolor="rgba(148,163,184,0.18)")
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.18)")
    return fig


def plot_drawdowns(timeseries_records: list[dict[str, Any]]) -> go.Figure:
    series = pd.DataFrame(timeseries_records)
    fig = go.Figure()
    if not series.empty:
        series["date"] = pd.to_datetime(series["date"])
        x_values = series["date"].dt.strftime("%Y-%m-%d").tolist()
        portfolio_drawdown = (pd.to_numeric(series["portfolio_drawdown"], errors="coerce").fillna(0.0) * 100).tolist()
        benchmark_drawdown = (pd.to_numeric(series["benchmark_drawdown"], errors="coerce").fillna(0.0) * 100).tolist()
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
    fig.update_layout(
        title="Drawdown Through the Same Time Horizon",
        height=400,
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
    )
    fig.update_yaxes(ticksuffix="%", gridcolor="rgba(148,163,184,0.18)")
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.18)")
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
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True, gridcolor="rgba(148,163,184,0.18)")
    fig.update_xaxes(dtick=1, gridcolor="rgba(148,163,184,0.18)")
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


def run_analysis(file_obj: Any, risk_profile: int, model_name: str, use_ollama: bool) -> tuple[Any, ...]:
    started_at = time.perf_counter()
    if file_obj is None:
        raise gr.Error("Upload a Robinhood CSV export first.")

    csv_path = Path(file_obj.name)
    transactions = load_transactions(csv_path)
    if transactions.empty:
        raise gr.Error("No valid transaction rows were found in the uploaded file.")

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
    payload = build_analysis_payload(portfolio_summary, market_metrics)
    tables = format_display_tables(market_metrics)
    overview_md = build_overview_markdown(portfolio_summary, market_metrics)
    benchmark_md = build_benchmark_markdown(market_metrics)
    insight_md = resolve_ollama_analysis(payload, risk_profile, model_name, use_ollama)
    headline = market_metrics["headline_metrics"]
    risk = market_metrics["risk_score"]
    eq_fig = plot_equity_curves(market_metrics["timeseries"])
    dd_fig = plot_drawdowns(market_metrics["timeseries"])
    proj_fig = plot_projection(tables["projection"])
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

    risk_md = f"""
### Observed Risk Framework
- Stated risk score: `{risk["stated_score"]}/100` (`{risk["stated_band"]}`)
- Observed portfolio risk: `{risk["score"]}/100` (`{risk["band"]}`)
- Difference vs stated: `{risk["difference_vs_stated"]}`
- Alignment: `{risk["alignment"]}`
- Confidence: `{risk["confidence_band"]}` (`{risk["confidence_score"]}`)

### Dimension Scores
- Concentration risk: `{risk["dimension_scores"]["concentration_risk"]}`
- Market risk: `{risk["dimension_scores"]["market_risk"]}`
- Behavioral risk: `{risk["dimension_scores"]["behavioral_risk"]}`
"""

    return (
        summary_cards,
        overview_md,
        benchmark_md,
        risk_md,
        insight_md,
        tables["holdings"],
        tables["attribution"],
        tables["sold"],
        tables["selection_alpha"],
        tables["risk_components"],
        tables["projection"],
        eq_fig,
        dd_fig,
        proj_fig,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Portfolio Analyzer Dashboard",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css="""
        .app-shell {max-width: 1400px; margin: 0 auto;}
        .metric-strip {display:grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; width: 100%; align-items: stretch;}
        @media (max-width: 1200px) {.metric-strip {grid-template-columns: repeat(2, minmax(220px, 1fr));}}
        @media (max-width: 820px) {.metric-strip {grid-template-columns: 1fr;}}
        """,
    ) as demo:
        gr.Markdown(
            """
            # Portfolio Analyzer Dashboard
            Upload your Robinhood CSV, compare your portfolio to the S&P 500 over your actual investing horizon,
            inspect observed portfolio risk, and generate an AI summary from Ollama if you want one.
            """
        )

        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(scale=1, min_width=280):
                upload = gr.File(label="Robinhood CSV", file_types=[".csv"])
                risk_profile = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=60,
                    step=1,
                    label="Stated Risk Profile (0-100)",
                )
                use_ollama = gr.Checkbox(label="Generate AI summary with Ollama", value=True)
                model_name = gr.Textbox(label="Ollama Model", value=DEFAULT_MODEL_NAME)
                analyze_btn = gr.Button("Run Analysis", variant="primary")
                gr.Markdown(
                    """
                    **Notes**
                    - Benchmark uses `^GSPC`
                    - Benchmark comparison excludes idle cash
                    - Timeframe stats are shown using your actual investing horizon
                    """
                )

            with gr.Column(scale=3):
                cards = gr.HTML(label="Summary Cards")
                with gr.Tabs():
                    with gr.Tab("Overview"):
                        overview_md = gr.Markdown()
                        equity_plot = gr.Plot(label="Portfolio vs S&P 500")
                    with gr.Tab("Holdings"):
                        holdings_df = gr.Dataframe(label="Open Holdings", interactive=False)
                        attribution_df = gr.Dataframe(label="Performance Attribution", interactive=False)
                    with gr.Tab("Risk"):
                        risk_md = gr.Markdown()
                        risk_components_df = gr.Dataframe(label="Risk Component Scores", interactive=False)
                        drawdown_plot = gr.Plot(label="Drawdown Comparison")
                    with gr.Tab("Benchmark"):
                        benchmark_md = gr.Markdown()
                        selection_alpha_df = gr.Dataframe(label="Ticker Alpha vs Benchmark", interactive=False)
                        sold_df = gr.Dataframe(label="Potential Sold-Too-Early Signals", interactive=False)
                    with gr.Tab("Projection"):
                        projection_df = gr.Dataframe(label="18-Year Projection Table", interactive=False)
                        projection_plot = gr.Plot(label="Projection Chart")
                    with gr.Tab("AI Insights"):
                        insight_md = gr.Markdown()

        analyze_btn.click(
            fn=run_analysis,
            inputs=[upload, risk_profile, model_name, use_ollama],
            outputs=[
                cards,
                overview_md,
                benchmark_md,
                risk_md,
                insight_md,
                holdings_df,
                attribution_df,
                sold_df,
                selection_alpha_df,
                risk_components_df,
                projection_df,
                equity_plot,
                drawdown_plot,
                projection_plot,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)

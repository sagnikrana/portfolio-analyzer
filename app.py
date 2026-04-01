from __future__ import annotations

import json
import math
import re
import subprocess
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


DEFAULT_MODEL_NAME = "llama3.1:latest"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
BENCHMARK_SYMBOL = "^GSPC"
PROJECTION_YEARS = 18


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
    weighted_ordinal = sum(
        date.toordinal() * weight for date, weight in zip(dates, weights, strict=False)
    ) / sum(weights)
    return datetime.fromordinal(int(round(weighted_ordinal))).date().isoformat()


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

    current_positions.sort(key=lambda item: item["cost_basis_total"], reverse=True)
    return {
        "current_positions": current_positions,
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

    holding_periods = [lot["hold_days"] for lot in lot_data["closed_lots"]]
    holding_counter = Counter(holding_period_bucket(days) for days in holding_periods)
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


def fetch_market_data(traded_symbols: list[str], benchmark_symbol: str, start_date: str) -> dict[str, Any]:
    tickers = sorted(set(symbol for symbol in traded_symbols if symbol))
    history_start = (pd.Timestamp(start_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    ticker_history_raw = yf.download(
        tickers=tickers,
        start=history_start,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    ticker_close = extract_close_frame(ticker_history_raw, tickers)

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
    share_changes = share_changes.reindex(market_index, fill_value=0.0)
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
    annualized_turnover: float | None,
    median_holding_days: float | None,
    equity_exposure: float | None,
    max_drawdown: float | None,
    downside_capture_ratio: float | None,
    annualized_volatility: float | None,
    beta_to_benchmark: float | None,
    years_of_history: float,
    closed_lot_count: int,
) -> dict[str, Any]:
    concentration_components = {
        "single_position_weight": clip01((max_position_weight or 0.0) / 0.22),
        "top_5_weight": clip01((top_5_weight or 0.0) / 0.65),
        "effective_holdings": clip01((10 - (effective_holdings or 10)) / 8),
    }
    market_components = {
        "volatility": clip01((annualized_volatility or 0.0) / 0.32),
        "drawdown": clip01(abs(max_drawdown or 0.0) / 0.40),
        "downside_capture": clip01((downside_capture_ratio or 0.0) / 1.15),
        "beta": clip01((beta_to_benchmark or 0.0) / 1.25),
        "equity_exposure": clip01((equity_exposure or 0.0) / 1.0),
    }
    behavior_components = {
        "turnover": clip01((annualized_turnover or 0.0) / 0.50),
        "short_holding_period": clip01((540 - (median_holding_days or 540)) / 540),
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
        "dimension_scores": {k: round(v * 100, 1) for k, v in dimension_scores.items()},
        "component_scores": {
            **{f"concentration::{k}": round(v * 100, 1) for k, v in concentration_components.items()},
            **{f"market::{k}": round(v * 100, 1) for k, v in market_components.items()},
            **{f"behavior::{k}": round(v * 100, 1) for k, v in behavior_components.items()},
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
    share_matrix = build_daily_share_matrix(df, market_index, tracked_tickers)
    cash_balance_series = build_cash_balance_series(df, market_index)
    portfolio_equity_series = (share_matrix * price_matrix).sum(axis=1)
    account_value_series = portfolio_equity_series + cash_balance_series

    active_mask = portfolio_equity_series.ne(0) | benchmark_value_series.ne(0)
    if active_mask.any():
        first_active = active_mask[active_mask].index[0]
        portfolio_equity_series = portfolio_equity_series.loc[first_active:]
        account_value_series = account_value_series.loc[first_active:]
        benchmark_value_series = benchmark_value_series.loc[first_active:]

    portfolio_returns = portfolio_equity_series.pct_change().replace([math.inf, -math.inf], pd.NA).fillna(0.0)
    benchmark_returns = benchmark_value_series.pct_change().replace([math.inf, -math.inf], pd.NA).fillna(0.0)
    aligned_returns = pd.concat(
        [portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")], axis=1
    ).dropna()
    portfolio_drawdown = (portfolio_equity_series / portfolio_equity_series.cummax()) - 1
    benchmark_drawdown = (benchmark_value_series / benchmark_value_series.cummax()) - 1
    annualized_volatility = float(portfolio_returns.std() * math.sqrt(252)) if len(portfolio_returns) > 2 else None
    benchmark_annualized_volatility = (
        float(benchmark_returns.std() * math.sqrt(252)) if len(benchmark_returns) > 2 else None
    )
    beta_to_benchmark = None
    tracking_error = None
    info_ratio_proxy = None
    if len(aligned_returns) > 5 and aligned_returns["benchmark"].var() > 0:
        beta_to_benchmark = float(
            aligned_returns["portfolio"].cov(aligned_returns["benchmark"])
            / aligned_returns["benchmark"].var()
        )
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
    downside_mask = benchmark_returns < 0
    downside_capture_ratio = None
    if downside_mask.any() and benchmark_returns[downside_mask].mean() != 0:
        downside_capture_ratio = float(
            portfolio_returns[downside_mask].mean() / benchmark_returns[downside_mask].mean()
        )

    years = float(portfolio_summary["date_range"]["years"])
    portfolio_cagr = annualized_return(total_account_value_estimate, net_deposits, years)
    invested_only_cagr = annualized_return(current_portfolio_value, invested_net_cash_estimate, years)
    benchmark_cagr = annualized_return(benchmark_current_value, benchmark_invested_net_cash, years)
    median_holding_days = portfolio_summary["behavioral_metrics"].get("median_holding_period_days")
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
        annualized_turnover=annualized_turnover,
        median_holding_days=median_holding_days,
        equity_exposure=equity_exposure,
        max_drawdown=float(portfolio_drawdown.min()) if not portfolio_drawdown.empty else None,
        downside_capture_ratio=downside_capture_ratio,
        annualized_volatility=annualized_volatility,
        beta_to_benchmark=beta_to_benchmark,
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
        "beta_to_benchmark": round(beta_to_benchmark, 4) if beta_to_benchmark is not None else None,
        "tracking_error": round(tracking_error, 4) if tracking_error is not None else None,
        "information_ratio_proxy": round(info_ratio_proxy, 4) if info_ratio_proxy is not None else None,
        "concentration_hhi": round(hhi, 4),
        "max_position_weight": round(max_position_weight, 4),
        "effective_holdings": round(effective_holdings, 2) if effective_holdings is not None else None,
        "top_5_weight": round(top_5_weight, 4),
        "concentration_adjusted_return_proxy": round(concentration_adjusted_return_proxy, 4)
        if concentration_adjusted_return_proxy is not None
        else None,
        "max_portfolio_drawdown": round(float(portfolio_drawdown.min()), 4) if not portfolio_drawdown.empty else None,
        "max_benchmark_drawdown": round(float(benchmark_drawdown.min()), 4) if not benchmark_drawdown.empty else None,
        "portfolio_drawdown_on_worst_benchmark_day": round(portfolio_drawdown_on_worst_benchmark_day, 4)
        if portfolio_drawdown_on_worst_benchmark_day is not None
        else None,
        "downside_capture_ratio": round(downside_capture_ratio, 4) if downside_capture_ratio is not None else None,
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
        f"{json.dumps(analysis_payload, indent=2)}\n\n"
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


def metric_card(label: str, value: str, subtitle: str = "") -> str:
    subtitle_md = f"<div style='color:#5f6c7b;font-size:12px'>{subtitle}</div>" if subtitle else ""
    return (
        "<div style='padding:14px 16px;border:1px solid #dde4ee;border-radius:14px;"
        "background:#f8fbff;height:100%'>"
        f"<div style='font-size:12px;color:#4f5d6b;text-transform:uppercase;letter-spacing:.08em'>{label}</div>"
        f"<div style='font-size:28px;font-weight:700;margin-top:6px'>{value}</div>"
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
    attribution = dataframe_from_records(
        market_metrics["performance_attribution"],
        ["ticker", "realized_pnl", "unrealized_pnl", "combined_pnl", "pnl_contribution_pct"],
    )
    sold = dataframe_from_records(
        market_metrics["sold_too_early"],
        ["ticker", "missed_upside", "realized_pnl", "if_held_pnl"],
    )
    selection_alpha = dataframe_from_records(market_metrics["selection_alpha"], ["ticker", "alpha_pnl"])
    risk_components = pd.DataFrame(
        [
            {"metric": key, "score": value}
            for key, value in market_metrics["risk_score"]["component_scores"].items()
        ]
    ).sort_values("score", ascending=False)
    projection = dataframe_from_records(
        market_metrics["projection_scenarios_no_new_contributions"]["table"],
        ["year", "conservative", "base", "optimistic"],
    )
    return {
        "holdings": holdings.round(4),
        "attribution": attribution.round(2),
        "sold": sold.round(2),
        "selection_alpha": selection_alpha.round(2),
        "risk_components": risk_components.round(1),
        "projection": projection.round(2),
    }


def plot_equity_curves(timeseries_records: list[dict[str, Any]]) -> plt.Figure:
    series = pd.DataFrame(timeseries_records)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if not series.empty:
        series["date"] = pd.to_datetime(series["date"])
        ax.plot(series["date"], series["portfolio_invested_value"], label="Invested Portfolio", linewidth=2.2)
        ax.plot(series["date"], series["benchmark_value"], label="Trade-Matched S&P 500", linewidth=2.2)
    ax.set_title("Portfolio vs S&P 500 Over Your Investing Horizon")
    ax.set_ylabel("Value ($)")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    return fig


def plot_drawdowns(timeseries_records: list[dict[str, Any]]) -> plt.Figure:
    series = pd.DataFrame(timeseries_records)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if not series.empty:
        series["date"] = pd.to_datetime(series["date"])
        ax.plot(series["date"], series["portfolio_drawdown"] * 100, label="Portfolio Drawdown", linewidth=2.0)
        ax.plot(series["date"], series["benchmark_drawdown"] * 100, label="S&P 500 Drawdown", linewidth=2.0)
    ax.set_title("Drawdown Through the Same Time Horizon")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    return fig


def plot_projection(projection_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if not projection_df.empty:
        ax.plot(projection_df["year"], projection_df["conservative"], label="Conservative", linewidth=2.0)
        ax.plot(projection_df["year"], projection_df["base"], label="Base", linewidth=2.0)
        ax.plot(projection_df["year"], projection_df["optimistic"], label="Optimistic", linewidth=2.0)
    ax.set_title("18-Year Projection Without New Contributions")
    ax.set_xlabel("Years From Today")
    ax.set_ylabel("Projected Value ($)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
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
- Median closed-lot holding period: `{behavior["median_holding_period_days"]}` days
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
- Beta to S&P 500: `{headline["beta_to_benchmark"]}`
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

    summary_cards = (
        metric_card("Analysis Window", f'{headline["analysis_years"]:.2f}y', f'{headline["analysis_start"]} to {headline["analysis_end"]}')
        + metric_card("Invested Value", money_text(headline["current_portfolio_value"]), "Current invested sleeve")
        + metric_card("Unrealized P&L", money_text(headline["total_unrealized_pnl"]), "Open positions only")
        + metric_card("Observed Risk", f'{risk["score"]}/100', f'{risk["band"]} | {risk["alignment"]}')
        + metric_card("Vs S&P 500", pct_text(headline["excess_money_weighted_return_vs_benchmark"]), "Excess money-weighted return")
    )

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
        .metric-strip {display:grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px;}
        @media (max-width: 1100px) {.metric-strip {grid-template-columns: repeat(2, minmax(0, 1fr));}}
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
                cards = gr.HTML(label="Summary Cards", elem_classes=["metric-strip"])
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

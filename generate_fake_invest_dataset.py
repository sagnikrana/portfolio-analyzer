from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
DEFAULT_OUTPUT = RAW_DIR / "fake_mantis_invest.csv"


@dataclass(frozen=True)
class Security:
    ticker: str
    name: str
    cusip: str
    start_price: float
    dividend_yield: float = 0.0
    split_date: date | None = None
    split_ratio: float | None = None


SECURITIES: list[Security] = [
    Security("AAPL", "Apple", "037833100", 118.0, dividend_yield=0.0045, split_date=date(2020, 8, 31), split_ratio=4.0),
    Security("MSFT", "Microsoft", "594918104", 205.0, dividend_yield=0.0065),
    Security("NVDA", "NVIDIA", "67066G104", 135.0, dividend_yield=0.001),
    Security("GOOGL", "Alphabet Class A", "02079K305", 72.0, split_date=date(2022, 7, 18), split_ratio=20.0),
    Security("AMZN", "Amazon", "023135106", 95.0),
    Security("META", "Meta Platforms", "30303M102", 155.0),
    Security("TSLA", "Tesla", "88160R101", 185.0),
    Security("V", "Visa", "92826C839", 190.0, dividend_yield=0.006),
    Security("BRK.B", "Berkshire Hathaway Class B", "084670702", 250.0),
    Security("AVGO", "Broadcom", "11135F101", 480.0, dividend_yield=0.012),
    Security("QQQ", "Invesco QQQ Trust", "46090E103", 290.0),
    Security("VOO", "Vanguard S&P 500 ETF", "922908363", 330.0, dividend_yield=0.012),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a fake Robinhood-style investment CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write the fake CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation")
    return parser.parse_args()


def format_date(value: date) -> str:
    return f"{value.month}/{value.day}/{str(value.year)[-2:]}"


def format_money(value: float | None, negative_parens: bool = False) -> str:
    if value is None:
        return ""
    if negative_parens and value < 0:
        return f"(${abs(value):,.2f})"
    return f"${value:,.2f}"


def format_quantity(value: float | None) -> str:
    if value is None:
        return ""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def price_on_day(security: Security, current_day: date, anchor: date) -> float:
    day_index = (current_day - anchor).days
    trend = 0.00045 + (hash(security.ticker) % 7) * 0.00006
    seasonal = 0.035 * __import__("math").sin(day_index / 47.0)
    drifted = security.start_price * (1 + trend) ** max(day_index, 0)
    return max(3.0, drifted * (1 + seasonal))


def build_description(security: Security) -> str:
    return f"{security.name}\\nCUSIP: {security.cusip}"


def make_row(
    *,
    activity_date: date,
    settle_offset: int,
    instrument: str,
    description: str,
    trans_code: str,
    quantity: float | None,
    price: float | None,
    amount: float | None,
    namount: float = 0.0,
) -> dict[str, str]:
    settle_date = activity_date + timedelta(days=settle_offset)
    return {
        "Activity Date": format_date(activity_date),
        "Process Date": format_date(activity_date),
        "Settle Date": format_date(settle_date),
        "Instrument": instrument,
        "Description": description,
        "Trans Code": trans_code,
        "Quantity": format_quantity(quantity),
        "Price": format_money(price),
        "Amount": format_money(amount, negative_parens=True),
        "Namount": format_money(namount),
    }


def generate_dataset(seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    holdings: defaultdict[str, float] = defaultdict(float)
    cash_balance = 0.0
    anchor = date(2020, 8, 10)
    end_date = date(2026, 3, 6)

    def add_cash_event(day: date, description: str, amount: float) -> None:
        nonlocal cash_balance
        cash_balance += amount
        rows.append(
            make_row(
                activity_date=day,
                settle_offset=0,
                instrument="",
                description=description,
                trans_code="ACH" if "ACH" in description else "INT",
                quantity=None,
                price=None,
                amount=amount,
            )
        )

    current_day = anchor
    while current_day <= end_date:
        if current_day.day in {1, 15}:
            deposit = rng.randint(350, 1200)
            add_cash_event(current_day, "ACH Deposit", float(deposit))

        if current_day.day == 28:
            interest = round(max(0.01, cash_balance * 0.00012), 2)
            add_cash_event(current_day, "Interest Payment", interest)

        if current_day.weekday() in {1, 3} and cash_balance > 150:
            security = rng.choice(SECURITIES)
            price = round(price_on_day(security, current_day, anchor), 2)
            budget = min(cash_balance * rng.uniform(0.08, 0.24), rng.uniform(80, 1600))
            quantity = round(max(0.02, budget / price), 6)
            cost = round(quantity * price, 2)
            if cost <= cash_balance:
                cash_balance -= cost
                holdings[security.ticker] += quantity
                rows.append(
                    make_row(
                        activity_date=current_day,
                        settle_offset=2,
                        instrument=security.ticker,
                        description=build_description(security),
                        trans_code="Buy",
                        quantity=quantity,
                        price=price,
                        amount=-cost,
                        namount=cost,
                    )
                )

        if current_day.weekday() == 4 and rng.random() < 0.12:
            sellable = [ticker for ticker, qty in holdings.items() if qty > 0.05]
            if sellable:
                ticker = rng.choice(sellable)
                security = next(sec for sec in SECURITIES if sec.ticker == ticker)
                price = round(price_on_day(security, current_day, anchor) * rng.uniform(0.96, 1.05), 2)
                max_qty = holdings[ticker]
                quantity = round(min(max_qty, max(0.02, max_qty * rng.uniform(0.12, 0.45))), 6)
                proceeds = round(quantity * price, 2)
                holdings[ticker] -= quantity
                cash_balance += proceeds
                rows.append(
                    make_row(
                        activity_date=current_day,
                        settle_offset=2,
                        instrument=ticker,
                        description=build_description(security),
                        trans_code="Sell",
                        quantity=quantity,
                        price=price,
                        amount=proceeds,
                        namount=proceeds,
                    )
                )

        if current_day.day in {12, 27}:
            for security in SECURITIES:
                if security.dividend_yield <= 0 or holdings[security.ticker] <= 0:
                    continue
                if rng.random() < 0.18:
                    annual_div_per_share = security.start_price * security.dividend_yield
                    div_amount = round(holdings[security.ticker] * annual_div_per_share / 4, 2)
                    if div_amount > 0:
                        cash_balance += div_amount
                        rows.append(
                            make_row(
                                activity_date=current_day,
                                settle_offset=0,
                                instrument=security.ticker,
                                description=(
                                    f"Cash Div: R/D {current_day.isoformat()} "
                                    f"P/D {current_day.isoformat()} - {holdings[security.ticker]:.6f} shares at "
                                    f"{round(div_amount / holdings[security.ticker], 4)}"
                                ),
                                trans_code="CDIV",
                                quantity=None,
                                price=None,
                                amount=div_amount,
                            )
                        )

        for security in SECURITIES:
            if security.split_date == current_day and security.split_ratio and holdings[security.ticker] > 0:
                add_qty = round(holdings[security.ticker] * (security.split_ratio - 1), 6)
                holdings[security.ticker] += add_qty
                rows.append(
                    make_row(
                        activity_date=current_day,
                        settle_offset=0,
                        instrument=security.ticker,
                        description=build_description(security),
                        trans_code="SPL",
                        quantity=add_qty,
                        price=None,
                        amount=None,
                    )
                )

        current_day += timedelta(days=1)

    # Add a small gifted share credit so the analyzer sees one REC event like the real file.
    gifted_security = SECURITIES[0]
    holdings[gifted_security.ticker] += 1.0
    rows.append(
        make_row(
            activity_date=anchor,
            settle_offset=0,
            instrument=gifted_security.ticker,
            description=build_description(gifted_security),
            trans_code="REC",
            quantity=1.0,
            price=None,
            amount=None,
        )
    )

    rows.sort(key=lambda row: (row["Activity Date"].split("/")[2], row["Activity Date"], row["Trans Code"], row["Instrument"]))
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Activity Date",
                "Process Date",
                "Settle Date",
                "Instrument",
                "Description",
                "Trans Code",
                "Quantity",
                "Price",
                "Amount",
                "Namount",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def row_sort_key(row: dict[str, str]) -> tuple[date, str, str]:
    month, day, year = row["Activity Date"].split("/")
    parsed = date(2000 + int(year), int(month), int(day))
    return parsed, row["Trans Code"], row["Instrument"]


def main() -> None:
    args = parse_args()
    rows = generate_dataset(seed=args.seed)
    rows.sort(key=row_sort_key)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

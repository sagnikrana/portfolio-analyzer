from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path


RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
DEFAULT_OUTPUT = RAW_DIR / "fake_mantis_invest.csv"

ANCHOR = date(2020, 8, 10)


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
    # Income / defensive names used by the dividend profile.
    Security("JNJ", "Johnson & Johnson", "478160104", 150.0, dividend_yield=0.026),
    Security("KO", "Coca-Cola", "191216100", 48.0, dividend_yield=0.030),
    Security("PG", "Procter & Gamble", "742718109", 138.0, dividend_yield=0.024),
    Security("JPM", "JPMorgan Chase", "46625H100", 100.0, dividend_yield=0.030),
    Security("SCHD", "Schwab US Dividend Equity ETF", "808524797", 56.0, dividend_yield=0.033),
    Security("AMD", "Advanced Micro Devices", "007903107", 85.0),
]

SECURITIES_BY_TICKER: dict[str, Security] = {sec.ticker: sec for sec in SECURITIES}


@dataclass(frozen=True)
class Profile:
    """Knobs that shape one synthetic investor's transaction history."""

    key: str
    output: str  # filename under data/raw
    description: str
    tickers: tuple[str, ...]
    seed: int
    deposit_days: tuple[int, ...] = (1, 15)
    deposit_range: tuple[int, int] = (350, 1200)
    buy_weekdays: tuple[int, ...] = (1, 3)  # Tue / Thu
    buy_budget_frac: tuple[float, float] = (0.08, 0.24)
    buy_abs_range: tuple[float, float] = (80.0, 1600.0)
    min_cash_to_buy: float = 150.0
    sell_weekdays: tuple[int, ...] = (4,)  # Fri
    sell_prob: float = 0.12
    sell_frac: tuple[float, float] = (0.12, 0.45)
    # Optional buy bias toward favored tickers (concentration).
    weights: dict[str, float] = field(default_factory=dict)


PROFILES: list[Profile] = [
    Profile(
        key="balanced",
        output="fake_mantis_invest.csv",
        description="Balanced blend across megacaps + ETFs, moderate trading.",
        tickers=("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "V", "BRK.B", "AVGO", "QQQ", "VOO"),
        seed=42,
    ),
    Profile(
        key="aggressive",
        output="fake_aggressive_growth.csv",
        description="Tech-heavy, concentrated, large deposits, rarely sells.",
        tickers=("NVDA", "TSLA", "META", "AMZN", "AVGO", "AAPL", "GOOGL", "AMD"),
        seed=101,
        deposit_days=(1, 15),
        deposit_range=(800, 2500),
        buy_weekdays=(0, 2, 4),
        buy_budget_frac=(0.18, 0.45),
        buy_abs_range=(300.0, 3500.0),
        sell_prob=0.04,
        sell_frac=(0.08, 0.20),
        weights={"NVDA": 3.0, "TSLA": 2.5, "META": 2.0, "AMD": 1.5},
    ),
    Profile(
        key="dividend",
        output="fake_dividend_income.csv",
        description="Dividend / defensive names, steady contributions, low turnover.",
        tickers=("JNJ", "KO", "PG", "JPM", "SCHD", "V", "MSFT", "BRK.B", "AVGO", "VOO"),
        seed=202,
        deposit_days=(1, 15),
        deposit_range=(400, 900),
        buy_weekdays=(1, 3),
        buy_budget_frac=(0.10, 0.25),
        buy_abs_range=(100.0, 1200.0),
        sell_prob=0.03,
        sell_frac=(0.08, 0.18),
        weights={"SCHD": 2.0, "JNJ": 1.6, "KO": 1.4, "PG": 1.4, "VOO": 1.6},
    ),
    Profile(
        key="index",
        output="fake_index_buy_and_hold.csv",
        description="Index ETFs only, dollar-cost averaging, essentially never sells.",
        tickers=("VOO", "QQQ", "SCHD"),
        seed=303,
        deposit_days=(1, 15),
        deposit_range=(500, 1500),
        buy_weekdays=(0, 1, 2, 3, 4),  # buys whenever cash is available after a deposit
        buy_budget_frac=(0.85, 1.0),   # deploy nearly all cash (DCA)
        buy_abs_range=(100.0, 2000.0),
        min_cash_to_buy=100.0,
        sell_weekdays=(),
        sell_prob=0.0,
        weights={"VOO": 3.0, "QQQ": 1.5, "SCHD": 1.0},
    ),
    Profile(
        key="active",
        output="fake_active_trader.csv",
        description="High-turnover trader: frequent small buys and sells, lots of churn.",
        tickers=("TSLA", "NVDA", "META", "AMD", "AAPL", "AMZN", "QQQ"),
        seed=404,
        deposit_days=(1, 15),
        deposit_range=(400, 1100),
        buy_weekdays=(0, 1, 2, 3),
        buy_budget_frac=(0.10, 0.30),
        buy_abs_range=(60.0, 900.0),
        min_cash_to_buy=80.0,
        sell_weekdays=(0, 1, 2, 3, 4),
        sell_prob=0.5,
        sell_frac=(0.20, 0.75),
        weights={"TSLA": 2.0, "NVDA": 2.0, "META": 1.5, "AMD": 1.5},
    ),
]

PROFILES_BY_KEY: dict[str, Profile] = {p.key: p for p in PROFILES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fake Robinhood-style investment CSV(s).")
    parser.add_argument(
        "--profile",
        default="all",
        choices=["all", *PROFILES_BY_KEY.keys()],
        help="Which profile to generate (default: all).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Override output path (single profile only).")
    parser.add_argument(
        "--end",
        type=lambda s: date.fromisoformat(s),
        default=date.today(),
        help="End date (YYYY-MM-DD); defaults to today so data runs Aug 2020 -> now.",
    )
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


def _pick_ticker(rng: random.Random, profile: Profile) -> str:
    if profile.weights:
        weights = [profile.weights.get(t, 1.0) for t in profile.tickers]
        return rng.choices(list(profile.tickers), weights=weights, k=1)[0]
    return rng.choice(list(profile.tickers))


def generate_dataset(profile: Profile, end_date: date) -> list[dict[str, str]]:
    rng = random.Random(profile.seed)
    rows: list[dict[str, str]] = []
    holdings: defaultdict[str, float] = defaultdict(float)
    cash_balance = 0.0
    anchor = ANCHOR
    profile_securities = [SECURITIES_BY_TICKER[t] for t in profile.tickers]

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
        if current_day.day in profile.deposit_days:
            deposit = rng.randint(*profile.deposit_range)
            add_cash_event(current_day, "ACH Deposit", float(deposit))

        if current_day.day == 28:
            interest = round(max(0.01, cash_balance * 0.00012), 2)
            add_cash_event(current_day, "Interest Payment", interest)

        if current_day.weekday() in profile.buy_weekdays and cash_balance > profile.min_cash_to_buy:
            security = SECURITIES_BY_TICKER[_pick_ticker(rng, profile)]
            price = round(price_on_day(security, current_day, anchor), 2)
            budget = min(
                cash_balance * rng.uniform(*profile.buy_budget_frac),
                rng.uniform(*profile.buy_abs_range),
            )
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

        if current_day.weekday() in profile.sell_weekdays and rng.random() < profile.sell_prob:
            sellable = [ticker for ticker, qty in holdings.items() if qty > 0.05]
            if sellable:
                ticker = rng.choice(sellable)
                security = SECURITIES_BY_TICKER[ticker]
                price = round(price_on_day(security, current_day, anchor) * rng.uniform(0.96, 1.05), 2)
                max_qty = holdings[ticker]
                quantity = round(min(max_qty, max(0.02, max_qty * rng.uniform(*profile.sell_frac))), 6)
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
            for security in profile_securities:
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

        for security in profile_securities:
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
    gifted_security = profile_securities[0]
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


def build_profile(profile: Profile, end_date: date, output: Path | None = None) -> Path:
    rows = generate_dataset(profile, end_date)
    rows.sort(key=row_sort_key)
    out_path = output or (RAW_DIR / profile.output)
    write_csv(rows, out_path)
    print(f"[{profile.key}] wrote {len(rows)} rows to {out_path}  ({profile.description})")
    return out_path


def main() -> None:
    args = parse_args()
    if args.profile == "all":
        for profile in PROFILES:
            build_profile(profile, args.end)
    else:
        profile = PROFILES_BY_KEY[args.profile]
        build_profile(profile, args.end, output=args.output)


if __name__ == "__main__":
    main()

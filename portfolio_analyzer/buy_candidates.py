from __future__ import annotations

from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
import yfinance as yf
from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STRUCTURED_DIR = DATA_DIR / "processed" / "structured"

BUY_CANDIDATE_UNIVERSE_PATH = RAW_DIR / "buy_candidate_universe.csv"
SP500_CONSTITUENTS_PATH = STRUCTURED_DIR / "sp500_constituents.csv"
NASDAQ100_CONSTITUENTS_PATH = STRUCTURED_DIR / "nasdaq100_constituents.csv"
DOW30_CONSTITUENTS_PATH = STRUCTURED_DIR / "dow30_constituents.csv"
MAJOR_ETF_HOLDINGS_PATH = STRUCTURED_DIR / "major_etf_holdings.csv"

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
DOW30_WIKI_URL = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}

DEFENSIVE_SECTORS = {"Health Care", "Consumer Defensive", "Utilities", "Fixed Income"}
GROWTH_SECTORS = {"Technology", "Communication Services", "Consumer Cyclical"}
INCOME_SECTORS = {"Utilities", "Real Estate", "Financial Services", "Consumer Defensive", "Fixed Income"}

CURATED_SEED_DATA: list[dict[str, Any]] = [
    {
        "ticker": "VOO",
        "market_data_symbol": "VOO",
        "security_name": "Vanguard S&P 500 ETF",
        "asset_type": "ETF",
        "primary_role": "Core benchmark ballast",
        "sector": "Broad Market",
        "style_tilt": "Blend",
        "region": "US",
        "is_core": True,
        "is_defensive": False,
        "is_income": False,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Default broad-market core holding",
        "universe_source": "Curated ETF list",
        "source_count": 1,
    },
    {
        "ticker": "IVV",
        "market_data_symbol": "IVV",
        "security_name": "iShares Core S&P 500 ETF",
        "asset_type": "ETF",
        "primary_role": "Core benchmark ballast",
        "sector": "Broad Market",
        "style_tilt": "Blend",
        "region": "US",
        "is_core": True,
        "is_defensive": False,
        "is_income": False,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Alternative S&P 500 core exposure",
        "universe_source": "Curated ETF list",
        "source_count": 1,
    },
    {
        "ticker": "VTI",
        "market_data_symbol": "VTI",
        "security_name": "Vanguard Total Stock Market ETF",
        "asset_type": "ETF",
        "primary_role": "Core diversification",
        "sector": "Broad Market",
        "style_tilt": "Blend",
        "region": "US",
        "is_core": True,
        "is_defensive": False,
        "is_income": False,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Broader US equity exposure than S&P 500 only",
        "universe_source": "Curated ETF list",
        "source_count": 1,
    },
    {
        "ticker": "SCHD",
        "market_data_symbol": "SCHD",
        "security_name": "Schwab U.S. Dividend Equity ETF",
        "asset_type": "ETF",
        "primary_role": "Income and quality ballast",
        "sector": "Financial Services",
        "style_tilt": "Income",
        "region": "US",
        "is_core": False,
        "is_defensive": True,
        "is_income": True,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Dividend-oriented ETF with quality tilt",
        "universe_source": "Curated ETF list",
        "source_count": 1,
    },
    {
        "ticker": "VIG",
        "market_data_symbol": "VIG",
        "security_name": "Vanguard Dividend Appreciation ETF",
        "asset_type": "ETF",
        "primary_role": "Quality ballast",
        "sector": "Financial Services",
        "style_tilt": "Quality",
        "region": "US",
        "is_core": False,
        "is_defensive": True,
        "is_income": True,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Dividend growth ETF for steadier quality exposure",
        "universe_source": "Curated ETF list",
        "source_count": 1,
    },
    {
        "ticker": "QUAL",
        "market_data_symbol": "QUAL",
        "security_name": "iShares MSCI USA Quality Factor ETF",
        "asset_type": "ETF",
        "primary_role": "Quality ballast",
        "sector": "Broad Market",
        "style_tilt": "Quality",
        "region": "US",
        "is_core": False,
        "is_defensive": True,
        "is_income": False,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Quality factor ETF for stronger balance sheets",
        "universe_source": "Curated ETF list",
        "source_count": 1,
    },
    {
        "ticker": "USMV",
        "market_data_symbol": "USMV",
        "security_name": "iShares MSCI USA Min Vol Factor ETF",
        "asset_type": "ETF",
        "primary_role": "Lower-volatility ballast",
        "sector": "Broad Market",
        "style_tilt": "Low Volatility",
        "region": "US",
        "is_core": False,
        "is_defensive": True,
        "is_income": False,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Designed to reduce portfolio swings",
        "universe_source": "Curated ETF list",
        "source_count": 1,
    },
    {
        "ticker": "XLV",
        "market_data_symbol": "XLV",
        "security_name": "Health Care Select Sector SPDR Fund",
        "asset_type": "ETF",
        "primary_role": "Defensive diversifier",
        "sector": "Health Care",
        "style_tilt": "Defensive",
        "region": "US",
        "is_core": False,
        "is_defensive": True,
        "is_income": False,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Healthcare sector ETF for defensive diversification",
        "universe_source": "Curated ETF list",
        "source_count": 1,
    },
    {
        "ticker": "XLP",
        "market_data_symbol": "XLP",
        "security_name": "Consumer Staples Select Sector SPDR Fund",
        "asset_type": "ETF",
        "primary_role": "Defensive diversifier",
        "sector": "Consumer Defensive",
        "style_tilt": "Defensive",
        "region": "US",
        "is_core": False,
        "is_defensive": True,
        "is_income": True,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Staples ETF for resilience in weaker markets",
        "universe_source": "Curated ETF list",
        "source_count": 1,
    },
    {
        "ticker": "XLU",
        "market_data_symbol": "XLU",
        "security_name": "Utilities Select Sector SPDR Fund",
        "asset_type": "ETF",
        "primary_role": "Defensive diversifier",
        "sector": "Utilities",
        "style_tilt": "Defensive",
        "region": "US",
        "is_core": False,
        "is_defensive": True,
        "is_income": True,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Utilities ETF for lower-beta diversification",
        "universe_source": "Curated ETF list",
        "source_count": 1,
    },
    {
        "ticker": "BND",
        "market_data_symbol": "BND",
        "security_name": "Vanguard Total Bond Market ETF",
        "asset_type": "ETF",
        "primary_role": "Stability sleeve",
        "sector": "Fixed Income",
        "style_tilt": "Defensive",
        "region": "US",
        "is_core": False,
        "is_defensive": True,
        "is_income": True,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Broad bond exposure for steadier ballast",
        "universe_source": "Curated ETF list",
        "source_count": 1,
    },
    {
        "ticker": "VGIT",
        "market_data_symbol": "VGIT",
        "security_name": "Vanguard Intermediate-Term Treasury ETF",
        "asset_type": "ETF",
        "primary_role": "Stability sleeve",
        "sector": "Fixed Income",
        "style_tilt": "Defensive",
        "region": "US",
        "is_core": False,
        "is_defensive": True,
        "is_income": True,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Treasury exposure for a more defensive mix",
        "universe_source": "Curated ETF list",
        "source_count": 1,
    },
    {
        "ticker": "BRK.B",
        "market_data_symbol": "BRK-B",
        "security_name": "Berkshire Hathaway Inc. Class B",
        "asset_type": "Stock",
        "primary_role": "Quality compounder",
        "sector": "Financial Services",
        "style_tilt": "Quality",
        "region": "US",
        "is_core": False,
        "is_defensive": False,
        "is_income": False,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Large diversified quality compounder",
        "universe_source": "Curated stock list",
        "source_count": 1,
    },
    {
        "ticker": "JPM",
        "market_data_symbol": "JPM",
        "security_name": "JPMorgan Chase & Co.",
        "asset_type": "Stock",
        "primary_role": "Financial quality",
        "sector": "Financial Services",
        "style_tilt": "Quality",
        "region": "US",
        "is_core": False,
        "is_defensive": False,
        "is_income": True,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Large-cap financial with broad earnings base",
        "universe_source": "Curated stock list",
        "source_count": 1,
    },
    {
        "ticker": "PG",
        "market_data_symbol": "PG",
        "security_name": "Procter & Gamble Co.",
        "asset_type": "Stock",
        "primary_role": "Defensive compounder",
        "sector": "Consumer Defensive",
        "style_tilt": "Defensive",
        "region": "US",
        "is_core": False,
        "is_defensive": True,
        "is_income": True,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Consumer staples leader with steadier demand",
        "universe_source": "Curated stock list",
        "source_count": 1,
    },
    {
        "ticker": "JNJ",
        "market_data_symbol": "JNJ",
        "security_name": "Johnson & Johnson",
        "asset_type": "Stock",
        "primary_role": "Defensive compounder",
        "sector": "Health Care",
        "style_tilt": "Defensive",
        "region": "US",
        "is_core": False,
        "is_defensive": True,
        "is_income": True,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Diversified healthcare franchise",
        "universe_source": "Curated stock list",
        "source_count": 1,
    },
    {
        "ticker": "WMT",
        "market_data_symbol": "WMT",
        "security_name": "Walmart Inc.",
        "asset_type": "Stock",
        "primary_role": "Defensive compounder",
        "sector": "Consumer Defensive",
        "style_tilt": "Defensive",
        "region": "US",
        "is_core": False,
        "is_defensive": True,
        "is_income": False,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Defensive retail exposure with resilient cash flow",
        "universe_source": "Curated stock list",
        "source_count": 1,
    },
    {
        "ticker": "COST",
        "market_data_symbol": "COST",
        "security_name": "Costco Wholesale Corporation",
        "asset_type": "Stock",
        "primary_role": "Resilient quality",
        "sector": "Consumer Defensive",
        "style_tilt": "Quality",
        "region": "US",
        "is_core": False,
        "is_defensive": True,
        "is_income": False,
        "is_growth": False,
        "eligible_for_buy_engine": True,
        "notes": "Resilient consumer franchise with strong membership model",
        "universe_source": "Curated stock list",
        "source_count": 1,
    },
    {
        "ticker": "MSFT",
        "market_data_symbol": "MSFT",
        "security_name": "Microsoft Corporation",
        "asset_type": "Stock",
        "primary_role": "Quality growth",
        "sector": "Technology",
        "style_tilt": "Quality Growth",
        "region": "US",
        "is_core": False,
        "is_defensive": False,
        "is_income": False,
        "is_growth": True,
        "eligible_for_buy_engine": True,
        "notes": "High-quality large-cap growth if portfolio still needs selective growth",
        "universe_source": "Curated stock list",
        "source_count": 1,
    },
]

BUY_UNIVERSE_BOOL_COLUMNS = [
    "is_core",
    "is_defensive",
    "is_income",
    "is_growth",
    "eligible_for_buy_engine",
]


class BuyCandidateUniverseEntry(BaseModel):
    """One curated security the buy engine is allowed to consider.

    The buy side should not search the whole market blindly. It should reason
    over an explicit, reviewable universe. This object captures one row from
    that saved candidate universe after the raw known universes have been
    merged, normalized, and written to disk.
    """

    ticker: str
    market_data_symbol: str
    security_name: str
    asset_type: str
    primary_role: str
    sector: str
    style_tilt: str
    region: str
    is_core: bool
    is_defensive: bool
    is_income: bool
    is_growth: bool
    eligible_for_buy_engine: bool
    notes: str = ""
    universe_source: str = ""
    source_count: int = 0


def _normalize_ratio(value: Any) -> float | None:
    """Convert raw finance ratios into decimal form when possible."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    if numeric > 1:
        return numeric / 100.0
    return numeric


@lru_cache(maxsize=1024)
def fetch_candidate_market_metadata(market_data_symbol: str) -> dict[str, Any]:
    """Fetch lightweight market metadata for one candidate ticker.

    This helper is intentionally cached because the app may repeatedly render
    the same top ideas while the user tweaks preferences. The data is
    supplemental, not part of the core offline universe build.
    """
    ticker = yf.Ticker(str(market_data_symbol))
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    expense_ratio = (
        _normalize_ratio(info.get("netExpenseRatio"))
        or _normalize_ratio(info.get("annualReportExpenseRatio"))
        or _normalize_ratio(info.get("expenseRatio"))
    )
    dividend_yield = _normalize_ratio(info.get("yield")) or _normalize_ratio(info.get("dividendYield"))
    trailing_pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    try:
        trailing_pe = float(trailing_pe) if trailing_pe is not None else None
    except (TypeError, ValueError):
        trailing_pe = None
    try:
        forward_pe = float(forward_pe) if forward_pe is not None else None
    except (TypeError, ValueError):
        forward_pe = None

    return {
        "full_name": str(info.get("longName") or info.get("shortName") or "").strip(),
        "expense_ratio": expense_ratio,
        "dividend_yield": dividend_yield,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
    }


@lru_cache(maxsize=1024)
def fetch_candidate_news_signals(market_data_symbol: str, limit: int = 2) -> list[str]:
    """Fetch a few recent human-readable market/news signals for one candidate."""
    ticker = yf.Ticker(str(market_data_symbol))
    try:
        items = ticker.news or []
    except Exception:
        return []

    signals: list[str] = []
    for item in items[: max(1, int(limit))]:
        content = item.get("content") or {}
        title = str(content.get("title") or "").strip()
        summary = str(content.get("summary") or content.get("description") or "").strip()
        provider = ((content.get("provider") or {}) or {}).get("displayName")
        if not title:
            continue
        signal = title
        if provider:
            signal += f" ({provider})"
        if summary:
            signal += f" — {summary[:160].strip()}"
        signals.append(signal)
    return signals


def _normalize_display_ticker(symbol: Any) -> str:
    """Convert raw upstream symbols into display tickers.

    Wikipedia and ETF-holdings datasets generally use dots for share classes,
    while Yahoo market-data fetches often prefer dashes. We keep the display
    ticker in the more familiar dotted form and derive the market-data symbol
    separately.
    """
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    return text.replace("-", ".")


def _normalize_market_data_symbol(symbol: Any) -> str:
    """Convert a display ticker into the Yahoo-friendly market-data symbol."""
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    return text.replace(".", "-")


def _read_bool(value: Any) -> bool:
    """Parse loosely typed truthy values from CSV rows."""
    text = str(value).strip().upper()
    return text in {"TRUE", "1", "YES"}


def _coerce_bool_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize standard boolean columns in a buy-universe dataframe."""
    result = frame.copy()
    for column in BUY_UNIVERSE_BOOL_COLUMNS:
        if column in result.columns:
            result[column] = result[column].map(_read_bool) if result[column].dtype == object else result[column].fillna(False).astype(bool)
    return result


def _request_html(url: str) -> str:
    """Download HTML from a source page using an explicit browser-like header."""
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def _read_html_tables(url: str) -> list[pd.DataFrame]:
    """Read all tables from an HTML page after fetching it with a custom header."""
    html = _request_html(url)
    return pd.read_html(StringIO(html))


def _extract_sp500_constituents() -> pd.DataFrame:
    """Fetch and normalize the current S&P 500 constituent table."""
    table = _read_html_tables(SP500_WIKI_URL)[0].copy()
    frame = pd.DataFrame(
        {
            "ticker": table["Symbol"].map(_normalize_display_ticker),
            "market_data_symbol": table["Symbol"].map(_normalize_market_data_symbol),
            "security_name": table["Security"].astype(str).str.strip(),
            "sector": table["GICS Sector"].astype(str).str.strip(),
            "sub_industry": table["GICS Sub-Industry"].astype(str).str.strip(),
            "date_added": table["Date added"].astype(str).str.strip(),
            "universe_source": "S&P 500",
        }
    )
    return frame.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def _extract_nasdaq100_constituents() -> pd.DataFrame:
    """Fetch and normalize the current Nasdaq-100 constituent table."""
    tables = _read_html_tables(NASDAQ100_WIKI_URL)
    table = next(
        candidate
        for candidate in tables
        if {"Ticker", "Company"}.issubset({str(column) for column in candidate.columns})
    ).copy()
    frame = pd.DataFrame(
        {
            "ticker": table["Ticker"].map(_normalize_display_ticker),
            "market_data_symbol": table["Ticker"].map(_normalize_market_data_symbol),
            "security_name": table["Company"].astype(str).str.strip(),
            "sector": table[[column for column in table.columns if "Industry" in str(column)][0]].astype(str).str.strip(),
            "sub_industry": table[[column for column in table.columns if "Subsector" in str(column)][0]].astype(str).str.strip(),
            "universe_source": "Nasdaq-100",
        }
    )
    return frame.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def _extract_dow30_constituents() -> pd.DataFrame:
    """Fetch and normalize the current Dow 30 component table."""
    tables = _read_html_tables(DOW30_WIKI_URL)
    table = next(
        candidate
        for candidate in tables
        if {"Company", "Symbol", "Sector"}.issubset({str(column) for column in candidate.columns})
    ).copy()
    frame = pd.DataFrame(
        {
            "ticker": table["Symbol"].map(_normalize_display_ticker),
            "market_data_symbol": table["Symbol"].map(_normalize_market_data_symbol),
            "security_name": table["Company"].astype(str).str.strip(),
            "sector": table["Sector"].astype(str).str.strip(),
            "date_added": table["Date added"].astype(str).str.strip(),
            "universe_source": "Dow 30",
        }
    )
    return frame.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def curated_buy_seed_entries() -> list[BuyCandidateUniverseEntry]:
    """Return the intentionally small curated seed we started the buy side with."""
    return [BuyCandidateUniverseEntry.model_validate(row) for row in CURATED_SEED_DATA]


def fetch_major_etf_holdings(
    etf_entries: Iterable[BuyCandidateUniverseEntry] | None = None,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Fetch top holdings for the curated ETF list using `yfinance`.

    This is an offline build step, not an app-load operation. The saved output
    tells us which individual stocks show up repeatedly inside the ETFs we are
    already willing to recommend.
    """
    rows: list[dict[str, Any]] = []
    source_entries = list(etf_entries or curated_buy_seed_entries())
    for entry in source_entries:
        if entry.asset_type != "ETF":
            continue
        ticker = yf.Ticker(entry.market_data_symbol)
        try:
            top_holdings = ticker.funds_data.top_holdings
        except Exception:
            continue
        if top_holdings is None or top_holdings.empty:
            continue
        holding_frame = top_holdings.reset_index().head(top_n).copy()
        for row in holding_frame.to_dict(orient="records"):
            holding_ticker = _normalize_display_ticker(row.get("Symbol"))
            if not holding_ticker:
                continue
            rows.append(
                {
                    "etf_ticker": entry.ticker,
                    "etf_name": entry.security_name,
                    "holding_ticker": holding_ticker,
                    "market_data_symbol": _normalize_market_data_symbol(holding_ticker),
                    "holding_name": str(row.get("Name") or "").strip(),
                    "holding_percent": float(row.get("Holding Percent") or 0.0),
                    "universe_source": "Major ETF holdings",
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["etf_ticker", "holding_percent"], ascending=[True, False]).reset_index(drop=True)


def _style_tilt_for_stock(sector: str, source_labels: set[str]) -> str:
    """Infer a simple style tilt for non-ETF candidates."""
    if sector in DEFENSIVE_SECTORS:
        return "Defensive"
    if sector in GROWTH_SECTORS:
        return "Growth"
    if "Dow 30" in source_labels:
        return "Quality"
    return "Blend"


def _primary_role_for_stock(sector: str, source_labels: set[str]) -> str:
    """Infer a first-pass portfolio role for a stock candidate."""
    if sector in {"Health Care", "Consumer Defensive", "Utilities"}:
        return "Defensive large-cap"
    if sector in {"Technology", "Communication Services", "Consumer Cyclical"}:
        return "Quality growth"
    if "Dow 30" in source_labels:
        return "Blue-chip compounder"
    if sector in {"Financials", "Financial Services"}:
        return "Financial quality"
    return "Large-cap diversifier"


def _aggregate_known_universe_stock_rows(
    sp500: pd.DataFrame,
    nasdaq100: pd.DataFrame,
    dow30: pd.DataFrame,
    etf_holdings: pd.DataFrame,
) -> pd.DataFrame:
    """Merge known-universe stock sources into one deduplicated candidate table."""
    stock_sources: list[tuple[str, pd.DataFrame]] = [
        ("S&P 500", sp500),
        ("Nasdaq-100", nasdaq100),
        ("Dow 30", dow30),
    ]
    rows: list[dict[str, Any]] = []
    for source_name, frame in stock_sources:
        if frame.empty:
            continue
        for row in frame.to_dict(orient="records"):
            rows.append(
                {
                    "ticker": row.get("ticker"),
                    "market_data_symbol": row.get("market_data_symbol"),
                    "security_name": row.get("security_name"),
                    "sector": row.get("sector"),
                    "sub_industry": row.get("sub_industry"),
                    "source_label": source_name,
                }
            )
    if not etf_holdings.empty:
        for row in etf_holdings.to_dict(orient="records"):
            rows.append(
                {
                    "ticker": row.get("holding_ticker"),
                    "market_data_symbol": row.get("market_data_symbol"),
                    "security_name": row.get("holding_name"),
                    "sector": "",
                    "sub_industry": "",
                    "source_label": f"ETF holding: {row.get('etf_ticker')}",
                }
            )
    if not rows:
        return pd.DataFrame()

    combined = pd.DataFrame(rows)
    grouped_rows: list[dict[str, Any]] = []
    for ticker, frame in combined.groupby("ticker", dropna=False):
        if not ticker:
            continue
        security_name = next((str(value).strip() for value in frame["security_name"] if str(value).strip()), "")
        market_data_symbol = next((str(value).strip() for value in frame["market_data_symbol"] if str(value).strip()), _normalize_market_data_symbol(ticker))
        sector = next((str(value).strip() for value in frame["sector"] if str(value).strip()), "")
        sub_industry = next((str(value).strip() for value in frame["sub_industry"] if str(value).strip()), "")
        source_labels = {str(value).strip() for value in frame["source_label"] if str(value).strip()}
        grouped_rows.append(
            {
                "ticker": ticker,
                "market_data_symbol": market_data_symbol,
                "security_name": security_name,
                "sector": sector or "Unknown",
                "sub_industry": sub_industry,
                "source_labels": sorted(source_labels),
            }
        )
    return pd.DataFrame(grouped_rows).sort_values("ticker").reset_index(drop=True)


def build_buy_candidate_universe_seed(
    *,
    sp500: pd.DataFrame,
    nasdaq100: pd.DataFrame,
    dow30: pd.DataFrame,
    etf_holdings: pd.DataFrame,
) -> pd.DataFrame:
    """Build the saved raw buy universe from known universes plus the curated seed.

    The resulting CSV is intentionally app-friendly:
    - it contains the same columns the app already knows how to read
    - it can be loaded instantly at app startup
    - it only changes when we intentionally rerun the offline build
    """
    seed_entries = curated_buy_seed_entries()
    stock_universe = _aggregate_known_universe_stock_rows(sp500, nasdaq100, dow30, etf_holdings)
    existing_tickers = {entry.ticker for entry in seed_entries}
    rows: list[dict[str, Any]] = [entry.model_dump() for entry in seed_entries]

    for row in stock_universe.to_dict(orient="records"):
        ticker = str(row.get("ticker") or "").strip()
        if not ticker or ticker in existing_tickers:
            continue
        source_labels = set(row.get("source_labels") or [])
        sector = str(row.get("sector") or "Unknown")
        primary_role = _primary_role_for_stock(sector, source_labels)
        style_tilt = _style_tilt_for_stock(sector, source_labels)
        notes = "Known-universe stock candidate added from " + ", ".join(sorted(source_labels))
        rows.append(
            BuyCandidateUniverseEntry(
                ticker=ticker,
                market_data_symbol=str(row.get("market_data_symbol") or _normalize_market_data_symbol(ticker)),
                security_name=str(row.get("security_name") or ticker),
                asset_type="Stock",
                primary_role=primary_role,
                sector=sector,
                style_tilt=style_tilt,
                region="US",
                is_core=False,
                is_defensive=sector in DEFENSIVE_SECTORS,
                is_income=sector in INCOME_SECTORS,
                is_growth=sector in GROWTH_SECTORS,
                eligible_for_buy_engine=True,
                notes=notes,
                universe_source="; ".join(sorted(source_labels)),
                source_count=len(source_labels),
            ).model_dump()
        )

    frame = pd.DataFrame(rows)
    frame = _coerce_bool_columns(frame)
    frame["source_count"] = pd.to_numeric(frame.get("source_count"), errors="coerce").fillna(0).astype(int)
    frame = frame.sort_values(
        ["asset_type", "source_count", "ticker"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return frame


def build_known_universe_datasets() -> dict[str, pd.DataFrame]:
    """Fetch and save the known universe datasets used by the buy side.

    This is the main offline builder for the buy-side candidate layer. It:
    - fetches S&P 500, Nasdaq-100, and Dow 30 constituents
    - fetches top holdings for the curated ETF list
    - merges them into the saved buy-candidate seed

    The app should never call this during normal startup. It is meant to be run
    intentionally when we want to refresh the buy universe.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)

    sp500 = _extract_sp500_constituents()
    nasdaq100 = _extract_nasdaq100_constituents()
    dow30 = _extract_dow30_constituents()
    etf_holdings = fetch_major_etf_holdings(top_n=10)
    buy_universe = build_buy_candidate_universe_seed(
        sp500=sp500,
        nasdaq100=nasdaq100,
        dow30=dow30,
        etf_holdings=etf_holdings,
    )

    sp500.to_csv(SP500_CONSTITUENTS_PATH, index=False)
    nasdaq100.to_csv(NASDAQ100_CONSTITUENTS_PATH, index=False)
    dow30.to_csv(DOW30_CONSTITUENTS_PATH, index=False)
    etf_holdings.to_csv(MAJOR_ETF_HOLDINGS_PATH, index=False)
    buy_universe.to_csv(BUY_CANDIDATE_UNIVERSE_PATH, index=False)

    return {
        "sp500": sp500,
        "nasdaq100": nasdaq100,
        "dow30": dow30,
        "major_etf_holdings": etf_holdings,
        "buy_candidate_universe": buy_universe,
    }


def load_buy_candidate_universe(path: Path) -> list[BuyCandidateUniverseEntry]:
    """Read the saved buy-candidate universe CSV into typed entries."""
    if not path.exists():
        return []

    frame = _coerce_bool_columns(pd.read_csv(path))
    if "source_count" in frame.columns:
        frame["source_count"] = pd.to_numeric(frame["source_count"], errors="coerce").fillna(0).astype(int)

    return [
        BuyCandidateUniverseEntry.model_validate(row)
        for row in frame.to_dict(orient="records")
    ]


def buy_candidate_universe_frame(entries: list[BuyCandidateUniverseEntry]) -> pd.DataFrame:
    """Convert typed universe entries into a dataframe for review surfaces."""
    if not entries:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Market Data Symbol",
                "Name",
                "Asset Type",
                "Primary Role",
                "Sector",
                "Style",
                "Core",
                "Defensive",
                "Income",
                "Growth",
                "Eligible",
                "Universe Source",
                "Source Count",
                "Notes",
            ]
        )

    rows: list[dict[str, Any]] = []
    for item in entries:
        rows.append(
            {
                "Ticker": item.ticker,
                "Market Data Symbol": item.market_data_symbol,
                "Name": item.security_name,
                "Asset Type": item.asset_type,
                "Primary Role": item.primary_role,
                "Sector": item.sector,
                "Style": item.style_tilt,
                "Core": item.is_core,
                "Defensive": item.is_defensive,
                "Income": item.is_income,
                "Growth": item.is_growth,
                "Eligible": item.eligible_for_buy_engine,
                "Universe Source": item.universe_source,
                "Source Count": item.source_count,
                "Notes": item.notes,
            }
        )
    return pd.DataFrame(rows)

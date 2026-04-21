from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel


class BuyCandidateUniverseEntry(BaseModel):
    """One curated security that the future buy engine is allowed to consider.

    The buy side should not start by searching the whole market. It should start
    from a deliberately chosen universe of names we understand well enough to
    explain. This object captures that curated layer before enrichment.
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


def load_buy_candidate_universe(path: Path) -> list[BuyCandidateUniverseEntry]:
    """Read the curated buy-candidate universe CSV into typed entries."""
    if not path.exists():
        return []

    frame = pd.read_csv(path)
    bool_columns = [
        "is_core",
        "is_defensive",
        "is_income",
        "is_growth",
        "eligible_for_buy_engine",
    ]
    for column in bool_columns:
        if column in frame.columns:
            frame[column] = (
                frame[column]
                .astype(str)
                .str.strip()
                .str.upper()
                .map({"TRUE": True, "FALSE": False})
                .fillna(False)
            )

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
                "Notes": item.notes,
            }
        )
    return pd.DataFrame(rows)

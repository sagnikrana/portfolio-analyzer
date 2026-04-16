from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field


class RiskMetricObservation(BaseModel):
    metric_key: str
    group: Optional[str] = None
    label: Optional[str] = None
    raw_value: Optional[Any] = None
    score: Optional[float] = None
    score_readout: Optional[str] = None
    meaning: Optional[str] = None
    bigger_picture: Optional[str] = None


class DiagnosisConcern(BaseModel):
    concern_key: str
    label: str
    severity_score: float
    severity_band: str
    summary: str
    evidence_metric_keys: list[str] = Field(default_factory=list)
    related_tickers: list[str] = Field(default_factory=list)
    related_sectors: list[str] = Field(default_factory=list)


class HoldingRiskDriver(BaseModel):
    ticker: str
    sector: Optional[str] = None
    current_weight: Optional[float] = None
    excess_return_vs_benchmark: Optional[float] = None
    variance_contribution_pct: Optional[float] = None
    driver_reasons: list[str] = Field(default_factory=list)


class SectorRiskDriver(BaseModel):
    sector: str
    weight_pct: Optional[float] = None
    excess_return_vs_benchmark: Optional[float] = None
    driver_reasons: list[str] = Field(default_factory=list)


class DiagnosisSourceCoverage(BaseModel):
    risk_metrics_available: bool
    open_positions_available: bool
    sector_allocation_available: bool
    volatility_drivers_available: bool
    performance_attribution_available: bool
    filing_text_available: bool = False
    news_text_available: bool = False
    company_facts_available: bool = False
    company_profiles_available: bool = False
    macro_series_available: bool = False


class MacroRegimeSnapshot(BaseModel):
    as_of_date: Optional[str] = None
    fed_funds_rate: Optional[float] = None
    inflation_yoy: Optional[float] = None
    unemployment_rate: Optional[float] = None
    ten_year_yield: Optional[float] = None
    two_year_yield: Optional[float] = None
    yield_curve_spread: Optional[float] = None
    regime_flags: list[str] = Field(default_factory=list)
    summary: str


class CompanyFundamentalSnapshot(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    beta: Optional[float] = None
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    stockholders_equity: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    latest_filed_date: Optional[str] = None
    signals: list[str] = Field(default_factory=list)


class NarrativeEvidence(BaseModel):
    ticker: str
    source_type: str
    document_date: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: str


class PortfolioRiskDiagnosis(BaseModel):
    run_id: str
    generated_at: Optional[str] = None
    dataset_source: Optional[str] = None
    analysis_start: str
    analysis_end: str
    benchmark_symbol: str
    observed_risk_score: float
    observed_risk_band: str
    stated_risk_score: float
    stated_risk_band: str
    alignment: str
    confidence_band: str
    diagnostic_summary: str
    top_concerns: list[DiagnosisConcern] = Field(default_factory=list)
    top_holding_drivers: list[HoldingRiskDriver] = Field(default_factory=list)
    top_sector_drivers: list[SectorRiskDriver] = Field(default_factory=list)
    supporting_metrics: list[RiskMetricObservation] = Field(default_factory=list)
    macro_context: Optional[MacroRegimeSnapshot] = None
    holding_fundamentals: list[CompanyFundamentalSnapshot] = Field(default_factory=list)
    narrative_evidence: list[NarrativeEvidence] = Field(default_factory=list)
    data_coverage: DiagnosisSourceCoverage
    evidence_gaps: list[str] = Field(default_factory=list)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _severity_band(score: Optional[float]) -> str:
    value = float(score or 0.0)
    if value < 20:
        return "Low concern"
    if value < 40:
        return "Mild concern"
    if value < 60:
        return "Moderate concern"
    if value < 80:
        return "High concern"
    return "Very high concern"


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def load_diagnosis_bundle(base_dir: Path) -> dict[str, Any]:
    processed_dir = base_dir.parent
    return {
        "manifest": _load_json(base_dir / "diagnosis_manifest.json"),
        "headline": _load_json(base_dir / "headline_metrics.json"),
        "risk": _load_json(base_dir / "risk_score.json"),
        "portfolio_summary": _load_json(base_dir / "portfolio_summary.json"),
        "risk_metrics": _load_csv(base_dir / "risk_metric_scores.csv"),
        "risk_dimensions": _load_csv(base_dir / "risk_dimension_scores.csv"),
        "open_positions": _load_csv(base_dir / "open_positions.csv"),
        "sector_allocation": _load_csv(base_dir / "sector_allocation.csv"),
        "performance_attribution": _load_csv(base_dir / "performance_attribution.csv"),
        "volatility_drivers": _load_csv(base_dir / "top_volatility_drivers_2025.csv"),
        "company_facts": _load_csv(processed_dir / "structured" / "company_facts.csv"),
        "company_profiles": _load_csv(processed_dir / "structured" / "company_profiles.csv"),
        "macro_series": _load_csv(processed_dir / "structured" / "macro_series.csv"),
        "news_metadata": _load_csv(processed_dir / "structured" / "news_metadata.csv"),
        "document_corpus": _load_csv(processed_dir / "unstructured" / "document_corpus.csv"),
    }


def _build_metric_observations(bundle: dict[str, Any]) -> list[RiskMetricObservation]:
    risk_metrics = bundle["risk_metrics"].copy()
    raw_value_map = bundle["risk"].get("component_raw_values", {})
    observations: list[RiskMetricObservation] = []
    for row in risk_metrics.to_dict(orient="records"):
        metric_key = str(row.get("metric_key"))
        raw_value = row.get("raw_value")
        if pd.isna(raw_value) or raw_value is None:
            raw_value = raw_value_map.get(metric_key)
        observations.append(
            RiskMetricObservation(
                metric_key=metric_key,
                group=row.get("group"),
                label=row.get("label"),
                raw_value=_safe_float(raw_value) if _safe_float(raw_value) is not None else raw_value,
                score=_safe_float(row.get("score")),
                score_readout=row.get("score_readout"),
                meaning=row.get("meaning"),
                bigger_picture=row.get("bigger_picture"),
            )
        )
    return observations


def _build_holding_drivers(bundle: dict[str, Any]) -> list[HoldingRiskDriver]:
    positions = bundle["open_positions"].copy()
    volatility = bundle["volatility_drivers"].copy()
    if positions.empty:
        return []

    if not volatility.empty:
        volatility = volatility.rename(columns={"avg_weight_pct": "avg_weight_pct_2025"})
    merged = positions.merge(volatility, on="ticker", how="left")

    driver_rows: list[HoldingRiskDriver] = []
    for row in merged.sort_values("current_weight", ascending=False).to_dict(orient="records"):
        reasons: list[str] = []
        current_weight = _safe_float(row.get("current_weight"))
        variance_contribution_pct = _safe_float(row.get("variance_contribution_pct"))
        excess_return_vs_benchmark = _safe_float(row.get("excess_return_vs_benchmark"))

        if current_weight is not None and current_weight >= 0.15:
            reasons.append("large position size")
        if variance_contribution_pct is not None and variance_contribution_pct >= 0.08:
            reasons.append("meaningful 2025 volatility contributor")
        if excess_return_vs_benchmark is not None and excess_return_vs_benchmark < 0:
            reasons.append("lagging benchmark since buy")

        if reasons:
            driver_rows.append(
                HoldingRiskDriver(
                    ticker=str(row.get("ticker")),
                    sector=row.get("sector"),
                    current_weight=current_weight,
                    excess_return_vs_benchmark=excess_return_vs_benchmark,
                    variance_contribution_pct=variance_contribution_pct,
                    driver_reasons=reasons,
                )
            )

    if not driver_rows:
        top_rows = merged.sort_values("current_weight", ascending=False).head(5).to_dict(orient="records")
        for row in top_rows:
            driver_rows.append(
                HoldingRiskDriver(
                    ticker=str(row.get("ticker")),
                    sector=row.get("sector"),
                    current_weight=_safe_float(row.get("current_weight")),
                    excess_return_vs_benchmark=_safe_float(row.get("excess_return_vs_benchmark")),
                    variance_contribution_pct=_safe_float(row.get("variance_contribution_pct")),
                    driver_reasons=["largest current positions"],
                )
            )

    return driver_rows[:5]


def _latest_metric_value(
    company_facts: pd.DataFrame,
    ticker: str,
    canonical_metric: str,
) -> tuple[Optional[float], Optional[str]]:
    subset = company_facts[
        (company_facts["ticker"] == ticker)
        & (company_facts["canonical_metric"] == canonical_metric)
    ].copy()
    if subset.empty:
        return None, None
    subset["filed_date"] = pd.to_datetime(subset["filed_date"], errors="coerce")
    subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
    subset = subset.dropna(subset=["filed_date", "value"]).sort_values("filed_date")
    if subset.empty:
        return None, None
    row = subset.iloc[-1]
    return _safe_float(row.get("value")), str(row.get("filed_date").date())


def _build_macro_context(bundle: dict[str, Any]) -> Optional[MacroRegimeSnapshot]:
    macro = bundle["macro_series"].copy()
    if macro.empty:
        return None
    macro["date"] = pd.to_datetime(macro["date"], errors="coerce")
    macro["value"] = pd.to_numeric(macro["value"], errors="coerce")
    macro = macro.dropna(subset=["date", "value"])
    if macro.empty:
        return None

    def latest_value(series_id: str) -> tuple[Optional[float], Optional[pd.Timestamp]]:
        subset = macro[macro["series_id"] == series_id].sort_values("date")
        if subset.empty:
            return None, None
        row = subset.iloc[-1]
        return _safe_float(row["value"]), row["date"]

    fed_funds_rate, fed_date = latest_value("FEDFUNDS")
    unemployment_rate, unrate_date = latest_value("UNRATE")
    ten_year_yield, ten_year_date = latest_value("DGS10")
    two_year_yield, two_year_date = latest_value("DGS2")

    cpi = macro[macro["series_id"] == "CPIAUCSL"].sort_values("date")
    inflation_yoy = None
    cpi_date = None
    if len(cpi) >= 13:
        latest = cpi.iloc[-1]
        prior = cpi[cpi["date"] <= latest["date"] - pd.DateOffset(years=1)]
        if not prior.empty and prior.iloc[-1]["value"] not in (0, None):
            inflation_yoy = (_safe_float(latest["value"]) / _safe_float(prior.iloc[-1]["value"])) - 1
            cpi_date = latest["date"]

    yield_curve_spread = None
    if ten_year_yield is not None and two_year_yield is not None:
        yield_curve_spread = ten_year_yield - two_year_yield

    regime_flags: list[str] = []
    if fed_funds_rate is not None and fed_funds_rate >= 3.0:
        regime_flags.append("rates still restrictive")
    if inflation_yoy is not None and inflation_yoy >= 0.03:
        regime_flags.append("inflation still sticky")
    if unemployment_rate is not None and unemployment_rate <= 4.5:
        regime_flags.append("labor market still stable")
    if yield_curve_spread is not None and yield_curve_spread > 0:
        regime_flags.append("yield curve positively sloped")
    elif yield_curve_spread is not None:
        regime_flags.append("yield curve still inverted")

    as_of_date = max(
        [
            d for d in [fed_date, unrate_date, ten_year_date, two_year_date, cpi_date] if d is not None
        ],
        default=None,
    )
    summary = (
        "Macro backdrop looks moderately restrictive but stable. "
        f"Fed funds are around {fed_funds_rate:.2f}% "
        f"with inflation near {inflation_yoy:.2%} year over year and unemployment around {unemployment_rate:.1f}%. "
        f"The 10Y-2Y spread is about {yield_curve_spread:.2f}%."
        if fed_funds_rate is not None and inflation_yoy is not None and unemployment_rate is not None and yield_curve_spread is not None
        else "Macro backdrop is partially available, but not all core regime series were present."
    )
    return MacroRegimeSnapshot(
        as_of_date=as_of_date.date().isoformat() if as_of_date is not None else None,
        fed_funds_rate=fed_funds_rate,
        inflation_yoy=inflation_yoy,
        unemployment_rate=unemployment_rate,
        ten_year_yield=ten_year_yield,
        two_year_yield=two_year_yield,
        yield_curve_spread=yield_curve_spread,
        regime_flags=regime_flags,
        summary=summary,
    )


def _build_holding_fundamentals(
    bundle: dict[str, Any],
    top_holding_drivers: list[HoldingRiskDriver],
) -> list[CompanyFundamentalSnapshot]:
    company_facts = bundle["company_facts"].copy()
    profiles = bundle["company_profiles"].copy()
    if company_facts.empty and profiles.empty:
        return []

    snapshots: list[CompanyFundamentalSnapshot] = []
    for driver in top_holding_drivers:
        ticker = driver.ticker
        profile_row = profiles[profiles["symbol"] == ticker].head(1)
        profile = profile_row.iloc[0].to_dict() if not profile_row.empty else {}

        revenue, revenue_date = _latest_metric_value(company_facts, ticker, "revenue")
        net_income, net_income_date = _latest_metric_value(company_facts, ticker, "net_income")
        cash, cash_date = _latest_metric_value(company_facts, ticker, "cash_and_equivalents")
        assets, assets_date = _latest_metric_value(company_facts, ticker, "total_assets")
        liabilities, liabilities_date = _latest_metric_value(company_facts, ticker, "liability_related")
        equity, equity_date = _latest_metric_value(company_facts, ticker, "stockholders_equity")
        operating_cash_flow, operating_cash_flow_date = _latest_metric_value(
            company_facts, ticker, "operating_cash_flow"
        )

        dates = [d for d in [revenue_date, net_income_date, cash_date, assets_date, liabilities_date, equity_date, operating_cash_flow_date] if d]
        latest_filed_date = max(dates) if dates else None

        signals: list[str] = []
        beta = _safe_float(profile.get("beta"))
        if beta is not None and beta > 1.2:
            signals.append("above-market beta")
        if net_income is not None and net_income > 0:
            signals.append("latest net income positive")
        elif net_income is not None:
            signals.append("latest net income negative")
        if operating_cash_flow is not None and operating_cash_flow > 0:
            signals.append("operating cash flow positive")
        if assets is not None and liabilities is not None and assets > 0:
            if liabilities / assets > 0.75:
                signals.append("liabilities are a large share of assets")
            else:
                signals.append("balance sheet coverage looks reasonable")

        snapshots.append(
            CompanyFundamentalSnapshot(
                ticker=ticker,
                company_name=profile.get("companyName"),
                sector=profile.get("sector"),
                industry=profile.get("industry"),
                market_cap=_safe_float(profile.get("marketCap")),
                beta=beta,
                revenue=revenue,
                net_income=net_income,
                cash_and_equivalents=cash,
                total_assets=assets,
                total_liabilities=liabilities,
                stockholders_equity=equity,
                operating_cash_flow=operating_cash_flow,
                latest_filed_date=latest_filed_date,
                signals=signals,
            )
        )
    return snapshots


def _clean_snippet(text: Any, limit: int = 280) -> str:
    snippet = str(text or "").replace("\n", " ").replace("\r", " ").strip()
    snippet = " ".join(snippet.split())
    return snippet[:limit]


def _build_narrative_evidence(
    bundle: dict[str, Any],
    top_holding_drivers: list[HoldingRiskDriver],
) -> list[NarrativeEvidence]:
    corpus = bundle["document_corpus"].copy()
    if corpus.empty:
        return []
    evidence: list[NarrativeEvidence] = []
    tickers = [driver.ticker for driver in top_holding_drivers]
    for ticker in tickers:
        ticker_docs = corpus[corpus["ticker"] == ticker].copy()
        if ticker_docs.empty:
            continue
        ticker_docs["document_date"] = pd.to_datetime(ticker_docs["document_date"], errors="coerce")
        for source_type in ["sec_filing", "news_article"]:
            subset = ticker_docs[ticker_docs["source_type"] == source_type].sort_values("document_date", ascending=False)
            if subset.empty:
                continue
            row = subset.iloc[0]
            evidence.append(
                NarrativeEvidence(
                    ticker=ticker,
                    source_type=str(source_type),
                    document_date=row["document_date"].date().isoformat() if pd.notna(row["document_date"]) else None,
                    title=row.get("title"),
                    url=row.get("url"),
                    snippet=_clean_snippet(row.get("body_text")),
                )
            )
    return evidence


def _build_sector_drivers(bundle: dict[str, Any]) -> list[SectorRiskDriver]:
    sectors = bundle["sector_allocation"].copy()
    if sectors.empty:
        return []

    drivers: list[SectorRiskDriver] = []
    for row in sectors.sort_values("weight_pct", ascending=False).to_dict(orient="records"):
        reasons: list[str] = []
        weight_pct = _safe_float(row.get("weight_pct"))
        excess_return_vs_benchmark = _safe_float(row.get("excess_return_vs_benchmark"))
        if weight_pct is not None and weight_pct >= 0.25:
            reasons.append("large sector concentration")
        if excess_return_vs_benchmark is not None and excess_return_vs_benchmark < 0:
            reasons.append("sector is lagging trade-matched benchmark")
        if reasons:
            drivers.append(
                SectorRiskDriver(
                    sector=str(row.get("sector")),
                    weight_pct=weight_pct,
                    excess_return_vs_benchmark=excess_return_vs_benchmark,
                    driver_reasons=reasons,
                )
            )
    return drivers[:3]


def _concern_summary(concern_key: str, bundle: dict[str, Any]) -> str:
    risk = bundle["risk"]
    raw = risk.get("component_raw_values", {})
    headline = bundle["headline"]
    if concern_key == "concentration":
        largest = _safe_float(raw.get("concentration::single_position_weight"))
        top5 = _safe_float(raw.get("concentration::top_5_weight"))
        effective = _safe_float(raw.get("concentration::effective_holdings"))
        return (
            "The portfolio looks concentrated in a small number of names. "
            f"Largest position weight is {largest:.1%} and top five holdings account for about {top5:.1%} of capital. "
            f"Effective holdings are only about {effective:.2f}, so diversification is thinner than the ticker count suggests."
        )
    if concern_key == "market":
        rel_vol = _safe_float(raw.get("market::relative_volatility_to_benchmark"))
        rel_dd = _safe_float(raw.get("market::relative_drawdown_to_benchmark"))
        rel_beta = _safe_float(raw.get("market::relative_market_sensitivity_to_benchmark"))
        return (
            "The portfolio has recently behaved more aggressively than the S&P 500. "
            f"Relative volatility is about {rel_vol:.2f}x, relative drawdown depth is about {rel_dd:.2f}x, "
            f"and market sensitivity is about {rel_beta:.2f}x versus the benchmark."
        )
    turnover = _safe_float(raw.get("behavior::turnover"))
    holding_period = _safe_float(raw.get("behavior::short_holding_period"))
    target_days = risk.get("short_holding_period_target_days")
    return (
        "Behavioral risk looks relatively contained, but it still contributes context to the diagnosis. "
        f"Turnover is about {turnover:.1%} and weighted holding period is about {holding_period:.0f} days "
        f"against a target of {target_days} days."
    )


def _build_top_concerns(
    bundle: dict[str, Any],
    top_holding_drivers: list[HoldingRiskDriver],
    top_sector_drivers: list[SectorRiskDriver],
) -> list[DiagnosisConcern]:
    risk = bundle["risk"]
    dimension_scores = risk.get("dimension_scores", {})
    concern_configs = [
        (
            "concentration",
            "Concentration risk",
            float(dimension_scores.get("concentration_risk", 0.0)),
            [
                "concentration::single_position_weight",
                "concentration::top_5_weight",
                "concentration::effective_holdings",
            ],
        ),
        (
            "market",
            "Market-relative risk",
            float(dimension_scores.get("market_risk", 0.0)),
            [
                "market::relative_volatility_to_benchmark",
                "market::relative_drawdown_to_benchmark",
                "market::relative_downside_capture_to_benchmark",
                "market::relative_market_sensitivity_to_benchmark",
                "market::equity_exposure",
            ],
        ),
        (
            "behavior",
            "Behavioral risk",
            float(dimension_scores.get("behavioral_risk", 0.0)),
            [
                "behavior::turnover",
                "behavior::short_holding_period",
            ],
        ),
    ]

    concerns: list[DiagnosisConcern] = []
    related_tickers = [driver.ticker for driver in top_holding_drivers]
    related_sectors = [driver.sector for driver in top_sector_drivers if driver.sector]
    for concern_key, label, severity_score, metric_keys in concern_configs:
        if severity_score <= 0:
            continue
        concerns.append(
            DiagnosisConcern(
                concern_key=concern_key,
                label=label,
                severity_score=round(severity_score, 1),
                severity_band=_severity_band(severity_score),
                summary=_concern_summary(concern_key, bundle),
                evidence_metric_keys=metric_keys,
                related_tickers=related_tickers[:3] if concern_key != "behavior" else [],
                related_sectors=related_sectors[:2] if concern_key in {"concentration", "market"} else [],
            )
        )
    concerns.sort(key=lambda concern: concern.severity_score, reverse=True)
    return concerns


def _build_diagnostic_summary(bundle: dict[str, Any], top_concerns: list[DiagnosisConcern]) -> str:
    risk = bundle["risk"]
    headline = bundle["headline"]
    concern_labels = ", ".join(concern.label.lower() for concern in top_concerns[:2]) or "overall portfolio risk"
    return (
        f"Observed portfolio risk is {risk['score']:.1f}/100 ({risk['band']}) versus a stated risk of "
        f"{risk['stated_score']:.1f}/100 ({risk['stated_band']}). The portfolio currently looks "
        f"{risk['alignment'].replace('Observed portfolio risk is ', '').lower()}. "
        f"The main diagnosis is driven by {concern_labels}, over the analysis window "
        f"{headline['analysis_start']} to {headline['analysis_end']}."
    )


def _build_evidence_gaps(bundle: dict[str, Any], macro_context: Optional[MacroRegimeSnapshot], narrative_evidence: list[NarrativeEvidence]) -> list[str]:
    gaps = [
        "User goals and constraints beyond the stated risk score are not yet modeled in the diagnosis object.",
        "Company-specific narrative evidence is available, but it is not yet weighted quantitatively into concern ranking.",
        "Fundamental snapshots are based on broad canonical metrics and still need tighter metric selection for production use.",
    ]
    if bundle["volatility_drivers"].empty:
        gaps.append("No volatility driver table was available for holding-level variance attribution.")
    if macro_context is None:
        gaps.append("Macro regime summary could not be built from the processed macro dataset.")
    if not narrative_evidence:
        gaps.append("No usable narrative evidence was available for the main holding drivers.")
    return gaps


def portfolio_risk_diagnosis_from_saved_artifacts(base_dir: Path) -> PortfolioRiskDiagnosis:
    bundle = load_diagnosis_bundle(base_dir)
    top_holding_drivers = _build_holding_drivers(bundle)
    top_sector_drivers = _build_sector_drivers(bundle)
    top_concerns = _build_top_concerns(bundle, top_holding_drivers, top_sector_drivers)
    supporting_metrics = _build_metric_observations(bundle)
    macro_context = _build_macro_context(bundle)
    holding_fundamentals = _build_holding_fundamentals(bundle, top_holding_drivers)
    narrative_evidence = _build_narrative_evidence(bundle, top_holding_drivers)
    manifest = bundle["manifest"]
    headline = bundle["headline"]
    risk = bundle["risk"]

    return PortfolioRiskDiagnosis(
        run_id=str(manifest.get("run_id")),
        generated_at=manifest.get("generated_at"),
        dataset_source=manifest.get("dataset_source"),
        analysis_start=str(headline.get("analysis_start")),
        analysis_end=str(headline.get("analysis_end")),
        benchmark_symbol=str(headline.get("benchmark_symbol")),
        observed_risk_score=float(risk.get("score", 0.0)),
        observed_risk_band=str(risk.get("band")),
        stated_risk_score=float(risk.get("stated_score", 0.0)),
        stated_risk_band=str(risk.get("stated_band")),
        alignment=str(risk.get("alignment")),
        confidence_band=str(risk.get("confidence_band")),
        diagnostic_summary=_build_diagnostic_summary(bundle, top_concerns),
        top_concerns=top_concerns,
        top_holding_drivers=top_holding_drivers,
        top_sector_drivers=top_sector_drivers,
        supporting_metrics=supporting_metrics,
        macro_context=macro_context,
        holding_fundamentals=holding_fundamentals,
        narrative_evidence=narrative_evidence,
        data_coverage=DiagnosisSourceCoverage(
            risk_metrics_available=not bundle["risk_metrics"].empty,
            open_positions_available=not bundle["open_positions"].empty,
            sector_allocation_available=not bundle["sector_allocation"].empty,
            volatility_drivers_available=not bundle["volatility_drivers"].empty,
            performance_attribution_available=not bundle["performance_attribution"].empty,
            filing_text_available=not bundle["document_corpus"][bundle["document_corpus"]["source_type"] == "sec_filing"].empty,
            news_text_available=not bundle["document_corpus"][bundle["document_corpus"]["source_type"] == "news_article"].empty,
            company_facts_available=not bundle["company_facts"].empty,
            company_profiles_available=not bundle["company_profiles"].empty,
            macro_series_available=not bundle["macro_series"].empty,
        ),
        evidence_gaps=_build_evidence_gaps(bundle, macro_context, narrative_evidence),
    )

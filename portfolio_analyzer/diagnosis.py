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


def _build_evidence_gaps(bundle: dict[str, Any]) -> list[str]:
    gaps = [
        "User goals and constraints beyond the stated risk score are not yet modeled in the diagnosis object.",
        "Filing and news text are available in the pipeline, but company-specific narrative risk is not yet linked into concern ranking.",
        "Macro data exists, but there is not yet a dedicated macro regime summary object feeding the diagnosis.",
    ]
    if bundle["volatility_drivers"].empty:
        gaps.append("No volatility driver table was available for holding-level variance attribution.")
    return gaps


def portfolio_risk_diagnosis_from_saved_artifacts(base_dir: Path) -> PortfolioRiskDiagnosis:
    bundle = load_diagnosis_bundle(base_dir)
    top_holding_drivers = _build_holding_drivers(bundle)
    top_sector_drivers = _build_sector_drivers(bundle)
    top_concerns = _build_top_concerns(bundle, top_holding_drivers, top_sector_drivers)
    supporting_metrics = _build_metric_observations(bundle)
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
        data_coverage=DiagnosisSourceCoverage(
            risk_metrics_available=not bundle["risk_metrics"].empty,
            open_positions_available=not bundle["open_positions"].empty,
            sector_allocation_available=not bundle["sector_allocation"].empty,
            volatility_drivers_available=not bundle["volatility_drivers"].empty,
            performance_attribution_available=not bundle["performance_attribution"].empty,
        ),
        evidence_gaps=_build_evidence_gaps(bundle),
    )

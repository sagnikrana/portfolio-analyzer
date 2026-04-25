from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

try:
    from portfolio_analyzer.buy_candidates import (
        BuyCandidateUniverseEntry,
        fetch_candidate_market_metadata,
        load_buy_candidate_universe,
    )
except ModuleNotFoundError:
    from buy_candidates import (
        BuyCandidateUniverseEntry,
        fetch_candidate_market_metadata,
        load_buy_candidate_universe,
    )


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
STRUCTURED_DIR = REPO_ROOT / "data" / "processed" / "structured"
BUY_CANDIDATE_UNIVERSE_PATH = RAW_DIR / "buy_candidate_universe.csv"
BUY_CANDIDATE_UNIVERSE_ENRICHED_PATH = STRUCTURED_DIR / "buy_candidate_universe_enriched.csv"


class RiskMetricObservation(BaseModel):
    """One scored risk metric carried forward into the diagnosis layer.

    These objects preserve the original metric identity, numeric value, score,
    and plain-English explanation from the app bundle so later diagnosis objects
    can cite evidence without having to re-derive or rename everything.
    """

    metric_key: str
    group: Optional[str] = None
    label: Optional[str] = None
    raw_value: Optional[Any] = None
    score: Optional[float] = None
    score_readout: Optional[str] = None
    meaning: Optional[str] = None
    bigger_picture: Optional[str] = None


class DiagnosisConcern(BaseModel):
    """A ranked portfolio-level concern such as concentration or market risk.

    A concern sits above holding-level reasoning. It tells us *what kind* of risk
    is dominating the account before we ask which holdings are causing that risk.
    """

    concern_key: str
    label: str
    base_severity_score: float
    external_adjustment_score: float = 0.0
    severity_score: float
    severity_band: str
    summary: str
    adjustment_reasons: list[str] = Field(default_factory=list)
    evidence_metric_keys: list[str] = Field(default_factory=list)
    related_tickers: list[str] = Field(default_factory=list)
    related_sectors: list[str] = Field(default_factory=list)


class HoldingRiskDriver(BaseModel):
    """A holding that materially contributes to the diagnosis.

    This model is intentionally richer than the original version. It now keeps:
    - the raw measurements that made the holding relevant
    - explicit reason codes describing *why* it is risky
    - primary vs secondary reasons for prioritization
    - driver-level confidence so the UI can stay honest
    """

    ticker: str
    sector: Optional[str] = None
    current_weight: Optional[float] = None
    excess_return_vs_benchmark: Optional[float] = None
    variance_contribution_pct: Optional[float] = None
    driver_reasons: list[str] = Field(default_factory=list)
    primary_reason_code: Optional[str] = None
    primary_reason_label: Optional[str] = None
    primary_reason_summary: Optional[str] = None
    secondary_reason_codes: list[str] = Field(default_factory=list)
    driver_confidence_score: float = 0.0
    driver_confidence_band: str = "Low"
    evidence_summary: list[str] = Field(default_factory=list)
    reason_codes: list["ReasonCode"] = Field(default_factory=list)


class SectorRiskDriver(BaseModel):
    """A sector exposure that materially contributes to the diagnosis."""

    sector: str
    weight_pct: Optional[float] = None
    excess_return_vs_benchmark: Optional[float] = None
    driver_reasons: list[str] = Field(default_factory=list)
    primary_reason_code: Optional[str] = None
    primary_reason_label: Optional[str] = None
    primary_reason_summary: Optional[str] = None
    driver_confidence_score: float = 0.0
    driver_confidence_band: str = "Low"
    reason_codes: list["ReasonCode"] = Field(default_factory=list)


class DiagnosisSourceCoverage(BaseModel):
    """Coverage flags for the data sources feeding the diagnosis."""

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
    """A compact macro regime summary derived from processed FRED series."""

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
    """A narrow fundamental view for holdings already identified as important."""

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
    """One recent filing or news artifact attached to a diagnosis driver."""

    ticker: str
    source_type: str
    document_date: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: str


NEGATIVE_NARRATIVE_KEYWORDS = {
    "delay",
    "concern",
    "concerns",
    "cybersecurity",
    "downgrade",
    "lower",
    "litigation",
    "regulation",
    "antitrust",
    "headwind",
    "risk",
    "risks",
    "volatile",
    "pressure",
}


class ReasonCode(BaseModel):
    """A structured explanation token for why something is risky.

    The reason-code framework is the bridge between raw evidence and later UI or
    LLM explanation. Each code captures:
    - a stable machine-readable key
    - a user-facing label
    - the category of evidence it came from
    - a severity score used for prioritization
    - a short explanation and supporting evidence

    This makes the diagnosis layer easier to audit, test, and redesign because
    we can talk about *reason codes* directly instead of inferring intent from
    free-form strings.
    """

    code: str
    label: str
    category: str
    severity_score: float
    summary: str
    evidence: list[str] = Field(default_factory=list)


class HoldingConcernContribution(BaseModel):
    """One holding's contribution to one specific diagnosis concern.

    This object lets the system answer a more precise question than
    `HoldingRiskDriver` alone can answer:

    "How much is this holding contributing to *this specific concern*?"

    The contribution object keeps the concern link explicit, preserves the
    supporting reason codes that led to the contribution score, and carries its
    own confidence so later action logic can stay honest about uncertainty.
    """

    concern_key: str
    concern_label: str
    contribution_score: float
    contribution_band: str
    contribution_summary: str
    supporting_reason_codes: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    confidence_band: str = "Low"


class HoldingRiskContribution(BaseModel):
    """A holding-centric view of how one position feeds portfolio diagnosis.

    `PortfolioRiskDiagnosis` tells us what is wrong overall. `HoldingRiskDriver`
    tells us which positions show up as important. This object is the bridge
    between those layers and later action logic:

    - one object per important holding
    - a ranked set of concern-specific contributions
    - a primary concern and overall contribution strength
    - concise evidence and confidence we can later surface in UI or actions
    """

    ticker: str
    sector: Optional[str] = None
    current_weight: Optional[float] = None
    overall_contribution_score: float
    overall_contribution_band: str
    primary_concern_key: str
    primary_concern_label: str
    primary_concern_summary: str
    contribution_confidence_score: float = 0.0
    contribution_confidence_band: str = "Low"
    contribution_summary: str
    concern_contributions: list["HoldingConcernContribution"] = Field(default_factory=list)
    evidence_summary: list[str] = Field(default_factory=list)


class HoldingActionNeed(BaseModel):
    """First action-layer object built on top of diagnosis contributions.

    This object does not yet recommend replacements or full rebalancing. Its
    job is narrower and safer: decide whether an important holding currently
    looks like something to hold steady, monitor closely, trim, or reduce.

    The design is intentionally conservative because we want the first action
    layer to be auditable and grounded in diagnosis evidence rather than overly
    eager portfolio advice.
    """

    ticker: str
    sector: Optional[str] = None
    current_weight: Optional[float] = None
    action_label: str
    action_code: str
    action_pressure_score: float
    action_pressure_band: str
    action_urgency: str
    primary_action_reason: str
    action_summary: str
    linked_primary_concern: str
    supporting_concerns: list[str] = Field(default_factory=list)
    supporting_reason_codes: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    confidence_band: str = "Low"


class HoldingActionRecommendation(BaseModel):
    """Portfolio action recommendation grounded in relative underperformance.

    This layer is intentionally narrower than a full portfolio optimizer. It is
    designed to answer:

    - should this holding actually be trimmed or sold, rather than merely watched?
    - how much of the position should be reduced right now?
    - what specific evidence made that recommendation reasonable?

    The core rule is conservative by design: underperformance versus the
    trade-matched S&P 500 over a clearly stated holding period must be present
    before a sell-style recommendation appears. Diagnosis pressure and
    contribution evidence can strengthen the case, but they do not replace the
    relative-performance requirement.
    """

    ticker: str
    sector: Optional[str] = None
    current_weight: Optional[float] = None
    current_value: Optional[float] = None
    quantity: Optional[float] = None
    recommendation_label: str
    recommendation_code: str
    position_reduction_pct: float = 0.0
    shares_to_sell: Optional[float] = None
    value_to_sell: Optional[float] = None
    target_weight_after_action: Optional[float] = None
    projected_weight_reduction_pct_points: Optional[float] = None
    projected_variance_reduction_pct_points: Optional[float] = None
    projected_relative_drag_reduction_pct_points: Optional[float] = None
    performance_window_start: Optional[str] = None
    performance_window_end: Optional[str] = None
    performance_window_label: str
    holding_return_pct: Optional[float] = None
    benchmark_return_pct: Optional[float] = None
    relative_performance_vs_benchmark: Optional[float] = None
    stock_1y_return_pct: Optional[float] = None
    benchmark_1y_return_pct: Optional[float] = None
    relative_1y_return_pct: Optional[float] = None
    stock_3y_return_pct: Optional[float] = None
    benchmark_3y_return_pct: Optional[float] = None
    relative_3y_return_pct: Optional[float] = None
    stock_5y_return_pct: Optional[float] = None
    benchmark_5y_return_pct: Optional[float] = None
    relative_5y_return_pct: Optional[float] = None
    linked_action_need_label: Optional[str] = None
    linked_primary_concern: Optional[str] = None
    diagnosis_pressure_score: float = 0.0
    recommendation_summary: str
    what_changed: str = ""
    why_it_matters: str = ""
    amount_rationale: str = ""
    explicit_sell_modifiers: list[str] = Field(default_factory=list)
    modifier_score: float = 0.0
    portfolio_impact_summary: str = ""
    portfolio_impact_bullets: list[str] = Field(default_factory=list)
    reasoning_points: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    guardrail_notes: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    confidence_band: str = "Low"
    is_actionable: bool = False


class PortfolioActionImpact(BaseModel):
    """Portfolio-level preview of what improves if current actions are followed.

    This object aggregates all actionable holding recommendations into one
    simple question for the user:

    - if I actually trim or sell these names, what gets better overall?

    The preview is intentionally directional rather than fully optimized. It is
    meant to explain the likely benefit of the current action set before we
    build a deeper rebalance engine.
    """

    actionable_count: int = 0
    actionable_tickers: list[str] = Field(default_factory=list)
    total_value_to_sell: float = 0.0
    total_weight_reduction_pct_points: float = 0.0
    total_variance_reduction_pct_points: float = 0.0
    total_relative_drag_reduction_pct_points: float = 0.0
    current_largest_position_pct: Optional[float] = None
    projected_largest_position_pct: Optional[float] = None
    current_top5_weight_pct: Optional[float] = None
    projected_top5_weight_pct: Optional[float] = None
    impact_summary: str = ""
    impact_bullets: list[str] = Field(default_factory=list)


class PortfolioTraitSnapshot(BaseModel):
    """Before/after portfolio traits used in the rebalance-plan comparison view."""

    label: str
    invested_value: float = 0.0
    cash_value: float = 0.0
    total_account_value: float = 0.0
    invested_share_of_account: float = 0.0
    cash_share_of_account: float = 0.0
    largest_position_pct_of_invested: float = 0.0
    top5_weight_pct_of_invested: float = 0.0
    effective_holdings: float = 0.0
    top_sector: str = ""
    top_sector_weight_pct_of_invested: float = 0.0


class RebalanceHoldingChange(BaseModel):
    """One holding-level change inside the portfolio rebalance plan."""

    ticker: str
    security_name: str
    sector: str = ""
    action_label: str
    before_value: float = 0.0
    after_value: float = 0.0
    value_change: float = 0.0
    before_weight_pct_of_invested: float = 0.0
    after_weight_pct_of_invested: float = 0.0
    weight_change_pct_points: float = 0.0
    explanation: str = ""


class RebalanceSectorChange(BaseModel):
    """Sector-level before/after comparison for the rebalance plan."""

    sector: str
    before_value: float = 0.0
    after_value: float = 0.0
    before_weight_pct_of_invested: float = 0.0
    after_weight_pct_of_invested: float = 0.0
    weight_change_pct_points: float = 0.0


class PortfolioRebalancePlan(BaseModel):
    """Integrated before/after plan combining current sells and proposed adds.

    This object is the first time the project answers the full portfolio-level
    question:

    - what does the portfolio look like today?
    - what does it look like if we follow the current sell ideas and buy ideas?

    The plan is still deterministic and explanation-first. It does not try to
    be a perfect optimizer yet; it creates a reviewable proposal that ties
    together the action layer and the buy layer in one place.
    """

    summary: str = ""
    plan_assumptions: list[str] = Field(default_factory=list)
    improvement_bullets: list[str] = Field(default_factory=list)
    before_snapshot: Optional["PortfolioTraitSnapshot"] = None
    after_snapshot: Optional["PortfolioTraitSnapshot"] = None
    total_value_to_sell: float = 0.0
    total_value_to_buy: float = 0.0
    projected_cash_after_plan: float = 0.0
    sell_tickers: list[str] = Field(default_factory=list)
    buy_tickers: list[str] = Field(default_factory=list)
    holding_changes: list["RebalanceHoldingChange"] = Field(default_factory=list)
    sector_changes: list["RebalanceSectorChange"] = Field(default_factory=list)


class PortfolioGap(BaseModel):
    """A portfolio-level gap that should be understood before suggesting buys.

    The purpose of this object is explanation-first. It answers:

    - what is still missing after the current trims or sells?
    - why does that matter for this portfolio specifically?
    - what would likely improve if the gap were filled later?

    This keeps the future buy-side grounded in a portfolio need instead of
    turning directly into "here are some stocks to buy."
    """

    gap_key: str
    label: str
    severity_score: float
    severity_band: str
    what_is_missing: str
    why_this_gap_exists: str
    what_would_change: list[str] = Field(default_factory=list)
    linked_concerns: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    suggested_vehicle_tilt: str = ""


class PortfolioPreferences(BaseModel):
    """Default preferences and constraints for the future buy-side.

    The project does not yet ask the user for buy preferences directly, but the
    buy-side needs an object that says what constraints are already known and
    what still needs an explicit answer. This model captures:

    - capital that is already available or could be freed by current actions
    - risk tolerance carried forward from the dashboard input
    - simple default guardrails for new adds
    - unresolved decisions that the user should eventually confirm
    """

    available_cash_now: float = 0.0
    available_cash_if_actions_followed: float = 0.0
    current_invested_value: float = 0.0
    current_total_portfolio_value: float = 0.0
    budget_to_deploy: Optional[float] = None
    reinvest_freed_cash: Optional[bool] = None
    reinvestment_preference_label: str = ""
    stated_risk_score: float = 0.0
    stated_risk_band: str = ""
    suggested_max_new_position_pct: Optional[float] = None
    max_new_position_interpretation: str = ""
    allow_etfs: bool = True
    allow_single_stocks: bool = True
    single_stocks_preferred: Optional[bool] = None
    prefer_high_dividend_etfs: bool = False
    prefer_low_expense_for_dividend_etfs: bool = False
    buy_idea_limit: int = 10
    include_existing_holdings: bool = False
    vehicle_preference_label: str = ""
    sector_preferences: list[str] = Field(default_factory=list)
    inferred_sector_avoidances: list[str] = Field(default_factory=list)
    constraints_summary: str = ""
    unresolved_preferences: list[str] = Field(default_factory=list)
    assumption_notes: list[str] = Field(default_factory=list)
    preference_source: str = "inferred_defaults"


class ReplacementCandidate(BaseModel):
    """A first-pass buy-side idea tied to a portfolio gap and user constraints.

    This object is intentionally narrower than a full rebalance optimizer. Its
    job is to answer:

    - which curated candidate looks worth considering next?
    - which specific portfolio gap would it help fill?
    - why does it fit the current buy preferences?
    - what simple evidence supports keeping it in the idea set?

    The object stays explanation-first so the user can understand *why the name
    is here* before we ever get into a harder buy-size or optimization layer.
    """

    ticker: str
    security_name: str
    asset_type: str
    sector: str
    primary_role: str
    style_tilt: str
    linked_gap_key: str
    linked_gap_label: str
    fit_score: float
    fit_band: str
    why_it_fits: str
    what_it_improves: list[str] = Field(default_factory=list)
    preference_fit_summary: str = ""
    evidence_summary: list[str] = Field(default_factory=list)
    universe_source: str = ""
    suggested_allocation_pct_of_budget: Optional[float] = None
    suggested_allocation_amount: Optional[float] = None
    relative_1y_return_pct: Optional[float] = None
    relative_3y_return_pct: Optional[float] = None
    relative_5y_return_pct: Optional[float] = None
    annualized_volatility_1y: Optional[float] = None
    beta: Optional[float] = None
    confidence_score: float = 0.0
    confidence_band: str = "Low"
    stock_5y_return_pct: Optional[float] = None
    expense_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    trailing_pe: Optional[float] = None
    is_existing_holding: bool = False
    external_signal_summary: list[str] = Field(default_factory=list)


class CurrentHoldingSnapshot(BaseModel):
    """Minimal current-holding record used for reinvestment-aware buy ideas."""

    ticker: str
    security_name: str
    sector: Optional[str] = None
    current_weight: Optional[float] = None
    current_value: Optional[float] = None
    excess_return_vs_benchmark: Optional[float] = None
    weighted_avg_buy_date: Optional[str] = None


class PortfolioRiskDiagnosis(BaseModel):
    """Top-level typed diagnosis object for one analyzed portfolio snapshot.

    This is the main object the notebook and dashboard should consume. It keeps
    the diagnosis layered and inspectable:
    - top-level portfolio conclusion
    - ranked concern categories
    - holding and sector drivers
    - supporting evidence from metrics, fundamentals, macro, and narrative
    - explicit coverage and evidence gaps

    The model is intentionally richer than a reporting dataframe so later stages
    can build relationships between concerns, drivers, and actions cleanly.
    """
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
    holding_risk_contributions: list[HoldingRiskContribution] = Field(default_factory=list)
    holding_action_needs: list[HoldingActionNeed] = Field(default_factory=list)
    holding_action_recommendations: list[HoldingActionRecommendation] = Field(default_factory=list)
    portfolio_action_impact: Optional[PortfolioActionImpact] = None
    portfolio_rebalance_plan: Optional[PortfolioRebalancePlan] = None
    portfolio_gaps: list[PortfolioGap] = Field(default_factory=list)
    portfolio_preferences: Optional[PortfolioPreferences] = None
    current_holdings: list[CurrentHoldingSnapshot] = Field(default_factory=list)
    replacement_candidates: list[ReplacementCandidate] = Field(default_factory=list)
    supporting_metrics: list[RiskMetricObservation] = Field(default_factory=list)
    macro_context: Optional[MacroRegimeSnapshot] = None
    holding_fundamentals: list[CompanyFundamentalSnapshot] = Field(default_factory=list)
    narrative_evidence: list[NarrativeEvidence] = Field(default_factory=list)
    data_coverage: DiagnosisSourceCoverage
    evidence_gaps: list[str] = Field(default_factory=list)


HoldingRiskDriver.model_rebuild()
SectorRiskDriver.model_rebuild()
HoldingConcernContribution.model_rebuild()
HoldingRiskContribution.model_rebuild()
HoldingActionNeed.model_rebuild()
HoldingActionRecommendation.model_rebuild()
PortfolioActionImpact.model_rebuild()
PortfolioTraitSnapshot.model_rebuild()
RebalanceHoldingChange.model_rebuild()
RebalanceSectorChange.model_rebuild()
PortfolioRebalancePlan.model_rebuild()
PortfolioGap.model_rebuild()
PortfolioPreferences.model_rebuild()
ReplacementCandidate.model_rebuild()
CurrentHoldingSnapshot.model_rebuild()


def _load_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file from disk and return the parsed dictionary.

    The diagnosis pipeline is intentionally file-backed right now. The app writes
    analysis artifacts to disk, and this module reconstructs typed diagnosis
    objects from those persisted snapshots. This helper keeps that boundary
    explicit and centralized.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV into a DataFrame, returning an empty frame when absent.

    Missing files should not crash the diagnosis build immediately because some
    enrichment layers are optional. Returning an empty frame allows the object
    builder to degrade gracefully and describe coverage gaps explicitly.
    """
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _severity_band(score: Optional[float]) -> str:
    """Map a numeric concern score onto a human-readable severity label.

    The diagnosis layer uses the same broad score language throughout the object
    model so that concern ranking, summaries, and later UI presentation all speak
    in a consistent vocabulary.
    """
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
    """Coerce loosely typed source values into floats when possible.

    The diagnosis object reads from JSON, CSV, and dataframe-derived values, so a
    single metric may arrive as a string, number, NaN, or missing value. This
    helper standardizes those cases before they are used in scoring logic.
    """
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _driver_confidence_band(score: float) -> str:
    """Translate a driver-level confidence score into a human-readable band.

    Driver confidence is intentionally separate from the top-level diagnosis
    confidence. A portfolio diagnosis may be strong overall while one specific
    holding driver is only weakly supported.
    """
    if score < 0.35:
        return "Low"
    if score < 0.7:
        return "Medium"
    return "High"


def _sort_reason_codes(reason_codes: list[ReasonCode]) -> list[ReasonCode]:
    """Sort reason codes from strongest to weakest.

    We rank primarily by severity score and secondarily by label so downstream
    UI and notebook review produce stable ordering.
    """
    return sorted(reason_codes, key=lambda item: (-item.severity_score, item.label))


def load_diagnosis_bundle(base_dir: Path) -> dict[str, Any]:
    """Load the full diagnosis input bundle from processed app and pipeline data.

    Parameters
    ----------
    base_dir:
        Directory containing the app-generated diagnosis artifacts such as
        `risk_score.json`, `headline_metrics.json`, and the saved CSV tables.

    Returns
    -------
    dict[str, Any]
        A dictionary containing both:
        - app-side analysis outputs used for the initial diagnosis
        - external structured and unstructured datasets used for enrichment

    Notes
    -----
    This is the bridge between the app layer and the external data pipeline.
    The app contributes the portfolio-specific measurement layer, while the
    processed external datasets contribute macro, fundamental, and narrative
    context.
    """
    processed_dir = base_dir.parent
    return {
        "manifest": _load_json(base_dir / "diagnosis_manifest.json"),
        "headline": _load_json(base_dir / "headline_metrics.json"),
        "risk": _load_json(base_dir / "risk_score.json"),
        "portfolio_summary": _load_json(base_dir / "portfolio_summary.json"),
        "risk_metrics": _load_csv(base_dir / "risk_metric_scores.csv"),
        "risk_dimensions": _load_csv(base_dir / "risk_dimension_scores.csv"),
        "open_positions": _load_csv(base_dir / "open_positions.csv"),
        "holding_performance_context": _load_csv(base_dir / "holding_performance_context.csv"),
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
    """Convert saved risk metric tables into typed metric observation objects.

    These observations are the lowest-level diagnostic evidence items. They keep
    the original metric identity, score, raw value, and plain-English meaning so
    later layers can explain *why* a concern exists rather than just ranking it.
    """
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


def _narrative_evidence_by_ticker(narrative_evidence: list["NarrativeEvidence"]) -> dict[str, list["NarrativeEvidence"]]:
    """Group narrative evidence by ticker for easier driver-level reasoning."""
    grouped: dict[str, list[NarrativeEvidence]] = {}
    for item in narrative_evidence:
        grouped.setdefault(item.ticker, []).append(item)
    return grouped


def _build_holding_reason_codes(
    *,
    row: dict[str, Any],
    macro_context: Optional["MacroRegimeSnapshot"],
    fundamentals: Optional["CompanyFundamentalSnapshot"],
    narrative_items: list["NarrativeEvidence"],
) -> list[ReasonCode]:
    """Build structured reason codes for one holding driver.

    This is the core of the holding-level reason-code framework. It translates
    raw holding measurements plus external evidence into explicit reasons that
    later layers can rank, display, and explain.
    """
    reason_codes: list[ReasonCode] = []
    ticker = str(row.get("ticker"))
    current_weight = _safe_float(row.get("current_weight")) or 0.0
    variance_contribution_pct = _safe_float(row.get("variance_contribution_pct")) or 0.0
    excess_return_vs_benchmark = _safe_float(row.get("excess_return_vs_benchmark"))

    if current_weight >= 0.15:
        reason_codes.append(
            ReasonCode(
                code="concentration_pressure",
                label="Concentration pressure",
                category="positioning",
                severity_score=round(min(100.0, current_weight * 350), 1),
                summary=f"{ticker} is large enough to materially influence the whole account on its own.",
                evidence=[f"Current weight: {current_weight:.1%}"],
            )
        )
    elif current_weight >= 0.05:
        reason_codes.append(
            ReasonCode(
                code="meaningful_position",
                label="Meaningful position size",
                category="positioning",
                severity_score=round(min(100.0, current_weight * 220), 1),
                summary=f"{ticker} still carries enough capital to matter for portfolio outcomes.",
                evidence=[f"Current weight: {current_weight:.1%}"],
            )
        )

    if variance_contribution_pct >= 0.08:
        reason_codes.append(
            ReasonCode(
                code="volatility_contributor",
                label="Volatility contributor",
                category="market_behavior",
                severity_score=round(min(100.0, variance_contribution_pct * 300), 1),
                summary=f"{ticker} has been one of the biggest contributors to recent portfolio swings.",
                evidence=[f"Variance contribution: {variance_contribution_pct:.1%}"],
            )
        )
    elif variance_contribution_pct >= 0.03:
        reason_codes.append(
            ReasonCode(
                code="volatility_supporting_driver",
                label="Noticeable volatility contributor",
                category="market_behavior",
                severity_score=round(min(100.0, variance_contribution_pct * 220), 1),
                summary=f"{ticker} has contributed meaningfully to recent volatility even if it is not the top source of it.",
                evidence=[f"Variance contribution: {variance_contribution_pct:.1%}"],
            )
        )

    if excess_return_vs_benchmark is not None and excess_return_vs_benchmark < 0:
        reason_codes.append(
            ReasonCode(
                code="benchmark_lag",
                label="Lagging benchmark",
                category="performance",
                severity_score=round(min(100.0, abs(excess_return_vs_benchmark) * 100), 1),
                summary=f"{ticker} has underperformed the trade-matched S&P 500 since it entered the portfolio.",
                evidence=[f"Excess return vs benchmark: {excess_return_vs_benchmark:.1%}"],
            )
        )

    if fundamentals:
        if fundamentals.beta is not None and fundamentals.beta > 1.2:
            reason_codes.append(
                ReasonCode(
                    code="high_beta",
                    label="High beta",
                    category="fundamentals",
                    severity_score=round(min(100.0, (fundamentals.beta - 1.0) * 45), 1),
                    summary=f"{ticker} tends to move more than the market, which can amplify portfolio swings.",
                    evidence=[f"Beta: {fundamentals.beta:.2f}"],
                )
            )
        if "latest net income negative" in fundamentals.signals:
            reason_codes.append(
                ReasonCode(
                    code="negative_earnings",
                    label="Negative earnings",
                    category="fundamentals",
                    severity_score=62.0,
                    summary=f"{ticker}'s latest fundamental snapshot shows negative net income.",
                    evidence=["Latest net income was below zero"],
                )
            )
        if "liabilities are a large share of assets" in fundamentals.signals:
            reason_codes.append(
                ReasonCode(
                    code="balance_sheet_stretch",
                    label="Balance-sheet stretch",
                    category="fundamentals",
                    severity_score=58.0,
                    summary=f"{ticker}'s balance sheet looks more stretched than steadier holdings in the portfolio.",
                    evidence=["Liabilities are a large share of assets"],
                )
            )

    if narrative_items:
        narrative_titles = [item.title for item in narrative_items if item.title]
        reason_codes.append(
            ReasonCode(
                code="narrative_risk",
                label="Narrative or event risk",
                category="external_evidence",
                severity_score=float(min(80, 35 + (len(narrative_items) * 10))),
                summary=f"Recent filings or news around {ticker} contain risk-relevant language worth monitoring.",
                evidence=narrative_titles[:2] or ["Recent filing/news evidence available"],
            )
        )

    if macro_context and "rates still restrictive" in macro_context.regime_flags:
        sector = str(row.get("sector") or "")
        if sector in {"Technology", "Consumer Cyclical", "Communication Services"}:
            reason_codes.append(
                ReasonCode(
                    code="restrictive_rates_sensitivity",
                    label="Restrictive-rate sensitivity",
                    category="macro",
                    severity_score=42.0,
                    summary=f"{ticker} sits in a part of the market that can be more sensitive when rates stay restrictive.",
                    evidence=[f"Macro flag: {macro_context.regime_flags[0]}"],
                )
            )

    return _sort_reason_codes(reason_codes)


def _driver_confidence_from_reason_codes(reason_codes: list[ReasonCode]) -> tuple[float, str]:
    """Estimate holding/sector-driver confidence from independent evidence types."""
    if not reason_codes:
        return 0.0, "Low"
    categories = {item.category for item in reason_codes}
    evidence_bonus = 1 if any(item.evidence for item in reason_codes) else 0
    score = min(1.0, 0.2 + len(categories) * 0.22 + evidence_bonus * 0.12)
    return round(score, 2), _driver_confidence_band(score)


def _concern_contribution_band(score: float) -> str:
    """Translate a holding-level contribution score into a readable band."""
    if score < 20:
        return "Low contribution"
    if score < 40:
        return "Moderate contribution"
    if score < 60:
        return "Meaningful contribution"
    if score < 80:
        return "High contribution"
    return "Very high contribution"


def _holding_concern_label(concern_key: str) -> str:
    """Map internal concern keys to stable human-readable labels."""
    labels = {
        "concentration": "Concentration risk",
        "market": "Market-relative risk",
        "behavior": "Behavioral risk",
        "company_specific": "Company-specific risk",
        "macro": "Macro sensitivity",
    }
    return labels.get(concern_key, concern_key.replace("_", " ").title())


def _action_pressure_band(score: float) -> str:
    """Translate an action-pressure score into a user-reviewable band."""
    if score < 20:
        return "Low action pressure"
    if score < 40:
        return "Watchlist pressure"
    if score < 60:
        return "Moderate action pressure"
    if score < 80:
        return "High action pressure"
    return "Very high action pressure"


def _reason_code_concern_weights(reason_code: ReasonCode) -> list[tuple[str, float]]:
    """Map one reason code onto the concern families it most naturally supports.

    The contribution layer needs to know which concern(s) each reason code
    should strengthen. These mappings are intentionally explicit and heuristic:
    they keep the system inspectable while we are still learning the domain.
    """
    mapping = {
        "concentration_pressure": [("concentration", 1.0)],
        "meaningful_position": [("concentration", 0.65)],
        "volatility_contributor": [("market", 1.0)],
        "volatility_supporting_driver": [("market", 0.7)],
        "benchmark_lag": [("company_specific", 0.7), ("market", 0.3)],
        "high_beta": [("market", 0.8), ("macro", 0.2)],
        "negative_earnings": [("company_specific", 1.0)],
        "balance_sheet_stretch": [("company_specific", 1.0)],
        "narrative_risk": [("company_specific", 0.75), ("market", 0.25)],
        "restrictive_rates_sensitivity": [("macro", 1.0)],
    }
    return mapping.get(reason_code.code, [("company_specific", 0.5)])


def _action_urgency_label(score: float) -> str:
    """Map an action-pressure score to an urgency phrase."""
    if score < 30:
        return "No immediate action"
    if score < 55:
        return "Monitor next"
    if score < 75:
        return "Action worth considering soon"
    return "Action likely warranted soon"


def _unique_nonempty(values: Iterable[Any]) -> list[str]:
    """Return non-empty values once, preserving their original order.

    The review notebook and the dashboard read much more clearly when repeated
    reasons or evidence cues are collapsed. We keep this helper local to the
    diagnosis module so later action-layer objects can stay concise without
    losing the original ordering of signals.
    """
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in {None, ""}:
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        unique_values.append(text)
    return unique_values


def _candidate_driver_rows(bundle: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    """Build a small candidate set of holding rows for downstream diagnosis work.

    The diagnosis engine should not depend on fully materialized dataframes in
    later layers. This helper gives us a stable object-friendly intermediate:
    one merged record per holding with a rough priority score based on position
    size, volatility contribution, and benchmark lag.

    Parameters
    ----------
    bundle:
        Full saved diagnosis bundle from the app and processed datasets.
    limit:
        Maximum number of candidate holding rows to return.

    Returns
    -------
    list[dict[str, Any]]
        Ranked holding records ready to feed fundamental enrichment, narrative
        enrichment, and the final `HoldingRiskDriver` objects.
    """
    positions = bundle["open_positions"].copy()
    volatility = bundle["volatility_drivers"].copy()
    if positions.empty:
        return []

    if not volatility.empty:
        volatility = volatility.rename(columns={"avg_weight_pct": "avg_weight_pct_2025"})
    merged = positions.merge(volatility, on="ticker", how="left")

    merged["current_weight"] = pd.to_numeric(merged.get("current_weight"), errors="coerce").fillna(0.0)
    merged["variance_contribution_pct"] = pd.to_numeric(
        merged.get("variance_contribution_pct"), errors="coerce"
    ).fillna(0.0)
    merged["excess_return_vs_benchmark"] = pd.to_numeric(
        merged.get("excess_return_vs_benchmark"), errors="coerce"
    ).fillna(0.0)

    # A light-weight preselection score. This is not the final diagnosis score;
    # it simply helps us decide which holdings deserve deeper enrichment first.
    merged["candidate_driver_score"] = (
        merged["current_weight"] * 140.0
        + merged["variance_contribution_pct"] * 110.0
        + (-merged["excess_return_vs_benchmark"]).clip(lower=0.0) * 22.0
    )
    merged = merged.sort_values(
        ["candidate_driver_score", "current_weight", "variance_contribution_pct"],
        ascending=False,
    )
    return merged.head(limit).to_dict(orient="records")


def _build_holding_drivers(
    bundle: dict[str, Any],
    macro_context: Optional["MacroRegimeSnapshot"],
    holding_fundamentals: list["CompanyFundamentalSnapshot"],
    narrative_evidence: list["NarrativeEvidence"],
) -> list[HoldingRiskDriver]:
    """Identify and explain the holdings most responsible for portfolio fragility.

    This upgraded version moves from raw strings toward a true reason-code
    framework. Each holding driver now has:
    - explicit reason codes
    - a primary reason
    - secondary reasons
    - driver-level confidence
    - evidence summaries
    """
    candidate_rows = _candidate_driver_rows(bundle, limit=10)
    if not candidate_rows:
        return []
    fundamentals_by_ticker = {item.ticker: item for item in holding_fundamentals}
    narrative_by_ticker = _narrative_evidence_by_ticker(narrative_evidence)

    driver_rows: list[HoldingRiskDriver] = []
    for row in candidate_rows:
        ticker = str(row.get("ticker"))
        reason_codes = _build_holding_reason_codes(
            row=row,
            macro_context=macro_context,
            fundamentals=fundamentals_by_ticker.get(ticker),
            narrative_items=narrative_by_ticker.get(ticker, []),
        )
        if not reason_codes:
            continue
        primary = reason_codes[0]
        secondary = reason_codes[1:3]
        confidence_score, confidence_band = _driver_confidence_from_reason_codes(reason_codes)
        evidence_summary = [evidence for code in reason_codes[:3] for evidence in code.evidence[:2]]
        driver_importance_score = round(
            sum(code.severity_score for code in reason_codes[:3]) + ((confidence_score or 0.0) * 20.0),
            1,
        )
        driver_rows.append(
            HoldingRiskDriver(
                ticker=ticker,
                sector=row.get("sector"),
                current_weight=_safe_float(row.get("current_weight")),
                excess_return_vs_benchmark=_safe_float(row.get("excess_return_vs_benchmark")),
                variance_contribution_pct=_safe_float(row.get("variance_contribution_pct")),
                driver_reasons=[code.label for code in reason_codes],
                primary_reason_code=primary.code,
                primary_reason_label=primary.label,
                primary_reason_summary=primary.summary,
                secondary_reason_codes=[code.code for code in secondary],
                driver_confidence_score=confidence_score,
                driver_confidence_band=confidence_band,
                evidence_summary=evidence_summary,
                reason_codes=[code.model_dump() for code in reason_codes],
            )
        )

    if not driver_rows:
        for row in candidate_rows[:5]:
            ticker = str(row.get("ticker"))
            fallback_code = ReasonCode(
                code="largest_remaining_position",
                label="Largest remaining position",
                category="positioning",
                severity_score=50.0,
                summary=f"{ticker} is one of the portfolio's largest remaining positions.",
                evidence=[f"Current weight: {(_safe_float(row.get('current_weight')) or 0.0):.1%}"],
            )
            driver_rows.append(
                HoldingRiskDriver(
                    ticker=ticker,
                    sector=row.get("sector"),
                    current_weight=_safe_float(row.get("current_weight")),
                    excess_return_vs_benchmark=_safe_float(row.get("excess_return_vs_benchmark")),
                    variance_contribution_pct=_safe_float(row.get("variance_contribution_pct")),
                    driver_reasons=[fallback_code.label],
                    primary_reason_code=fallback_code.code,
                    primary_reason_label=fallback_code.label,
                    primary_reason_summary=fallback_code.summary,
                    secondary_reason_codes=[],
                    driver_confidence_score=0.3,
                    driver_confidence_band="Low",
                    evidence_summary=fallback_code.evidence,
                    reason_codes=[fallback_code.model_dump()],
                )
            )

    driver_rows.sort(
        key=lambda item: (
            -(sum(code.severity_score for code in item.reason_codes[:3]) + item.driver_confidence_score * 20.0),
            -(item.current_weight or 0.0),
            item.ticker,
        )
    )
    return driver_rows[:5]


def _latest_metric_value(
    company_facts: pd.DataFrame,
    ticker: str,
    canonical_metric: str,
) -> tuple[Optional[float], Optional[str]]:
    """Fetch the latest available SEC company-facts value for one canonical metric.

    Parameters
    ----------
    company_facts:
        Flattened SEC company facts dataset.
    ticker:
        The security whose fundamental series should be queried.
    canonical_metric:
        Normalized metric name from the fact-tag mapping layer, such as
        `revenue`, `net_income`, or `operating_cash_flow`.

    Returns
    -------
    tuple[Optional[float], Optional[str]]
        The latest numeric value and its filed date, if a usable observation
        exists.

    Notes
    -----
    This helper intentionally works on the canonical metric layer rather than raw
    SEC tags. That keeps the diagnosis object aligned with the business-friendly
    taxonomy we created in the data review notebook.
    """
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
    """Summarize the macro environment into a typed regime snapshot.

    This is the diagnosis layer's first answer to the question:
    "What does the broader economic weather look like right now?"

    The logic currently uses a compact set of macro series:
    - Fed funds rate
    - CPI inflation
    - unemployment
    - 10Y Treasury
    - 2Y Treasury

    From those series it derives a short regime summary and lightweight flags
    that can influence risk interpretation.
    """
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
    tickers: list[str],
) -> list[CompanyFundamentalSnapshot]:
    """Build fundamental snapshots for the holdings driving the diagnosis.

    The diagnosis engine does not need the full company-facts universe on every
    object. Instead, it narrows the fundamental layer to the holdings already
    identified as important drivers, then assembles a compact snapshot with:
    - identity data from company profiles
    - core fundamentals from SEC company facts
    - a few descriptive signals for quick interpretation
    """
    company_facts = bundle["company_facts"].copy()
    profiles = bundle["company_profiles"].copy()
    if company_facts.empty and profiles.empty:
        return []

    snapshots: list[CompanyFundamentalSnapshot] = []
    for ticker in tickers:
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
    """Normalize raw text into a short review-friendly evidence snippet.

    Diagnosis objects should carry compact evidence, not huge unbounded text
    blobs. This helper strips whitespace clutter and trims text to a length that
    is realistic for debugging, review, and later UI display.
    """
    snippet = str(text or "").replace("\n", " ").replace("\r", " ").strip()
    snippet = " ".join(snippet.split())
    return snippet[:limit]


def _build_narrative_evidence(
    bundle: dict[str, Any],
    tickers: list[str],
) -> list[NarrativeEvidence]:
    """Attach recent filing/news evidence for the main holding drivers.

    For each important holding, the diagnosis engine tries to keep one recent SEC
    filing artifact and one recent news artifact. This does not yet score the
    text deeply, but it gives the diagnosis object concrete external evidence that
    can be surfaced in later reasoning and UI layers.
    """
    corpus = bundle["document_corpus"].copy()
    if corpus.empty:
        return []
    evidence: list[NarrativeEvidence] = []
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


def _narrative_risk_map(narrative_evidence: list[NarrativeEvidence]) -> dict[str, int]:
    """Create a simple ticker-level narrative risk count from attached evidence.

    The current version uses a keyword-based heuristic as a bridge toward a more
    sophisticated narrative scoring system. The goal is not to produce a final
    NLP judgment yet, but to let external text influence the diagnosis ranking in
    a transparent, reviewable way.
    """
    risk_counts: dict[str, int] = {}
    for item in narrative_evidence:
        text = f"{item.title or ''} {item.snippet}".lower()
        matches = sum(1 for keyword in NEGATIVE_NARRATIVE_KEYWORDS if keyword in text)
        if matches > 0:
            risk_counts[item.ticker] = risk_counts.get(item.ticker, 0) + matches
    return risk_counts


def _build_sector_drivers(bundle: dict[str, Any]) -> list[SectorRiskDriver]:
    """Identify sectors that appear to be amplifying portfolio risk.

    Sector drivers use the same reason-code pattern as holding drivers, just at a
    higher level of abstraction. The goal is to make sector crowding explainable
    with the same primitives:
    - explicit reason codes
    - one primary reason
    - driver-level confidence
    """
    sectors = bundle["sector_allocation"].copy()
    if sectors.empty:
        return []

    drivers: list[SectorRiskDriver] = []
    for row in sectors.sort_values("weight_pct", ascending=False).to_dict(orient="records"):
        reason_codes: list[ReasonCode] = []
        weight_pct = _safe_float(row.get("weight_pct"))
        excess_return_vs_benchmark = _safe_float(row.get("excess_return_vs_benchmark"))
        if weight_pct is not None and weight_pct >= 0.25:
            reason_codes.append(
                ReasonCode(
                    code="sector_crowding",
                    label="Sector crowding",
                    category="positioning",
                    severity_score=round(min(100.0, weight_pct * 220), 1),
                    summary=f"{row.get('sector')} represents a large enough share of capital to shape portfolio behavior.",
                    evidence=[f"Sector weight: {weight_pct:.1%}"],
                )
            )
        if excess_return_vs_benchmark is not None and excess_return_vs_benchmark < 0:
            reason_codes.append(
                ReasonCode(
                    code="sector_benchmark_lag",
                    label="Sector benchmark lag",
                    category="performance",
                    severity_score=round(min(100.0, abs(excess_return_vs_benchmark) * 100), 1),
                    summary=f"{row.get('sector')} has lagged the trade-matched benchmark inside this portfolio.",
                    evidence=[f"Excess return vs benchmark: {excess_return_vs_benchmark:.1%}"],
                )
            )
        if reason_codes:
            reason_codes = _sort_reason_codes(reason_codes)
            confidence_score, confidence_band = _driver_confidence_from_reason_codes(reason_codes)
            primary = reason_codes[0]
            drivers.append(
                SectorRiskDriver(
                    sector=str(row.get("sector")),
                    weight_pct=weight_pct,
                    excess_return_vs_benchmark=excess_return_vs_benchmark,
                    driver_reasons=[code.label for code in reason_codes],
                    primary_reason_code=primary.code,
                    primary_reason_label=primary.label,
                    primary_reason_summary=primary.summary,
                    driver_confidence_score=confidence_score,
                    driver_confidence_band=confidence_band,
                    reason_codes=[code.model_dump() for code in reason_codes],
                )
            )
    return drivers[:3]


def _concern_summary(concern_key: str, bundle: dict[str, Any]) -> str:
    """Generate the base plain-English explanation for a diagnosis concern.

    This summary is intentionally tied to the app-side measurement layer first.
    External evidence can then *adjust* and *reinforce* the concern, but the
    summary starts from the concrete portfolio metrics already computed by the
    dashboard.
    """
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


def _concern_adjustments(
    concern_key: str,
    bundle: dict[str, Any],
    top_holding_drivers: list[HoldingRiskDriver],
    top_sector_drivers: list[SectorRiskDriver],
    macro_context: Optional[MacroRegimeSnapshot],
    holding_fundamentals: list[CompanyFundamentalSnapshot],
    narrative_evidence: list[NarrativeEvidence],
) -> tuple[float, list[str]]:
    """Adjust concern severity using external macro, fundamental, and narrative evidence.

    Parameters
    ----------
    concern_key:
        Concern currently being evaluated, such as `concentration` or `market`.
    bundle:
        Complete diagnosis input bundle.
    top_holding_drivers:
        Holdings already identified as central portfolio drivers.
    top_sector_drivers:
        Sector-level drivers identified from the app bundle.
    macro_context:
        Macro regime snapshot built from FRED series.
    holding_fundamentals:
        Fundamental snapshots for the key holdings.
    narrative_evidence:
        Filing and news evidence attached to the key holdings.

    Returns
    -------
    tuple[float, list[str]]
        A numeric adjustment and a list of human-readable reasons explaining why
        the adjustment was applied.

    Notes
    -----
    This is the heart of the "enriched diagnosis" idea. The app risk score gives
    us the base portfolio measurement. This function lets the outside world push
    that diagnosis up or down in a reviewable way.
    """
    adjustment = 0.0
    reasons: list[str] = []
    narrative_risk = _narrative_risk_map(narrative_evidence)
    fundamentals_by_ticker = {item.ticker: item for item in holding_fundamentals}

    if concern_key == "concentration":
        top_driver = top_holding_drivers[0] if top_holding_drivers else None
        if top_driver and (top_driver.current_weight or 0.0) >= 0.5:
            adjustment += 8.0
            reasons.append("largest holding still dominates after external review")
            top_fundamental = fundamentals_by_ticker.get(top_driver.ticker)
            if top_fundamental and top_fundamental.beta and top_fundamental.beta > 1.1:
                adjustment += 3.0
                reasons.append(f"{top_driver.ticker} also carries above-market beta")
            if narrative_risk.get(top_driver.ticker, 0) > 0:
                adjustment += 3.0
                reasons.append(f"{top_driver.ticker} also has risk-labeled narrative evidence")
        if top_sector_drivers and (top_sector_drivers[0].weight_pct or 0.0) >= 0.5:
            adjustment += 4.0
            reasons.append("one sector dominates the portfolio along with the top holding")

    elif concern_key == "market":
        if macro_context:
            if "rates still restrictive" in macro_context.regime_flags:
                adjustment += 3.0
                reasons.append("macro backdrop still has restrictive rates")
            if "inflation still sticky" in macro_context.regime_flags:
                adjustment += 2.0
                reasons.append("sticky inflation can pressure richly priced risk assets")
        high_beta_count = sum(1 for item in holding_fundamentals if (item.beta or 0.0) > 1.2)
        if high_beta_count >= 2:
            adjustment += 4.0
            reasons.append("multiple main drivers still have above-market beta")
        risky_narrative_names = [ticker for ticker, count in narrative_risk.items() if count > 0]
        if risky_narrative_names:
            adjustment += 3.0
            reasons.append("external narrative evidence reinforces market sensitivity concerns")

    elif concern_key == "behavior":
        lagging_driver_count = sum(
            1
            for driver in top_holding_drivers
            if (driver.excess_return_vs_benchmark or 0.0) < 0
        )
        if lagging_driver_count >= 3:
            adjustment += 1.5
            reasons.append("several top drivers are lagging benchmark despite low trading churn")
        if narrative_risk:
            adjustment += 1.0
            reasons.append("behavior should still be interpreted in light of evolving company news")

    return adjustment, reasons


def _build_top_concerns(
    bundle: dict[str, Any],
    top_holding_drivers: list[HoldingRiskDriver],
    top_sector_drivers: list[SectorRiskDriver],
    macro_context: Optional[MacroRegimeSnapshot],
    holding_fundamentals: list[CompanyFundamentalSnapshot],
    narrative_evidence: list[NarrativeEvidence],
) -> list[DiagnosisConcern]:
    """Construct and rank the portfolio's top diagnosis concerns.

    This function combines:
    - base dimension scores from the app bundle
    - metric keys supporting each concern
    - external evidence adjustments
    - related holdings and sectors

    The output is the first real ranked diagnosis layer, which is why this
    function matters so much for the overall architecture.
    """
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
    for concern_key, label, base_score, metric_keys in concern_configs:
        if base_score <= 0:
            continue
        external_adjustment_score, adjustment_reasons = _concern_adjustments(
            concern_key,
            bundle,
            top_holding_drivers,
            top_sector_drivers,
            macro_context,
            holding_fundamentals,
            narrative_evidence,
        )
        severity_score = min(100.0, round(base_score + external_adjustment_score, 1))
        concerns.append(
            DiagnosisConcern(
                concern_key=concern_key,
                label=label,
                base_severity_score=round(base_score, 1),
                external_adjustment_score=round(external_adjustment_score, 1),
                severity_score=severity_score,
                severity_band=_severity_band(severity_score),
                summary=_concern_summary(concern_key, bundle),
                adjustment_reasons=adjustment_reasons,
                evidence_metric_keys=metric_keys,
                related_tickers=related_tickers[:3] if concern_key != "behavior" else [],
                related_sectors=related_sectors[:2] if concern_key in {"concentration", "market"} else [],
            )
        )
    concerns.sort(key=lambda concern: concern.severity_score, reverse=True)
    return concerns


def _build_holding_risk_contributions(
    top_holding_drivers: list[HoldingRiskDriver],
    top_concerns: list[DiagnosisConcern],
) -> list[HoldingRiskContribution]:
    """Convert holding drivers into concern-specific contribution objects.

    This function is the next layer after `HoldingRiskDriver`. It answers:

    - which concern(s) a holding contributes to
    - how strongly it contributes to each one
    - which reason codes support that read
    - how confident we are in the contribution view

    The output is designed to feed both notebook review and later action logic.
    """
    concern_score_lookup = {
        concern.concern_key: concern.severity_score for concern in top_concerns
    }
    contribution_objects: list[HoldingRiskContribution] = []

    for driver in top_holding_drivers:
        contribution_rows: list[HoldingConcernContribution] = []
        reason_codes = list(driver.reason_codes or [])
        grouped_codes: dict[str, list[ReasonCode]] = {}
        weighted_scores: dict[str, float] = {}

        for reason_code in reason_codes:
            for concern_key, weight in _reason_code_concern_weights(reason_code):
                grouped_codes.setdefault(concern_key, []).append(reason_code)
                weighted_scores[concern_key] = weighted_scores.get(concern_key, 0.0) + (reason_code.severity_score * weight)

        if not weighted_scores:
            continue

        for concern_key, raw_score in weighted_scores.items():
            concern_multiplier = (concern_score_lookup.get(concern_key, 55.0)) / 100.0
            contribution_score = min(100.0, round(raw_score * concern_multiplier, 1))
            supporting_codes = _sort_reason_codes(grouped_codes.get(concern_key, []))
            supporting_code_keys = [item.code for item in supporting_codes]
            supporting_evidence = [
                evidence
                for item in supporting_codes[:3]
                for evidence in item.evidence[:2]
                if evidence
            ]
            confidence_score = min(
                1.0,
                round(driver.driver_confidence_score * (0.72 + 0.08 * len(set(supporting_code_keys))), 2),
            )
            confidence_band = _driver_confidence_band(confidence_score)
            lead_label = supporting_codes[0].label if supporting_codes else "supporting evidence"
            contribution_rows.append(
                HoldingConcernContribution(
                    concern_key=concern_key,
                    concern_label=_holding_concern_label(concern_key),
                    contribution_score=contribution_score,
                    contribution_band=_concern_contribution_band(contribution_score),
                    contribution_summary=(
                        f"{driver.ticker} mainly contributes to {_holding_concern_label(concern_key).lower()} "
                        f"through {lead_label.lower()}."
                    ),
                    supporting_reason_codes=supporting_code_keys,
                    supporting_evidence=supporting_evidence[:4],
                    confidence_score=confidence_score,
                    confidence_band=confidence_band,
                )
            )

        contribution_rows.sort(key=lambda item: item.contribution_score, reverse=True)
        primary_contribution = contribution_rows[0]
        overall_contribution_score = min(
            100.0,
            round(
                primary_contribution.contribution_score
                + max(0.0, sum(item.contribution_score for item in contribution_rows[1:3]) * 0.2),
                1,
            ),
        )
        overall_confidence = min(
            1.0,
            round(sum(item.confidence_score for item in contribution_rows[:2]) / max(1, len(contribution_rows[:2])), 2),
        )
        contribution_objects.append(
            HoldingRiskContribution(
                ticker=driver.ticker,
                sector=driver.sector,
                current_weight=driver.current_weight,
                overall_contribution_score=overall_contribution_score,
                overall_contribution_band=_concern_contribution_band(overall_contribution_score),
                primary_concern_key=primary_contribution.concern_key,
                primary_concern_label=primary_contribution.concern_label,
                primary_concern_summary=primary_contribution.contribution_summary,
                contribution_confidence_score=overall_confidence,
                contribution_confidence_band=_driver_confidence_band(overall_confidence),
                contribution_summary=(
                    f"{driver.ticker} is primarily a {_holding_concern_label(primary_contribution.concern_key).lower()} "
                    f"driver, with secondary spillover into "
                    + ", ".join(item.concern_label.lower() for item in contribution_rows[1:3])
                    + "."
                    if len(contribution_rows) > 1
                    else f"{driver.ticker} is primarily a {_holding_concern_label(primary_contribution.concern_key).lower()} driver."
                ),
                concern_contributions=contribution_rows,
                evidence_summary=driver.evidence_summary[:4],
            )
        )

    contribution_objects.sort(key=lambda item: item.overall_contribution_score, reverse=True)
    return contribution_objects


def _determine_action_label(
    contribution: HoldingRiskContribution,
) -> tuple[str, str, str]:
    """Choose the first action label from current contribution evidence.

    The output is:
    - user-facing action label
    - stable machine code
    - short explanation of why that action family was chosen
    """
    primary = contribution.primary_concern_key
    score = contribution.overall_contribution_score
    weight = contribution.current_weight or 0.0

    if primary == "concentration" and (weight >= 0.25 or score >= 80):
        return (
            "Reduce exposure",
            "reduce_exposure",
            "the holding is large enough that reducing size would directly lower portfolio concentration",
        )
    if primary in {"market", "macro"} and score >= 60:
        return (
            "Trim and monitor",
            "trim_and_monitor",
            "the holding is feeding market-sensitive risk strongly enough that trimming is worth considering",
        )
    if primary == "company_specific" and score >= 35:
        return (
            "Monitor closely",
            "monitor_closely",
            "the holding carries company-specific pressure, but the evidence is not yet strong enough to force a reduction call",
        )
    if score >= 35:
        return (
            "Monitor closely",
            "monitor_closely",
            "the holding contributes enough risk to stay on the watchlist even if no immediate trade is required",
        )
    return (
        "Hold steady",
        "hold_steady",
        "the holding is part of the diagnosis, but current contribution pressure does not yet justify an action call",
    )


def _build_holding_action_needs(
    holding_risk_contributions: list[HoldingRiskContribution],
) -> list[HoldingActionNeed]:
    """Turn contribution objects into the first concrete action layer.

    `HoldingActionNeed` is intentionally more conservative than a full sell
    engine. It answers:

    - does this holding merely need monitoring?
    - is trimming reasonable to consider?
    - is reduction pressure already high?

    It does *not* yet choose replacements or compute rebalance budgets.
    """
    action_needs: list[HoldingActionNeed] = []
    for contribution in holding_risk_contributions:
        pressure_score = min(
            100.0,
            round(
                contribution.overall_contribution_score * 0.82
                + ((contribution.current_weight or 0.0) * 100.0 * 0.35),
                1,
            ),
        )
        action_label, action_code, primary_reason = _determine_action_label(contribution)
        urgency = _action_urgency_label(pressure_score)
        supporting_concerns = _unique_nonempty(
            item.concern_label for item in contribution.concern_contributions[1:3]
        )
        supporting_reason_codes = _unique_nonempty(
            code
            for item in contribution.concern_contributions[:3]
            for code in item.supporting_reason_codes[:2]
        )
        supporting_evidence = _unique_nonempty(
            evidence
            for item in contribution.concern_contributions[:3]
            for evidence in item.supporting_evidence[:2]
        )
        action_needs.append(
            HoldingActionNeed(
                ticker=contribution.ticker,
                sector=contribution.sector,
                current_weight=contribution.current_weight,
                action_label=action_label,
                action_code=action_code,
                action_pressure_score=pressure_score,
                action_pressure_band=_action_pressure_band(pressure_score),
                action_urgency=urgency,
                primary_action_reason=primary_reason,
                action_summary=(
                    f"{contribution.ticker} currently looks like a **{action_label.lower()}** name because "
                    f"its strongest contribution is to {contribution.primary_concern_label.lower()} and its "
                    f"overall contribution score is {contribution.overall_contribution_score:.1f}/100. "
                    f"Its current action pressure is {pressure_score:.1f}/100, which falls into "
                    f"**{_action_pressure_band(pressure_score).lower()}**."
                ),
                linked_primary_concern=contribution.primary_concern_label,
                supporting_concerns=supporting_concerns,
                supporting_reason_codes=supporting_reason_codes,
                supporting_evidence=supporting_evidence[:5],
                confidence_score=contribution.contribution_confidence_score,
                confidence_band=contribution.contribution_confidence_band,
            )
        )
    action_needs.sort(key=lambda item: item.action_pressure_score, reverse=True)
    return action_needs


def _recommendation_confidence(
    *,
    holding_days: Optional[int],
    diagnosis_pressure_score: float,
    variance_contribution_pct: float,
    is_actionable: bool,
) -> tuple[float, str]:
    """Estimate confidence for a trim/sell recommendation.

    Recommendation confidence is intentionally tied to evidence quality rather
    than confidence theater. Longer holding periods, stronger diagnosis pressure,
    and measurable volatility contribution all make the recommendation easier to
    defend.
    """
    score = 0.35 if is_actionable else 0.25
    if holding_days is not None and holding_days >= 365:
        score += 0.2
    elif holding_days is not None and holding_days >= 180:
        score += 0.1
    if diagnosis_pressure_score >= 70:
        score += 0.2
    elif diagnosis_pressure_score >= 40:
        score += 0.12
    if variance_contribution_pct >= 0.03:
        score += 0.15
    elif variance_contribution_pct >= 0.01:
        score += 0.08
    score = min(0.95, round(score, 2))
    return score, _driver_confidence_band(score)


def _relative_window_readout(
    performance_row: dict[str, Any],
) -> tuple[int, int, list[str]]:
    """Summarize whether 1Y/3Y/5Y windows reinforce or soften the sell case."""
    underperform_count = 0
    outperform_count = 0
    evidence: list[str] = []
    for years in (1, 3, 5):
        relative = _safe_float(performance_row.get(f"relative_{years}y_return_pct"))
        if relative is None:
            continue
        if relative < 0:
            underperform_count += 1
            evidence.append(f"Trailing {years}Y vs S&P 500: {relative:.1%}")
        elif relative > 0:
            outperform_count += 1
            evidence.append(f"Trailing {years}Y vs S&P 500: +{relative:.1%}")
    return underperform_count, outperform_count, evidence


def _recommendation_reduction_from_underperformance(
    *,
    excess_return_vs_benchmark: float,
    current_weight: float,
    diagnosis_pressure_score: float,
    variance_contribution_pct: float,
) -> tuple[str, str, float]:
    """Choose a trim/sell action from benchmark underperformance.

    This function intentionally requires underperformance versus the S&P 500.
    Risk pressure can increase the trim size, but poor relative performance is
    the gating signal that makes a sell-style recommendation appropriate.
    """
    underperformance = abs(excess_return_vs_benchmark)

    if underperformance >= 1.0 and current_weight <= 0.03:
        return (
            "Sell all shares",
            "sell_all",
            1.0,
        )
    if underperformance >= 0.85:
        reduction = 0.5 if diagnosis_pressure_score >= 35 or variance_contribution_pct >= 0.01 else 0.35
        return ("Reduce by 50%" if reduction == 0.5 else "Trim 35%", "deep_underperformance_trim", reduction)
    if underperformance >= 0.65:
        reduction = 0.35 if diagnosis_pressure_score >= 35 or variance_contribution_pct >= 0.01 else 0.25
        return ("Trim 35%" if reduction == 0.35 else "Trim 25%", "heavy_underperformance_trim", reduction)
    if underperformance >= 0.45:
        reduction = 0.25 if diagnosis_pressure_score >= 30 or variance_contribution_pct >= 0.008 else 0.2 if diagnosis_pressure_score >= 20 or variance_contribution_pct >= 0.005 else 0.15
        if reduction == 0.25:
            return ("Trim 25%", "meaningful_underperformance_trim", reduction)
        return ("Trim 20%" if reduction == 0.2 else "Trim 15%", "meaningful_underperformance_trim", reduction)
    if underperformance >= 0.3:
        reduction = 0.2 if diagnosis_pressure_score >= 25 or variance_contribution_pct >= 0.006 else 0.15
        return ("Trim 20%" if reduction == 0.2 else "Trim 15%", "moderate_underperformance_trim", reduction)
    if underperformance >= 0.2:
        reduction = 0.15 if diagnosis_pressure_score >= 15 or variance_contribution_pct >= 0.004 else 0.10
        return ("Trim 15%" if reduction == 0.15 else "Trim 10%", "watchlist_trim", reduction)
    if underperformance >= 0.12:
        return ("Trim 10%", "small_but_weak_trim", 0.10)
    if underperformance >= 0.08 and (diagnosis_pressure_score >= 15 or variance_contribution_pct >= 0.004):
        return ("Trim 10%", "early_weakness_trim", 0.10)
    return ("Hold for now", "hold_for_now", 0.0)


def _recommendation_label_from_reduction(reduction_pct: float) -> str:
    """Map a normalized trim size back to the user-facing action label."""
    if reduction_pct >= 0.999:
        return "Sell all shares"
    if reduction_pct >= 0.5:
        return "Reduce by 50%"
    if reduction_pct >= 0.35:
        return "Trim 35%"
    if reduction_pct >= 0.25:
        return "Trim 25%"
    if reduction_pct >= 0.2:
        return "Trim 20%"
    if reduction_pct >= 0.15:
        return "Trim 15%"
    if reduction_pct >= 0.1:
        return "Trim 10%"
    return "Hold for now"


def _normalize_recommendation_reduction(reduction_pct: float) -> float:
    """Snap raw trim math to a small set of user-facing, explainable action sizes."""
    if reduction_pct >= 0.999:
        return 1.0
    if reduction_pct >= 0.5:
        return 0.5
    if reduction_pct >= 0.35:
        return 0.35
    if reduction_pct >= 0.25:
        return 0.25
    if reduction_pct >= 0.2:
        return 0.2
    if reduction_pct >= 0.15:
        return 0.15
    if reduction_pct >= 0.1:
        return 0.1
    return 0.0


def _build_explicit_sell_modifiers(
    *,
    action_need: Optional[HoldingActionNeed],
    linked_primary_concern: Optional[str],
) -> tuple[list[str], float]:
    """Translate external evidence into explicit sell modifiers.

    These modifiers are intentionally separate from the raw performance trigger.
    They help the system say, in plain language, what else in the market or in
    the company is making the underperformance harder to ignore.

    The score is deliberately small relative to the core benchmark-lag logic:
    external evidence can strengthen or tip a borderline case, but it should not
    replace observable market underperformance as the main trigger.
    """
    if action_need is None:
        return [], 0.0

    codes = set(action_need.supporting_reason_codes or [])
    evidence = list(action_need.supporting_evidence or [])
    modifiers: list[str] = []
    score = 0.0

    if "negative_earnings" in codes:
        modifiers.append("**Profits are under pressure** in the latest company filing, which makes weak market performance more concerning.")
        score += 0.10
    if "balance_sheet_stretch" in codes:
        modifiers.append("**The balance sheet looks more stretched** than steadier holdings, so the downside can be harder to absorb.")
        score += 0.06
    if "narrative_risk" in codes:
        headline = next((item for item in evidence if item and "Macro flag:" not in str(item) and "Beta:" not in str(item) and "Current weight:" not in str(item) and "Excess return vs benchmark:" not in str(item)), None)
        if headline:
            headline_text = str(headline)
            if "10-Q" in headline_text or "10-K" in headline_text:
                headline_text = "the latest company report"
            modifiers.append(f"**Recent company reports or news added caution** around this name, including {headline_text}.")
        else:
            modifiers.append("**Recent company reports or news added caution** around this holding.")
        score += 0.06
    if "restrictive_rates_sensitivity" in codes or linked_primary_concern == "Macro sensitivity":
        modifiers.append("**Higher interest rates are still a headwind** for this part of the market.")
        score += 0.05
    if "high_beta" in codes and linked_primary_concern in {"Market-relative risk", "Macro sensitivity"}:
        modifiers.append("**This stock tends to move more than the market**, which can make a weak trend hit the portfolio harder.")
        score += 0.05

    return _unique_nonempty(modifiers), round(score, 2)


def _should_sell_all_recommendation(
    *,
    excess_return_vs_benchmark: Optional[float],
    underperform_windows: int,
    outperform_windows: int,
    current_weight: float,
    diagnosis_pressure_score: float,
    modifier_score: float,
) -> bool:
    """Decide when a trim should escalate into a full exit.

    A full-exit recommendation should stay rare. We only use it when the
    pattern is strong enough that keeping a small residual position would add
    little portfolio value:

    - the holding has badly lagged the trade-matched S&P 500,
    - the weakness is persistent across the longer trailing windows we track,
    - there is no meaningful longer-horizon offset,
    - and the position is small or medium enough that exiting does not create
      a new concentration gap elsewhere.

    This helper intentionally sits *above* the trim-sizing logic. That keeps
    the main underperformance thresholds simple while making the "sell all"
    rule explicit, auditable, and easier to explain in notebooks and the app.
    """
    if excess_return_vs_benchmark is None or excess_return_vs_benchmark >= 0:
        return False

    underperformance = abs(excess_return_vs_benchmark)
    no_long_horizon_support = outperform_windows == 0
    small_or_medium_position = current_weight <= 0.06

    if (
        underperformance >= 1.0
        and underperform_windows >= 3
        and no_long_horizon_support
        and small_or_medium_position
    ):
        return True

    if (
        underperformance >= 0.85
        and underperform_windows >= 3
        and no_long_horizon_support
        and current_weight <= 0.05
        and (modifier_score >= 0.10 or diagnosis_pressure_score >= 35)
    ):
        return True

    if (
        underperformance >= 1.2
        and underperform_windows >= 2
        and no_long_horizon_support
        and current_weight <= 0.06
        and diagnosis_pressure_score >= 45
    ):
        return True

    return False


def _build_recommendation_summary(
    *,
    ticker: str,
    recommendation_label: str,
    reduction_pct: float,
    weighted_avg_buy_date: Optional[str],
    unrealized_return_pct: Optional[float],
    benchmark_return_since_buy: Optional[float],
    excess_return_vs_benchmark: Optional[float],
    diagnosis_pressure_score: float,
    linked_primary_concern: Optional[str],
    is_actionable: bool,
) -> str:
    """Write the plain-English recommendation summary shown in review surfaces."""
    period_text = (
        f"since the weighted-average buy date of {weighted_avg_buy_date}"
        if weighted_avg_buy_date
        else "over the tracked holding period"
    )
    if not is_actionable:
        return (
            f"{ticker} is **not** a sell recommendation right now. Even though it still appears in the diagnosis, "
            f"this rule set only recommends trimming when a holding has clearly lagged the S&P 500, and that case is not strong enough here {period_text}."
        )

    action_clause = (
        "selling all shares is the current recommendation."
        if reduction_pct >= 0.999
        else f"trimming about {reduction_pct:.0%} of the position is the current recommendation."
    )
    pressure_clause = (
        f" Because it also carries {diagnosis_pressure_score:.1f}/100 diagnosis pressure"
        + (f" through {linked_primary_concern.lower()}" if linked_primary_concern else "")
        + ","
        if diagnosis_pressure_score > 0
        else " Even without strong diagnosis pressure, the size of the benchmark lag alone is enough to justify action,"
    )
    return (
        f"{ticker} is currently a **{recommendation_label.lower()}** candidate. "
        f"{ticker} has underperformed the trade-matched S&P 500 {period_text}: the holding return is "
        f"{(unrealized_return_pct or 0.0):.1%} versus benchmark return {(benchmark_return_since_buy or 0.0):.1%}, "
        f"which is a lag of {(excess_return_vs_benchmark or 0.0):.1%}."
        + pressure_clause
        + " "
        + action_clause
    )


def _build_recommendation_explanation_parts(
    *,
    ticker: str,
    recommendation_label: str,
    reduction_pct: float,
    weighted_avg_buy_date: Optional[str],
    excess_return_vs_benchmark: Optional[float],
    diagnosis_pressure_score: float,
    linked_primary_concern: Optional[str],
    underperform_windows: int,
    outperform_windows: int,
    current_weight: float,
    variance_contribution_pct: float,
    is_actionable: bool,
) -> tuple[str, str, str]:
    """Split the recommendation into user-facing explanation parts.

    We keep the overall summary for compact surfaces, but these fields make it
    easier for notebooks and dashboards to explain:
    - what changed
    - why that matters for the portfolio
    - why this specific trim size was chosen
    """
    period_text = (
        f"since the weighted-average buy date of {weighted_avg_buy_date}"
        if weighted_avg_buy_date
        else "over the tracked holding period"
    )
    lag_text = f"{abs(excess_return_vs_benchmark or 0.0):.1%}"
    concern_text = linked_primary_concern.lower() if linked_primary_concern else "portfolio risk"

    what_changed = (
        f"{ticker} has lagged the trade-matched S&P 500 by {lag_text} {period_text}."
        if (excess_return_vs_benchmark or 0.0) < 0
        else f"{ticker} has not built a strong enough lag versus the S&P 500 {period_text} to justify action."
    )
    if underperform_windows > 0:
        what_changed += f" It also underperformed in {underperform_windows} of the trailing 1Y/3Y/5Y windows that were available."

    why_it_matters = (
        f"This matters because the holding is still feeding {concern_text}"
        if linked_primary_concern
        else "This matters because the holding is still adding measurable portfolio pressure"
    )
    detail_parts: list[str] = []
    if diagnosis_pressure_score > 0:
        detail_parts.append(f"{diagnosis_pressure_score:.1f}/100 diagnosis pressure")
    if current_weight >= 0.08:
        detail_parts.append(f"a meaningful position size at {current_weight:.1%} of the portfolio")
    if variance_contribution_pct >= 0.02:
        detail_parts.append(f"{variance_contribution_pct:.1%} of tracked variance")
    if detail_parts:
        why_it_matters += " through " + ", ".join(detail_parts)
    why_it_matters += "."

    if not is_actionable:
        amount_rationale = (
            "No sell-down amount is being recommended because the evidence is not strong enough after the current guardrails are applied."
        )
    elif reduction_pct >= 0.999:
        amount_rationale = "The current recommendation is to sell all shares because the underperformance is severe and there is not enough offsetting evidence to justify keeping even a small residual position."
    else:
        amount_rationale = (
            f"The current size is a {reduction_pct:.0%} trim, rather than a full exit, because the system is balancing benchmark lag against the strength of the supporting evidence."
        )
        if outperform_windows > 0:
            amount_rationale += f" Longer-horizon strength in {outperform_windows} trailing window(s) kept the recommendation from becoming even more aggressive."

    return what_changed, why_it_matters, amount_rationale


def _build_recommendation_guardrail_notes(
    *,
    is_fund: bool,
    holding_days: Optional[int],
    outperform_windows: int,
    underperform_windows: int,
    diagnosis_pressure_score: float,
    current_weight: float,
    reduction_pct: float,
) -> list[str]:
    """Explain which guardrails softened or blocked the action recommendation."""
    notes: list[str] = []
    if is_fund:
        notes.append("ETF and fund positions are intentionally kept out of the single-stock sell rule.")
    if holding_days is not None and holding_days < 120:
        notes.append("The holding period is still short, so the system avoids treating the early return gap as durable evidence.")
    if outperform_windows >= 2:
        notes.append("Strong trailing outperformance in multiple longer windows softened the recommendation.")
    elif outperform_windows == 1:
        notes.append("Some longer-horizon strength is still present, so the recommendation is moderated rather than absolute.")
    if reduction_pct > 0 and diagnosis_pressure_score < 30 and current_weight < 0.05 and underperform_windows <= 1:
        notes.append("The trim stays relatively small because the portfolio impact is still limited.")
    if not notes and reduction_pct == 0:
        notes.append("Current guardrails blocked a sell call because the evidence was not consistent enough.")
    return notes


def _build_portfolio_impact_preview(
    *,
    ticker: str,
    current_weight: float,
    reduction_pct: float,
    variance_contribution_pct: float,
    excess_return_vs_benchmark: Optional[float],
) -> tuple[float, float, float, str, list[str]]:
    """Estimate what likely improves if the holding is trimmed or sold.

    This is intentionally a preview, not a full portfolio re-optimization. The
    goal is to explain directionally useful outcomes in user language:
    - lower concentration from a smaller position
    - lower volatility from reducing a variance contributor
    - less exposure to a holding that has lagged the market
    """
    weight_reduction = round(current_weight * reduction_pct, 4)
    variance_reduction = round((variance_contribution_pct or 0.0) * reduction_pct, 4)
    relative_drag_reduction = round(weight_reduction * max(0.0, abs(excess_return_vs_benchmark or 0.0)), 4)

    bullets: list[str] = []
    if weight_reduction > 0:
        bullets.append(
            f"**Lower concentration:** {ticker}'s portfolio weight would fall by about {weight_reduction:.1%}."
        )
    if variance_reduction >= 0.005:
        bullets.append(
            f"**Lower volatility:** this could remove about {variance_reduction:.1%} of the holding's recent contribution to portfolio swings."
        )
    elif variance_contribution_pct > 0 and reduction_pct > 0:
        bullets.append(
            f"**Slightly steadier portfolio behavior:** this would trim a small part of the holding's recent volatility contribution."
        )
    if relative_drag_reduction > 0:
        bullets.append(
            f"**Less benchmark-lag exposure:** it would cut about {relative_drag_reduction:.1%} of weighted exposure to a holding that has been trailing the S&P 500."
        )

    if not bullets:
        bullets.append("**Portfolio impact looks limited:** this action is mostly about discipline and monitoring rather than a large immediate portfolio change.")

    summary = "If you make this move, the most likely near-term benefit is a cleaner portfolio shape with less reliance on this single lagging position."
    if len(bullets) >= 3:
        summary = "If you make this move, the portfolio should become less concentrated, a bit steadier, and less exposed to a name that has been lagging the market."
    elif len(bullets) == 2:
        summary = "If you make this move, the portfolio should become less dependent on this holding and somewhat less exposed to its recent weakness."

    return weight_reduction, variance_reduction, relative_drag_reduction, summary, bullets


def _build_portfolio_action_impact(
    bundle: dict[str, Any],
    holding_action_recommendations: list[HoldingActionRecommendation],
) -> PortfolioActionImpact:
    """Aggregate all actionable holding recommendations into one impact view."""
    actionable = [item for item in holding_action_recommendations if item.is_actionable]
    if not actionable:
        return PortfolioActionImpact(
            impact_summary=(
                "No portfolio-level action impact is being projected right now because the current rule set did not produce any actionable trims or sells."
            ),
            impact_bullets=[
                "The portfolio shape would stay broadly the same under the current recommendation set."
            ],
        )

    positions = bundle["open_positions"].copy()
    if positions.empty:
        return PortfolioActionImpact(
            actionable_count=len(actionable),
            actionable_tickers=[item.ticker for item in actionable],
            total_value_to_sell=round(sum(item.value_to_sell or 0.0 for item in actionable), 2),
            total_weight_reduction_pct_points=round(sum(item.projected_weight_reduction_pct_points or 0.0 for item in actionable), 4),
            total_variance_reduction_pct_points=round(sum(item.projected_variance_reduction_pct_points or 0.0 for item in actionable), 4),
            total_relative_drag_reduction_pct_points=round(sum(item.projected_relative_drag_reduction_pct_points or 0.0 for item in actionable), 4),
            impact_summary="The current action set should reduce exposure to the flagged names, but full concentration math could not be rebuilt from the saved positions snapshot.",
            impact_bullets=["The sell set still lowers exposure to the names currently flagged as laggards."],
        )

    positions["current_weight"] = pd.to_numeric(positions.get("current_weight"), errors="coerce").fillna(0.0)
    action_map = {item.ticker: (item.position_reduction_pct or 0.0) for item in actionable}
    positions["projected_weight"] = positions.apply(
        lambda row: max(
            0.0,
            float(row["current_weight"]) * (1.0 - float(action_map.get(str(row.get("ticker")), 0.0))),
        ),
        axis=1,
    )

    current_weights = sorted(positions["current_weight"].tolist(), reverse=True)
    projected_weights = sorted(positions["projected_weight"].tolist(), reverse=True)

    current_largest = current_weights[0] if current_weights else None
    projected_largest = projected_weights[0] if projected_weights else None
    current_top5 = sum(current_weights[:5]) if current_weights else None
    projected_top5 = sum(projected_weights[:5]) if projected_weights else None

    total_value_to_sell = round(sum(item.value_to_sell or 0.0 for item in actionable), 2)
    total_weight_reduction = round(sum(item.projected_weight_reduction_pct_points or 0.0 for item in actionable), 4)
    total_variance_reduction = round(sum(item.projected_variance_reduction_pct_points or 0.0 for item in actionable), 4)
    total_drag_reduction = round(sum(item.projected_relative_drag_reduction_pct_points or 0.0 for item in actionable), 4)

    bullets: list[str] = []
    if current_largest is not None and projected_largest is not None and projected_largest < current_largest:
        bullets.append(
            f"The largest position would fall from about {current_largest:.1%} to about {projected_largest:.1%}."
        )
    if current_top5 is not None and projected_top5 is not None and projected_top5 < current_top5:
        bullets.append(
            f"The top 5 holdings would fall from about {current_top5:.1%} of the portfolio to about {projected_top5:.1%}."
        )
    if total_variance_reduction > 0:
        bullets.append(
            f"Taken together, these moves could remove about {total_variance_reduction:.1%} of recent variance contribution from the flagged names."
        )
    if total_drag_reduction > 0:
        bullets.append(
            f"They would also cut about {total_drag_reduction:.1%} of weighted exposure to names that have been lagging the S&P 500."
        )
    if total_value_to_sell > 0:
        bullets.append(
            f"In total, the current action set would free up about ${total_value_to_sell:,.2f} to redeploy or hold in cash."
        )

    summary = (
        "If you follow the current action set, the portfolio should become less concentrated and a bit less exposed to recent laggards."
    )
    if total_variance_reduction > 0 and total_drag_reduction > 0:
        summary = (
            "If you follow the current action set, the portfolio should become less concentrated, a bit steadier, and less exposed to names that have been trailing the S&P 500."
        )

    return PortfolioActionImpact(
        actionable_count=len(actionable),
        actionable_tickers=[item.ticker for item in actionable],
        total_value_to_sell=total_value_to_sell,
        total_weight_reduction_pct_points=total_weight_reduction,
        total_variance_reduction_pct_points=total_variance_reduction,
        total_relative_drag_reduction_pct_points=total_drag_reduction,
        current_largest_position_pct=round(current_largest, 4) if current_largest is not None else None,
        projected_largest_position_pct=round(projected_largest, 4) if projected_largest is not None else None,
        current_top5_weight_pct=round(current_top5, 4) if current_top5 is not None else None,
        projected_top5_weight_pct=round(projected_top5, 4) if projected_top5 is not None else None,
        impact_summary=summary,
        impact_bullets=bullets,
    )


def _effective_holdings_from_values(position_values: list[float]) -> float:
    """Estimate effective holdings from positive position values."""
    positives = [float(value) for value in position_values if float(value or 0.0) > 0]
    total = sum(positives)
    if total <= 0:
        return 0.0
    weights = [value / total for value in positives]
    denominator = sum(weight * weight for weight in weights)
    if denominator <= 0:
        return 0.0
    return round(1.0 / denominator, 1)


def _sector_mix_from_position_map(position_map: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Aggregate position values by sector from a holding-value map."""
    sector_totals: dict[str, float] = {}
    for item in position_map.values():
        sector = str(item.get("sector") or "Unclassified").strip() or "Unclassified"
        sector_totals[sector] = sector_totals.get(sector, 0.0) + float(item.get("value") or 0.0)
    return sector_totals


def _build_trait_snapshot(
    *,
    label: str,
    position_map: dict[str, dict[str, Any]],
    cash_value: float,
    total_account_value: float,
) -> PortfolioTraitSnapshot:
    """Build one before/after portfolio trait snapshot from holding values."""
    invested_value = round(sum(float(item.get("value") or 0.0) for item in position_map.values()), 2)
    total_account_value = round(max(total_account_value, invested_value + max(0.0, cash_value)), 2)
    positions = sorted(
        [float(item.get("value") or 0.0) for item in position_map.values() if float(item.get("value") or 0.0) > 0],
        reverse=True,
    )
    largest_pct = (positions[0] / invested_value) if positions and invested_value > 0 else 0.0
    top5_pct = (sum(positions[:5]) / invested_value) if invested_value > 0 else 0.0
    sector_mix = _sector_mix_from_position_map(position_map)
    top_sector = ""
    top_sector_weight = 0.0
    if sector_mix and invested_value > 0:
        top_sector, top_sector_value = max(sector_mix.items(), key=lambda item: item[1])
        top_sector_weight = top_sector_value / invested_value
    return PortfolioTraitSnapshot(
        label=label,
        invested_value=invested_value,
        cash_value=round(max(0.0, cash_value), 2),
        total_account_value=total_account_value,
        invested_share_of_account=round((invested_value / total_account_value) if total_account_value > 0 else 0.0, 4),
        cash_share_of_account=round((max(0.0, cash_value) / total_account_value) if total_account_value > 0 else 0.0, 4),
        largest_position_pct_of_invested=round(largest_pct, 4),
        top5_weight_pct_of_invested=round(top5_pct, 4),
        effective_holdings=_effective_holdings_from_values(positions),
        top_sector=top_sector,
        top_sector_weight_pct_of_invested=round(top_sector_weight, 4),
    )


def _build_portfolio_rebalance_plan(
    *,
    current_holdings: list[CurrentHoldingSnapshot],
    portfolio_preferences: Optional[PortfolioPreferences],
    holding_action_recommendations: list[HoldingActionRecommendation],
    replacement_candidates: list[ReplacementCandidate],
) -> PortfolioRebalancePlan:
    """Build the first integrated before/after rebalance-plan object.

    The plan assumes the current sell recommendations are followed and the
    current ranked buy ideas are funded according to their suggested starting
    slices. This creates a reviewable proposal that lets the user compare:

    - current portfolio shape
    - projected portfolio shape after the current plan
    """
    if portfolio_preferences is None:
        return PortfolioRebalancePlan(
            summary="A rebalance plan could not be built because the portfolio preference object is missing.",
            plan_assumptions=["Run the portfolio analysis first so the buy-side constraints can be inferred."],
        )

    total_account_value = float(
        portfolio_preferences.current_total_portfolio_value
        or (portfolio_preferences.current_invested_value + portfolio_preferences.available_cash_now)
        or 0.0
    )
    current_cash = float(portfolio_preferences.available_cash_now or 0.0)

    before_positions: dict[str, dict[str, Any]] = {}
    for holding in current_holdings:
        before_positions[holding.ticker] = {
            "ticker": holding.ticker,
            "name": holding.security_name,
            "sector": holding.sector or "Unclassified",
            "value": float(holding.current_value or 0.0),
        }

    after_positions = {ticker: dict(item) for ticker, item in before_positions.items()}
    action_map = {item.ticker: item for item in holding_action_recommendations if item.is_actionable}
    buy_total = 0.0

    for ticker, recommendation in action_map.items():
        current_value = float(after_positions.get(ticker, {}).get("value", recommendation.current_value or 0.0))
        reduction_value = float(recommendation.value_to_sell or (current_value * float(recommendation.position_reduction_pct or 0.0)))
        if ticker not in after_positions:
            after_positions[ticker] = {
                "ticker": ticker,
                "name": ticker,
                "sector": recommendation.sector or "Unclassified",
                "value": max(0.0, current_value - reduction_value),
            }
        else:
            after_positions[ticker]["value"] = max(0.0, current_value - reduction_value)

    for candidate in replacement_candidates:
        allocation = float(candidate.suggested_allocation_amount or 0.0)
        if allocation <= 0:
            continue
        buy_total += allocation
        entry = after_positions.setdefault(
            candidate.ticker,
            {
                "ticker": candidate.ticker,
                "name": candidate.security_name,
                "sector": candidate.sector or "Unclassified",
                "value": 0.0,
            },
        )
        entry["name"] = candidate.security_name
        entry["sector"] = candidate.sector or entry.get("sector") or "Unclassified"
        entry["value"] = float(entry.get("value") or 0.0) + allocation

    before_snapshot = _build_trait_snapshot(
        label="Before plan",
        position_map=before_positions,
        cash_value=current_cash,
        total_account_value=total_account_value,
    )
    projected_cash = max(0.0, current_cash + float(sum(item.value_to_sell or 0.0 for item in action_map.values())) - buy_total)
    after_snapshot = _build_trait_snapshot(
        label="After plan",
        position_map=after_positions,
        cash_value=projected_cash,
        total_account_value=total_account_value,
    )

    invested_before = before_snapshot.invested_value or 1.0
    invested_after = after_snapshot.invested_value or 1.0
    candidate_map = {item.ticker: item for item in replacement_candidates}

    holding_changes: list[RebalanceHoldingChange] = []
    for ticker in sorted(set(before_positions) | set(after_positions)):
        before_item = before_positions.get(ticker, {})
        after_item = after_positions.get(ticker, {})
        before_value = float(before_item.get("value") or 0.0)
        after_value = float(after_item.get("value") or 0.0)
        if abs(after_value - before_value) < 1e-6:
            continue
        recommendation = action_map.get(ticker)
        candidate = candidate_map.get(ticker)
        if before_value > 0 and after_value <= 0:
            action_label = "Exit"
        elif before_value > 0 and after_value < before_value and candidate is not None:
            action_label = "Trim then add back"
        elif before_value > 0 and after_value < before_value:
            action_label = recommendation.recommendation_label if recommendation is not None else "Trim"
        elif before_value <= 0 and after_value > 0:
            action_label = "New buy"
        else:
            action_label = "Add more"
        explanation = (
            recommendation.recommendation_summary
            if recommendation is not None
            else candidate.why_it_fits
            if candidate is not None
            else "Projected holding change in the current rebalance plan."
        )
        holding_changes.append(
            RebalanceHoldingChange(
                ticker=ticker,
                security_name=str(after_item.get("name") or before_item.get("name") or ticker),
                sector=str(after_item.get("sector") or before_item.get("sector") or "Unclassified"),
                action_label=action_label,
                before_value=round(before_value, 2),
                after_value=round(after_value, 2),
                value_change=round(after_value - before_value, 2),
                before_weight_pct_of_invested=round(before_value / invested_before, 4) if invested_before > 0 else 0.0,
                after_weight_pct_of_invested=round(after_value / invested_after, 4) if invested_after > 0 else 0.0,
                weight_change_pct_points=round(((after_value / invested_after) - (before_value / invested_before)) if invested_before > 0 and invested_after > 0 else 0.0, 4),
                explanation=explanation,
            )
        )
    holding_changes.sort(key=lambda item: abs(item.value_change), reverse=True)

    before_sector_mix = _sector_mix_from_position_map(before_positions)
    after_sector_mix = _sector_mix_from_position_map(after_positions)
    sector_changes: list[RebalanceSectorChange] = []
    for sector in sorted(set(before_sector_mix) | set(after_sector_mix)):
        before_value = float(before_sector_mix.get(sector, 0.0))
        after_value = float(after_sector_mix.get(sector, 0.0))
        sector_changes.append(
            RebalanceSectorChange(
                sector=sector,
                before_value=round(before_value, 2),
                after_value=round(after_value, 2),
                before_weight_pct_of_invested=round(before_value / invested_before, 4) if invested_before > 0 else 0.0,
                after_weight_pct_of_invested=round(after_value / invested_after, 4) if invested_after > 0 else 0.0,
                weight_change_pct_points=round(((after_value / invested_after) - (before_value / invested_before)) if invested_before > 0 and invested_after > 0 else 0.0, 4),
            )
        )
    sector_changes.sort(key=lambda item: abs(item.weight_change_pct_points), reverse=True)

    improvement_bullets: list[str] = []
    if after_snapshot.cash_share_of_account < before_snapshot.cash_share_of_account:
        improvement_bullets.append(
            f"More of the account gets put to work: invested share rises from about {before_snapshot.invested_share_of_account:.1%} to about {after_snapshot.invested_share_of_account:.1%}."
        )
    if after_snapshot.largest_position_pct_of_invested < before_snapshot.largest_position_pct_of_invested:
        improvement_bullets.append(
            f"The largest holding falls from about {before_snapshot.largest_position_pct_of_invested:.1%} of invested assets to about {after_snapshot.largest_position_pct_of_invested:.1%}."
        )
    if after_snapshot.top5_weight_pct_of_invested < before_snapshot.top5_weight_pct_of_invested:
        improvement_bullets.append(
            f"Top-5 concentration improves from about {before_snapshot.top5_weight_pct_of_invested:.1%} to about {after_snapshot.top5_weight_pct_of_invested:.1%}."
        )
    if after_snapshot.effective_holdings > before_snapshot.effective_holdings:
        improvement_bullets.append(
            f"Effective holdings rise from about {before_snapshot.effective_holdings:.1f} to about {after_snapshot.effective_holdings:.1f}, which points to broader diversification."
        )
    if (
        after_snapshot.top_sector
        and before_snapshot.top_sector
        and after_snapshot.top_sector == before_snapshot.top_sector
        and after_snapshot.top_sector_weight_pct_of_invested < before_snapshot.top_sector_weight_pct_of_invested
    ):
        improvement_bullets.append(
            f"The most crowded sector ({before_snapshot.top_sector}) gets lighter, falling from about {before_snapshot.top_sector_weight_pct_of_invested:.1%} to about {after_snapshot.top_sector_weight_pct_of_invested:.1%} of invested assets."
        )
    if not improvement_bullets:
        improvement_bullets.append(
            "The main change is a rotation of capital rather than a dramatic reshaping of the portfolio."
        )

    sell_tickers = [item.ticker for item in holding_action_recommendations if item.is_actionable]
    buy_tickers = [item.ticker for item in replacement_candidates if (item.suggested_allocation_amount or 0.0) > 0]
    assumptions = [
        "This plan assumes the current sell recommendations are followed as shown in Risk Actions.",
        "It also assumes the current Buy Ideas budget slices are funded as suggested.",
        "The comparison is a portfolio-shape preview, not a price forecast.",
    ]
    summary = (
        f"This rebalance plan trims or exits {len(sell_tickers)} holding(s) and funds {len(buy_tickers)} buy idea(s). "
        f"It moves about ${sum(item.value_to_sell or 0.0 for item in action_map.values()):,.2f} out of current holdings and about ${buy_total:,.2f} into the proposed adds, so you can compare the portfolio before vs after in one view."
    )

    return PortfolioRebalancePlan(
        summary=summary,
        plan_assumptions=assumptions,
        improvement_bullets=improvement_bullets,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        total_value_to_sell=round(sum(item.value_to_sell or 0.0 for item in action_map.values()), 2),
        total_value_to_buy=round(buy_total, 2),
        projected_cash_after_plan=round(projected_cash, 2),
        sell_tickers=sell_tickers,
        buy_tickers=buy_tickers,
        holding_changes=holding_changes[:15],
        sector_changes=sector_changes[:12],
    )


def portfolio_rebalance_plan_from_user_preferences(
    *,
    diagnosis: PortfolioRiskDiagnosis,
    preferences: PortfolioPreferences,
) -> PortfolioRebalancePlan:
    """Rebuild the rebalance plan after the user changes buy preferences."""
    candidates = replacement_candidates_from_user_preferences(
        diagnosis=diagnosis,
        preferences=preferences,
    )
    return _build_portfolio_rebalance_plan(
        current_holdings=diagnosis.current_holdings,
        portfolio_preferences=preferences,
        holding_action_recommendations=diagnosis.holding_action_recommendations,
        replacement_candidates=candidates,
    )


def _suggested_max_new_position_pct(stated_risk_score: float) -> float:
    """Infer a simple default cap for new adds from stated risk tolerance.

    This is a temporary guardrail for the buy-side design layer. It prevents the
    first replacement ideas from recreating the same concentration problem we
    are trying to solve, even before we ask the user for explicit preferences.
    """
    if stated_risk_score <= 35:
        return 0.05
    if stated_risk_score <= 65:
        return 0.08
    return 0.10


def _risk_band_from_score(stated_risk_score: float) -> str:
    """Convert a 0-100 stated risk score into a broad user-facing band."""
    if stated_risk_score < 35:
        return "Conservative"
    if stated_risk_score < 70:
        return "Moderate"
    return "Aggressive"


def _build_portfolio_preferences(
    bundle: dict[str, Any],
    portfolio_action_impact: Optional[PortfolioActionImpact],
    top_sector_drivers: list[SectorRiskDriver],
) -> PortfolioPreferences:
    """Create the first buy-side constraint object from known portfolio facts.

    We are not collecting explicit buy preferences from the user yet, so this
    builder produces an inferred default object. It keeps a clean separation
    between:

    - what we already know from the current analysis
    - what the user still needs to tell us before true buy recommendations
    """
    headline = bundle["headline"]
    risk = bundle["risk"]
    available_cash_now = _safe_float(headline.get("uninvested_cash_estimate")) or 0.0
    action_cash = portfolio_action_impact.total_value_to_sell if portfolio_action_impact is not None else 0.0
    stated_risk_score = float(risk.get("stated_score", 0.0))
    stated_risk_band = str(risk.get("stated_band", ""))
    suggested_max_new_position_pct = _suggested_max_new_position_pct(stated_risk_score)

    crowded_sector = next(
        (
            driver.sector
            for driver in top_sector_drivers
            if (driver.weight_pct or 0.0) >= 0.25 and "crowd" in str(driver.primary_reason_label or "").lower()
        ),
        None,
    )
    inferred_sector_avoidances = [crowded_sector] if crowded_sector else []

    if stated_risk_score <= 35:
        vehicle_preference_label = "Default to steadier ETF-led adds unless you explicitly want individual stocks."
    elif stated_risk_score <= 65:
        vehicle_preference_label = "Default to a blend of diversified ETFs and selective single stocks."
    else:
        vehicle_preference_label = "Higher-risk profile allows either ETFs or single stocks, but new adds should still avoid rebuilding concentration."

    assumption_notes = [
        "No explicit buy-side preferences have been captured yet, so this object uses conservative defaults.",
        f"New adds are temporarily capped near {suggested_max_new_position_pct:.0%} of the portfolio to avoid recreating concentration too quickly.",
    ]
    if crowded_sector:
        assumption_notes.append(
            f"Until you say otherwise, the most crowded current sector ({crowded_sector}) is treated as a lower-priority place for fresh capital."
        )

    unresolved_preferences = [
        "Should freed cash be fully reinvested, partly reinvested, or left partly in cash?",
        "Do you prefer ETFs, single stocks, or a blend for new capital?",
        "Are there any sectors you want to emphasize or avoid?",
        "What is the maximum size you are comfortable giving to a new position as a share of the total portfolio?",
    ]

    constraints_summary = (
        f"Right now the system can assume about ${available_cash_now + action_cash:,.2f} could be available for future adds, "
        f"but reinvestment rules and sector preferences are still user decisions. Until those are set, the safest default is to favor diversified adds and keep any one new position below about {suggested_max_new_position_pct:.0%} of the total portfolio."
    )

    return PortfolioPreferences(
        available_cash_now=round(available_cash_now, 2),
        available_cash_if_actions_followed=round(available_cash_now + action_cash, 2),
        current_invested_value=round(_safe_float(headline.get("current_portfolio_value")) or 0.0, 2),
        current_total_portfolio_value=round(_safe_float(headline.get("total_account_value_estimate")) or 0.0, 2),
        budget_to_deploy=round(available_cash_now + action_cash, 2),
        reinvest_freed_cash=None,
        reinvestment_preference_label="Not set yet — decide whether freed cash should be reinvested or partly left in cash.",
        stated_risk_score=stated_risk_score,
        stated_risk_band=stated_risk_band,
        suggested_max_new_position_pct=suggested_max_new_position_pct,
        max_new_position_interpretation="This cap is measured as a share of the total portfolio, not just the buy budget.",
        allow_etfs=True,
        allow_single_stocks=True,
        single_stocks_preferred=None,
        prefer_high_dividend_etfs=False,
        prefer_low_expense_for_dividend_etfs=False,
        buy_idea_limit=10,
        include_existing_holdings=False,
        vehicle_preference_label=vehicle_preference_label,
        sector_preferences=[],
        inferred_sector_avoidances=inferred_sector_avoidances,
        constraints_summary=constraints_summary,
        unresolved_preferences=unresolved_preferences,
        assumption_notes=assumption_notes,
        preference_source="inferred_defaults",
    )


def portfolio_preferences_from_user_inputs(
    *,
    base_preferences: PortfolioPreferences,
    budget_to_deploy: Optional[float],
    reinvest_choice: str,
    sector_preferences: Optional[list[str]],
    max_new_position_pct: Optional[float],
    stated_risk_score: Optional[float],
    allow_etfs: bool,
    allow_single_stocks: bool,
    vehicle_preference: str,
    prefer_high_dividend_etfs: bool,
    prefer_low_expense_for_dividend_etfs: bool,
    buy_idea_limit: Optional[int],
    include_existing_holdings: bool,
) -> PortfolioPreferences:
    """Create a user-defined `PortfolioPreferences` object from dashboard inputs.

    This is the bridge from UI controls into the typed buy-side constraints
    layer. It starts from the inferred base preferences, then replaces them with
    the choices the user has explicitly made.

    The resulting object is meant to be stable enough for later buy-side logic
    to consume directly, instead of scraping loosely typed UI values again.
    """
    resolved_budget = budget_to_deploy
    if resolved_budget is None or resolved_budget < 0:
        resolved_budget = base_preferences.available_cash_if_actions_followed

    resolved_risk_score = (
        float(stated_risk_score)
        if stated_risk_score is not None
        else float(base_preferences.stated_risk_score)
    )
    resolved_risk_band = _risk_band_from_score(resolved_risk_score)

    resolved_max_position_pct = (
        float(max_new_position_pct)
        if max_new_position_pct is not None
        else float(base_preferences.suggested_max_new_position_pct or _suggested_max_new_position_pct(resolved_risk_score))
    )
    resolved_buy_idea_limit = int(buy_idea_limit or base_preferences.buy_idea_limit or 10)
    resolved_buy_idea_limit = max(5, min(20, resolved_buy_idea_limit))

    cleaned_sector_preferences = [item for item in (sector_preferences or []) if str(item).strip()]

    if reinvest_choice == "Yes":
        reinvest_freed_cash = True
        reinvestment_preference_label = "Freed cash should be reinvested."
    elif reinvest_choice == "No":
        reinvest_freed_cash = False
        reinvestment_preference_label = "Freed cash does not have to be reinvested."
    else:
        reinvest_freed_cash = None
        reinvestment_preference_label = "Reinvestment is still undecided."

    if vehicle_preference == "ETFs only":
        single_stocks_preferred = False
        vehicle_preference_label = "Only ETFs should be considered for future adds."
    elif vehicle_preference == "Single stocks only":
        single_stocks_preferred = True
        vehicle_preference_label = "Only single stocks should be considered for future adds."
    elif vehicle_preference == "Prefer ETFs":
        single_stocks_preferred = False
        vehicle_preference_label = "Prefer ETFs for future adds."
    elif vehicle_preference == "Prefer single stocks":
        single_stocks_preferred = True
        vehicle_preference_label = "Prefer single stocks for future adds."
    else:
        single_stocks_preferred = None
        vehicle_preference_label = "Blend ETFs and single stocks unless a later rule says otherwise."

    unresolved_preferences: list[str] = []
    if reinvest_freed_cash is None:
        unresolved_preferences.append("Decide whether freed cash should be fully reinvested or partly kept in cash.")
    if not allow_etfs and not allow_single_stocks:
        unresolved_preferences.append("At least one vehicle type must be allowed before buy suggestions can work.")
    if not cleaned_sector_preferences:
        unresolved_preferences.append("Sector preferences are still open, so future adds will default to portfolio-need first.")

    assumption_notes = [
        f"Budget to deploy is set to about ${resolved_budget:,.2f}.",
        f"New adds should stay at or below about {resolved_max_position_pct:.0%} of the total portfolio unless you later override that.",
        f"The buy view is currently set to show the top {resolved_buy_idea_limit} ideas.",
    ]
    if base_preferences.inferred_sector_avoidances:
        assumption_notes.append(
            "Current crowded sectors are still treated as lower-priority places for fresh capital unless you explicitly choose them."
        )
    if prefer_high_dividend_etfs:
        assumption_notes.append(
            "High-dividend ETFs will get a boost when they still fit the portfolio gap and performance filters."
        )
    if prefer_low_expense_for_dividend_etfs:
        assumption_notes.append(
            "Among dividend ETFs, lower expense ratios get extra priority when possible."
        )
    if include_existing_holdings:
        assumption_notes.append(
            "Existing holdings are allowed back into the buy universe for reinvestment ideas."
        )

    constraints_summary = (
        f"The buy side should currently work with about ${resolved_budget:,.2f}, a stated risk tolerance of {resolved_risk_score:.0f}/100 ({resolved_risk_band}), "
        f"and a cap near {resolved_max_position_pct:.0%} of the total portfolio for any one new position. "
        f"{vehicle_preference_label}"
    )

    return PortfolioPreferences(
        available_cash_now=base_preferences.available_cash_now,
        available_cash_if_actions_followed=base_preferences.available_cash_if_actions_followed,
        current_invested_value=base_preferences.current_invested_value,
        current_total_portfolio_value=base_preferences.current_total_portfolio_value,
        budget_to_deploy=round(float(resolved_budget), 2),
        reinvest_freed_cash=reinvest_freed_cash,
        reinvestment_preference_label=reinvestment_preference_label,
        stated_risk_score=round(resolved_risk_score, 1),
        stated_risk_band=resolved_risk_band,
        suggested_max_new_position_pct=round(resolved_max_position_pct, 4),
        max_new_position_interpretation="This cap is measured as a share of the total portfolio, not just the buy budget.",
        allow_etfs=allow_etfs,
        allow_single_stocks=allow_single_stocks,
        single_stocks_preferred=single_stocks_preferred,
        prefer_high_dividend_etfs=prefer_high_dividend_etfs,
        prefer_low_expense_for_dividend_etfs=prefer_low_expense_for_dividend_etfs,
        buy_idea_limit=resolved_buy_idea_limit,
        include_existing_holdings=include_existing_holdings,
        vehicle_preference_label=vehicle_preference_label,
        sector_preferences=cleaned_sector_preferences,
        inferred_sector_avoidances=base_preferences.inferred_sector_avoidances,
        constraints_summary=constraints_summary,
        unresolved_preferences=unresolved_preferences,
        assumption_notes=assumption_notes,
        preference_source="user_defined",
    )


def _build_portfolio_gaps(
    bundle: dict[str, Any],
    top_concerns: list[DiagnosisConcern],
    top_sector_drivers: list[SectorRiskDriver],
    portfolio_action_impact: Optional[PortfolioActionImpact],
) -> list[PortfolioGap]:
    """Explain what the portfolio still needs before any buy ideas are shown.

    The goal here is not to recommend securities. It is to surface the portfolio
    needs that future buys would have to solve. That makes later buy-side logic
    much easier to trust because it starts with:

    - what is missing?
    - why is it missing?
    - what changes if we fill that gap?
    """
    headline = bundle["headline"]
    concentration_concern = next((item for item in top_concerns if item.concern_key == "concentration"), None)
    market_concern = next((item for item in top_concerns if item.concern_key == "market"), None)
    macro_series = bundle.get("macro_series", pd.DataFrame())
    sector_allocation = bundle.get("sector_allocation", pd.DataFrame()).copy()
    if not sector_allocation.empty:
        sector_allocation["weight_pct"] = pd.to_numeric(sector_allocation.get("weight_pct"), errors="coerce").fillna(0.0)
        sector_allocation = sector_allocation.sort_values("weight_pct", ascending=False)

    gaps: list[PortfolioGap] = []
    max_position_weight = _safe_float(headline.get("max_position_weight")) or 0.0
    top5_weight = _safe_float(headline.get("top_5_weight")) or 0.0
    relative_volatility = _safe_float(headline.get("relative_volatility_to_benchmark")) or 0.0
    relative_market_sensitivity = _safe_float(headline.get("relative_market_sensitivity_to_benchmark")) or 0.0
    cash_freed = portfolio_action_impact.total_value_to_sell if portfolio_action_impact is not None else 0.0
    restrictive_rates = False
    if not macro_series.empty and "series_key" in macro_series.columns and "value" in macro_series.columns:
        fed_row = macro_series.loc[macro_series["series_key"].astype(str) == "fed_funds_rate"].tail(1)
        fed_value = _safe_float(fed_row["value"].iloc[-1]) if not fed_row.empty else None
        restrictive_rates = fed_value is not None and fed_value >= 3.0

    if concentration_concern is not None or max_position_weight >= 0.15 or top5_weight >= 0.55:
        largest_after = portfolio_action_impact.projected_largest_position_pct if portfolio_action_impact else None
        top5_after = portfolio_action_impact.projected_top5_weight_pct if portfolio_action_impact else None
        gaps.append(
            PortfolioGap(
                gap_key="diversified_core",
                label="Needs more diversification and steadier core exposure",
                severity_score=round(max(concentration_concern.severity_score if concentration_concern else 0.0, max_position_weight * 120), 1),
                severity_band=_severity_band(max(concentration_concern.severity_score if concentration_concern else 0.0, max_position_weight * 120)),
                what_is_missing="The portfolio still needs more capital in holdings that do not depend so heavily on one or two names.",
                why_this_gap_exists=(
                    f"The largest position still makes up about {max_position_weight:.1%} of the portfolio"
                    + (f", and even after the current trims the largest position would still be about {largest_after:.1%}" if largest_after is not None else "")
                    + f". The top 5 holdings currently account for about {top5_weight:.1%}"
                    + (f", falling only to about {top5_after:.1%} under the current action set." if top5_after is not None else ".")
                ),
                what_would_change=[
                    "Less dependence on one holding to carry the whole portfolio.",
                    "A steadier portfolio if the biggest current winner reverses.",
                    "More room to add future ideas without immediately recreating concentration pressure.",
                ],
                linked_concerns=[item.label for item in top_concerns[:2] if item.label],
                supporting_evidence=_unique_nonempty(
                    [
                        f"Largest position: {max_position_weight:.1%}",
                        f"Top 5 holdings: {top5_weight:.1%}",
                        (
                            f"Projected top 5 after current actions: {top5_after:.1%}"
                            if top5_after is not None else None
                        ),
                    ]
                ),
                suggested_vehicle_tilt="Prefer benchmark-like core holdings or diversified funds before adding more single-name risk.",
            )
        )

    if not sector_allocation.empty and float(sector_allocation["weight_pct"].iloc[0]) >= 0.25:
        top_sector = str(sector_allocation.iloc[0]["sector"])
        top_sector_weight = float(sector_allocation.iloc[0]["weight_pct"])
        gaps.append(
            PortfolioGap(
                gap_key="sector_balance",
                label="Needs less reliance on one sector",
                severity_score=round(min(100.0, top_sector_weight * 120), 1),
                severity_band=_severity_band(min(100.0, top_sector_weight * 120)),
                what_is_missing="The portfolio needs more balance outside its most crowded sector.",
                why_this_gap_exists=f"{top_sector} currently makes up about {top_sector_weight:.1%} of invested capital, which means sector-specific weakness can spread across too much of the account.",
                what_would_change=[
                    "Less cluster risk from one part of the market dominating the account.",
                    "More balanced sector exposure after the current trims free up capital.",
                    "A cleaner path to future adds that do not stack onto the same theme.",
                ],
                linked_concerns=["Concentration risk", "Sector crowding"],
                supporting_evidence=_unique_nonempty(
                    [
                        f"Top sector: {top_sector}",
                        f"Top sector weight: {top_sector_weight:.1%}",
                        (
                            f"Lead sector driver: {top_sector_drivers[0].primary_reason_label}"
                            if top_sector_drivers else None
                        ),
                    ]
                ),
                suggested_vehicle_tilt="Favor adds outside the most crowded sector, or use broader funds that naturally dilute sector concentration.",
            )
        )

    if market_concern is not None or relative_volatility >= 1.4 or relative_market_sensitivity >= 1.15:
        gaps.append(
            PortfolioGap(
                gap_key="defensive_ballast",
                label="Needs steadier, lower-volatility ballast",
                severity_score=round(max(market_concern.severity_score if market_concern else 0.0, relative_volatility * 35), 1),
                severity_band=_severity_band(max(market_concern.severity_score if market_concern else 0.0, relative_volatility * 35)),
                what_is_missing="The portfolio still needs holdings that behave more like a steady core than a high-swing growth sleeve.",
                why_this_gap_exists=(
                    f"Recent portfolio volatility has been about {relative_volatility:.2f}x the S&P 500"
                    f" and market sensitivity is about {relative_market_sensitivity:.2f}x."
                    + (" Higher interest rates also make that kind of high-swing mix less forgiving." if restrictive_rates else "")
                ),
                what_would_change=[
                    "Smaller portfolio swings during normal market pullbacks.",
                    "Less pressure to keep trimming individual laggards just to calm the portfolio down.",
                    "A more stable base for any future stock ideas added on top.",
                ],
                linked_concerns=[
                    label for label in [market_concern.label if market_concern else None, "Macro sensitivity" if restrictive_rates else None] if label
                ],
                supporting_evidence=_unique_nonempty(
                    [
                        f"Relative volatility vs S&P 500: {relative_volatility:.2f}x",
                        f"Relative market sensitivity vs S&P 500: {relative_market_sensitivity:.2f}x",
                        "Rates are still relatively high." if restrictive_rates else None,
                    ]
                ),
                suggested_vehicle_tilt="Favor lower-beta, benchmark-like, or more defensive holdings over additional high-swing names.",
            )
        )

    if cash_freed > 0:
        gaps.append(
            PortfolioGap(
                gap_key="capital_redeployment_plan",
                label="Needs a clear plan for freed cash",
                severity_score=52.0,
                severity_band=_severity_band(52.0),
                what_is_missing="The portfolio will need a clear rule for what to do with the cash created by the current action set.",
                why_this_gap_exists=f"The current trims and sells would free up about ${cash_freed:,.2f}. Without a plan, that cash can either sit idle by accident or get pushed back into the same crowded parts of the portfolio.",
                what_would_change=[
                    "Future adds can be tied to a real portfolio need instead of impulse redeployment.",
                    "Cash can be split intentionally between reinvestment and dry powder if that fits the user.",
                    "Buy recommendations later can be sized around an actual budget instead of vague opportunity lists.",
                ],
                linked_concerns=["Action follow-through"],
                supporting_evidence=[f"Cash freed by current actions: ${cash_freed:,.2f}"],
                suggested_vehicle_tilt="Decide first how much cash should be redeployed, then match that budget to the most important gap rather than the most exciting ticker.",
            )
        )

    return sorted(gaps, key=lambda item: (-item.severity_score, item.label))


@lru_cache(maxsize=1)
def _load_buy_candidate_universe_inputs() -> tuple[list[BuyCandidateUniverseEntry], pd.DataFrame]:
    """Load the curated buy universe and its enriched metrics once per process.

    The buy-side candidate layer reads from a deliberately small curated
    universe. Caching keeps the first-pass buy engine fast while preserving the
    explicit file-backed boundary between research data and recommendation logic.
    """
    entries = load_buy_candidate_universe(BUY_CANDIDATE_UNIVERSE_PATH)
    enriched = _load_csv(BUY_CANDIDATE_UNIVERSE_ENRICHED_PATH)
    return entries, enriched


def _fit_band(score: float) -> str:
    """Map a buy-candidate fit score onto a quick review label."""
    if score < 35:
        return "Weak fit"
    if score < 55:
        return "Plausible fit"
    if score < 75:
        return "Good fit"
    return "Strong fit"


def _normalized_sector_key(sector: Any) -> str:
    """Normalize sector labels so portfolio and buy-universe naming can match.

    Different sources use slightly different labels for the same economic area,
    for example `Technology` versus `Information Technology`. The buy side
    should compare those as the same sector when applying preferences and
    crowding penalties.
    """
    text = str(sector or "").strip().lower()
    aliases = {
        "technology": "information technology",
        "information technology": "information technology",
        "financial services": "financials",
        "financials": "financials",
        "consumer cyclical": "consumer discretionary",
        "consumer discretionary": "consumer discretionary",
        "consumer defensive": "consumer defensive",
        "consumer staples": "consumer defensive",
        "health care": "health care",
        "communication services": "communication services",
        "industrials": "industrials",
        "utilities": "utilities",
        "real estate": "real estate",
        "energy": "energy",
        "materials": "materials",
        "broad market": "broad market",
        "fixed income": "fixed income",
    }
    return aliases.get(text, text)


def _replacement_candidate_confidence(score: float, evidence_count: int) -> tuple[float, str]:
    """Estimate confidence for a replacement candidate from fit and evidence."""
    confidence_score = min(1.0, 0.2 + (score / 100.0) * 0.5 + min(evidence_count, 4) * 0.08)
    if confidence_score < 0.4:
        band = "Low"
    elif confidence_score < 0.72:
        band = "Medium"
    else:
        band = "High"
    return round(confidence_score, 2), band


def _candidate_priority_key(item: dict[str, Any]) -> tuple[Any, ...]:
    """Return a stable sort key for replacement-candidate ranking.

    The larger universe now includes many near-duplicate candidates. A pure
    fit-score sort tends to over-index on a few almost interchangeable names.
    This key still respects fit first, but then prefers candidates that:

    - appear in more trusted source universes
    - have steadier recent volatility when the fit score is otherwise similar
    - preserve deterministic ordering for review and tests
    """
    entry = item["entry"]
    row = item["row"]
    volatility = _safe_float(row.get("annualized_volatility_1y"))
    source_count = int(_safe_float(row.get("source_count")) or entry.source_count or 0)
    return (
        -float(item["fit_score"]),
        -source_count,
        volatility if volatility is not None else 999.0,
        entry.asset_type != "ETF",
        entry.ticker,
    )


def _candidate_diversity_signature(candidate: BuyCandidateUniverseEntry) -> tuple[str, str, str, str]:
    """Build a compact signature so the buy slate avoids obvious duplicates.

    Examples:
    - `IVV` and `VOO` share the same broad role and should not both dominate
      the top of the slate.
    - a broad-market core ETF and a low-volatility ETF should both be allowed,
      because they play different jobs in the portfolio.
    """
    return (
        candidate.asset_type,
        candidate.primary_role,
        candidate.sector,
        candidate.style_tilt,
    )


def _candidate_passes_buy_filters(
    candidate: BuyCandidateUniverseEntry,
    candidate_row: dict[str, Any],
    preferences: PortfolioPreferences,
) -> tuple[bool, list[str]]:
    """Apply first-pass buy-side quality filters before ranking candidates.

    The bigger universe makes it easier for noisy ideas to sneak into the top
    list. These filters are intentionally simple and conservative:

    - avoid very high-swing single stocks for moderate-or-lower risk profiles
    - avoid names that have lagged the S&P 500 across multiple windows
    - keep extremely rate-sensitive fixed-income ballast from crowding out
      better first-pass buy ideas when it has badly lagged for years
    """
    notes: list[str] = []
    volatility_1y = _safe_float(candidate_row.get("annualized_volatility_1y"))
    rel_1y = _safe_float(candidate_row.get("relative_1y_return_pct"))
    rel_3y = _safe_float(candidate_row.get("relative_3y_return_pct"))
    rel_5y = _safe_float(candidate_row.get("relative_5y_return_pct"))
    beta = _safe_float(candidate_row.get("beta"))
    source_count = int(_safe_float(candidate_row.get("source_count")) or candidate.source_count or 0)

    if candidate.asset_type == "Stock":
        if preferences.stated_risk_score <= 65 and volatility_1y is not None and volatility_1y >= 0.42:
            return False, ["Recent volatility is still too high for the current buy profile."]
        if preferences.stated_risk_score <= 55 and beta is not None and beta >= 1.45:
            return False, ["This stock still moves much more than the market for the current risk tolerance."]
        if (
            rel_1y is not None and rel_1y < -0.10
            and rel_3y is not None and rel_3y < -0.12
            and rel_5y is not None and rel_5y < -0.08
        ):
            return False, ["It has lagged the S&P 500 across multiple windows, so it is not a strong first-pass add."]
        if source_count >= 3:
            notes.append("it also shows up across several trusted source universes")
    else:
        if (
            candidate.sector == "Fixed Income"
            and rel_3y is not None and rel_3y < -0.45
            and rel_5y is not None and rel_5y < -0.45
        ):
            notes.append("its long-term relative returns are weak, so it should compete mainly as stability ballast, not as a core add")

    return True, notes


def _build_current_holdings(bundle: dict[str, Any]) -> list[CurrentHoldingSnapshot]:
    """Build a compact list of current holdings for reinvestment-aware buy ideas."""
    positions = bundle["open_positions"].copy()
    profiles = bundle["company_profiles"].copy()
    if positions.empty:
        return []

    profile_lookup: dict[str, str] = {}
    if not profiles.empty and "symbol" in profiles.columns:
        profile_lookup = {
            str(row.get("symbol")): str(row.get("companyName") or row.get("company_name") or row.get("symbol") or "").strip()
            for row in profiles.to_dict(orient="records")
        }

    snapshots: list[CurrentHoldingSnapshot] = []
    for row in positions.to_dict(orient="records"):
        ticker = str(row.get("ticker") or "").strip()
        if not ticker:
            continue
        snapshots.append(
            CurrentHoldingSnapshot(
                ticker=ticker,
                security_name=profile_lookup.get(ticker) or ticker,
                sector=row.get("sector"),
                current_weight=_safe_float(row.get("current_weight")),
                current_value=_safe_float(row.get("current_value")),
                excess_return_vs_benchmark=_safe_float(row.get("excess_return_vs_benchmark")),
                weighted_avg_buy_date=str(row.get("weighted_avg_buy_date") or "").strip() or None,
            )
        )
    return snapshots


def _gap_score_for_candidate(
    candidate: BuyCandidateUniverseEntry,
    gap: PortfolioGap,
    candidate_row: dict[str, Any],
    preferences: PortfolioPreferences,
) -> tuple[float, list[str], list[str]]:
    """Score how well one candidate fits one portfolio gap.

    The first buy engine should stay explainable. This helper gives each
    candidate a deterministic score against each gap, plus human-readable
    reasons that later power the `Buy Ideas` tab.
    """
    score = 0.0
    reasons: list[str] = []
    evidence: list[str] = []

    volatility_1y = _safe_float(candidate_row.get("annualized_volatility_1y"))
    rel_1y = _safe_float(candidate_row.get("relative_1y_return_pct"))
    rel_3y = _safe_float(candidate_row.get("relative_3y_return_pct"))
    rel_5y = _safe_float(candidate_row.get("relative_5y_return_pct"))
    beta = _safe_float(candidate_row.get("beta"))
    candidate_sector_key = _normalized_sector_key(candidate.sector)
    avoided_sector_keys = {
        _normalized_sector_key(item) for item in (preferences.inferred_sector_avoidances or [])
    }
    preferred_sector_keys = {
        _normalized_sector_key(item) for item in (preferences.sector_preferences or [])
    }

    if gap.gap_key == "diversified_core":
        if candidate.is_core:
            score += 26
            reasons.append("it adds broader, steadier core exposure")
        if candidate.primary_role in {"Core benchmark ballast", "Core diversification"}:
            score += 20
            reasons.append("its role is already aligned with rebuilding the portfolio's core")
        if candidate.asset_type == "ETF":
            score += 8
        if candidate.sector == "Broad Market":
            score += 8
        if rel_3y is not None and rel_3y >= 0:
            score += 6
            evidence.append(f"3Y vs S&P 500: {rel_3y:.1%}")
        if rel_5y is not None and rel_5y >= 0:
            score += 6
            evidence.append(f"5Y vs S&P 500: {rel_5y:.1%}")
        if volatility_1y is not None and volatility_1y <= 0.16:
            score += 6
            evidence.append(f"1Y volatility: {volatility_1y:.1%}")

    if gap.gap_key == "sector_balance":
        if candidate_sector_key not in avoided_sector_keys:
            score += 16
            reasons.append("it does not add more weight to the most crowded current sector")
        if candidate_sector_key in preferred_sector_keys:
            score += 18
            reasons.append("it also matches the sector preference you set")
        if candidate.sector == "Broad Market":
            score += 10
            reasons.append("broad-market exposure naturally spreads sector risk")
        if candidate.is_defensive:
            score += 8
            reasons.append("it helps diversify away from the crowded area without adding another high-swing sleeve")

    if gap.gap_key == "defensive_ballast":
        if candidate.is_defensive:
            score += 24
            reasons.append("it is designed to be steadier than a high-swing equity sleeve")
        if "defensive" in candidate.primary_role.lower() or "stability" in candidate.primary_role.lower():
            score += 15
        if volatility_1y is not None and volatility_1y <= 0.14:
            score += 16
            evidence.append(f"1Y volatility: {volatility_1y:.1%}")
        elif volatility_1y is not None and volatility_1y <= 0.18:
            score += 8
            evidence.append(f"1Y volatility: {volatility_1y:.1%}")
        if candidate.asset_type == "ETF":
            score += 6
        if beta is not None and beta <= 1.0:
            score += 8
            evidence.append(f"Beta: {beta:.2f}")
        if rel_3y is not None and rel_3y >= -0.10:
            score += 6
            evidence.append(f"3Y vs S&P 500: {rel_3y:.1%}")

    if gap.gap_key == "capital_redeployment_plan":
        if candidate.is_core:
            score += 14
            reasons.append("it gives freed cash a clear job instead of pushing it back into a narrow theme")
        if candidate.asset_type == "ETF":
            score += 8
        if rel_1y is not None and rel_1y >= -0.05:
            score += 6
            evidence.append(f"1Y vs S&P 500: {rel_1y:.1%}")

    if rel_1y is not None and rel_1y > 0:
        score += min(10.0, rel_1y * 55.0)
    elif rel_1y is not None and rel_1y < -0.20:
        score -= 8

    if rel_3y is not None and rel_3y > 0:
        score += min(12.0, rel_3y * 18.0)
    elif rel_3y is not None and rel_3y < -0.25:
        score -= 10

    if rel_5y is not None and rel_5y > 0:
        score += min(12.0, rel_5y * 12.0)
    elif rel_5y is not None and rel_5y < -0.30:
        score -= 10

    return round(score, 1), _unique_nonempty(reasons), _unique_nonempty(evidence)


def _select_replacement_candidate_slate(
    scored_candidates: list[dict[str, Any]],
    gaps_to_score: list[PortfolioGap],
    preferences: PortfolioPreferences,
) -> list[dict[str, Any]]:
    """Choose a scan-friendly slate from the scored candidate pool.

    The buy tab should feel useful within a few seconds. That means the top of
    the slate cannot be five tiny variations of the same broad-market ETF.
    This selector deliberately aims for:

    - at least one candidate for each of the most important gaps
    - limited duplication by role/sector/style
    - a final list that is diverse enough to compare meaningfully
    """
    target_count = max(5, int(preferences.buy_idea_limit or 10))
    selected: list[dict[str, Any]] = []
    used_tickers: set[str] = set()
    used_signatures: set[tuple[str, str, str, str]] = set()

    def can_add(item: dict[str, Any], *, allow_duplicate_gap: bool) -> bool:
        entry = item["entry"]
        gap = item["gap"]
        signature = _candidate_diversity_signature(entry)
        if entry.ticker in used_tickers:
            return False
        if signature in used_signatures:
            return False
        if not allow_duplicate_gap and any(existing["gap"].gap_key == gap.gap_key for existing in selected):
            return False
        broad_market_count = sum(existing["entry"].sector == "Broad Market" for existing in selected)
        if broad_market_count >= 2 and entry.sector == "Broad Market" and float(item["fit_score"]) < 80.0:
            return False
        same_sector_count = sum(existing["entry"].sector == entry.sector for existing in selected)
        if (
            same_sector_count >= 2
            and entry.sector not in {"Broad Market", "Fixed Income"}
            and entry.sector not in set(preferences.sector_preferences or [])
        ):
            return False
        return True

    ordered_all = sorted(scored_candidates, key=_candidate_priority_key)
    for gap in gaps_to_score:
        gap_matches = [
            item for item in ordered_all
            if item["gap"].gap_key == gap.gap_key and float(item["fit_score"]) >= 50.0
        ]
        for item in gap_matches:
            if can_add(item, allow_duplicate_gap=False):
                selected.append(item)
                used_tickers.add(item["entry"].ticker)
                used_signatures.add(_candidate_diversity_signature(item["entry"]))
                break

    for item in ordered_all:
        if len(selected) >= target_count:
            break
        if can_add(item, allow_duplicate_gap=True):
            selected.append(item)
            used_tickers.add(item["entry"].ticker)
            used_signatures.add(_candidate_diversity_signature(item["entry"]))

    return selected[:target_count]


def _preference_penalty_or_bonus(
    candidate: BuyCandidateUniverseEntry,
    preferences: PortfolioPreferences,
    candidate_row: Optional[dict[str, Any]] = None,
) -> tuple[float, list[str]]:
    """Adjust candidate fit for explicit user constraints."""
    delta = 0.0
    notes: list[str] = []
    candidate_row = candidate_row or {}

    if candidate.asset_type == "ETF":
        if not preferences.allow_etfs:
            return -999.0, ["ETFs are turned off in the current buy preferences."]
        if preferences.single_stocks_preferred is False:
            delta += 8.0
            notes.append("it matches the current ETF preference")
        if preferences.prefer_high_dividend_etfs:
            metadata = fetch_candidate_market_metadata(candidate.market_data_symbol)
            expense_ratio = _safe_float(candidate_row.get("expense_ratio"))
            if expense_ratio is None:
                expense_ratio = _safe_float(metadata.get("expense_ratio"))
            dividend_yield = _safe_float(candidate_row.get("dividend_yield"))
            if dividend_yield is None:
                dividend_yield = _safe_float(metadata.get("dividend_yield"))
            if candidate.is_income or "dividend" in candidate.primary_role.lower() or candidate.style_tilt.lower() == "income":
                delta += 12.0
                notes.append("it lines up with the current high-dividend ETF preference")
                if dividend_yield is not None and dividend_yield >= 0.025:
                    delta += min(8.0, dividend_yield * 120.0)
                if preferences.prefer_low_expense_for_dividend_etfs and expense_ratio is not None:
                    if expense_ratio <= 0.0006:
                        delta += 12.0
                        notes.append("its expense ratio is extremely low for a dividend ETF")
                    elif expense_ratio <= 0.0015:
                        delta += 6.0
                        notes.append("its expense ratio is still relatively low")
                    else:
                        delta -= 4.0
                        notes.append("its expense ratio is higher than the low-cost dividend preference")
                rel_1y = _safe_float(candidate_row.get("relative_1y_return_pct"))
                rel_3y = _safe_float(candidate_row.get("relative_3y_return_pct"))
                if rel_1y is not None and rel_1y > -0.08:
                    delta += 4.0
                if rel_3y is not None and rel_3y > -0.18:
                    delta += 4.0
            else:
                delta -= 2.0
    else:
        if not preferences.allow_single_stocks:
            return -999.0, ["Single stocks are turned off in the current buy preferences."]
        if preferences.single_stocks_preferred is True:
            delta += 8.0
            notes.append("it matches the current single-stock preference")

    candidate_sector_key = _normalized_sector_key(candidate.sector)
    preferred_sectors = {_normalized_sector_key(item) for item in (preferences.sector_preferences or [])}
    if preferred_sectors:
        if candidate_sector_key in preferred_sectors:
            delta += 10.0
            notes.append("it falls inside the sectors you prefer")
        elif candidate_sector_key != "broad market":
            delta -= 4.0

    avoided_sectors = {_normalized_sector_key(item) for item in (preferences.inferred_sector_avoidances or [])}
    if candidate_sector_key in avoided_sectors and candidate_sector_key not in preferred_sectors:
        delta -= 18.0
        notes.append("it sits in a sector the current portfolio already leans on heavily")

    return delta, notes


def _candidate_allocation_from_rank(
    *,
    candidate_rank: int,
    candidate_count: int,
    budget_to_deploy: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    """Suggest a simple starting budget split across the top buy ideas."""
    if budget_to_deploy is None or budget_to_deploy <= 0 or candidate_count <= 0:
        return None, None

    weights = [1.0 / (idx + 1) for idx in range(candidate_count)]
    total_weight = sum(weights)
    pct = weights[candidate_rank] / total_weight
    return round(pct, 4), round(float(budget_to_deploy) * pct, 2)


def _build_replacement_candidates(
    *,
    portfolio_gaps: list[PortfolioGap],
    portfolio_preferences: Optional[PortfolioPreferences],
    holding_action_recommendations: list[HoldingActionRecommendation],
    current_holdings: Optional[list[CurrentHoldingSnapshot]] = None,
) -> list[ReplacementCandidate]:
    """Build the first explanation-first set of buy ideas.

    This is not yet a full optimizer or a broad stock-search engine. It is a
    deterministic bridge from:

    - what the portfolio still needs
    - what the user currently allows
    - what the curated buy universe contains

    into a short list of candidates worth reviewing next.
    """
    if portfolio_preferences is None or not portfolio_gaps:
        return []

    entries, enriched = _load_buy_candidate_universe_inputs()
    if not entries or enriched.empty:
        return []

    actionable_tickers = {
        item.ticker
        for item in holding_action_recommendations
        if item.is_actionable
    }
    current_holding_map = {
        item.ticker: item for item in (current_holdings or [])
    }
    gaps_to_score = portfolio_gaps[:3]
    enriched_records = {
        str(row.get("ticker")): row for row in enriched.to_dict(orient="records")
    }

    scored_candidates: list[dict[str, Any]] = []
    for entry in entries:
        if not entry.eligible_for_buy_engine:
            continue
        if entry.ticker in actionable_tickers:
            continue
        if not portfolio_preferences.include_existing_holdings and entry.ticker in current_holding_map:
            continue
        row = enriched_records.get(entry.ticker, {})
        preference_adjustment, preference_notes = _preference_penalty_or_bonus(
            entry,
            portfolio_preferences,
            row,
        )
        if preference_adjustment <= -900:
            continue
        passes_filters, filter_notes = _candidate_passes_buy_filters(
            entry,
            row,
            portfolio_preferences,
        )
        if not passes_filters:
            continue

        best_gap: Optional[PortfolioGap] = None
        best_gap_score = -999.0
        best_gap_reasons: list[str] = []
        best_gap_evidence: list[str] = []
        for gap in gaps_to_score:
            gap_score, gap_reasons, gap_evidence = _gap_score_for_candidate(
                entry,
                gap,
                row,
                portfolio_preferences,
            )
            total_gap_score = gap_score + preference_adjustment
            if total_gap_score > best_gap_score:
                best_gap = gap
                best_gap_score = total_gap_score
                best_gap_reasons = gap_reasons + filter_notes
                best_gap_evidence = gap_evidence

        if best_gap is None:
            continue

        reason_parts = best_gap_reasons[:2] + preference_notes[:1]
        why_it_fits = (
            f"{entry.ticker} stands out because "
            + "; ".join(reason_parts)
            + "."
            if reason_parts
            else f"{entry.ticker} is a plausible fit for the current portfolio gap."
        )
        evidence_summary = _unique_nonempty(
            best_gap_evidence
            + [
                (
                    f"Known-universe support: appears in {entry.source_count} source list(s)"
                    if entry.source_count else None
                ),
                (
                    f"Universe source: {entry.universe_source}"
                    if entry.universe_source else None
                ),
                f"Primary role: {entry.primary_role}",
                f"Sector: {entry.sector}",
                (
                    f"1Y volatility: {_safe_float(row.get('annualized_volatility_1y')):.1%}"
                    if _safe_float(row.get("annualized_volatility_1y")) is not None else None
                ),
                (
                    f"1Y vs S&P 500: {_safe_float(row.get('relative_1y_return_pct')):.1%}"
                    if _safe_float(row.get("relative_1y_return_pct")) is not None else None
                ),
                (
                    f"3Y vs S&P 500: {_safe_float(row.get('relative_3y_return_pct')):.1%}"
                    if _safe_float(row.get("relative_3y_return_pct")) is not None else None
                ),
                (
                    f"5Y vs S&P 500: {_safe_float(row.get('relative_5y_return_pct')):.1%}"
                    if _safe_float(row.get("relative_5y_return_pct")) is not None else None
                ),
            ]
        )
        what_it_improves = _unique_nonempty(best_gap.what_would_change[:3])
        preference_fit_summary = (
            "Fits the current constraints by "
            + "; ".join(preference_notes)
            if preference_notes
            else "Fits the current constraints without breaking the vehicle or sector rules already in place."
        )
        scored_candidates.append(
            {
                "entry": entry,
                "row": row,
                "gap": best_gap,
                "fit_score": round(max(0.0, min(100.0, best_gap_score)), 1),
                "why_it_fits": why_it_fits,
                "what_it_improves": what_it_improves,
                "preference_fit_summary": preference_fit_summary,
                "evidence_summary": evidence_summary[:5],
            }
        )

    top_candidates = _select_replacement_candidate_slate(
        scored_candidates=scored_candidates,
        gaps_to_score=gaps_to_score,
        preferences=portfolio_preferences,
    )[: max(1, int(portfolio_preferences.buy_idea_limit or 10))]
    top_candidates = sorted(top_candidates, key=lambda item: (-float(item["fit_score"]), item["entry"].ticker))

    replacement_candidates: list[ReplacementCandidate] = []
    for rank, item in enumerate(top_candidates):
        allocation_pct, allocation_amount = _candidate_allocation_from_rank(
            candidate_rank=rank,
            candidate_count=len(top_candidates),
            budget_to_deploy=portfolio_preferences.budget_to_deploy,
        )
        confidence_score, confidence_band = _replacement_candidate_confidence(
            item["fit_score"],
            len(item["evidence_summary"]),
        )
        entry = item["entry"]
        row = item["row"]
        gap = item["gap"]
        replacement_candidates.append(
            ReplacementCandidate(
                ticker=entry.ticker,
                security_name=entry.security_name,
                asset_type=entry.asset_type,
                sector=entry.sector,
                primary_role=entry.primary_role,
                style_tilt=entry.style_tilt,
                linked_gap_key=gap.gap_key,
                linked_gap_label=gap.label,
                fit_score=item["fit_score"],
                fit_band=_fit_band(item["fit_score"]),
                why_it_fits=item["why_it_fits"],
                what_it_improves=item["what_it_improves"],
                preference_fit_summary=item["preference_fit_summary"],
                evidence_summary=item["evidence_summary"],
                universe_source=entry.universe_source,
                suggested_allocation_pct_of_budget=allocation_pct,
                suggested_allocation_amount=allocation_amount,
                relative_1y_return_pct=_safe_float(row.get("relative_1y_return_pct")),
                relative_3y_return_pct=_safe_float(row.get("relative_3y_return_pct")),
                relative_5y_return_pct=_safe_float(row.get("relative_5y_return_pct")),
                stock_5y_return_pct=_safe_float(row.get("stock_5y_return_pct")),
                annualized_volatility_1y=_safe_float(row.get("annualized_volatility_1y")),
                beta=_safe_float(row.get("beta")),
                expense_ratio=_safe_float(row.get("expense_ratio")) or _safe_float(fetch_candidate_market_metadata(entry.market_data_symbol).get("expense_ratio")),
                dividend_yield=_safe_float(row.get("dividend_yield")) or _safe_float(fetch_candidate_market_metadata(entry.market_data_symbol).get("dividend_yield")),
                trailing_pe=_safe_float(row.get("trailing_pe")) or _safe_float(fetch_candidate_market_metadata(entry.market_data_symbol).get("trailing_pe")),
                is_existing_holding=entry.ticker in current_holding_map,
                confidence_score=confidence_score,
                confidence_band=confidence_band,
            )
        )
    return replacement_candidates


def replacement_candidates_from_user_preferences(
    *,
    diagnosis: PortfolioRiskDiagnosis,
    preferences: PortfolioPreferences,
) -> list[ReplacementCandidate]:
    """Rebuild buy ideas after the user changes buy-side constraints.

    The dashboard should not need to rerun the whole portfolio analysis just to
    refresh buy ideas after a user changes the budget, vehicle mix, or sector
    preferences. This helper lets the app keep the diagnosis fixed while
    recomputing replacement candidates from the updated preferences object.
    """
    return _build_replacement_candidates(
        portfolio_gaps=diagnosis.portfolio_gaps,
        portfolio_preferences=preferences,
        holding_action_recommendations=diagnosis.holding_action_recommendations,
        current_holdings=diagnosis.current_holdings,
    )


def _build_holding_action_recommendations(
    bundle: dict[str, Any],
    holding_action_needs: list[HoldingActionNeed],
    holding_risk_contributions: list[HoldingRiskContribution],
    analysis_end: str,
) -> list[HoldingActionRecommendation]:
    """Turn diagnosis pressure into actual sell/trim guidance.

    This layer is stricter than `HoldingActionNeed`. A holding only becomes a
    sell-style recommendation when it has actually lagged the trade-matched
    S&P 500 over a clearly stated period. Diagnosis pressure and contribution
    evidence help size the trim, but they do not create a sell call on their own.
    """
    positions = bundle["open_positions"].copy()
    performance_context = bundle["holding_performance_context"].copy()
    volatility = bundle["volatility_drivers"].copy()
    if positions.empty:
        return []

    positions["current_weight"] = pd.to_numeric(positions.get("current_weight"), errors="coerce").fillna(0.0)
    positions["current_value"] = pd.to_numeric(positions.get("current_value"), errors="coerce")
    positions["quantity"] = pd.to_numeric(positions.get("quantity"), errors="coerce")
    positions["unrealized_return_pct"] = pd.to_numeric(positions.get("unrealized_return_pct"), errors="coerce")
    positions["benchmark_return_since_buy"] = pd.to_numeric(positions.get("benchmark_return_since_buy"), errors="coerce")
    positions["excess_return_vs_benchmark"] = pd.to_numeric(
        positions.get("excess_return_vs_benchmark"), errors="coerce"
    )
    if not volatility.empty:
        volatility = volatility[["ticker", "variance_contribution_pct"]].copy()
        volatility["variance_contribution_pct"] = pd.to_numeric(
            volatility.get("variance_contribution_pct"), errors="coerce"
        ).fillna(0.0)
        positions = positions.merge(volatility, on="ticker", how="left")
    positions["variance_contribution_pct"] = pd.to_numeric(
        positions.get("variance_contribution_pct"), errors="coerce"
    ).fillna(0.0)
    performance_lookup = {}
    if not performance_context.empty:
        for row in performance_context.to_dict(orient="records"):
            performance_lookup[str(row.get("ticker"))] = row

    action_need_lookup = {item.ticker: item for item in holding_action_needs}
    contribution_lookup = {item.ticker: item for item in holding_risk_contributions}
    analysis_end_ts = pd.to_datetime(analysis_end, errors="coerce")

    recommendations: list[HoldingActionRecommendation] = []
    for row in positions.to_dict(orient="records"):
        ticker = str(row.get("ticker"))
        sector = row.get("sector")
        is_fund = str(sector or "").strip() == "ETF / Fund"
        action_need = action_need_lookup.get(ticker)
        contribution = contribution_lookup.get(ticker)
        weighted_avg_buy_date = row.get("weighted_avg_buy_date")
        buy_date_ts = pd.to_datetime(weighted_avg_buy_date, errors="coerce")
        holding_days = None
        if pd.notna(analysis_end_ts) and pd.notna(buy_date_ts):
            holding_days = int((analysis_end_ts - buy_date_ts).days)

        current_weight = _safe_float(row.get("current_weight")) or 0.0
        current_value = _safe_float(row.get("current_value"))
        quantity = _safe_float(row.get("quantity"))
        unrealized_return_pct = _safe_float(row.get("unrealized_return_pct"))
        benchmark_return_since_buy = _safe_float(row.get("benchmark_return_since_buy"))
        excess_return_vs_benchmark = _safe_float(row.get("excess_return_vs_benchmark"))
        variance_contribution_pct = _safe_float(row.get("variance_contribution_pct")) or 0.0
        diagnosis_pressure_score = (
            action_need.action_pressure_score if action_need is not None else 0.0
        )
        performance_row = performance_lookup.get(ticker, {})
        underperform_windows, outperform_windows, trailing_window_evidence = _relative_window_readout(performance_row)
        linked_primary_concern = (
            action_need.linked_primary_concern
            if action_need is not None
            else contribution.primary_concern_label if contribution is not None else None
        )
        explicit_sell_modifiers, modifier_score = _build_explicit_sell_modifiers(
            action_need=action_need,
            linked_primary_concern=linked_primary_concern,
        )

        if is_fund:
            recommendation_label = "Hold for now"
            recommendation_code = "fund_hold_for_now"
            reduction_pct = 0.0
            reasoning_points = [
                "This is an ETF/fund position, so the stock-specific underperformance rule is not being used to force a sell recommendation here.",
                "Broad fund positions usually need portfolio-construction review rather than single-stock sell logic.",
            ]
        elif excess_return_vs_benchmark is None:
            recommendation_label = "Hold for now"
            recommendation_code = "missing_relative_performance"
            reduction_pct = 0.0
            reasoning_points = [
                "Relative performance versus the S&P 500 was not available, so the system is not making a sell-style recommendation.",
            ]
        elif holding_days is not None and holding_days < 120:
            recommendation_label = "Hold for now"
            recommendation_code = "holding_period_too_short"
            reduction_pct = 0.0
            reasoning_points = [
                f"The weighted-average holding period is only about {holding_days} days, which is too short for this sell rule to treat as durable underperformance.",
            ]
        elif excess_return_vs_benchmark >= -0.10:
            recommendation_label = "Hold for now"
            recommendation_code = "not_enough_underperformance"
            reduction_pct = 0.0
            reasoning_points = [
                "The holding has not lagged the trade-matched S&P 500 by enough to justify trimming under the current rule set.",
            ]
        elif underperform_windows == 0 and outperform_windows >= 1:
            recommendation_label = "Hold for now"
            recommendation_code = "long_horizon_support_still_positive"
            reduction_pct = 0.0
            reasoning_points = [
                "The holding has lagged since your buy date, but the trailing 1Y/3Y/5Y comparison still shows enough longer-horizon support that the system is not issuing a sell recommendation yet.",
            ]
        else:
            recommendation_label, recommendation_code, reduction_pct = _recommendation_reduction_from_underperformance(
                excess_return_vs_benchmark=excess_return_vs_benchmark,
                current_weight=current_weight,
                diagnosis_pressure_score=diagnosis_pressure_score,
                variance_contribution_pct=variance_contribution_pct,
            )
            if reduction_pct == 0 and modifier_score >= 0.10 and (excess_return_vs_benchmark or 0.0) <= -0.12:
                reduction_pct = 0.10 if modifier_score < 0.16 else 0.15
                recommendation_code = "external_modifiers_trigger_trim"
            elif reduction_pct > 0 and modifier_score >= 0.10:
                reduction_pct = min(1.0, reduction_pct + 0.05)
                recommendation_code = "external_modifiers_strengthen_trim"
                if modifier_score >= 0.18:
                    reduction_pct = min(1.0, reduction_pct + 0.05)
                    recommendation_code = "external_modifiers_strongly_strengthen_trim"
            elif reduction_pct > 0 and modifier_score >= 0.05 and underperform_windows >= 1:
                reduction_pct = min(1.0, reduction_pct + 0.05)
                recommendation_code = "external_modifiers_support_existing_trim"
            if underperform_windows >= 2:
                escalation = 0.10 if underperform_windows == 2 else 0.15
                if diagnosis_pressure_score >= 45 or current_weight >= 0.08 or variance_contribution_pct >= 0.02:
                    escalation += 0.05
                reduction_pct = min(1.0, reduction_pct + escalation)
                recommendation_code = "persistent_underperformance_escalation"
            elif underperform_windows >= 1 and (diagnosis_pressure_score >= 50 or current_weight >= 0.08):
                reduction_pct = min(1.0, reduction_pct + 0.05)
                recommendation_code = "persistent_underperformance_supports_trim"
            if outperform_windows >= 2:
                reduction_pct = max(0.0, reduction_pct - 0.10)
                if reduction_pct == 0 and (excess_return_vs_benchmark or 0.0) <= -0.45:
                    reduction_pct = 0.10
                    recommendation_code = "long_horizon_strength_softens_but_does_not_cancel_trim"
                if reduction_pct == 0:
                    recommendation_label = "Hold for now"
                    recommendation_code = "long_horizon_outperformance_offsets_since_buy_lag"
            if _should_sell_all_recommendation(
                excess_return_vs_benchmark=excess_return_vs_benchmark,
                underperform_windows=underperform_windows,
                outperform_windows=outperform_windows,
                current_weight=current_weight,
                diagnosis_pressure_score=diagnosis_pressure_score,
                modifier_score=modifier_score,
            ):
                reduction_pct = 1.0
                recommendation_code = "sell_all_persistent_underperformer"
            reduction_pct = _normalize_recommendation_reduction(reduction_pct)
            recommendation_label = _recommendation_label_from_reduction(reduction_pct)
            reasoning_points = [
                f"Since {weighted_avg_buy_date or 'the weighted-average buy date'}, the holding has lagged the trade-matched S&P 500 by {(excess_return_vs_benchmark or 0.0):.1%}.",
            ]
            if diagnosis_pressure_score > 0:
                reasoning_points.append(
                    f"It also carries {diagnosis_pressure_score:.1f}/100 diagnosis pressure, which makes trimming easier to defend."
                )
            if variance_contribution_pct >= 0.02:
                reasoning_points.append(
                    f"It contributed {variance_contribution_pct:.1%} of the tracked 2025 portfolio variance, so it is not just a weak performer in isolation."
                )
            if underperform_windows > 0:
                reasoning_points.append(
                    f"It has also trailed the S&P 500 in {underperform_windows} of the trailing 1Y/3Y/5Y windows that were available."
                )
            if current_weight >= 0.08 and excess_return_vs_benchmark <= -0.15:
                reasoning_points.append(
                    f"It is also a large enough position at {current_weight:.1%} of the portfolio that even moderate underperformance can do real damage."
                )
            if explicit_sell_modifiers:
                reasoning_points.append(
                    f"External market and company signals added {modifier_score:.0%} of extra trim pressure on top of the performance case."
                )
            if outperform_windows > 0:
                reasoning_points.append(
                    f"At the same time, it has still beaten the S&P 500 in {outperform_windows} trailing windows, so the recommendation is moderated rather than absolute."
                )
            if reduction_pct >= 0.999:
                reasoning_points.append(
                    "The weakness is now persistent enough across the longer trailing windows that the system no longer sees a strong reason to keep a small residual position."
                )

        shares_to_sell = round((quantity or 0.0) * reduction_pct, 4) if quantity is not None else None
        value_to_sell = round((current_value or 0.0) * reduction_pct, 2) if current_value is not None else None
        target_weight_after_action = round(current_weight * (1.0 - reduction_pct), 4) if current_weight is not None else None
        is_actionable = reduction_pct > 0
        (
            projected_weight_reduction_pct_points,
            projected_variance_reduction_pct_points,
            projected_relative_drag_reduction_pct_points,
            portfolio_impact_summary,
            portfolio_impact_bullets,
        ) = _build_portfolio_impact_preview(
            ticker=ticker,
            current_weight=current_weight,
            reduction_pct=reduction_pct,
            variance_contribution_pct=variance_contribution_pct,
            excess_return_vs_benchmark=excess_return_vs_benchmark,
        )
        confidence_score, confidence_band = _recommendation_confidence(
            holding_days=holding_days,
            diagnosis_pressure_score=diagnosis_pressure_score,
            variance_contribution_pct=variance_contribution_pct,
            is_actionable=is_actionable,
        )

        supporting_evidence = _unique_nonempty(
            [
                f"Holding return since weighted-average buy date: {(unrealized_return_pct or 0.0):.1%}",
                f"Benchmark return over the same period: {(benchmark_return_since_buy or 0.0):.1%}",
                f"Relative performance vs benchmark: {(excess_return_vs_benchmark or 0.0):.1%}",
                f"Diagnosis pressure: {diagnosis_pressure_score:.1f}/100" if diagnosis_pressure_score else None,
                f"Variance contribution (2025): {variance_contribution_pct:.1%}" if variance_contribution_pct else None,
            ]
            + trailing_window_evidence
            + (list(getattr(action_need, "supporting_evidence", [])[:2]) if action_need is not None else [])
        )
        what_changed, why_it_matters, amount_rationale = _build_recommendation_explanation_parts(
            ticker=ticker,
            recommendation_label=recommendation_label,
            reduction_pct=reduction_pct,
            weighted_avg_buy_date=weighted_avg_buy_date,
            excess_return_vs_benchmark=excess_return_vs_benchmark,
            diagnosis_pressure_score=diagnosis_pressure_score,
            linked_primary_concern=linked_primary_concern,
            underperform_windows=underperform_windows,
            outperform_windows=outperform_windows,
            current_weight=current_weight,
            variance_contribution_pct=variance_contribution_pct,
            is_actionable=is_actionable,
        )
        guardrail_notes = _build_recommendation_guardrail_notes(
            is_fund=is_fund,
            holding_days=holding_days,
            outperform_windows=outperform_windows,
            underperform_windows=underperform_windows,
            diagnosis_pressure_score=diagnosis_pressure_score,
            current_weight=current_weight,
            reduction_pct=reduction_pct,
        )

        recommendations.append(
            HoldingActionRecommendation(
                ticker=ticker,
                sector=sector,
                current_weight=current_weight,
                current_value=current_value,
                quantity=quantity,
                recommendation_label=recommendation_label,
                recommendation_code=recommendation_code,
                position_reduction_pct=reduction_pct,
                shares_to_sell=shares_to_sell,
                value_to_sell=value_to_sell,
                target_weight_after_action=target_weight_after_action,
                projected_weight_reduction_pct_points=projected_weight_reduction_pct_points,
                projected_variance_reduction_pct_points=projected_variance_reduction_pct_points,
                projected_relative_drag_reduction_pct_points=projected_relative_drag_reduction_pct_points,
                performance_window_start=weighted_avg_buy_date,
                performance_window_end=analysis_end,
                performance_window_label=(
                    f"Since weighted-average buy date ({weighted_avg_buy_date}) through {analysis_end}"
                    if weighted_avg_buy_date
                    else f"Through {analysis_end}"
                ),
                holding_return_pct=unrealized_return_pct,
                benchmark_return_pct=benchmark_return_since_buy,
                relative_performance_vs_benchmark=excess_return_vs_benchmark,
                stock_1y_return_pct=_safe_float(performance_row.get("stock_1y_return_pct")),
                benchmark_1y_return_pct=_safe_float(performance_row.get("benchmark_1y_return_pct")),
                relative_1y_return_pct=_safe_float(performance_row.get("relative_1y_return_pct")),
                stock_3y_return_pct=_safe_float(performance_row.get("stock_3y_return_pct")),
                benchmark_3y_return_pct=_safe_float(performance_row.get("benchmark_3y_return_pct")),
                relative_3y_return_pct=_safe_float(performance_row.get("relative_3y_return_pct")),
                stock_5y_return_pct=_safe_float(performance_row.get("stock_5y_return_pct")),
                benchmark_5y_return_pct=_safe_float(performance_row.get("benchmark_5y_return_pct")),
                relative_5y_return_pct=_safe_float(performance_row.get("relative_5y_return_pct")),
                linked_action_need_label=action_need.action_label if action_need is not None else None,
                linked_primary_concern=linked_primary_concern,
                diagnosis_pressure_score=diagnosis_pressure_score,
                recommendation_summary=_build_recommendation_summary(
                    ticker=ticker,
                    recommendation_label=recommendation_label,
                    reduction_pct=reduction_pct,
                    weighted_avg_buy_date=weighted_avg_buy_date,
                    unrealized_return_pct=unrealized_return_pct,
                    benchmark_return_since_buy=benchmark_return_since_buy,
                    excess_return_vs_benchmark=excess_return_vs_benchmark,
                    diagnosis_pressure_score=diagnosis_pressure_score,
                    linked_primary_concern=linked_primary_concern,
                    is_actionable=is_actionable,
                ),
                what_changed=what_changed,
                why_it_matters=why_it_matters,
                amount_rationale=amount_rationale,
                explicit_sell_modifiers=explicit_sell_modifiers,
                modifier_score=modifier_score,
                portfolio_impact_summary=portfolio_impact_summary,
                portfolio_impact_bullets=portfolio_impact_bullets,
                reasoning_points=reasoning_points,
                supporting_evidence=supporting_evidence[:6],
                guardrail_notes=guardrail_notes,
                confidence_score=confidence_score,
                confidence_band=confidence_band,
                is_actionable=is_actionable,
            )
        )

    recommendations.sort(
        key=lambda item: (
            not item.is_actionable,
            -(item.position_reduction_pct or 0.0),
            -(abs(item.relative_performance_vs_benchmark) if item.relative_performance_vs_benchmark is not None else 0.0),
            -(item.current_weight or 0.0),
            item.ticker,
        )
    )
    return recommendations


def _build_diagnostic_summary(bundle: dict[str, Any], top_concerns: list[DiagnosisConcern]) -> str:
    """Create the top-level diagnosis narrative for the portfolio.

    This summary is intended to read like the opening paragraph of a trustworthy
    explanation: observed risk, stated risk, top diagnosis categories, analysis
    window, and the main external reasons that reinforced the read.
    """
    risk = bundle["risk"]
    headline = bundle["headline"]
    concern_labels = ", ".join(concern.label.lower() for concern in top_concerns[:2]) or "overall portfolio risk"
    adjustment_clause = ""
    if top_concerns:
        top_adjustments = [
            reason
            for concern in top_concerns[:2]
            for reason in concern.adjustment_reasons[:2]
        ]
        if top_adjustments:
            adjustment_clause = " External evidence reinforced this read through " + "; ".join(top_adjustments[:3]) + "."
    return (
        f"Observed portfolio risk is {risk['score']:.1f}/100 ({risk['band']}) versus a stated risk of "
        f"{risk['stated_score']:.1f}/100 ({risk['stated_band']}). The portfolio currently looks "
        f"{risk['alignment'].replace('Observed portfolio risk is ', '').lower()}. "
        f"The main diagnosis is driven by {concern_labels}, over the analysis window "
        f"{headline['analysis_start']} to {headline['analysis_end']}.{adjustment_clause}"
    )


def _build_evidence_gaps(bundle: dict[str, Any], macro_context: Optional[MacroRegimeSnapshot], narrative_evidence: list[NarrativeEvidence]) -> list[str]:
    """Describe what the diagnosis still cannot claim confidently.

    A financial diagnosis object should be honest about its limits. These gaps
    are not errors; they are explicit reminders of what the current system still
    does not model deeply enough for production-grade recommendations.
    """
    gaps = [
        "User goals and constraints beyond the stated risk score are not yet modeled in the diagnosis object.",
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
    """Build the full `PortfolioRiskDiagnosis` object from saved pipeline artifacts.

    Parameters
    ----------
    base_dir:
        Directory containing the app-generated diagnosis bundle written during a
        dashboard analysis run.

    Returns
    -------
    PortfolioRiskDiagnosis
        A typed diagnosis object that combines:
        - portfolio risk measurements from the app
        - holding and sector driver attribution
        - macro regime context
        - fundamental snapshots
        - narrative evidence
        - source coverage and known evidence gaps

    Notes
    -----
    This is the current public entrypoint for the diagnosis engine. It is the
    function notebooks and later application layers should call when they want a
    structured diagnosis object instead of raw dataframes.
    """
    bundle = load_diagnosis_bundle(base_dir)
    supporting_metrics = _build_metric_observations(bundle)
    macro_context = _build_macro_context(bundle)
    candidate_rows = _candidate_driver_rows(bundle, limit=10)
    candidate_tickers = [str(row.get("ticker")) for row in candidate_rows if row.get("ticker")]
    holding_fundamentals = _build_holding_fundamentals(bundle, candidate_tickers)
    narrative_evidence = _build_narrative_evidence(bundle, candidate_tickers)
    top_holding_drivers = _build_holding_drivers(
        bundle,
        macro_context,
        holding_fundamentals,
        narrative_evidence,
    )
    top_sector_drivers = _build_sector_drivers(bundle)
    top_concerns = _build_top_concerns(
        bundle,
        top_holding_drivers,
        top_sector_drivers,
        macro_context,
        holding_fundamentals,
        narrative_evidence,
    )
    holding_risk_contributions = _build_holding_risk_contributions(
        top_holding_drivers,
        top_concerns,
    )
    holding_action_needs = _build_holding_action_needs(
        holding_risk_contributions,
    )
    manifest = bundle["manifest"]
    headline = bundle["headline"]
    risk = bundle["risk"]
    holding_action_recommendations = _build_holding_action_recommendations(
        bundle,
        holding_action_needs,
        holding_risk_contributions,
        analysis_end=str(headline.get("analysis_end")),
    )
    portfolio_action_impact = _build_portfolio_action_impact(
        bundle,
        holding_action_recommendations,
    )
    portfolio_preferences = _build_portfolio_preferences(
        bundle,
        portfolio_action_impact,
        top_sector_drivers,
    )
    current_holdings = _build_current_holdings(bundle)
    portfolio_gaps = _build_portfolio_gaps(
        bundle,
        top_concerns,
        top_sector_drivers,
        portfolio_action_impact,
    )
    replacement_candidates = _build_replacement_candidates(
        portfolio_gaps=portfolio_gaps,
        portfolio_preferences=portfolio_preferences,
        holding_action_recommendations=holding_action_recommendations,
        current_holdings=current_holdings,
    )
    portfolio_rebalance_plan = _build_portfolio_rebalance_plan(
        current_holdings=current_holdings,
        portfolio_preferences=portfolio_preferences,
        holding_action_recommendations=holding_action_recommendations,
        replacement_candidates=replacement_candidates,
    )

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
        holding_risk_contributions=holding_risk_contributions,
        holding_action_needs=holding_action_needs,
        holding_action_recommendations=holding_action_recommendations,
        portfolio_action_impact=portfolio_action_impact,
        portfolio_rebalance_plan=portfolio_rebalance_plan,
        portfolio_gaps=portfolio_gaps,
        portfolio_preferences=portfolio_preferences,
        current_holdings=current_holdings,
        replacement_candidates=replacement_candidates,
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

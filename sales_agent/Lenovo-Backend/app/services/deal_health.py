"""Deal Health calculator — pure scoring functions.

Implements the five-component formula from the "Deal Detailed View" user
story. Every threshold/weight/cadence comes from `DealHealthSettings`, which
is loaded from the `lvo_dealhealthconfig` row by the orchestrator and passed
in here. That keeps these functions independent of SQLAlchemy / Postgres so
they are trivially unit-testable.

Final score formula (composite):

    score = sum(weight_i * component_score_i) / 100

Components (defaults, all configurable):

    Stage Progress (25%)         — average of stage-position and stage-velocity
    Activity Freshness (25%)     — last activity vs tempo-class cadence
    Stakeholder (20%)            — coverage * threading factor
    Close Date Confidence (20%)  — time-progress vs expected stage progress
    Risk Adjustment (10%)        — start at 100, subtract 20 per active risk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

HealthBand = Literal["GREEN", "YELLOW", "RED"]

ComponentKey = Literal[
    "stage_progress",
    "activity_freshness",
    "stakeholder",
    "close_confidence",
    "risk_adjustment",
]


@dataclass(frozen=True)
class DealHealthSettings:
    """Tunables for the calculator. Mirrors the JSON in lvo_dealhealthconfig.

    Construct via `DealHealthSettings.from_config_dict(...)` so the same
    JSON shape stored in the DB is the canonical input — no field-by-field
    re-mapping in the orchestrator.
    """

    weights: dict[str, int]
    stage_position_score: dict[str, int]
    stage_time_distribution: dict[str, float]
    close_confidence_distribution: dict[str, float]
    tempo_class_target_days: dict[str, int]
    tempo_class_cadence_days: dict[str, int]
    stakeholder_required_count: int
    stakeholder_threading_factor: dict[str, float]
    low_stakeholder_threshold: int
    low_activity_days_threshold: int
    risk_penalty_per_risk: int
    health_band_thresholds: dict[str, int]

    @classmethod
    def from_config_dict(cls, raw: dict[str, Any]) -> "DealHealthSettings":
        """Build from the JSON shape stored in lvo_dealhealthconfig.lvo_settings."""
        return cls(
            weights=dict(raw["weights"]),
            stage_position_score=dict(raw["stage_position_score"]),
            stage_time_distribution=dict(raw["stage_time_distribution"]),
            close_confidence_distribution=dict(raw["close_confidence_distribution"]),
            tempo_class_target_days=dict(raw["tempo_class_target_days"]),
            tempo_class_cadence_days=dict(raw["tempo_class_cadence_days"]),
            stakeholder_required_count=int(raw["stakeholder_required_count"]),
            stakeholder_threading_factor=dict(raw["stakeholder_threading_factor"]),
            low_stakeholder_threshold=int(raw["low_stakeholder_threshold"]),
            low_activity_days_threshold=int(raw["low_activity_days_threshold"]),
            risk_penalty_per_risk=int(raw["risk_penalty_per_risk"]),
            health_band_thresholds=dict(raw["health_band_thresholds"]),
        )


@dataclass(frozen=True)
class DealHealthBreakdown:
    """Structured result returned by `compose_deal_health`.

    `components` matches the shape of the API's DealHealthInfo.components dict
    so the orchestrator can serialise it directly without remapping.
    """

    score: int
    band: HealthBand
    components: dict[ComponentKey, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default settings (mirror of sql/2026_06_create_dealhealth_config.sql).
# Used as a fallback by the orchestrator when the config row is missing —
# ensures the API never 500s just because the seed migration wasn't run.
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS = DealHealthSettings.from_config_dict(
    {
        "weights": {
            "stage_progress": 25,
            "activity_freshness": 25,
            "stakeholder": 20,
            "close_confidence": 20,
            "risk_adjustment": 10,
        },
        "stage_position_score": {
            "Qualify": 20,
            "Develop": 40,
            "Propose": 60,
            "Execute": 80,
            "Closed Won": 100,
            "Closed Lost": 0,
        },
        "stage_time_distribution": {
            "Qualify": 0.20,
            "Develop": 0.30,
            "Propose": 0.20,
            "Execute": 0.30,
        },
        "close_confidence_distribution": {
            "Qualify": 0.25,
            "Develop": 0.25,
            "Propose": 0.30,
            "Execute": 0.20,
        },
        "tempo_class_target_days": {
            "Fast": 30,
            "Quarterly": 90,
            "Programmatic": 365,
            "Strategic": 730,
        },
        "tempo_class_cadence_days": {
            "Fast": 4,
            "Quarterly": 10,
            "Programmatic": 20,
            "Strategic": 40,
        },
        "stakeholder_required_count": 5,
        "stakeholder_threading_factor": {"1": 0.60, "2": 0.80, "3+": 1.00},
        "low_stakeholder_threshold": 40,
        "low_activity_days_threshold": 14,
        "risk_penalty_per_risk": 20,
        "health_band_thresholds": {"GREEN": 75, "YELLOW": 50},
    }
)

# Tempo-class fallback. The DB constraint enforces these four exact values
# but defensive lookups go through this map so a stray legacy value (e.g.
# "Quarterly / Enterprise" written by a script) still resolves.
_TEMPO_ALIASES: dict[str, str] = {
    "fast": "Fast",
    "fast / transactional": "Fast",
    "quarterly": "Quarterly",
    "quarterly / enterprise": "Quarterly",
    "programmatic": "Programmatic",
    "programmatic / annual": "Programmatic",
    "strategic": "Strategic",
    "strategic / multiyear": "Strategic",
}


def _normalise_tempo_class(raw: str | None) -> str:
    """Map any legacy tempo-class label to one of the four canonical values.

    Falls back to ``"Quarterly"`` per the Phase-1 plan decision so the
    calculator never raises KeyError for legacy/unset rows.
    """
    if raw:
        canon = _TEMPO_ALIASES.get(raw.strip().lower())
        if canon:
            return canon
    return "Quarterly"


# ---------------------------------------------------------------------------
# Stage Progress  (25%)
# ---------------------------------------------------------------------------


def stage_position_score(stage: str | None, settings: DealHealthSettings) -> float:
    """Return the configured 0–100 score for the given stage (case-sensitive).

    Unknown stages collapse to 0 — never raise. The user story explicitly
    enumerates Qualify/Develop/Propose/Execute/Closed Won/Closed Lost and
    every other value is treated as "no progress data".
    """
    if stage is None:
        return 0.0
    return float(settings.stage_position_score.get(stage, 0))


def stage_velocity_score(actual_days: float, expected_days: float) -> float:
    """Map velocity = actual / expected to a 0–100 score per the user story.

    Bands:
        velocity ≤ 1.00            → 100
        1.01 ≤ velocity ≤ 1.25     →  80
        1.26 ≤ velocity ≤ 1.50     →  60
        1.51 ≤ velocity ≤ 2.00     →  40
                  > 2.00            →  20

    A non-positive `expected_days` is impossible per the spec; we treat it
    as "no velocity signal" and return 100 (i.e. "not late") rather than
    raising — keeps the API forgiving of malformed config.
    """
    if expected_days <= 0:
        return 100.0
    velocity = max(actual_days, 0) / expected_days
    if velocity <= 1.0:
        return 100.0
    if velocity <= 1.25:
        return 80.0
    if velocity <= 1.50:
        return 60.0
    if velocity <= 2.00:
        return 40.0
    return 20.0


def expected_days_in_stage(
    stage: str | None,
    target_close_total_days: float,
    settings: DealHealthSettings,
) -> float:
    """Days the deal *should* spend in `stage` given its tempo class."""
    if stage is None:
        return 0.0
    fraction = settings.stage_time_distribution.get(stage, 0.0)
    return float(target_close_total_days) * float(fraction)


def stage_progress_score(
    stage: str | None,
    actual_days_in_stage: float,
    expected_days: float,
    settings: DealHealthSettings,
) -> tuple[float, dict[str, Any]]:
    """Combined Stage Progress component score — average of position + velocity."""
    position = stage_position_score(stage, settings)
    velocity = stage_velocity_score(actual_days_in_stage, expected_days)
    score = 0.5 * position + 0.5 * velocity
    inputs = {
        "stage": stage,
        "stagePositionScore": position,
        "stageVelocityScore": velocity,
        "actualDaysInStage": actual_days_in_stage,
        "expectedDaysInStage": expected_days,
    }
    return score, inputs


# ---------------------------------------------------------------------------
# Activity Freshness  (25%)
# ---------------------------------------------------------------------------


def activity_freshness_score(
    activity_age_days: float | None,
    tempo_class: str | None,
    settings: DealHealthSettings,
) -> tuple[float, dict[str, Any]]:
    """Map (age / cadence-midpoint) ratio to a 0–100 score.

    Bands per the user story:
        ratio ≤ 1.0   → 100  (Healthy / On track)
        1.01–2.0      →  70  (Slight delay / Monitor)
        2.01–3.0      →  40  (Risky)
              > 3.0   →  20  (Critical / No engagement)

    A deal with **no activity at all** is treated as "Critical" → 20.
    """
    canonical_tempo = _normalise_tempo_class(tempo_class)
    cadence = float(
        settings.tempo_class_cadence_days.get(canonical_tempo)
        or settings.tempo_class_cadence_days["Quarterly"]
    )
    inputs: dict[str, Any] = {
        "tempoClass": canonical_tempo,
        "cadenceDays": cadence,
        "activityAgeDays": activity_age_days,
    }

    if activity_age_days is None:
        # No activity ever — treat as critical.
        inputs["ratio"] = None
        return 20.0, inputs

    if cadence <= 0:
        inputs["ratio"] = None
        return 100.0, inputs

    ratio = max(activity_age_days, 0) / cadence
    inputs["ratio"] = ratio
    if ratio <= 1.0:
        return 100.0, inputs
    if ratio <= 2.0:
        return 70.0, inputs
    if ratio <= 3.0:
        return 40.0, inputs
    return 20.0, inputs


# ---------------------------------------------------------------------------
# Stakeholder  (20%)
# ---------------------------------------------------------------------------


def stakeholder_score(
    active_stakeholders: int,
    settings: DealHealthSettings,
) -> tuple[float, dict[str, Any]]:
    """Coverage % of required stakeholders, scaled by threading factor.

    `required_count` defaults to 5 per the user story and is configurable.
    """
    required = max(int(settings.stakeholder_required_count), 1)
    coverage = min(active_stakeholders, required) / required * 100.0

    if active_stakeholders <= 0:
        threading_key = "1"  # treat as worst case
    elif active_stakeholders == 1:
        threading_key = "1"
    elif active_stakeholders == 2:
        threading_key = "2"
    else:
        threading_key = "3+"
    threading_factor = float(
        settings.stakeholder_threading_factor.get(threading_key, 1.0)
    )

    score = coverage * threading_factor
    inputs = {
        "activeStakeholders": active_stakeholders,
        "requiredStakeholders": required,
        "coverageScore": coverage,
        "threadingFactor": threading_factor,
    }
    return score, inputs


# ---------------------------------------------------------------------------
# Close Date Confidence  (20%)
# ---------------------------------------------------------------------------


def close_date_confidence_score(
    today: date,
    created_at: datetime | None,
    close_date: date | None,
    stage: str | None,
    settings: DealHealthSettings,
    *,
    is_closed: bool,
) -> tuple[float, dict[str, Any]]:
    """Score the alignment of elapsed time with where the stage *should* be.

    Per the user story:
        Time progress       = elapsed / total
        Expected progress   = cumulative sum of stage_time_distribution up to
                              and including the current stage
        Gap                 = time_progress - expected_progress
        Score:
            < 10%   → 100   strong alignment
            10–25%  →  70   slight misalignment
            25–40%  →  40   risky
            > 40%   →   0   unrealistic

    Special-cases:
        * Already closed deals → 100 (the score isn't relevant)
        * close_date < today (overdue) → 0 (explicit user-story rule)
        * Missing inputs → 0 (we can't make a confident statement)
    """
    inputs: dict[str, Any] = {
        "today": today.isoformat(),
        "closeDate": close_date.isoformat() if close_date else None,
        "stage": stage,
    }

    if is_closed:
        inputs["reason"] = "deal_closed"
        return 100.0, inputs

    if close_date is None:
        inputs["reason"] = "no_close_date"
        return 0.0, inputs

    if close_date < today:
        inputs["reason"] = "close_date_overdue"
        return 0.0, inputs

    if created_at is None:
        inputs["reason"] = "no_created_date"
        return 0.0, inputs

    created_date = created_at.date()
    total_days = (close_date - created_date).days
    elapsed_days = (today - created_date).days
    if total_days <= 0:
        inputs["reason"] = "non_positive_total"
        return 100.0, inputs

    time_progress = elapsed_days / total_days

    # Cumulative expected progress up to and including the current stage.
    stage_order = ["Qualify", "Develop", "Propose", "Execute"]
    cum = 0.0
    expected_progress = 0.0
    for s in stage_order:
        cum += float(settings.close_confidence_distribution.get(s, 0.0))
        if s == stage:
            expected_progress = cum
            break
    else:
        # Stage not in the open-deal list (e.g. Closed Won/Lost).
        expected_progress = cum

    gap = time_progress - expected_progress
    inputs.update(
        {
            "elapsedDays": elapsed_days,
            "totalDays": total_days,
            "timeProgress": time_progress,
            "expectedProgress": expected_progress,
            "gap": gap,
        }
    )

    abs_gap = abs(gap)
    if abs_gap < 0.10:
        return 100.0, inputs
    if abs_gap < 0.25:
        return 70.0, inputs
    if abs_gap < 0.40:
        return 40.0, inputs
    return 0.0, inputs


# ---------------------------------------------------------------------------
# Risk Adjustment  (10%)
# ---------------------------------------------------------------------------


def risk_adjustment_score(
    risk_count: int,
    settings: DealHealthSettings,
) -> tuple[float, dict[str, Any]]:
    """Start at 100, subtract risk_penalty_per_risk for each active risk.

    Floored at 0 so a deal with 6 risks doesn't go negative.
    """
    penalty = max(int(risk_count), 0) * int(settings.risk_penalty_per_risk)
    score = max(100.0 - penalty, 0.0)
    return score, {"riskCount": int(risk_count), "penaltyPerRisk": int(settings.risk_penalty_per_risk)}


# ---------------------------------------------------------------------------
# Composite — combine all five components into a final 0–100 score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DealHealthInputs:
    """Everything the calculator needs in one struct.

    The orchestrator builds this from the DB row + dependents; tests build
    it directly. Keeps the composition function pure.
    """

    stage: str | None
    tempo_class: str | None
    actual_days_in_stage: float
    target_close_total_days: float
    last_activity_age_days: float | None
    active_stakeholders: int
    created_at: datetime | None
    close_date: date | None
    is_closed: bool
    risk_count: int
    today: date | None = None  # Override only in tests; otherwise UTC today.


def compose_deal_health(
    inputs: DealHealthInputs,
    settings: DealHealthSettings = DEFAULT_SETTINGS,
) -> DealHealthBreakdown:
    """Run all five components, weight them, return the structured breakdown."""
    today = inputs.today or datetime.now(timezone.utc).date()

    expected_stage_days = expected_days_in_stage(
        inputs.stage, inputs.target_close_total_days, settings
    )
    stage_score, stage_inputs = stage_progress_score(
        inputs.stage,
        inputs.actual_days_in_stage,
        expected_stage_days,
        settings,
    )

    activity_score, activity_inputs = activity_freshness_score(
        inputs.last_activity_age_days, inputs.tempo_class, settings
    )

    stake_score, stake_inputs = stakeholder_score(
        inputs.active_stakeholders, settings
    )

    confidence_score, confidence_inputs = close_date_confidence_score(
        today,
        inputs.created_at,
        inputs.close_date,
        inputs.stage,
        settings,
        is_closed=inputs.is_closed,
    )

    risk_score, risk_inputs = risk_adjustment_score(inputs.risk_count, settings)

    weights = settings.weights
    weighted_total = (
        stage_score * weights["stage_progress"]
        + activity_score * weights["activity_freshness"]
        + stake_score * weights["stakeholder"]
        + confidence_score * weights["close_confidence"]
        + risk_score * weights["risk_adjustment"]
    ) / 100.0

    final_score = max(0, min(100, round(weighted_total)))
    band = health_band(final_score, settings)

    components: dict[ComponentKey, dict[str, Any]] = {
        "stage_progress": {
            "weight": weights["stage_progress"],
            "score": round(stage_score, 2),
            "inputs": stage_inputs,
        },
        "activity_freshness": {
            "weight": weights["activity_freshness"],
            "score": round(activity_score, 2),
            "inputs": activity_inputs,
        },
        "stakeholder": {
            "weight": weights["stakeholder"],
            "score": round(stake_score, 2),
            "inputs": stake_inputs,
        },
        "close_confidence": {
            "weight": weights["close_confidence"],
            "score": round(confidence_score, 2),
            "inputs": confidence_inputs,
        },
        "risk_adjustment": {
            "weight": weights["risk_adjustment"],
            "score": round(risk_score, 2),
            "inputs": risk_inputs,
        },
    }

    return DealHealthBreakdown(score=final_score, band=band, components=components)


def health_band(score: int, settings: DealHealthSettings = DEFAULT_SETTINGS) -> HealthBand:
    """≥75 GREEN, 50–74 YELLOW, <50 RED (defaults from the user story)."""
    green_threshold = settings.health_band_thresholds.get("GREEN", 75)
    yellow_threshold = settings.health_band_thresholds.get("YELLOW", 50)
    if score >= green_threshold:
        return "GREEN"
    if score >= yellow_threshold:
        return "YELLOW"
    return "RED"


__all__ = [
    "ComponentKey",
    "DealHealthBreakdown",
    "DealHealthInputs",
    "DealHealthSettings",
    "DEFAULT_SETTINGS",
    "HealthBand",
    "activity_freshness_score",
    "close_date_confidence_score",
    "compose_deal_health",
    "expected_days_in_stage",
    "health_band",
    "risk_adjustment_score",
    "stage_position_score",
    "stage_progress_score",
    "stage_velocity_score",
    "stakeholder_score",
]

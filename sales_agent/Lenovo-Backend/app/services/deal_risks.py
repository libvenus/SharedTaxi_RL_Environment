"""Deal Risk derivation — pure rule evaluators.

Implements the 13 risk rules from the "Deal Detailed View" user story.
Each rule is a small predicate function that takes a ``DealRiskInputs``
struct and returns either ``None`` (no risk) or a ``Risk`` instance.

The recalc orchestrator runs risks **before** health so the resulting
``len(risks)`` can feed the Risk Adjustment component (10% weight).

Adding a new rule
-----------------
1. Add a function ``risk_<short_name>(inputs) -> Risk | None``.
2. Append it to ``ALL_RULES`` in the order you want it evaluated.
3. Add the corresponding (category, name) pair to the SQL CHECK constraint
   in ``sql/2026_06_create_dealrisk.sql`` if needed.
4. Add a unit test in ``tests/test_deal_risks.py`` covering both the
   trigger case and the non-trigger case.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Literal

from app.services.deal_health import (
    DEFAULT_SETTINGS,
    DealHealthSettings,
    _normalise_tempo_class,
    activity_freshness_score,
    close_date_confidence_score,
    expected_days_in_stage,
    stage_velocity_score,
    stakeholder_score,
)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

RiskCategory = Literal[
    "Activity & Engagement",
    "Stakeholder",
    "Deal Execution",
    "Timeline & Forecast",
]


@dataclass(frozen=True)
class Risk:
    """One detected risk on a deal.

    Mirrors the columns on `lvo_dealrisk` and the public RiskInfo schema —
    keeping all three in sync is enforced by tests.
    """

    category: RiskCategory
    name: str
    message: str


@dataclass(frozen=True)
class DealRiskInputs:
    """Everything needed to evaluate every risk rule.

    Built by `app.services.deal_recalc` from the live DB state and passed
    in here. Tests build it directly so they don't need a DB.
    """

    # Opportunity row
    stage: str | None
    statecode: str | None
    estimated_value: float | None
    close_date: date | None
    name: str | None
    tempo_class: str | None

    # Computed time inputs
    today: date | None = None
    last_activity_date: datetime | None = None
    stage_entry_date: datetime | None = None
    created_at: datetime | None = None

    # Stakeholders
    active_stakeholders: int = 0
    decision_maker_present: bool = False
    decision_maker_engaged: bool = False  # any activity tied to the DM contact

    # Next actions
    open_next_actions: int = 0
    next_actions_missing_date: int = 0  # actions with no due_date

    # Misc deal hygiene
    has_notes: bool = True  # placeholder; the API doesn't read notes today

    # ---- helpers ----------------------------------------------------------
    @property
    def resolved_today(self) -> date:
        return self.today or datetime.now(timezone.utc).date()

    @property
    def is_closed(self) -> bool:
        if (self.statecode or "").lower() in {"won", "closed won", "lost", "closed lost"}:
            return True
        if (self.stage or "") in {"Closed Won", "Closed Lost"}:
            return True
        return False


# ---------------------------------------------------------------------------
# Activity & Engagement risks
# ---------------------------------------------------------------------------


def risk_low_activity(
    inputs: DealRiskInputs, settings: DealHealthSettings
) -> Risk | None:
    """Last activity older than ``low_activity_days_threshold`` (14 by default).

    Closed deals are exempt — they're not supposed to have ongoing activity.
    """
    if inputs.is_closed or inputs.last_activity_date is None:
        # No activity at all is handled by `risk_no_recent_engagement` (and is
        # covered by Activity Freshness component getting a low score).
        return None
    age_days = (inputs.resolved_today - inputs.last_activity_date.date()).days
    if age_days > int(settings.low_activity_days_threshold):
        return Risk(
            category="Activity & Engagement",
            name="Low Activity",
            message=f"Low activity on deal ({age_days} days since last touch)",
        )
    return None


def risk_no_recent_engagement(
    inputs: DealRiskInputs, settings: DealHealthSettings
) -> Risk | None:
    """Activity-freshness ratio > 1 (i.e. older than the tempo-class cadence).

    This deliberately uses the same calculation as the health calculator so
    the two stay in lock-step.
    """
    if inputs.is_closed:
        return None
    if inputs.last_activity_date is None:
        return Risk(
            category="Activity & Engagement",
            name="No Recent Engagement",
            message="No recent customer engagement",
        )
    age_days = max((inputs.resolved_today - inputs.last_activity_date.date()).days, 0)
    score, _ = activity_freshness_score(age_days, inputs.tempo_class, settings)
    if score < 100:
        return Risk(
            category="Activity & Engagement",
            name="No Recent Engagement",
            message="No recent customer engagement",
        )
    return None


# ---------------------------------------------------------------------------
# Stakeholder risks
# ---------------------------------------------------------------------------


def risk_decision_maker_not_engaged(
    inputs: DealRiskInputs, settings: DealHealthSettings
) -> Risk | None:
    """No DM mapped, OR DM mapped but no activity tied to them."""
    if inputs.is_closed:
        return None
    if not inputs.decision_maker_present or not inputs.decision_maker_engaged:
        return Risk(
            category="Stakeholder",
            name="Decision Maker Not Engaged",
            message="Decision maker not engaged",
        )
    return None


def risk_single_threaded(
    inputs: DealRiskInputs, settings: DealHealthSettings
) -> Risk | None:
    """Only one active stakeholder."""
    if inputs.is_closed:
        return None
    if inputs.active_stakeholders == 1:
        return Risk(
            category="Stakeholder",
            name="Single-Threaded Engagement",
            message="Single-threaded engagement",
        )
    return None


def risk_low_stakeholder_score(
    inputs: DealRiskInputs, settings: DealHealthSettings
) -> Risk | None:
    """Stakeholder score < configured threshold (40 by default)."""
    if inputs.is_closed:
        return None
    score, _ = stakeholder_score(inputs.active_stakeholders, settings)
    if score < int(settings.low_stakeholder_threshold):
        return Risk(
            category="Stakeholder",
            name="Low Stakeholder Score",
            message="Low stakeholder engagement detected.",
        )
    return None


# ---------------------------------------------------------------------------
# Deal Execution risks
# ---------------------------------------------------------------------------


def risk_no_next_steps(
    inputs: DealRiskInputs, settings: DealHealthSettings
) -> Risk | None:
    if inputs.is_closed:
        return None
    if inputs.open_next_actions == 0:
        return Risk(
            category="Deal Execution",
            name="No Next Steps Defined",
            message="No next steps defined",
        )
    return None


def risk_missing_action_date(
    inputs: DealRiskInputs, settings: DealHealthSettings
) -> Risk | None:
    if inputs.is_closed:
        return None
    if inputs.next_actions_missing_date > 0:
        return Risk(
            category="Deal Execution",
            name="Missing Action Date",
            message="Next action date not defined",
        )
    return None


def risk_stale_stage(
    inputs: DealRiskInputs, settings: DealHealthSettings
) -> Risk | None:
    """Today > stage_entry_date + (target_total_days * stage_fraction).

    Triggered when a deal has overstayed its allotted time in the current
    stage based on its tempo class.
    """
    if inputs.is_closed:
        return None
    if inputs.stage is None or inputs.stage_entry_date is None:
        return None

    canonical_tempo = _normalise_tempo_class(inputs.tempo_class)
    target_total = int(
        settings.tempo_class_target_days.get(canonical_tempo, 90)
    )
    stage_duration = expected_days_in_stage(inputs.stage, target_total, settings)
    if stage_duration <= 0:
        return None

    expected_exit = inputs.stage_entry_date.date() + timedelta(days=int(stage_duration))
    if inputs.resolved_today > expected_exit:
        return Risk(
            category="Deal Execution",
            name="Stale Deal Stage",
            message=f"Deal stage not progressing as expected (stuck in {inputs.stage})",
        )
    return None


def risk_incomplete_deal_information(
    inputs: DealRiskInputs, settings: DealHealthSettings
) -> Risk | None:
    """Any of value / closeDate / stage / notes is missing.

    `notes` is currently always True (placeholder until we add a notes
    column to the API). Kept in the rule so swapping it in later is one
    line change.
    """
    if inputs.is_closed:
        return None
    missing: list[str] = []
    if inputs.estimated_value is None or float(inputs.estimated_value) <= 0:
        missing.append("estimated value")
    if inputs.close_date is None:
        missing.append("close date")
    if inputs.stage is None:
        missing.append("stage")
    if not inputs.has_notes:
        missing.append("notes")
    if not missing:
        return None
    return Risk(
        category="Deal Execution",
        name="Incomplete Deal Information",
        message=f"Incomplete deal information ({', '.join(missing)})",
    )


# ---------------------------------------------------------------------------
# Timeline & Forecast risks
# ---------------------------------------------------------------------------


def risk_no_close_date(
    inputs: DealRiskInputs, settings: DealHealthSettings
) -> Risk | None:
    if inputs.is_closed:
        return None
    if inputs.close_date is None:
        return Risk(
            category="Timeline & Forecast",
            name="No Close Date",
            message="Close date not defined",
        )
    return None


def risk_close_date_overdue(
    inputs: DealRiskInputs, settings: DealHealthSettings
) -> Risk | None:
    if inputs.is_closed or inputs.close_date is None:
        return None
    if inputs.close_date < inputs.resolved_today:
        return Risk(
            category="Timeline & Forecast",
            name="Close Date Overdue",
            message="Close date overdue",
        )
    return None


def risk_unrealistic_close_timeline(
    inputs: DealRiskInputs, settings: DealHealthSettings
) -> Risk | None:
    """Triggered when Close Date Confidence resolves to 0 (gap > 40%)."""
    if inputs.is_closed or inputs.close_date is None:
        return None
    score, info = close_date_confidence_score(
        inputs.resolved_today,
        inputs.created_at,
        inputs.close_date,
        inputs.stage,
        settings,
        is_closed=inputs.is_closed,
    )
    # Score==0 due to overdue close-date is already covered by
    # risk_close_date_overdue. Skip when the reason is overdue to avoid
    # double-counting.
    if score == 0 and info.get("reason") not in {"close_date_overdue", "no_close_date"}:
        return Risk(
            category="Timeline & Forecast",
            name="Unrealistic Close Timeline",
            message="Unrealistic close timeline",
        )
    return None


def risk_deal_stuck_in_stage(
    inputs: DealRiskInputs, settings: DealHealthSettings
) -> Risk | None:
    """Triggered when stage velocity > 2.0 (i.e. velocity score == 20).

    Different from `risk_stale_stage` which compares against the absolute
    expected exit date. This one is the velocity-band trigger called out
    in the user story's Stage Velocity table.
    """
    if inputs.is_closed:
        return None
    if inputs.stage_entry_date is None or inputs.stage is None:
        return None
    canonical_tempo = _normalise_tempo_class(inputs.tempo_class)
    target_total = int(settings.tempo_class_target_days.get(canonical_tempo, 90))
    expected_days = expected_days_in_stage(inputs.stage, target_total, settings)
    actual_days = (inputs.resolved_today - inputs.stage_entry_date.date()).days
    if expected_days <= 0:
        return None
    if stage_velocity_score(actual_days, expected_days) <= 20:
        return Risk(
            category="Timeline & Forecast",
            name="Deal Stuck in Stage",
            message=f"Deal stuck in {inputs.stage} stage",
        )
    return None


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------

# Order matters only in two places:
#   * `evaluate_risks` returns the rules in this order, so consumers using
#     "first-risk" semantics (e.g. populating opportunity.lvo_riskreason
#     with the headline risk) get a predictable answer.
#   * `risk_unrealistic_close_timeline` defers to `risk_close_date_overdue`
#     to avoid double-counting overdue deals — keep `overdue` before
#     `unrealistic` in this list.
ALL_RULES: list[Callable[[DealRiskInputs, DealHealthSettings], Risk | None]] = [
    risk_close_date_overdue,
    risk_no_close_date,
    risk_unrealistic_close_timeline,
    risk_stale_stage,
    risk_deal_stuck_in_stage,
    risk_decision_maker_not_engaged,
    risk_single_threaded,
    risk_low_stakeholder_score,
    risk_low_activity,
    risk_no_recent_engagement,
    risk_no_next_steps,
    risk_missing_action_date,
    risk_incomplete_deal_information,
]


def evaluate_risks(
    inputs: DealRiskInputs,
    settings: DealHealthSettings = DEFAULT_SETTINGS,
) -> list[Risk]:
    """Run every rule, return the list of risks that fired."""
    out: list[Risk] = []
    for rule in ALL_RULES:
        risk = rule(inputs, settings)
        if risk is not None:
            out.append(risk)
    return out


__all__ = [
    "ALL_RULES",
    "DealRiskInputs",
    "Risk",
    "RiskCategory",
    "evaluate_risks",
]

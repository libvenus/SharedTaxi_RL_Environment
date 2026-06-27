"""Unit tests for app/services/deal_risks.py.

One trigger / non-trigger pair per rule, plus a couple of cross-cutting
sanity checks (closed deals never flag, the rules don't double-count when
a deal is both overdue and unrealistic).
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.services.deal_risks import (
    ALL_RULES,
    DealRiskInputs,
    Risk,
    evaluate_risks,
    risk_close_date_overdue,
    risk_deal_stuck_in_stage,
    risk_decision_maker_not_engaged,
    risk_incomplete_deal_information,
    risk_low_activity,
    risk_low_stakeholder_score,
    risk_missing_action_date,
    risk_no_close_date,
    risk_no_next_steps,
    risk_no_recent_engagement,
    risk_single_threaded,
    risk_stale_stage,
    risk_unrealistic_close_timeline,
)
from app.services.deal_health import DEFAULT_SETTINGS


SETTINGS = DEFAULT_SETTINGS


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _baseline(**overrides) -> DealRiskInputs:
    """Healthy default deal — no risks should fire on it.

    Tests override individual fields to flip a single rule.
    """
    base = dict(
        stage="Develop",
        statecode="Open",
        estimated_value=2_000_000.0,
        close_date=date(2026, 12, 31),
        name="Acme — Workstation Refresh",
        tempo_class="Quarterly",
        today=date(2026, 6, 9),
        last_activity_date=_ts("2026-06-05"),
        stage_entry_date=_ts("2026-04-01"),  # ~70 days ago, expected ~27 → velocity > 2 risk!
        created_at=_ts("2026-01-01"),
        active_stakeholders=4,
        decision_maker_present=True,
        decision_maker_engaged=True,
        open_next_actions=2,
        next_actions_missing_date=0,
    )
    # Adjust stage_entry_date so the baseline is healthy (no stale stage).
    base["stage_entry_date"] = _ts("2026-05-15")  # ~25 days ago, expected ~27 → velocity ~0.93 → score 100
    base.update(overrides)
    return DealRiskInputs(**base)


# ---------------------------------------------------------------------------
# Healthy baseline triggers no risks
# ---------------------------------------------------------------------------


def test_baseline_deal_has_no_risks() -> None:
    risks = evaluate_risks(_baseline(), SETTINGS)
    assert risks == []


# ---------------------------------------------------------------------------
# Activity & Engagement
# ---------------------------------------------------------------------------


def test_low_activity_triggers_after_threshold() -> None:
    inputs = _baseline(last_activity_date=_ts("2026-05-01"))  # 39 days ago > 14
    risk = risk_low_activity(inputs, SETTINGS)
    assert risk is not None
    assert risk.category == "Activity & Engagement"
    assert risk.name == "Low Activity"


def test_low_activity_does_not_trigger_within_threshold() -> None:
    inputs = _baseline(last_activity_date=_ts("2026-06-08"))
    assert risk_low_activity(inputs, SETTINGS) is None


def test_no_recent_engagement_triggers_when_no_activity_ever() -> None:
    inputs = _baseline(last_activity_date=None)
    risk = risk_no_recent_engagement(inputs, SETTINGS)
    assert risk is not None
    assert risk.name == "No Recent Engagement"


def test_no_recent_engagement_does_not_trigger_when_fresh() -> None:
    inputs = _baseline(last_activity_date=_ts("2026-06-08"))
    assert risk_no_recent_engagement(inputs, SETTINGS) is None


# ---------------------------------------------------------------------------
# Stakeholder
# ---------------------------------------------------------------------------


def test_decision_maker_not_engaged_triggers_when_no_dm() -> None:
    inputs = _baseline(decision_maker_present=False, decision_maker_engaged=False)
    risk = risk_decision_maker_not_engaged(inputs, SETTINGS)
    assert risk is not None


def test_decision_maker_not_engaged_triggers_when_dm_silent() -> None:
    inputs = _baseline(decision_maker_present=True, decision_maker_engaged=False)
    assert risk_decision_maker_not_engaged(inputs, SETTINGS) is not None


def test_decision_maker_not_engaged_no_trigger_when_engaged() -> None:
    inputs = _baseline(decision_maker_present=True, decision_maker_engaged=True)
    assert risk_decision_maker_not_engaged(inputs, SETTINGS) is None


def test_single_threaded_triggers_with_one_stakeholder() -> None:
    inputs = _baseline(active_stakeholders=1)
    risk = risk_single_threaded(inputs, SETTINGS)
    assert risk is not None


def test_single_threaded_does_not_trigger_with_two_or_more() -> None:
    inputs = _baseline(active_stakeholders=2)
    assert risk_single_threaded(inputs, SETTINGS) is None


def test_low_stakeholder_score_triggers_below_threshold() -> None:
    # 1 stakeholder → score 12 < threshold 40
    inputs = _baseline(active_stakeholders=1)
    assert risk_low_stakeholder_score(inputs, SETTINGS) is not None


def test_low_stakeholder_score_no_trigger_above_threshold() -> None:
    # 4 stakeholders → score 80 > 40
    inputs = _baseline(active_stakeholders=4)
    assert risk_low_stakeholder_score(inputs, SETTINGS) is None


# ---------------------------------------------------------------------------
# Deal Execution
# ---------------------------------------------------------------------------


def test_no_next_steps_triggers_when_zero_open() -> None:
    inputs = _baseline(open_next_actions=0)
    assert risk_no_next_steps(inputs, SETTINGS) is not None


def test_no_next_steps_no_trigger_when_open_actions_exist() -> None:
    inputs = _baseline(open_next_actions=1)
    assert risk_no_next_steps(inputs, SETTINGS) is None


def test_missing_action_date_triggers() -> None:
    inputs = _baseline(open_next_actions=1, next_actions_missing_date=1)
    assert risk_missing_action_date(inputs, SETTINGS) is not None


def test_missing_action_date_no_trigger_when_all_have_dates() -> None:
    inputs = _baseline(open_next_actions=2, next_actions_missing_date=0)
    assert risk_missing_action_date(inputs, SETTINGS) is None


def test_stale_stage_triggers_when_today_past_expected_exit() -> None:
    # Stage entered 200 days ago (very old) — Develop expected ~27 days.
    inputs = _baseline(
        stage_entry_date=_ts("2025-11-01"),
        today=date(2026, 6, 9),
    )
    assert risk_stale_stage(inputs, SETTINGS) is not None


def test_stale_stage_no_trigger_within_window() -> None:
    inputs = _baseline(
        stage_entry_date=_ts("2026-05-25"),  # ~15 days ago, expected ~27
        today=date(2026, 6, 9),
    )
    assert risk_stale_stage(inputs, SETTINGS) is None


def test_incomplete_deal_information_flags_missing_value() -> None:
    inputs = _baseline(estimated_value=None)
    risk = risk_incomplete_deal_information(inputs, SETTINGS)
    assert risk is not None
    assert "value" in risk.message.lower()


def test_incomplete_deal_information_no_trigger_when_complete() -> None:
    assert risk_incomplete_deal_information(_baseline(), SETTINGS) is None


# ---------------------------------------------------------------------------
# Timeline & Forecast
# ---------------------------------------------------------------------------


def test_no_close_date_triggers() -> None:
    assert risk_no_close_date(_baseline(close_date=None), SETTINGS) is not None


def test_no_close_date_no_trigger_when_set() -> None:
    assert risk_no_close_date(_baseline(), SETTINGS) is None


def test_close_date_overdue_triggers() -> None:
    inputs = _baseline(close_date=date(2026, 6, 1), today=date(2026, 6, 9))
    assert risk_close_date_overdue(inputs, SETTINGS) is not None


def test_close_date_overdue_no_trigger_in_future() -> None:
    inputs = _baseline(close_date=date(2026, 12, 31), today=date(2026, 6, 9))
    assert risk_close_date_overdue(inputs, SETTINGS) is None


def test_unrealistic_close_timeline_triggers_when_gap_huge() -> None:
    """Qualify-stage deal that's 90% through its time budget."""
    inputs = _baseline(
        stage="Qualify",
        created_at=_ts("2026-01-01"),
        close_date=date(2026, 6, 30),
        today=date(2026, 6, 1),
    )
    assert risk_unrealistic_close_timeline(inputs, SETTINGS) is not None


def test_unrealistic_close_timeline_does_not_double_count_overdue() -> None:
    """An overdue deal must trigger overdue, NOT unrealistic-timeline."""
    inputs = _baseline(close_date=date(2026, 5, 1), today=date(2026, 6, 9))
    # Overdue should fire …
    assert risk_close_date_overdue(inputs, SETTINGS) is not None
    # … and unrealistic should NOT (avoids double-counting in the same UI badge).
    assert risk_unrealistic_close_timeline(inputs, SETTINGS) is None


def test_deal_stuck_in_stage_triggers_when_velocity_high() -> None:
    inputs = _baseline(
        stage_entry_date=_ts("2025-12-01"),  # ~190 days ago in Develop, expected 27
        today=date(2026, 6, 9),
    )
    assert risk_deal_stuck_in_stage(inputs, SETTINGS) is not None


def test_deal_stuck_in_stage_no_trigger_when_velocity_normal() -> None:
    inputs = _baseline(
        stage_entry_date=_ts("2026-05-25"),
        today=date(2026, 6, 9),
    )
    assert risk_deal_stuck_in_stage(inputs, SETTINGS) is None


# ---------------------------------------------------------------------------
# Cross-cutting: closed deals never flag risks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("statecode", ["Won", "Closed Won", "Lost", "Closed Lost"])
def test_closed_deals_have_no_risks(statecode: str) -> None:
    inputs = _baseline(
        statecode=statecode,
        close_date=date(2026, 5, 1),  # would be overdue if still open
        last_activity_date=None,
        active_stakeholders=0,
        decision_maker_present=False,
        open_next_actions=0,
        next_actions_missing_date=0,
        estimated_value=None,
    )
    risks = evaluate_risks(inputs, SETTINGS)
    assert risks == [], f"Closed deal in state '{statecode}' should not flag risks: {risks}"


# ---------------------------------------------------------------------------
# evaluate_risks integration
# ---------------------------------------------------------------------------


def test_evaluate_risks_returns_in_rule_declaration_order() -> None:
    """Ordering matters because lvo_riskreason takes the *first* message."""
    inputs = _baseline(
        last_activity_date=None,
        active_stakeholders=1,
        decision_maker_present=False,
        open_next_actions=0,
    )
    risks = evaluate_risks(inputs, SETTINGS)
    names = [r.name for r in risks]

    # Rules are evaluated in ALL_RULES order; assert that the order in the
    # returned list matches the declaration so first-message semantics stay
    # deterministic.
    rule_names_in_order = [
        rule(inputs, SETTINGS).name
        for rule in ALL_RULES
        if rule(inputs, SETTINGS) is not None
    ]
    assert names == rule_names_in_order


def test_evaluate_risks_flags_multiple_for_unhealthy_deal() -> None:
    """Sanity: a clearly unhealthy deal should fire ≥ 3 risks at once."""
    inputs = _baseline(
        last_activity_date=None,
        active_stakeholders=1,
        decision_maker_present=False,
        open_next_actions=0,
    )
    risks = evaluate_risks(inputs, SETTINGS)
    assert len(risks) >= 3
    # All risks are well-formed Risk dataclasses.
    assert all(isinstance(r, Risk) for r in risks)

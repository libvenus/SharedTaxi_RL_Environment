"""Unit tests for app/services/deal_health.py.

The calculator has zero DB dependency, so each test builds the inputs in
memory and asserts the structured output. Coverage targets every band /
branch from the user story's scoring tables.

Run from the project root:

    pytest tests/test_deal_health.py -v
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.services.deal_health import (
    DEFAULT_SETTINGS,
    DealHealthInputs,
    activity_freshness_score,
    close_date_confidence_score,
    compose_deal_health,
    expected_days_in_stage,
    health_band,
    risk_adjustment_score,
    stage_position_score,
    stage_progress_score,
    stage_velocity_score,
    stakeholder_score,
)


# ---------------------------------------------------------------------------
# Stage position
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage, expected",
    [
        ("Qualify", 20),
        ("Develop", 40),
        ("Propose", 60),
        ("Execute", 80),
        ("Closed Won", 100),
        ("Closed Lost", 0),
        (None, 0),
        ("Unknown stage", 0),
    ],
)
def test_stage_position_score_matches_user_story(stage: str | None, expected: int) -> None:
    assert stage_position_score(stage, DEFAULT_SETTINGS) == expected


# ---------------------------------------------------------------------------
# Stage velocity (5 bands)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "actual, expected_days, expected_score",
    [
        # velocity ≤ 1.0  → 100
        (0, 10, 100.0),
        (5, 10, 100.0),
        (10, 10, 100.0),
        # 1.01 – 1.25 → 80
        (12, 10, 80.0),
        (12.5, 10, 80.0),
        # 1.26 – 1.50 → 60
        (13, 10, 60.0),
        (15, 10, 60.0),
        # 1.51 – 2.00 → 40
        (16, 10, 40.0),
        (20, 10, 40.0),
        # > 2.00 → 20
        (21, 10, 20.0),
        (100, 10, 20.0),
        # Defensive: zero/negative expected_days → 100 (no signal)
        (5, 0, 100.0),
        (5, -1, 100.0),
    ],
)
def test_stage_velocity_score_bands(
    actual: float, expected_days: float, expected_score: float
) -> None:
    assert stage_velocity_score(actual, expected_days) == expected_score


# ---------------------------------------------------------------------------
# Activity freshness (4 bands + no-activity case)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "age_days, tempo, expected_score",
    [
        # ratio ≤ 1.0 → 100
        (1, "Fast", 100.0),
        (4, "Fast", 100.0),
        (10, "Quarterly", 100.0),
        (20, "Programmatic", 100.0),
        (40, "Strategic", 100.0),
        # 1.01 – 2.0 → 70
        (5, "Fast", 70.0),
        (8, "Fast", 70.0),
        (15, "Quarterly", 70.0),
        # 2.01 – 3.0 → 40
        (9, "Fast", 40.0),
        (12, "Fast", 40.0),
        (25, "Quarterly", 40.0),
        # > 3.0 → 20
        (13, "Fast", 20.0),
        (60, "Quarterly", 20.0),
    ],
)
def test_activity_freshness_bands(
    age_days: int, tempo: str, expected_score: float
) -> None:
    score, _ = activity_freshness_score(age_days, tempo, DEFAULT_SETTINGS)
    assert score == expected_score


def test_activity_freshness_no_activity_is_critical() -> None:
    score, info = activity_freshness_score(None, "Quarterly", DEFAULT_SETTINGS)
    assert score == 20.0
    assert info["activityAgeDays"] is None


def test_activity_freshness_falls_back_to_quarterly_for_unknown_tempo() -> None:
    # Unknown tempo class → cadence falls back to Quarterly midpoint (10 days).
    score, info = activity_freshness_score(11, "????", DEFAULT_SETTINGS)
    assert info["tempoClass"] == "Quarterly"
    # 11 / 10 = 1.1 → second band → 70
    assert score == 70.0


# ---------------------------------------------------------------------------
# Stakeholder (coverage * threading factor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "active, expected_score",
    [
        # 1 active → coverage 20 * threading 0.60 = 12
        (1, 12.0),
        # 2 active → coverage 40 * threading 0.80 = 32
        (2, 32.0),
        # 3 active → coverage 60 * threading 1.00 = 60
        (3, 60.0),
        # 4 active → coverage 80 * threading 1.00 = 80
        (4, 80.0),
        # 5 active → coverage 100 * threading 1.00 = 100
        (5, 100.0),
        # 6 active → coverage capped at 100 * 1.00 = 100
        (6, 100.0),
    ],
)
def test_stakeholder_score(active: int, expected_score: float) -> None:
    score, _ = stakeholder_score(active, DEFAULT_SETTINGS)
    assert score == pytest.approx(expected_score)


def test_stakeholder_score_zero_active_is_zero() -> None:
    score, _ = stakeholder_score(0, DEFAULT_SETTINGS)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Close date confidence
# ---------------------------------------------------------------------------


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_close_confidence_overdue_is_zero() -> None:
    score, info = close_date_confidence_score(
        today=date(2026, 6, 9),
        created_at=_ts("2026-01-01"),
        close_date=date(2026, 5, 1),  # past
        stage="Develop",
        settings=DEFAULT_SETTINGS,
        is_closed=False,
    )
    assert score == 0.0
    assert info["reason"] == "close_date_overdue"


def test_close_confidence_no_close_date_is_zero() -> None:
    score, info = close_date_confidence_score(
        today=date(2026, 6, 9),
        created_at=_ts("2026-01-01"),
        close_date=None,
        stage="Qualify",
        settings=DEFAULT_SETTINGS,
        is_closed=False,
    )
    assert score == 0.0
    assert info["reason"] == "no_close_date"


def test_close_confidence_strong_alignment_is_100() -> None:
    """Develop stage at 50% time progress → expected 50%, gap 0 → 100."""
    score, info = close_date_confidence_score(
        today=date(2026, 4, 1),
        created_at=_ts("2026-01-01"),
        close_date=date(2026, 7, 1),
        stage="Develop",
        settings=DEFAULT_SETTINGS,
        is_closed=False,
    )
    # Expected progress for Develop = Qualify + Develop = 0.25 + 0.25 = 0.50
    # Time progress = 90/181 ≈ 0.497 → gap ≈ -0.003 → < 10% → 100
    assert score == 100.0
    assert info["expectedProgress"] == pytest.approx(0.50, abs=0.001)


def test_close_confidence_unrealistic_returns_zero() -> None:
    """Qualify-stage deal that's 90% through its time budget → gap > 40% → 0."""
    score, info = close_date_confidence_score(
        today=date(2026, 6, 1),
        created_at=_ts("2026-01-01"),
        close_date=date(2026, 6, 30),
        stage="Qualify",
        settings=DEFAULT_SETTINGS,
        is_closed=False,
    )
    # Time progress ≈ 0.84, expected = 0.25 (Qualify only) → gap ≈ 0.59 → 0
    assert score == 0.0
    assert info["gap"] > 0.40


def test_close_confidence_closed_deal_is_100() -> None:
    score, info = close_date_confidence_score(
        today=date(2026, 6, 1),
        created_at=_ts("2026-01-01"),
        close_date=date(2026, 6, 30),
        stage="Closed Won",
        settings=DEFAULT_SETTINGS,
        is_closed=True,
    )
    assert score == 100.0
    assert info["reason"] == "deal_closed"


# ---------------------------------------------------------------------------
# Risk adjustment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 100.0),
        (1, 80.0),
        (2, 60.0),
        (3, 40.0),
        (4, 20.0),
        (5, 0.0),
        (6, 0.0),  # floored
        (-1, 100.0),  # never goes above 100
    ],
)
def test_risk_adjustment_score(n: int, expected: float) -> None:
    score, _ = risk_adjustment_score(n, DEFAULT_SETTINGS)
    assert score == expected


# ---------------------------------------------------------------------------
# Health band thresholds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score, expected",
    [
        (100, "GREEN"),
        (75, "GREEN"),
        (74, "YELLOW"),
        (50, "YELLOW"),
        (49, "RED"),
        (0, "RED"),
    ],
)
def test_health_band(score: int, expected: str) -> None:
    assert health_band(score) == expected


# ---------------------------------------------------------------------------
# Stage progress (combined)
# ---------------------------------------------------------------------------


def test_stage_progress_is_average_of_position_and_velocity() -> None:
    expected_days = expected_days_in_stage("Develop", 90, DEFAULT_SETTINGS)
    assert expected_days == 90 * 0.30
    score, info = stage_progress_score("Develop", actual_days_in_stage=27, expected_days=expected_days, settings=DEFAULT_SETTINGS)
    # Position = 40, velocity (27/27=1.0) = 100 → average = 70
    assert score == 70.0
    assert info["stagePositionScore"] == 40
    assert info["stageVelocityScore"] == 100


# ---------------------------------------------------------------------------
# Composite — end-to-end
# ---------------------------------------------------------------------------


def test_compose_healthy_strong_deal() -> None:
    """Mid-stage Develop deal, on-cadence activity, 4 stakeholders, no risks."""
    inputs = DealHealthInputs(
        stage="Develop",
        tempo_class="Quarterly",
        actual_days_in_stage=20,        # < expected 27 → velocity 100
        target_close_total_days=90,
        last_activity_age_days=5,       # < cadence 10 → freshness 100
        active_stakeholders=4,           # coverage 80 * threading 1.00 = 80
        created_at=_ts("2026-01-01"),
        close_date=date(2026, 7, 1),
        is_closed=False,
        risk_count=0,                    # → adjustment 100
        today=date(2026, 4, 1),
    )
    breakdown = compose_deal_health(inputs)

    # Stage progress = 0.5*40 + 0.5*100 = 70
    # Activity 100, Stakeholder 80, Close 100, Risk 100
    # Weighted = (70*25 + 100*25 + 80*20 + 100*20 + 100*10) / 100
    #         = (1750 + 2500 + 1600 + 2000 + 1000) / 100 = 88.5
    # Python's round-half-to-even gives 88; we accept either to stay
    # robust against banker's-rounding edge cases.
    assert breakdown.score in {88, 89}
    assert breakdown.band == "GREEN"


def test_compose_unhealthy_deal_is_red_or_yellow() -> None:
    """Stalled qualify deal, no activity, single stakeholder, 3 risks."""
    inputs = DealHealthInputs(
        stage="Qualify",
        tempo_class="Quarterly",
        actual_days_in_stage=120,        # >> expected 18 → velocity 20
        target_close_total_days=90,
        last_activity_age_days=None,     # no activity → freshness 20
        active_stakeholders=1,            # → 12
        created_at=_ts("2026-01-01"),
        close_date=date(2026, 6, 30),    # near future
        is_closed=False,
        risk_count=3,                    # → 40
        today=date(2026, 6, 1),
    )
    breakdown = compose_deal_health(inputs)
    # The exact composite score is band-checked rather than equality-asserted —
    # the goal here is "this deal must NOT be GREEN".
    assert breakdown.band in {"YELLOW", "RED"}
    assert breakdown.score < 75


def test_compose_clamps_to_0_100() -> None:
    """Synthetic worst-case input — score must stay within [0, 100]."""
    inputs = DealHealthInputs(
        stage="Closed Lost",
        tempo_class="Quarterly",
        actual_days_in_stage=10000,
        target_close_total_days=90,
        last_activity_age_days=None,
        active_stakeholders=0,
        created_at=_ts("2020-01-01"),
        close_date=date(2020, 12, 31),
        is_closed=True,
        risk_count=10,
    )
    breakdown = compose_deal_health(inputs)
    assert 0 <= breakdown.score <= 100


def test_compose_returns_all_five_components() -> None:
    inputs = DealHealthInputs(
        stage="Propose",
        tempo_class="Strategic",
        actual_days_in_stage=50,
        target_close_total_days=730,
        last_activity_age_days=10,
        active_stakeholders=3,
        created_at=_ts("2025-01-01"),
        close_date=date(2026, 12, 31),
        is_closed=False,
        risk_count=1,
        today=date(2026, 6, 9),
    )
    breakdown = compose_deal_health(inputs)
    assert set(breakdown.components.keys()) == {
        "stage_progress",
        "activity_freshness",
        "stakeholder",
        "close_confidence",
        "risk_adjustment",
    }
    for c in breakdown.components.values():
        assert 0 <= c["score"] <= 100
        assert c["weight"] in {25, 20, 10}

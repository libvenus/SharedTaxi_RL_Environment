"""Pure-function tests for app/services/kpi_snapshots.py.

Tests focus on ``compute_trend_info`` because every other function in
that module is a thin wrapper around the database. Direction logic and
delta math are independently regression-checked here.
"""

from __future__ import annotations

import pytest

from app.services.kpi_snapshots import (
    BucketAggregate,
    compute_trend_info,
)


def _bucket(bucket: str = "open", value: float = 0.0, count: int = 0) -> BucketAggregate:
    return BucketAggregate(bucket=bucket, value=value, count=count)


# ---------------------------------------------------------------------------
# Null comparison
# ---------------------------------------------------------------------------


def test_returns_none_when_previous_missing():
    """No historical row → caller must surface trend=null."""
    assert compute_trend_info(_bucket(value=100.0, count=3), None) is None


# ---------------------------------------------------------------------------
# Up direction
# ---------------------------------------------------------------------------


def test_direction_up_when_value_grew():
    delta = compute_trend_info(
        _bucket(value=120_000.0, count=10),
        _bucket(value=100_000.0, count=8),
    )
    assert delta is not None
    assert delta.direction == "up"
    assert delta.delta_value == pytest.approx(20_000.0)
    assert delta.delta_count == 2


def test_direction_up_when_only_count_grew_and_value_flat():
    """Value tie → fall through to count for direction."""
    delta = compute_trend_info(
        _bucket(value=50_000.0, count=12),
        _bucket(value=50_000.0, count=10),
    )
    assert delta is not None
    assert delta.direction == "up"
    assert delta.delta_value == 0.0
    assert delta.delta_count == 2


# ---------------------------------------------------------------------------
# Down direction
# ---------------------------------------------------------------------------


def test_direction_down_when_value_shrank():
    delta = compute_trend_info(
        _bucket(value=80_000.0, count=6),
        _bucket(value=100_000.0, count=8),
    )
    assert delta is not None
    assert delta.direction == "down"
    assert delta.delta_value == pytest.approx(-20_000.0)
    assert delta.delta_count == -2


def test_direction_down_when_only_count_shrank():
    delta = compute_trend_info(
        _bucket(value=100_000.0, count=5),
        _bucket(value=100_000.0, count=8),
    )
    assert delta is not None
    assert delta.direction == "down"
    assert delta.delta_count == -3


# ---------------------------------------------------------------------------
# Flat direction
# ---------------------------------------------------------------------------


def test_direction_flat_when_no_change():
    delta = compute_trend_info(
        _bucket(value=100_000.0, count=8),
        _bucket(value=100_000.0, count=8),
    )
    assert delta is not None
    assert delta.direction == "flat"
    assert delta.delta_value == 0.0
    assert delta.delta_count == 0


def test_direction_flat_when_both_zero():
    """Edge case: empty bucket on both sides — should not raise."""
    delta = compute_trend_info(_bucket(), _bucket())
    assert delta is not None
    assert delta.direction == "flat"


# ---------------------------------------------------------------------------
# Mixed signals — value wins over count
# ---------------------------------------------------------------------------


def test_value_growth_outranks_count_drop():
    """One large deal closing replaces five small deals — direction is up."""
    delta = compute_trend_info(
        _bucket(value=500_000.0, count=3),
        _bucket(value=400_000.0, count=5),
    )
    assert delta is not None
    assert delta.direction == "up"
    assert delta.delta_value == pytest.approx(100_000.0)
    assert delta.delta_count == -2


def test_value_drop_outranks_count_growth():
    """Several small deals replacing one big one — direction is down."""
    delta = compute_trend_info(
        _bucket(value=120_000.0, count=10),
        _bucket(value=200_000.0, count=4),
    )
    assert delta is not None
    assert delta.direction == "down"
    assert delta.delta_value == pytest.approx(-80_000.0)
    assert delta.delta_count == 6

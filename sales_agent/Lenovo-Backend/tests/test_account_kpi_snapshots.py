"""Pure-function tests for app/services/account_kpi_snapshots.py.

The trend-math helper (`compute_trend_info`) is reused from
``app.services.kpi_snapshots`` and already has its own test suite in
``tests/test_kpi_snapshots.py``. Here we verify:

* The bucket whitelist (``BUCKETS``) covers exactly the four cards on
  the Accounts page.
* ``BucketAggregate`` is duck-compatible with the trend-math helper —
  i.e. it satisfies ``compute_trend_info`` without raising on attribute
  access.
* The lookback table (``PERIOD_LOOKBACK_DAYS``) maps the three accepted
  ``ComparePeriod`` literals to canonical day counts.

Database-touching paths (``compute_buckets`` / ``take_snapshot`` /
``lookup_snapshot``) are integration-level and exercised by Postman /
the dev server — keeping the unit tests pure preserves the "no DB" rule
the rest of ``tests/`` follows.
"""

from __future__ import annotations

import pytest

from app.services.account_kpi_snapshots import (
    BUCKETS,
    PERIOD_LOOKBACK_DAYS,
    BucketAggregate,
)
from app.services.kpi_snapshots import compute_trend_info


# ---------------------------------------------------------------------------
# Bucket whitelist
# ---------------------------------------------------------------------------


def test_buckets_cover_the_four_cards():
    """The Accounts page ships exactly four cards in v1; nothing more, nothing less."""
    assert BUCKETS == ("total", "acv", "active", "at_risk")


def test_period_lookback_days_matches_compare_period_literals():
    """Adding a new ComparePeriod must update both this map and the schemas Literal."""
    assert PERIOD_LOOKBACK_DAYS == {
        "last_week": 7,
        "past_month": 30,
        "last_quarter": 90,
    }


# ---------------------------------------------------------------------------
# Duck-compatibility with the shared trend-math helper
# ---------------------------------------------------------------------------


def _bucket(bucket: str = "total", value: float = 0.0, count: int = 0) -> BucketAggregate:
    return BucketAggregate(bucket=bucket, value=value, count=count)


def test_compute_trend_info_accepts_account_bucket_aggregate_up():
    """An account-side BucketAggregate must work with the shared trend helper."""
    delta = compute_trend_info(
        _bucket(bucket="active", count=24),
        _bucket(bucket="active", count=22),
    )
    assert delta is not None
    assert delta.direction == "up"
    assert delta.delta_count == 2
    # value is 0 on count-only buckets — direction comes from delta_count.
    assert delta.delta_value == pytest.approx(0.0)


def test_compute_trend_info_accepts_account_bucket_aggregate_down():
    delta = compute_trend_info(
        _bucket(bucket="at_risk", count=5),
        _bucket(bucket="at_risk", count=7),
    )
    assert delta is not None
    assert delta.direction == "down"
    assert delta.delta_count == -2


def test_compute_trend_info_handles_acv_value_swing():
    """The ACV bucket carries dollars — direction must follow delta_value."""
    delta = compute_trend_info(
        _bucket(bucket="acv", value=21_000_000.0, count=40),
        _bucket(bucket="acv", value=19_750_000.0, count=38),
    )
    assert delta is not None
    assert delta.direction == "up"
    assert delta.delta_value == pytest.approx(1_250_000.0)
    assert delta.delta_count == 2


def test_compute_trend_info_returns_none_when_previous_missing():
    assert compute_trend_info(_bucket(count=10), None) is None

"""KPI snapshot service — powers period-over-period trend math on the
``/api/opportunities/kpi-summary`` endpoint.

Architecture mirrors ``deal_recalc.py``:

* ``compute_buckets``      — pure-ish DB read; the same six SUM/COUNT
  aggregates the live endpoint runs, optionally restricted to deals that
  already existed by ``existing_by`` (used during backfill).
* ``take_snapshot``        — UPSERTs one ``lvo_opportunitysnapshot`` row
  per bucket for a given ``snapshot_date``.
* ``lookup_snapshot``      — fetches the most-recent snapshot row on or
  before a target date for each bucket. Robust to missed days.
* ``compute_trend_info``   — pure helper: turns (current, previous) into a
  ``TrendInfo`` payload (or ``None`` when no comparison row exists).

Trend semantics for v1
----------------------
* Direction is the **sign of deltaValue**: ``up`` if positive, ``down`` if
  negative, ``flat`` if exactly zero.
* The router emits ``trend: null`` when filters are applied (Strategy A).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

from sqlalchemy import Numeric, cast, desc, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models import Opportunity, OpportunitySnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — kept in sync with `_bucket_predicate` in opportunities.py
# ---------------------------------------------------------------------------

BUCKETS: tuple[str, ...] = (
    "open",
    "pipeline",
    "best_case",
    "commit",
    "most_likely",
    "won",
    "loss",
)

WON_STATECODES = ("Won", "Closed Won")
LOST_STATECODES = ("Lost", "Closed Lost")
WON_STAGES = ("Closed Won",)
LOST_STAGES = ("Closed Lost",)

# Map a ComparePeriod literal to a lookback in days. The router only
# accepts these three; anything new must be added on both sides.
PERIOD_LOOKBACK_DAYS: dict[str, int] = {
    "last_week": 7,
    "past_month": 30,
    "last_quarter": 90,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BucketAggregate:
    """One bucket's SUM(estimatedvalue) + COUNT(*)."""

    bucket: str
    value: float = 0.0
    count: int = 0


@dataclass(frozen=True)
class TrendDelta:
    """Pure data shape returned by ``compute_trend_info``.

    The router converts this into a ``TrendInfo`` Pydantic model — keeping
    the service module Pydantic-free preserves the "pure-ish" contract.
    """

    direction: str  # "up" | "down" | "flat"
    delta_value: float
    delta_count: int


# ---------------------------------------------------------------------------
# Bucket predicate (mirror of opportunities._bucket_predicate)
# ---------------------------------------------------------------------------


def _bucket_predicate(bucket: str):
    """The same six predicates the live endpoint uses.

    Duplicated rather than imported to keep services free of router-side
    deps. Keep this in sync with ``app/routers/opportunities.py`` —
    every bucket lives in both places.
    """
    if bucket == "open":
        return Opportunity.statecode == "Open"
    if bucket == "pipeline":
        # UI label is "Identified"; predicate and bucket id stay "pipeline".
        return Opportunity.lvo_forecastcategory == "Pipeline"
    if bucket == "best_case":
        return Opportunity.lvo_forecastcategory == "Best Case"
    if bucket == "commit":
        return Opportunity.lvo_forecastcategory == "Commit"
    if bucket == "most_likely":
        return Opportunity.lvo_forecastcategory == "Most Likely"
    if bucket == "won":
        return Opportunity.statecode.in_(WON_STATECODES) | Opportunity.stagename.in_(
            WON_STAGES
        )
    if bucket == "loss":
        return Opportunity.statecode.in_(LOST_STATECODES) | Opportunity.stagename.in_(
            LOST_STAGES
        )
    raise ValueError(f"Unknown KPI bucket: {bucket!r}")


# ---------------------------------------------------------------------------
# Bucket computation
# ---------------------------------------------------------------------------


def compute_buckets(
    db: Session,
    *,
    existing_by: date | None = None,
) -> dict[str, BucketAggregate]:
    """Compute (sum, count) for every bucket in one round-trip.

    Parameters
    ----------
    existing_by
        When set, restricts the aggregate to deals that existed by that
        calendar date — the backfill path uses this to reconstruct
        historical snapshots from the current ``opportunity`` table.
        Falls back to ``opportunity.lvo_createdat`` (which the
        Deal-Detailed-View migration backfilled) so the cutoff can be
        evaluated even when the base table does not expose ``createdon``.
    """
    value_col = cast(Opportunity.estimatedvalue, Numeric)

    def _bucket_select(label: str, *predicates):
        sum_expr = func.coalesce(
            func.sum(value_col).filter(*predicates), 0
        ).label(f"{label}_value")
        count_expr = func.count().filter(*predicates).label(f"{label}_count")
        return sum_expr, count_expr

    cols = []
    for bucket in BUCKETS:
        sum_expr, count_expr = _bucket_select(bucket, _bucket_predicate(bucket))
        cols.append(sum_expr)
        cols.append(count_expr)

    stmt = select(*cols).select_from(Opportunity)
    if existing_by is not None:
        # Use lvo_createdat (set by the Deal-Detailed-View migration) as the
        # "existed-by" filter. Treat NULL as "always existed" so older rows
        # without a created-at value still appear in historical snapshots.
        stmt = stmt.where(
            (Opportunity.lvo_createdat.is_(None))
            | (Opportunity.lvo_createdat <= datetime.combine(
                existing_by, datetime.max.time(), tzinfo=timezone.utc
            ))
        )

    row = db.execute(stmt).one()

    out: dict[str, BucketAggregate] = {}
    for bucket in BUCKETS:
        out[bucket] = BucketAggregate(
            bucket=bucket,
            value=float(getattr(row, f"{bucket}_value") or 0),
            count=int(getattr(row, f"{bucket}_count") or 0),
        )
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def take_snapshot(
    db: Session,
    *,
    snapshot_date: date | None = None,
    existing_by: date | None = None,
) -> dict[str, BucketAggregate]:
    """Compute every bucket and UPSERT the rows into the snapshot table.

    Idempotent — re-running on the same ``snapshot_date`` overwrites the
    existing values rather than duplicating rows (uniqueness is enforced
    by ``uq_lvo_opportunitysnapshot_date_bucket``).

    Returns the freshly-computed buckets so the caller can log them.
    """
    snapshot_date = snapshot_date or datetime.now(timezone.utc).date()
    buckets = compute_buckets(db, existing_by=existing_by or snapshot_date)

    now = datetime.now(timezone.utc)
    rows = [
        {
            "lvo_opportunitysnapshotid": str(uuid.uuid4()),
            "lvo_snapshotdate": snapshot_date,
            "lvo_bucket": bucket,
            "lvo_value": agg.value,
            "lvo_count": agg.count,
            "lvo_createdat": now,
        }
        for bucket, agg in buckets.items()
    ]

    insert_stmt = pg_insert(OpportunitySnapshot.__table__).values(rows)
    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=("lvo_snapshotdate", "lvo_bucket"),
        set_={
            "lvo_value": insert_stmt.excluded.lvo_value,
            "lvo_count": insert_stmt.excluded.lvo_count,
            "lvo_createdat": insert_stmt.excluded.lvo_createdat,
        },
    )
    db.execute(upsert_stmt)
    db.commit()
    logger.debug("Snapshot %s recorded for %s buckets", snapshot_date, len(rows))
    return buckets


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def lookup_snapshot(
    db: Session, *, on_or_before: date
) -> dict[str, BucketAggregate]:
    """Return the most-recent snapshot row ≤ ``on_or_before`` for each bucket.

    Buckets without any qualifying snapshot are absent from the dict — the
    caller can treat that as "no comparison row available".
    """
    # DISTINCT ON (bucket) ORDER BY bucket, snapshot_date DESC is the
    # canonical "latest per group" idiom in Postgres. We fall back to a
    # generic GROUP BY for portability with other backends.
    rows = (
        db.execute(
            select(
                OpportunitySnapshot.lvo_bucket,
                OpportunitySnapshot.lvo_value,
                OpportunitySnapshot.lvo_count,
                OpportunitySnapshot.lvo_snapshotdate,
            )
            .where(OpportunitySnapshot.lvo_snapshotdate <= on_or_before)
            .order_by(
                OpportunitySnapshot.lvo_bucket.asc(),
                desc(OpportunitySnapshot.lvo_snapshotdate),
            )
        )
        .all()
    )

    seen: set[str] = set()
    out: dict[str, BucketAggregate] = {}
    for r in rows:
        if r.lvo_bucket in seen:
            continue
        seen.add(r.lvo_bucket)
        out[r.lvo_bucket] = BucketAggregate(
            bucket=r.lvo_bucket,
            value=float(r.lvo_value or 0),
            count=int(r.lvo_count or 0),
        )
    return out


def lookup_previous_for_period(
    db: Session, *, period: str, today: date | None = None
) -> dict[str, BucketAggregate]:
    """Find the comparison snapshot for a ComparePeriod literal.

    Returns ``{}`` when the lookback maps to an unsupported period or no
    snapshot exists ≤ the target date.
    """
    if period not in PERIOD_LOOKBACK_DAYS:
        return {}
    today = today or datetime.now(timezone.utc).date()
    target = today - timedelta(days=PERIOD_LOOKBACK_DAYS[period])
    return lookup_snapshot(db, on_or_before=target)


# ---------------------------------------------------------------------------
# Trend math (pure)
# ---------------------------------------------------------------------------


def compute_trend_info(
    current: BucketAggregate,
    previous: BucketAggregate | None,
) -> TrendDelta | None:
    """Pure helper: turn (current, previous) into a TrendDelta.

    Returns ``None`` when there's no historical row to compare against.
    Direction is determined by ``deltaValue`` first; ties (deltaValue == 0)
    fall back to ``deltaCount``; if both are zero the trend is ``flat``.
    """
    if previous is None:
        return None
    delta_value = float(current.value - previous.value)
    delta_count = int(current.count - previous.count)

    if delta_value > 0:
        direction = "up"
    elif delta_value < 0:
        direction = "down"
    elif delta_count > 0:
        direction = "up"
    elif delta_count < 0:
        direction = "down"
    else:
        direction = "flat"

    return TrendDelta(
        direction=direction,
        delta_value=delta_value,
        delta_count=delta_count,
    )


# ---------------------------------------------------------------------------
# Backfill helper
# ---------------------------------------------------------------------------


def backfill(
    db: Session,
    *,
    dates: Iterable[date] | None = None,
    today: date | None = None,
) -> dict[date, dict[str, BucketAggregate]]:
    """Generate snapshots for a sequence of historical dates.

    Defaults to ``today``, ``today-7``, ``today-30``, ``today-90`` so the
    KPI strip can show non-null trends right after a fresh deploy. Uses
    ``opportunity.lvo_createdat`` as the existed-by-date filter — won/loss
    backfills are approximate because the snapshot service has no record
    of historical state changes.
    """
    today = today or datetime.now(timezone.utc).date()
    if dates is None:
        dates = [
            today,
            today - timedelta(days=PERIOD_LOOKBACK_DAYS["last_week"]),
            today - timedelta(days=PERIOD_LOOKBACK_DAYS["past_month"]),
            today - timedelta(days=PERIOD_LOOKBACK_DAYS["last_quarter"]),
        ]

    out: dict[date, dict[str, BucketAggregate]] = {}
    for d in dates:
        buckets = take_snapshot(db, snapshot_date=d, existing_by=d)
        out[d] = buckets
    return out


__all__ = [
    "BUCKETS",
    "BucketAggregate",
    "PERIOD_LOOKBACK_DAYS",
    "TrendDelta",
    "backfill",
    "compute_buckets",
    "compute_trend_info",
    "lookup_previous_for_period",
    "lookup_snapshot",
    "take_snapshot",
]

"""Account-KPI snapshot service — powers period-over-period trend math on
``GET /api/accounts/kpi-summary``.

Architecture mirrors ``app/services/kpi_snapshots.py``:

* ``compute_buckets``      — pure-ish DB read; computes the four account-side
  aggregates (total / acv / active / at_risk) in two round-trips.
  Optionally restricted to rows that already existed by ``existing_by``
  (used during backfill).
* ``take_snapshot``        — UPSERTs one ``lvo_accountsnapshot`` row per
  bucket for a given ``snapshot_date``.
* ``lookup_snapshot``      — fetches the most-recent snapshot row on or
  before a target date for each bucket. Robust to missed days.
* ``lookup_previous_for_period`` — wraps ``lookup_snapshot`` with the
  ``ComparePeriod`` literal the router accepts.
* ``backfill``             — generates snapshots for today / today-7 /
  today-30 / today-90 so the strip shows non-null trends right after a
  fresh deploy.

We intentionally **reuse** ``compute_trend_info`` from
``app/services/kpi_snapshots.py`` — the function is bucket-agnostic, so
duplicating it across two services would only invite drift.

Trend semantics (Strategy A)
----------------------------
The router emits ``trend: null`` whenever any filter parameter is supplied.
Snapshots are global / unfiltered in v1; per-dimension snapshots are a
planned follow-up.

Backfill caveats
----------------
* ``total`` — exact when ``account.createdon`` exists on the live schema.
  Partial dumps that lack the column silently skip the existed-by filter
  and fall back to "today" values for every backfill date.
* ``acv`` — exact whenever ``opportunity.lvo_createdat`` is populated
  (set by ``sql/2026_06_deal_detail_schema.sql``).
* ``active`` / ``at_risk`` — **approximate**. ``lvo_accountstatus`` flips
  over time aren't journaled, so backfill uses the *current* status for
  every historical date. Trend deltas across a period covered only by
  backfill will read 0; once nightly snapshots accumulate, trends become
  meaningful.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

from sqlalchemy import Numeric, String, cast, desc, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.filters_query import CANCELED_STATECODE
from app.models import Account, AccountSnapshot, Opportunity
from app.services.account_columns import get_account_columns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — kept in sync with ``_account_bucket_predicate`` in the router
# ---------------------------------------------------------------------------

BUCKETS: tuple[str, ...] = (
    "total",
    "acv",
    "active",
    "at_risk",
)

ACTIVE_STATUS = "Active"
AT_RISK_STATUS = "At-Risk"

# Map a ComparePeriod literal to a lookback in days. The router only accepts
# these three; anything new must be added on both sides.
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
    """One bucket's value + count.

    Intentionally NOT shared with ``app.services.kpi_snapshots.BucketAggregate``
    so the two snapshot services stay independently importable. The trend-math
    helper consumes anything with ``.value`` and ``.count`` attributes.
    """

    bucket: str
    value: float = 0.0
    count: int = 0


# ---------------------------------------------------------------------------
# Bucket computation
# ---------------------------------------------------------------------------


def compute_buckets(
    db: Session,
    *,
    existing_by: date | None = None,
) -> dict[str, BucketAggregate]:
    """Compute (value, count) for every account-side bucket.

    Uses two round-trips:
      1. Account-side counts (``total`` / ``active`` / ``at_risk``).
      2. Opportunity-side ACV roll-up (``acv``).

    Parameters
    ----------
    existing_by
        When set, restricts each aggregate to rows that existed by that
        calendar date. The accuracy of the restriction depends on which
        column is available — see the module docstring for caveats.
    """
    cutoff: datetime | None = None
    if existing_by is not None:
        cutoff = datetime.combine(existing_by, datetime.max.time(), tzinfo=timezone.utc)

    # ---- 1. Account-side aggregates -----------------------------------
    acct_total = func.count().label("total_count")
    acct_active = (
        func.count()
        .filter(Account.lvo_accountstatus == ACTIVE_STATUS)
        .label("active_count")
    )
    acct_at_risk = (
        func.count()
        .filter(Account.lvo_accountstatus == AT_RISK_STATUS)
        .label("at_risk_count")
    )

    acct_stmt = select(acct_total, acct_active, acct_at_risk).select_from(Account)
    if cutoff is not None and "createdon" in get_account_columns(db):
        # createdon is NULL for the few legacy rows in the dump — treat NULL
        # as "always existed" so the historical count never under-reports.
        acct_stmt = acct_stmt.where(
            (Account.createdon.is_(None)) | (Account.createdon <= cutoff)
        )

    acct_row = db.execute(acct_stmt).one()

    # ---- 2. Opportunity-side ACV --------------------------------------
    value_col = cast(Opportunity.estimatedvalue, Numeric)
    acv_value = func.coalesce(func.sum(value_col), 0).label("acv_value")
    acv_count = func.count(func.distinct(func.upper(cast(Opportunity.accountid, String)))).label(
        "acv_count"
    )

    acv_stmt = (
        select(acv_value, acv_count)
        .select_from(Opportunity)
        .where(
            func.coalesce(Opportunity.statecode, "") != CANCELED_STATECODE,
            Opportunity.accountid.is_not(None),
        )
    )
    if cutoff is not None:
        # lvo_createdat is set by sql/2026_06_deal_detail_schema.sql; treat
        # NULL as "always existed" same as above.
        acv_stmt = acv_stmt.where(
            (Opportunity.lvo_createdat.is_(None))
            | (Opportunity.lvo_createdat <= cutoff)
        )

    acv_row = db.execute(acv_stmt).one()

    return {
        "total": BucketAggregate(
            bucket="total", value=0.0, count=int(acct_row.total_count or 0)
        ),
        "acv": BucketAggregate(
            bucket="acv",
            value=float(acv_row.acv_value or 0),
            count=int(acv_row.acv_count or 0),
        ),
        "active": BucketAggregate(
            bucket="active", value=0.0, count=int(acct_row.active_count or 0)
        ),
        "at_risk": BucketAggregate(
            bucket="at_risk", value=0.0, count=int(acct_row.at_risk_count or 0)
        ),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def take_snapshot(
    db: Session,
    *,
    snapshot_date: date | None = None,
    existing_by: date | None = None,
) -> dict[str, BucketAggregate]:
    """Compute every bucket and UPSERT one row per bucket into
    ``lvo_accountsnapshot``.

    Idempotent — re-running on the same ``snapshot_date`` overwrites the
    existing values rather than duplicating rows (uniqueness is enforced
    by ``uq_lvo_accountsnapshot_date_bucket``).

    Returns the freshly-computed buckets so the caller can log them.
    """
    snapshot_date = snapshot_date or datetime.now(timezone.utc).date()
    buckets = compute_buckets(db, existing_by=existing_by or snapshot_date)

    now = datetime.now(timezone.utc)
    rows = [
        {
            "lvo_accountsnapshotid": str(uuid.uuid4()),
            "lvo_snapshotdate": snapshot_date,
            "lvo_bucket": bucket,
            "lvo_value": agg.value,
            "lvo_count": agg.count,
            "lvo_createdat": now,
        }
        for bucket, agg in buckets.items()
    ]

    insert_stmt = pg_insert(AccountSnapshot.__table__).values(rows)
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
    logger.debug(
        "Account snapshot %s recorded for %s buckets", snapshot_date, len(rows)
    )
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
    rows = (
        db.execute(
            select(
                AccountSnapshot.lvo_bucket,
                AccountSnapshot.lvo_value,
                AccountSnapshot.lvo_count,
                AccountSnapshot.lvo_snapshotdate,
            )
            .where(AccountSnapshot.lvo_snapshotdate <= on_or_before)
            .order_by(
                AccountSnapshot.lvo_bucket.asc(),
                desc(AccountSnapshot.lvo_snapshotdate),
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
    """Find the comparison snapshot for a ``ComparePeriod`` literal.

    Returns ``{}`` when the lookback maps to an unsupported period or no
    snapshot exists ≤ the target date.
    """
    if period not in PERIOD_LOOKBACK_DAYS:
        return {}
    today = today or datetime.now(timezone.utc).date()
    target = today - timedelta(days=PERIOD_LOOKBACK_DAYS[period])
    return lookup_snapshot(db, on_or_before=target)


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

    Defaults to ``today``, ``today-7``, ``today-30`` and ``today-90`` so
    the KPI strip shows non-null trends right after a fresh deploy. Each
    date uses itself as the ``existed_by`` filter — see the module
    docstring for accuracy caveats per bucket.
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
    "backfill",
    "compute_buckets",
    "lookup_previous_for_period",
    "lookup_snapshot",
    "take_snapshot",
]

"""Account read endpoints — backs the Deal Detailed View, the View
Account user story, the Customer Information tab, and the per-account
Opportunities tab.

Endpoints
---------
GET    /api/accounts                              Paginated, filterable list.
GET    /api/accounts/{id}                         Full account-detail payload.
GET    /api/accounts/{id}/opportunities           Linked deals — filterable, with timeline + risk count.
GET    /api/accounts/{id}/opportunities/export    CSV stream of the filtered set.
GET    /api/accounts/{id}/customer-information    Sectioned read-only profile.
GET    /api/accounts/filters                      Distinct values for the filter pickers.
GET    /api/accounts/value-range                  Min/max totalAccountValue (slider).
POST   /api/accounts/{id}/recompute-status        Force-recompute derived fields.

Account-level contact CRUD lives in ``app/routers/account_contacts.py``.

Soft-deleted opportunities (``statecode='Canceled'``) are excluded from
every aggregate by ``apply_opportunity_filters(... include_canceled=False)``.
"""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterator, Sequence

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import Numeric, String, asc, cast, desc, func, inspect, select, text
from sqlalchemy.orm import Session, defer, undefer

from app.database import get_db
from app.filters_query import (
    CANCELED_STATECODE,
    apply_account_filters,
    apply_opportunity_filters,
    total_account_value_subquery,
)
from app.models import (
    Account,
    Activity,
    DealRisk,
    Opportunity,
    OpportunityCompetitor,
)
from app.normalizers import normalise_risk_score, normalise_sale_motion, normalise_stage
from app.routers.deals_read import build_account_summary
from app.schemas import (
    AccountDetail,
    AccountFiltersResponse,
    AccountKpiBucket,
    AccountKpiCard,
    AccountKpiSummaryResponse,
    AccountListItem,
    AccountListResponse,
    AccountOpportunitiesResponse,
    AccountRecomputeResponse,
    AccountSort,
    AccountSummary,
    AccountValueRangeResponse,
    ActivityItem,
    ComparePeriod,
    CustomerInfoAddress,
    CustomerInfoBasicInformation,
    CustomerInfoCommercialTerms,
    CustomerInfoIdentityLegal,
    CustomerInfoTerritoryOwnership,
    CustomerInformationResponse,
    OpportunityListItem,
    OpportunitySort,
    SaleMotionRef,
    SortOrder,
    StageRef,
    TrendInfo,
)
from app.services.account_columns import (
    LENOVO_CUSTOM_COLUMNS,
    STANDARD_D365_COLUMNS,
    get_account_columns,
)
from app.services.account_kpi_snapshots import (
    BucketAggregate as AccountBucketAggregate,
    PERIOD_LOOKBACK_DAYS as ACCOUNT_PERIOD_LOOKBACK_DAYS,
    compute_buckets as compute_account_buckets,
    lookup_previous_for_period as lookup_account_previous_for_period,
)
from app.services.account_recalc import recalculate_account, recalculate_async
from app.services.kpi_snapshots import compute_trend_info

router = APIRouter(prefix="/api/accounts", tags=["accounts"])


# ---------------------------------------------------------------------------
# Internal constants / helpers
# ---------------------------------------------------------------------------

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100

# Maps the public sortBy enum to the column the listing query orders on.
# Resolved AFTER the value-rollup subquery is built so the
# totalAccountValue / openDealsCount sorts can target the alias columns.
_VALID_SORT_COLUMNS: tuple[AccountSort, ...] = (
    "name",
    "totalAccountValue",
    "openDealsCount",
    "lastInteraction",
    "status",
    "lvoAccountType",
)


def _has_table(db: Session, table_name: str) -> bool:
    return inspect(db.bind).has_table(table_name)


def _has_account_view_columns(db: Session) -> bool:
    """True only when sql/2026_06_account_view_schema.sql has been applied.

    The list endpoint defers the optional columns when they are missing so
    the rest of the page keeps working on a partially-migrated DB.
    """
    cols = {c["name"] for c in inspect(db.bind).get_columns("account")}
    return {
        "lvo_accounttype",
        "lvo_accountstatus",
        "lvo_lastinteractiondate",
    }.issubset(cols)


def _normalise_csv(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None


def _ensure_account(db: Session, account_id: str) -> Account:
    # Case-insensitive UUID match — Postgres serialises native ``uuid`` to
    # lowercase text, but FE URLs carry uppercase hyphen-blocks. Without
    # the UPPER on both sides we 404 every uppercase URL.
    account = db.execute(
        select(Account).where(
            func.upper(cast(Account.accountid, String)) == account_id.upper()
        )
    ).scalar_one_or_none()
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account '{account_id}' not found.",
        )
    return account


def _pick_region_label(account: Account) -> str | None:
    """Single-string region for the grid column.

    Highest-precedence label among lvo_territory > lvo_businessgroupid > lvo_countryid.
    """
    return account.lvo_territory or account.lvo_businessgroupid or account.lvo_countryid


# ---------------------------------------------------------------------------
# Seller name resolution
# ---------------------------------------------------------------------------


def _resolve_seller_names(
    db: Session, owner_ids: Sequence[str]
) -> dict[str, str]:
    """Resolve owninguser UUIDs to display names.

    Uses runtime introspection — joins to ``systemuser.fullname`` only when
    that table has the fields we need; otherwise returns an empty dict and
    the caller falls back to the raw UUID. Keeps this endpoint usable on
    dumps that don't ship a systemuser table.
    """
    if not owner_ids:
        return {}
    inspector = inspect(db.bind)
    if not inspector.has_table("systemuser"):
        return {}
    cols = {c["name"] for c in inspector.get_columns("systemuser")}
    if "systemuserid" not in cols:
        return {}
    name_col = (
        "fullname" if "fullname" in cols
        else "internalemailaddress" if "internalemailaddress" in cols
        else None
    )
    if name_col is None:
        return {}

    upper_ids = [oid.upper() for oid in owner_ids if oid]
    if not upper_ids:
        return {}

    # Raw SQL is the lightest touch here — we don't model systemuser as an
    # ORM class (it's not present on every dump).
    sql = (
        f"SELECT UPPER(systemuserid::TEXT) AS uid, {name_col} AS label "
        "FROM systemuser "
        "WHERE UPPER(systemuserid::TEXT) = ANY(:ids)"
    )
    rows = db.execute(text(sql), {"ids": upper_ids}).all()
    return {row.uid: (row.label or row.uid) for row in rows}


# ---------------------------------------------------------------------------
# GET /api/accounts/filters — distinct values for the multi-select pickers
# ---------------------------------------------------------------------------


@router.get(
    "/filters",
    response_model=AccountFiltersResponse,
    summary="Distinct filter options for the View Account grid",
)
def get_account_filter_options(
    db: Session = Depends(get_db),
) -> AccountFiltersResponse:
    has_view_cols = _has_account_view_columns(db)

    if has_view_cols:
        types = (
            db.execute(
                select(Account.lvo_accounttype)
                .where(Account.lvo_accounttype.is_not(None))
                .distinct()
                .order_by(Account.lvo_accounttype)
            )
            .scalars()
            .all()
        )
        statuses = (
            db.execute(
                select(Account.lvo_accountstatus)
                .where(Account.lvo_accountstatus.is_not(None))
                .distinct()
                .order_by(Account.lvo_accountstatus)
            )
            .scalars()
            .all()
        )
    else:
        types = []
        statuses = []

    segments = (
        db.execute(
            select(Account.lvo_segment)
            .where(Account.lvo_segment.is_not(None))
            .distinct()
            .order_by(Account.lvo_segment)
        )
        .scalars()
        .all()
    )
    industries = (
        db.execute(
            select(Account.industrycode)
            .where(Account.industrycode.is_not(None))
            .distinct()
            .order_by(Account.industrycode)
        )
        .scalars()
        .all()
    )
    region_rows = db.execute(
        select(
            Account.lvo_territory,
            Account.lvo_businessgroupid,
            Account.lvo_countryid,
        )
    ).all()
    regions: list[str] = sorted(
        {
            value
            for row in region_rows
            for value in (row.lvo_territory, row.lvo_businessgroupid, row.lvo_countryid)
            if value
        }
    )

    return AccountFiltersResponse(
        account_types=list(types),
        account_statuses=list(statuses),
        segments=list(segments),
        regions=regions,
        industries=list(industries),
    )


# ---------------------------------------------------------------------------
# GET /api/accounts/value-range — min / max totalAccountValue for the slider
# ---------------------------------------------------------------------------


@router.get(
    "/value-range",
    response_model=AccountValueRangeResponse,
    summary="Dynamic min/max totalAccountValue for the value-range filter",
)
def get_account_value_range(
    db: Session = Depends(get_db),
) -> AccountValueRangeResponse:
    """Compute the min/max sum of estimatedvalue grouped per account.

    Returns the raw min/max so the FE can position the range slider; the
    range only includes accounts that have at least one non-Canceled deal.
    """
    rollup = total_account_value_subquery()
    row = db.execute(
        select(
            func.coalesce(func.min(rollup.c.total_value), 0).label("min_v"),
            func.coalesce(func.max(rollup.c.total_value), 0).label("max_v"),
        )
    ).one()
    return AccountValueRangeResponse(
        min=float(row.min_v or 0),
        max=float(row.max_v or 0),
        currency="USD",
    )


# ---------------------------------------------------------------------------
# GET /api/accounts/kpi-summary — Top-of-page KPI strip
#
# Four cards: Total Accounts / Account Value (ACV) / Active Accounts /
# Accounts at Risk. Mirrors /api/opportunities/kpi-summary (Strategy A on
# filters: trend is null whenever any filter is supplied).
# ---------------------------------------------------------------------------


def _compute_filtered_account_buckets(
    db: Session,
    *,
    search: str | None,
    account_types: list[str] | None,
    account_statuses: list[str] | None,
    segments: list[str] | None,
    regions: list[str] | None,
    industries: list[str] | None,
    value_min: float | None,
    value_max: float | None,
) -> dict[str, AccountBucketAggregate]:
    """Compute the four KPI buckets for a filtered account set.

    Two queries:
      1. The filtered account list (subquery) → ``total`` /  ``active``
         / ``at_risk`` counts.
      2. ACV → SUM(estimatedvalue) for non-canceled opps whose accountid
         is in the filtered set (DISTINCT-counted for ``count``).
    """
    rollup = total_account_value_subquery()
    total_value_col = func.coalesce(rollup.c.total_value, 0)

    filtered_acct_stmt = (
        select(Account.accountid, Account.lvo_accountstatus)
        .select_from(Account)
        .join(
            rollup,
            rollup.c.account_key == func.upper(cast(Account.accountid, String)),
            isouter=True,
        )
    )
    filtered_acct_stmt = apply_account_filters(
        filtered_acct_stmt,
        search=search,
        account_types=account_types,
        account_statuses=account_statuses,
        segments=segments,
        regions=regions,
        industries=industries,
        value_min=value_min,
        value_max=value_max,
        total_account_value_col=total_value_col,
    )
    filtered_subq = filtered_acct_stmt.subquery()

    acct_aggs = db.execute(
        select(
            func.count().label("total_count"),
            func.count()
            .filter(filtered_subq.c.lvo_accountstatus == "Active")
            .label("active_count"),
            func.count()
            .filter(filtered_subq.c.lvo_accountstatus == "At-Risk")
            .label("at_risk_count"),
        ).select_from(filtered_subq)
    ).one()

    value_col = cast(Opportunity.estimatedvalue, Numeric)
    acv_row = db.execute(
        select(
            func.coalesce(func.sum(value_col), 0).label("acv_value"),
            func.count(
                func.distinct(func.upper(cast(Opportunity.accountid, String)))
            ).label("acv_count"),
        )
        .select_from(Opportunity)
        .where(
            func.coalesce(Opportunity.statecode, "") != CANCELED_STATECODE,
            func.upper(cast(Opportunity.accountid, String)).in_(
                select(func.upper(cast(filtered_subq.c.accountid, String)))
            ),
        )
    ).one()

    return {
        "total": AccountBucketAggregate(
            bucket="total", value=0.0, count=int(acct_aggs.total_count or 0)
        ),
        "acv": AccountBucketAggregate(
            bucket="acv",
            value=float(acv_row.acv_value or 0),
            count=int(acv_row.acv_count or 0),
        ),
        "active": AccountBucketAggregate(
            bucket="active", value=0.0, count=int(acct_aggs.active_count or 0)
        ),
        "at_risk": AccountBucketAggregate(
            bucket="at_risk", value=0.0, count=int(acct_aggs.at_risk_count or 0)
        ),
    }


@router.get(
    "/kpi-summary",
    response_model=AccountKpiSummaryResponse,
    summary="KPI cards strip — Total / Account Value / Active / At-Risk",
)
def account_kpi_summary(
    compare_period: ComparePeriod = Query(
        "last_week",
        alias="comparePeriod",
        description=(
            "Period to compare against. Trend will be null until a "
            "lvo_accountsnapshot row exists for the lookback date."
        ),
    ),
    search: str | None = Query(None, description="ILIKE on account name + accountnumber."),
    account_types: str | None = Query(
        None,
        alias="accountTypes",
        description="Comma-separated list of Prospect / Customer.",
    ),
    account_statuses: str | None = Query(
        None,
        alias="accountStatuses",
        description="Comma-separated list of Active / Inactive / At-Risk.",
    ),
    segments: str | None = Query(
        None,
        description="Comma-separated list of SMB / Mid-Market / Enterprise / Strategic.",
    ),
    regions: str | None = Query(
        None,
        description=(
            "Comma-separated list. Matches lvo_territory, lvo_businessgroupid "
            "or lvo_countryid (case-insensitive)."
        ),
    ),
    industries: str | None = Query(
        None,
        description="Comma-separated list of industry codes.",
    ),
    value_min: float | None = Query(None, alias="valueMin", ge=0),
    value_max: float | None = Query(None, alias="valueMax", ge=0),
    db: Session = Depends(get_db),
) -> AccountKpiSummaryResponse:
    """Return the four KPI cards plus optional period-over-period trends.

    The unfiltered call computes aggregates against the entire account
    table (matching the snapshot service's ``compute_buckets``) so the
    "now" numbers and "N days ago" numbers come from comparable inputs.

    Filtered calls run a second code path that joins the same filter set
    used by ``/api/accounts``; the returned ``trend`` is forced to ``null``
    in that case (Strategy A — snapshots are global / unfiltered in v1).
    """
    types_list = _normalise_csv(account_types)
    statuses_list = _normalise_csv(account_statuses)
    segments_list = _normalise_csv(segments)
    regions_list = _normalise_csv(regions)
    industries_list = _normalise_csv(industries)

    has_filters = any(
        [
            bool(search),
            bool(types_list),
            bool(statuses_list),
            bool(segments_list),
            bool(regions_list),
            bool(industries_list),
            value_min is not None,
            value_max is not None,
        ]
    )

    if has_filters:
        live_buckets = _compute_filtered_account_buckets(
            db,
            search=search,
            account_types=types_list,
            account_statuses=statuses_list,
            segments=segments_list,
            regions=regions_list,
            industries=industries_list,
            value_min=value_min,
            value_max=value_max,
        )
    else:
        # Unfiltered path uses the same primitive the snapshot service does
        # so today's numbers are directly comparable with snapshot rows.
        live_buckets = compute_account_buckets(db)

    previous_buckets: dict[str, AccountBucketAggregate] = {}
    if not has_filters:
        previous_buckets = lookup_account_previous_for_period(
            db, period=compare_period
        )

    def _trend_for(bucket: str) -> TrendInfo | None:
        if has_filters:
            return None
        prev = previous_buckets.get(bucket)
        delta = compute_trend_info(live_buckets[bucket], prev)
        if delta is None:
            return None
        return TrendInfo(
            direction=delta.direction,
            delta_value=delta.delta_value,
            delta_count=delta.delta_count,
        )

    def card(bucket: str) -> AccountKpiCard:
        live = live_buckets[bucket]
        return AccountKpiCard(
            value=live.value, count=live.count, trend=_trend_for(bucket)
        )

    notes: list[str] = []
    if has_filters:
        notes.append(
            "Trend is suppressed because filters are applied — v1 snapshots "
            "are global only. Re-issue without filters to get period-over-period change."
        )
    elif not previous_buckets:
        lookback_days = ACCOUNT_PERIOD_LOOKBACK_DAYS.get(compare_period)
        notes.append(
            "Trend is null because no lvo_accountsnapshot row exists on or "
            f"before today minus {lookback_days} days. Run "
            "`python -m app.jobs.snapshot_account_kpis --backfill` once, "
            "then schedule `python -m app.jobs.snapshot_account_kpis` nightly."
        )

    return AccountKpiSummaryResponse(
        compare_period=compare_period,
        as_of=datetime.now(timezone.utc),
        currency="USD",
        total_accounts=card("total"),
        account_value=card("acv"),
        active_accounts=card("active"),
        accounts_at_risk=card("at_risk"),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# GET /api/accounts/export — CSV download of the filtered set
# ---------------------------------------------------------------------------


_EXPORT_COLUMNS: list[tuple[str, str]] = [
    ("id", "Account ID"),
    ("account_number", "Account Number"),
    ("name", "Account Name"),
    ("account_type", "Type"),
    ("industry", "Industry"),
    ("segment", "Segment"),
    ("region", "Region"),
    ("business_group", "Business Group"),
    ("country", "Country"),
    ("territory", "Territory"),
    ("status", "Status"),
    ("statecode", "Statecode"),
    ("last_interaction", "Last Interaction"),
    ("active_opportunities_count", "Active Opportunities"),
    ("total_account_value", "Total Account Value"),
    ("currency", "Currency"),
    ("revenue", "Revenue"),
    ("employee_count", "Employees"),
    ("owner_id", "Seller ID"),
    ("owner_name", "Seller"),
]


def _format_export_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        # Two-decimal currency-friendly rendering with no thousands sep.
        return f"{value:.2f}"
    return str(value)


def _stream_account_csv(items: list[AccountListItem]) -> Iterator[str]:
    """Generator yielding CSV chunks for ``StreamingResponse``."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([label for _, label in _EXPORT_COLUMNS])
    yield buffer.getvalue()
    buffer.seek(0)
    buffer.truncate()

    for item in items:
        # Pydantic returns the field by name; populate_by_name lets us index
        # via the snake_case attribute regardless of the wire alias.
        writer.writerow(
            [_format_export_value(getattr(item, field)) for field, _ in _EXPORT_COLUMNS]
        )
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate()


@router.get(
    "/export",
    summary="CSV export of the filtered account list",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/csv": {}},
            "description": "CSV stream of accounts matching the filter set.",
        }
    },
)
def export_accounts(
    search: str | None = Query(None),
    account_types: str | None = Query(None, alias="accountTypes"),
    account_statuses: str | None = Query(None, alias="accountStatuses"),
    segments: str | None = Query(None),
    regions: str | None = Query(None),
    industries: str | None = Query(None),
    value_min: float | None = Query(None, alias="valueMin", ge=0),
    value_max: float | None = Query(None, alias="valueMax", ge=0),
    bucket: AccountKpiBucket | None = Query(
        None,
        description=(
            "KPI-strip drill-down — same semantics as /api/accounts. "
            "Lets the Export button continue to work after the user has "
            "clicked a KPI card."
        ),
    ),
    sort_by: AccountSort = Query("name", alias="sortBy"),
    sort_order: SortOrder = Query("asc", alias="sortOrder"),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Export the same dataset the grid is showing as CSV.

    Honours every filter/sort the list endpoint accepts so the FE can simply
    forward the active query string. Response is streamed in chunks so very
    large exports don't pin the whole table in memory.
    """
    # Re-use the list endpoint's logic to keep the export 100% in-sync. The
    # MAX_PAGE_SIZE cap is bypassed here on purpose: an export should
    # contain everything that matches.
    response = list_accounts(  # type: ignore[call-arg]
        page=1,
        page_size=MAX_PAGE_SIZE,
        search=search,
        account_types=account_types,
        account_statuses=account_statuses,
        segments=segments,
        regions=regions,
        industries=industries,
        value_min=value_min,
        value_max=value_max,
        bucket=bucket,
        sort_by=sort_by,
        sort_order=sort_order,
        db=db,
    )
    items: list[AccountListItem] = list(response.items)

    # If the dataset is bigger than one page, keep paging server-side until
    # we've collected everything. We rely on the existing total + total_pages
    # so callers don't have to loop.
    page = 2
    while page <= response.total_pages:
        next_page = list_accounts(  # type: ignore[call-arg]
            page=page,
            page_size=MAX_PAGE_SIZE,
            search=search,
            account_types=account_types,
            account_statuses=account_statuses,
            segments=segments,
            regions=regions,
            industries=industries,
            value_min=value_min,
            value_max=value_max,
            bucket=bucket,
            sort_by=sort_by,
            sort_order=sort_order,
            db=db,
        )
        items.extend(next_page.items)
        page += 1

    filename = f"accounts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        _stream_account_csv(items),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# GET /api/accounts — paginated list (the core View Account endpoint)
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=AccountListResponse,
    summary="Paginated, filterable, sortable list of accounts",
)
def list_accounts(
    page: int = Query(1, ge=1, description="1-based page number."),
    page_size: int = Query(
        DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        alias="pageSize",
    ),
    search: str | None = Query(None, description="ILIKE on account name + accountnumber."),
    account_types: str | None = Query(
        None,
        alias="accountTypes",
        description="Comma-separated list of Prospect / Customer.",
    ),
    account_statuses: str | None = Query(
        None,
        alias="accountStatuses",
        description=(
            "Comma-separated list of Active / Inactive / At-Risk. "
            "When omitted, only Active and At-Risk accounts are returned."
        ),
    ),
    segments: str | None = Query(
        None,
        description="Comma-separated list of SMB / Mid-Market / Enterprise / Strategic.",
    ),
    regions: str | None = Query(
        None,
        description=(
            "Comma-separated list. Matches lvo_territory, lvo_businessgroupid "
            "or lvo_countryid (case-insensitive)."
        ),
    ),
    industries: str | None = Query(
        None,
        description="Comma-separated list of industry codes.",
    ),
    value_min: float | None = Query(
        None,
        alias="valueMin",
        ge=0,
        description="Lower bound on totalAccountValue (USD).",
    ),
    value_max: float | None = Query(
        None,
        alias="valueMax",
        ge=0,
        description="Upper bound on totalAccountValue (USD).",
    ),
    bucket: AccountKpiBucket | None = Query(
        None,
        description=(
            "KPI-strip drill-down. One of total / acv / active / at_risk. "
            "Mirrors the predicate used by /api/accounts/kpi-summary so the "
            "card count and the resulting grid count agree. Composes with "
            "every other filter — pass `bucket=at_risk&regions=APAC` for the "
            "intersection. `bucket=total` additionally allows Inactive rows, "
            "which the default filter set hides."
        ),
    ),
    sort_by: AccountSort = Query("name", alias="sortBy"),
    sort_order: SortOrder = Query("asc", alias="sortOrder"),
    db: Session = Depends(get_db),
) -> AccountListResponse:
    """Build the Accounts grid in one query.

    The aggregate rollup (total value, open deals count, last won date) is
    pulled in via a single LEFT JOIN to a SUM/COUNT subquery so the grid
    scales to thousands of accounts without N+1.
    """
    has_view_cols = _has_account_view_columns(db)
    has_systemuser = _has_table(db, "systemuser")

    rollup = total_account_value_subquery()

    select_cols: list[Any] = [
        Account,
        func.coalesce(rollup.c.total_value, 0).label("total_value"),
        func.coalesce(rollup.c.open_count, 0).label("open_count"),
        rollup.c.last_won_date.label("last_won_date"),
    ]

    base_stmt = select(*select_cols).select_from(Account).join(
        rollup,
        rollup.c.account_key == func.upper(cast(Account.accountid, String)),
        isouter=True,
    )
    base_stmt = apply_account_filters(
        base_stmt,
        search=search,
        account_types=_normalise_csv(account_types),
        account_statuses=_normalise_csv(account_statuses),
        segments=_normalise_csv(segments),
        regions=_normalise_csv(regions),
        industries=_normalise_csv(industries),
        owner_id=None,  # Owner-based filtering disabled by design (q1 = no_filter_yet).
        value_min=value_min,
        value_max=value_max,
        total_account_value_col=func.coalesce(rollup.c.total_value, 0),
        bucket=bucket,
    )

    # ---- Total count -------------------------------------------------------
    count_stmt = select(func.count(Account.accountid.distinct())).select_from(
        Account
    ).join(
        rollup,
        rollup.c.account_key == func.upper(cast(Account.accountid, String)),
        isouter=True,
    )
    count_stmt = apply_account_filters(
        count_stmt,
        search=search,
        account_types=_normalise_csv(account_types),
        account_statuses=_normalise_csv(account_statuses),
        segments=_normalise_csv(segments),
        regions=_normalise_csv(regions),
        industries=_normalise_csv(industries),
        owner_id=None,
        value_min=value_min,
        value_max=value_max,
        total_account_value_col=func.coalesce(rollup.c.total_value, 0),
    )
    total: int = int(db.execute(count_stmt).scalar_one() or 0)

    # ---- Sort --------------------------------------------------------------
    sort_col_map: dict[str, Any] = {
        "name": Account.name,
        "totalAccountValue": func.coalesce(rollup.c.total_value, 0),
        "openDealsCount": func.coalesce(rollup.c.open_count, 0),
        "lastInteraction": (
            Account.lvo_lastinteractiondate if has_view_cols else Account.name
        ),
        "status": (
            Account.lvo_accountstatus if has_view_cols else Account.name
        ),
        "lvoAccountType": (
            Account.lvo_accounttype if has_view_cols else Account.name
        ),
    }
    sort_col = sort_col_map.get(sort_by, Account.name)
    order_fn = asc if sort_order == "asc" else desc

    rows_stmt = (
        base_stmt.order_by(order_fn(sort_col), asc(Account.accountid))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    if not has_view_cols:
        # Defensive — keeps the grid alive on a DB where the migration
        # hasn't been applied. The columns simply read as None.
        rows_stmt = rows_stmt.options(
            defer(Account.lvo_accounttype, raiseload=True),
            defer(Account.lvo_accountstatus, raiseload=True),
            defer(Account.lvo_lastinteractiondate, raiseload=True),
        )

    rows = db.execute(rows_stmt).all()

    # ---- Resolve seller names in one batch (best-effort) ------------------
    owner_ids = [r.Account.owninguser for r in rows if r.Account.owninguser]
    seller_names = (
        _resolve_seller_names(db, owner_ids) if has_systemuser else {}
    )

    items: list[AccountListItem] = []
    for row in rows:
        a: Account = row.Account
        owner_id = a.owninguser
        items.append(
            AccountListItem(
                id=str(a.accountid),
                name=a.name,
                account_number=a.accountnumber,
                account_type=a.lvo_accounttype if has_view_cols else None,
                industry=a.industrycode,
                segment=a.lvo_segment,
                region=_pick_region_label(a),
                business_group=a.lvo_businessgroupid,
                country=a.lvo_countryid,
                territory=a.lvo_territory,
                status=a.lvo_accountstatus if has_view_cols else None,
                statecode=a.statecode,
                last_interaction=(
                    a.lvo_lastinteractiondate if has_view_cols else None
                ),
                active_opportunities_count=int(row.open_count or 0),
                total_account_value=float(row.total_value or 0),
                currency=a.lvo_defaultcurrency or "USD",
                employee_count=a.numberofemployees,
                revenue=float(a.revenue) if a.revenue is not None else None,
                owner_id=owner_id,
                owner_name=(
                    seller_names.get(owner_id.upper()) if owner_id else None
                ),
            )
        )

    total_pages = math.ceil(total / page_size) if page_size else 0

    notes: list[str] = []
    if total == 0:
        # Canonical empty-state code from the user story.
        notes.append("ERR_MSG_0010")
    if not has_view_cols:
        notes.append(
            "account.lvo_accounttype / lvo_accountstatus / lvo_lastinteractiondate "
            "not found — run sql/2026_06_account_view_schema.sql to populate them."
        )

    return AccountListResponse(
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        sort_by=sort_by,
        sort_order=sort_order,
        items=items,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# GET /api/accounts/{id} — full detail (REPLACES the previous minimal version)
# ---------------------------------------------------------------------------


@router.get(
    "/{account_id}",
    response_model=AccountDetail,
    summary="Account profile + rollups + status/type for the detail page",
    responses={404: {"description": "Account not found"}},
)
def get_account(
    account_id: str = Path(..., description="account.accountid (UUID)"),
    db: Session = Depends(get_db),
) -> AccountDetail:
    """Return the full ``AccountDetail`` payload.

    Combines the cached row + an on-the-fly recalc of derived fields so the
    detail page never shows a stale status. Persistence is opt-in here — the
    GET stays read-only — sellers can hit POST /recompute-status to write
    the recomputed values back.
    """
    account = _ensure_account(db, account_id)
    summary: AccountSummary = build_account_summary(db, account_id)

    # Live derivation (no write) so the page is idempotent.
    derived = recalculate_account(db, account_id, write=False)

    has_view_cols = _has_account_view_columns(db)

    # --- Total / Won / Lost / Canceled counts --------------------------------
    rollup_row = db.execute(
        select(
            func.count().label("total"),
            func.count()
                .filter(
                    Opportunity.statecode.in_(("Won", "Closed Won"))
                    | Opportunity.stagename.in_(("Closed Won",))
                )
                .label("won"),
            func.count()
                .filter(
                    Opportunity.statecode.in_(("Lost", "Closed Lost"))
                    | Opportunity.stagename.in_(("Closed Lost",))
                )
                .label("lost"),
            func.count()
                .filter(Opportunity.statecode == "Canceled")
                .label("canceled"),
        )
        .select_from(Opportunity)
        .where(func.upper(Opportunity.accountid) == account_id.upper())
    ).one()

    seller_names = _resolve_seller_names(
        db, [account.owninguser] if account.owninguser else []
    )

    return AccountDetail(
        id=str(account.accountid),
        name=account.name,
        account_number=account.accountnumber,
        account_type=(
            account.lvo_accounttype
            if has_view_cols
            else (derived.account_type if derived else None)
        ),
        industry=account.industrycode,
        segment=account.lvo_segment,
        region=_pick_region_label(account),
        business_group=account.lvo_businessgroupid,
        country=account.lvo_countryid,
        territory=account.lvo_territory,
        status=(
            account.lvo_accountstatus
            if has_view_cols
            else (derived.status if derived else None)
        ),
        statecode=account.statecode,
        last_interaction=(
            account.lvo_lastinteractiondate
            if has_view_cols
            else (derived.last_interaction if derived else None)
        ),
        active_opportunities_count=summary.open_deals_count,
        total_account_value=summary.total_account_value,
        currency=account.lvo_defaultcurrency or "USD",
        employee_count=account.numberofemployees,
        revenue=float(account.revenue) if account.revenue is not None else None,
        owner_id=account.owninguser,
        owner_name=(
            seller_names.get((account.owninguser or "").upper())
            if account.owninguser
            else None
        ),
        won_deals_count=int(rollup_row.won or 0),
        lost_deals_count=int(rollup_row.lost or 0),
        canceled_deals_count=int(rollup_row.canceled or 0),
        total_deals_count=int(rollup_row.total or 0),
    )


# ---------------------------------------------------------------------------
# GET /api/accounts/{id}/opportunities — linked deals (per-account grid)
#
# Powers the "Opportunities" tab on the Account Detail screen. Returns the
# same OpportunityListItem shape the main grid uses so the FE can re-use
# its row component, but pre-filtered to ``account_id = :id`` and enriched
# with:
#
#   * a populated ``activities`` array (last 90 days by default, with
#     same-day events collapsed into a ``type='multiple'`` marker) — drives
#     the timeline dots in the row;
#   * ``riskCount`` (count of Active rows in lvo_dealrisk) — drives the
#     '⚠ N' badge in the Risk column;
#   * the same ``search`` / ``regions`` / ``industries`` / ``stages`` /
#     ``products`` filters the main grid accepts, so the toolbar
#     dropdowns work with no special-casing.
# ---------------------------------------------------------------------------


# Sort columns surfaced by the per-account grid. Mirrors the main
# /api/opportunities endpoint so the FE can use one sortBy enum across
# both grids. Default order is most-recent-close-date first, matching the
# previous implementation.
_OPP_SORT_COLUMN_MAP: dict[OpportunitySort, Any] = {
    "name": Opportunity.name,
    "value": Opportunity.estimatedvalue,
    "closeDate": Opportunity.estimatedclosedate,
    "closeProbability": Opportunity.closeprobability,
    "stage": Opportunity.stagename,
}

# Cap how many activity events we surface per opportunity row — the
# timeline strip can't render more dots than this and the day-bucketing
# already collapses bursts.
_MAX_ACTIVITY_EVENTS_PER_OPP = 30
_DEFAULT_TIMELINE_DAYS = 90


def _bucket_activities_per_opportunity(
    rows: Sequence[Activity],
) -> list[ActivityItem]:
    """Collapse a chronological list of Activity rows into timeline events.

    Bucketing rule (matches the legend in the UI mockup):

    * **1 event on a calendar day**  → emit a normal ActivityItem with the
      event's own type (``email`` / ``meeting`` / ``crm``).
    * **2+ events on the same day**  → emit a single ``type='multiple'``
      marker carrying ``groupedCount=N``. The FE renders this as the
      numbered circle ("5") shown in the mockup legend.

    Events are ordered most-recent-first within the returned list so the
    UI can slice ``activities[:N]`` without re-sorting.
    """
    by_day: dict[date, list[Activity]] = defaultdict(list)
    for event in rows:
        if event.lvo_activitydate is None:
            continue
        day_key = event.lvo_activitydate.date()
        by_day[day_key].append(event)

    out: list[ActivityItem] = []
    # Sort days descending so most-recent dots come first.
    for day_key in sorted(by_day.keys(), reverse=True):
        day_events = by_day[day_key]
        if len(day_events) == 1:
            event = day_events[0]
            out.append(
                ActivityItem(
                    id=event.lvo_activityid,
                    type=_normalise_activity_type(event.lvo_activitytype),
                    direction=event.lvo_direction,
                    subject=event.lvo_subject,
                    body=event.lvo_body,
                    activity_date=event.lvo_activitydate,
                    grouped_count=event.lvo_groupedcount,
                )
            )
        else:
            # Pick the most recent event on the day to anchor the marker;
            # its body/subject aren't used by the timeline strip but are
            # surfaced for offcanvas hover-state parity.
            anchor = max(
                day_events,
                key=lambda e: e.lvo_activitydate or datetime.min,
            )
            out.append(
                ActivityItem(
                    id=anchor.lvo_activityid,
                    type="multiple",
                    direction=None,
                    subject=anchor.lvo_subject,
                    body=anchor.lvo_body,
                    activity_date=anchor.lvo_activitydate,
                    grouped_count=len(day_events),
                )
            )
        if len(out) >= _MAX_ACTIVITY_EVENTS_PER_OPP:
            break
    return out


def _normalise_activity_type(raw: str | None) -> str:
    """Map raw lvo_activitytype values onto the ActivityType literal."""
    if not raw:
        return "crm"
    lowered = raw.strip().lower()
    if lowered in {"email", "meeting", "crm", "multiple"}:
        return lowered
    if "email" in lowered:
        return "email"
    if "meeting" in lowered or "call" in lowered:
        return "meeting"
    return "crm"


def _opportunity_sort_clause(
    sort_by: OpportunitySort, sort_order: SortOrder
) -> tuple[Any, Any]:
    """Return the primary + tie-breaker ORDER BY columns."""
    primary = _OPP_SORT_COLUMN_MAP[sort_by]
    direction = asc if sort_order == "asc" else desc
    return direction(primary), asc(Opportunity.opportunityid)


def _resolve_account_opportunities(
    db: Session,
    *,
    account_id: str,
    page: int,
    page_size: int,
    include_canceled: bool,
    search: str | None,
    regions: list[str] | None,
    industries: list[str] | None,
    stages: list[str] | None,
    products: list[str] | None,
    sort_by: OpportunitySort,
    sort_order: SortOrder,
    timeline_days: int,
) -> tuple[int, list[OpportunityListItem]]:
    """Shared resolver — backs the list endpoint AND the CSV export.

    Returns ``(total, items)`` so the export can stream every page in a
    single loop without re-implementing the filter/activity/risk-count
    pipeline.
    """
    base_filter = func.upper(Opportunity.accountid) == account_id.upper()
    filter_kwargs = {
        "search": search,
        "regions": regions,
        "industries": industries,
        "stages": stages,
        "products": products,
        "include_canceled": include_canceled,
    }

    # ---- Total count ----------------------------------------------------
    count_stmt = (
        select(func.count(Opportunity.opportunityid.distinct()))
        .select_from(Opportunity)
        .where(base_filter)
    )
    count_stmt = apply_opportunity_filters(count_stmt, **filter_kwargs)
    total = int(db.execute(count_stmt).scalar_one() or 0)

    has_activity_table = _has_table(db, "lvo_activity")
    has_dealrisk_table = _has_table(db, "lvo_dealrisk")
    inspector = inspect(db.bind)
    opportunity_columns = {c["name"] for c in inspector.get_columns("opportunity")}
    has_deal_health = {
        "lvo_dealhealthscore",
        "lvo_riskscore",
        "lvo_riskreason",
    }.issubset(opportunity_columns)
    has_deal_detail = {
        "lvo_tempoclass",
        "lvo_stageentrydate",
        "lvo_createdat",
        "lvo_dealhealthupdatedat",
    }.issubset(opportunity_columns)

    # ---- Subqueries (last_activity + risk_count) ------------------------
    if has_activity_table:
        last_activity_subq = (
            select(
                func.upper(Activity.lvo_opportunityid).label("opportunity_key"),
                func.max(Activity.lvo_activitydate).label("last_activity"),
            )
            .where(Activity.statecode == "Active")
            .group_by(func.upper(Activity.lvo_opportunityid))
            .subquery()
        )
    else:
        last_activity_subq = None

    if has_dealrisk_table:
        risk_count_subq = (
            select(
                func.upper(DealRisk.lvo_opportunityid).label("opportunity_key"),
                func.count().label("risk_count"),
            )
            .where(DealRisk.statecode == "Active")
            .group_by(func.upper(DealRisk.lvo_opportunityid))
            .subquery()
        )
    else:
        risk_count_subq = None

    # ---- Build the page query -------------------------------------------
    select_cols: list[Any] = [
        Opportunity,
        Account.name.label("account_name"),
        Account.industrycode.label("industry"),
    ]
    if last_activity_subq is not None:
        select_cols.append(last_activity_subq.c.last_activity.label("last_activity"))
    if risk_count_subq is not None:
        select_cols.append(risk_count_subq.c.risk_count.label("risk_count"))

    rows_stmt = select(*select_cols).join(
        Account,
        cast(Account.accountid, String) == Opportunity.accountid,
        isouter=True,
    )
    if last_activity_subq is not None:
        rows_stmt = rows_stmt.join(
            last_activity_subq,
            last_activity_subq.c.opportunity_key
            == func.upper(cast(Opportunity.opportunityid, String)),
            isouter=True,
        )
    if risk_count_subq is not None:
        rows_stmt = rows_stmt.join(
            risk_count_subq,
            risk_count_subq.c.opportunity_key
            == func.upper(cast(Opportunity.opportunityid, String)),
            isouter=True,
        )

    rows_stmt = rows_stmt.where(base_filter)
    rows_stmt = apply_opportunity_filters(
        rows_stmt, account_already_joined=True, **filter_kwargs
    )

    if not has_deal_health:
        rows_stmt = rows_stmt.options(
            defer(Opportunity.lvo_dealhealthscore, raiseload=True),
            defer(Opportunity.lvo_riskscore, raiseload=True),
            defer(Opportunity.lvo_riskreason, raiseload=True),
        )
    if not has_deal_detail:
        rows_stmt = rows_stmt.options(
            defer(Opportunity.lvo_tempoclass, raiseload=True),
            defer(Opportunity.lvo_stageentrydate, raiseload=True),
            defer(Opportunity.lvo_createdat, raiseload=True),
            defer(Opportunity.lvo_dealhealthupdatedat, raiseload=True),
        )

    primary_order, tiebreak_order = _opportunity_sort_clause(sort_by, sort_order)
    rows_stmt = (
        rows_stmt.order_by(primary_order, tiebreak_order)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows = db.execute(rows_stmt).all()

    opportunity_ids_upper = [str(r.Opportunity.opportunityid).upper() for r in rows]

    # ---- Competitor preview (batched) -----------------------------------
    competitor_map: dict[str, list[str]] = defaultdict(list)
    competitor_counts: dict[str, int] = defaultdict(int)
    if opportunity_ids_upper:
        comp_rows = db.execute(
            select(
                OpportunityCompetitor.lvo_opportunityid,
                OpportunityCompetitor.lvo_competitorname,
            )
            .where(
                func.upper(OpportunityCompetitor.lvo_opportunityid).in_(
                    opportunity_ids_upper
                ),
                OpportunityCompetitor.statecode == "Active",
            )
            .order_by(
                OpportunityCompetitor.lvo_opportunityid,
                OpportunityCompetitor.lvo_competitorname,
            )
        ).all()
        for opp_id, comp_name in comp_rows:
            key = (opp_id or "").upper()
            competitor_counts[key] += 1
            if len(competitor_map[key]) < 3 and comp_name:
                competitor_map[key].append(comp_name)

    # ---- Activity timeline (batched, bucketed) --------------------------
    activity_map: dict[str, list[ActivityItem]] = defaultdict(list)
    if has_activity_table and opportunity_ids_upper:
        activity_window_start = datetime.now(timezone.utc) - timedelta(
            days=timeline_days
        )
        # SQLAlchemy + psycopg2 happily compares TIMESTAMP WITH TIME ZONE
        # against a tz-aware Python datetime, but the column is declared
        # as plain DateTime in the ORM — strip tzinfo to be safe across
        # both naive and aware database columns.
        window_naive = activity_window_start.replace(tzinfo=None)
        per_opp_buffer: dict[str, list[Activity]] = defaultdict(list)
        act_rows = (
            db.execute(
                select(Activity)
                .where(
                    func.upper(Activity.lvo_opportunityid).in_(opportunity_ids_upper),
                    Activity.statecode == "Active",
                    Activity.lvo_activitydate >= window_naive,
                )
                .order_by(
                    func.upper(Activity.lvo_opportunityid),
                    desc(Activity.lvo_activitydate),
                )
            )
            .scalars()
            .all()
        )
        for a in act_rows:
            key = (a.lvo_opportunityid or "").upper()
            per_opp_buffer[key].append(a)
        for key, events in per_opp_buffer.items():
            activity_map[key] = _bucket_activities_per_opportunity(events)

    # ---- Assemble OpportunityListItem rows ------------------------------
    items: list[OpportunityListItem] = []
    for row in rows:
        opp = row.Opportunity
        opp_key = str(opp.opportunityid).upper()

        last_activity = (
            row.last_activity if last_activity_subq is not None else None
        )
        risk_count_val = (
            int(row.risk_count or 0) if risk_count_subq is not None else None
        )

        items.append(
            OpportunityListItem(
                id=opp.opportunityid,
                name=opp.name,
                account_id=opp.accountid,
                account_name=row.account_name,
                industry=row.industry,
                country=opp.lvo_country,
                region=opp.lvo_businessgroup,
                stage=StageRef(
                    raw=opp.stagename,
                    label=normalise_stage(opp.stagename),
                ),
                sale_motion=SaleMotionRef(
                    raw=opp.lvo_salesmotion,
                    label=normalise_sale_motion(opp.lvo_salesmotion),
                ),
                forecast_category=opp.lvo_forecastcategory,
                value=(
                    float(opp.estimatedvalue)
                    if opp.estimatedvalue is not None
                    else None
                ),
                currency="USD",
                close_date=opp.estimatedclosedate,
                close_probability=(
                    float(opp.closeprobability)
                    if opp.closeprobability is not None
                    else None
                ),
                competitor_count=competitor_counts.get(opp_key, 0),
                competitors=competitor_map.get(opp_key, []),
                owner_id=opp.owninguser,
                statecode=opp.statecode,
                risk=opp.lvo_riskreason if has_deal_health else None,
                risk_score=(
                    normalise_risk_score(opp.lvo_riskscore)
                    if has_deal_health
                    else None
                ),
                risk_count=risk_count_val,
                deal_health=opp.lvo_dealhealthscore if has_deal_health else None,
                last_activity=last_activity,
                activities=activity_map.get(opp_key, []),
            )
        )

    return total, items


@router.get(
    "/{account_id}/opportunities",
    response_model=AccountOpportunitiesResponse,
    summary="Opportunities linked to this account, with filters + timeline + risk count",
)
def list_account_opportunities(
    account_id: str = Path(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, alias="pageSize"),
    include_canceled: bool = Query(
        False,
        alias="includeCanceled",
        description="Include soft-deleted deals (statecode='Canceled').",
    ),
    search: str | None = Query(
        None,
        description=(
            "ILIKE on opportunity.name OR account.name. "
            "Backs the 'Search Opportunity' box in the toolbar."
        ),
    ),
    regions: str | None = Query(
        None,
        description=(
            "Comma-separated list. Matches opportunity.lvo_businessgroup "
            "or lvo_country (case-insensitive). Mirrors the global "
            "/api/filters/regions vocabulary."
        ),
    ),
    industries: str | None = Query(
        None, description="Comma-separated list of account.industrycode."
    ),
    stages: str | None = Query(
        None, description="Comma-separated list of opportunity.stagename."
    ),
    products: str | None = Query(
        None,
        description=(
            "Comma-separated list. Matches via EXISTS over "
            "quote + lvo_quoteitem.lvo_productseries; accepts slugs or "
            "raw labels."
        ),
    ),
    sort_by: OpportunitySort = Query("closeDate", alias="sortBy"),
    sort_order: SortOrder = Query("desc", alias="sortOrder"),
    timeline_days: int = Query(
        _DEFAULT_TIMELINE_DAYS,
        alias="timelineDays",
        ge=7,
        le=365,
        description=(
            "Activity-timeline window length in days. The UI shows tick "
            "marks at 90/60/30/0 by default; bump this to e.g. 180 to "
            "render a longer strip without changing the FE rendering."
        ),
    ),
    db: Session = Depends(get_db),
) -> AccountOpportunitiesResponse:
    """List deals belonging to the account using the same shape as the main grid."""
    _ensure_account(db, account_id)

    total, items = _resolve_account_opportunities(
        db,
        account_id=account_id,
        page=page,
        page_size=page_size,
        include_canceled=include_canceled,
        search=search,
        regions=_normalise_csv(regions),
        industries=_normalise_csv(industries),
        stages=_normalise_csv(stages),
        products=_normalise_csv(products),
        sort_by=sort_by,
        sort_order=sort_order,
        timeline_days=timeline_days,
    )

    total_pages = math.ceil(total / page_size) if page_size else 0

    return AccountOpportunitiesResponse(
        account_id=account_id,
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        items=items,
    )


# ---------------------------------------------------------------------------
# GET /api/accounts/{id}/opportunities/export — CSV stream of the same set
# ---------------------------------------------------------------------------


_OPP_EXPORT_COLUMNS: list[tuple[str, str]] = [
    ("name", "Name"),
    ("account_name", "Account"),
    ("stage_label", "Stage"),
    ("sale_motion_label", "Sale Motion"),
    ("forecast_category", "Forecast"),
    ("value", "Value"),
    ("currency", "Currency"),
    ("close_date", "Close Date"),
    ("close_probability", "Probability"),
    ("deal_health", "Deal Health"),
    ("risk", "Risk Reason"),
    ("risk_count", "Risk Count"),
    ("competitor_count", "Competitors"),
    ("last_activity", "Last Activity"),
    ("statecode", "Statecode"),
]


def _format_opp_export_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _stream_account_opp_csv(
    items: list[OpportunityListItem],
) -> Iterator[str]:
    """Yield CSV chunks for the per-account Opportunities export."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([label for _, label in _OPP_EXPORT_COLUMNS])
    yield buffer.getvalue()
    buffer.seek(0)
    buffer.truncate()

    for item in items:
        # Flatten the StageRef / SaleMotionRef wrappers into plain labels
        # so the CSV reads cleanly.
        flat = {
            "name": item.name,
            "account_name": item.account_name,
            "stage_label": item.stage.label or item.stage.raw,
            "sale_motion_label": item.sale_motion.label or item.sale_motion.raw,
            "forecast_category": item.forecast_category,
            "value": item.value,
            "currency": item.currency,
            "close_date": item.close_date,
            "close_probability": item.close_probability,
            "deal_health": item.deal_health,
            "risk": item.risk,
            "risk_count": item.risk_count,
            "competitor_count": item.competitor_count,
            "last_activity": item.last_activity,
            "statecode": item.statecode,
        }
        writer.writerow(
            [_format_opp_export_value(flat[field]) for field, _ in _OPP_EXPORT_COLUMNS]
        )
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate()


@router.get(
    "/{account_id}/opportunities/export",
    summary="CSV export of the per-account Opportunities tab",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/csv": {}},
            "description": "CSV stream of opportunities matching the filter set.",
        }
    },
)
def export_account_opportunities(
    account_id: str = Path(...),
    include_canceled: bool = Query(False, alias="includeCanceled"),
    search: str | None = Query(None),
    regions: str | None = Query(None),
    industries: str | None = Query(None),
    stages: str | None = Query(None),
    products: str | None = Query(None),
    sort_by: OpportunitySort = Query("closeDate", alias="sortBy"),
    sort_order: SortOrder = Query("desc", alias="sortOrder"),
    timeline_days: int = Query(
        _DEFAULT_TIMELINE_DAYS, alias="timelineDays", ge=7, le=365
    ),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Export the full filtered set as CSV, no pagination cap.

    Re-runs the same resolver the list endpoint uses, paging server-side
    until every match is collected. ``MAX_PAGE_SIZE`` is the page chunk —
    not a cap on the export — so the download contains every match.
    """
    _ensure_account(db, account_id)

    filters = {
        "include_canceled": include_canceled,
        "search": search,
        "regions": _normalise_csv(regions),
        "industries": _normalise_csv(industries),
        "stages": _normalise_csv(stages),
        "products": _normalise_csv(products),
        "sort_by": sort_by,
        "sort_order": sort_order,
        "timeline_days": timeline_days,
    }

    all_items: list[OpportunityListItem] = []
    page = 1
    while True:
        total, items = _resolve_account_opportunities(
            db,
            account_id=account_id,
            page=page,
            page_size=MAX_PAGE_SIZE,
            **filters,
        )
        all_items.extend(items)
        # ``items`` shorter than the page size means we've drained the result.
        if len(items) < MAX_PAGE_SIZE or len(all_items) >= total:
            break
        page += 1

    filename = (
        f"account_{account_id}_opportunities_"
        f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    return StreamingResponse(
        _stream_account_opp_csv(all_items),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# POST /api/accounts/{id}/recompute-status — force a refresh
# ---------------------------------------------------------------------------


@router.post(
    "/{account_id}/recompute-status",
    response_model=AccountRecomputeResponse,
    summary="Force-recompute account-type / status / lastInteraction",
)
def force_recompute_account_status(
    background_tasks: BackgroundTasks,
    account_id: str = Path(..., description="account.accountid (UUID)"),
    db: Session = Depends(get_db),
) -> AccountRecomputeResponse:
    """Manually trigger a write-back of the derived account fields.

    Useful from the detail page or from ops scripts. The deal-write paths
    already enqueue this asynchronously after every PATCH/DELETE that could
    flip the account's status.
    """
    _ensure_account(db, account_id)
    result = recalculate_account(db, account_id, write=True)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recompute failed.",
        )
    # Schedule a follow-up async pass too — useful when the caller is an admin
    # and wants the next page-load to reflect any racy writes that landed
    # while the synchronous recompute was running.
    background_tasks.add_task(recalculate_async, account_id)
    return AccountRecomputeResponse(
        id=result.account_id,
        account_type=result.account_type,
        status=result.status,
        last_interaction=result.last_interaction,
    )


# ---------------------------------------------------------------------------
# GET /api/accounts/{id}/customer-information — sectioned read-only profile
#
# Phase 1 of the "View Customer Information" user story. Returns six cards
# (Basic Info, Billing/Shipping Address, Identity & Legal, Commercial Terms,
# Territory & Ownership) using a mix of standard D365 columns and the
# ``lvo_*`` columns added by sql/2026_06_account_customer_info_schema.sql.
# Columns absent on the live DB are deferred and read as ``None``.
# ---------------------------------------------------------------------------


# Standard D365 paymenttermscode option-set. Values come from CRM 2011+ and
# haven't changed since. We surface the label so the FE doesn't have to ship
# its own lookup; an unrecognised code falls back to "Code <n>" so the user
# at least sees the raw value rather than an empty cell.
_PAYMENT_TERMS_LABELS: dict[int, str] = {
    1: "Net 30",
    2: "2% 10, Net 30",
    3: "Net 45",
    4: "Net 60",
}


def _label_payment_terms(code: int | None) -> str | None:
    if code is None:
        return None
    return _PAYMENT_TERMS_LABELS.get(int(code), f"Code {int(code)}")


def _resolve_lookup_label(
    db: Session,
    table_name: str,
    pk_column: str,
    name_columns: tuple[str, ...],
    pk_value: str | None,
) -> str | None:
    """Best-effort lookup-id → display-name resolver.

    Used for ``territoryid`` (→ territory.name) and
    ``defaultpricelevelid`` (→ pricelevel.name). When the target table or
    columns aren't on the dump we return ``None`` and the caller falls
    back to the raw UUID.
    """
    if not pk_value:
        return None
    inspector = inspect(db.bind)
    if not inspector.has_table(table_name):
        return None
    cols = {c["name"] for c in inspector.get_columns(table_name)}
    if pk_column not in cols:
        return None
    name_col = next((c for c in name_columns if c in cols), None)
    if name_col is None:
        return None
    sql = (
        f"SELECT {name_col} AS label "
        f"FROM {table_name} "
        f"WHERE UPPER({pk_column}::TEXT) = :pk "
        "LIMIT 1"
    )
    row = db.execute(text(sql), {"pk": str(pk_value).upper()}).first()
    return (row.label if row else None) or None


def _attr(account: Account, name: str, present: frozenset[str]) -> Any:
    """Read ``account.<name>`` only when the column exists on the live DB.

    Belt-and-braces — the ORM's ``defer(... raiseload=False)`` already
    gives us ``None`` for missing columns, but Postgres still raises if we
    SELECT a column that genuinely doesn't exist. The router uses defer +
    raiseload=True for the heavy GET, so this fallback is just defensive
    sugar for code paths that bypass the eager-load.
    """
    if name not in present:
        return None
    return getattr(account, name, None)


@router.get(
    "/{account_id}/customer-information",
    response_model=CustomerInformationResponse,
    summary="Sectioned read-only Customer Information payload",
    responses={404: {"description": "Account not found"}},
)
def get_account_customer_information(
    account_id: str = Path(..., description="account.accountid (UUID)"),
    db: Session = Depends(get_db),
) -> CustomerInformationResponse:
    """Return a sectioned payload for the Customer Information tab.

    Six cards, every field optional. Rows missing from the dump or from a
    partially-applied migration come back as ``null`` rather than 500.
    """
    present_cols = get_account_columns(db)

    # ---- The Customer-Information columns are declared ``deferred=True``
    # ---- on the Account ORM (see app/models.py). That keeps every other
    # ---- ``SELECT Account`` lean and \\N-safe on stripped D365 dumps.
    # ---- For THIS endpoint we ``undefer()`` only the columns we know are
    # ---- physically present, pulling them into the same SELECT (no
    # ---- per-attribute lazy load round-trips).
    undeferred_attrs = []
    candidate_columns = (*STANDARD_D365_COLUMNS, *LENOVO_CUSTOM_COLUMNS)
    for col in candidate_columns:
        if col not in present_cols:
            continue
        attr = getattr(Account, col, None)
        if attr is not None:
            undeferred_attrs.append(undefer(attr))

    stmt = select(Account).where(
        func.upper(cast(Account.accountid, String)) == account_id.upper()
    )
    if undeferred_attrs:
        stmt = stmt.options(*undeferred_attrs)
    account: Account | None = db.execute(stmt).scalar_one_or_none()
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account '{account_id}' not found.",
        )

    has_view_cols = _has_account_view_columns(db)

    # ---- Owner (Assigned Owner) and Record Owner (createdby) name lookup -----
    owner_uuids: list[str] = []
    if account.owninguser:
        owner_uuids.append(account.owninguser)
    record_owner_id = _attr(account, "createdby", present_cols)
    if record_owner_id and record_owner_id != account.owninguser:
        owner_uuids.append(record_owner_id)
    seller_names = _resolve_seller_names(db, owner_uuids) if owner_uuids else {}

    # ---- Territory + Price-list lookup labels (best-effort) ------------------
    territory_id = _attr(account, "territoryid", present_cols)
    sales_territory_label = (
        account.lvo_salesterritory
        if "lvo_salesterritory" in present_cols and account.lvo_salesterritory
        else _resolve_lookup_label(
            db, "territory", "territoryid", ("name",), territory_id
        )
        or territory_id
    )

    price_level_id = _attr(account, "defaultpricelevelid", present_cols)
    price_list_label = (
        _resolve_lookup_label(
            db, "pricelevel", "pricelevelid", ("name",), price_level_id
        )
        or price_level_id
    )

    # ---- Build sections -------------------------------------------------------
    basic = CustomerInfoBasicInformation(
        account_id=str(account.accountid),
        account_name=account.name,
        account_type=(account.lvo_accounttype if has_view_cols else None),
        segment=account.lvo_segment,
        sub_segment=_attr(account, "lvo_subsegment", present_cols),
        industry_segment=account.industrycode,
        gtm_segment=_attr(account, "lvo_gtmsegment", present_cols),
        annual_revenue=(
            float(account.revenue) if account.revenue is not None else None
        ),
        employee_count=account.numberofemployees,
        seller_known_as=_attr(account, "lvo_sellerknownas", present_cols),
    )

    billing = CustomerInfoAddress(
        line1=_attr(account, "address1_line1", present_cols),
        line2=_attr(account, "address1_line2", present_cols),
        city=_attr(account, "address1_city", present_cols),
        state_province=_attr(account, "address1_stateorprovince", present_cols),
        postal_code=_attr(account, "address1_postalcode", present_cols),
        country=_attr(account, "address1_country", present_cols),
    )

    shipping = CustomerInfoAddress(
        line1=_attr(account, "address2_line1", present_cols),
        line2=_attr(account, "address2_line2", present_cols),
        city=_attr(account, "address2_city", present_cols),
        state_province=_attr(account, "address2_stateorprovince", present_cols),
        postal_code=_attr(account, "address2_postalcode", present_cols),
        country=_attr(account, "address2_country", present_cols),
    )

    identity = CustomerInfoIdentityLegal(
        legal_name_local=_attr(account, "lvo_legalnamelocal", present_cols),
        local_language=_attr(account, "lvo_locallanguage", present_cols),
        alias=_attr(account, "lvo_alias", present_cols),
        tax_vat_number=_attr(account, "lvo_taxvatnumber", present_cols),
        legal_entity=_attr(account, "lvo_legalentity", present_cols),
        main_phone=_attr(account, "telephone1", present_cols),
        website=_attr(account, "websiteurl", present_cols),
        linkedin_url=_attr(account, "lvo_linkedinurl", present_cols),
    )

    commercial = CustomerInfoCommercialTerms(
        default_currency=account.lvo_defaultcurrency,
        payment_terms=_label_payment_terms(
            _attr(account, "paymenttermscode", present_cols)
        ),
        price_list=price_list_label,
        deal_sign_config=_attr(account, "lvo_dealsignconfig", present_cols),
    )

    owning_user = account.owninguser
    territory = CustomerInfoTerritoryOwnership(
        region=_pick_region_label(account),
        sales_territory=sales_territory_label,
        future_territory=_attr(account, "lvo_futureterritory", present_cols),
        sales_org=_attr(account, "lvo_salesorg", present_cols),
        territory_move_reason=_attr(
            account, "lvo_territorymovereason", present_cols
        ),
        geographic_unit=_attr(account, "lvo_geographicunit", present_cols),
        sales_office=_attr(account, "lvo_salesoffice", present_cols),
        assigned_owner_id=owning_user,
        assigned_owner_name=(
            seller_names.get((owning_user or "").upper())
            if owning_user
            else None
        ),
        record_owner_id=record_owner_id,
        record_owner_name=(
            seller_names.get((record_owner_id or "").upper())
            if record_owner_id
            else None
        ),
    )

    # ---- Diagnostics ----------------------------------------------------------
    notes: list[str] = []
    missing_lvo = [c for c in LENOVO_CUSTOM_COLUMNS if c not in present_cols]
    if missing_lvo:
        notes.append(
            "Lenovo-custom columns missing — run "
            "sql/2026_06_account_customer_info_schema.sql to populate "
            f"({len(missing_lvo)} column(s) absent: "
            f"{', '.join(missing_lvo[:5])}{'…' if len(missing_lvo) > 5 else ''})."
        )
    missing_d365 = [c for c in STANDARD_D365_COLUMNS if c not in present_cols]
    if missing_d365:
        notes.append(
            "Standard D365 columns absent on this dump "
            f"({len(missing_d365)}): the affected fields read as null."
        )

    return CustomerInformationResponse(
        id=str(account.accountid),
        basic_information=basic,
        billing_address=billing,
        shipping_address=shipping,
        identity_and_legal=identity,
        commercial_terms=commercial,
        territory_and_ownership=territory,
        notes=notes,
    )

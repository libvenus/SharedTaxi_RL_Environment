"""Opportunity endpoints used by the Opportunities UI.

Endpoints
---------
- GET /api/opportunities                                  (#2)
- GET /api/opportunities/kpi-summary                      (#1)
- GET /api/opportunities/{opportunity_id}/competitors     (#5)
- GET /api/opportunities/{opportunity_id}/sale-motion     (#13)
"""

import math
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import Numeric, String, asc, case, cast, desc, func, inspect, select, text
from sqlalchemy.orm import Session, defer

from app.database import get_db
from app.filters_query import apply_opportunity_filters
from app.models import Account, Activity, DealRisk, Opportunity, OpportunityCompetitor
from app.normalizers import normalise_risk_score, normalise_sale_motion, normalise_stage
from app.schemas import (
    ActivityItem,
    ComparePeriod,
    Competitor,
    CompetitorList,
    CreateOpportunityRequest,
    KpiBucket,
    KpiCard,
    KpiSummaryResponse,
    OpportunityListItem,
    OpportunityListResponse,
    OpportunityRef,
    OpportunitySearchResponse,
    OpportunitySort,
    SaleMotionRef,
    SaleMotionResponse,
    SortOrder,
    StageRef,
    TrendInfo,
)
from app.services.kpi_snapshots import (
    BucketAggregate,
    PERIOD_LOOKBACK_DAYS,
    compute_trend_info,
    lookup_previous_for_period,
)

from app.schemas import CompetitorsResponse,CompetitorsRequest
from app.services.opportunity import get_opportunity_competitors, save_competitors

router = APIRouter(prefix="/api/opportunities", tags=["opportunities"])


# ---------------------------------------------------------------------------
# Forecast / stage / state vocabulary used in aggregates
# ---------------------------------------------------------------------------
WON_STATECODES = ("Won", "Closed Won")
LOST_STATECODES = ("Lost", "Closed Lost")
WON_STAGES = ("Closed Won",)
LOST_STAGES = ("Closed Lost",)



def _bucket_predicate(bucket: KpiBucket):
    """Single source of truth for the WHERE clause behind each KPI card.

    Used by /kpi-summary to build per-bucket aggregates and by the main
    /opportunities list when the user clicks a card to filter the grid.
    """
    if bucket == "open":
        return Opportunity.statecode == "Open"
    if bucket == "pipeline":
        # UI label is "Identified"; bucket id and predicate stay "pipeline"
        # so existing snapshots, drill-down URLs and tests keep working.
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
# Shared helpers
# ---------------------------------------------------------------------------
def _ensure_opportunity_exists(db: Session, opportunity_id: str) -> Opportunity:
    opp = db.get(Opportunity, opportunity_id)
    if opp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Opportunity '{opportunity_id}' not found",
        )
    return opp


def _normalise_csv(raw: str | None) -> list[str] | None:
    """Turn a comma-separated query-string value into a clean list."""
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None

@router.post("/competitors")
def create_or_update_competitors(
    request: CompetitorsRequest,
    db: Session = Depends(get_db)
):
    return save_competitors(db, request)
# ---------------------------------------------------------------------------
# #1 — KPI summary
# ---------------------------------------------------------------------------
@router.get(
    "/kpi-summary",
    response_model=KpiSummaryResponse,
    summary="KPI cards strip — Open Deals / Pipeline / Best Case / Commit / Won / Loss",
)
def kpi_summary(
    compare_period: ComparePeriod = Query(
        "last_week",
        alias="comparePeriod",
        description="Period to compare against. Trend will be null until snapshots exist.",
    ),
    search: str | None = Query(None, description="Free-text search across name / account."),
    regions: str | None = Query(None, description="Comma-separated business groups or countries."),
    industries: str | None = Query(None, description="Comma-separated industry codes."),
    stages: str | None = Query(None, description="Comma-separated raw stagename values."),
    products: str | None = Query(None, description="Comma-separated product series."),
    db: Session = Depends(get_db),
) -> KpiSummaryResponse:
    filters = {
        "search": search,
        "regions": _normalise_csv(regions),
        "industries": _normalise_csv(industries),
        "stages": _normalise_csv(stages),
        "products": _normalise_csv(products),
    }

    value_col = cast(Opportunity.estimatedvalue, Numeric)

    def bucket(label: str, *predicates) -> tuple:
        """Return (sum, count) aggregates filtered by the given predicates."""
        sum_expr = func.coalesce(func.sum(value_col).filter(*predicates), 0).label(
            f"{label}_value"
        )
        count_expr = func.count().filter(*predicates).label(f"{label}_count")
        return sum_expr, count_expr

    open_v, open_c = bucket("open", _bucket_predicate("open"))
    pipe_v, pipe_c = bucket("pipe", _bucket_predicate("pipeline"))
    best_v, best_c = bucket("best", _bucket_predicate("best_case"))
    commit_v, commit_c = bucket("commit", _bucket_predicate("commit"))
    ml_v, ml_c = bucket("ml", _bucket_predicate("most_likely"))
    won_v, won_c = bucket("won", _bucket_predicate("won"))
    loss_v, loss_c = bucket("loss", _bucket_predicate("loss"))

    stmt = select(
        open_v, open_c,
        pipe_v, pipe_c,
        best_v, best_c,
        commit_v, commit_c,
        ml_v, ml_c,
        won_v, won_c,
        loss_v, loss_c,
    ).select_from(Opportunity)
    stmt = apply_opportunity_filters(stmt, **filters)

    row = db.execute(stmt).one()

    # ---- Trend wiring ------------------------------------------------------
    # v1 strategy: trend is only emitted on the unfiltered view, since the
    # snapshot table only stores global (non-filtered) aggregates. Per-
    # dimension snapshots are a planned follow-up.
    has_filters = any(filters.values())

    # Build the live "current" aggregates per bucket so we can pair them
    # with historical rows the same way the snapshot service does.
    live_buckets: dict[str, BucketAggregate] = {
        "open": BucketAggregate("open", float(row.open_value or 0), int(row.open_count or 0)),
        "pipeline": BucketAggregate("pipeline", float(row.pipe_value or 0), int(row.pipe_count or 0)),
        "best_case": BucketAggregate("best_case", float(row.best_value or 0), int(row.best_count or 0)),
        "commit": BucketAggregate("commit", float(row.commit_value or 0), int(row.commit_count or 0)),
        "most_likely": BucketAggregate("most_likely", float(row.ml_value or 0), int(row.ml_count or 0)),
        "won": BucketAggregate("won", float(row.won_value or 0), int(row.won_count or 0)),
        "loss": BucketAggregate("loss", float(row.loss_value or 0), int(row.loss_count or 0)),
    }

    previous_buckets: dict[str, BucketAggregate] = {}
    if not has_filters:
        previous_buckets = lookup_previous_for_period(db, period=compare_period)

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

    def card(bucket: str) -> KpiCard:
        live = live_buckets[bucket]
        return KpiCard(value=live.value, count=live.count, trend=_trend_for(bucket))

    notes: list[str] = [
        "Pipeline bucket relies on opportunity.lvo_forecastcategory='Pipeline'; "
        "the sample data does not contain that label, so the bucket may read 0.",
    ]
    if has_filters:
        notes.insert(
            0,
            "Trend is suppressed because filters are applied — v1 snapshots "
            "are global only. Re-issue without filters to get period-over-period change.",
        )
    elif not previous_buckets:
        lookback_days = PERIOD_LOOKBACK_DAYS.get(compare_period)
        notes.insert(
            0,
            (
                "Trend is null because no lvo_opportunitysnapshot row exists "
                f"on or before today minus {lookback_days} days. Run "
                "`python -m app.jobs.snapshot_kpis --backfill` once, then "
                "schedule `python -m app.jobs.snapshot_kpis` nightly."
            ),
        )

    return KpiSummaryResponse(
        compare_period=compare_period,
        as_of=datetime.now(timezone.utc),
        currency="USD",
        open_deals=card("open"),
        pipeline=card("pipeline"),
        best_case=card("best_case"),
        commit=card("commit"),
        most_likely=card("most_likely"),
        won=card("won"),
        loss=card("loss"),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# GET /api/opportunities/search — typeahead for the Parent / Child Opportunity
# pickers on the Complete-Information form.
#
# Routing note: declared *before* the bare ``GET /{opportunity_id}`` route in
# ``deals_read.py`` so FastAPI's literal-path-wins resolution sends ``/search``
# here every time. (Even if ordering changed, FastAPI prefers literal segments
# over path-parameters, so this would still work — but defining it on the
# router that doesn't own a bare ``/{id}`` keeps the intent obvious.)
# ---------------------------------------------------------------------------

_SEARCH_DEFAULT_LIMIT = 20
_SEARCH_MAX_LIMIT = 50
from fastapi import HTTPException
import uuid


def create_opportunity(
    payload: CreateOpportunityRequest,
    db: Session
):
    try:
        data = payload.model_dump(
            exclude_none=True,
            exclude_unset=True
        )

        if not data.get("opportunityid"):
            data["opportunityid"] = str(uuid.uuid4())

        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())

        query = text(f"""
            INSERT INTO public.opportunity
            ({columns})
            VALUES
            ({placeholders})
            RETURNING opportunityid
        """)

        result = db.execute(query, data)
        opportunity_id = result.scalar()

        db.commit()

        return {
            "success": True,
            "message": "Opportunity created successfully",
            "opportunity_id": str(opportunity_id)
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get(
    "/competitors",
    response_model=CompetitorsResponse
)
def get_competitors(
    opportunity_id: str = Query(...),
    db: Session = Depends(get_db)
):
    return get_opportunity_competitors(
        db=db,
        opportunity_id=opportunity_id
    )

@router.post("/opportunity_create")
def create_opportunity_api(
    payload: CreateOpportunityRequest,
    db: Session = Depends(get_db)
):
    return create_opportunity(payload, db)

from sqlalchemy import text
from fastapi import HTTPException


def update_opportunity(
    opportunity_id: str,
    payload: CreateOpportunityRequest,
    db: Session
):
    try:
        data = payload.model_dump(
            exclude_none=True,
            exclude_unset=True
        )

        # Prevent updating primary key
        data.pop("opportunityid", None)

        if not data:
            raise HTTPException(
                status_code=400,
                detail="No fields provided for update"
            )

        set_clause = ", ".join(
            [f"{column} = :{column}" for column in data.keys()]
        )

        data["opportunity_id"] = opportunity_id

        query = text(f"""
            UPDATE public.opportunity
            SET {set_clause}
            WHERE opportunityid = :opportunity_id
            RETURNING opportunityid
        """)

        result = db.execute(query, data)
        updated_id = result.scalar()

        if not updated_id:
            db.rollback()
            raise HTTPException(
                status_code=404,
                detail=f"Opportunity {opportunity_id} not found"
            )

        db.commit()

        return {
            "success": True,
            "message": "Opportunity updated successfully",
            "opportunity_id": updated_id
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@router.patch("/opportunity_update/{opportunity_id}")
def update_opportunity_api(
    opportunity_id: str,
    payload: CreateOpportunityRequest,
    db: Session = Depends(get_db)
):
    return update_opportunity(
        opportunity_id=opportunity_id,
        payload=payload,
        db=db
    )

@router.get(
    "/search",
    response_model=OpportunitySearchResponse,
    summary="Typeahead search for the Parent / Child Opportunity pickers",
)
def search_opportunities(
    q: str = Query(
        ...,
        min_length=1,
        max_length=200,
        description="Substring to ILIKE against opportunity.name (case-insensitive).",
    ),
    exclude_id: str | None = Query(
        None,
        alias="excludeId",
        description=(
            "Optional opportunityid to exclude from the results — typically "
            "the deal currently being edited so the picker can't pick itself."
        ),
    ),
    limit: int = Query(
        _SEARCH_DEFAULT_LIMIT,
        ge=1,
        le=_SEARCH_MAX_LIMIT,
        description=(
            f"Max items returned. Default {_SEARCH_DEFAULT_LIMIT}, max "
            f"{_SEARCH_MAX_LIMIT} — keeps the picker snappy."
        ),
    ),
    db: Session = Depends(get_db),
) -> OpportunitySearchResponse:
    """Lightweight typeahead — returns ``{id, name}`` only.

    Ordering rules (most relevant first):
      1. Names that *start with* the needle (prefix match)
      2. Names that contain the needle anywhere
      3. Alphabetical tiebreaker

    Cancelled deals are filtered out so the FE never sees a dead reference
    in the picker. Cycle prevention happens server-side at PATCH time
    (``INVALID_PARENT_OPPORTUNITY``); the picker may still surface a
    descendant of the current deal but selecting it will fail validation.
    """
    needle = q.strip()
    if not needle:
        return OpportunitySearchResponse(query=q, total=0, items=[])

    substr_pattern = f"%{needle}%"
    prefix_pattern = f"{needle}%"

    # Use an explicit relevance score so the SQL planner can use the
    # ix_opportunity_name_lower index for both the WHERE filter and the
    # ORDER BY (Postgres only — gracefully degrades on other engines).
    prefix_match = case(
        (Opportunity.name.ilike(prefix_pattern), 0),
        else_=1,
    )

    stmt = (
        select(Opportunity.opportunityid, Opportunity.name)
        .where(
            Opportunity.name.is_not(None),
            Opportunity.name.ilike(substr_pattern),
            func.coalesce(Opportunity.statecode, "") != "Canceled",
        )
        .order_by(prefix_match.asc(), func.lower(Opportunity.name).asc())
        .limit(limit)
    )

    if exclude_id:
        stmt = stmt.where(
            func.upper(cast(Opportunity.opportunityid, String))
            != str(exclude_id).upper()
        )

    rows = db.execute(stmt).all()
    items = [OpportunityRef(id=str(r.opportunityid), name=r.name) for r in rows]
    return OpportunitySearchResponse(query=needle, total=len(items), items=items)


# ---------------------------------------------------------------------------
# #2 — List opportunities (main grid)
# ---------------------------------------------------------------------------
_SORT_COLUMN_MAP = {
    "name": Opportunity.name,
    "value": Opportunity.estimatedvalue,
    "closeDate": Opportunity.estimatedclosedate,
    "closeProbability": Opportunity.closeprobability,
    "stage": Opportunity.stagename,
}


@router.get(
    "",
    response_model=OpportunityListResponse,
    summary="Paginated list of opportunities for the main grid",
)
def list_opportunities(
    page: int = Query(1, ge=1, description="1-based page number."),
    page_size: int = Query(
        10,
        ge=1,
        le=100,
        alias="pageSize",
        description="Number of items per page (1-100).",
    ),
    search: str | None = Query(None, description="Free-text search across name / account."),
    regions: str | None = Query(None, description="Comma-separated business groups or countries."),
    industries: str | None = Query(None, description="Comma-separated industry codes."),
    stages: str | None = Query(None, description="Comma-separated raw stagename values."),
    products: str | None = Query(None, description="Comma-separated product series."),
    bucket: KpiBucket | None = Query(
        None,
        description=(
            "Restrict the grid to one KPI-card bucket "
            "(open / pipeline / best_case / commit / won / loss). "
            "Mirrors the predicate used by /kpi-summary."
        ),
    ),
    sort_by: OpportunitySort = Query("closeDate", alias="sortBy"),
    sort_order: SortOrder = Query("desc", alias="sortOrder"),
    db: Session = Depends(get_db),
) -> OpportunityListResponse:
    filters = {
        "search": search,
        "regions": _normalise_csv(regions),
        "industries": _normalise_csv(industries),
        "stages": _normalise_csv(stages),
        "products": _normalise_csv(products),
    }
    bucket_predicate = _bucket_predicate(bucket) if bucket else None

    sort_col = _SORT_COLUMN_MAP[sort_by]
    order_fn = asc if sort_order == "asc" else desc

    # -- Total count -------------------------------------------------------
    count_stmt = select(func.count(Opportunity.opportunityid.distinct()))
    count_stmt = apply_opportunity_filters(count_stmt, **filters)
    if bucket_predicate is not None:
        count_stmt = count_stmt.where(bucket_predicate)
    total: int = db.execute(count_stmt).scalar_one()

    # If sql/2026_06_create_lvo_activity.sql has not been applied yet, fall
    # back to "no activity data" rather than 500ing the whole endpoint.
    inspector = inspect(db.bind)
    has_activity_table = inspector.has_table("lvo_activity")
    has_dealrisk_table = inspector.has_table("lvo_dealrisk")

    # sql/2026_06_add_dealhealth.sql adds three columns to `opportunity`.
    # sql/2026_06_deal_detail_schema.sql adds four more.
    # If a migration hasn't been applied yet, defer the missing columns so
    # the existing grid keeps working on a partially-migrated DB.
    opportunity_columns = {c["name"] for c in inspector.get_columns("opportunity")}
    has_deal_health = {
        "lvo_dealhealthscore",
        "lvo_riskscore",
        "lvo_riskreason",
    }.issubset(opportunity_columns)
    has_deal_detail_columns = {
        "lvo_tempoclass",
        "lvo_stageentrydate",
        "lvo_createdat",
        "lvo_dealhealthupdatedat",
    }.issubset(opportunity_columns)

    if has_activity_table:
        # Most recent activity timestamp per opportunity — drives the
        # "Last Activity" column and orders the timeline dots.
        # Compare on UPPER() because the TEXT FK can be mixed case in the data.
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

    select_cols: list = [
        Opportunity,
        Account.name.label("account_name"),
        Account.industrycode.label("industry"),
    ]
    if last_activity_subq is not None:
        select_cols.append(last_activity_subq.c.last_activity.label("last_activity"))
    if risk_count_subq is not None:
        select_cols.append(risk_count_subq.c.risk_count.label("risk_count"))

    # -- Page query --------------------------------------------------------
    rows_stmt = select(*select_cols).join(
        Account,
        # account.accountid is UUID; opportunity.accountid is TEXT — must cast.
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
    # Pass account_already_joined=True so the helper doesn't add a second join.
    rows_stmt = apply_opportunity_filters(
        rows_stmt, **filters, account_already_joined=True
    )
    if bucket_predicate is not None:
        rows_stmt = rows_stmt.where(bucket_predicate)
    if not has_deal_health:
        # raiseload=True surfaces a clear error if someone later tries to
        # read the missing attribute, instead of silently issuing a lazy
        # query that would also fail with UndefinedColumn.
        rows_stmt = rows_stmt.options(
            defer(Opportunity.lvo_dealhealthscore, raiseload=True),
            defer(Opportunity.lvo_riskscore, raiseload=True),
            defer(Opportunity.lvo_riskreason, raiseload=True),
        )
    if not has_deal_detail_columns:
        # Same pattern for the Deal Detailed View columns — keeps the
        # main grid endpoint working on a DB that hasn't yet had
        # sql/2026_06_deal_detail_schema.sql applied.
        rows_stmt = rows_stmt.options(
            defer(Opportunity.lvo_tempoclass, raiseload=True),
            defer(Opportunity.lvo_stageentrydate, raiseload=True),
            defer(Opportunity.lvo_createdat, raiseload=True),
            defer(Opportunity.lvo_dealhealthupdatedat, raiseload=True),
        )
    rows_stmt = (
        rows_stmt.order_by(
            order_fn(sort_col),
            asc(Opportunity.opportunityid),
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows = db.execute(rows_stmt).all()

    # opportunity.opportunityid is UUID in the DB; lvo_opportunitycompetitor.
    # lvo_opportunityid is TEXT — and the sample data stores UPPERCASE UUIDs
    # while Python's str(uuid.UUID(...)) is lowercase. Compare case-insensitive.
    opportunity_ids_upper = [str(r.Opportunity.opportunityid).upper() for r in rows]

    # -- Competitor preview (batched, avoids N+1) --------------------------
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

    # -- Activities preview (batched, avoids N+1) --------------------------
    # We surface up to 5 most recent activities per opportunity. They drive
    # the timeline dots in the grid and the offcanvas detail panel.
    ACTIVITIES_PER_OPP_LIMIT = 5
    activity_map: dict[str, list[ActivityItem]] = defaultdict(list)
    if has_activity_table and opportunity_ids_upper:
        act_rows = (
            db.execute(
                select(Activity)
                .where(
                    func.upper(Activity.lvo_opportunityid).in_(opportunity_ids_upper),
                    Activity.statecode == "Active",
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
            if len(activity_map[key]) >= ACTIVITIES_PER_OPP_LIMIT:
                continue
            activity_map[key].append(
                ActivityItem(
                    id=a.lvo_activityid,
                    type=a.lvo_activitytype,
                    direction=a.lvo_direction,
                    subject=a.lvo_subject,
                    body=a.lvo_body,
                    activity_date=a.lvo_activitydate,
                    grouped_count=a.lvo_groupedcount,
                )
            )

    items = [
        OpportunityListItem(
            id=row.Opportunity.opportunityid,
            name=row.Opportunity.name,
            account_id=row.Opportunity.accountid,
            account_name=row.account_name,
            industry=row.industry,
            country=row.Opportunity.lvo_country,
            region=row.Opportunity.lvo_businessgroup,
            stage=StageRef(
                raw=row.Opportunity.stagename,
                label=normalise_stage(row.Opportunity.stagename),
            ),
            sale_motion=SaleMotionRef(
                raw=row.Opportunity.lvo_salesmotion,
                label=normalise_sale_motion(row.Opportunity.lvo_salesmotion),
            ),
            forecast_category=row.Opportunity.lvo_forecastcategory,
            value=float(row.Opportunity.estimatedvalue)
            if row.Opportunity.estimatedvalue is not None
            else None,
            currency="USD",
            close_date=row.Opportunity.estimatedclosedate,
            close_probability=float(row.Opportunity.closeprobability)
            if row.Opportunity.closeprobability is not None
            else None,
            competitor_count=competitor_counts.get(
                str(row.Opportunity.opportunityid).upper(), 0
            ),
            competitors=competitor_map.get(
                str(row.Opportunity.opportunityid).upper(), []
            ),
            owner_id=row.Opportunity.owninguser,
            statecode=row.Opportunity.statecode,
            risk=row.Opportunity.lvo_riskreason if has_deal_health else None,
            risk_score=(
                normalise_risk_score(row.Opportunity.lvo_riskscore)
                if has_deal_health
                else None
            ),
            risk_count=(
                int(row.risk_count or 0)
                if risk_count_subq is not None and hasattr(row, "risk_count")
                else None
            ),
            deal_health=row.Opportunity.lvo_dealhealthscore if has_deal_health else None,
            last_activity=row.last_activity if has_activity_table else None,
            activities=activity_map.get(
                str(row.Opportunity.opportunityid).upper(), []
            ),
        )
        for row in rows
    ]

    total_pages = math.ceil(total / page_size) if page_size else 0

    return OpportunityListResponse(
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        sort_by=sort_by,
        sort_order=sort_order,
        items=items,
        notes=[
            "nextAction is still a placeholder — needs lvo_nextaction* columns.",
            (
                "lastActivity and activities[] come from lvo_activity "
                "(seeded via sql/2026_06_create_lvo_activity.sql)."
                if has_activity_table
                else "lvo_activity table not found — run sql/2026_06_create_lvo_activity.sql "
                "to populate lastActivity and the timeline."
            ),
            (
                "risk, riskScore, dealHealth come from opportunity columns "
                "(seeded via sql/2026_06_add_dealhealth.sql)."
                if has_deal_health
                else "opportunity.lvo_dealhealthscore not found — run "
                "sql/2026_06_add_dealhealth.sql to populate risk/dealHealth."
            ),
        ],
    )


# ---------------------------------------------------------------------------
# #5 — Competitors per opportunity
# ---------------------------------------------------------------------------
@router.get(
    "/{opportunity_id}/competitors",
    response_model=CompetitorList,
    summary="Competitors associated with an opportunity",
)
def list_competitors(
    opportunity_id: str = Path(..., description="opportunity.opportunityid (UUID)"),
    db: Session = Depends(get_db),
) -> CompetitorList:
    _ensure_opportunity_exists(db, opportunity_id)

    # lvo_opportunityid is TEXT and the sample data stores UPPERCASE UUIDs,
    # so compare case-insensitively to tolerate either casing in the URL.
    rows = (
        db.execute(
            select(OpportunityCompetitor)
            .where(
                func.upper(OpportunityCompetitor.lvo_opportunityid)
                == opportunity_id.upper(),
                OpportunityCompetitor.statecode == "Active",
            )
            .order_by(OpportunityCompetitor.lvo_competitorname)
        )
        .scalars()
        .all()
    )

    items = [
        Competitor(
            id=r.lvo_opportunitycompetitorid,
            opportunity_id=r.lvo_opportunityid or opportunity_id,
            name=r.lvo_name,
            competitor_name=r.lvo_competitorname,
            competitor_type=r.lvo_competitortype,
            reselling_partner_id=r.lvo_resellingpartner,
        )
        for r in rows
    ]

    return CompetitorList(
        opportunity_id=opportunity_id,
        total=len(items),
        items=items,
    )


# ---------------------------------------------------------------------------
# #13 — Sale motion per opportunity (Net new / Expansion / Renewal)
# ---------------------------------------------------------------------------
@router.get(
    "/{opportunity_id}/sale-motion",
    response_model=SaleMotionResponse,
    summary="Sale motion pill value (Net new / Expansion / Renewal)",
)
def get_sale_motion(
    opportunity_id: str = Path(..., description="opportunity.opportunityid (UUID)"),
    db: Session = Depends(get_db),
) -> SaleMotionResponse:
    opp = _ensure_opportunity_exists(db, opportunity_id)
    return SaleMotionResponse(
        opportunity_id=opportunity_id,
        raw=opp.lvo_salesmotion,
        label=normalise_sale_motion(opp.lvo_salesmotion),
    )

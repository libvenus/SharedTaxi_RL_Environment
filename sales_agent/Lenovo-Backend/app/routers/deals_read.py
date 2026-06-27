"""Read endpoints for the Deal Detailed View user story.

Endpoints
---------
GET  /api/opportunities/{id}                       Full deal-detail payload.
GET  /api/opportunities/{id}/timeline              Paginated chronological events.
GET  /api/opportunities/{id}/contacts              Decision maker + additional contacts.
GET  /api/opportunities/{id}/health                Live health breakdown.
GET  /api/opportunities/{id}/risks                 Active risks.
POST /api/opportunities/{id}/health/recalculate    Force a synchronous recalc.

The detail endpoint hydrates the entire screen in one round-trip; the other
endpoints exist for partial refreshes (e.g. paginated timeline) and ops use
(forced recalc).
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Path, Query, status
from sqlalchemy import String, cast, desc, func, inspect, or_, select, text
from sqlalchemy.orm import Session, undefer

from app.database import get_db
from app.models import (
    Account,
    Activity,
    AuditLog,
    Contact,
    DealRisk,
    NextAction,
    Opportunity,
    OpportunityCompetitor,
    OpportunityContact,
)
from app.normalizers import (
    normalise_deal_priority,
    normalise_sale_motion,
    normalise_stage,
)
from app.schemas import (
    AccountSummary,
    ActivityItem,
    Competitor,
    ContactListResponse,
    ContactRef,
    DealHealthInfo,
    HealthComponent,
    NextActionItem,
    OpportunityDetail,
    OpportunityRef,
    RecalculateHealthResponse,
    RiskInfo,
    RiskListResponse,
    SaleMotionRef,
    StageRef,
    TimelineEvent,
    TimelineResponse,
)
from app.services.account_recalc import recalculate_async as recalculate_account_async
from app.services.contact_phone import bulk_read_phones
from app.services.deal_recalc import RecalculationResult, recalculate_deal_health

router = APIRouter(prefix="/api/opportunities", tags=["opportunities-detail"])


# ---------------------------------------------------------------------------
# Constants & shared helpers
# ---------------------------------------------------------------------------

CLOSED_STATES = {"won", "closed won", "lost", "closed lost"}
CLOSED_STAGES = {"Closed Won", "Closed Lost"}
CANCELED_STATE = "Canceled"

STAGE_LOCK_STAGES = {"Qualify"}

# Cap how many activities are inlined into OpportunityDetail. Anything
# beyond this lives behind /timeline (paginated). 10 matches the design
# of the offcanvas panel.
DETAIL_ACTIVITY_PREVIEW_LIMIT = 10

# Default page size for the detailed timeline. The grid offcanvas only
# shows the latest 5; the dedicated /timeline page wants a fuller window.
TIMELINE_DEFAULT_PAGE_SIZE = 25
TIMELINE_MAX_PAGE_SIZE = 100


def _ensure_opportunity(db: Session, opportunity_id: str) -> Opportunity:
    opp = db.get(Opportunity, opportunity_id)
    if opp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Opportunity '{opportunity_id}' not found.",
        )
    return opp


def _is_closed(opp: Opportunity) -> bool:
    return (opp.statecode or "").lower() in CLOSED_STATES or (
        opp.stagename or ""
    ) in CLOSED_STAGES


def _is_canceled(opp: Opportunity) -> bool:
    return (opp.statecode or "") == CANCELED_STATE


def _is_stage_locked(opp: Opportunity) -> bool:
    return (opp.stagename or "") in STAGE_LOCK_STAGES or _is_closed(opp) or _is_canceled(opp)


def _has_table(db: Session, table_name: str) -> bool:
    """Defensive — lets the API survive when an optional migration is missing."""
    return inspect(db.bind).has_table(table_name)


def _has_column(db: Session, table_name: str, column_name: str) -> bool:
    """Runtime introspection helper — used by paths that read columns added
    by an optional migration (e.g. ``opportunity.modifiedon``).

    Returning ``False`` short-circuits the SELECT so we never try to read
    a non-existent column from a stripped dump.
    """
    inspector = inspect(db.bind)
    if not inspector.has_table(table_name):
        return False
    return any(c["name"] == column_name for c in inspector.get_columns(table_name))


# ---------------------------------------------------------------------------
# Owner-name resolution — used by Overview / Complete-Information form
# ---------------------------------------------------------------------------


def _resolve_owner_name(db: Session, owner_id: str | None) -> str | None:
    """Resolve ``opportunity.owninguser`` (UUID) to a human-readable name.

    Mirrors the helper in ``app/routers/accounts.py`` — same defensive
    introspection so we don't 500 on dumps that don't ship ``systemuser``.
    Returns ``None`` only when the lookup is genuinely impossible
    (no table, no matching row, or no name column); the caller treats
    ``None`` as "fall back to UUID".
    """
    if not owner_id:
        return None
    inspector = inspect(db.bind)
    if not inspector.has_table("systemuser"):
        return None
    cols = {c["name"] for c in inspector.get_columns("systemuser")}
    if "systemuserid" not in cols:
        return None
    name_col = (
        "fullname" if "fullname" in cols
        else "internalemailaddress" if "internalemailaddress" in cols
        else None
    )
    if name_col is None:
        return None

    sql = (
        f"SELECT {name_col} AS label "
        "FROM systemuser "
        "WHERE UPPER(systemuserid::TEXT) = :uid "
        "LIMIT 1"
    )
    row = db.execute(text(sql), {"uid": owner_id.upper()}).first()
    return row.label if row and row.label else None


# ---------------------------------------------------------------------------
# Days-in-stage derivation — used by Complete-Information form
# ---------------------------------------------------------------------------


def _compute_days_in_stage(stage_entry_date: datetime | None) -> int | None:
    """Pure function — whole days between ``stage_entry_date`` and today (UTC).

    Negative results (a stage_entry_date in the future, e.g. data drift)
    are clamped to 0 so the FE never has to render a negative number.
    Returns ``None`` only when ``stage_entry_date`` itself is ``None``.
    """
    if stage_entry_date is None:
        return None
    today = datetime.now(timezone.utc)
    # Normalise both to naive-UTC for the subtraction so a timezone-aware
    # vs. naive ``stage_entry_date`` (varies by dump) doesn't raise.
    if stage_entry_date.tzinfo is None:
        anchor = today.replace(tzinfo=None)
    else:
        anchor = today
    delta = anchor - stage_entry_date
    return max(0, delta.days)


# ---------------------------------------------------------------------------
# Parent / Child Opportunity resolution — Complete-Information form pickers
# ---------------------------------------------------------------------------


def _load_parent_ref(db: Session, parent_id: str | None) -> OpportunityRef | None:
    """Return the ``{id, name}`` pair for the parent deal so the FE can render
    the chip without an extra round-trip.

    Returns ``None`` when:
      * ``parent_id`` itself is ``None``;
      * the parent row was cancelled (``statecode == 'Canceled'``); or
      * the parent has been hard-deleted from the DB (broken reference).

    Case-insensitive comparison — the caller may have stored the FK in any
    case, while ``cast(uuid, String)`` always returns canonical lowercase.
    """
    if not parent_id:
        return None
    row = db.execute(
        select(Opportunity.opportunityid, Opportunity.name)
        .where(
            func.upper(cast(Opportunity.opportunityid, String))
            == str(parent_id).upper(),
            func.coalesce(Opportunity.statecode, "") != CANCELED_STATE,
        )
        .limit(1)
    ).first()
    if row is None:
        return None
    return OpportunityRef(id=str(row.opportunityid), name=row.name)


def _load_children(db: Session, opportunity_id: str) -> list[OpportunityRef]:
    """Return every non-cancelled deal whose parent points back to this one.

    The result is sorted by name (case-insensitive) so the picker renders
    deterministically in screenshots and tests.
    """
    rows = db.execute(
        select(Opportunity.opportunityid, Opportunity.name)
        .where(
            func.upper(Opportunity.lvo_parentopportunityid) == opportunity_id.upper(),
            func.coalesce(Opportunity.statecode, "") != CANCELED_STATE,
        )
        .order_by(func.lower(func.coalesce(Opportunity.name, "")))
    ).all()
    return [
        OpportunityRef(id=str(r.opportunityid), name=r.name)
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Account summary (used inline in detail; also exposed via /api/accounts)
# ---------------------------------------------------------------------------


def build_account_summary(db: Session, account_id: str) -> AccountSummary:
    """Hydrate AccountSummary including the on-the-fly rollups.

    Rollups are computed in two cheap aggregate queries rather than via
    eager-loaded relationships so the page can scale to accounts with many
    deals. Excludes Canceled deals from both totals (they were soft-deleted).
    """
    account = db.execute(
        select(Account).where(
            func.upper(cast(Account.accountid, String)) == account_id.upper()
        )
    ).scalar_one_or_none()

    if account is None:
        return AccountSummary(
            id=account_id,
            name=None,
            total_account_value=0.0,
            open_deals_count=0,
        )

    # Total Account Value — sum of estimatedvalue across non-Canceled deals.
    total_value = (
        db.execute(
            select(
                func.coalesce(func.sum(Opportunity.estimatedvalue), 0)
            ).where(
                func.upper(Opportunity.accountid) == account_id.upper(),
                func.coalesce(Opportunity.statecode, "") != CANCELED_STATE,
            )
        ).scalar_one()
        or 0
    )

    # Open Deals Count.
    open_count = (
        db.execute(
            select(func.count()).where(
                func.upper(Opportunity.accountid) == account_id.upper(),
                Opportunity.statecode == "Open",
            )
        ).scalar_one()
        or 0
    )

    return AccountSummary(
        id=str(account.accountid),
        name=account.name,
        segment=account.lvo_segment,
        industry=account.industrycode,
        territory=account.lvo_territory,
        employee_count=account.numberofemployees,
        total_account_value=float(total_value),
        open_deals_count=int(open_count),
        business_group=account.lvo_businessgroupid,
        country=account.lvo_countryid,
    )


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


def _build_contact_refs(
    db: Session, opportunity_id: str
) -> tuple[ContactRef | None, list[ContactRef]]:
    """Return (decision_maker, additional_contacts).

    LEFT-joins lvo_opportunitycontact to contact so a link with a missing
    contact row (broken FK in the seed) still appears with placeholder name.

    Phone numbers are batched in a single query via
    ``app.services.contact_phone.bulk_read_phones`` — the resolver picks
    whichever of telephone1 / mobilephone / lvo_phone exists in this dump,
    so we never SELECT a column that isn't there.
    """
    if not _has_table(db, "lvo_opportunitycontact"):
        return None, []

    rows = db.execute(
        select(OpportunityContact, Contact)
        .join(
            Contact,
            func.upper(cast(Contact.contactid, String))
            == func.upper(OpportunityContact.lvo_contactid),
            isouter=True,
        )
        .where(
            func.upper(OpportunityContact.lvo_opportunityid)
            == opportunity_id.upper(),
            OpportunityContact.statecode == "Active",
        )
        .order_by(
            desc(OpportunityContact.lvo_isdecisionmaker),
            OpportunityContact.lvo_role,
            OpportunityContact.lvo_createdat,
        )
    ).all()

    contact_ids = [link.lvo_contactid for link, _ in rows if link.lvo_contactid]
    phones = bulk_read_phones(db, contact_ids) if contact_ids else {}

    decision_maker: ContactRef | None = None
    additional: list[ContactRef] = []
    for link, contact in rows:
        first_name = contact.firstname if contact else None
        last_name = contact.lastname if contact else None
        ref = ContactRef(
            id=link.lvo_opportunitycontactid,
            contact_id=link.lvo_contactid,
            name=(contact.fullname if contact else None)
            or (
                f"{(first_name or '').strip()} {(last_name or '').strip()}".strip()
                if contact
                else None
            ),
            first_name=first_name,
            last_name=last_name,
            role=link.lvo_role,
            is_decision_maker=bool(link.lvo_isdecisionmaker),
            last_touch_date=link.lvo_lasttouchdate,
            job_title=contact.jobtitle if contact else None,
            email=contact.emailaddress1 if contact else None,
            phone=phones.get((link.lvo_contactid or "").upper()),
        )
        if link.lvo_isdecisionmaker and decision_maker is None:
            decision_maker = ref
        else:
            additional.append(ref)
    return decision_maker, additional


# ---------------------------------------------------------------------------
# Activities preview (inlined into detail) and full timeline
# ---------------------------------------------------------------------------


def _load_activity_preview(
    db: Session, opportunity_id: str, limit: int = DETAIL_ACTIVITY_PREVIEW_LIMIT
) -> list[ActivityItem]:
    if not _has_table(db, "lvo_activity"):
        return []
    rows = (
        db.execute(
            select(Activity)
            .where(
                func.upper(Activity.lvo_opportunityid) == opportunity_id.upper(),
                Activity.statecode == "Active",
            )
            .order_by(desc(Activity.lvo_activitydate))
            .limit(limit)
        )
        .scalars()
        .all()
    )
    return [
        ActivityItem(
            id=a.lvo_activityid,
            type=a.lvo_activitytype,
            direction=a.lvo_direction,
            subject=a.lvo_subject,
            body=a.lvo_body,
            activity_date=a.lvo_activitydate,
            grouped_count=a.lvo_groupedcount,
        )
        for a in rows
    ]


# ---------------------------------------------------------------------------
# Competitors / next-actions snapshots (already exposed via their own
# endpoints; we re-use the loading logic here so the detail payload is
# self-contained and doesn't force the FE to make 4 extra calls).
# ---------------------------------------------------------------------------


def _load_competitors(db: Session, opportunity_id: str) -> list[Competitor]:
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
    return [
        Competitor(
            id=c.lvo_opportunitycompetitorid,
            opportunity_id=c.lvo_opportunityid or opportunity_id,
            name=c.lvo_name,
            competitor_name=c.lvo_competitorname,
            competitor_type=c.lvo_competitortype,
            reselling_partner_id=c.lvo_resellingpartner,
        )
        for c in rows
    ]


def _load_next_actions(db: Session, opportunity_id: str) -> list[NextActionItem]:
    if not _has_table(db, "lvo_nextaction"):
        return []
    rows = (
        db.execute(
            select(NextAction)
            .where(
                func.upper(NextAction.lvo_opportunityid) == opportunity_id.upper(),
                NextAction.statecode == "Active",
            )
            .order_by(desc(NextAction.lvo_createdat))
        )
        .scalars()
        .all()
    )
    return [
        NextActionItem(
            id=a.lvo_nextactionid,
            opportunity_id=a.lvo_opportunityid,
            description=a.lvo_description,
            due_date=a.lvo_duedate,
            status=a.lvo_status,
            verbal_commit_date=a.verbal_commit_date,
            verbal_written_acceptance=a.verbal_written_acceptance,
            created_at=a.lvo_createdat,
            updated_at=a.lvo_updatedat,
            created_by=a.lvo_createdby,
        )
        for a in rows
    ]


# ---------------------------------------------------------------------------
# Health + Risks (live, no DB write — uses the recalc orchestrator with
# write=False so reads never mutate state).
# ---------------------------------------------------------------------------


def _to_health_info(result: RecalculationResult) -> DealHealthInfo:
    components = {
        key: HealthComponent(
            weight=int(c["weight"]),
            score=float(c["score"]),
            inputs=c.get("inputs", {}),
        )
        for key, c in result.breakdown.components.items()
    }
    return DealHealthInfo(
        score=int(result.breakdown.score),
        band=result.breakdown.band,
        updated_at=result.updated_at,
        components=components,
    )


def _to_risk_infos(result: RecalculationResult) -> list[RiskInfo]:
    infos: list[RiskInfo] = []
    persisted_ids = list(result.persisted_risk_ids) if result.persisted else []
    for idx, r in enumerate(result.risks):
        infos.append(
            RiskInfo(
                id=persisted_ids[idx] if idx < len(persisted_ids) else None,
                category=r.category,
                name=r.name,
                message=r.message,
                detected_at=result.updated_at if result.persisted else None,
            )
        )
    return infos


def _load_persisted_risks(db: Session, opportunity_id: str) -> list[RiskInfo]:
    """Return the most recent persisted risks for read-only endpoints."""
    if not _has_table(db, "lvo_dealrisk"):
        return []
    rows = (
        db.execute(
            select(DealRisk)
            .where(
                func.upper(DealRisk.lvo_opportunityid) == opportunity_id.upper(),
                DealRisk.statecode == "Active",
            )
            .order_by(DealRisk.lvo_detectedat.desc())
        )
        .scalars()
        .all()
    )
    return [
        RiskInfo(
            id=r.lvo_dealriskid,
            category=r.lvo_riskcategory,
            name=r.lvo_riskname,
            message=r.lvo_message,
            detected_at=r.lvo_detectedat,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# GET /api/opportunities/{id}  — full detail
# ---------------------------------------------------------------------------


@router.get(
    "/{opportunity_id}",
    response_model=OpportunityDetail,
    summary="Full Deal Detail payload",
    responses={404: {"description": "Opportunity not found"}},
)
def get_opportunity_detail(
    opportunity_id: str = Path(..., description="opportunity.opportunityid (UUID)"),
    db: Session = Depends(get_db),
) -> OpportunityDetail:
    """Hydrate the entire Deal Detail page in one round-trip.

    Combines:
        * Opportunity row (stage, value, motion, dates, ownership)
        * Account panel with Total Account Value + Open Deals Count rollups
        * Decision maker + additional contacts
        * Most recent 10 activities (full history at /timeline)
        * Competitors and Next Actions (also exposed via their own endpoints)
        * Live deal-health breakdown + active risks (no write).
        * Complete-Information (Deal Summary) form fields — summary,
          priority, lead origin, partner-involved toggle, parent/child
          opportunity refs, days-in-stage, owner display name, and
          ``createdby`` / ``modifiedon`` / ``modifiedby`` audit columns.

    The audit columns are deferred-by-default on the ORM (see ``app/models.py``)
    so we explicitly ``undefer()`` them here. If a stripped dump genuinely
    lacks the columns, the runtime check skips the undefer — the response
    falls back to ``None`` for those fields rather than 500'ing the page.
    """
    # ---- Load the deal -----------------------------------------------------
    # Undefer audit columns only when they actually ship in the dump. Without
    # this guard, ``undefer()`` against a missing column would raise on the
    # SELECT. ``_has_column`` is cheap (cached metadata).
    audit_columns_present = (
        _has_column(db, "opportunity", "createdby")
        and _has_column(db, "opportunity", "modifiedon")
        and _has_column(db, "opportunity", "modifiedby")
    )
    stmt = select(Opportunity).where(
        func.upper(cast(Opportunity.opportunityid, String))
        == str(opportunity_id).upper()
    )
    if audit_columns_present:
        stmt = stmt.options(
            undefer(Opportunity.createdby),
            undefer(Opportunity.modifiedon),
            undefer(Opportunity.modifiedby),
        )
    opp = db.execute(stmt).scalar_one_or_none()
    if opp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Opportunity '{opportunity_id}' not found.",
        )

    account = build_account_summary(db, opp.accountid) if opp.accountid else AccountSummary(
        id="",
        name=None,
        total_account_value=0.0,
        open_deals_count=0,
    )
    decision_maker, additional_contacts = _build_contact_refs(db, opportunity_id)
    competitors = _load_competitors(db, opportunity_id)
    next_actions = _load_next_actions(db, opportunity_id)
    activities_preview = _load_activity_preview(db, opportunity_id)

    # Live evaluation — persist so grid (stored score + risk count) stays aligned.
    result = recalculate_deal_health(db, opportunity_id, write=True)
    if result is None:
        # _ensure_opportunity already raised 404; defensive guard.
        raise HTTPException(status_code=500, detail="Failed to compute deal health.")
    health = _to_health_info(result)
    # Use the risks we just evaluated (and persisted when write=True). Reloading
    # lvo_dealrisk here can return stale rows when the session hasn't flushed yet
    # or when an older partial persist left fewer rows than the live rules fire.
    risks = _to_risk_infos(result)

    # ---- Complete-Information extras --------------------------------------
    owner_name = _resolve_owner_name(db, opp.owninguser)
    parent_ref = _load_parent_ref(db, opp.lvo_parentopportunityid)
    children = _load_children(db, opportunity_id)
    days_in_stage = _compute_days_in_stage(opp.lvo_stageentrydate)

    # Audit columns only — read directly when present, else None.
    created_by = opp.createdby if audit_columns_present else None
    modified_at = opp.modifiedon if audit_columns_present else None
    modified_by = opp.modifiedby if audit_columns_present else None

    return OpportunityDetail(
        id=str(opp.opportunityid),
        name=opp.name,
        account_id=opp.accountid,
        stage=StageRef(raw=opp.stagename, label=normalise_stage(opp.stagename)),
        sale_motion=SaleMotionRef(
            raw=opp.lvo_salesmotion,
            label=normalise_sale_motion(opp.lvo_salesmotion),
        ),
        forecast_category=opp.lvo_forecastcategory,
        value=float(opp.estimatedvalue) if opp.estimatedvalue is not None else None,
        currency="USD",
        close_date=opp.estimatedclosedate,
        close_probability=(
            float(opp.closeprobability) if opp.closeprobability is not None else None
        ),
        owner_id=opp.owninguser,
        owner_name=owner_name,
        statecode=opp.statecode,
        tempo_class=opp.lvo_tempoclass,
        created_at=opp.lvo_createdat,
        stage_entry_date=opp.lvo_stageentrydate,
        is_closed=_is_closed(opp),
        is_canceled=_is_canceled(opp),
        is_stage_locked=_is_stage_locked(opp),
        # ---- Complete-Information form fields ------------------------------
        summary=opp.lvo_summary,
        priority=normalise_deal_priority(opp.lvo_priority),
        lead_origin=opp.lvo_leadorigin,
        partner_involved=bool(opp.lvo_partnerinvolved),
        parent_opportunity_id=opp.lvo_parentopportunityid,
        parent_opportunity_name=parent_ref.name if parent_ref else None,
        child_opportunities=children,
        days_in_stage=days_in_stage,
        created_by=created_by,
        modified_at=modified_at,
        modified_by=modified_by,
        # ---- Composite panels ---------------------------------------------
        account=account,
        decision_maker=decision_maker,
        additional_contacts=additional_contacts,
        competitors=competitors,
        next_actions=next_actions,
        activities_preview=activities_preview,
        health=health,
        risks=risks,
        #---Newly added fields --#
       actual_revenue=opp.actual_revenue ,
        actual_close_date=opp.actual_close_date ,
        close_reason=opp.close_reason ,
        sales_order_reference=opp.sales_order_reference ,
        won_solution_category=opp.won_solution_category ,
        win_notes_commentary=opp.win_notes_commentary , 
        invoice_number=opp.invoice_number ,
        loss_reason=opp.loss_reason ,
        competitor_won=opp.competitor_won ,
        lost_solution_category=opp.lost_solution_category ,
        lost_revenue_value=opp.lost_revenue_value ,
        loss_notes_commentary=opp.loss_notes_commentary ,
        deal_appeal=opp.deal_appeal ,
        re_engagement_date=opp.re_engagement_date ,
        solution_area=opp.solution_area ,
        sub_solution_area=opp.sub_solution_area ,
        solution_certifications=opp.solution_certifications ,
        solution_offerings=opp.solution_offerings ,
        leasing_vendor=opp.leasing_vendor ,
        sales_model=opp.sales_model ,
        service_model=opp.service_model ,

        budget_confirmed=opp.budget_confirmed ,

        quote_reference=opp.quote_reference ,
        partner_commercial_model=opp.partner_commercial_model ,
        actual_confirmed_revenue=opp.actual_confirmed_revenue ,
        reseller_channel_account=opp.reseller_channel_account ,

        deal_protection_status=opp.deal_protection_status ,
        deal_registration_ref=opp.deal_registration_ref ,
        number_of_countries=opp.number_of_countries ,



 sow_required=opp.sow_required ,
 multi_country_solution_required=opp.multi_country_solution_required ,
 deal_qualification_review=opp.deal_qualification_review ,
 solution_handover_artefacts=opp.solution_handover_artefacts ,
 solution_service_executive=opp.solution_service_executive ,
 solution_service_domain_specialist=opp.solution_service_domain_specialist ,
 lgfs_sales_representatives=opp.lgfs_sales_representatives ,
 lgfs_sales_support=opp.lgfs_sales_support ,
 deal_desk_analyst=opp.deal_desk_analyst ,
 deal_engagement_manager=opp.deal_engagement_manager ,
 ssds_channel=opp.ssds_channel ,
 sell_through_week_auto=opp.sell_through_week_auto ,
 competitor_type =opp.competitor_type ,

 order_date =opp.order_date ,
 shipping_date =opp.shipping_date ,
 sales_order_reference_po =opp.sales_order_reference_po ,
 created_date =opp.created_date ,
 order_number =opp.order_number ,


    )


# ---------------------------------------------------------------------------
# GET /api/opportunities/{id}/timeline  — paginated chronological events
# ---------------------------------------------------------------------------


@router.get(
    "/{opportunity_id}/timeline",
    response_model=TimelineResponse,
    summary="Paginated chronological timeline (activities + CRM field changes)",
)
def get_opportunity_timeline(
    opportunity_id: str = Path(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(
        TIMELINE_DEFAULT_PAGE_SIZE,
        ge=1,
        le=TIMELINE_MAX_PAGE_SIZE,
        alias="pageSize",
    ),
    types: str | None = Query(
        None,
        description=(
            "Comma-separated activity types to include "
            "(email, meeting, crm, multiple, crm_change). "
            "Omit to include all."
        ),
    ),
    db: Session = Depends(get_db),
) -> TimelineResponse:
    """Return a fully-merged chronological feed of activities + CRM changes.

    * `lvo_activity` rows surface as ``source='activity'`` with their original
      type.  These are the customer-facing touchpoints.
    * `lvo_audit_log` opportunity-update rows surface as
      ``source='crm_change'`` with type=``<changed_field>`` so the user can
      audit who changed forecast / value / close-date and when.

    Pagination is server-side because the merged result-set can be large
    on long-running deals.
    """
    _ensure_opportunity(db, opportunity_id)

    requested_types: set[str] | None = None
    if types:
        requested_types = {t.strip() for t in types.split(",") if t.strip()}

    events: list[TimelineEvent] = []

    # --- Activities ---------------------------------------------------------
    if _has_table(db, "lvo_activity") and (
        requested_types is None
        or any(t in {"email", "meeting", "crm", "multiple"} for t in requested_types)
    ):
        activity_filter_types = (
            requested_types & {"email", "meeting", "crm", "multiple"}
            if requested_types
            else None
        )
        stmt = select(Activity).where(
            func.upper(Activity.lvo_opportunityid) == opportunity_id.upper(),
            Activity.statecode == "Active",
        )
        if activity_filter_types:
            stmt = stmt.where(Activity.lvo_activitytype.in_(activity_filter_types))
        for a in db.execute(stmt).scalars().all():
            events.append(
                TimelineEvent(
                    id=a.lvo_activityid,
                    source="activity",
                    type=a.lvo_activitytype,
                    direction=a.lvo_direction,
                    subject=a.lvo_subject,
                    body=a.lvo_body,
                    event_date=a.lvo_activitydate,
                    grouped_count=a.lvo_groupedcount,
                    changed_by=None,
                )
            )

    # --- Audit-log derived CRM changes -------------------------------------
    if _has_table(db, "lvo_audit_log") and (
        requested_types is None or "crm_change" in requested_types
    ):
        audit_rows = (
            db.execute(
                select(AuditLog).where(
                    AuditLog.lvo_entitytype == "opportunity",
                    AuditLog.lvo_action == "update",
                    or_(
                        func.upper(AuditLog.lvo_opportunityid) == opportunity_id.upper(),
                        func.upper(AuditLog.lvo_entityid) == opportunity_id.upper(),
                    ),
                )
            )
            .scalars()
            .all()
        )
        for log in audit_rows:
            for changed_field, body in _explode_audit_diff(log.lvo_diff):
                events.append(
                    TimelineEvent(
                        id=f"{log.lvo_auditlogid}:{changed_field}",
                        source="crm_change",
                        type=changed_field,
                        direction=None,
                        subject=f"{changed_field} changed",
                        body=body,
                        event_date=log.lvo_changedat,
                        grouped_count=None,
                        changed_by=log.lvo_changedby,
                    )
                )

    # --- Sort + paginate ----------------------------------------------------
    events.sort(key=lambda e: e.event_date, reverse=True)
    total = len(events)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = events[start:end]
    total_pages = math.ceil(total / page_size) if page_size else 0

    return TimelineResponse(
        opportunity_id=opportunity_id,
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        items=page_items,
    )


def _explode_audit_diff(diff_json: str | None) -> list[tuple[str, str]]:
    """Turn an opportunity audit-log diff into per-field timeline entries.

    Returns a list of (field_name, "before -> after") tuples. The top-level
    structure of a diff written by deals_write.py is::

        {"before": {...snapshot...}, "after": {...snapshot...}}

    Anything else (including legacy / malformed diffs) is dropped.
    """
    if not diff_json:
        return []
    try:
        diff = json.loads(diff_json)
    except (TypeError, ValueError):
        return []
    if not isinstance(diff, dict):
        return []
    before = diff.get("before") or {}
    after = diff.get("after") or {}
    if not isinstance(before, dict) or not isinstance(after, dict):
        return []
    out: list[tuple[str, str]] = []
    for key in sorted(set(before.keys()) | set(after.keys())):
        b, a = before.get(key), after.get(key)
        if b == a:
            continue
        out.append((key, f"{b!r} → {a!r}"))
    return out


# ---------------------------------------------------------------------------
# GET /api/opportunities/{id}/contacts
# ---------------------------------------------------------------------------


@router.get(
    "/{opportunity_id}/contacts",
    response_model=ContactListResponse,
    summary="Decision maker + additional contacts on a deal",
)
def get_opportunity_contacts(
    opportunity_id: str = Path(...),
    db: Session = Depends(get_db),
) -> ContactListResponse:
    _ensure_opportunity(db, opportunity_id)
    decision_maker, additional = _build_contact_refs(db, opportunity_id)
    total = (1 if decision_maker else 0) + len(additional)
    return ContactListResponse(
        opportunity_id=opportunity_id,
        decision_maker=decision_maker,
        additional_contacts=additional,
        total=total,
    )


# ---------------------------------------------------------------------------
# GET /api/opportunities/{id}/health
# ---------------------------------------------------------------------------


@router.get(
    "/{opportunity_id}/health",
    response_model=DealHealthInfo,
    summary="Live deal-health breakdown (no DB write)",
)
def get_opportunity_health(
    opportunity_id: str = Path(...),
    db: Session = Depends(get_db),
) -> DealHealthInfo:
    _ensure_opportunity(db, opportunity_id)
    result = recalculate_deal_health(db, opportunity_id, write=False)
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to compute deal health.")
    return _to_health_info(result)


# ---------------------------------------------------------------------------
# POST /api/opportunities/{id}/health/recalculate  — synchronous force-recalc
# ---------------------------------------------------------------------------


@router.post(
    "/{opportunity_id}/health/recalculate",
    response_model=RecalculateHealthResponse,
    summary="Force a synchronous recalculation (admin / debug use)",
)
def recalculate_opportunity_health(
    background_tasks: BackgroundTasks,
    opportunity_id: str = Path(...),
    db: Session = Depends(get_db),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> RecalculateHealthResponse:
    """Force-recalc the deal's health and persist the result.

    This endpoint is intended for admin / ops use — the regular write
    endpoints already enqueue an async recalc on their happy path. Audit
    log rows are written so we can see who triggered manual recalcs.
    The owning account's status is recomputed asynchronously so an At-Risk
    flip caused by the new score lands on the account row too.
    """
    opp = _ensure_opportunity(db, opportunity_id)
    result = recalculate_deal_health(
        db,
        opportunity_id,
        write=True,
        audit_actor_type="admin",
        audit_category="admin_action",
        audit_changed_by=x_user_id,
    )
    if result is None:
        raise HTTPException(status_code=500, detail="Recalculation failed.")

    if opp.accountid:
        background_tasks.add_task(recalculate_account_async, str(opp.accountid))

    return RecalculateHealthResponse(
        opportunity_id=opportunity_id,
        health=_to_health_info(result),
        risks=_to_risk_infos(result),
    )


# ---------------------------------------------------------------------------
# GET /api/opportunities/{id}/risks
# ---------------------------------------------------------------------------


@router.get(
    "/{opportunity_id}/risks",
    response_model=RiskListResponse,
    summary="Active risks for a deal (persisted; falls back to live eval)",
)
def get_opportunity_risks(
    opportunity_id: str = Path(...),
    db: Session = Depends(get_db),
) -> RiskListResponse:
    _ensure_opportunity(db, opportunity_id)
    result = recalculate_deal_health(db, opportunity_id, write=False)
    live_items = _to_risk_infos(result) if result else []
    persisted = _load_persisted_risks(db, opportunity_id)
    # Prefer persisted rows only when the count matches live evaluation —
    # otherwise an old partial persist (e.g. before a full batch recalc) would
    # under-report relative to the rules engine.
    if persisted and len(persisted) == len(live_items):
        items = persisted
    else:
        items = live_items
    return RiskListResponse(
        opportunity_id=opportunity_id, total=len(items), items=items
    )

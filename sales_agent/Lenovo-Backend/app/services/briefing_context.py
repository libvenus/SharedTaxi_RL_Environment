"""Sprint 2 — Pre-meeting briefing facts and signals (D365 mirror only)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session

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
from app.services.what_changed import (
    build_activity_titles,
    build_crm_update_titles,
    classify_activity,
    explode_audit_diff,
    normalize_event_at,
)

INF_MSG_0004 = "INF_MSG_0004"
STALE_FIELD_DAYS = 90
ACTIVITY_GAP_DAYS = 5
DEFAULT_SIGNAL_LIMIT = 20
DEFAULT_MAX_SUMMARY_WORDS = 100

BriefingSourceType = Literal[
    "d365_account",
    "d365_opportunity",
    "d365_activity",
    "d365_audit_log",
    "d365_competitor",
    "d365_contact",
    "d365_next_action",
    "d365_deal_risk",
    "derived_activity_gap",
]


@dataclass(frozen=True)
class BriefingSourceRef:
    source_type: BriefingSourceType
    source_id: str
    label: str


@dataclass(frozen=True)
class BriefingFactField:
    field_name: str
    display_label: str
    value: str | None
    is_missing: bool
    is_unverified: bool
    source: BriefingSourceRef | None


@dataclass(frozen=True)
class BriefingCompetitorItem:
    competitor_name: str
    competitor_type: str | None
    reselling_partner: str | None
    primary_risk: str | None
    source: BriefingSourceRef


@dataclass(frozen=True)
class BriefingSignalItem:
    signal_id: str
    summary: str
    why_shown: str
    event_at: datetime
    involved_parties: list[str]
    source: BriefingSourceRef


@dataclass(frozen=True)
class BriefingAccountFacts:
    account_id: str | None
    account_name: str | None
    fields: list[BriefingFactField]
    paragraph: str
    word_count: int
    max_words: int
    gaps: list[str]
    unverified_labels: list[str]


@dataclass(frozen=True)
class BriefingDealFacts:
    opportunity_id: str
    opportunity_name: str | None
    stage: str | None
    fields: list[BriefingFactField]
    paragraph: str
    word_count: int
    max_words: int
    gaps: list[str]
    competitor_intel: list[BriefingCompetitorItem] | None
    competitor_message_code: str | None


@dataclass(frozen=True)
class BriefingContextData:
    seller_id: str
    opportunity_id: str
    account_id: str | None
    generated_at: datetime
    account: BriefingAccountFacts
    deal: BriefingDealFacts
    signals: list[BriefingSignalItem]


def _has_table(db: Session, table_name: str) -> bool:
    bind = db.get_bind()
    if bind is None:
        return False
    return inspect(bind).has_table(table_name)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _normalise_id(value: str) -> str:
    return value.strip().upper()


def _is_stale(dt: datetime | date | None, today: date) -> bool:
    if dt is None:
        return False
    if isinstance(dt, datetime):
        ref = dt.date()
    else:
        ref = dt
    return (today - ref).days > STALE_FIELD_DAYS


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _trim_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + "…"


def _ensure_seller_opportunity(
    db: Session,
    seller_id: str,
    opportunity_id: str,
) -> Opportunity:
    opp = db.execute(
        select(Opportunity).where(
            func.upper(cast(Opportunity.opportunityid, String))
            == _normalise_id(opportunity_id),
        )
    ).scalar_one_or_none()
    if opp is None:
        raise LookupError("Opportunity not found.")
    owner = (opp.owninguser or "").strip().upper()
    if owner != _normalise_id(seller_id):
        raise PermissionError("Seller does not own this opportunity.")
    return opp


def _load_decision_maker(
    db: Session,
    opportunity_id: str,
) -> tuple[str | None, BriefingSourceRef | None]:
    if not _has_table(db, "lvo_opportunitycontact"):
        return None, None
    link = db.execute(
        select(OpportunityContact, Contact)
        .join(
            Contact,
            func.upper(cast(Contact.contactid, String))
            == func.upper(cast(OpportunityContact.lvo_contactid, String)),
        )
        .where(
            func.upper(cast(OpportunityContact.lvo_opportunityid, String))
            == _normalise_id(opportunity_id),
            OpportunityContact.statecode == "Active",
            OpportunityContact.lvo_isdecisionmaker.is_(True),
        )
        .limit(1)
    ).first()
    if link is None:
        return None, None
    oc, contact = link
    name = contact.fullname or f"{contact.firstname or ''} {contact.lastname or ''}".strip()
    return name or None, BriefingSourceRef(
        source_type="d365_contact",
        source_id=str(contact.contactid),
        label="Decision maker contact",
    )


def _fact(
    field_name: str,
    display_label: str,
    value: str | None,
    source: BriefingSourceRef | None,
    *,
    stale: bool = False,
) -> BriefingFactField:
    missing = value is None or not str(value).strip()
    return BriefingFactField(
        field_name=field_name,
        display_label=display_label,
        value=None if missing else str(value).strip(),
        is_missing=missing,
        is_unverified=stale and not missing,
        source=source,
    )


def _build_account_facts(
    db: Session,
    account: Account | None,
    opportunity_id: str,
    *,
    max_words: int,
    today: date,
) -> BriefingAccountFacts:
    if account is None:
        return BriefingAccountFacts(
            account_id=None,
            account_name=None,
            fields=[],
            paragraph="Account data is not linked to this meeting.",
            word_count=6,
            max_words=max_words,
            gaps=["account"],
            unverified_labels=[],
        )

    acct_src = BriefingSourceRef(
        source_type="d365_account",
        source_id=str(account.accountid),
        label=account.name or "Account",
    )
    dm_name, dm_src = _load_decision_maker(db, opportunity_id)

    open_deals = 0
    if _has_table(db, "opportunity"):
        open_deals = int(
            db.scalar(
                select(func.count())
                .select_from(Opportunity)
                .where(
                    func.upper(cast(Opportunity.accountid, String))
                    == _normalise_id(str(account.accountid)),
                    Opportunity.statecode == "Open",
                )
            )
            or 0
        )

    stale_acct = _is_stale(account.lvo_lastinteractiondate, today)
    fields = [
        _fact("name", "Account name", account.name, acct_src),
        _fact("segment", "Segment", account.lvo_segment, acct_src, stale=stale_acct),
        _fact(
            "employees",
            "Employee count",
            str(account.numberofemployees) if account.numberofemployees is not None else None,
            acct_src,
            stale=stale_acct,
        ),
        _fact(
            "revenue",
            "Annual revenue",
            str(account.revenue) if account.revenue is not None else None,
            acct_src,
            stale=stale_acct,
        ),
        _fact("decision_maker", "Key decision maker", dm_name, dm_src),
        _fact(
            "open_deals",
            "Open Lenovo opportunities",
            str(open_deals) if open_deals else None,
            acct_src,
        ),
        _fact("industry", "Industry", account.industrycode, acct_src, stale=stale_acct),
    ]

    gaps = [f.display_label for f in fields if f.is_missing]
    unverified = [f.display_label for f in fields if f.is_unverified]

    parts: list[str] = []
    if account.name:
        parts.append(f"{account.name}")
    if account.lvo_segment:
        parts.append(f"operates in the {account.lvo_segment} segment")
    if account.numberofemployees is not None:
        parts.append(f"with approximately {account.numberofemployees:,} employees")
    if dm_name:
        parts.append(f"Key decision maker: {dm_name}")
    if account.revenue is not None:
        parts.append(f"reported revenue of {account.revenue:,.0f}")
    if open_deals:
        parts.append(f"{open_deals} open Lenovo opportunity(ies) on this account")
    if not parts:
        paragraph = "Insufficient account data in D365 to generate a summary."
    else:
        paragraph = _trim_words(". ".join(parts) + ".", max_words)

    return BriefingAccountFacts(
        account_id=str(account.accountid),
        account_name=account.name,
        fields=fields,
        paragraph=paragraph,
        word_count=_word_count(paragraph),
        max_words=max_words,
        gaps=gaps,
        unverified_labels=unverified,
    )


def _build_competitor_intel(
    db: Session,
    opportunity_id: str,
) -> tuple[list[BriefingCompetitorItem] | None, str | None]:
    if not _has_table(db, "lvo_opportunitycompetitor"):
        return None, INF_MSG_0004
    rows = (
        db.execute(
            select(OpportunityCompetitor).where(
                func.upper(cast(OpportunityCompetitor.lvo_opportunityid, String))
                == _normalise_id(opportunity_id),
                OpportunityCompetitor.statecode == "Active",
            )
        )
        .scalars()
        .all()
    )
    items: list[BriefingCompetitorItem] = []
    for row in rows:
        name = row.lvo_competitorname or row.lvo_name
        if not name or not str(name).strip():
            continue
        ctype = row.lvo_competitortype
        risk = (
            f"{ctype} competitor"
            if ctype
            else "Competitor logged on this opportunity"
        )
        items.append(
            BriefingCompetitorItem(
                competitor_name=str(name).strip(),
                competitor_type=ctype,
                reselling_partner=row.lvo_resellingpartner,
                primary_risk=risk,
                source=BriefingSourceRef(
                    source_type="d365_competitor",
                    source_id=str(row.lvo_opportunitycompetitorid),
                    label=str(name).strip(),
                ),
            )
        )
    if not items:
        return None, INF_MSG_0004
    return items, None


def _build_deal_facts(
    db: Session,
    opp: Opportunity,
    *,
    max_words: int,
    today: date,
) -> BriefingDealFacts:
    opp_id = str(opp.opportunityid)
    opp_src = BriefingSourceRef(
        source_type="d365_opportunity",
        source_id=opp_id,
        label=opp.name or "Opportunity",
    )
    stale_opp = _is_stale(opp.modifiedon, today) if opp.modifiedon else False

    health = None
    if opp.lvo_dealhealthscore is not None:
        health = str(opp.lvo_dealhealthscore)
    elif opp.lvo_riskreason:
        health = opp.lvo_riskreason

    fields = [
        _fact("name", "Deal name", opp.name, opp_src),
        _fact("stage", "Stage", opp.stagename, opp_src),
        _fact(
            "value",
            "Deal value",
            str(opp.estimatedvalue) if opp.estimatedvalue is not None else None,
            opp_src,
            stale=stale_opp,
        ),
        _fact(
            "close_date",
            "Close date",
            opp.estimatedclosedate.isoformat() if opp.estimatedclosedate else None,
            opp_src,
            stale=stale_opp,
        ),
        _fact("summary", "Deal description", opp.lvo_summary, opp_src, stale=stale_opp),
        _fact("budget_confirmed", "Budget approval", opp.budget_confirmed, opp_src),
        _fact("priority", "Priority", opp.lvo_priority, opp_src),
        _fact("forecast", "Forecast category", opp.lvo_forecastcategory, opp_src),
        _fact("deal_health", "Deal health", health, opp_src),
        _fact("service_model", "Services / DaaS", opp.service_model, opp_src),
    ]

    gaps = [f.display_label for f in fields if f.is_missing]
    competitors, comp_code = _build_competitor_intel(db, opp_id)

    parts: list[str] = []
    if opp.name:
        parts.append(f"Deal: {opp.name}")
    if opp.stagename:
        parts.append(f"currently at {opp.stagename} stage in D365")
    if opp.lvo_summary:
        parts.append(opp.lvo_summary)
    if opp.estimatedvalue is not None:
        parts.append(f"estimated value {opp.estimatedvalue:,.0f}")
    if opp.estimatedclosedate:
        parts.append(f"target close {opp.estimatedclosedate.isoformat()}")
    if opp.budget_confirmed:
        parts.append(f"budget status: {opp.budget_confirmed}")
    if opp.service_model:
        parts.append(f"service model: {opp.service_model}")
    if health:
        parts.append(f"deal health indicator: {health}")

    paragraph = (
        _trim_words(". ".join(parts) + ".", max_words)
        if parts
        else "Insufficient opportunity data in D365 to generate a deal summary."
    )

    return BriefingDealFacts(
        opportunity_id=opp_id,
        opportunity_name=opp.name,
        stage=opp.stagename,
        fields=fields,
        paragraph=paragraph,
        word_count=_word_count(paragraph),
        max_words=max_words,
        gaps=gaps,
        competitor_intel=competitors,
        competitor_message_code=comp_code,
    )


def _build_opportunity_signals(
    db: Session,
    opportunity_id: str,
    *,
    limit: int,
    today: date,
) -> list[BriefingSignalItem]:
    opp_upper = _normalise_id(opportunity_id)
    cutoff = _utc_now() - timedelta(days=30)
    items: list[BriefingSignalItem] = []

    if _has_table(db, "lvo_activity"):
        acts = (
            db.execute(
                select(Activity)
                .where(
                    func.upper(cast(Activity.lvo_opportunityid, String)) == opp_upper,
                    Activity.statecode == "Active",
                    Activity.lvo_activitydate >= cutoff,
                )
                .order_by(Activity.lvo_activitydate.desc())
            )
            .scalars()
            .all()
        )
        for act in acts:
            mapped = classify_activity(act.lvo_activitytype, act.lvo_direction)
            if mapped is None:
                continue
            title, summary = build_activity_titles(
                mapped, act.lvo_subject, act.lvo_body
            )
            why = (
                "Inbound email detected — customer-initiated contact is prioritised."
                if mapped == "email"
                else f"Recent {mapped} activity logged in D365."
            )
            items.append(
                BriefingSignalItem(
                    signal_id=f"activity:{act.lvo_activityid}",
                    summary=summary or title,
                    why_shown=why,
                    event_at=normalize_event_at(act.lvo_activitydate),
                    involved_parties=[],
                    source=BriefingSourceRef(
                        source_type="d365_activity",
                        source_id=act.lvo_activityid,
                        label=title,
                    ),
                )
            )

    if _has_table(db, "lvo_audit_log"):
        logs = (
            db.execute(
                select(AuditLog).where(
                    AuditLog.lvo_entitytype == "opportunity",
                    AuditLog.lvo_action == "update",
                    AuditLog.lvo_changedat >= cutoff,
                    or_(
                        func.upper(cast(AuditLog.lvo_opportunityid, String)) == opp_upper,
                        func.upper(cast(AuditLog.lvo_entityid, String)) == opp_upper,
                    ),
                )
            )
            .scalars()
            .all()
        )
        for log in logs:
            for field, body in explode_audit_diff(log.lvo_diff):
                title, summary = build_crm_update_titles(field, body)
                items.append(
                    BriefingSignalItem(
                        signal_id=f"audit:{log.lvo_auditlogid}:{field}",
                        summary=summary,
                        why_shown="D365 opportunity field change logged in audit trail.",
                        event_at=normalize_event_at(log.lvo_changedat),
                        involved_parties=[log.lvo_changedby] if log.lvo_changedby else [],
                        source=BriefingSourceRef(
                            source_type="d365_audit_log",
                            source_id=log.lvo_auditlogid,
                            label=title,
                        ),
                    )
                )

    if _has_table(db, "lvo_activity"):
        last_act = db.execute(
            select(func.max(Activity.lvo_activitydate)).where(
                func.upper(cast(Activity.lvo_opportunityid, String)) == opp_upper,
                Activity.statecode == "Active",
            )
        ).scalar_one_or_none()
        if last_act is not None:
            last_dt = normalize_event_at(last_act)
            gap_days = (today - last_dt.date()).days
            if gap_days >= ACTIVITY_GAP_DAYS:
                items.append(
                    BriefingSignalItem(
                        signal_id=f"gap:{opportunity_id}",
                        summary=(
                            f"No outbound activity logged against this opportunity "
                            f"in the last {gap_days} days."
                        ),
                        why_shown=(
                            "Activity gap detected from D365 activity timeline — "
                            "seller-side inactivity on an active deal is flagged."
                        ),
                        event_at=last_dt,
                        involved_parties=[],
                        source=BriefingSourceRef(
                            source_type="derived_activity_gap",
                            source_id=opportunity_id,
                            label="Activity gap",
                        ),
                    )
                )

    items.sort(key=lambda s: s.event_at, reverse=True)
    return items[:limit]


def build_briefing_context(
    db: Session,
    seller_id: str,
    opportunity_id: str,
    account_id: str | None = None,
    *,
    max_summary_words: int = DEFAULT_MAX_SUMMARY_WORDS,
    signal_limit: int = DEFAULT_SIGNAL_LIMIT,
    today: date | None = None,
) -> BriefingContextData:
    today = today or date.today()
    opp = _ensure_seller_opportunity(db, seller_id, opportunity_id)
    resolved_account_id = account_id or opp.accountid

    account: Account | None = None
    if resolved_account_id and _has_table(db, "account"):
        account = db.execute(
            select(Account).where(
                func.upper(cast(Account.accountid, String))
                == _normalise_id(str(resolved_account_id)),
            )
        ).scalar_one_or_none()

    return BriefingContextData(
        seller_id=seller_id.strip(),
        opportunity_id=str(opp.opportunityid),
        account_id=str(resolved_account_id) if resolved_account_id else None,
        generated_at=_utc_now(),
        account=_build_account_facts(
            db,
            account,
            str(opp.opportunityid),
            max_words=max_summary_words,
            today=today,
        ),
        deal=_build_deal_facts(db, opp, max_words=max_summary_words, today=today),
        signals=_build_opportunity_signals(
            db,
            str(opp.opportunityid),
            limit=signal_limit,
            today=today,
        ),
    )

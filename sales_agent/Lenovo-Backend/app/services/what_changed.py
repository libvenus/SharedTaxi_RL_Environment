"""Sprint 2 US 1.1 — portfolio-level What Changed feed aggregation.

Builds a unified chronological feed for a seller's open opportunities by
merging ``lvo_activity``, ``lvo_audit_log``, ``lvo_dealrisk``, and overdue
``lvo_nextaction`` rows. The notification panel excludes seller-initiated
CRM changes; the full activity timeline includes everything.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from sqlalchemy import String, cast, func, or_, select, text
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session

from app.models import (
    Account,
    Activity,
    AuditLog,
    DealRisk,
    NextAction,
    NotificationRead,
    Opportunity,
)
from app.schemas import WhatChangedItem

WhatChangedActivityType = Literal["email", "meeting", "crm_update", "risk", "task"]
WhatChangedLinkType = Literal["opportunity", "account", "outreach", "activity", "todo"]
WhatChangedDirection = Literal["inbound", "outbound"]

ERR_MSG_0020 = "ERR_MSG_0020"

NOTIFICATION_PANEL_DEFAULT_LIMIT = 6
NOTIFICATION_PANEL_MAX_LIMIT = 6
ACTIVITY_TIMELINE_DEFAULT_PAGE_SIZE = 25
ACTIVITY_TIMELINE_MAX_PAGE_SIZE = 100
FEED_LOOKBACK_DAYS = 30


@dataclass(frozen=True)
class _OpportunityContext:
    opportunity_id: str
    opportunity_name: str | None
    account_id: str | None
    account_name: str | None


def _has_table(db: Session, table_name: str) -> bool:
    return inspect(db.bind).has_table(table_name)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def normalize_event_at(dt: datetime) -> datetime:
    """Coerce DB timestamps to naive UTC so feed items sort reliably.

    Postgres ``TIMESTAMPTZ`` columns come back timezone-aware via SQLAlchemy;
    ``TIMESTAMP`` / ``datetime.combine`` values are naive. Mixing both breaks
    ``list.sort(key=event_at)``.
    """
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _normalise_actor(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped.upper() if stripped else None


def is_seller_actor(
    actor: str | None,
    seller_id: str,
    viewer_id: str | None = None,
) -> bool:
    """True when the event was triggered by the seller (or the X-User-Id viewer)."""
    actor_norm = _normalise_actor(actor)
    if actor_norm is None:
        return False
    if actor_norm == seller_id.strip().upper():
        return True
    if viewer_id and actor_norm == viewer_id.strip().upper():
        return True
    return False


def explode_audit_diff(diff_json: str | None) -> list[tuple[str, str]]:
    """Turn an opportunity audit-log diff into per-field timeline entries."""
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


def normalize_direction(raw: str | None) -> WhatChangedDirection | None:
    """Normalise ``lvo_activity.lvo_direction`` for the API."""
    if not raw:
        return None
    value = raw.strip().lower()
    if value in {"inbound", "outbound"}:
        return value  # type: ignore[return-value]
    return None


def activity_category_label(
    activity_type: WhatChangedActivityType,
    *,
    changed_field: str | None = None,
) -> str:
    """Subtitle prefix shown before the actor name on Activity Timeline rows."""
    if activity_type == "email":
        return "Email received"
    if activity_type == "meeting":
        return "Meeting completed"
    if activity_type == "crm_update":
        if changed_field and changed_field.strip().lower() == "stagename":
            return "Stage progression tracked"
        return "CRM field updated"
    if activity_type == "risk":
        return "Risk signal detected"
    return "Task overdue"


def resolve_actor_names(db: Session, actor_values: set[str]) -> dict[str, str | None]:
    """Batch-resolve actor UUIDs or emails to ``systemuser`` display names.

    Returns a dict keyed by UPPER(raw value) → fullname (or email fallback).
    """
    if not actor_values or not _has_table(db, "systemuser"):
        return {}

    inspector = inspect(db.bind)
    cols = {c["name"] for c in inspector.get_columns("systemuser")}
    if "systemuserid" not in cols:
        return {}

    name_col = (
        "fullname"
        if "fullname" in cols
        else "internalemailaddress"
        if "internalemailaddress" in cols
        else None
    )
    if name_col is None:
        return {}

    uuids: list[str] = []
    emails: list[str] = []
    for raw in actor_values:
        if not raw or not raw.strip():
            continue
        if "@" in raw:
            emails.append(raw.strip().lower())
        else:
            uuids.append(raw.strip().upper())

    out: dict[str, str | None] = {}

    if uuids:
        placeholders = ", ".join(f":u{i}" for i in range(len(uuids)))
        params = {f"u{i}": uid for i, uid in enumerate(uuids)}
        sql = (
            f"SELECT UPPER(systemuserid::TEXT) AS actor_key, {name_col} AS label "
            f"FROM systemuser WHERE UPPER(systemuserid::TEXT) IN ({placeholders})"
        )
        for row in db.execute(text(sql), params):
            if row.label:
                out[row.actor_key] = row.label

    if emails and "internalemailaddress" in cols:
        placeholders = ", ".join(f":e{i}" for i in range(len(emails)))
        params = {f"e{i}": email for i, email in enumerate(emails)}
        sql = (
            f"SELECT LOWER(internalemailaddress) AS actor_key, {name_col} AS label "
            f"FROM systemuser WHERE LOWER(internalemailaddress) IN ({placeholders})"
        )
        for row in db.execute(text(sql), params):
            if row.label:
                out[row.actor_key.upper()] = row.label

    return out


def _enrich_feed_items(db: Session, items: list[WhatChangedItem]) -> list[WhatChangedItem]:
    actor_values = {item.actor for item in items if item.actor}
    names = resolve_actor_names(db, actor_values)
    if not names:
        return items
    enriched: list[WhatChangedItem] = []
    for item in items:
        key = _normalise_actor(item.actor)
        actor_name = names.get(key) if key else None
        enriched.append(item.model_copy(update={"actor_name": actor_name}))
    return enriched


def classify_activity(
    activity_type: str,
    direction: str | None,
) -> WhatChangedActivityType | None:
    """Map ``lvo_activity`` rows to notification activity types."""
    at = (activity_type or "").strip().lower()
    dir_norm = (direction or "").strip().lower()
    if at == "email":
        if dir_norm == "outbound":
            return None
        return "email"
    if at == "meeting":
        return "meeting"
    if at in {"crm", "multiple"}:
        return "crm_update"
    return None


def build_activity_titles(
    activity_type: WhatChangedActivityType,
    subject: str | None,
    body: str | None,
) -> tuple[str, str]:
    subj = (subject or "").strip()
    snippet = (body or "").strip()
    if len(snippet) > 120:
        snippet = snippet[:117] + "..."
    if activity_type == "email":
        title = subj or "Email received"
        summary = snippet or "New inbound email on this deal."
        return title, summary
    if activity_type == "meeting":
        title = f"Meeting completed: {subj}" if subj else "Meeting completed"
        summary = snippet or "A calendar meeting was logged on this deal."
        return title, summary
    title = subj or "CRM activity"
    summary = snippet or "CRM touchpoint recorded on this deal."
    return title, summary


def build_crm_update_titles(changed_field: str, body: str) -> tuple[str, str]:
    field = changed_field.strip()
    if field.lower() == "stagename":
        return "Stage changed", body
    label = field.replace("_", " ").strip().title() or "Field"
    return f"{label} updated", body


def feed_key_activity(activity_id: str) -> str:
    return f"activity:{activity_id}"


def feed_key_audit(audit_id: str, field: str) -> str:
    return f"audit:{audit_id}:{field}"


def feed_key_risk(risk_id: str) -> str:
    return f"risk:{risk_id}"


def feed_key_task(action_id: str) -> str:
    return f"task:{action_id}"


def _load_seller_opportunities(db: Session, seller_id: str) -> dict[str, _OpportunityContext]:
    stmt = select(Opportunity).where(
        func.upper(cast(Opportunity.owninguser, String)) == seller_id.strip().upper(),
        or_(Opportunity.statecode.is_(None), Opportunity.statecode == "Open"),
    )
    opps = db.execute(stmt).scalars().all()
    if not opps:
        return {}

    account_ids = {o.accountid for o in opps if o.accountid}
    account_names: dict[str, str | None] = {}
    if account_ids and _has_table(db, "account"):
        for acct in (
            db.execute(select(Account).where(Account.accountid.in_(account_ids)))
            .scalars()
            .all()
        ):
            account_names[acct.accountid] = acct.name

    out: dict[str, _OpportunityContext] = {}
    for opp in opps:
        oid = str(opp.opportunityid)
        aid = opp.accountid
        out[oid.upper()] = _OpportunityContext(
            opportunity_id=oid,
            opportunity_name=opp.name,
            account_id=aid,
            account_name=account_names.get(aid) if aid else None,
        )
    return out


def _load_read_keys(db: Session, seller_id: str) -> set[str]:
    if not _has_table(db, "lvo_notification_read"):
        return set()
    rows = (
        db.execute(
            select(NotificationRead.feed_item_key).where(
                func.upper(NotificationRead.seller_id) == seller_id.strip().upper()
            )
        )
        .scalars()
        .all()
    )
    return set(rows)


def _to_item(
    *,
    feed_key: str,
    activity_type: WhatChangedActivityType,
    title: str,
    summary: str,
    ctx: _OpportunityContext,
    event_at: datetime,
    link_type: WhatChangedLinkType,
    link_id: str,
    actor: str | None,
    read_keys: set[str],
    direction: WhatChangedDirection | None = None,
    changed_field: str | None = None,
) -> WhatChangedItem:
    return WhatChangedItem(
        id=feed_key,
        activity_type=activity_type,
        title=title,
        summary=summary,
        account_id=ctx.account_id,
        account_name=ctx.account_name,
        opportunity_id=ctx.opportunity_id,
        opportunity_name=ctx.opportunity_name,
        event_at=normalize_event_at(event_at),
        is_read=feed_key in read_keys,
        link_type=link_type,
        link_id=link_id,
        actor=actor,
        actor_name=None,
        direction=direction,
        category_label=activity_category_label(
            activity_type, changed_field=changed_field
        ),
    )


def build_seller_feed(
    db: Session,
    seller_id: str,
    *,
    viewer_id: str | None = None,
    exclude_self_changes: bool = True,
    activity_types: set[WhatChangedActivityType] | None = None,
    lookback_days: int = FEED_LOOKBACK_DAYS,
    today: date | None = None,
) -> list[WhatChangedItem]:
    """Aggregate portfolio feed items for one seller, newest first."""
    opp_map = _load_seller_opportunities(db, seller_id)
    if not opp_map:
        return []

    opp_ids_upper = list(opp_map.keys())
    read_keys = _load_read_keys(db, seller_id)
    cutoff = _utc_now() - timedelta(days=lookback_days)
    today = today or date.today()
    items: list[WhatChangedItem] = []

    def _type_allowed(t: WhatChangedActivityType) -> bool:
        return activity_types is None or t in activity_types

    # --- Activities ---------------------------------------------------------
    if _has_table(db, "lvo_activity"):
        stmt = (
            select(Activity)
            .where(
                func.upper(cast(Activity.lvo_opportunityid, String)).in_(opp_ids_upper),
                Activity.statecode == "Active",
                Activity.lvo_activitydate >= cutoff,
            )
            .order_by(Activity.lvo_activitydate.desc())
        )
        for act in db.execute(stmt).scalars().all():
            mapped = classify_activity(act.lvo_activitytype, act.lvo_direction)
            if mapped is None or not _type_allowed(mapped):
                continue
            # Outbound emails are excluded via classify_activity(). Seller-owned
            # CRM audit rows are excluded below. Do not skip all activities
            # where owninguser == seller — inbound email / meetings should still
            # surface on the panel.
            ctx = opp_map.get(str(act.lvo_opportunityid).upper())
            if ctx is None:
                continue
            title, summary = build_activity_titles(mapped, act.lvo_subject, act.lvo_body)
            link_type: WhatChangedLinkType = "outreach" if mapped == "email" else "activity"
            link_id = ctx.opportunity_id if mapped == "email" else act.lvo_activityid
            items.append(
                _to_item(
                    feed_key=feed_key_activity(act.lvo_activityid),
                    activity_type=mapped,
                    title=title,
                    summary=summary,
                    ctx=ctx,
                    event_at=act.lvo_activitydate,
                    link_type=link_type,
                    link_id=link_id,
                    actor=act.owninguser,
                    read_keys=read_keys,
                    direction=normalize_direction(act.lvo_direction),
                )
            )

    # --- CRM audit changes --------------------------------------------------
    if _has_table(db, "lvo_audit_log") and (
        activity_types is None or "crm_update" in activity_types
    ):
        audit_rows = (
            db.execute(
                select(AuditLog).where(
                    AuditLog.lvo_entitytype == "opportunity",
                    AuditLog.lvo_action == "update",
                    AuditLog.lvo_changedat >= cutoff,
                    or_(
                        func.upper(cast(AuditLog.lvo_opportunityid, String)).in_(
                            opp_ids_upper
                        ),
                        func.upper(cast(AuditLog.lvo_entityid, String)).in_(opp_ids_upper),
                    ),
                )
            )
            .scalars()
            .all()
        )
        for log in audit_rows:
            if exclude_self_changes and is_seller_actor(
                log.lvo_changedby, seller_id, viewer_id
            ):
                continue
            opp_key = (
                str(log.lvo_opportunityid or log.lvo_entityid or "").upper()
            )
            ctx = opp_map.get(opp_key)
            if ctx is None:
                continue
            for field, body in explode_audit_diff(log.lvo_diff):
                title, summary = build_crm_update_titles(field, body)
                items.append(
                    _to_item(
                        feed_key=feed_key_audit(log.lvo_auditlogid, field),
                        activity_type="crm_update",
                        title=title,
                        summary=summary,
                        ctx=ctx,
                        event_at=log.lvo_changedat,
                        link_type="opportunity",
                        link_id=ctx.opportunity_id,
                        actor=log.lvo_changedby,
                        read_keys=read_keys,
                        changed_field=field,
                    )
                )

    # --- Deal risks ---------------------------------------------------------
    if _has_table(db, "lvo_dealrisk") and (
        activity_types is None or "risk" in activity_types
    ):
        risk_rows = (
            db.execute(
                select(DealRisk).where(
                    func.upper(cast(DealRisk.lvo_opportunityid, String)).in_(
                        opp_ids_upper
                    ),
                    DealRisk.statecode == "Active",
                    DealRisk.lvo_detectedat >= cutoff,
                )
            )
            .scalars()
            .all()
        )
        for risk in risk_rows:
            ctx = opp_map.get(str(risk.lvo_opportunityid).upper())
            if ctx is None:
                continue
            items.append(
                _to_item(
                    feed_key=feed_key_risk(risk.lvo_dealriskid),
                    activity_type="risk",
                    title=risk.lvo_riskname,
                    summary=risk.lvo_message,
                    ctx=ctx,
                    event_at=risk.lvo_detectedat,
                    link_type="opportunity",
                    link_id=ctx.opportunity_id,
                    actor=None,
                    read_keys=read_keys,
                )
            )

    # --- Overdue next actions -----------------------------------------------
    if _has_table(db, "lvo_nextaction") and (
        activity_types is None or "task" in activity_types
    ):
        task_rows = (
            db.execute(
                select(NextAction).where(
                    func.upper(cast(NextAction.lvo_opportunityid, String)).in_(
                        opp_ids_upper
                    ),
                    NextAction.statecode == "Active",
                    NextAction.lvo_status == "Open",
                    NextAction.lvo_duedate.is_not(None),
                    NextAction.lvo_duedate < today,
                )
            )
            .scalars()
            .all()
        )
        for task in task_rows:
            ctx = opp_map.get(str(task.lvo_opportunityid).upper())
            if ctx is None:
                continue
            due = task.lvo_duedate
            items.append(
                _to_item(
                    feed_key=feed_key_task(task.lvo_nextactionid),
                    activity_type="task",
                    title="Overdue task",
                    summary=task.lvo_description,
                    ctx=ctx,
                    event_at=datetime.combine(due, datetime.min.time())
                    if due
                    else task.lvo_updatedat,
                    link_type="todo",
                    link_id=task.lvo_nextactionid,
                    actor=task.lvo_createdby,
                    read_keys=read_keys,
                )
            )

    items.sort(key=lambda i: i.event_at, reverse=True)
    return _enrich_feed_items(db, items)


def paginate_feed(
    items: list[WhatChangedItem],
    page: int,
    page_size: int,
) -> tuple[list[WhatChangedItem], int, int]:
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    total_pages = math.ceil(total / page_size) if page_size else 0
    return items[start:end], total, total_pages


def mark_notification_read(
    db: Session,
    seller_id: str,
    notification_id: str,
) -> datetime:
    """Idempotently mark one feed item as read for a seller."""
    now = _utc_now()
    if not _has_table(db, "lvo_notification_read"):
        return now

    existing = db.execute(
        select(NotificationRead).where(
            func.upper(NotificationRead.seller_id) == seller_id.strip().upper(),
            NotificationRead.feed_item_key == notification_id,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing.read_at

    row = NotificationRead(
        lvo_notificationreadid=str(uuid.uuid4()),
        seller_id=seller_id.strip(),
        feed_item_key=notification_id,
        read_at=now,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row.read_at


def parse_activity_type_filter(raw: str | None) -> set[WhatChangedActivityType] | None:
    if not raw:
        return None
    mapping = {
        "email": "email",
        "meeting": "meeting",
        "crm": "crm_update",
        "crm_update": "crm_update",
        "risk": "risk",
        "task": "task",
    }
    out: set[WhatChangedActivityType] = set()
    for token in raw.split(","):
        key = token.strip().lower()
        if not key:
            continue
        mapped = mapping.get(key)
        if mapped:
            out.add(mapped)  # type: ignore[arg-type]
    return out or None

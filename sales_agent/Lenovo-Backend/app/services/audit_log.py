"""Central compliance audit log writer.

All platform mutations and configurable read actions should flow through
``write_audit_event`` so records are consistent for Azure Monitor / Log
Analytics export and compliance queries.

Records are append-only at the application layer; Postgres triggers block
UPDATE and block DELETE except during the retention purge job.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session

from app.models import AuditConfig, AuditLog

logger = logging.getLogger(__name__)

ActorType = Literal["seller", "admin", "ai", "system", "event_spine"]
AuditCategory = Literal[
    "seller_action",
    "admin_action",
    "ai_automated",
    "crm_writeback",
    "event_spine",
    "read_action",
    "system",
]
AuditOutcome = Literal["success", "failure"]

_DEFAULT_RETENTION_DAYS = 90


@dataclass(frozen=True)
class AuditConfigSnapshot:
    retention_days: int
    log_seller_reads: bool
    log_admin_reads: bool
    log_ai_output_reads: bool
    updated_at: datetime | None
    updated_by: str | None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _has_audit_table(db: Session) -> bool:
    bind = db.get_bind()
    if bind is None:
        return False
    return inspect(bind).has_table("lvo_audit_log")


def _has_compliance_audit_columns(db: Session) -> bool:
    """True when sql/2026_14_audit_compliance.sql has been applied."""
    bind = db.get_bind()
    if bind is None or not _has_audit_table(db):
        return False
    return "lvo_actortype" in inspect(bind).get_columns("lvo_audit_log")


def _has_config_table(db: Session) -> bool:
    bind = db.get_bind()
    if bind is None:
        return False
    return inspect(bind).has_table("lvo_audit_config")


def get_audit_config(db: Session) -> AuditConfigSnapshot:
    """Load singleton config or return safe defaults when not migrated."""
    if not _has_config_table(db):
        return AuditConfigSnapshot(
            retention_days=_DEFAULT_RETENTION_DAYS,
            log_seller_reads=False,
            log_admin_reads=False,
            log_ai_output_reads=False,
            updated_at=None,
            updated_by=None,
        )
    row = db.get(AuditConfig, 1)
    if row is None:
        return AuditConfigSnapshot(
            retention_days=_DEFAULT_RETENTION_DAYS,
            log_seller_reads=False,
            log_admin_reads=False,
            log_ai_output_reads=False,
            updated_at=None,
            updated_by=None,
        )
    return AuditConfigSnapshot(
        retention_days=int(row.retention_days),
        log_seller_reads=bool(row.log_seller_reads),
        log_admin_reads=bool(row.log_admin_reads),
        log_ai_output_reads=bool(row.log_ai_output_reads),
        updated_at=row.updated_at,
        updated_by=row.updated_by,
    )


def update_audit_config(
    db: Session,
    *,
    retention_days: int | None = None,
    log_seller_reads: bool | None = None,
    log_admin_reads: bool | None = None,
    log_ai_output_reads: bool | None = None,
    updated_by: str | None = None,
) -> AuditConfigSnapshot:
    if not _has_config_table(db):
        raise RuntimeError("lvo_audit_config table is not migrated.")

    row = db.get(AuditConfig, 1)
    if row is None:
        row = AuditConfig(id=1)
        db.add(row)

    before = {
        "retention_days": row.retention_days,
        "log_seller_reads": row.log_seller_reads,
        "log_admin_reads": row.log_admin_reads,
        "log_ai_output_reads": row.log_ai_output_reads,
    }

    if retention_days is not None:
        row.retention_days = int(retention_days)
    if log_seller_reads is not None:
        row.log_seller_reads = bool(log_seller_reads)
    if log_admin_reads is not None:
        row.log_admin_reads = bool(log_admin_reads)
    if log_ai_output_reads is not None:
        row.log_ai_output_reads = bool(log_ai_output_reads)

    now = _utc_now()
    row.updated_at = now
    row.updated_by = updated_by

    after = {
        "retention_days": row.retention_days,
        "log_seller_reads": row.log_seller_reads,
        "log_admin_reads": row.log_admin_reads,
        "log_ai_output_reads": row.log_ai_output_reads,
    }

    write_audit_event(
        db,
        entity_type="audit_config",
        entity_id="1",
        action="update",
        category="admin_action",
        actor_type="admin",
        changed_by=updated_by,
        diff={"before": before, "after": after},
        source_service="d365_sales",
    )
    db.commit()
    return get_audit_config(db)


def build_field_changes(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Expand a before/after dict into per-field CRM write-back rows."""
    before = before or {}
    after = after or {}
    keys = sorted(set(before) | set(after))
    changes: list[dict[str, Any]] = []
    for key in keys:
        b, a = before.get(key), after.get(key)
        if b != a:
            changes.append({"field": key, "before": b, "after": a})
    return changes


def write_audit_event(
    db: Session,
    *,
    entity_type: str,
    entity_id: str,
    action: str,
    category: AuditCategory = "crm_writeback",
    actor_type: ActorType = "seller",
    changed_by: str | None = None,
    opportunity_id: str | None = None,
    outcome: AuditOutcome = "success",
    correlation_id: str | None = None,
    failure_reason: str | None = None,
    event_type: str | None = None,
    delivery_attempts: int | None = None,
    source_service: str = "d365_sales",
    diff: dict[str, Any] | None = None,
    field_changes: list[dict[str, Any]] | None = None,
) -> str | None:
    """Append one immutable audit row. Caller commits the session.

    Returns the new audit id, or ``None`` when the table is not migrated.
    """
    if not _has_audit_table(db):
        logger.warning("lvo_audit_log missing — audit event skipped for %s", entity_type)
        return None

    payload = dict(diff) if diff else {}
    if field_changes:
        payload["fieldChanges"] = field_changes
    elif "before" in payload and "after" in payload and "fieldChanges" not in payload:
        payload["fieldChanges"] = build_field_changes(
            payload.get("before"), payload.get("after")
        )

    audit_id = str(uuid.uuid4())
    diff_json = json.dumps(payload) if payload else None
    changed_at = _utc_now()

    if not _has_compliance_audit_columns(db):
        # Pre-2026_14 schema — legacy columns only. Map actions the old
        # CHECK constraints allow so batch jobs (e.g. recalc_health) still run.
        legacy_action = action if action in {"create", "update", "delete"} else "update"
        legacy_entity = (
            entity_type
            if entity_type in {
                "opportunity",
                "competitor",
                "next_action",
                "opportunity_contact",
                "account_contact",
            }
            else "opportunity"
        )
        db.execute(
            text(
                """
                INSERT INTO lvo_audit_log (
                    lvo_auditlogid, lvo_entitytype, lvo_entityid,
                    lvo_opportunityid, lvo_action, lvo_changedby,
                    lvo_changedat, lvo_diff
                ) VALUES (
                    :id, :entity_type, :entity_id,
                    :opportunity_id, :action, :changed_by,
                    :changed_at, CAST(:diff AS JSONB)
                )
                """
            ),
            {
                "id": audit_id,
                "entity_type": legacy_entity,
                "entity_id": entity_id,
                "opportunity_id": opportunity_id,
                "action": legacy_action,
                "changed_by": changed_by,
                "changed_at": changed_at,
                "diff": diff_json,
            },
        )
    else:
        row = AuditLog(
            lvo_auditlogid=audit_id,
            lvo_entitytype=entity_type,
            lvo_entityid=entity_id,
            lvo_opportunityid=opportunity_id,
            lvo_action=action,
            lvo_changedby=changed_by,
            lvo_changedat=changed_at,
            lvo_diff=diff_json,
            lvo_actortype=actor_type,
            lvo_category=category,
            lvo_outcome=outcome,
            lvo_correlationid=correlation_id,
            lvo_failurereason=failure_reason,
            lvo_eventtype=event_type,
            lvo_deliveryattempts=delivery_attempts,
            lvo_sourceservice=source_service,
        )
        db.add(row)

    logger.info(
        "audit_event",
        extra={
            "audit_id": audit_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "category": category,
            "actor_type": actor_type,
            "outcome": outcome,
            "correlation_id": correlation_id,
            "source_service": source_service,
        },
    )
    return audit_id


def query_audit_events(
    db: Session,
    *,
    entity_type: str | None = None,
    entity_id: str | None = None,
    opportunity_id: str | None = None,
    category: str | None = None,
    correlation_id: str | None = None,
    actor_type: str | None = None,
    changed_by: str | None = None,
    from_at: datetime | None = None,
    to_at: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Compliance query — paginated, newest first."""
    if not _has_audit_table(db):
        return [], 0

    stmt = select(AuditLog)
    if entity_type:
        stmt = stmt.where(AuditLog.lvo_entitytype == entity_type)
    if entity_id:
        stmt = stmt.where(AuditLog.lvo_entityid == entity_id)
    if opportunity_id:
        stmt = stmt.where(AuditLog.lvo_opportunityid == opportunity_id)
    if category:
        stmt = stmt.where(AuditLog.lvo_category == category)
    if correlation_id:
        stmt = stmt.where(AuditLog.lvo_correlationid == correlation_id)
    if actor_type:
        stmt = stmt.where(AuditLog.lvo_actortype == actor_type)
    if changed_by:
        stmt = stmt.where(AuditLog.lvo_changedby == changed_by)
    if from_at:
        stmt = stmt.where(AuditLog.lvo_changedat >= from_at)
    if to_at:
        stmt = stmt.where(AuditLog.lvo_changedat <= to_at)

    count_stmt = select(func.count()).select_from(AuditLog)
    if entity_type:
        count_stmt = count_stmt.where(AuditLog.lvo_entitytype == entity_type)
    if entity_id:
        count_stmt = count_stmt.where(AuditLog.lvo_entityid == entity_id)
    if opportunity_id:
        count_stmt = count_stmt.where(AuditLog.lvo_opportunityid == opportunity_id)
    if category:
        count_stmt = count_stmt.where(AuditLog.lvo_category == category)
    if correlation_id:
        count_stmt = count_stmt.where(AuditLog.lvo_correlationid == correlation_id)
    if actor_type:
        count_stmt = count_stmt.where(AuditLog.lvo_actortype == actor_type)
    if changed_by:
        count_stmt = count_stmt.where(AuditLog.lvo_changedby == changed_by)
    if from_at:
        count_stmt = count_stmt.where(AuditLog.lvo_changedat >= from_at)
    if to_at:
        count_stmt = count_stmt.where(AuditLog.lvo_changedat <= to_at)

    total = int(db.execute(count_stmt).scalar_one() or 0)
    rows = (
        db.execute(
            stmt.order_by(AuditLog.lvo_changedat.desc()).limit(limit).offset(offset)
        )
        .scalars()
        .all()
    )
    return [_serialize_row(r) for r in rows], total


def _serialize_row(row: AuditLog) -> dict[str, Any]:
    diff: dict[str, Any] | None = None
    if row.lvo_diff:
        try:
            diff = json.loads(row.lvo_diff)
        except json.JSONDecodeError:
            diff = {"raw": row.lvo_diff}
    return {
        "id": row.lvo_auditlogid,
        "entityType": row.lvo_entitytype,
        "entityId": row.lvo_entityid,
        "opportunityId": row.lvo_opportunityid,
        "action": row.lvo_action,
        "category": row.lvo_category,
        "actorType": row.lvo_actortype,
        "changedBy": row.lvo_changedby,
        "changedAt": row.lvo_changedat,
        "outcome": row.lvo_outcome,
        "correlationId": row.lvo_correlationid,
        "failureReason": row.lvo_failurereason,
        "eventType": row.lvo_eventtype,
        "deliveryAttempts": row.lvo_deliveryattempts,
        "sourceService": row.lvo_sourceservice,
        "diff": diff,
    }


def purge_expired_audit_rows(db: Session, *, retention_days: int) -> int:
    """Delete rows older than retention. Requires audit.purge_mode session flag."""
    if not _has_audit_table(db):
        return 0
    cutoff = _utc_now() - timedelta(days=int(retention_days))
    db.execute(text("SET LOCAL audit.purge_mode = 'on'"))
    result = db.execute(
        text(
            "DELETE FROM lvo_audit_log WHERE lvo_changedat < :cutoff"
        ),
        {"cutoff": cutoff},
    )
    db.commit()
    return int(result.rowcount or 0)


def should_log_read(
    config: AuditConfigSnapshot,
    *,
    actor_type: ActorType,
    path: str,
) -> bool:
    """Return True when this GET should be captured per runtime config."""
    if not path.startswith("/api/"):
        return False
    if actor_type == "seller":
        return config.log_seller_reads
    if actor_type == "admin":
        return config.log_admin_reads
    if actor_type == "ai":
        return config.log_ai_output_reads
    return False


def infer_actor_type_from_path(path: str, header_actor: str | None) -> ActorType:
    if header_actor in {"seller", "admin", "ai", "system", "event_spine"}:
        return header_actor  # type: ignore[return-value]
    if path.startswith("/api/sales-operating-model"):
        return "admin"
    if path.startswith("/api/compliance"):
        return "admin"
    return "seller"


__all__ = [
    "AuditConfigSnapshot",
    "build_field_changes",
    "get_audit_config",
    "infer_actor_type_from_path",
    "purge_expired_audit_rows",
    "query_audit_events",
    "should_log_read",
    "update_audit_config",
    "write_audit_event",
]

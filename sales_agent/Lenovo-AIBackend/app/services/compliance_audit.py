"""Cross-cutting compliance audit writer for Lenovo-AIBackend.

Writes to the shared ``lvo_audit_log`` table (same Postgres as D365 Sales).
Uses HTTP ingest when ``COMPLIANCE_API_KEY`` is set; otherwise inserts
directly via SQL so local dev works without a round-trip.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import D365_BASE_URL, D365_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

_COMPLIANCE_KEY_ENV = "COMPLIANCE_API_KEY"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _table_exists(db: Session) -> bool:
    row = db.execute(
        text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'lvo_audit_log' LIMIT 1"
        )
    ).scalar()
    return row is not None


def write_compliance_audit(
    db: Session,
    *,
    entity_type: str,
    entity_id: str,
    action: str,
    category: str = "ai_automated",
    actor_type: str = "ai",
    changed_by: str | None = None,
    opportunity_id: str | None = None,
    outcome: str = "success",
    correlation_id: str | None = None,
    failure_reason: str | None = None,
    event_type: str | None = None,
    delivery_attempts: int | None = None,
    diff: dict[str, Any] | None = None,
    field_changes: list[dict[str, Any]] | None = None,
) -> str | None:
    """Append one compliance audit row. Caller may commit the session."""
    import os

    payload = dict(diff) if diff else {}
    if field_changes:
        payload["fieldChanges"] = field_changes

    body = {
        "entity_type": entity_type,
        "entity_id": str(entity_id),
        "action": action,
        "category": category,
        "actor_type": actor_type,
        "changed_by": changed_by,
        "opportunity_id": opportunity_id,
        "outcome": outcome,
        "correlation_id": correlation_id,
        "failure_reason": failure_reason,
        "event_type": event_type,
        "delivery_attempts": delivery_attempts,
        "source_service": "ai_backend",
        "diff": payload or None,
        "field_changes": field_changes,
    }

    api_key = os.getenv(_COMPLIANCE_KEY_ENV)
    if api_key:
        try:
            url = f"{D365_BASE_URL.rstrip('/')}/api/compliance/audit-events"
            resp = httpx.post(
                url,
                json=body,
                headers={"X-Compliance-Api-Key": api_key},
                timeout=D365_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json().get("id")
        except Exception:
            logger.exception("Compliance audit HTTP ingest failed; falling back to SQL")

    if not _table_exists(db):
        logger.warning("lvo_audit_log missing — skipped audit for %s", entity_type)
        return None

    audit_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO lvo_audit_log (
                lvo_auditlogid, lvo_entitytype, lvo_entityid, lvo_opportunityid,
                lvo_action, lvo_changedby, lvo_changedat, lvo_diff,
                lvo_actortype, lvo_category, lvo_outcome, lvo_correlationid,
                lvo_failurereason, lvo_eventtype, lvo_deliveryattempts,
                lvo_sourceservice
            ) VALUES (
                :id, :entity_type, :entity_id, :opportunity_id,
                :action, :changed_by, :changed_at, CAST(:diff AS jsonb),
                :actor_type, :category, :outcome, :correlation_id,
                :failure_reason, :event_type, :delivery_attempts,
                :source_service
            )
            """
        ),
        {
            "id": audit_id,
            "entity_type": entity_type,
            "entity_id": str(entity_id),
            "opportunity_id": opportunity_id,
            "action": action,
            "changed_by": changed_by,
            "changed_at": _utc_now(),
            "diff": json.dumps(payload) if payload else None,
            "actor_type": actor_type,
            "category": category,
            "outcome": outcome,
            "correlation_id": correlation_id,
            "failure_reason": failure_reason,
            "event_type": event_type,
            "delivery_attempts": delivery_attempts,
            "source_service": "ai_backend",
        },
    )
    logger.info(
        "compliance_audit",
        extra={
            "audit_id": audit_id,
            "entity_type": entity_type,
            "action": action,
            "category": category,
        },
    )
    return audit_id


def record_event_spine(
    db: Session,
    *,
    event_id: str,
    event_type: str,
    outcome: str,
    delivery_attempts: int,
    failure_reason: str | None = None,
    correlation_id: str | None = None,
    diff: dict[str, Any] | None = None,
) -> str | None:
    """Helper for future Event Spine delivery / dead-letter rows."""
    return write_compliance_audit(
        db,
        entity_type="event_spine",
        entity_id=event_id,
        action="dead_letter" if outcome == "failure" else "deliver",
        category="event_spine",
        actor_type="event_spine",
        outcome=outcome,
        event_type=event_type,
        delivery_attempts=delivery_attempts,
        failure_reason=failure_reason,
        correlation_id=correlation_id or event_id,
        diff=diff,
    )

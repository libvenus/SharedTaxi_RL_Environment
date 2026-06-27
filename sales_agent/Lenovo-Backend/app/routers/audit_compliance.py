"""Compliance audit API — ingest, query, and runtime config.

No seller-facing UI. Intended for compliance/ops tooling and cross-service
ingest (Lenovo-AIBackend). Protect with ``X-Compliance-Api-Key``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.services.audit_log import (
    get_audit_config,
    query_audit_events,
    update_audit_config,
    write_audit_event,
)

router = APIRouter(prefix="/api/compliance", tags=["Compliance · Audit Log"])


class AuditEventIngestRequest(BaseModel):
    entity_type: str
    entity_id: str
    action: str
    category: str = "system"
    actor_type: str = "system"
    changed_by: str | None = None
    opportunity_id: str | None = None
    outcome: str = "success"
    correlation_id: str | None = None
    failure_reason: str | None = None
    event_type: str | None = None
    delivery_attempts: int | None = None
    source_service: str = "ai_backend"
    diff: dict[str, Any] | None = None
    field_changes: list[dict[str, Any]] | None = None


class AuditEventIngestResponse(BaseModel):
    id: str


class AuditConfigResponse(BaseModel):
    retention_days: int
    log_seller_reads: bool
    log_admin_reads: bool
    log_ai_output_reads: bool
    updated_at: datetime | None = None
    updated_by: str | None = None


class AuditConfigUpdateRequest(BaseModel):
    retention_days: int | None = Field(default=None, ge=1, le=3650)
    log_seller_reads: bool | None = None
    log_admin_reads: bool | None = None
    log_ai_output_reads: bool | None = None


class AuditEventListResponse(BaseModel):
    total: int
    items: list[dict[str, Any]]


def _verify_compliance_key(
    x_compliance_api_key: str | None = Header(default=None, alias="X-Compliance-Api-Key"),
) -> None:
    expected = get_settings().compliance_api_key
    if not expected:
        return
    if x_compliance_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Compliance-Api-Key.",
        )


@router.post(
    "/audit-events",
    response_model=AuditEventIngestResponse,
    summary="Ingest an audit event (AIBackend / Event Spine)",
)
def ingest_audit_event(
    body: AuditEventIngestRequest,
    db: Session = Depends(get_db),
    _: None = Depends(_verify_compliance_key),
) -> AuditEventIngestResponse:
    audit_id = write_audit_event(
        db,
        entity_type=body.entity_type,
        entity_id=body.entity_id,
        action=body.action,
        category=body.category,  # type: ignore[arg-type]
        actor_type=body.actor_type,  # type: ignore[arg-type]
        changed_by=body.changed_by,
        opportunity_id=body.opportunity_id,
        outcome=body.outcome,  # type: ignore[arg-type]
        correlation_id=body.correlation_id,
        failure_reason=body.failure_reason,
        event_type=body.event_type,
        delivery_attempts=body.delivery_attempts,
        source_service=body.source_service,
        diff=body.diff,
        field_changes=body.field_changes,
    )
    if audit_id is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audit log table is not available.",
        )
    db.commit()
    return AuditEventIngestResponse(id=audit_id)


@router.get(
    "/audit-events",
    response_model=AuditEventListResponse,
    summary="Query audit events (compliance / ops)",
)
def list_audit_events(
    entity_type: str | None = Query(default=None, alias="entityType"),
    entity_id: str | None = Query(default=None, alias="entityId"),
    opportunity_id: str | None = Query(default=None, alias="opportunityId"),
    category: str | None = None,
    correlation_id: str | None = Query(default=None, alias="correlationId"),
    actor_type: str | None = Query(default=None, alias="actorType"),
    changed_by: str | None = Query(default=None, alias="changedBy"),
    from_at: datetime | None = Query(default=None, alias="from"),
    to_at: datetime | None = Query(default=None, alias="to"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    _: None = Depends(_verify_compliance_key),
) -> AuditEventListResponse:
    items, total = query_audit_events(
        db,
        entity_type=entity_type,
        entity_id=entity_id,
        opportunity_id=opportunity_id,
        category=category,
        correlation_id=correlation_id,
        actor_type=actor_type,
        changed_by=changed_by,
        from_at=from_at,
        to_at=to_at,
        limit=limit,
        offset=offset,
    )
    return AuditEventListResponse(total=total, items=items)


@router.get(
    "/audit-config",
    response_model=AuditConfigResponse,
    summary="Read audit retention and read-logging toggles",
)
def read_audit_config(
    db: Session = Depends(get_db),
    _: None = Depends(_verify_compliance_key),
) -> AuditConfigResponse:
    cfg = get_audit_config(db)
    return AuditConfigResponse(
        retention_days=cfg.retention_days,
        log_seller_reads=cfg.log_seller_reads,
        log_admin_reads=cfg.log_admin_reads,
        log_ai_output_reads=cfg.log_ai_output_reads,
        updated_at=cfg.updated_at,
        updated_by=cfg.updated_by,
    )


@router.patch(
    "/audit-config",
    response_model=AuditConfigResponse,
    summary="Update retention / read-logging toggles (no code deploy)",
)
def patch_audit_config(
    body: AuditConfigUpdateRequest,
    db: Session = Depends(get_db),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    _: None = Depends(_verify_compliance_key),
) -> AuditConfigResponse:
    try:
        cfg = update_audit_config(
            db,
            retention_days=body.retention_days,
            log_seller_reads=body.log_seller_reads,
            log_admin_reads=body.log_admin_reads,
            log_ai_output_reads=body.log_ai_output_reads,
            updated_by=x_user_id,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return AuditConfigResponse(
        retention_days=cfg.retention_days,
        log_seller_reads=cfg.log_seller_reads,
        log_admin_reads=cfg.log_admin_reads,
        log_ai_output_reads=cfg.log_ai_output_reads,
        updated_at=cfg.updated_at,
        updated_by=cfg.updated_by,
    )

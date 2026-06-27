"""Service layer for the data-hygiene task queue.

Sprint 1A · US04 — Data Hygiene, Validation & Intelligent Alerts

This module owns the task CRUD lifecycle. The detector logic that *decides
when to create a task* lives next door in
``app/services/data_task_detectors.py``; this file is purely about persisting,
querying, resolving, and dismissing tasks once a detector (or the AI team's
NLP pipeline, or an inline validator) has decided they're warranted.

Idempotency
-----------
``create_task`` is idempotent on ``(entity_kind, entity_id, task_kind)``
when the existing row is OPEN or DISMISSED. The function catches the
``IntegrityError`` from the partial UNIQUE index in
``sql/2026_06_us04_data_task.sql`` and returns the existing row. This
means the daily scan can be re-run as often as you like with zero
duplicate-task risk, AND a dismissed task continues to suppress
re-detection (AC #5).

Audit trail
-----------
We keep the audit on the row itself for S1A (resolved_at, dismissed_at,
dismissal_note, resolved_value, actor_id). When a unified
``lvo_audit_log``-style cross-cutting table arrives, we'll add a thin
adapter here without touching the API or service signatures.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple
from uuid import UUID

from sqlalchemy import case
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.data_task import DataTask
from app.schema.data_task import DataTaskCreateRequest
from app.services.compliance_audit import write_compliance_audit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CASE expressions used by ``list_tasks`` to implement
# "ORDER BY confidence DESC NULLS LAST, severity DESC, created_at ASC"
# in a way that works on both SQLite (test) and Postgres (prod).
#
# Postgres has native ``NULLS LAST`` and a built-in enum ordering, but
# SQLite has neither, so we project the labels to integers explicitly.
# Higher integer = comes first.
# ---------------------------------------------------------------------------
_CONFIDENCE_RANK = case(
    (DataTask.confidence == "high", 3),
    (DataTask.confidence == "medium", 2),
    (DataTask.confidence == "low", 1),
    else_=0,  # NULL / unknown sorts last
)

_SEVERITY_RANK = case(
    (DataTask.severity == "high", 3),
    (DataTask.severity == "medium", 2),
    (DataTask.severity == "low", 1),
    else_=0,
)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def create_task(
    db: Session,
    payload: DataTaskCreateRequest,
) -> Tuple[DataTask, bool]:
    """Persist a new data-hygiene task — or return the existing one.

    Returns
    -------
    (task, was_existing)
        ``was_existing=True`` iff there was already an OPEN or DISMISSED
        task with the same ``(entity_kind, entity_id, task_kind)``.

        For OPEN tasks: this is plain idempotency. The scan re-detected
        what's already on the queue.

        For DISMISSED tasks: this is AC #5 suppression. The seller
        previously dismissed this exact issue and we honour that
        decision — no new alert is generated.

    Concurrency
    -----------
    We use a SELECT-then-INSERT pattern with a defensive ``IntegrityError``
    catch on commit. The SELECT handles the common case cheaply (and works
    on SQLite for tests, which doesn't enforce the partial unique index
    that lives in the production SQL migration). The IntegrityError catch
    closes the race window where two concurrent callers both see "no
    existing row" and both attempt to insert — Postgres' partial UNIQUE
    index in ``sql/2026_06_us04_data_task.sql`` will reject the loser,
    and we return the winner's row instead of bubbling up the error.
    """
    existing = (
        db.query(DataTask)
        .filter(
            DataTask.entity_kind == payload.entity_kind,
            DataTask.entity_id == payload.entity_id,
            DataTask.task_kind == payload.task_kind,
            DataTask.status.in_(("open", "dismissed")),
        )
        .one_or_none()
    )
    if existing is not None:
        return existing, True

    new_task = DataTask(
        owner_id=payload.owner_id,
        entity_kind=payload.entity_kind,
        entity_id=payload.entity_id,
        task_kind=payload.task_kind,
        severity=payload.severity,
        confidence=payload.confidence,
        field_name=payload.field_name,
        current_value=payload.current_value,
        suggested_value=payload.suggested_value,
        evidence_ref=payload.evidence_ref,
        evidence_text=payload.evidence_text,
        created_by_source=payload.created_by_source,
    )
    db.add(new_task)
    try:
        db.commit()
    except IntegrityError:
        # Lost the race against a concurrent insert. Find the winner
        # and return it — caller sees was_existing=True.
        db.rollback()
        winner = (
            db.query(DataTask)
            .filter(
                DataTask.entity_kind == payload.entity_kind,
                DataTask.entity_id == payload.entity_id,
                DataTask.task_kind == payload.task_kind,
                DataTask.status.in_(("open", "dismissed")),
            )
            .one_or_none()
        )
        if winner is None:
            logger.error(
                "create_task IntegrityError but no matching row found "
                "(entity=%s/%s, kind=%s)",
                payload.entity_kind,
                payload.entity_id,
                payload.task_kind,
            )
            raise
        return winner, True

    db.refresh(new_task)
    return new_task, False


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def get_task(db: Session, task_id: UUID) -> Optional[DataTask]:
    """Single-row fetch. Returns None when not found."""
    return db.query(DataTask).filter(DataTask.task_id == task_id).one_or_none()


def list_tasks(
    db: Session,
    *,
    owner_id: Optional[UUID] = None,
    status: Optional[str] = None,
    task_kind: Optional[str] = None,
    entity_kind: Optional[str] = None,
    entity_id: Optional[UUID] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[list[DataTask], int]:
    """List tasks with optional filters, ordered by confidence then severity then age.

    AC #11 portfolio scoping is enforced at the call-site by passing the
    seller's ``owner_id``. We don't have user-auth context in this layer
    yet — ``owner_id=None`` returns ALL tasks (used by ops / admin only).

    Returns
    -------
    (items, total)
        ``total`` is the unpaginated count, so the FE can render
        "showing X of Y" correctly.
    """
    base = db.query(DataTask)
    if owner_id is not None:
        base = base.filter(DataTask.owner_id == owner_id)
    if status is not None:
        base = base.filter(DataTask.status == status)
    if task_kind is not None:
        base = base.filter(DataTask.task_kind == task_kind)
    if entity_kind is not None:
        base = base.filter(DataTask.entity_kind == entity_kind)
    if entity_id is not None:
        base = base.filter(DataTask.entity_id == entity_id)

    total = base.count()

    items = (
        base.order_by(
            _CONFIDENCE_RANK.desc(),
            _SEVERITY_RANK.desc(),
            DataTask.created_at.asc(),
        )
        .limit(limit)
        .offset(offset)
        .all()
    )
    return items, total


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


def resolve_task(
    db: Session,
    task: DataTask,
    *,
    actor_id: UUID,
    resolved_value: Optional[str],
) -> Tuple[DataTask, bool]:
    """Transition an OPEN task to RESOLVED.

    Idempotent: if already resolved, returns ``(task, True)`` without
    rewriting fields (we don't want a second resolver to overwrite the
    first actor's audit trail).

    Returns
    -------
    (task, was_already_resolved)

    Raises
    ------
    ValueError
        If the task is currently DISMISSED — dismissed tasks must be
        re-opened first (a flow we don't expose in S1A).
    """
    if task.status == "resolved":
        return task, True
    if task.status == "dismissed":
        # AC #5 says dismissed pairs are suppressed from future alerts.
        # That doesn't preclude re-opening, but we don't expose that
        # flow in S1A — so reject the transition explicitly.
        raise ValueError(
            f"Cannot resolve a dismissed task (task_id={task.task_id}); "
            "re-open via a new POST /ai-api/data-tasks first."
        )

    task.status = "resolved"
    task.resolved_at = datetime.now(timezone.utc)
    task.resolved_value = resolved_value
    task.actor_id = actor_id
    db.add(task)
    write_compliance_audit(
        db,
        entity_type="data_task",
        entity_id=str(task.task_id),
        action="resolve",
        category="seller_action",
        actor_type="seller",
        changed_by=str(actor_id),
        diff={
            "before": {"status": "open"},
            "after": {"status": "resolved", "resolvedValue": resolved_value},
        },
    )
    db.commit()
    db.refresh(task)
    return task, False


def dismiss_task(
    db: Session,
    task: DataTask,
    *,
    actor_id: UUID,
    note: str,
) -> Tuple[DataTask, bool]:
    """Transition an OPEN task to DISMISSED.

    Idempotent: if already dismissed, returns ``(task, True)`` without
    overwriting the original dismissal note.

    The DB has a CHECK constraint that enforces a non-empty note when
    status='dismissed'; we ALSO validate at this layer so the error
    message is friendly rather than a Postgres CHECK violation.

    Returns
    -------
    (task, was_already_dismissed)

    Raises
    ------
    ValueError
        On empty / whitespace note, or on attempt to dismiss a resolved task.
    """
    if not note or not note.strip():
        raise ValueError("Dismissal note must not be empty (AC #4).")

    if task.status == "dismissed":
        return task, True
    if task.status == "resolved":
        raise ValueError(
            f"Cannot dismiss a resolved task (task_id={task.task_id})."
        )

    task.status = "dismissed"
    task.dismissed_at = datetime.now(timezone.utc)
    task.dismissal_note = note.strip()
    task.actor_id = actor_id
    db.add(task)
    write_compliance_audit(
        db,
        entity_type="data_task",
        entity_id=str(task.task_id),
        action="dismiss",
        category="seller_action",
        actor_type="seller",
        changed_by=str(actor_id),
        diff={
            "before": {"status": "open"},
            "after": {"status": "dismissed", "dismissalNote": note.strip()},
        },
    )
    db.commit()
    db.refresh(task)
    return task, False

"""FastAPI router for the data-hygiene task queue.

Sprint 1A · US04 — Data Hygiene, Validation & Intelligent Alerts

Six routes (all JSON):

  POST   /ai-api/data-tasks                       # create (AI / scan / inline)
  GET    /ai-api/data-tasks                       # list with filters
  GET    /ai-api/data-tasks/{task_id}             # detail
  POST   /ai-api/data-tasks/{task_id}/resolve     # resolve + audit fields
  POST   /ai-api/data-tasks/{task_id}/dismiss     # dismiss with required note
  POST   /ai-api/data-tasks/scan                  # admin/manual trigger of daily scan

The scan endpoint imports lazily so the heavy ``app.jobs.scan_data_tasks``
module isn't pulled in for the FE happy path.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schema.data_task import (
    DataTaskCreateRequest,
    DataTaskCreateResponse,
    DataTaskDismissRequest,
    DataTaskDismissResponse,
    DataTaskListResponse,
    DataTaskOut,
    DataTaskResolveRequest,
    DataTaskResolveResponse,
    EntityKind,
    ScanRunRequest,
    ScanRunResponse,
    TaskStatus,
)
from app.services import data_task_service

router = APIRouter(
    prefix="/data-tasks",
    tags=["Data Hygiene Tasks"],
)


# ---------------------------------------------------------------------------
# POST /ai-api/data-tasks  — generic create endpoint
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=DataTaskCreateResponse,
    status_code=status.HTTP_200_OK,
    summary="Create a data-hygiene task (idempotent on entity+kind)",
)
def create_data_task(
    payload: DataTaskCreateRequest = Body(...),
    db: Session = Depends(get_db),
) -> DataTaskCreateResponse:
    """Create a task — or return the existing one for the same
    ``(entity_kind, entity_id, task_kind)`` if it's open OR dismissed.

    Returns
    -------
    200 OK with ``was_existing`` indicating which path was taken.
    We don't return 201 because callers (AI team, scan job) treat
    "newly created" and "already existed" the same way — both are
    success — and a 200 keeps the shape consistent.
    """
    task, was_existing = data_task_service.create_task(db, payload)
    return DataTaskCreateResponse(
        task=DataTaskOut.model_validate(task),
        was_existing=was_existing,
    )


# ---------------------------------------------------------------------------
# GET /ai-api/data-tasks  — list with filters
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=DataTaskListResponse,
    summary="List data-hygiene tasks (filtered + ordered)",
)
def list_data_tasks(
    owner_id: Optional[UUID] = Query(
        None,
        alias="ownerId",
        description=(
            "Restrict to one seller's portfolio. AC #11 portfolio scoping. "
            "Omit ONLY for admin / debug calls."
        ),
    ),
    task_status: Optional[TaskStatus] = Query(
        None,
        alias="status",
        description="Filter by status. The FE typically passes 'open' here.",
    ),
    task_kind: Optional[str] = Query(
        None,
        alias="kind",
        description="Filter by detector / signal kind (e.g. 'past_close_date').",
    ),
    entity_kind: Optional[EntityKind] = Query(
        None,
        alias="entityKind",
        description="Filter by entity type.",
    ),
    entity_id: Optional[UUID] = Query(
        None,
        alias="entityId",
        description=(
            "Filter to all tasks on a single record — used by the deal-detail page."
        ),
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> DataTaskListResponse:
    items, total = data_task_service.list_tasks(
        db,
        owner_id=owner_id,
        status=task_status,
        task_kind=task_kind,
        entity_kind=entity_kind,
        entity_id=entity_id,
        limit=limit,
        offset=offset,
    )
    return DataTaskListResponse(
        items=[DataTaskOut.model_validate(t) for t in items],
        total=total,
    )


# ---------------------------------------------------------------------------
# GET /ai-api/data-tasks/{task_id}  — detail
# ---------------------------------------------------------------------------


@router.get(
    "/{task_id}",
    response_model=DataTaskOut,
    summary="Fetch a single data-hygiene task by id",
)
def get_data_task(
    task_id: UUID = Path(...),
    db: Session = Depends(get_db),
) -> DataTaskOut:
    task = data_task_service.get_task(db, task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data task {task_id} not found.",
        )
    return DataTaskOut.model_validate(task)


# ---------------------------------------------------------------------------
# POST /ai-api/data-tasks/{task_id}/resolve
# ---------------------------------------------------------------------------


@router.post(
    "/{task_id}/resolve",
    response_model=DataTaskResolveResponse,
    summary="Resolve a data-hygiene task (writes audit fields)",
)
def resolve_data_task(
    task_id: UUID = Path(...),
    payload: DataTaskResolveRequest = Body(...),
    db: Session = Depends(get_db),
) -> DataTaskResolveResponse:
    task = data_task_service.get_task(db, task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data task {task_id} not found.",
        )
    try:
        task, was_already = data_task_service.resolve_task(
            db,
            task,
            actor_id=payload.actor_id,
            resolved_value=payload.resolved_value,
        )
    except ValueError as exc:
        # Triggered when the task is dismissed — re-open via a new POST first.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return DataTaskResolveResponse(
        task=DataTaskOut.model_validate(task),
        was_already_resolved=was_already,
    )


# ---------------------------------------------------------------------------
# POST /ai-api/data-tasks/{task_id}/dismiss
# ---------------------------------------------------------------------------


@router.post(
    "/{task_id}/dismiss",
    response_model=DataTaskDismissResponse,
    summary="Dismiss a data-hygiene task with a required note (AC #4)",
)
def dismiss_data_task(
    task_id: UUID = Path(...),
    payload: DataTaskDismissRequest = Body(...),
    db: Session = Depends(get_db),
) -> DataTaskDismissResponse:
    task = data_task_service.get_task(db, task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data task {task_id} not found.",
        )
    try:
        task, was_already = data_task_service.dismiss_task(
            db,
            task,
            actor_id=payload.actor_id,
            note=payload.note,
        )
    except ValueError as exc:
        # Empty note (covered by Pydantic too) or attempt to dismiss
        # an already-resolved task.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return DataTaskDismissResponse(
        task=DataTaskOut.model_validate(task),
        was_already_dismissed=was_already,
    )


# ---------------------------------------------------------------------------
# POST /ai-api/data-tasks/scan  — manual scan trigger
# ---------------------------------------------------------------------------


@router.post(
    "/scan",
    response_model=ScanRunResponse,
    summary="Trigger the data-hygiene scan immediately (admin / cron)",
)
def trigger_scan(
    payload: Optional[ScanRunRequest] = Body(default=None),
    db: Session = Depends(get_db),
) -> ScanRunResponse:
    """Synchronous scan trigger — runs detectors against every active opp
    visible to D365 and persists tasks accordingly.

    Lazy import keeps cold-start fast for the much more common FE path.

    The cron-driven CLI invocation
    (``python -m app.jobs.scan_data_tasks``) does the same thing without
    going through HTTP.
    """
    # Lazy import — pulls in d365_client + httpx work that the FE GETs
    # don't need.
    from app.jobs.scan_data_tasks import run_scan

    request = payload or ScanRunRequest()
    summary = run_scan(
        db,
        limit=request.limit,
        dry_run=request.dry_run,
    )
    return summary

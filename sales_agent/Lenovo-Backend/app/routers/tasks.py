"""Sprint 2 US 1.3 — Task Pending badge summary on the Home dashboard."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import TaskPendingSummaryResponse
from app.services.task_pending import (
    ERR_MSG_0022,
    TaskPendingSummaryData,
    build_task_pending_summary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


def _require_seller_id(seller_id: str | None) -> str:
    if not seller_id or not seller_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="sellerId is required.",
        )
    return seller_id.strip()


def _to_response(data: TaskPendingSummaryData) -> TaskPendingSummaryResponse:
    return TaskPendingSummaryResponse(
        seller_id=data.seller_id,
        count=data.count,
        overdue_count=data.overdue_count,
        due_today_count=data.due_today_count,
        has_overdue=data.has_overdue,
        badge_color=data.badge_color,
        label=data.label,
        last_updated_at=data.last_updated_at,
        source=data.source,
    )


@router.get(
    "/pending-summary",
    response_model=TaskPendingSummaryResponse,
    summary="Task Pending badge — open next-action count for the Home header",
)
def get_task_pending_summary(
    seller_id: str | None = Query(
        default=None,
        alias="sellerId",
        description="Seller UUID — matches opportunity.owninguser.",
    ),
    db: Session = Depends(get_db),
) -> TaskPendingSummaryResponse:
    seller = _require_seller_id(seller_id)
    try:
        return _to_response(build_task_pending_summary(db, seller))
    except Exception:
        logger.exception("Task pending summary failed for seller %s", seller)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERR_MSG_0022,
        ) from None

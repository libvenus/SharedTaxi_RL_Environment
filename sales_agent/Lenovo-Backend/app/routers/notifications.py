"""Sprint 2 US 1.1 — What Changed notification panel + activity timeline."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    ActivityTimelineResponse,
    NotificationMarkReadResponse,
    NotificationPanelResponse,
)
from app.services.what_changed import (
    ACTIVITY_TIMELINE_DEFAULT_PAGE_SIZE,
    ACTIVITY_TIMELINE_MAX_PAGE_SIZE,
    ERR_MSG_0020,
    NOTIFICATION_PANEL_DEFAULT_LIMIT,
    NOTIFICATION_PANEL_MAX_LIMIT,
    build_seller_feed,
    mark_notification_read,
    paginate_feed,
    parse_activity_type_filter,
)

logger = logging.getLogger(__name__)

notifications_router = APIRouter(prefix="/api/notifications", tags=["notifications"])
timeline_router = APIRouter(prefix="/api/activity-timeline", tags=["activity-timeline"])


def _require_seller_id(seller_id: str | None) -> str:
    if not seller_id or not seller_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="sellerId is required.",
        )
    return seller_id.strip()


@notifications_router.get(
    "",
    response_model=NotificationPanelResponse,
    summary="Recent What Changed items for the home notification panel",
)
def get_notification_panel(
    seller_id: str | None = Query(
        default=None,
        alias="sellerId",
        description="Seller UUID — matches opportunity.owninguser.",
    ),
    limit: int = Query(
        NOTIFICATION_PANEL_DEFAULT_LIMIT,
        ge=1,
        le=NOTIFICATION_PANEL_MAX_LIMIT,
        description="Panel shows up to 6 recent items.",
    ),
    types: str | None = Query(
        default=None,
        description="Comma-separated filter: email, meeting, crm_update, risk, task.",
    ),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> NotificationPanelResponse:
    """Return the 4–6 most recent portfolio events for the signed-in seller.

    Seller-initiated CRM audit changes are excluded so the seller does not
    receive notifications for their own edits.
    """
    seller = _require_seller_id(seller_id)
    try:
        feed = build_seller_feed(
            db,
            seller,
            viewer_id=x_user_id,
            exclude_self_changes=True,
            activity_types=parse_activity_type_filter(types),
        )
        items = feed[:limit]
        return NotificationPanelResponse(seller_id=seller, limit=limit, items=items)
    except HTTPException:
        raise
    except Exception:
        logger.exception("notification panel feed failed for seller=%s", seller)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERR_MSG_0020,
        ) from None


@notifications_router.patch(
    "/{notification_id}/read",
    response_model=NotificationMarkReadResponse,
    summary="Mark one notification feed item as read",
)
def patch_notification_read(
    notification_id: str = Path(
        ...,
        description="Synthetic feed key, e.g. activity:<uuid>.",
    ),
    seller_id: str | None = Query(default=None, alias="sellerId"),
    db: Session = Depends(get_db),
) -> NotificationMarkReadResponse:
    seller = _require_seller_id(seller_id)
    try:
        read_at = mark_notification_read(db, seller, notification_id)
        return NotificationMarkReadResponse(
            seller_id=seller,
            notification_id=notification_id,
            read_at=read_at,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "mark read failed seller=%s notification=%s", seller, notification_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERR_MSG_0020,
        ) from None


@timeline_router.get(
    "",
    response_model=ActivityTimelineResponse,
    summary="Paginated portfolio activity timeline for the home dashboard",
)
def get_activity_timeline(
    seller_id: str | None = Query(default=None, alias="sellerId"),
    page: int = Query(1, ge=1),
    page_size: int = Query(
        ACTIVITY_TIMELINE_DEFAULT_PAGE_SIZE,
        ge=1,
        le=ACTIVITY_TIMELINE_MAX_PAGE_SIZE,
        alias="pageSize",
    ),
    types: str | None = Query(
        default=None,
        description="Comma-separated filter: email, meeting, crm_update, risk, task.",
    ),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> ActivityTimelineResponse:
    """Full chronological feed across the seller's open opportunities."""
    seller = _require_seller_id(seller_id)
    try:
        feed = build_seller_feed(
            db,
            seller,
            viewer_id=x_user_id,
            exclude_self_changes=False,
            activity_types=parse_activity_type_filter(types),
        )
        page_items, total, total_pages = paginate_feed(feed, page, page_size)
        return ActivityTimelineResponse(
            seller_id=seller,
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
            items=page_items,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("activity timeline feed failed for seller=%s", seller)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERR_MSG_0020,
        ) from None

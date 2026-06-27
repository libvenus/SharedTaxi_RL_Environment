from uuid import UUID

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from typing import Optional
from datetime import date
from app.db.database import get_db
from fastapi import APIRouter, Request, Depends, HTTPException
from sqlalchemy.orm import Session
from app.services.meeting_details_service import (
    process_scheduler_response,
    upsert_meeting,
    get_meetings,
    get_matching_meetings,
    update_meeting_status,
    cancel_meeting,
)
from app.schema.schedulemeeting import (
    MeetingDetailsCreate,
    MeetingDetailsResponse,
    MeetingSearchRequest,
    MeetingStatusUpdate,
    MeetingStatusResponse,
)


router = APIRouter(prefix="/meeting-details", tags=["Meeting Details"])


@router.post("/")
async def ai_create_meeting(
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or missing JSON payload."
        )

    if not payload:
        raise HTTPException(
            status_code=400,
            detail="Request payload is required."
        )

    return process_scheduler_response(
        db,
        payload
    )


@router.get(
    "/",
    response_model=list[MeetingDetailsResponse]
)
def fetch_meetings(
    attendees: str,
    organiser_name: str,
    recurrence_start_date: date,
    title: Optional[str] = None,
    db: Session = Depends(get_db)
):
    return get_meetings(
        db=db,
        attendees=attendees,
        organiser_name=organiser_name,
        recurrence_start_date=recurrence_start_date,
        title=title
    )


@router.post("/meetings/search")
def ai_search_meetings(
    payload: MeetingSearchRequest,
    db: Session = Depends(get_db)
):
    return get_matching_meetings(
        db=db,
        payload=payload)


@router.patch(
    "/{meeting_id}/status",
    response_model=MeetingStatusResponse,
)
def patch_meeting_status(
    meeting_id: UUID,
    payload: MeetingStatusUpdate,
    db: Session = Depends(get_db),
):
    meeting = update_meeting_status(
        db=db,
        meeting_id=meeting_id,
        bot_status=payload.bot_status,
        reason=payload.reason,
    )
    return meeting


@router.delete(
    "/{meeting_id}",
    response_model=MeetingStatusResponse,
)
def delete_meeting(
    meeting_id: UUID,
    reason: Optional[str] = None,
    db: Session = Depends(get_db),
):
    meeting = cancel_meeting(db=db, meeting_id=meeting_id, reason=reason)
    return meeting

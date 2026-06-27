from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schema.meeting_briefing import (
    MeetingBriefingResponse,
    PrepNoteCreate,
    PrepNoteUpdate,
    PrepTaskStatusUpdate,
    VoiceNoteCreate,
)
from app.schema.meeting_prep import CreateMeetingRequest
from app.services.briefing_service import (
    ERR_MSG_0025,
    create_prep_note,
    delete_prep_note,
    generate_briefing,
    update_prep_note,
    update_prep_task_status,
)
from app.services.meeting_prep import create_meeting, get_upcoming_meetings, send_bot_to_meeting

router = APIRouter(
    prefix="/meeting-prep",
    tags=["Execute : Meeting Prep"],
)


def _require_seller_id(seller_id: str | None) -> str:
    if not seller_id or not seller_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="sellerId is required.",
        )
    return seller_id.strip()


@router.get("/meetings")
def upcoming_meetings(
    filter: str = Query(
        ...,
        description="all_meetings | today | tomorrow | this_week",
    ),
    seller_id: str | None = Query(default=None, alias="sellerId"),
    db: Session = Depends(get_db),
):
    try:
        return get_upcoming_meetings(db, filter, seller_id=seller_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get(
    "/{meeting_id}/briefing",
    response_model=MeetingBriefingResponse,
    summary="Pre-meeting briefing card (auto-generated on first open)",
)
def get_meeting_briefing(
    meeting_id: str,
    seller_id: str | None = Query(default=None, alias="sellerId"),
    refresh: bool = Query(default=False, description="Force regenerate briefing"),
    db: Session = Depends(get_db),
):
    seller = _require_seller_id(seller_id)
    try:
        payload = generate_briefing(
            db,
            meeting_id,
            seller,
            refresh=refresh,
        )
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERR_MSG_0025,
        ) from exc


@router.patch("/{meeting_id}/prep-tasks/{task_id}")
def patch_prep_task(
    meeting_id: str,
    task_id: int,
    body: PrepTaskStatusUpdate,
    seller_id: str | None = Query(default=None, alias="sellerId"),
    db: Session = Depends(get_db),
):
    seller = _require_seller_id(seller_id)
    return update_prep_task_status(
        db,
        meeting_id,
        seller,
        task_id,
        done=body.done,
    )


@router.get("/{meeting_id}/prep-notes")
def list_prep_notes(
    meeting_id: str,
    seller_id: str | None = Query(default=None, alias="sellerId"),
    db: Session = Depends(get_db),
):
    from app.services.briefing_service import (
        _load_meeting,
        _load_prep_notes,
        _parse_uuid,
        _verify_seller,
    )

    seller = _require_seller_id(seller_id)
    meeting = _load_meeting(db, meeting_id)
    seller_uuid = _verify_seller(meeting, seller)
    return _load_prep_notes(db, meeting_id, seller_uuid)


@router.post("/{meeting_id}/prep-notes")
def post_prep_note(
    meeting_id: str,
    body: PrepNoteCreate,
    seller_id: str | None = Query(default=None, alias="sellerId"),
    db: Session = Depends(get_db),
):
    seller = _require_seller_id(seller_id)
    return create_prep_note(
        db,
        meeting_id,
        seller,
        body=body.body,
        note_type=body.note_type,
    )


@router.post("/{meeting_id}/prep-notes/voice")
def post_voice_prep_note(
    meeting_id: str,
    body: VoiceNoteCreate,
    seller_id: str | None = Query(default=None, alias="sellerId"),
    db: Session = Depends(get_db),
):
    """Accepts transcript text — wire STT in AI/FE integration layer."""
    seller = _require_seller_id(seller_id)
    return create_prep_note(
        db,
        meeting_id,
        seller,
        body=body.transcript.strip(),
        note_type="voice_transcript",
    )


@router.patch("/{meeting_id}/prep-notes/{note_id}")
def patch_prep_note(
    meeting_id: str,
    note_id: int,
    body: PrepNoteUpdate,
    seller_id: str | None = Query(default=None, alias="sellerId"),
    db: Session = Depends(get_db),
):
    seller = _require_seller_id(seller_id)
    return update_prep_note(db, meeting_id, seller, note_id, body=body.body)


@router.delete("/{meeting_id}/prep-notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_prep_note(
    meeting_id: str,
    note_id: int,
    seller_id: str | None = Query(default=None, alias="sellerId"),
    db: Session = Depends(get_db),
):
    seller = _require_seller_id(seller_id)
    delete_prep_note(db, meeting_id, seller, note_id)


@router.post("")
def create_meeting_endpoint(
    payload: CreateMeetingRequest,
    db: Session = Depends(get_db),
):
    return create_meeting(payload=payload, db=db)


@router.post(
    "/{meeting_id}/join",
    summary="Send the Vexa bot to join a Teams meeting",
    responses={
        200: {"description": "Bot dispatched; vexa_bot_id stored on the meeting"},
        400: {"description": "Meeting has no join URL"},
        404: {"description": "Meeting not found"},
        502: {"description": "Vexa unreachable or returned an error"},
        503: {"description": "Vexa not configured on this server"},
    },
)
def join_meeting_with_bot(meeting_id: str, db: Session = Depends(get_db)):
    return send_bot_to_meeting(db=db, meeting_id=meeting_id)

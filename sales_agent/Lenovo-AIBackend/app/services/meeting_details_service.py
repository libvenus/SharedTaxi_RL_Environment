from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from dateutil import parser
from datetime import datetime
from app.models.schedulemeeting import MeetingDetails
from app.schema.schedulemeeting import MeetingSearchRequest
def parse_datetime(value):
    if not value:
        return None

    try:
        return parser.parse(str(value), dayfirst=True)
    except Exception:
        return None



def process_scheduler_response(db, response_data: dict):
    try:
        print(f"Processing scheduler response: {response_data}")
        actions = response_data.get("actions", [])
        print(f"Received actions: {actions}")

        for action in actions:
            print(f"Processing action: {action}")

            # if action.get("type") != "schedule_meeting":
            #     continue

            meeting_payload = action.get("payload", {})

            meeting_data = {
                "meeting_id": meeting_payload.get("meeting_id"),
                "meeting_start_time": datetime.fromisoformat(
                    meeting_payload["meeting_start_time"].replace("Z", "+00:00")
                ) if meeting_payload.get("meeting_start_time") else None,

                "meeting_end_time": datetime.fromisoformat(
                    meeting_payload["meeting_end_time"].replace("Z", "+00:00")
                ) if meeting_payload.get("meeting_end_time") else None,

                "platform": meeting_payload.get("platform"),
                "title": meeting_payload.get("title"),
                "account_name": meeting_payload.get("account_name"),
                "attendees_emails": meeting_payload.get("attendees"),
                "organiser_name": meeting_payload.get("organiser_name"),
                "action": meeting_payload.get("action"),
                "body": meeting_payload.get("body"),
                "meeting_url": meeting_payload.get("body"),
                "recurrence_pattern": meeting_payload.get("recurrence_pattern"),
                "recurrence_interval": meeting_payload.get("recurrence_interval"),
                "recurrence_start_date": meeting_payload.get("recurrence_start_date"),
                "recurrence_end_date": meeting_payload.get("recurrence_end_date"),
            }

            return upsert_meeting(db, meeting_data)

        return {"message": "No schedule_meeting action found"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


def upsert_meeting(
    db,
    meeting_data: dict
):
    """Insert-or-update a meeting row keyed on ``meeting_id``.

    The bot may POST the same meeting more than once — initially when the
    Outlook event is created, and again after the D365 resolver returns
    the matching ``opportunity_id`` / ``account_id``. The upsert pattern
    keeps both calls idempotent.

    NOTE: ``bot_status`` and ``bot_last_event_at`` are deliberately NOT
    overwritten by this upsert — those are owned by the lifecycle PATCH
    endpoint. Re-POSTing a meeting that's already ``joined`` should not
    silently revert it to ``pending``.
    """
    try:
       stmt = insert(MeetingDetails).values(**meeting_data)

       stmt = stmt.on_conflict_do_update(
        index_elements=["meeting_id"],
        set_={
            "meeting_start_time": stmt.excluded.meeting_start_time,
            "meeting_end_time": stmt.excluded.meeting_end_time,
            "platform": stmt.excluded.platform,
            "title": stmt.excluded.title,
            "account_name": stmt.excluded.account_name,
            "attendees_emails": stmt.excluded.attendees_emails,
            "organiser_name": stmt.excluded.organiser_name,
            "action": stmt.excluded.action,
            "body": stmt.excluded.body,
            "meeting_url": stmt.excluded.meeting_url,
            "recurrence_pattern": stmt.excluded.recurrence_pattern,
            "recurrence_interval": stmt.excluded.recurrence_interval,
            "recurrence_start_date": stmt.excluded.recurrence_start_date,
            "recurrence_end_date": stmt.excluded.recurrence_end_date,
            "updated_at": func.now(),
        }
    )

       db.execute(stmt)
       db.commit()

       return {"message": f"{meeting_data.get('action')} meeting  is done successfully."}

    except SQLAlchemyError as e:
       db.rollback()
 

       raise HTTPException(
        status_code=500,
        detail=f"Database error: {str(e)}"
    )

    except Exception as e:
       db.rollback()
    
       raise HTTPException(
        status_code=500,
        detail=f"Unexpected error: {str(e)}"
    )



def get_meetings(
    db: Session,
    attendees: str,
    organiser_name: str,
    recurrence_start_date,
    title: str = None
):
    query = db.query(MeetingDetails)

    query = query.filter(
        MeetingDetails.attendees.ilike(f"%{attendees}%")
    )

    query = query.filter(
        MeetingDetails.organiser_name.ilike(f"%{organiser_name}%")
    )

    query = query.filter(
        MeetingDetails.recurrence_start_date == recurrence_start_date
    )

    if title:
        query = query.filter(
            MeetingDetails.title.ilike(f"%{title}%")
        )

    return query.all()


# ---------------------------------------------------------------------------
# Sprint 1A · US01 — bot lifecycle helpers
# ---------------------------------------------------------------------------


def _ensure_meeting(db: Session, meeting_id: UUID) -> MeetingDetails:
    """Load a meeting or raise 404.

    Centralised so the PATCH and DELETE endpoints surface identical errors.
    """
    meeting = db.get(MeetingDetails, meeting_id)
    if meeting is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting '{meeting_id}' not found.",
        )
    return meeting


def update_meeting_status(
    db: Session,
    meeting_id: UUID,
    bot_status: str,
    reason: Optional[str] = None,
) -> MeetingDetails:
    """Transition a meeting to a new bot lifecycle state.

    Always stamps ``bot_last_event_at = now()`` so callers can audit when
    the transition happened. ``reason`` is overwritten on every call —
    we keep the most recent context only (the audit trail of all past
    transitions is intentionally out of scope for v1; revisit if compliance
    asks for it).

    Raises HTTPException(404) if the meeting doesn't exist. Pydantic
    enforces the ``bot_status`` whitelist before this is called, so we
    don't re-validate here.
    """
    meeting = _ensure_meeting(db, meeting_id)

    meeting.bot_status = bot_status
    meeting.bot_status_reason = reason
    meeting.bot_last_event_at = datetime.now(timezone.utc)

    db.add(meeting)
    db.commit()
    db.refresh(meeting)
    return meeting


def cancel_meeting(
    db: Session,
    meeting_id: UUID,
    reason: Optional[str] = None,
) -> MeetingDetails:
    """Soft-delete: flip the row's ``bot_status`` to ``'cancelled'``.

    The row itself is preserved so we keep an audit trail of cancelled
    meetings (and so transcripts that arrive late don't 404).
    """
    return update_meeting_status(
        db=db,
        meeting_id=meeting_id,
        bot_status="cancelled",
        reason=reason or "Meeting cancelled.",
    )


def get_matching_meetings(db, payload: MeetingSearchRequest):
    try:
        meeting_start_time = parse_datetime(payload.meeting_start_time)
       

        if not meeting_start_time:
            raise HTTPException(
                status_code=400,
                detail="Invalid meeting_start_time format"
            )

        query = (
            db.query(MeetingDetails)
            .filter(
                MeetingDetails.organiser_name == payload.organiser_name,
                MeetingDetails.meeting_start_time == meeting_start_time
            )
        )
        print(f"Querying meetings with organiser_name='{payload.organiser_name}' and meeting_start_time='{meeting_start_time}'")
        print(query.all())
        meetings = query.all()
     
        if payload.title:
            query = query.filter(
                MeetingDetails.title == payload.title
            )

        
        print(f"Found meetings: {len(meetings)}")
        

        if not meetings:
            raise HTTPException(
                status_code=404,
                detail="No matching meetings found"
            )
        email_list = [email.strip().lower() for email in payload.attendees_emails.replace(";", ",").split(",") if email.strip()]
        
        matching_rows = []

        for meeting in meetings:
            db_attendees = meeting.attendees_emails or ""
            db_email_list = [email.strip().lower() for email in db_attendees.replace(";", ",").split(",") if email.strip()]

           
            if set(db_email_list) == set(email_list):
                matching_rows.append(meeting)

        if not matching_rows:
            raise HTTPException(
                status_code=404,
                detail="No meeting found with matching attendees"
            )

        return matching_rows

    except HTTPException:
        raise

    except SQLAlchemyError as e:
       
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )

    except Exception as e:
       
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

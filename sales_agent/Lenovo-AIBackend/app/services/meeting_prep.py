import os
from datetime import datetime, timedelta, timezone
from uuid import UUID

import httpx
from sqlalchemy.orm import Session
from sqlalchemy import func, table, column
from app.schema.meeting_prep import CreateMeetingRequest
from app.models.schedulemeeting import MeetingDetails
import uuid
from app.graphapi.meetinginvite import (
    create_teams_meeting_invite,
    reschedule_teams_meeting_invite,
    cancel_teams_meeting_invite
)

VEXA_BOT_URL = os.getenv("VEXA_BOT_URL", "").rstrip("/")
VEXA_API_KEY = os.getenv("VEXA_API_KEY", "")
VEXA_CALLBACK_BASE_URL = os.getenv("VEXA_CALLBACK_BASE_URL", "").rstrip("/")


def get_upcoming_meetings(
    db: Session,
    filter_type: str,
    seller_id: str | None = None,
):
    now = datetime.now(timezone.utc)

    query = db.query(MeetingDetails).filter(
        MeetingDetails.bot_status != "completed"
    )
    
    # now = MeetingDetails.meeting_endtime if MeetingDetails.meeting_endtime else datetime.now() + timedelta(minutes=30)

    seller_uuid = None
    if seller_id and seller_id.strip():
        try:
            seller_uuid = UUID(seller_id.strip())
            query = query.filter(MeetingDetails.seller_id == seller_uuid)
        except ValueError:
            raise ValueError("sellerId must be a valid UUID.")

    if filter_type == "all_meetings":
        query = query.filter(
            MeetingDetails.meeting_start_time > now
        )
  


    elif filter_type == "today":
        query = query.filter(
            func.date(
                MeetingDetails.meeting_start_time
            ) == now.date()
        )

    elif filter_type == "tomorrow":
        tomorrow = now.date() + timedelta(days=1)

        query = query.filter(
            func.date(
                MeetingDetails.meeting_start_time
            ) == tomorrow
        )

    elif filter_type == "this_week":

        today = now.date()

        # Sunday -> Saturday
        days_since_sunday = (today.weekday() + 1) % 7

        start_of_week = today - timedelta(days=days_since_sunday)
        end_of_week = start_of_week + timedelta(days=6)

        query = query.filter(
            func.date(MeetingDetails.meeting_start_time) >= start_of_week
        ).filter(
            func.date(MeetingDetails.meeting_start_time) <= end_of_week
        ).filter(
            MeetingDetails.meeting_start_time > now
        )

    else:
        raise ValueError(
            "filter_type must be one of: "
            "all_meetings, today, tomorrow, this_week"
        )

    # Left join calendar_events on join_url = tbl_schedule_meetings.body to
    # pull passcode and join_meeting_id (meetings without a calendar_events
    # row still appear, with NULL passcode/join_meeting_id).
    calendar_events = table(
        "calendar_events",
        column("join_url"),
        column("passcode"),
        column("join_meeting_id"),
    )

    rows = (
        query
        .add_columns(
            calendar_events.c.passcode,
            calendar_events.c.join_meeting_id,
        )
        .outerjoin(
            calendar_events,
            calendar_events.c.join_url == MeetingDetails.body,
        )
        .order_by(MeetingDetails.meeting_start_time.asc())
        .all()
    )
    
    # Today's and tomorrow's upcoming meeting counts (independent of the
    # requested filter, but scoped to the same seller).
    today = now.date()
    tomorrow = today + timedelta(days=1)

    count_base = db.query(func.count(MeetingDetails.meeting_id))
    if seller_uuid is not None:
        count_base = count_base.filter(MeetingDetails.seller_id == seller_uuid)

    today_upcoming_count = count_base.filter(
        func.date(MeetingDetails.meeting_start_time) == today
    ).filter(
        MeetingDetails.meeting_start_time > now
    ).scalar()

    today_completed_count = count_base.filter(
        func.date(MeetingDetails.meeting_start_time) == today
    ).filter(
        MeetingDetails.meeting_start_time <= now
    ).scalar()

    tomorrow_meeting_count = count_base.filter(
        func.date(MeetingDetails.meeting_start_time) == tomorrow
    ).scalar()

    response = []
   

    for meeting, passcode, join_meeting_id in rows:
        if meeting.attendees_emails:
            attendees_list = [
                email.strip() for email in meeting.attendees_emails.split(",")
            ]
        else:
            attendees_list = []

        prep_pending = 0
        if meeting.seller_id:
            from app.services.briefing_service import count_open_prep_tasks

            prep_pending = count_open_prep_tasks(
                db, str(meeting.meeting_id), meeting.seller_id
            )

        response.append({
            "meeting_id": str(meeting.meeting_id),
            "meeting_start_time": meeting.meeting_start_time,
            "meeting_end_time": meeting.meeting_end_time,
            "platform": meeting.platform,
            "title": meeting.title,
            "account_name": meeting.account_name,
            "organiser_name": meeting.organiser_name,
            "attendees_emails": meeting.attendees_emails,
            "body": meeting.body,
            "attendee_count": len(attendees_list or []),
            "opportunity_id": str(meeting.opportunity_id) if meeting.opportunity_id else None,
            "account_id": str(meeting.account_id) if meeting.account_id else None,
            "meeting_url": meeting.meeting_url,
            "passcode": passcode,
            "join_meeting_id": join_meeting_id,
            "prep_tasks_pending_count": prep_pending,
        })

    return {
        "today_meeting_count": {
            "completed": today_completed_count or 0,
            "upcoming": today_upcoming_count or 0,
            "total": (today_completed_count or 0) + (today_upcoming_count or 0),
        },
        "tomorrow_meeting_count": tomorrow_meeting_count or 0,
        "meetings": response,
    }

def create_meeting(
    payload: CreateMeetingRequest,
    db: Session
):
    
    meeting_end = (
    payload.meeting_end_time
    if payload.meeting_end_time
    else payload.meeting_start_time + timedelta(minutes=30)
)
    seller_uuid = None
    if getattr(payload, "seller_id", None):
        seller_uuid = payload.seller_id

    url = payload.meeting_url or payload.body or None
    meeting = MeetingDetails(
        meeting_id=str(uuid.uuid4()),
        meeting_start_time=payload.meeting_start_time,
        meeting_end_time=payload.meeting_end_time or meeting_end,
        platform=payload.platform,
        title=payload.title,
        account_name=payload.account_name,
        attendees_emails=payload.attendees_emails,
        opportunity=payload.opportunity,
        meeting_url=url,
        body=url,
        prep_notes=payload.prep_notes,
        seller_id=seller_uuid,
    )
    
    db.add(meeting)
    db.commit()
    db.refresh(meeting)

    if payload.attendees_emails:
        try:
            graph_result = create_teams_meeting_invite(
                recipient_email=payload.attendees_emails,
                meeting_start=payload.meeting_start_time.strftime("%Y-%m-%d %H:%M"),
                meeting_end=meeting_end.strftime("%Y-%m-%d %H:%M"),
                timezone="Asia/Kolkata",
                meeting_subject=payload.title,
            )
            if isinstance(graph_result, dict) and graph_result.get("success"):
                if graph_result.get("join_url"):
                    meeting.meeting_url = graph_result["join_url"]
                    meeting.body = graph_result["join_url"]
                if graph_result.get("event_id"):
                    meeting.graph_event_id = graph_result["event_id"]
                db.commit()
                db.refresh(meeting)
        except Exception:
            pass

    return {
        "success": True,
        "message": "Meeting created successfully",
        "meeting_id": str(meeting.meeting_id),
        "join_url": meeting.meeting_url,
    }


def send_bot_to_meeting(db: Session, meeting_id: str) -> dict:
    """Proxy a bot-join request to Vexa, store the returned bot_id, and
    advance bot_status to 'scheduled'.  Returns the full Vexa response so
    the API layer can forward it to the client."""
    from fastapi import HTTPException

    meeting = db.get(MeetingDetails, meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id}' not found.")

    join_url = meeting.meeting_url or meeting.body
    if not join_url:
        raise HTTPException(
            status_code=400,
            detail="Meeting has no join URL. Create a Teams invite first.",
        )

    if not VEXA_BOT_URL or not VEXA_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Vexa bot not configured (VEXA_BOT_URL / VEXA_API_KEY missing).",
        )

    vexa_payload: dict = {
        "platform": meeting.platform or "teams",
        "meeting_url": join_url,
        "native_meeting_id": meeting.meeting_id,
        "bot_name": "Lenovo Sales Assistant",
    }

    # Only pass a webhook URL when the server is publicly reachable
    if VEXA_CALLBACK_BASE_URL and "localhost" not in VEXA_CALLBACK_BASE_URL:
        vexa_payload["webhook_url"] = f"{VEXA_CALLBACK_BASE_URL}/ai-api/vexa/webhook"

    try:
        resp = httpx.post(
            f"{VEXA_BOT_URL}/bots",
            headers={"x-api-key": VEXA_API_KEY, "Content-Type": "application/json"},
            json=vexa_payload,
            timeout=15.0,
        )
        resp.raise_for_status()
        vexa_data = resp.json() or {}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Vexa returned {exc.response.status_code}: {exc.response.text}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vexa unreachable: {exc}") from exc

    bot_id = vexa_data.get("bot_id") or vexa_data.get("id") or vexa_data.get("botId")
    if bot_id:
        meeting.vexa_bot_id = str(bot_id)
    meeting.bot_status = "scheduled"
    db.commit()
    db.refresh(meeting)

    return {
        "success": True,
        "meeting_id": meeting_id,
        "vexa_bot_id": meeting.vexa_bot_id,
        "bot_status": meeting.bot_status,
        "join_url": join_url,
        "vexa_response": vexa_data,
    }
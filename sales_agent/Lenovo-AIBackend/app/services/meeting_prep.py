import logging
import os
from datetime import datetime, timedelta, timezone
from uuid import UUID

log = logging.getLogger(__name__)

import httpx
from sqlalchemy.orm import Session
from sqlalchemy import func, table, column, text
from app.schema.meeting_prep import CreateMeetingRequest
from app.models.schedulemeeting import MeetingDetails
import uuid
from app.graphapi.meetinginvite import (
    create_teams_meeting_invite,
    reschedule_teams_meeting_invite,
    cancel_teams_meeting_invite,
    apply_lobby_bypass,
    _get_access_token,
    _ssl_verify,
)

VEXA_BOT_URL = os.getenv("VEXA_BOT_URL", "").rstrip("/")
VEXA_API_KEY = os.getenv("VEXA_API_KEY", "")
VEXA_CALLBACK_BASE_URL = os.getenv("VEXA_CALLBACK_BASE_URL", "").rstrip("/")

# Vexa bot status → our bot_status
_VEXA_STATUS_MAP = {
    "joining": "joining",
    "in_waiting_room": "lobby_waiting",
    "lobby_waiting": "lobby_waiting",
    "awaiting_admission": "lobby_waiting",
    "in_call_not_recording": "joining",
    "in_call_recording": "joined",
    "joined": "joined",
    "call_ended": "done",
    "done": "done",
    "fatal": "failed",
    "failed": "failed",
    "cancelled": "cancelled",
}


def _poll_vexa_bot(meeting_id: str, bot_id: str) -> None:
    """Background task: poll Vexa every 30 s, sync status + auto-fetch transcript."""
    import time
    from app.db.database import SessionLocal

    _POLL_INTERVAL = 30
    _MAX_POLLS = 480  # 4 hours max

    for _ in range(_MAX_POLLS):
        time.sleep(_POLL_INTERVAL)
        try:
            resp = httpx.get(
                f"{VEXA_BOT_URL}/bots/id/{bot_id}",
                headers={"x-api-key": VEXA_API_KEY},
                timeout=10.0,
            )
            if resp.status_code != 200:
                continue
            vexa = resp.json() or {}
        except Exception:
            continue

        vexa_status = vexa.get("status", "")
        our_status = _VEXA_STATUS_MAP.get(vexa_status, "")

        db = SessionLocal()
        try:
            meeting = db.get(MeetingDetails, meeting_id)
            if meeting is None:
                break
            if our_status and meeting.bot_status != our_status:
                meeting.bot_status = our_status
                meeting.bot_last_event_at = datetime.now(timezone.utc)
                db.commit()
                log.info("Poll updated meeting %s bot_status=%s (vexa=%s)", meeting_id, our_status, vexa_status)

            if vexa_status in ("call_ended", "done"):
                # Auto-fetch transcript
                try:
                    from app.api.vexa_webhook import _fetch_vexa_transcript
                    db.refresh(meeting)
                    _fetch_vexa_transcript(db, meeting, vexa)
                    log.info("Auto-fetched transcript for meeting %s after call end", meeting_id)
                except Exception as exc:
                    log.warning("Auto-fetch transcript failed for %s: %s", meeting_id, exc)
                break
            if vexa_status in ("fatal", "failed", "cancelled"):
                break
        except Exception as exc:
            log.warning("Poll error for meeting %s: %s", meeting_id, exc)
        finally:
            db.close()


def get_upcoming_meetings(
    db: Session,
    filter_type: str,
    seller_id: str | None = None,
):
    now = datetime.now(timezone.utc)

    _UUID_RE = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    query = db.query(MeetingDetails).filter(
        MeetingDetails.bot_status.notin_(["cancelled", "rescheduled"])
    ).filter(
        MeetingDetails.meeting_id.regexp_match(_UUID_RE, flags="i")
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
        pass  # no date filter — return all meetings regardless of time



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
            "bot_status": meeting.bot_status,
            "vexa_bot_id": meeting.vexa_bot_id,
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

    if payload.attendees_emails and not url:
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

    # Apply lobby bypass just before dispatching the bot, so any meeting URL works
    # regardless of whether the bypass was set at meeting-creation time.
    _GRAPH_TENANT = os.getenv("TENANT_ID", "")
    _GRAPH_CLIENT_ID = os.getenv("CLIENT_ID", "")
    _GRAPH_CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
    _GRAPH_USER_ID = os.getenv("USER_ID", "")
    if all([_GRAPH_TENANT, _GRAPH_CLIENT_ID, _GRAPH_CLIENT_SECRET, _GRAPH_USER_ID]):
        try:
            _token = _get_access_token(_GRAPH_TENANT, _GRAPH_CLIENT_ID, _GRAPH_CLIENT_SECRET)
            ok = apply_lobby_bypass(_token, _GRAPH_USER_ID, join_url)
            log.info("Lobby bypass before bot dispatch: %s for %s", "OK" if ok else "failed", meeting_id)
        except Exception as _lbe:
            log.warning("Lobby bypass error (non-fatal): %s", _lbe)

    vexa_payload: dict = {
        "meeting_url": join_url,
        "bot_name": "Lenovo Sales Assistant",
    }

    vexa_headers = {"x-api-key": VEXA_API_KEY, "Content-Type": "application/json"}

    # Vexa reads the webhook URL from the X-User-Webhook-URL header, not the body
    if VEXA_CALLBACK_BASE_URL and "localhost" not in VEXA_CALLBACK_BASE_URL and "127.0.0.1" not in VEXA_CALLBACK_BASE_URL:
        webhook_url = f"{VEXA_CALLBACK_BASE_URL}/ai-api/vexa/webhook"
        vexa_headers["X-User-Webhook-URL"] = webhook_url
        log.info("Sending webhook_url to Vexa: %s", webhook_url)
    else:
        log.warning("VEXA_CALLBACK_BASE_URL is localhost/unset — Vexa will NOT call back for transcripts.")

    try:
        resp = httpx.post(
            f"{VEXA_BOT_URL}/bots",
            headers=vexa_headers,
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


def fetch_and_store_vexa_transcript(db: Session, meeting_id: str) -> dict:
    """Pull the completed transcript from Vexa and persist it locally.

    Works by looking up the Vexa bot_id stored on the meeting, then calling
    GET /transcripts/{platform}/{native_meeting_id} on the Vexa API.
    """
    from fastapi import HTTPException

    meeting = db.get(MeetingDetails, meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id}' not found.")

    if not VEXA_BOT_URL or not VEXA_API_KEY:
        raise HTTPException(status_code=503, detail="Vexa not configured.")

    # Get the Vexa internal meeting record to find platform/native_meeting_id
    if not meeting.vexa_bot_id:
        raise HTTPException(status_code=404, detail="No Vexa bot associated with this meeting. Join the meeting first.")

    # Fetch the Vexa meeting record to get platform and native_meeting_id
    try:
        meta_resp = httpx.get(
            f"{VEXA_BOT_URL}/bots/id/{meeting.vexa_bot_id}",
            headers={"x-api-key": VEXA_API_KEY},
            timeout=10.0,
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vexa unreachable: {exc}") from exc

    platform = meta.get("platform", "teams")
    native_id = meta.get("native_meeting_id")
    vexa_status = meta.get("status", "")

    if not native_id:
        raise HTTPException(status_code=404, detail="Vexa has no native_meeting_id for this bot.")

    # Fetch transcript segments
    try:
        tx_resp = httpx.get(
            f"{VEXA_BOT_URL}/transcripts/{platform}/{native_id}",
            headers={"x-api-key": VEXA_API_KEY},
            timeout=30.0,
        )
        tx_resp.raise_for_status()
        vexa_transcript = tx_resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Vexa transcript returned {exc.response.status_code}: {exc.response.text[:200]}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vexa unreachable: {exc}") from exc

    segments_raw = vexa_transcript.get("segments") or []

    # Store via the webhook handler's transcript pipeline
    from app.api.vexa_webhook import _fetch_vexa_transcript
    _fetch_vexa_transcript(db, meeting, {
        "platform": platform,
        "native_meeting_id": native_id,
        "status": vexa_status,
    })

    return {
        "success": True,
        "meeting_id": meeting_id,
        "vexa_bot_id": meeting.vexa_bot_id,
        "vexa_status": vexa_status,
        "segments_found": len(segments_raw),
        "platform": platform,
        "native_meeting_id": native_id,
    }
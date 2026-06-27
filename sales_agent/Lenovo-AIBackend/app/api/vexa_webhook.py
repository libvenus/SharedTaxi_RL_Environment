"""Vexa bot webhook receiver.

Vexa calls this endpoint to report:
  - Bot status changes (joining → joined → call_ended → fatal)
  - Transcript segments (batched utterances during the meeting)

The bot_id in each event is looked up against tbl_schedule_meetings.vexa_bot_id
to find the matching meeting row.  Unknown bot_ids are silently ignored so
that a misconfigured bot can't crash the service.

Vexa status → our bot_status mapping:
  joining              → joining
  in_waiting_room      → lobby_waiting
  in_call_not_recording → joining
  in_call_recording    → joined
  call_ended           → joined   (transcript already closed by finalize)
  fatal                → failed
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.schedulemeeting import MeetingDetails
from app.services.meeting_details_service import update_meeting_status
from app.services.transcript_service import (
    append_segments,
    finalize_transcript,
    start_transcript,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/vexa", tags=["Vexa Bot Webhook"])

# Vexa raw status → our bot_status enum
_STATUS_MAP: dict[str, str] = {
    # Vexa raw status         → our bot_status
    "joining":                  "joining",
    "in_waiting_room":          "lobby_waiting",
    "lobby_waiting":            "lobby_waiting",
    "in_call_not_recording":    "joining",
    "in_call_recording":        "joined",
    "joined":                   "joined",
    "call_ended":               "joined",
    "done":                     "joined",
    "fatal":                    "failed",
    "failed":                   "failed",
    "cancelled":                "cancelled",
}

# Consent message sent by the bot in the Teams chat before recording starts.
# Vexa fires the webhook with transcript data AFTER the bot has already
# announced consent in-meeting, so we record that announcement here.
_BOT_CONSENT_TEXT = (
    "This meeting is being recorded and transcribed by the Lenovo Sales AI "
    "assistant. By continuing to participate you consent to this recording."
)


def _find_meeting(db: Session, bot_id: str) -> MeetingDetails | None:
    return (
        db.query(MeetingDetails)
        .filter(MeetingDetails.vexa_bot_id == bot_id)
        .first()
    )


def _handle_status(db: Session, meeting: MeetingDetails, raw_status: str) -> None:
    mapped = _STATUS_MAP.get(raw_status.lower())
    if mapped and mapped != meeting.bot_status:
        update_meeting_status(db=db, meeting_id=meeting.meeting_id, bot_status=mapped)
        log.info("Vexa status %s → bot_status=%s for meeting %s", raw_status, mapped, meeting.meeting_id)


def _handle_transcript(db: Session, meeting: MeetingDetails, data: dict[str, Any]) -> None:
    """Accept a Vexa transcript payload and feed it into the transcript pipeline."""
    import uuid as _uuid
    from app.models.transcript import MeetingTranscript

    meeting_id = meeting.meeting_id
    now = datetime.now(timezone.utc)

    # Transcript table requires a UUID meeting_id; skip for non-UUID rows
    # (e.g. seeded test meetings like "mtg-004")
    try:
        _uuid.UUID(str(meeting_id))
    except ValueError:
        log.debug("Skipping transcript for non-UUID meeting_id %s", meeting_id)
        return

    # Ensure a transcript row exists
    existing = db.query(MeetingTranscript).filter(MeetingTranscript.meeting_id == meeting_id).first()
    if existing is None:
        try:
            start_transcript(
                db=db,
                meeting_id=meeting_id,
                consent_message_text=_BOT_CONSENT_TEXT,
                consent_sent_at=now,
                started_at=now,
            )
        except Exception as exc:
            log.warning("Could not start transcript for %s: %s", meeting_id, exc)
            return

    # Normalise Vexa segment shapes into our schema
    raw_segments: list[dict] = data.get("transcript") or data.get("segments") or []
    if not raw_segments:
        return

    from app.schema.transcript import TranscriptSegmentInput
    from datetime import timedelta
    from decimal import Decimal

    def _secs_to_dt(secs: float) -> datetime:
        """Convert Vexa offset-seconds to an absolute datetime.
        If the value looks like a Unix timestamp (>1e9), use it directly.
        Otherwise treat it as seconds-since-meeting-start relative to now.
        """
        if secs > 1_000_000_000:
            return datetime.fromtimestamp(secs, tz=timezone.utc)
        return now + timedelta(seconds=secs)

    segments = []
    for seg in raw_segments:
        text = seg.get("text") or seg.get("utterance_text") or seg.get("words") or ""
        if isinstance(text, list):
            text = " ".join(w.get("word", "") for w in text)
        if not text:
            continue

        start_secs = float(seg.get("start_time") or seg.get("start") or 0)
        end_secs = float(seg.get("end_time") or seg.get("end") or start_secs + 1)
        if end_secs < start_secs:
            end_secs = start_secs + 0.001

        raw_conf = seg.get("confidence") or seg.get("confidence_score") or 1.0
        confidence = min(max(Decimal(str(raw_conf)), Decimal("0")), Decimal("1"))

        try:
            segments.append(
                TranscriptSegmentInput(
                    speaker_name=seg.get("speaker") or seg.get("speaker_name") or "Unknown Attendee",
                    utterance_text=text.strip(),
                    start_time=_secs_to_dt(start_secs),
                    end_time=_secs_to_dt(end_secs),
                    confidence_score=confidence,
                )
            )
        except Exception as exc:
            log.warning("Skipping invalid segment: %s — %s", seg, exc)

    if segments:
        try:
            append_segments(db=db, meeting_id=meeting_id, segments=segments)
            log.info("Appended %d segments for meeting %s", len(segments), meeting_id)
        except Exception as exc:
            log.warning("Could not append segments for %s: %s", meeting_id, exc)

    # Auto-finalize when Vexa signals the call has ended
    if data.get("status") in ("call_ended", "done"):
        try:
            finalize_transcript(
                db=db,
                meeting_id=meeting_id,
                overall_confidence_score=data.get("confidence") or 1.0,
                finalized_at=now,
            )
        except Exception:
            pass


@router.post(
    "/webhook",
    status_code=status.HTTP_200_OK,
    summary="Receive status and transcript events from the Vexa bot",
)
async def vexa_webhook(request: Request, db: Session = Depends(get_db)) -> dict:
    try:
        payload: dict = await request.json()
    except Exception:
        return {"ok": True, "note": "invalid json, ignored"}

    bot_id: str | None = (
        payload.get("bot_id")
        or payload.get("botId")
        or payload.get("id")
    )
    if not bot_id:
        return {"ok": True, "note": "no bot_id, ignored"}

    meeting = _find_meeting(db, str(bot_id))
    if meeting is None:
        log.debug("Vexa webhook: unknown bot_id %s", bot_id)
        return {"ok": True, "note": "unknown bot_id"}

    event_type: str = (
        payload.get("type")
        or payload.get("event")
        or ""
    ).lower()

    data: dict = payload.get("data") or {}

    if event_type in ("status_change", "bot_status", "bot.status_changed"):
        raw = data.get("status") or payload.get("status") or ""
        _handle_status(db, meeting, raw)

    elif event_type in ("transcript", "transcription", "transcript.segment"):
        _handle_transcript(db, meeting, data)
        # Some Vexa versions also carry a status in transcript events
        raw_status = data.get("status") or payload.get("status") or ""
        if raw_status:
            _handle_status(db, meeting, raw_status)

    else:
        # Unknown event — still update status if present at top level
        raw_status = payload.get("status") or data.get("status") or ""
        if raw_status:
            _handle_status(db, meeting, raw_status)

    return {"ok": True}

"""Transcript persistence — Sprint 1A · US02 (Consent & Recording).

Five service functions that the /transcripts/* router delegates to:

    start_transcript      — POST /transcripts/
    append_segments       — POST /transcripts/{meeting_id}/segments
    finalize_transcript   — POST /transcripts/{meeting_id}/finalize
    terminate_transcript  — POST /transcripts/{meeting_id}/terminate
    get_transcript        — GET  /transcripts/{meeting_id}

These are kept separate from the router so they can be unit-tested
without spinning up FastAPI, and so the call sites stay legible.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Tuple
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.schedulemeeting import MeetingDetails
from app.models.transcript import MeetingTranscript, MeetingTranscriptSegment
from app.schema.transcript import (
    TerminatableReason,
    TranscriptSegmentInput,
)


# ---------------------------------------------------------------------------
# Eligibility — bot must have moved the meeting to one of these states
# (via US01's PATCH /meeting-details/{id}/status) before a transcript can
# start. Anything else means "the bot isn't actually in the meeting yet".
# ---------------------------------------------------------------------------
_TRANSCRIBABLE_BOT_STATUSES = ("joined", "lobby_waiting", "joining")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_meeting(db: Session, meeting_id: UUID) -> MeetingDetails:
    """Load the meeting row or 404.
    tbl_schedule_meetings uses a TEXT primary key, so cast UUID → str.
    """
    meeting = db.get(MeetingDetails, str(meeting_id))
    if meeting is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting '{meeting_id}' not found.",
        )
    return meeting


def _load_transcript(db: Session, meeting_id: UUID) -> MeetingTranscript:
    """Load the transcript row by meeting_id (UNIQUE) or 404."""
    transcript = db.execute(
        select(MeetingTranscript).where(
            MeetingTranscript.meeting_id == meeting_id
        )
    ).scalar_one_or_none()
    if transcript is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No transcript exists for meeting '{meeting_id}'.",
        )
    return transcript


def _assert_in_progress(transcript: MeetingTranscript) -> None:
    """Guard for /append, /finalize, /terminate — they all require active state."""
    if transcript.status != "in_progress":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Transcript is in status '{transcript.status}'; only "
                f"'in_progress' transcripts accept further writes."
            ),
        )


# ---------------------------------------------------------------------------
# 1. start_transcript
# ---------------------------------------------------------------------------


def start_transcript(
    db: Session,
    meeting_id: UUID,
    consent_message_text: str,
    consent_sent_at: datetime,
    started_at: Optional[datetime] = None,
) -> MeetingTranscript:
    """Create the transcript row.

    Validations (in order — earliest 4xx wins):
      1. Meeting exists                              → 404
      2. Bot is actually in / approaching the meeting → 400
         (bot_status must be one of joined / lobby_waiting / joining)
      3. consent_sent_at <= started_at                → 422
         (consent must precede capture; AC #1)
      4. No existing transcript for this meeting      → 409
    """
    meeting = _ensure_meeting(db, meeting_id)

    if meeting.bot_status not in _TRANSCRIBABLE_BOT_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Cannot start transcript: meeting bot_status is "
                f"'{meeting.bot_status}'. Bot must be one of "
                f"{list(_TRANSCRIBABLE_BOT_STATUSES)} first "
                f"(use PATCH /meeting-details/{{id}}/status)."
            ),
        )

    effective_started_at = started_at or consent_sent_at
    if effective_started_at < consent_sent_at:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "started_at must be >= consent_sent_at — consent must be "
                "sent BEFORE audio capture begins."
            ),
        )

    # Single-transcript-per-meeting guarantee (also enforced by the UNIQUE
    # index, but we surface a clean 409 before hitting the DB).
    existing = db.execute(
        select(MeetingTranscript.transcript_id).where(
            MeetingTranscript.meeting_id == meeting_id
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"A transcript already exists for meeting '{meeting_id}'. "
                f"Use the existing transcript_id or terminate it first."
            ),
        )

    transcript = MeetingTranscript(
        meeting_id=meeting_id,
        opportunity_id=meeting.opportunity_id,
        account_id=meeting.account_id,
        status="in_progress",
        consent_message_text=consent_message_text,
        consent_sent_at=consent_sent_at,
        started_at=effective_started_at,
    )
    db.add(transcript)
    db.commit()
    db.refresh(transcript)
    return transcript


# ---------------------------------------------------------------------------
# 2. append_segments
# ---------------------------------------------------------------------------


def append_segments(
    db: Session,
    meeting_id: UUID,
    segments: List[TranscriptSegmentInput],
) -> Tuple[MeetingTranscript, int]:
    """Bulk-INSERT a batch of segments.

    Returns ``(transcript, appended_count)`` so the router can echo the
    new totals back to the bot.

    No dedup in v1 — bot is responsible for at-most-once delivery. If
    the bot retries on network errors and we end up with duplicates,
    the read-side ordering by ``start_time`` keeps the transcript
    coherent; we can add a dedup constraint in a later sprint if it
    becomes a real problem in dev.
    """
    transcript = _load_transcript(db, meeting_id)
    _assert_in_progress(transcript)

    if not segments:
        # Defensive — the Pydantic schema enforces min_length=1, but a
        # service-level guard keeps unit tests honest.
        return transcript, 0

    rows = [
        MeetingTranscriptSegment(
            transcript_id=transcript.transcript_id,
            meeting_id=meeting_id,
            speaker_name=seg.speaker_name,
            speaker_email=seg.speaker_email,
            speaker_role=seg.speaker_role,
            speaker_contact_id=seg.speaker_contact_id,
            utterance_text=seg.utterance_text,
            start_time=seg.start_time,
            end_time=seg.end_time,
            confidence_score=seg.confidence_score,
        )
        for seg in segments
    ]
    db.add_all(rows)

    appended = len(rows)
    transcript.segment_count = (transcript.segment_count or 0) + appended

    db.commit()
    db.refresh(transcript)
    return transcript, appended


# ---------------------------------------------------------------------------
# 3. finalize_transcript
# ---------------------------------------------------------------------------


def finalize_transcript(
    db: Session,
    meeting_id: UUID,
    overall_confidence_score: Decimal,
    finalized_at: Optional[datetime] = None,
) -> MeetingTranscript:
    """Clean meeting end — sets status='finalized'.

    Always writes ``terminated_reason='meeting_ended'`` so consumers can
    distinguish from /terminate paths via a single column.
    """
    transcript = _load_transcript(db, meeting_id)
    _assert_in_progress(transcript)

    transcript.status = "finalized"
    transcript.terminated_reason = "meeting_ended"
    transcript.overall_confidence_score = overall_confidence_score
    transcript.finalized_at = finalized_at or datetime.now(timezone.utc)

    db.commit()
    db.refresh(transcript)
    return transcript


# ---------------------------------------------------------------------------
# 4. terminate_transcript
# ---------------------------------------------------------------------------


def terminate_transcript(
    db: Session,
    meeting_id: UUID,
    reason: TerminatableReason,
    terminated_at: Optional[datetime] = None,
    overall_confidence_score: Optional[Decimal] = None,
) -> MeetingTranscript:
    """Early stop — organiser kicked the bot, all-left, or bot crashed.

    Sets status='terminated_partial' and stamps the reason. The transcript
    is preserved; downstream consumers can decide whether a partial
    transcript is useful (typically yes — even kicked-bot transcripts
    capture meaningful conversation).
    """
    transcript = _load_transcript(db, meeting_id)
    _assert_in_progress(transcript)

    transcript.status = "terminated_partial"
    transcript.terminated_reason = reason
    transcript.finalized_at = terminated_at or datetime.now(timezone.utc)
    if overall_confidence_score is not None:
        transcript.overall_confidence_score = overall_confidence_score

    db.commit()
    db.refresh(transcript)
    return transcript


# ---------------------------------------------------------------------------
# 5. get_transcript
# ---------------------------------------------------------------------------


def get_transcript(
    db: Session,
    meeting_id: UUID,
) -> Tuple[MeetingTranscript, List[MeetingTranscriptSegment]]:
    """Return transcript metadata + all segments ordered by start_time.

    No pagination in v1. A 2-hour meeting at 1 segment/second is ~7200
    rows — tolerable for a single payload. Revisit if we hit longer
    meetings or smaller windows.
    """
    transcript = _load_transcript(db, meeting_id)

    segments = db.execute(
        select(MeetingTranscriptSegment)
        .where(MeetingTranscriptSegment.meeting_id == meeting_id)
        .order_by(MeetingTranscriptSegment.start_time.asc())
    ).scalars().all()

    return transcript, list(segments)

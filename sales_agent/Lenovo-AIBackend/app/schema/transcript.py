"""Pydantic request/response schemas for the meeting-transcript endpoints.

Sprint 1A · US02 — Consent & Recording

Casing convention is **snake_case** throughout (matches the rest of this
service — `/meeting-details/`, `/activity-details/`).
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Lifecycle whitelists — kept in sync with the CHECK constraints in
# sql/2026_06_us02_meeting_transcript.sql AND with TRANSCRIPT_STATUS_VALUES
# in app/models/transcript.py. Edit ALL THREE together when adding a new value.
# ---------------------------------------------------------------------------
TranscriptStatus = Literal["in_progress", "finalized", "terminated_partial"]

TerminatedReason = Literal[
    "meeting_ended",
    "organizer_removed",
    "all_left",
    "bot_failure",
]

# The bot may legitimately call /finalize OR /terminate — but the union
# here is "things the BOT can write via /terminate" only. /finalize always
# writes 'meeting_ended'.
TerminatableReason = Literal["organizer_removed", "all_left", "bot_failure"]


# ---------------------------------------------------------------------------
# POST /transcripts/  — start a transcript
# ---------------------------------------------------------------------------


class TranscriptStartRequest(BaseModel):
    """Body for POST /transcripts/.

    Both consent fields are REQUIRED — without them the transcript can't
    exist. This is the server-side enforcement of AC #1 ("consent message
    sent before audio capture begins").
    """

    meeting_id: UUID = Field(
        description="The meeting whose transcript is starting. Must already exist in tbl_schedule_meetings."
    )
    consent_message_text: str = Field(
        min_length=1,
        description=(
            "The actual chat message the bot just sent (e.g. CONF_MSG_0004). "
            "Stored verbatim for compliance audit."
        ),
    )
    consent_sent_at: datetime = Field(
        description="When the bot posted the consent message in the Teams chat."
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description=(
            "When audio capture began. If omitted, defaults to consent_sent_at. "
            "Must be >= consent_sent_at (consent must precede capture)."
        ),
    )


# ---------------------------------------------------------------------------
# POST /transcripts/{meeting_id}/segments  — append batched utterances
# ---------------------------------------------------------------------------


class TranscriptSegmentInput(BaseModel):
    """A single utterance posted by the bot.

    ``speaker_name`` is required — bot must default to "Unknown Attendee"
    for unidentified speakers (per AC #3). ``speaker_email`` /
    ``speaker_role`` / ``speaker_contact_id`` are nullable for unknowns.
    """

    speaker_name: str = Field(min_length=1)
    speaker_email: Optional[str] = None
    speaker_role: Optional[str] = None
    speaker_contact_id: Optional[UUID] = None

    utterance_text: str = Field(min_length=1)
    start_time: datetime
    end_time: datetime
    confidence_score: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("1"),
        description="0.000–1.000 inclusive. Pydantic + DB CHECK both enforce.",
    )

    @field_validator("end_time")
    @classmethod
    def _end_after_start(cls, v: datetime, info) -> datetime:
        start = info.data.get("start_time")
        if start is not None and v < start:
            raise ValueError("end_time must be >= start_time")
        return v


class TranscriptSegmentsAppendRequest(BaseModel):
    """Body for POST /transcripts/{meeting_id}/segments.

    Bot batches at its discretion — minimum 1 segment per call. We don't
    cap the upper bound; if it becomes a problem, the bot can self-throttle.
    """

    segments: List[TranscriptSegmentInput] = Field(min_length=1)


class TranscriptSegmentsAppendResponse(BaseModel):
    """Echo + new totals so the bot can confirm what we accepted."""

    transcript_id: UUID
    meeting_id: UUID
    appended_count: int
    segment_count: int  # post-append total — useful for the bot's logging


# ---------------------------------------------------------------------------
# POST /transcripts/{meeting_id}/finalize  — clean meeting end
# ---------------------------------------------------------------------------


class TranscriptFinalizeRequest(BaseModel):
    """Body for POST /transcripts/{meeting_id}/finalize.

    Bot computes overall confidence (typically a weighted average of
    segment confidences); we just record what it sends. Finalisation is
    only valid from status='in_progress'.
    """

    overall_confidence_score: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Aggregate confidence across all segments, 0.000–1.000.",
    )
    finalized_at: Optional[datetime] = Field(
        default=None,
        description="When the meeting ended. Defaults to server now() if omitted.",
    )


# ---------------------------------------------------------------------------
# POST /transcripts/{meeting_id}/terminate  — early stop
# ---------------------------------------------------------------------------


class TranscriptTerminateRequest(BaseModel):
    """Body for POST /transcripts/{meeting_id}/terminate.

    Use this when the bot leaves before the natural meeting end — kicked
    by organiser, last human left, or the bot crashed. The transcript is
    saved as 'terminated_partial' so consumers can distinguish from a
    clean finalisation.

    For a clean meeting end, use /finalize instead — this endpoint
    intentionally does NOT accept ``meeting_ended`` as a reason.
    """

    reason: TerminatableReason
    terminated_at: Optional[datetime] = Field(
        default=None,
        description="When the bot stopped capturing. Defaults to server now() if omitted.",
    )
    overall_confidence_score: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("1"),
        description=(
            "Optional. Bot may include a partial confidence score, but the "
            "value is informational only — terminated_partial transcripts "
            "are not authoritative."
        ),
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class TranscriptSegmentResponse(BaseModel):
    """A single segment as returned by GET /transcripts/{meeting_id}."""

    model_config = ConfigDict(from_attributes=True)

    segment_id: UUID
    speaker_name: str
    speaker_email: Optional[str] = None
    speaker_role: Optional[str] = None
    speaker_contact_id: Optional[UUID] = None
    utterance_text: str
    start_time: datetime
    end_time: datetime
    confidence_score: Decimal


class TranscriptResponse(BaseModel):
    """Top-level transcript metadata.

    Returned by:
      - POST /transcripts/                        (without segments)
      - POST /transcripts/{meeting_id}/finalize
      - POST /transcripts/{meeting_id}/terminate
    """

    model_config = ConfigDict(from_attributes=True)

    transcript_id: UUID
    meeting_id: UUID
    opportunity_id: Optional[UUID] = None
    account_id: Optional[UUID] = None
    status: TranscriptStatus
    consent_message_text: str
    consent_sent_at: datetime
    overall_confidence_score: Optional[Decimal] = None
    segment_count: int
    terminated_reason: Optional[TerminatedReason] = None
    started_at: datetime
    finalized_at: Optional[datetime] = None


class TranscriptWithSegmentsResponse(BaseModel):
    """Full transcript view — used by GET /transcripts/{meeting_id}.

    Segments are returned ordered by ``start_time`` ascending. For very
    long meetings (>3000 segments) consider adding pagination later;
    not in 1A scope.
    """

    model_config = ConfigDict(from_attributes=True)

    transcript: TranscriptResponse
    segments: List[TranscriptSegmentResponse]

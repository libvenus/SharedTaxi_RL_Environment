"""ORM models for the meeting-transcript pipeline.

Sprint 1A · US02 — Consent & Recording

Two tables:
  - tbl_meeting_transcript          — one row per meeting (metadata + lifecycle)
  - tbl_meeting_transcript_segment  — one row per utterance

See sql/2026_06_us02_meeting_transcript.sql for the migration.
"""

import uuid

from sqlalchemy import Column, DateTime, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.database import Base


# ---------------------------------------------------------------------------
# Lifecycle whitelists — kept in sync with the CHECK constraints in
# sql/2026_06_us02_meeting_transcript.sql AND with the Pydantic Literal in
# app/schema/transcript.py. Edit ALL THREE together when adding a new value.
# ---------------------------------------------------------------------------
TRANSCRIPT_STATUS_VALUES = (
    "in_progress",         # bot is actively capturing
    "finalized",           # meeting ended cleanly; overall confidence set
    "terminated_partial",  # organiser removed bot, all-left, or bot crashed
)

TERMINATED_REASON_VALUES = (
    "meeting_ended",
    "organizer_removed",
    "all_left",
    "bot_failure",
)


class MeetingTranscript(Base):
    """One row per meeting that the bot is recording.

    Created by ``POST /transcripts/`` after the bot sends ``CONF_MSG_0004``
    in the Teams chat. ``consent_message_text`` and ``consent_sent_at`` are
    NOT NULL — a transcript literally cannot exist without compliance proof.

    The ``opportunity_id`` / ``account_id`` are denormalised from the
    parent ``tbl_schedule_meetings`` row at start time so the FE Activity
    tab can filter by opportunity without joining.
    """

    __tablename__ = "tbl_meeting_transcript"

    transcript_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    meeting_id = Column(UUID(as_uuid=True), nullable=False, unique=True)
    opportunity_id = Column(UUID(as_uuid=True))
    account_id = Column(UUID(as_uuid=True))

    status = Column(
        String(32),
        nullable=False,
        server_default="in_progress",
    )
    consent_message_text = Column(Text, nullable=False)
    consent_sent_at = Column(DateTime(timezone=True), nullable=False)

    overall_confidence_score = Column(Numeric(4, 3))
    segment_count = Column(Integer, nullable=False, server_default="0")

    terminated_reason = Column(String(32))
    started_at = Column(DateTime(timezone=True), nullable=False)
    finalized_at = Column(DateTime(timezone=True))

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class MeetingTranscriptSegment(Base):
    """One row per utterance within a meeting transcript.

    Append-only — the bot pushes batches via
    ``POST /transcripts/{meeting_id}/segments``. Speaker attribution is
    done by the bot before write (using the D365 contact resolver), so
    the backend stores whatever the bot sends. ``speaker_name`` defaults
    to "Unknown Attendee" for unidentified speakers — never a fabricated
    name.

    ``meeting_id`` is denormalised from the parent transcript so the
    primary read path (``meeting_id, start_time``) doesn't need a join.
    """

    __tablename__ = "tbl_meeting_transcript_segment"

    segment_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    transcript_id = Column(UUID(as_uuid=True), nullable=False)
    meeting_id = Column(UUID(as_uuid=True), nullable=False)

    speaker_name = Column(Text, nullable=False)
    speaker_email = Column(Text)
    speaker_role = Column(Text)
    speaker_contact_id = Column(UUID(as_uuid=True))

    utterance_text = Column(Text, nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=False)
    confidence_score = Column(Numeric(4, 3), nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

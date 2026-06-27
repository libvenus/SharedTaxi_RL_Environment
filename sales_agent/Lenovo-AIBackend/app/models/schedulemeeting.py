import uuid

from sqlalchemy import Column, String, Integer, Text, Date, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import TEXT, UUID, JSONB
from app.db.database import Base


# ---------------------------------------------------------------------------
# Bot lifecycle state machine — kept in sync with the CHECK constraint on
# tbl_schedule_meetings.bot_status (see sql/2026_06_us01_meeting_lifecycle.sql)
#
# Legal transitions:
#
#     pending ──► scheduled ──► joining ──► joined
#                       │            │
#                       │            └─► lobby_waiting ──► joined
#                       │
#                       ├─► rescheduled ──► (back to scheduled with new time)
#                       └─► cancelled (terminal)
#
# Any state may transition to:
#   - cancelled (calendar event deleted, OR participant opted out per US03)
#   - failed    (bot encountered an error)
#
# Canonical bot_status_reason values for cancellations:
#   - "participant_opted_out"   — written by app/services/consent_email_service.py
#                                 when a participant clicks the email opt-out link
#                                 (Sprint 1A · US03 · AC #7). The bot must NOT
#                                 join under any circumstances after this state.
#   - "Meeting cancelled."      — default for direct DELETE /meeting-details/{id}
#   - <free-form>               — bot-side errors, lobby timeouts, etc.
# ---------------------------------------------------------------------------
BOT_STATUS_VALUES = (
    "pending",
    "scheduled",
    "joining",
    "joined",
    "lobby_waiting",
    "cancelled",
    "rescheduled",
    "failed",
)


class MeetingDetails(Base):
    __tablename__ = "tbl_schedule_meetings"

    meeting_id = Column(TEXT,
        primary_key=True
        
    )

    meeting_start_time = Column(DateTime, nullable=False)
    meeting_end_time = Column(DateTime, nullable=False)

    platform = Column(String(100))
    title = Column(String(500))
    account_name = Column(String(255))

    attendees = Column(JSONB, nullable=True)

    organiser_name = Column(String(255))

    action = Column(Text)
    body = Column(Text)

    recurrence_pattern = Column(String(100))
    recurrence_interval = Column(Integer)

    recurrence_start_date = Column(Date)
    recurrence_end_date = Column(Date)

    # -----------------------------------------------------------------------
    # Sprint 1A · US01 — Joining the Meetings
    # Added by sql/2026_06_us01_meeting_lifecycle.sql
    # -----------------------------------------------------------------------
    bot_status = Column(
        String(32),
        nullable=False,
        server_default="pending",
    )
    bot_status_reason = Column(Text)
    bot_last_event_at = Column(DateTime(timezone=True))

    # Resolved by D365 /api/meetings/resolve-opportunity. Nullable until the
    # bot or FE supplies them — US01 may POST before resolution succeeds.
    opportunity_id = Column(UUID(as_uuid=True))
    account_id = Column(UUID(as_uuid=True))
    opportunity = Column(String(255))
    meeting_url = Column(Text)
    attendees_emails = Column(Text)
    prep_notes = Column(Text)
    seller_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    # Vexa bot ID returned by POST /bots — correlates webhook events to meetings
    vexa_bot_id = Column(Text, nullable=True, index=True)
    # Microsoft Graph calendar event ID — used for reschedule/cancel
    graph_event_id = Column(Text, nullable=True)

    created_at = Column(
        DateTime,
        server_default=func.now()
    )

    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now()
    )
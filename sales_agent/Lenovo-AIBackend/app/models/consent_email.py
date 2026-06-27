"""ORM model for the pre-meeting consent email pipeline.

Sprint 1A · US03 — Consent Capture

One table:
  - tbl_meeting_consent_email — one row per (meeting_id, recipient_email)

See sql/2026_06_us03_consent_email.sql for the migration.
"""

import uuid

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.database import Base


# ---------------------------------------------------------------------------
# Lifecycle whitelists — kept in sync with the CHECK constraints in
# sql/2026_06_us03_consent_email.sql AND with the Pydantic Literal in
# app/schema/consent_email.py. Edit ALL THREE together when adding a new value.
# ---------------------------------------------------------------------------
DELIVERY_STATUS_VALUES = (
    "pending",                  # row created; bot hasn't tried sending yet
    "sent",                     # delivery succeeded
    "failed",                   # last attempt failed; may retry
    "fallback_to_in_meeting",   # gave up after retries; bot uses US02 chat msg
)

# Canonical reason we write to tbl_schedule_meetings.bot_status_reason when
# an opt-out cascades to a US01 cancellation. Kept here so the consent
# service and the US01 docs reference the same string.
BOT_STATUS_REASON_OPT_OUT = "participant_opted_out"


class MeetingConsentEmail(Base):
    """One row per (meeting_id, recipient_email).

    Created by ``POST /consent-emails/schedule`` when a meeting becomes
    eligible AND the consent window is still open. The bot uses
    ``opt_out_token`` to build the link inside the email body; clicks
    on that link land on ``GET /consent-emails/opt-out/{token}``.

    The (meeting_id, recipient_email) pair is UNIQUE — re-scheduling
    is idempotent and returns the existing row+token, so the bot can
    safely retry the schedule call without spamming recipients.
    """

    __tablename__ = "tbl_meeting_consent_email"

    consent_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    meeting_id = Column(UUID(as_uuid=True), nullable=False)
    recipient_email = Column(Text, nullable=False)
    recipient_name = Column(Text)

    # 256 bits of url-safe randomness — see app/services/consent_email_service.py
    # for the generator. UNIQUE in the DB so collisions can't leak access.
    opt_out_token = Column(Text, nullable=False)

    scheduled_send_at = Column(DateTime(timezone=True), nullable=False)
    attempt_count = Column(Integer, nullable=False, server_default="0")
    last_attempt_at = Column(DateTime(timezone=True))
    next_retry_at = Column(DateTime(timezone=True))

    delivery_status = Column(
        String(32),
        nullable=False,
        server_default="pending",
    )
    failure_reason = Column(Text)

    opted_out_at = Column(DateTime(timezone=True))
    opt_out_ip = Column(Text)

    seller_notified_at = Column(DateTime(timezone=True))

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

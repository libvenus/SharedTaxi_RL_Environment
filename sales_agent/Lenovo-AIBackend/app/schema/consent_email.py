"""Pydantic request/response schemas for the consent-email pipeline.

Sprint 1A · US03 — Consent Capture

Casing convention is **snake_case** throughout (matches the rest of this
service — `/meeting-details/`, `/transcripts/`, `/activity-details/`).

The opt-out endpoint itself returns ``HTMLResponse`` (a server-rendered
confirmation page), not a JSON body, so there is no Pydantic schema for
its response. SUCC_MSG_0010 is rendered inline in the router.
"""

from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Lifecycle whitelists — kept in sync with the CHECK constraint in
# sql/2026_06_us03_consent_email.sql AND with DELIVERY_STATUS_VALUES in
# app/models/consent_email.py. Edit ALL THREE together when adding a new value.
# ---------------------------------------------------------------------------
DeliveryStatus = Literal[
    "pending",
    "sent",
    "failed",
    "fallback_to_in_meeting",
]

# What the bot may write via PATCH /delivery — strict subset of
# DeliveryStatus that excludes server-managed states ('pending' is the
# initial state; 'fallback_to_in_meeting' is set by the service after
# retry exhaustion).
DeliveryStatusInput = Literal["sent", "failed"]

# What /consent-status returns to the bot — derived state, not stored.
ConsentMechanism = Literal[
    "pre_meeting_email",   # at least one row, all sent / opted-out
    "in_meeting_chat",     # no rows OR all fell back to in-meeting
    "mixed",               # some sent, some fell back — bot must use chat too
]


# ---------------------------------------------------------------------------
# POST /consent-emails/schedule  — create per-recipient rows
# ---------------------------------------------------------------------------


class ConsentRecipientInput(BaseModel):
    """Per-attendee input the bot passes when scheduling.

    The bot pulls the attendee list from the Outlook event; we filter
    out internal Lenovo domains here on the backend (single source of
    truth — bot doesn't have to know what counts as "internal").
    """

    email: str = Field(min_length=3)
    name: Optional[str] = Field(
        default=None,
        description="Display name from the meeting invite, if available.",
    )


class ConsentScheduleRequest(BaseModel):
    """Body for POST /consent-emails/schedule.

    Bot calls this after a meeting becomes eligible (US01 resolved an
    opportunity / account). Backend decides whether the consent window
    is still open and either creates rows or signals fallback.
    """

    meeting_id: UUID = Field(
        description="The meeting being scheduled. Must already exist in tbl_schedule_meetings."
    )
    recipients: List[ConsentRecipientInput] = Field(
        min_length=1,
        description=(
            "All meeting attendees (internal + external). Backend filters "
            "out internal Lenovo domains using INTERNAL_EMAIL_DOMAINS config."
        ),
    )


class ConsentRecipientRecord(BaseModel):
    """Per-recipient row returned in the schedule response.

    Bot uses ``opt_out_url`` to build the email body. The full URL is
    pre-built server-side (using ``OPT_OUT_BASE_URL`` config) so the
    bot doesn't have to know the AIBackend's public hostname.
    """

    model_config = ConfigDict(from_attributes=True)

    consent_id: UUID
    recipient_email: str
    recipient_name: Optional[str] = None
    opt_out_token: str
    opt_out_url: str
    scheduled_send_at: datetime
    delivery_status: DeliveryStatus


class ConsentScheduleResponse(BaseModel):
    """Response from POST /consent-emails/schedule.

    Two shapes:

    - ``should_send=True``  + ``recipients=[…]``           — window open
    - ``should_send=False`` + ``recipients=[]``
      + ``fallback='in_meeting_chat'``                      — window passed,
                                                              bot uses US02

    The ``filtered_internal_count`` field tells the bot how many
    attendees we filtered out, so it can log "skipped 3 internal".
    """

    meeting_id: UUID
    should_send: bool
    fallback: Optional[Literal["in_meeting_chat", "meeting_started"]] = Field(
        default=None,
        description=(
            "Why the bot should NOT send. 'in_meeting_chat' = window passed; "
            "'meeting_started' = even tighter — the meeting has already begun."
        ),
    )
    recipients: List[ConsentRecipientRecord]
    filtered_internal_count: int = Field(
        ge=0,
        description="How many supplied recipients were filtered out as internal Lenovo domains.",
    )
    seller_name: Optional[str] = Field(
        default=None,
        description=(
            "Pulled from tbl_schedule_meetings.organiser_name — bot uses this in "
            "the 'From: <Seller Name> via Lenovo Sales Assistant' header (AC #4)."
        ),
    )
    system_email_address: Optional[str] = Field(
        default=None,
        description="The system 'from' address (SYSTEM_EMAIL_ADDRESS config).",
    )


# ---------------------------------------------------------------------------
# PATCH /consent-emails/{consent_id}/delivery  — bot reports send result
# ---------------------------------------------------------------------------


class ConsentDeliveryUpdateRequest(BaseModel):
    """Body for PATCH /consent-emails/{consent_id}/delivery.

    Bot calls this after attempting (or retrying) to send the email.
    Backend updates the row, increments ``attempt_count``, and decides
    whether to schedule a retry or mark the row as fallback.
    """

    status: DeliveryStatusInput = Field(
        description=(
            "Outcome of the send attempt. 'sent' = email accepted by SMTP "
            "/ Graph; 'failed' = bounce, invalid email, or system error."
        ),
    )
    failure_reason: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Free-form reason. Required if status='failed', ignored if 'sent'.",
    )
    attempted_at: Optional[datetime] = Field(
        default=None,
        description="When the send was attempted. Defaults to server now() if omitted.",
    )


class ConsentDeliveryUpdateResponse(BaseModel):
    """Echo + retry-state so the bot knows what to do next.

    If ``next_retry_at`` is non-null, the bot should poll
    ``GET /consent-emails/due-for-retry`` after that time and re-send.
    If ``delivery_status == 'fallback_to_in_meeting'``, the bot has
    exhausted retries and must use US02 instead.
    """

    model_config = ConfigDict(from_attributes=True)

    consent_id: UUID
    meeting_id: UUID
    recipient_email: str
    delivery_status: DeliveryStatus
    attempt_count: int
    last_attempt_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# GET /meetings/{meeting_id}/consent-status  — bot pre-join check
# ---------------------------------------------------------------------------


class MeetingConsentStatusResponse(BaseModel):
    """Aggregated consent state for a meeting.

    Bot reads this right before joining. AC #7: "If any single
    participant opts out, the bot does not join the meeting — no
    exceptions and no seller override." So the bot's logic is
    literally ``if response.any_opted_out: return``.

    ``consent_mechanism`` tells the bot how to introduce itself once
    inside (US02 chat announcement) — relevant when the email pipeline
    fell back to in-meeting consent.
    """

    meeting_id: UUID
    any_opted_out: bool
    opt_out_count: int
    total_recipients: int
    consent_mechanism: ConsentMechanism
    pending_retries: int = Field(
        ge=0,
        description=(
            "How many recipients are still waiting for a retry (delivery_status="
            "'failed' AND attempt_count<2 AND next_retry_at>now). Bot may want "
            "to wait a few minutes before joining if this is non-zero."
        ),
    )
    sent_count: int
    failed_count: int
    fallback_count: int


# ---------------------------------------------------------------------------
# GET /consent-emails/due-for-retry  — bot polls this periodically
# ---------------------------------------------------------------------------


class ConsentRetryQueueItem(BaseModel):
    """Single retry-pending row returned by /due-for-retry."""

    model_config = ConfigDict(from_attributes=True)

    consent_id: UUID
    meeting_id: UUID
    recipient_email: str
    recipient_name: Optional[str] = None
    opt_out_url: str
    attempt_count: int
    last_attempt_at: Optional[datetime] = None
    next_retry_at: datetime
    failure_reason: Optional[str] = None


class ConsentRetryQueueResponse(BaseModel):
    """Wrapper so we can extend with pagination later without breaking callers."""

    items: List[ConsentRetryQueueItem]


# ---------------------------------------------------------------------------
# GET /consent-emails/{meeting_id}  — audit / FE list view
# ---------------------------------------------------------------------------


class ConsentEmailRecord(BaseModel):
    """Full record as returned by the per-meeting list endpoint.

    Includes every field the audit / FE Activity tab might want to
    render — opt-out timestamp, IP, delivery history.
    """

    model_config = ConfigDict(from_attributes=True)

    consent_id: UUID
    meeting_id: UUID
    recipient_email: str
    recipient_name: Optional[str] = None
    opt_out_token: str
    opt_out_url: str
    scheduled_send_at: datetime
    attempt_count: int
    last_attempt_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    delivery_status: DeliveryStatus
    failure_reason: Optional[str] = None
    opted_out_at: Optional[datetime] = None
    opt_out_ip: Optional[str] = None
    seller_notified_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class ConsentEmailListResponse(BaseModel):
    """Per-meeting list response."""

    meeting_id: UUID
    items: List[ConsentEmailRecord]

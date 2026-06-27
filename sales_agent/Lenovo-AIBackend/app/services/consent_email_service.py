"""Pre-meeting consent email persistence.

Sprint 1A · US03 — Consent Capture

Service functions delegated to by app/api/consent_emails.py:

    schedule_consent_emails    — POST /consent-emails/schedule
    record_delivery_status     — PATCH /consent-emails/{id}/delivery
    record_opt_out             — GET  /consent-emails/opt-out/{token}
    get_consent_status         — GET  /meetings/{id}/consent-status
    list_consent_emails        — GET  /consent-emails/{meeting_id}
    due_for_retry              — GET  /consent-emails/due-for-retry

Plus internal helpers:

    is_internal_email          — domain check against INTERNAL_EMAIL_DOMAINS
    generate_opt_out_token     — secrets.token_urlsafe(32)
    build_opt_out_url          — token → public URL using OPT_OUT_BASE_URL
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import (
    AIBACKEND_API_PREFIX,
    CONSENT_RETRY_DELAY_MINUTES,
    CONSENT_WINDOW_MINUTES,
    INTERNAL_EMAIL_DOMAINS,
    OPT_OUT_BASE_URL,
    SYSTEM_EMAIL_ADDRESS,
)
from app.models.consent_email import (
    BOT_STATUS_REASON_OPT_OUT,
    MeetingConsentEmail,
)
from app.models.schedulemeeting import MeetingDetails
from app.schema.consent_email import (
    ConsentMechanism,
    ConsentRecipientInput,
    DeliveryStatusInput,
)
from app.services.meeting_details_service import _ensure_meeting, cancel_meeting


# Cap on attempts (initial + 1 retry, per AC #10). Mirrored in the DB
# CHECK constraint and the model whitelist.
_MAX_ATTEMPTS = 2


# ===========================================================================
# Internal helpers
# ===========================================================================


def is_internal_email(email: str) -> bool:
    """True if the email domain is in INTERNAL_EMAIL_DOMAINS.

    AC #1: only EXTERNAL customer attendees receive the email. We compare
    case-insensitively because some calendar feeds upper-case the domain.
    """
    if "@" not in email:
        # Defensive — Pydantic already rejects these but cheap to handle here.
        return False
    domain = email.rsplit("@", 1)[-1].strip().lower()
    return domain in INTERNAL_EMAIL_DOMAINS


def generate_opt_out_token() -> str:
    """Cryptographically random URL-safe token, ~43 chars long.

    `secrets.token_urlsafe(32)` produces 256 bits of randomness, enough
    that a brute-force attacker would have to make ~2^128 guesses to
    find any one valid token. Way over our compliance bar.
    """
    return secrets.token_urlsafe(32)


def build_opt_out_url(token: str) -> str:
    """Compose the full URL the participant clicks in the email."""
    return f"{OPT_OUT_BASE_URL}{AIBACKEND_API_PREFIX}/consent-emails/opt-out/{token}"


def _normalise_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Make a datetime timezone-aware, defaulting naive values to UTC.

    `tbl_schedule_meetings.meeting_start_time` is `DateTime` (no tz) on
    purpose — the existing US01 schema keeps things simple — but the
    consent-window math has to be timezone-correct. We treat naive
    timestamps as UTC, which matches how the bot writes them.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _now_utc() -> datetime:
    """A single, mockable place to pin "current time"."""
    return datetime.now(timezone.utc)


# ===========================================================================
# 1. schedule_consent_emails
# ===========================================================================


def schedule_consent_emails(
    db: Session,
    meeting_id: UUID,
    recipients: List[ConsentRecipientInput],
) -> dict:
    """Create per-recipient consent rows IF the consent window is still open.

    Returns a dict shaped to ``ConsentScheduleResponse``:

        {
            "meeting_id": ...,
            "should_send": True/False,
            "fallback":   None | "in_meeting_chat" | "meeting_started",
            "recipients": [...],
            "filtered_internal_count": ...,
            "seller_name": ...,
            "system_email_address": ...,
        }

    Behaviour:
      1. 404 if the meeting doesn't exist.
      2. Filter out recipients with internal Lenovo domains (AC #1).
      3. If meeting is in the past or already started → ``fallback='meeting_started'``,
         no rows created (US02 in-meeting chat is the only consent mechanism left).
      4. If meeting is within ``CONSENT_WINDOW_MINUTES`` → ``fallback='in_meeting_chat'``,
         no rows created (AC #3).
      5. Otherwise, upsert one row per external recipient with a fresh token
         AND ``scheduled_send_at = meeting_start - consent_window``.

    Idempotent — re-calling for the same meeting returns existing tokens
    so the bot doesn't double-send. New recipients (e.g. someone added to
    the calendar invite) get fresh rows on a re-call.
    """
    meeting = _ensure_meeting(db, meeting_id)

    seller_name = meeting.organiser_name
    meeting_start = _normalise_aware(meeting.meeting_start_time)
    if meeting_start is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Meeting '{meeting_id}' has no meeting_start_time set; cannot schedule consent.",
        )

    # ---- Filter internal domains (AC #1) ----------------------------------
    external: list[ConsentRecipientInput] = []
    filtered_internal = 0
    seen_emails: set[str] = set()
    for r in recipients:
        email = r.email.strip().lower()
        if not email or "@" not in email:
            # Bot sent something nonsensical — drop silently rather than 422
            # the entire batch. Pydantic should already block this.
            continue
        if email in seen_emails:
            continue
        seen_emails.add(email)
        if is_internal_email(email):
            filtered_internal += 1
            continue
        external.append(
            ConsentRecipientInput(email=email, name=r.name)
        )

    base_response: dict = {
        "meeting_id": meeting_id,
        "filtered_internal_count": filtered_internal,
        "seller_name": seller_name,
        "system_email_address": SYSTEM_EMAIL_ADDRESS,
    }

    # ---- Window checks (AC #3) --------------------------------------------
    now = _now_utc()
    delta = meeting_start - now
    window = timedelta(minutes=CONSENT_WINDOW_MINUTES)

    if delta <= timedelta(0):
        # Meeting is in the past / has started. AC #3: bot uses US02 only.
        return {
            **base_response,
            "should_send": False,
            "fallback": "meeting_started",
            "recipients": [],
        }

    if delta < window:
        # Window has passed. AC #3: skip email; bot uses US02.
        return {
            **base_response,
            "should_send": False,
            "fallback": "in_meeting_chat",
            "recipients": [],
        }

    # ---- All-internal short-circuit ---------------------------------------
    # No-one to send to → nothing to do, but still NOT a fallback case
    # (the bot can use US02 if this happens, but typically all-internal
    # meetings shouldn't even get to this endpoint).
    if not external:
        return {
            **base_response,
            "should_send": False,
            "fallback": "in_meeting_chat",
            "recipients": [],
        }

    # ---- Upsert per-recipient rows ----------------------------------------
    scheduled_send_at = meeting_start - window

    persisted_rows: list[MeetingConsentEmail] = []
    for recipient in external:
        # Look up existing row (idempotency on (meeting_id, recipient_email)).
        existing = db.execute(
            select(MeetingConsentEmail).where(
                MeetingConsentEmail.meeting_id == meeting_id,
                MeetingConsentEmail.recipient_email == recipient.email,
            )
        ).scalar_one_or_none()

        if existing is not None:
            # Don't issue a new token — that would invalidate the link
            # already in the recipient's inbox. Just refresh the
            # scheduled_send_at in case the meeting was rescheduled.
            existing.scheduled_send_at = scheduled_send_at
            if recipient.name and not existing.recipient_name:
                existing.recipient_name = recipient.name
            persisted_rows.append(existing)
            continue

        row = MeetingConsentEmail(
            meeting_id=meeting_id,
            recipient_email=recipient.email,
            recipient_name=recipient.name,
            opt_out_token=generate_opt_out_token(),
            scheduled_send_at=scheduled_send_at,
            delivery_status="pending",
        )
        db.add(row)
        persisted_rows.append(row)

    db.commit()
    for row in persisted_rows:
        db.refresh(row)

    return {
        **base_response,
        "should_send": True,
        "fallback": None,
        "recipients": [
            {
                "consent_id": row.consent_id,
                "recipient_email": row.recipient_email,
                "recipient_name": row.recipient_name,
                "opt_out_token": row.opt_out_token,
                "opt_out_url": build_opt_out_url(row.opt_out_token),
                "scheduled_send_at": row.scheduled_send_at,
                "delivery_status": row.delivery_status,
            }
            for row in persisted_rows
        ],
    }


# ===========================================================================
# 2. record_delivery_status
# ===========================================================================


def _ensure_consent(db: Session, consent_id: UUID) -> MeetingConsentEmail:
    """Load a consent row by PK or 404."""
    row = db.get(MeetingConsentEmail, consent_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Consent record '{consent_id}' not found.",
        )
    return row


def record_delivery_status(
    db: Session,
    consent_id: UUID,
    new_status: DeliveryStatusInput,
    failure_reason: Optional[str] = None,
    attempted_at: Optional[datetime] = None,
) -> MeetingConsentEmail:
    """Record the bot's send-attempt outcome.

    On 'sent' (AC: success path):
      - delivery_status='sent'
      - attempt_count incremented
      - last_attempt_at stamped
      - next_retry_at cleared

    On 'failed' (AC #10: retry-once-then-fallback):
      - if attempt_count was 0 → schedule retry at now+10min, status='failed'
      - if attempt_count was >=1 → status='fallback_to_in_meeting' (no more
        retries — bot uses US02 in-meeting chat as the consent mechanism)

    Already-opted-out rows reject further delivery updates with 400
    (an opt-out has cancelled the bot dispatch — sending more email is
    pointless and looks like a bug).
    """
    row = _ensure_consent(db, consent_id)

    if row.opted_out_at is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Consent record '{consent_id}' has been opted out; "
                "delivery updates are no longer accepted."
            ),
        )

    if row.delivery_status == "fallback_to_in_meeting":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Consent record '{consent_id}' has exhausted retries "
                "and fallen back to in-meeting consent."
            ),
        )

    now = _now_utc()
    row.last_attempt_at = attempted_at or now
    row.attempt_count = (row.attempt_count or 0) + 1

    if new_status == "sent":
        row.delivery_status = "sent"
        row.failure_reason = None
        row.next_retry_at = None
    else:  # 'failed'
        row.failure_reason = failure_reason
        if row.attempt_count >= _MAX_ATTEMPTS:
            row.delivery_status = "fallback_to_in_meeting"
            row.next_retry_at = None
        else:
            row.delivery_status = "failed"
            row.next_retry_at = now + timedelta(minutes=CONSENT_RETRY_DELAY_MINUTES)

    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ===========================================================================
# 3. record_opt_out
# ===========================================================================


def record_opt_out(
    db: Session,
    token: str,
    client_ip: Optional[str] = None,
) -> Tuple[MeetingConsentEmail, MeetingDetails]:
    """Process a click on the email's opt-out link.

    Behaviour:
      - 404 if the token doesn't match any row
      - 410 Gone if the meeting has already started (link expired,
        AC #6: "active until meeting starts — not after")
      - Idempotent — second click returns the same row without re-stamping
        timestamps; AC #8 still satisfied (we re-render SUCC_MSG_0010)
      - On first opt-out: cascades to bot_status='cancelled' on the
        parent meeting (AC #7: bot does not join under any circumstances)
      - Stamps seller_notified_at so a future notifications service can
        pick up the INF_MSG_0001 event (AC #9)

    Returns ``(consent_row, meeting_row)`` so the router can render the
    confirmation HTML with meeting context.
    """
    row = db.execute(
        select(MeetingConsentEmail).where(
            MeetingConsentEmail.opt_out_token == token
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid opt-out link.",
        )

    meeting = _ensure_meeting(db, row.meeting_id)
    meeting_start = _normalise_aware(meeting.meeting_start_time)
    if meeting_start is not None and _now_utc() >= meeting_start:
        # Link has expired (AC #6). 410 Gone is the right code; the
        # router renders a friendly "this link is no longer valid" page.
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="The opt-out window for this meeting has closed.",
        )

    # Idempotent — already opted out → just return the existing rows
    # (router will render SUCC_MSG_0010 again).
    if row.opted_out_at is not None:
        return row, meeting

    # First opt-out: stamp + cascade.
    now = _now_utc()
    row.opted_out_at = now
    row.opt_out_ip = client_ip
    row.seller_notified_at = now

    db.add(row)
    db.commit()
    db.refresh(row)

    # Cascade to US01 lifecycle. Use the existing cancel_meeting helper
    # so the audit trail (bot_last_event_at + bot_status_reason) is
    # written by exactly one piece of code.
    if meeting.bot_status != "cancelled":
        meeting = cancel_meeting(
            db=db,
            meeting_id=row.meeting_id,
            reason=BOT_STATUS_REASON_OPT_OUT,
        )

    return row, meeting


# ===========================================================================
# 4. get_consent_status
# ===========================================================================


def _derive_mechanism(
    sent_count: int,
    fallback_count: int,
    total_recipients: int,
) -> ConsentMechanism:
    """Decide which consent mechanism the bot should use in-meeting.

    - All recipients sent successfully → 'pre_meeting_email'
    - All recipients fell back → 'in_meeting_chat'
    - No recipients on the meeting → 'in_meeting_chat'
      (e.g. all-internal meeting; bot still introduces itself in chat)
    - Mix of sent + fallback → 'mixed' (bot uses chat as belt-and-braces)
    """
    if total_recipients == 0:
        return "in_meeting_chat"
    if fallback_count == total_recipients:
        return "in_meeting_chat"
    if sent_count == total_recipients:
        return "pre_meeting_email"
    return "mixed"


def get_consent_status(
    db: Session,
    meeting_id: UUID,
) -> dict:
    """Return aggregated consent state for a meeting.

    Cheap to compute (single table, partial indexes). Bot calls this
    right before joining; AC #7 means ``any_opted_out=True`` is a hard
    "do not join".
    """
    _ensure_meeting(db, meeting_id)

    rows = db.execute(
        select(MeetingConsentEmail).where(
            MeetingConsentEmail.meeting_id == meeting_id
        )
    ).scalars().all()

    total = len(rows)
    opt_out_count = sum(1 for r in rows if r.opted_out_at is not None)
    sent_count = sum(1 for r in rows if r.delivery_status == "sent")
    failed_count = sum(1 for r in rows if r.delivery_status == "failed")
    fallback_count = sum(
        1 for r in rows if r.delivery_status == "fallback_to_in_meeting"
    )

    now = _now_utc()
    pending_retries = sum(
        1
        for r in rows
        if r.delivery_status == "failed"
        and (r.attempt_count or 0) < _MAX_ATTEMPTS
        and r.next_retry_at is not None
        and _normalise_aware(r.next_retry_at) > now
    )

    return {
        "meeting_id": meeting_id,
        "any_opted_out": opt_out_count > 0,
        "opt_out_count": opt_out_count,
        "total_recipients": total,
        "consent_mechanism": _derive_mechanism(sent_count, fallback_count, total),
        "pending_retries": pending_retries,
        "sent_count": sent_count,
        "failed_count": failed_count,
        "fallback_count": fallback_count,
    }


# ===========================================================================
# 5. due_for_retry
# ===========================================================================


def due_for_retry(
    db: Session,
    now: Optional[datetime] = None,
) -> List[MeetingConsentEmail]:
    """Rows the bot should re-send right now.

    Filter: delivery_status='failed' AND attempt_count<2 AND
    next_retry_at<=now AND not opted-out. Sorted by next_retry_at so
    the bot processes the oldest backlog first.
    """
    cutoff = now or _now_utc()

    rows = db.execute(
        select(MeetingConsentEmail)
        .where(
            MeetingConsentEmail.delivery_status == "failed",
            MeetingConsentEmail.attempt_count < _MAX_ATTEMPTS,
            MeetingConsentEmail.next_retry_at.is_not(None),
            MeetingConsentEmail.next_retry_at <= cutoff,
            MeetingConsentEmail.opted_out_at.is_(None),
        )
        .order_by(MeetingConsentEmail.next_retry_at.asc())
    ).scalars().all()

    return list(rows)


# ===========================================================================
# 6. list_consent_emails (audit / FE)
# ===========================================================================


def list_consent_emails(
    db: Session,
    meeting_id: UUID,
) -> List[MeetingConsentEmail]:
    """Every consent row for a meeting, ordered by recipient email.

    Used by the FE Activity tab + ad-hoc compliance audits. 404 if the
    meeting itself doesn't exist (a meeting with zero rows returns []).
    """
    _ensure_meeting(db, meeting_id)

    rows = db.execute(
        select(MeetingConsentEmail)
        .where(MeetingConsentEmail.meeting_id == meeting_id)
        .order_by(MeetingConsentEmail.recipient_email.asc())
    ).scalars().all()

    return list(rows)

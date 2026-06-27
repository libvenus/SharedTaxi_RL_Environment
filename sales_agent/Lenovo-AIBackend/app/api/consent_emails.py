"""FastAPI router for the pre-meeting consent email pipeline.

Sprint 1A · US03 — Consent Capture

Six routes:

  POST   /consent-emails/schedule                 # bot calls when meeting eligible
  PATCH  /consent-emails/{consent_id}/delivery    # bot reports send result
  GET    /consent-emails/opt-out/{token}          # PUBLIC — browser hits this
  GET    /consent-emails/due-for-retry            # bot polls periodically
  GET    /consent-emails/{meeting_id}             # audit / FE list view
  (and on meetings_status_router.py — see below)
  GET    /meetings/{meeting_id}/consent-status    # bot pre-join check

The opt-out endpoint returns server-rendered HTML; the rest are JSON.
"""

from html import escape
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Path, Request, status
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schema.consent_email import (
    ConsentDeliveryUpdateRequest,
    ConsentDeliveryUpdateResponse,
    ConsentEmailListResponse,
    ConsentEmailRecord,
    ConsentRecipientRecord,
    ConsentRetryQueueItem,
    ConsentRetryQueueResponse,
    ConsentScheduleRequest,
    ConsentScheduleResponse,
    MeetingConsentStatusResponse,
)
from app.services.consent_email_service import (
    build_opt_out_url,
    due_for_retry,
    get_consent_status,
    list_consent_emails,
    record_delivery_status,
    record_opt_out,
    schedule_consent_emails,
)

router = APIRouter(
    prefix="/consent-emails",
    tags=["Consent Emails"],
)

# Separate router so the /meetings/{id}/consent-status URL slots cleanly
# under /meetings (alongside /meeting-details/, which already exists).
# Mounted by app/main.py.
meeting_consent_status_router = APIRouter(
    prefix="/meetings",
    tags=["Consent Emails"],
)


# ---------------------------------------------------------------------------
# POST /consent-emails/schedule  — bot calls this when a meeting is eligible
# ---------------------------------------------------------------------------


@router.post(
    "/schedule",
    response_model=ConsentScheduleResponse,
    summary="Schedule consent emails for a meeting (or fall back to in-meeting consent)",
    responses={
        404: {"description": "Meeting not found"},
        400: {"description": "Meeting has no start time / not eligible"},
        422: {"description": "Schema validation (empty recipients, malformed email)"},
    },
)
def schedule_route(
    payload: ConsentScheduleRequest = Body(...),
    db: Session = Depends(get_db),
) -> ConsentScheduleResponse:
    """Decide whether to send the consent email and create per-recipient rows.

    AC #1 — internal Lenovo emails are filtered out.
    AC #3 — if the consent window has passed, returns ``should_send=false``
            and the bot falls back to US02's in-meeting chat announcement.
    """
    result = schedule_consent_emails(
        db=db,
        meeting_id=payload.meeting_id,
        recipients=list(payload.recipients),
    )
    return ConsentScheduleResponse(
        meeting_id=result["meeting_id"],
        should_send=result["should_send"],
        fallback=result.get("fallback"),
        recipients=[
            ConsentRecipientRecord(**r) for r in result["recipients"]
        ],
        filtered_internal_count=result["filtered_internal_count"],
        seller_name=result.get("seller_name"),
        system_email_address=result.get("system_email_address"),
    )


# ---------------------------------------------------------------------------
# PATCH /consent-emails/{consent_id}/delivery  — bot reports send result
# ---------------------------------------------------------------------------


@router.patch(
    "/{consent_id}/delivery",
    response_model=ConsentDeliveryUpdateResponse,
    summary="Record the bot's send-attempt outcome (sent / failed)",
    responses={
        400: {
            "description": (
                "Already opted-out or already in fallback state — no further "
                "delivery updates accepted."
            )
        },
        404: {"description": "Consent record not found"},
        422: {"description": "Invalid status / failure_reason missing for failed"},
    },
)
def delivery_route(
    consent_id: UUID = Path(...),
    payload: ConsentDeliveryUpdateRequest = Body(...),
    db: Session = Depends(get_db),
) -> ConsentDeliveryUpdateResponse:
    """Update the row, increment attempt_count, schedule retry on failure."""
    row = record_delivery_status(
        db=db,
        consent_id=consent_id,
        new_status=payload.status,
        failure_reason=payload.failure_reason,
        attempted_at=payload.attempted_at,
    )
    return ConsentDeliveryUpdateResponse(
        consent_id=row.consent_id,
        meeting_id=row.meeting_id,
        recipient_email=row.recipient_email,
        delivery_status=row.delivery_status,
        attempt_count=row.attempt_count,
        last_attempt_at=row.last_attempt_at,
        next_retry_at=row.next_retry_at,
        failure_reason=row.failure_reason,
    )


# ---------------------------------------------------------------------------
# GET /consent-emails/opt-out/{token}   — PUBLIC, browser-facing
# ---------------------------------------------------------------------------
# Renders SUCC_MSG_0010 (AC #8) as a self-contained HTML page. Inline CSS
# keeps the response a single small payload — no static assets, no FE
# deployment needed.
# ---------------------------------------------------------------------------


_OPT_OUT_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="robots" content="noindex,nofollow" />
  <title>Recording opt-out confirmed</title>
  <style>
    :root {{ color-scheme: light dark; }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                   Oxygen, Ubuntu, Cantarell, sans-serif;
      background: #f5f7fa;
      color: #1a202c;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }}
    .card {{
      background: #fff;
      max-width: 520px;
      width: 100%;
      border-radius: 12px;
      box-shadow: 0 4px 24px rgba(15, 23, 42, 0.08);
      padding: 40px 36px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: #e6f4ea;
      color: #137333;
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 24px;
    }}
    .badge::before {{
      content: "";
      width: 8px; height: 8px; border-radius: 50%;
      background: #137333;
      display: inline-block;
    }}
    h1 {{ font-size: 22px; margin: 0 0 12px; line-height: 1.3; }}
    p  {{ font-size: 15px; line-height: 1.55; margin: 0 0 16px; color: #2d3748; }}
    .meeting {{
      background: #f7fafc;
      border-left: 3px solid #4299e1;
      padding: 14px 16px;
      border-radius: 6px;
      font-size: 14px;
      margin: 20px 0;
    }}
    .meeting strong {{ display: block; margin-bottom: 4px; color: #1a202c; }}
    .footer {{
      margin-top: 32px;
      padding-top: 20px;
      border-top: 1px solid #e2e8f0;
      font-size: 12px;
      color: #718096;
    }}
  </style>
</head>
<body>
  <main class="card">
    <span class="badge">Opt-out recorded</span>
    <h1>Your preference has been saved.</h1>
    <p>
      Thanks for letting us know. The Lenovo Note-Taking Agent will not
      record or transcribe this meeting. The seller has been notified
      that you have opted out.
    </p>
    <div class="meeting">
      <strong>Meeting</strong>
      {meeting_title}<br />
      Scheduled: {meeting_when}<br />
      Recipient: {recipient}
    </div>
    <p>
      If this was a mistake, please contact the seller directly — they
      can re-add the agent manually using Microsoft Teams' "Add
      participant" feature once the meeting starts.
    </p>
    <div class="footer">
      Reference: {reference}<br />
      Sent by Lenovo Sales Assistant on behalf of {seller_name}.
    </div>
  </main>
</body>
</html>
"""

_OPT_OUT_GONE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Opt-out link expired</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 520px; margin: 60px auto; padding: 24px;
      background: #f5f7fa; color: #1a202c;
    }
    .card { background: #fff; border-radius: 12px; padding: 40px 36px;
            box-shadow: 0 4px 24px rgba(15, 23, 42, 0.08); }
    h1 { font-size: 20px; margin: 0 0 12px; }
    p  { font-size: 15px; line-height: 1.55; color: #2d3748; }
  </style>
</head>
<body>
  <div class="card">
    <h1>This opt-out link is no longer active.</h1>
    <p>
      The meeting has already started, so opt-out preferences can no
      longer be recorded through this link. Please contact the seller
      directly if you need to discuss recording.
    </p>
  </div>
</body>
</html>
"""

_OPT_OUT_INVALID_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Invalid opt-out link</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 520px; margin: 60px auto; padding: 24px;
      background: #f5f7fa; color: #1a202c;
    }
    .card { background: #fff; border-radius: 12px; padding: 40px 36px;
            box-shadow: 0 4px 24px rgba(15, 23, 42, 0.08); }
    h1 { font-size: 20px; margin: 0 0 12px; }
    p  { font-size: 15px; line-height: 1.55; color: #2d3748; }
  </style>
</head>
<body>
  <div class="card">
    <h1>This link doesn't look right.</h1>
    <p>
      The opt-out link you used isn't recognised. Make sure you used
      the most recent invitation email — older links may have been
      invalidated when the meeting was rescheduled.
    </p>
  </div>
</body>
</html>
"""


@router.get(
    "/opt-out/{token}",
    response_class=HTMLResponse,
    summary="Public opt-out endpoint (participant clicks the email link)",
    responses={
        200: {"description": "Opt-out recorded — confirmation page (SUCC_MSG_0010)"},
        404: {"description": "Token not recognised — invalid or rescheduled meeting"},
        410: {"description": "Meeting already started — link no longer active"},
    },
)
def opt_out_route(
    request: Request,
    token: str = Path(..., min_length=20, max_length=128),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Record the opt-out and return a confirmation HTML page.

    Idempotent: clicking the link a second time renders the same page
    without re-stamping the timestamp. AC #7 cascade (cancel the bot)
    is also idempotent — already-cancelled meetings stay cancelled.
    """
    # Capture client IP from the request — best-effort; behind a load
    # balancer this needs `X-Forwarded-For` parsing, deferred to ops.
    client_ip = request.client.host if request.client else None

    try:
        row, meeting = record_opt_out(db=db, token=token, client_ip=client_ip)
    except Exception as exc:
        # Render friendly HTML for 404/410 instead of FastAPI's default
        # JSON error page (these are browser-facing, not API-facing).
        from fastapi import HTTPException

        if isinstance(exc, HTTPException):
            if exc.status_code == status.HTTP_404_NOT_FOUND:
                return HTMLResponse(_OPT_OUT_INVALID_HTML, status_code=404)
            if exc.status_code == status.HTTP_410_GONE:
                return HTMLResponse(_OPT_OUT_GONE_HTML, status_code=410)
        raise

    # Success — render SUCC_MSG_0010 confirmation page.
    meeting_when = (
        meeting.meeting_start_time.strftime("%a, %d %b %Y · %H:%M UTC")
        if meeting.meeting_start_time
        else "TBD"
    )
    html = _OPT_OUT_HTML_TEMPLATE.format(
        meeting_title=escape(meeting.title or "Untitled meeting"),
        meeting_when=escape(meeting_when),
        recipient=escape(row.recipient_email),
        reference=escape(str(row.consent_id)),
        seller_name=escape(meeting.organiser_name or "the seller"),
    )
    return HTMLResponse(html, status_code=200)


# ---------------------------------------------------------------------------
# GET /consent-emails/due-for-retry  — bot polls this periodically
# ---------------------------------------------------------------------------


@router.get(
    "/due-for-retry",
    response_model=ConsentRetryQueueResponse,
    summary="Rows ready for re-send (failed + attempt<2 + next_retry_at<=now)",
)
def due_for_retry_route(
    db: Session = Depends(get_db),
) -> ConsentRetryQueueResponse:
    """Return rows the bot should re-send right now."""
    rows = due_for_retry(db=db)
    return ConsentRetryQueueResponse(
        items=[
            ConsentRetryQueueItem(
                consent_id=r.consent_id,
                meeting_id=r.meeting_id,
                recipient_email=r.recipient_email,
                recipient_name=r.recipient_name,
                opt_out_url=build_opt_out_url(r.opt_out_token),
                attempt_count=r.attempt_count,
                last_attempt_at=r.last_attempt_at,
                next_retry_at=r.next_retry_at,
                failure_reason=r.failure_reason,
            )
            for r in rows
        ]
    )


# ---------------------------------------------------------------------------
# GET /consent-emails/{meeting_id}  — audit / FE list view
# ---------------------------------------------------------------------------


@router.get(
    "/{meeting_id}",
    response_model=ConsentEmailListResponse,
    summary="List all consent records for a meeting (audit + FE)",
    responses={
        404: {"description": "Meeting not found"},
    },
)
def list_route(
    meeting_id: UUID = Path(...),
    db: Session = Depends(get_db),
) -> ConsentEmailListResponse:
    """Return every consent row for a meeting, ordered by recipient email."""
    rows = list_consent_emails(db=db, meeting_id=meeting_id)
    return ConsentEmailListResponse(
        meeting_id=meeting_id,
        items=[
            ConsentEmailRecord(
                consent_id=r.consent_id,
                meeting_id=r.meeting_id,
                recipient_email=r.recipient_email,
                recipient_name=r.recipient_name,
                opt_out_token=r.opt_out_token,
                opt_out_url=build_opt_out_url(r.opt_out_token),
                scheduled_send_at=r.scheduled_send_at,
                attempt_count=r.attempt_count,
                last_attempt_at=r.last_attempt_at,
                next_retry_at=r.next_retry_at,
                delivery_status=r.delivery_status,
                failure_reason=r.failure_reason,
                opted_out_at=r.opted_out_at,
                opt_out_ip=r.opt_out_ip,
                seller_notified_at=r.seller_notified_at,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in rows
        ],
    )


# ---------------------------------------------------------------------------
# GET /meetings/{meeting_id}/consent-status   — bot pre-join check
# ---------------------------------------------------------------------------


@meeting_consent_status_router.get(
    "/{meeting_id}/consent-status",
    response_model=MeetingConsentStatusResponse,
    summary="Aggregated consent state for a meeting (bot reads this before joining)",
    responses={
        404: {"description": "Meeting not found"},
    },
)
def consent_status_route(
    meeting_id: UUID = Path(...),
    db: Session = Depends(get_db),
) -> MeetingConsentStatusResponse:
    """Return whether anyone has opted out + what consent mechanism applies."""
    result = get_consent_status(db=db, meeting_id=meeting_id)
    return MeetingConsentStatusResponse(**result)

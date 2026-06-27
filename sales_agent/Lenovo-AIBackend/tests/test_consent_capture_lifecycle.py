"""Smoke tests for Sprint 1A US03 (Consent Capture) endpoints.

Pins the consent-email pipeline contract that the bot integrates against.
Run with:

    pytest -q tests/test_consent_capture_lifecycle.py

These tests assume the conftest fixtures set up:
  - SQLite in-memory engine with meeting + transcript + consent schemas
  - A FastAPI app with /meeting-details, /transcripts, /consent-emails,
    and /ai-api/meetings/{id}/consent-status routers wired

NOTE on key casing: snake_case throughout, matching the rest of this
repo (and US01 / US02 conventions).

NOTE on the meeting_start_time column:
  tbl_schedule_meetings.meeting_start_time is `DateTime` (no tz). The
  consent service treats naive timestamps as UTC. Tests pass tz-aware
  datetimes via isoformat() — FastAPI strips the tz on parse against
  the schedulemeeting Pydantic schema, but the consent service
  re-applies UTC, so the math is consistent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import select


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meeting_payload(
    meeting_id: str,
    *,
    start_offset_minutes: int = 180,  # default: 3 hours from now
    **overrides,
) -> dict:
    """Build a meeting payload whose start time is now + offset minutes.

    The default 180 min is comfortably outside the 60-min consent window
    so the schedule endpoint creates rows. Tests that need the
    window-passed branch override this explicitly.
    """
    start = datetime.now(timezone.utc) + timedelta(minutes=start_offset_minutes)
    end = start + timedelta(hours=1)
    payload = {
        "meeting_id": meeting_id,
        "meeting_start_time": start.isoformat(),
        "meeting_end_time": end.isoformat(),
        "platform": "Microsoft Teams",
        "title": "ThinkPad Fleet Review",
        "attendees": "k.richter@db.com, seller@lenovo.com",
        "organiser_name": "Maria Hofer",
    }
    payload.update(overrides)
    return payload


def _create_meeting(client, *, start_offset_minutes: int = 180) -> str:
    """Create a meeting row and return its UUID string."""
    meeting_id = str(uuid4())
    client.post(
        "/ai-api/meeting-details/",
        json=_meeting_payload(meeting_id, start_offset_minutes=start_offset_minutes),
    ).raise_for_status()
    return meeting_id


def _schedule_payload(
    meeting_id: str,
    *,
    recipients: list[dict] | None = None,
) -> dict:
    """Build a /ai-api/consent-emails/schedule body."""
    if recipients is None:
        recipients = [
            {"email": "k.richter@db.com", "name": "Klaus Richter"},
            {"email": "rajesh.k@infosys.com", "name": "Rajesh Kumar"},
        ]
    return {"meeting_id": meeting_id, "recipients": recipients}


# ---------------------------------------------------------------------------
# 1. POST /schedule happy path — creates rows for external recipients only
# ---------------------------------------------------------------------------
def test_schedule_creates_rows_for_external_recipients(client):
    meeting_id = _create_meeting(client)

    response = client.post(
        "/ai-api/consent-emails/schedule",
        json=_schedule_payload(meeting_id),
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["should_send"] is True
    assert body["fallback"] is None
    assert len(body["recipients"]) == 2
    assert body["filtered_internal_count"] == 0
    assert body["seller_name"] == "Maria Hofer"
    assert body["system_email_address"]  # non-empty

    for r in body["recipients"]:
        assert r["delivery_status"] == "pending"
        assert r["opt_out_token"]  # non-empty
        assert r["opt_out_url"].endswith("/ai-api/consent-emails/opt-out/" + r["opt_out_token"])


# ---------------------------------------------------------------------------
# 2. POST /schedule filters out internal Lenovo domains (AC #1)
# ---------------------------------------------------------------------------
def test_schedule_filters_internal_domains(client):
    meeting_id = _create_meeting(client)

    response = client.post(
        "/ai-api/consent-emails/schedule",
        json=_schedule_payload(
            meeting_id,
            recipients=[
                {"email": "k.richter@db.com", "name": "Klaus Richter"},
                {"email": "seller@lenovo.com", "name": "Maria Hofer"},     # internal
                {"email": "engineer@motorola.com", "name": "Eng One"},     # internal
                {"email": "rajesh.k@infosys.com", "name": "Rajesh Kumar"},
            ],
        ),
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["filtered_internal_count"] == 2
    emails = sorted(r["recipient_email"] for r in body["recipients"])
    assert emails == ["k.richter@db.com", "rajesh.k@infosys.com"]


# ---------------------------------------------------------------------------
# 3. POST /schedule when window has passed → should_send=false (AC #3)
# ---------------------------------------------------------------------------
def test_schedule_window_passed_returns_should_send_false(client):
    # Meeting starts in 30 min — INSIDE the default 60-min consent window.
    meeting_id = _create_meeting(client, start_offset_minutes=30)

    response = client.post(
        "/ai-api/consent-emails/schedule",
        json=_schedule_payload(meeting_id),
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["should_send"] is False
    assert body["fallback"] == "in_meeting_chat"
    assert body["recipients"] == []


# ---------------------------------------------------------------------------
# 4. POST /schedule when meeting already started → should_send=false
# ---------------------------------------------------------------------------
def test_schedule_meeting_started_returns_meeting_started_fallback(client):
    # Meeting started 5 min ago.
    meeting_id = _create_meeting(client, start_offset_minutes=-5)

    response = client.post(
        "/ai-api/consent-emails/schedule",
        json=_schedule_payload(meeting_id),
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["should_send"] is False
    assert body["fallback"] == "meeting_started"


# ---------------------------------------------------------------------------
# 5. POST /schedule is idempotent — re-call returns existing tokens
# ---------------------------------------------------------------------------
def test_schedule_is_idempotent(client):
    meeting_id = _create_meeting(client)

    first = client.post("/ai-api/consent-emails/schedule", json=_schedule_payload(meeting_id))
    assert first.status_code == 200, first.text
    first_tokens = sorted(r["opt_out_token"] for r in first.json()["recipients"])

    second = client.post("/ai-api/consent-emails/schedule", json=_schedule_payload(meeting_id))
    assert second.status_code == 200, second.text
    second_tokens = sorted(r["opt_out_token"] for r in second.json()["recipients"])

    assert first_tokens == second_tokens, (
        "Re-scheduling must NOT issue new tokens — old tokens are still in the recipient's inbox"
    )


# ---------------------------------------------------------------------------
# 6. POST /schedule for unknown meeting → 404
# ---------------------------------------------------------------------------
def test_schedule_unknown_meeting_returns_404(client):
    response = client.post(
        "/ai-api/consent-emails/schedule",
        json=_schedule_payload(str(uuid4())),
    )
    assert response.status_code == 404, response.text


# ---------------------------------------------------------------------------
# 7. PATCH /delivery with status=sent updates delivery_status
# ---------------------------------------------------------------------------
def test_delivery_sent_updates_status(client):
    meeting_id = _create_meeting(client)
    schedule = client.post(
        "/ai-api/consent-emails/schedule", json=_schedule_payload(meeting_id)
    ).json()
    consent_id = schedule["recipients"][0]["consent_id"]

    response = client.patch(
        f"/ai-api/consent-emails/{consent_id}/delivery",
        json={"status": "sent"},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["delivery_status"] == "sent"
    assert body["attempt_count"] == 1
    assert body["next_retry_at"] is None
    assert body["failure_reason"] is None


# ---------------------------------------------------------------------------
# 8. PATCH /delivery with status=failed schedules retry at +10 min (AC #10)
# ---------------------------------------------------------------------------
def test_delivery_failed_first_time_schedules_retry(client):
    meeting_id = _create_meeting(client)
    schedule = client.post(
        "/ai-api/consent-emails/schedule", json=_schedule_payload(meeting_id)
    ).json()
    consent_id = schedule["recipients"][0]["consent_id"]

    response = client.patch(
        f"/ai-api/consent-emails/{consent_id}/delivery",
        json={"status": "failed", "failure_reason": "SMTP bounce"},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["delivery_status"] == "failed"
    assert body["attempt_count"] == 1
    assert body["next_retry_at"] is not None  # retry scheduled
    assert body["failure_reason"] == "SMTP bounce"


# ---------------------------------------------------------------------------
# 9. PATCH /delivery failed twice → fallback_to_in_meeting (AC #10)
# ---------------------------------------------------------------------------
def test_delivery_failed_twice_falls_back(client):
    meeting_id = _create_meeting(client)
    schedule = client.post(
        "/ai-api/consent-emails/schedule", json=_schedule_payload(meeting_id)
    ).json()
    consent_id = schedule["recipients"][0]["consent_id"]

    client.patch(
        f"/ai-api/consent-emails/{consent_id}/delivery",
        json={"status": "failed", "failure_reason": "First fail"},
    ).raise_for_status()

    response = client.patch(
        f"/ai-api/consent-emails/{consent_id}/delivery",
        json={"status": "failed", "failure_reason": "Second fail"},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["delivery_status"] == "fallback_to_in_meeting"
    assert body["attempt_count"] == 2
    assert body["next_retry_at"] is None  # no more retries


# ---------------------------------------------------------------------------
# 10. GET /opt-out/{token} happy path — records opt-out, cancels bot (AC #7,8)
# ---------------------------------------------------------------------------
def test_opt_out_records_and_cancels_bot(client, db_session):
    from app.models.schedulemeeting import MeetingDetails
    from app.models.consent_email import MeetingConsentEmail

    meeting_id = _create_meeting(client)
    schedule = client.post(
        "/ai-api/consent-emails/schedule", json=_schedule_payload(meeting_id)
    ).json()
    token = schedule["recipients"][0]["opt_out_token"]

    response = client.get(f"/ai-api/consent-emails/opt-out/{token}")
    assert response.status_code == 200, response.text
    assert "text/html" in response.headers["content-type"]
    assert "preference has been saved" in response.text  # SUCC_MSG_0010 text

    # Opt-out timestamp recorded in the DB
    db_session.expire_all()
    consent_row = db_session.execute(
        select(MeetingConsentEmail).where(
            MeetingConsentEmail.opt_out_token == token
        )
    ).scalar_one()
    assert consent_row.opted_out_at is not None
    assert consent_row.seller_notified_at is not None

    # Cascade — bot_status flipped to cancelled with the canonical reason
    meeting = db_session.get(MeetingDetails, UUID(meeting_id))
    assert meeting.bot_status == "cancelled"
    assert meeting.bot_status_reason == "participant_opted_out"


# ---------------------------------------------------------------------------
# 11. GET /opt-out/{token} after meeting starts → 410 Gone (AC #6)
# ---------------------------------------------------------------------------
def test_opt_out_after_meeting_started_returns_410(client, db_session):
    from app.models.consent_email import MeetingConsentEmail
    from app.models.schedulemeeting import MeetingDetails

    meeting_id = _create_meeting(client)
    schedule = client.post(
        "/ai-api/consent-emails/schedule", json=_schedule_payload(meeting_id)
    ).json()
    token = schedule["recipients"][0]["opt_out_token"]

    # Backdate the meeting so the link is "expired".
    meeting = db_session.get(MeetingDetails, UUID(meeting_id))
    meeting.meeting_start_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    db_session.add(meeting)
    db_session.commit()

    response = client.get(f"/ai-api/consent-emails/opt-out/{token}")
    assert response.status_code == 410, response.text
    assert "no longer active" in response.text


# ---------------------------------------------------------------------------
# 12. GET /opt-out/{token} is idempotent — second click also 200
# ---------------------------------------------------------------------------
def test_opt_out_is_idempotent(client, db_session):
    from app.models.consent_email import MeetingConsentEmail

    meeting_id = _create_meeting(client)
    schedule = client.post(
        "/ai-api/consent-emails/schedule", json=_schedule_payload(meeting_id)
    ).json()
    token = schedule["recipients"][0]["opt_out_token"]

    first = client.get(f"/ai-api/consent-emails/opt-out/{token}")
    assert first.status_code == 200

    # Capture the recorded timestamp
    db_session.expire_all()
    row = db_session.execute(
        select(MeetingConsentEmail).where(
            MeetingConsentEmail.opt_out_token == token
        )
    ).scalar_one()
    first_ts = row.opted_out_at
    assert first_ts is not None

    # Second click — same response, but timestamp must NOT be re-stamped
    second = client.get(f"/ai-api/consent-emails/opt-out/{token}")
    assert second.status_code == 200
    assert "preference has been saved" in second.text

    db_session.expire_all()
    row2 = db_session.execute(
        select(MeetingConsentEmail).where(
            MeetingConsentEmail.opt_out_token == token
        )
    ).scalar_one()
    assert row2.opted_out_at == first_ts


# ---------------------------------------------------------------------------
# 13. GET /opt-out/{token} with bad token → 404 HTML
# ---------------------------------------------------------------------------
def test_opt_out_with_bad_token_returns_404_html(client):
    bad_token = "x" * 32  # right shape, wrong value
    response = client.get(f"/ai-api/consent-emails/opt-out/{bad_token}")
    assert response.status_code == 404
    assert "text/html" in response.headers["content-type"]
    assert "doesn't look right" in response.text


# ---------------------------------------------------------------------------
# 14. GET /ai-api/meetings/{id}/consent-status aggregates correctly
# ---------------------------------------------------------------------------
def test_consent_status_aggregates_after_opt_out(client):
    meeting_id = _create_meeting(client)
    schedule = client.post(
        "/ai-api/consent-emails/schedule", json=_schedule_payload(meeting_id)
    ).json()
    tokens = [r["opt_out_token"] for r in schedule["recipients"]]

    # Pre-opt-out — any_opted_out=False
    pre = client.get(f"/ai-api/meetings/{meeting_id}/consent-status").json()
    assert pre["any_opted_out"] is False
    assert pre["opt_out_count"] == 0
    assert pre["total_recipients"] == 2

    # One opt-out
    client.get(f"/ai-api/consent-emails/opt-out/{tokens[0]}").raise_for_status()

    post = client.get(f"/ai-api/meetings/{meeting_id}/consent-status").json()
    assert post["any_opted_out"] is True
    assert post["opt_out_count"] == 1
    assert post["total_recipients"] == 2


# ---------------------------------------------------------------------------
# 15. GET /due-for-retry returns ONLY due-for-retry rows
# ---------------------------------------------------------------------------
def test_due_for_retry_returns_only_failed_rows(client, db_session):
    from app.models.consent_email import MeetingConsentEmail

    meeting_id = _create_meeting(client)
    schedule = client.post(
        "/ai-api/consent-emails/schedule", json=_schedule_payload(meeting_id)
    ).json()
    consent_id_a = schedule["recipients"][0]["consent_id"]
    consent_id_b = schedule["recipients"][1]["consent_id"]

    # Recipient A: failed, retry due in past (eligible)
    client.patch(
        f"/ai-api/consent-emails/{consent_id_a}/delivery",
        json={"status": "failed", "failure_reason": "Bounce"},
    ).raise_for_status()
    # Backdate the next_retry_at so it's "due now".
    db_session.expire_all()
    row_a = db_session.get(MeetingConsentEmail, UUID(consent_id_a))
    row_a.next_retry_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db_session.add(row_a)
    db_session.commit()

    # Recipient B: sent (not eligible)
    client.patch(
        f"/ai-api/consent-emails/{consent_id_b}/delivery",
        json={"status": "sent"},
    ).raise_for_status()

    response = client.get("/ai-api/consent-emails/due-for-retry")
    assert response.status_code == 200, response.text

    items = response.json()["items"]
    consent_ids = [i["consent_id"] for i in items]
    assert consent_id_a in consent_ids
    assert consent_id_b not in consent_ids

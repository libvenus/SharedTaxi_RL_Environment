"""Smoke tests for Sprint 1A US01 endpoints.

Pins the contract that the bot / agent team integrates against. Run with:

    pytest -q

These are intentionally narrow — they confirm the happy path + the
"don't break things on idempotent re-POST" + the validation surface,
but stop short of full integration coverage. Add more as the bot's
behaviour solidifies.

NOTE on key casing: this service uses **snake_case** keys throughout
(``meeting_id``, ``bot_status``, ``bot_status_reason``, ...) to match
the existing convention in this repo. The frontend consumers are the
D365 backend (camelCase) — there's an integration translator there if /
when we wire them together end-to-end. Don't change to camelCase here
without first updating the schemas to add Pydantic ``alias_generator``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest


def _meeting_payload(meeting_id: str, **overrides) -> dict:
    """Build a minimal-valid meeting payload."""
    start = datetime(2026, 6, 15, 15, 0, tzinfo=timezone.utc)
    payload = {
        "meeting_id": meeting_id,
        "meeting_start_time": start.isoformat(),
        "meeting_end_time": (start + timedelta(hours=1)).isoformat(),
        "platform": "Microsoft Teams",
        "title": "ThinkPad Fleet Review",
        "attendees": "k.richter@db.com, seller@lenovo.com",
        "organiser_name": "Maria Hofer",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# 1. POST creates row with bot_status defaulting to 'pending'
# ---------------------------------------------------------------------------
def test_post_creates_meeting_with_pending_status(client, db_session):
    from app.models.schedulemeeting import MeetingDetails

    meeting_id = str(uuid4())
    response = client.post("/ai-api/meeting-details/", json=_meeting_payload(meeting_id))
    assert response.status_code == 200, response.text

    # The PG UUID type's bind processor expects a uuid.UUID, not a str.
    # Pydantic does this for us on the POST path; for direct ORM reads
    # we cast explicitly.
    db_session.expire_all()
    row = db_session.get(MeetingDetails, UUID(meeting_id))
    assert row is not None
    assert row.bot_status == "pending"
    assert row.bot_status_reason is None
    assert row.bot_last_event_at is None


# ---------------------------------------------------------------------------
# 2. Re-POSTing the same meeting_id is idempotent and preserves bot_status
# ---------------------------------------------------------------------------
def test_repost_preserves_bot_status(client, db_session):
    from app.models.schedulemeeting import MeetingDetails

    meeting_id = str(uuid4())
    client.post("/ai-api/meeting-details/", json=_meeting_payload(meeting_id)).raise_for_status()

    patched = client.patch(
        f"/ai-api/meeting-details/{meeting_id}/status",
        json={"bot_status": "joined"},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["bot_status"] == "joined"

    # Re-POST with a changed title — bot_status must stay 'joined', not
    # silently revert to 'pending'.
    second = client.post(
        "/ai-api/meeting-details/",
        json=_meeting_payload(meeting_id, title="ThinkPad Fleet Review (rev 2)"),
    )
    assert second.status_code == 200

    db_session.expire_all()
    row = db_session.get(MeetingDetails, UUID(meeting_id))
    assert row is not None
    assert row.bot_status == "joined"
    assert row.title == "ThinkPad Fleet Review (rev 2)"


# ---------------------------------------------------------------------------
# 3. PATCH status transitions through the legal states
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "next_status,reason",
    [
        ("scheduled", None),
        ("joining", None),
        ("joined", None),
        ("lobby_waiting", "Waiting for organiser to admit"),
        ("rescheduled", "Outlook event moved"),
        ("failed", "Network error during join"),
    ],
)
def test_patch_status_transitions(client, next_status, reason):
    meeting_id = str(uuid4())
    client.post("/ai-api/meeting-details/", json=_meeting_payload(meeting_id)).raise_for_status()

    body = {"bot_status": next_status}
    if reason:
        body["reason"] = reason

    response = client.patch(
        f"/ai-api/meeting-details/{meeting_id}/status",
        json=body,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert str(payload["meeting_id"]).lower() == meeting_id.lower()
    assert payload["bot_status"] == next_status
    assert payload["bot_status_reason"] == reason
    assert payload["bot_last_event_at"] is not None


# ---------------------------------------------------------------------------
# 4. PATCH with an invalid status returns 422 (Pydantic guards the whitelist)
# ---------------------------------------------------------------------------
def test_patch_status_rejects_unknown_value(client):
    meeting_id = str(uuid4())
    client.post("/ai-api/meeting-details/", json=_meeting_payload(meeting_id)).raise_for_status()

    response = client.patch(
        f"/ai-api/meeting-details/{meeting_id}/status",
        json={"bot_status": "totally-not-a-real-status"},
    )
    assert response.status_code == 422
    body = response.json()
    # FastAPI / Pydantic standard error envelope
    assert "detail" in body


# ---------------------------------------------------------------------------
# 5. PATCH on unknown meeting_id returns 404
# ---------------------------------------------------------------------------
def test_patch_status_404_for_unknown_meeting(client):
    response = client.patch(
        f"/ai-api/meeting-details/{uuid4()}/status",
        json={"bot_status": "joined"},
    )
    assert response.status_code == 404, response.text
    assert "not found" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 6. DELETE is a soft-delete (flips status, doesn't remove the row)
# ---------------------------------------------------------------------------
def test_delete_is_soft_delete(client, db_session):
    from app.models.schedulemeeting import MeetingDetails

    meeting_id = str(uuid4())
    client.post("/ai-api/meeting-details/", json=_meeting_payload(meeting_id)).raise_for_status()

    response = client.delete(
        f"/ai-api/meeting-details/{meeting_id}",
        params={"reason": "Cancelled by organiser"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["bot_status"] == "cancelled"

    # Row is still there.
    db_session.expire_all()
    row = db_session.get(MeetingDetails, UUID(meeting_id))
    assert row is not None
    assert row.bot_status == "cancelled"
    assert row.bot_status_reason == "Cancelled by organiser"

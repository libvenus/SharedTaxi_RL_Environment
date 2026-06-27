"""Smoke tests for Sprint 1A US02 (Consent & Recording) endpoints.

Pins the transcript-pipeline contract the bot / agent team integrates
against. Run with:

    pytest -q tests/test_transcript_lifecycle.py

These tests assume the conftest fixtures set up:
  - SQLite in-memory engine with both meeting + transcript schemas
  - A FastAPI app with /ai-api/meeting-details and /ai-api/transcripts routers wired

NOTE on key casing: snake_case throughout, matching the rest of this
repo. See ``tests/test_meeting_lifecycle.py`` for the rationale.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import select


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meeting_payload(meeting_id: str, **overrides) -> dict:
    """Build a minimal-valid meeting payload (mirror of test_meeting_lifecycle)."""
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


def _create_joined_meeting(client) -> str:
    """Create a meeting and PATCH its bot_status to 'joined'.

    Returns the meeting_id (string, no hyphens stripped — UUID format).
    Required precondition for every transcript-start test.
    """
    meeting_id = str(uuid4())
    client.post("/ai-api/meeting-details/", json=_meeting_payload(meeting_id)).raise_for_status()
    client.patch(
        f"/ai-api/meeting-details/{meeting_id}/status",
        json={"bot_status": "joined"},
    ).raise_for_status()
    return meeting_id


def _consent_now() -> dict:
    """A pair of consent fields the bot would send when starting the transcript."""
    now = datetime(2026, 6, 15, 15, 0, 30, tzinfo=timezone.utc)
    return {
        "consent_message_text": (
            "Hi everyone — this meeting is being recorded by the Lenovo "
            "Note-Taking Agent for transcript and follow-up purposes. "
            "(CONF_MSG_0004)"
        ),
        "consent_sent_at": now.isoformat(),
    }


def _segment(
    *,
    speaker_name: str = "Klaus Richter",
    speaker_email: str | None = "k.richter@db.com",
    speaker_role: str | None = "CTO",
    text: str = "Thanks for joining the call today.",
    start: datetime | None = None,
    end: datetime | None = None,
    confidence: float = 0.93,
) -> dict:
    """Build a single transcript segment payload."""
    if start is None:
        start = datetime(2026, 6, 15, 15, 1, 0, tzinfo=timezone.utc)
    if end is None:
        end = start + timedelta(seconds=4)
    return {
        "speaker_name": speaker_name,
        "speaker_email": speaker_email,
        "speaker_role": speaker_role,
        "utterance_text": text,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "confidence_score": confidence,
    }


# ---------------------------------------------------------------------------
# 1. POST /ai-api/transcripts/ creates row with status='in_progress' + consent fields
# ---------------------------------------------------------------------------
def test_post_transcript_creates_in_progress_row(client, db_session):
    from app.models.transcript import MeetingTranscript

    meeting_id = _create_joined_meeting(client)

    response = client.post(
        "/ai-api/transcripts/",
        json={"meeting_id": meeting_id, **_consent_now()},
    )
    assert response.status_code == 201, response.text

    body = response.json()
    assert body["status"] == "in_progress"
    assert body["consent_message_text"].startswith("Hi everyone")
    assert body["consent_sent_at"] is not None
    assert body["segment_count"] == 0
    assert body["finalized_at"] is None

    # And the row really landed in the DB.
    db_session.expire_all()
    row = db_session.execute(
        select(MeetingTranscript).where(
            MeetingTranscript.meeting_id == UUID(meeting_id)
        )
    ).scalar_one()
    assert str(row.status) == "in_progress"


# ---------------------------------------------------------------------------
# 2. POST /ai-api/transcripts/ without consent fields → 422
# ---------------------------------------------------------------------------
def test_post_transcript_without_consent_returns_422(client):
    meeting_id = _create_joined_meeting(client)

    response = client.post(
        "/ai-api/transcripts/",
        json={"meeting_id": meeting_id},
    )
    assert response.status_code == 422, response.text


# ---------------------------------------------------------------------------
# 3. POST /ai-api/transcripts/ for a meeting that's still 'pending' → 400
# ---------------------------------------------------------------------------
def test_post_transcript_when_bot_not_joined_returns_400(client):
    # Create the meeting but DON'T patch to 'joined'.
    meeting_id = str(uuid4())
    client.post("/ai-api/meeting-details/", json=_meeting_payload(meeting_id)).raise_for_status()

    response = client.post(
        "/ai-api/transcripts/",
        json={"meeting_id": meeting_id, **_consent_now()},
    )
    assert response.status_code == 400, response.text
    assert "bot_status" in response.text.lower()


# ---------------------------------------------------------------------------
# 4. POST /ai-api/transcripts/ twice for the same meeting → 409
# ---------------------------------------------------------------------------
def test_post_transcript_twice_returns_409(client):
    meeting_id = _create_joined_meeting(client)
    body = {"meeting_id": meeting_id, **_consent_now()}

    first = client.post("/ai-api/transcripts/", json=body)
    assert first.status_code == 201, first.text

    second = client.post("/ai-api/transcripts/", json=body)
    assert second.status_code == 409, second.text


# ---------------------------------------------------------------------------
# 5. POST /ai-api/transcripts/{id}/segments appends and increments segment_count
# ---------------------------------------------------------------------------
def test_post_segments_appends_and_increments(client):
    meeting_id = _create_joined_meeting(client)
    client.post(
        "/ai-api/transcripts/",
        json={"meeting_id": meeting_id, **_consent_now()},
    ).raise_for_status()

    # First batch — 2 segments
    first = client.post(
        f"/ai-api/transcripts/{meeting_id}/segments",
        json={"segments": [_segment(text="One"), _segment(text="Two")]},
    )
    assert first.status_code == 200, first.text
    assert first.json()["appended_count"] == 2
    assert first.json()["segment_count"] == 2

    # Second batch — 3 more
    second = client.post(
        f"/ai-api/transcripts/{meeting_id}/segments",
        json={"segments": [
            _segment(text="Three"),
            _segment(text="Four"),
            _segment(text="Five"),
        ]},
    )
    assert second.status_code == 200, second.text
    assert second.json()["appended_count"] == 3
    assert second.json()["segment_count"] == 5


# ---------------------------------------------------------------------------
# 6. POST /ai-api/transcripts/{id}/segments with confidence > 1 → 422
# ---------------------------------------------------------------------------
def test_post_segments_invalid_confidence_returns_422(client):
    meeting_id = _create_joined_meeting(client)
    client.post(
        "/ai-api/transcripts/",
        json={"meeting_id": meeting_id, **_consent_now()},
    ).raise_for_status()

    bad = _segment(confidence=1.5)
    response = client.post(
        f"/ai-api/transcripts/{meeting_id}/segments",
        json={"segments": [bad]},
    )
    assert response.status_code == 422, response.text


# ---------------------------------------------------------------------------
# 7. POST /ai-api/transcripts/{id}/finalize sets status + score + reason
# ---------------------------------------------------------------------------
def test_finalize_sets_status_and_overall_confidence(client):
    meeting_id = _create_joined_meeting(client)
    client.post(
        "/ai-api/transcripts/",
        json={"meeting_id": meeting_id, **_consent_now()},
    ).raise_for_status()

    response = client.post(
        f"/ai-api/transcripts/{meeting_id}/finalize",
        json={"overall_confidence_score": 0.91},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "finalized"
    assert float(body["overall_confidence_score"]) == 0.91
    assert body["terminated_reason"] == "meeting_ended"
    assert body["finalized_at"] is not None


# ---------------------------------------------------------------------------
# 8. POST /ai-api/transcripts/{id}/terminate marks 'terminated_partial' with reason
# ---------------------------------------------------------------------------
def test_terminate_marks_partial_with_reason(client):
    meeting_id = _create_joined_meeting(client)
    client.post(
        "/ai-api/transcripts/",
        json={"meeting_id": meeting_id, **_consent_now()},
    ).raise_for_status()

    response = client.post(
        f"/ai-api/transcripts/{meeting_id}/terminate",
        json={"reason": "organizer_removed"},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "terminated_partial"
    assert body["terminated_reason"] == "organizer_removed"
    assert body["finalized_at"] is not None


# ---------------------------------------------------------------------------
# 9. GET /ai-api/transcripts/{id} returns metadata + segments ordered by start_time
# ---------------------------------------------------------------------------
def test_get_transcript_returns_segments_ordered_by_start_time(client):
    meeting_id = _create_joined_meeting(client)
    client.post(
        "/ai-api/transcripts/",
        json={"meeting_id": meeting_id, **_consent_now()},
    ).raise_for_status()

    base = datetime(2026, 6, 15, 15, 1, 0, tzinfo=timezone.utc)
    # Insert deliberately out-of-order to confirm the ORDER BY does its job.
    out_of_order = [
        _segment(text="THIRD", start=base + timedelta(seconds=20),
                 end=base + timedelta(seconds=24)),
        _segment(text="FIRST", start=base,
                 end=base + timedelta(seconds=4)),
        _segment(text="SECOND", start=base + timedelta(seconds=10),
                 end=base + timedelta(seconds=14)),
    ]
    client.post(
        f"/ai-api/transcripts/{meeting_id}/segments",
        json={"segments": out_of_order},
    ).raise_for_status()

    response = client.get(f"/ai-api/transcripts/{meeting_id}")
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["transcript"]["segment_count"] == 3
    texts = [seg["utterance_text"] for seg in body["segments"]]
    assert texts == ["FIRST", "SECOND", "THIRD"], texts

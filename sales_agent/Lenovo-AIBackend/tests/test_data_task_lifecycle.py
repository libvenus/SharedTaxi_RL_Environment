"""Smoke tests for Sprint 1A US04 (Data Hygiene) endpoints + detectors.

Pins the data-hygiene task queue contract used by the AI team's
transcript-signal pipeline, the daily-scan CLI job, and the FE seller
To-Do list.

Run with:

    pytest -q tests/test_data_task_lifecycle.py

These tests assume the conftest fixtures set up:
  - SQLite in-memory engine with the data-task schema (Base.metadata)
  - A FastAPI app with /ai-api/data-tasks router mounted

NOTE on idempotency tests: the partial UNIQUE index from
``sql/2026_06_us04_data_task.sql`` is NOT created by SQLAlchemy's
``create_all()`` — it lives only in the SQL migration. The service layer
uses a SELECT-then-INSERT pattern so idempotency is enforced from
software too, which is what these tests exercise.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from uuid import uuid4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_payload(
    *,
    owner_id: str | None = None,
    entity_kind: str = "opportunity",
    entity_id: str | None = None,
    task_kind: str = "transcript_signal_close_date_different",
    severity: str = "medium",
    confidence: str | None = "high",
    field_name: str | None = "close_date",
    current_value: str | None = "2026-06-30",
    suggested_value: str | None = "2026-06-15",
    evidence_ref: str | None = "transcript_segment_id=ab12",
    evidence_text: str = "Customer mentioned 'we need it before June 15th' on the call.",
    created_by_source: str = "transcript",
) -> dict:
    """Build a POST /ai-api/data-tasks body with sensible defaults.

    Specific tests override fields they care about.
    """
    return {
        "owner_id": owner_id or str(uuid4()),
        "entity_kind": entity_kind,
        "entity_id": entity_id or str(uuid4()),
        "task_kind": task_kind,
        "severity": severity,
        "confidence": confidence,
        "field_name": field_name,
        "current_value": current_value,
        "suggested_value": suggested_value,
        "evidence_ref": evidence_ref,
        "evidence_text": evidence_text,
        "created_by_source": created_by_source,
    }


def _create_task(client, **overrides) -> dict:
    """POST a task and return the parsed response body. Asserts 200."""
    response = client.post("/ai-api/data-tasks", json=_task_payload(**overrides))
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# 1. POST /ai-api/data-tasks happy path — creates a brand-new open task
# ---------------------------------------------------------------------------
def test_create_task_happy_path(client):
    body = _create_task(client)
    assert body["was_existing"] is False
    task = body["task"]
    assert task["status"] == "open"
    assert task["task_kind"] == "transcript_signal_close_date_different"
    assert task["severity"] == "medium"
    assert task["confidence"] == "high"
    assert task["evidence_text"].startswith("Customer mentioned")
    assert task["resolved_at"] is None
    assert task["dismissed_at"] is None


# ---------------------------------------------------------------------------
# 2. POST /ai-api/data-tasks is idempotent — same (entity, kind) → same row
# ---------------------------------------------------------------------------
def test_create_task_is_idempotent_for_open_tasks(client):
    owner_id = str(uuid4())
    entity_id = str(uuid4())

    first = _create_task(client, owner_id=owner_id, entity_id=entity_id)
    assert first["was_existing"] is False
    first_id = first["task"]["task_id"]

    second = _create_task(client, owner_id=owner_id, entity_id=entity_id)
    assert second["was_existing"] is True
    assert second["task"]["task_id"] == first_id
    # Original evidence_text is preserved — no overwrite on idempotent re-call.
    assert second["task"]["evidence_text"] == first["task"]["evidence_text"]


# ---------------------------------------------------------------------------
# 3. AC #5 suppression — dismissed task with same key returned, no new row
# ---------------------------------------------------------------------------
def test_create_task_suppresses_when_dismissed_task_exists(client):
    actor_id = str(uuid4())
    owner_id = str(uuid4())
    entity_id = str(uuid4())

    created = _create_task(client, owner_id=owner_id, entity_id=entity_id)
    task_id = created["task"]["task_id"]

    # Dismiss it.
    dismiss_resp = client.post(
        f"/ai-api/data-tasks/{task_id}/dismiss",
        json={"actor_id": actor_id, "note": "Customer confirmed the original date."},
    )
    assert dismiss_resp.status_code == 200, dismiss_resp.text
    assert dismiss_resp.json()["task"]["status"] == "dismissed"

    # Re-detect — should NOT create a new task; should return the dismissed one.
    re_detect = _create_task(client, owner_id=owner_id, entity_id=entity_id)
    assert re_detect["was_existing"] is True
    assert re_detect["task"]["task_id"] == task_id
    assert re_detect["task"]["status"] == "dismissed"


# ---------------------------------------------------------------------------
# 4. POST /ai-api/data-tasks rejects empty evidence_text (AC #3)
# ---------------------------------------------------------------------------
def test_create_task_rejects_empty_evidence_text(client):
    payload = _task_payload(evidence_text="")
    response = client.post("/ai-api/data-tasks", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 5. GET /ai-api/data-tasks?ownerId=X scopes to a single seller (AC #11)
# ---------------------------------------------------------------------------
def test_list_filters_by_owner(client):
    owner_a = str(uuid4())
    owner_b = str(uuid4())
    _create_task(client, owner_id=owner_a)
    _create_task(client, owner_id=owner_a)
    _create_task(client, owner_id=owner_b)

    resp = client.get("/ai-api/data-tasks", params={"ownerId": owner_a})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert all(t["owner_id"] == owner_a for t in body["items"])


# ---------------------------------------------------------------------------
# 6. GET ordering — high confidence first, then severity, then age
# ---------------------------------------------------------------------------
def test_list_orders_by_confidence_then_severity_then_age(client):
    owner_id = str(uuid4())

    # Low confidence + high severity — should NOT come first because
    # confidence dominates.
    low_conf = _create_task(
        client,
        owner_id=owner_id,
        entity_id=str(uuid4()),
        task_kind="kind_a",
        confidence="low",
        severity="high",
    )

    # High confidence + low severity — should rank above low_conf.
    high_conf = _create_task(
        client,
        owner_id=owner_id,
        entity_id=str(uuid4()),
        task_kind="kind_b",
        confidence="high",
        severity="low",
    )

    # NULL confidence — sorts last regardless of severity.
    null_conf = _create_task(
        client,
        owner_id=owner_id,
        entity_id=str(uuid4()),
        task_kind="kind_c",
        confidence=None,
        severity="high",
    )

    resp = client.get("/ai-api/data-tasks", params={"ownerId": owner_id})
    items = resp.json()["items"]
    assert len(items) == 3
    assert items[0]["task_id"] == high_conf["task"]["task_id"]
    assert items[1]["task_id"] == low_conf["task"]["task_id"]
    assert items[2]["task_id"] == null_conf["task"]["task_id"]


# ---------------------------------------------------------------------------
# 7. GET /ai-api/data-tasks/{task_id} returns full record
# ---------------------------------------------------------------------------
def test_get_task_returns_full_record(client):
    created = _create_task(client)
    task_id = created["task"]["task_id"]

    resp = client.get(f"/ai-api/data-tasks/{task_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["evidence_text"] == created["task"]["evidence_text"]


# ---------------------------------------------------------------------------
# 8. GET /ai-api/data-tasks/{task_id} 404s on unknown id
# ---------------------------------------------------------------------------
def test_get_task_404s_on_unknown(client):
    resp = client.get(f"/ai-api/data-tasks/{uuid4()}")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 9. POST /resolve happy path
# ---------------------------------------------------------------------------
def test_resolve_task_writes_audit_fields(client):
    actor_id = str(uuid4())
    created = _create_task(client)
    task_id = created["task"]["task_id"]

    resp = client.post(
        f"/ai-api/data-tasks/{task_id}/resolve",
        json={"actor_id": actor_id, "resolved_value": "2026-06-15"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["was_already_resolved"] is False
    task = body["task"]
    assert task["status"] == "resolved"
    assert task["actor_id"] == actor_id
    assert task["resolved_value"] == "2026-06-15"
    assert task["resolved_at"] is not None


# ---------------------------------------------------------------------------
# 10. POST /resolve is idempotent — second resolve returns was_already_resolved
# ---------------------------------------------------------------------------
def test_resolve_task_is_idempotent(client):
    actor_id = str(uuid4())
    created = _create_task(client)
    task_id = created["task"]["task_id"]

    first = client.post(
        f"/ai-api/data-tasks/{task_id}/resolve",
        json={"actor_id": actor_id, "resolved_value": "2026-06-15"},
    )
    assert first.status_code == 200
    first_resolved_at = first.json()["task"]["resolved_at"]

    second = client.post(
        f"/ai-api/data-tasks/{task_id}/resolve",
        json={"actor_id": str(uuid4()), "resolved_value": "different"},
    )
    assert second.status_code == 200
    body = second.json()
    assert body["was_already_resolved"] is True
    # resolved_at should NOT have been overwritten by the second resolver.
    assert body["task"]["resolved_at"] == first_resolved_at


# ---------------------------------------------------------------------------
# 11. POST /dismiss rejects empty / whitespace-only note (AC #4)
# ---------------------------------------------------------------------------
def test_dismiss_task_rejects_empty_note(client):
    created = _create_task(client)
    task_id = created["task"]["task_id"]

    # Pydantic rejects min_length=1 → 422
    blank = client.post(
        f"/ai-api/data-tasks/{task_id}/dismiss",
        json={"actor_id": str(uuid4()), "note": ""},
    )
    assert blank.status_code == 422

    # Whitespace-only — slips past Pydantic min_length=1, but the service
    # layer rejects → 409.
    whitespace = client.post(
        f"/ai-api/data-tasks/{task_id}/dismiss",
        json={"actor_id": str(uuid4()), "note": "   "},
    )
    assert whitespace.status_code == 409


# ---------------------------------------------------------------------------
# 12. POST /dismiss happy path
# ---------------------------------------------------------------------------
def test_dismiss_task_writes_audit_fields(client):
    actor_id = str(uuid4())
    created = _create_task(client)
    task_id = created["task"]["task_id"]

    resp = client.post(
        f"/ai-api/data-tasks/{task_id}/dismiss",
        json={
            "actor_id": actor_id,
            "note": "Confirmed with customer — original date is correct.",
        },
    )
    assert resp.status_code == 200, resp.text
    task = resp.json()["task"]
    assert task["status"] == "dismissed"
    assert task["actor_id"] == actor_id
    assert task["dismissal_note"].startswith("Confirmed with customer")
    assert task["dismissed_at"] is not None


# ---------------------------------------------------------------------------
# 13. Detector unit: D1/D2/D3 happy paths
# ---------------------------------------------------------------------------
def test_deterministic_detectors_happy_paths():
    """Pure-function detector tests — no DB, no HTTP, just unit math."""
    from app.clients.d365_client import OpportunityScanRow
    from app.services.data_task_detectors import (
        detect_past_close_date,
        detect_stale_activity,
        detect_zero_or_missing_value,
    )

    today = date(2026, 6, 9)
    scan_run_at = datetime(2026, 6, 9, 2, 0, 0, tzinfo=timezone.utc)
    owner = uuid4()
    opp_id = uuid4()

    # D1 — past close date, still active
    overdue = OpportunityScanRow(
        opportunity_id=opp_id,
        owner_id=owner,
        statecode="Open",
        close_date=date(2026, 5, 1),  # 39 days ago
        estimated_value=200_000.0,
        last_activity=datetime(2026, 6, 8, tzinfo=timezone.utc),
    )
    d1 = detect_past_close_date(overdue, today=today, scan_run_at=scan_run_at)
    assert d1 is not None
    assert d1.task_kind == "past_close_date"
    assert d1.severity == "high"

    # D1 — close date in the future = no task
    future = OpportunityScanRow(
        opportunity_id=uuid4(),
        owner_id=owner,
        statecode="Open",
        close_date=date(2026, 9, 1),
        estimated_value=200_000.0,
    )
    assert detect_past_close_date(future, today=today, scan_run_at=scan_run_at) is None

    # D2 — zero value
    zero_val = OpportunityScanRow(
        opportunity_id=uuid4(),
        owner_id=owner,
        statecode="Open",
        estimated_value=0.0,
    )
    d2 = detect_zero_or_missing_value(zero_val, scan_run_at=scan_run_at)
    assert d2 is not None
    assert d2.task_kind == "zero_or_missing_value"
    assert d2.field_name == "estimated_value"

    # D2 — non-zero = no task
    has_value = OpportunityScanRow(
        opportunity_id=uuid4(),
        owner_id=owner,
        statecode="Open",
        estimated_value=42_000.0,
    )
    assert detect_zero_or_missing_value(has_value, scan_run_at=scan_run_at) is None

    # D3 — stale activity (45 days)
    stale = OpportunityScanRow(
        opportunity_id=uuid4(),
        owner_id=owner,
        statecode="Open",
        estimated_value=10_000.0,
        last_activity=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    d3 = detect_stale_activity(
        stale, today=today, stale_days=30, scan_run_at=scan_run_at
    )
    assert d3 is not None
    assert d3.task_kind == "stale_activity"
    # 45 days < 60 (2x threshold) → severity 'medium'
    assert d3.severity == "medium"

    # D3 — fresh activity = no task
    fresh = OpportunityScanRow(
        opportunity_id=uuid4(),
        owner_id=owner,
        statecode="Open",
        estimated_value=10_000.0,
        last_activity=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    assert detect_stale_activity(
        fresh, today=today, stale_days=30, scan_run_at=scan_run_at
    ) is None


# ---------------------------------------------------------------------------
# 14. Detector unit: D4 picks the most-severe risk and maps category → severity
# ---------------------------------------------------------------------------
def test_risk_flag_detector_picks_most_severe():
    from app.clients.d365_client import OpportunityRisk
    from app.services.data_task_detectors import detect_risk_flags

    owner = uuid4()
    opp_id = uuid4()
    scan_run_at = datetime(2026, 6, 9, 2, 0, 0, tzinfo=timezone.utc)

    risks = [
        OpportunityRisk(
            risk_id="r1",
            category="Activity & Engagement",  # → medium
            name="Low Activity",
            message="Only 1 activity in 60 days.",
        ),
        OpportunityRisk(
            risk_id="r2",
            category="Timeline & Forecast",  # → high
            name="Past Close Date",
            message="Close date passed 12 days ago.",
        ),
    ]

    out = detect_risk_flags(
        opp_id, risks, owner_id=owner, scan_run_at=scan_run_at
    )
    assert len(out) == 1
    payload = out[0]
    assert payload.task_kind == "risk_flag"
    assert payload.severity == "high"
    assert payload.field_name == "Past Close Date"
    assert payload.evidence_text == "Close date passed 12 days ago."
    assert "d365_risk_id=r2" in payload.evidence_ref

    # Empty risks → no tasks
    assert detect_risk_flags(
        opp_id, [], owner_id=owner, scan_run_at=scan_run_at
    ) == []

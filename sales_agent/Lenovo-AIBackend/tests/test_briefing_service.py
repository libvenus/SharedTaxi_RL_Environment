"""Unit tests for briefing derivation rules."""

from __future__ import annotations

from app.services.briefing_service import (
    ERR_MSG_0023,
    _derive_prep_tasks,
    _derive_talking_points,
)


def test_derive_prep_tasks_from_inbound_email_signal() -> None:
    signals = [
        {
            "summary": "Customer replied requesting revised pricing",
            "whyShown": "Inbound email detected — customer-initiated contact is prioritised.",
            "source": {
                "sourceType": "d365_activity",
                "sourceId": "act-1",
                "label": "Email",
            },
        }
    ]
    tasks = _derive_prep_tasks(signals)
    assert len(tasks) >= 1
    assert tasks[0]["priority"] == "HIGH"


def test_talking_points_empty_without_sources() -> None:
    points = _derive_talking_points([], {"stage": "Propose", "fields": []}, None)
    assert points == []


def test_err_msg_0023_for_empty_talking_points() -> None:
    assert ERR_MSG_0023 == "ERR_MSG_0023"

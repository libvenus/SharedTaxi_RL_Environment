"""Unit tests for Task Pending badge helpers (no database)."""

from __future__ import annotations

import pytest

from app.services.task_pending import badge_color_for, build_pending_label


@pytest.mark.parametrize(
    "count, expected",
    [
        (0, "0 tasks pending"),
        (1, "1 task pending"),
        (5, "5 tasks pending"),
    ],
)
def test_build_pending_label(count: int, expected: str) -> None:
    assert build_pending_label(count) == expected


@pytest.mark.parametrize(
    "has_overdue, color",
    [
        (True, "red"),
        (False, "default"),
    ],
)
def test_badge_color_for(has_overdue: bool, color: str) -> None:
    assert badge_color_for(has_overdue) == color

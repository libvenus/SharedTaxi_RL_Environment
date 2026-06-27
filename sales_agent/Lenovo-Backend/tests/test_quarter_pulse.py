"""Unit tests for Quarter Pulse metric math (no database)."""

from __future__ import annotations

import pytest

from app.services.quarter_pulse import (
    _build_coverage_metric,
    _build_quota_metric,
    _coverage_band,
    _coverage_progress_fill,
    _quota_attainment_band,
)


@pytest.mark.parametrize(
    "percent, band, color",
    [
        (0, "low", "red"),
        (49.9, "low", "red"),
        (50, "medium", "blue"),
        (79.9, "medium", "blue"),
        (80, "high", "green"),
        (120, "high", "green"),
    ],
)
def test_quota_attainment_bands(percent: float, band: str, color: str) -> None:
    assert _quota_attainment_band(percent) == (band, color)
    metric = _build_quota_metric(percent)
    assert metric.band == band
    assert metric.bar_color == color
    assert metric.display_value == f"{round(percent)}%"


@pytest.mark.parametrize(
    "ratio, band, color",
    [
        (0.5, "low", "red"),
        (1.0, "medium", "yellow"),
        (2.0, "medium", "yellow"),
        (2.5, "high", "green"),
    ],
)
def test_pipeline_coverage_bands(ratio: float, band: str, color: str) -> None:
    assert _coverage_band(ratio) == (band, color)
    metric = _build_coverage_metric(ratio)
    assert metric.display_value == f"{ratio:.1f}x"


def test_coverage_progress_fill_caps_at_three_x() -> None:
    assert _coverage_progress_fill(0.0) == 0.0
    assert _coverage_progress_fill(1.5) == 50.0
    assert _coverage_progress_fill(3.0) == 100.0
    assert _coverage_progress_fill(10.0) == 100.0

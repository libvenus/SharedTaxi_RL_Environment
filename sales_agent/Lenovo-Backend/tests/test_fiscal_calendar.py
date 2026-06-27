"""Unit tests for app/services/fiscal_calendar.py."""

from __future__ import annotations

from datetime import date

import pytest

from app.services.fiscal_calendar import get_current_fiscal_period


@pytest.mark.parametrize(
    "today, expected_fy, expected_q, expected_days_left",
    [
        # FY starts April — FY2026 = Apr 2025 – Mar 2026
        (date(2025, 4, 1), 2026, 1, 89),
        (date(2025, 6, 30), 2026, 1, 0),
        (date(2025, 7, 1), 2026, 2, 91),
        (date(2025, 12, 31), 2026, 3, 0),
        (date(2026, 1, 15), 2026, 4, 75),
        (date(2026, 3, 31), 2026, 4, 0),
        (date(2026, 4, 1), 2027, 1, 89),
    ],
)
def test_fiscal_period_quarters_april_fy_start(
    today: date,
    expected_fy: int,
    expected_q: int,
    expected_days_left: int,
) -> None:
    period = get_current_fiscal_period(today, fy_start_month=4)
    assert period.fiscal_year == expected_fy
    assert period.quarter_number == expected_q
    assert period.quarter_label == f"Q{expected_q}"
    assert period.days_left == expected_days_left
    assert period.start_date <= today <= period.end_date


def test_fiscal_period_january_fy_start() -> None:
    period = get_current_fiscal_period(date(2026, 6, 19), fy_start_month=1)
    assert period.fiscal_year == 2026
    assert period.quarter_number == 2
    assert period.start_date == date(2026, 4, 1)
    assert period.end_date == date(2026, 6, 30)

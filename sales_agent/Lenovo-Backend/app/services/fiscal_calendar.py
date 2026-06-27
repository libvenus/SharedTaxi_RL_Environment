"""Fiscal-quarter helpers for Sprint 2 US 1.2 Quarter Pulse.

Lenovo fiscal years default to starting in April (configurable via
``FISCAL_YEAR_START_MONTH``). Quarters are three-month blocks from that
anchor: Q1 = months 0–2, Q2 = 3–5, Q3 = 6–8, Q4 = 9–11 after FY start.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from app.config import get_settings

DEFAULT_FISCAL_YEAR_START_MONTH = 4


@dataclass(frozen=True)
class FiscalPeriod:
    fiscal_year: int
    quarter_number: int
    quarter_label: str
    start_date: date
    end_date: date
    days_left: int


def fiscal_year_start_month() -> int:
    return get_settings().fiscal_year_start_month


def _fiscal_year_for(day: date, fy_start_month: int) -> int:
    """Return the fiscal-year label for ``day`` (the year the FY is named after)."""
    if day.month >= fy_start_month:
        return day.year + 1
    return day.year


def _fiscal_year_start_date(fiscal_year: int, fy_start_month: int) -> date:
    """First calendar day of the named fiscal year."""
    return date(fiscal_year - 1, fy_start_month, 1)


def get_current_fiscal_period(
    today: date | None = None,
    *,
    fy_start_month: int | None = None,
) -> FiscalPeriod:
    """Resolve the fiscal quarter containing ``today`` (default: UTC date)."""
    day = today or date.today()
    start_month = fy_start_month if fy_start_month is not None else fiscal_year_start_month()
    fiscal_year = _fiscal_year_for(day, start_month)
    fy_start = _fiscal_year_start_date(fiscal_year, start_month)

    months_since_fy_start = (day.year - fy_start.year) * 12 + (day.month - fy_start.month)
    quarter_number = months_since_fy_start // 3 + 1
    quarter_number = min(max(quarter_number, 1), 4)

    quarter_start = _add_months(fy_start, (quarter_number - 1) * 3)
    quarter_end = _add_months(quarter_start, 3) - timedelta(days=1)
    days_left = max((quarter_end - day).days, 0)

    return FiscalPeriod(
        fiscal_year=fiscal_year,
        quarter_number=quarter_number,
        quarter_label=f"Q{quarter_number}",
        start_date=quarter_start,
        end_date=quarter_end,
        days_left=days_left,
    )


def _add_months(anchor: date, months: int) -> date:
    """Add ``months`` calendar months to ``anchor`` (day clamped to month end)."""
    month_index = anchor.month - 1 + months
    year = anchor.year + month_index // 12
    month = month_index % 12 + 1
    # Clamp day to last day of target month.
    next_month = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
    last_day = (next_month - timedelta(days=1)).day
    return date(year, month, min(anchor.day, last_day))

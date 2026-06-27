"""Sprint 2 US 1.2 — Quarter Pulse summary for the Home dashboard."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Literal

from sqlalchemy import String, cast, func, select
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session

from app.models import Opportunity, SellerQuota
from app.services.fiscal_calendar import FiscalPeriod, get_current_fiscal_period

WON_STATECODES = ("Won", "Closed Won")
WON_STAGES = ("Closed Won",)

QuotaBand = Literal["low", "medium", "high"]
CoverageBand = Literal["low", "medium", "high"]
BarColor = Literal["red", "blue", "yellow", "green"]

QUOTA_NOT_SET_PROMPT = (
    "Set your quota target in D365 to see attainment metrics."
)

ERR_MSG_0021 = "ERR_MSG_0021"


@dataclass(frozen=True)
class _QuotaMetric:
    display_value: str
    percent: float | None
    progress_fill_percent: float | None
    band: QuotaBand | None
    bar_color: BarColor | None


@dataclass(frozen=True)
class _CoverageMetric:
    display_value: str
    ratio: float | None
    progress_fill_percent: float | None
    band: CoverageBand | None
    bar_color: BarColor | None


@dataclass(frozen=True)
class QuarterPulseData:
    quarter_label: str
    fiscal_year: int
    days_left_in_quarter: int
    last_updated_at: datetime
    quota_configured: bool
    quota_target: float | None
    closed_revenue: float
    open_pipeline_value: float
    open_deal_count: int
    quota_attainment: _QuotaMetric
    pipeline_coverage: _CoverageMetric
    prompt: str | None


def _has_table(db: Session, table_name: str) -> bool:
    bind = db.get_bind()
    if bind is None:
        return False
    return inspect(bind).has_table(table_name)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _normalise_seller_id(seller_id: str) -> str:
    return seller_id.strip().upper()


def _won_predicate():
    return Opportunity.statecode.in_(WON_STATECODES) | Opportunity.stagename.in_(WON_STAGES)


def _sum_estimated_value(db: Session, *where_clauses) -> float:
    stmt = select(func.coalesce(func.sum(Opportunity.estimatedvalue), 0)).where(*where_clauses)
    return float(db.execute(stmt).scalar_one() or 0)


def _count_opportunities(db: Session, *where_clauses) -> int:
    stmt = select(func.count()).select_from(Opportunity).where(*where_clauses)
    return int(db.execute(stmt).scalar_one() or 0)


def _seller_filter(seller_id: str):
    normalised = _normalise_seller_id(seller_id)
    return func.upper(cast(Opportunity.owninguser, String)) == normalised


def _closed_revenue_in_period(
    db: Session,
    seller_id: str,
    period: FiscalPeriod,
) -> float:
    return _sum_estimated_value(
        db,
        _seller_filter(seller_id),
        _won_predicate(),
        Opportunity.estimatedclosedate.is_not(None),
        Opportunity.estimatedclosedate >= period.start_date,
        Opportunity.estimatedclosedate <= period.end_date,
    )


def _open_pipeline_value(db: Session, seller_id: str) -> tuple[float, int]:
    clauses = (
        _seller_filter(seller_id),
        Opportunity.statecode == "Open",
    )
    value = _sum_estimated_value(db, *clauses)
    count = _count_opportunities(db, *clauses)
    return value, count


def _lookup_quota(
    db: Session,
    seller_id: str,
    period: FiscalPeriod,
) -> float | None:
    if not _has_table(db, SellerQuota.__tablename__):
        return None

    row = db.execute(
        select(SellerQuota.quota_amount)
        .where(
            func.upper(cast(SellerQuota.seller_id, String)) == _normalise_seller_id(seller_id),
            SellerQuota.fiscal_year == period.fiscal_year,
            SellerQuota.fiscal_quarter == period.quarter_number,
        )
        .limit(1)
    ).scalar_one_or_none()

    if row is None:
        return None
    amount = float(row)
    return amount if amount > 0 else None


def _quota_attainment_band(percent: float) -> tuple[QuotaBand, BarColor]:
    if percent < 50:
        return "low", "red"
    if percent < 80:
        return "medium", "blue"
    return "high", "green"


def _coverage_band(ratio: float) -> tuple[CoverageBand, BarColor]:
    if ratio < 1.0:
        return "low", "red"
    if ratio <= 2.0:
        return "medium", "yellow"
    return "high", "green"


def _coverage_progress_fill(ratio: float) -> float:
    """Map coverage ratio to a 0–100 bar width (caps at 3.0x)."""
    return min(max(ratio / 3.0 * 100.0, 0.0), 100.0)


def _build_quota_metric(percent: float) -> _QuotaMetric:
    band, color = _quota_attainment_band(percent)
    display = f"{round(percent)}%"
    fill = min(max(percent, 0.0), 100.0)
    return _QuotaMetric(
        display_value=display,
        percent=round(percent, 2),
        progress_fill_percent=round(fill, 2),
        band=band,
        bar_color=color,
    )


def _build_coverage_metric(ratio: float) -> _CoverageMetric:
    band, color = _coverage_band(ratio)
    return _CoverageMetric(
        display_value=f"{ratio:.1f}x",
        ratio=round(ratio, 2),
        progress_fill_percent=round(_coverage_progress_fill(ratio), 2),
        band=band,
        bar_color=color,
    )


def _not_set_quota_metric() -> _QuotaMetric:
    return _QuotaMetric(
        display_value="Not set",
        percent=None,
        progress_fill_percent=None,
        band=None,
        bar_color=None,
    )


def _not_set_coverage_metric() -> _CoverageMetric:
    return _CoverageMetric(
        display_value="Not set",
        ratio=None,
        progress_fill_percent=None,
        band=None,
        bar_color=None,
    )


def build_quarter_pulse(db: Session, seller_id: str) -> QuarterPulseData:
    """Compute live Quarter Pulse metrics for a seller from D365 mirror data."""
    period = get_current_fiscal_period()
    closed_revenue = _closed_revenue_in_period(db, seller_id, period)
    open_pipeline_value, open_deal_count = _open_pipeline_value(db, seller_id)
    quota_target = _lookup_quota(db, seller_id, period)

    if quota_target is None:
        return QuarterPulseData(
            quarter_label=period.quarter_label,
            fiscal_year=period.fiscal_year,
            days_left_in_quarter=period.days_left,
            last_updated_at=_utc_now(),
            quota_configured=False,
            quota_target=None,
            closed_revenue=closed_revenue,
            open_pipeline_value=open_pipeline_value,
            open_deal_count=open_deal_count,
            quota_attainment=_not_set_quota_metric(),
            pipeline_coverage=_not_set_coverage_metric(),
            prompt=QUOTA_NOT_SET_PROMPT,
        )

    attainment_pct = (closed_revenue / quota_target) * 100.0 if quota_target else 0.0
    remaining_quota = max(quota_target - closed_revenue, 0.0)

    if remaining_quota > 0:
        coverage_ratio = open_pipeline_value / remaining_quota
        coverage_metric = _build_coverage_metric(coverage_ratio)
    elif open_pipeline_value > 0 and quota_target > 0:
        coverage_metric = _build_coverage_metric(open_pipeline_value / quota_target)
    else:
        coverage_metric = _CoverageMetric(
            display_value="0.0x",
            ratio=0.0,
            progress_fill_percent=0.0,
            band="low",
            bar_color="red",
        )

    return QuarterPulseData(
        quarter_label=period.quarter_label,
        fiscal_year=period.fiscal_year,
        days_left_in_quarter=period.days_left,
        last_updated_at=_utc_now(),
        quota_configured=True,
        quota_target=quota_target,
        closed_revenue=closed_revenue,
        open_pipeline_value=open_pipeline_value,
        open_deal_count=open_deal_count,
        quota_attainment=_build_quota_metric(attainment_pct),
        pipeline_coverage=coverage_metric,
        prompt=None,
    )


def upsert_seller_quota(
    db: Session,
    *,
    seller_id: str,
    fiscal_year: int,
    fiscal_quarter: int,
    quota_amount: float,
    set_by: str | None,
    currency_code: str = "USD",
) -> SellerQuota:
    if not _has_table(db, SellerQuota.__tablename__):
        raise RuntimeError("lvo_seller_quota table is not available")

    if quota_amount <= 0:
        raise ValueError("quota_amount must be positive")

    normalised_seller = seller_id.strip()
    existing = db.execute(
        select(SellerQuota).where(
            func.upper(cast(SellerQuota.seller_id, String))
            == _normalise_seller_id(normalised_seller),
            SellerQuota.fiscal_year == fiscal_year,
            SellerQuota.fiscal_quarter == fiscal_quarter,
        )
    ).scalar_one_or_none()

    now = _utc_now()
    if existing is not None:
        existing.quota_amount = Decimal(str(quota_amount))
        existing.currency_code = currency_code
        existing.source = "manual"
        existing.set_by = set_by
        existing.modified_at = now
        db.commit()
        db.refresh(existing)
        return existing

    row = SellerQuota(
        lvo_sellerquotaid=str(uuid.uuid4()),
        seller_id=normalised_seller,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        quota_amount=Decimal(str(quota_amount)),
        currency_code=currency_code,
        source="manual",
        set_by=set_by,
        modified_at=now,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row

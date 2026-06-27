"""Sprint 2 US 1.2 — Quarter Pulse summary on the Home dashboard."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    QuarterPulseQuotaUpsertRequest,
    QuarterPulseQuotaUpsertResponse,
    QuarterPulseResponse,
    QuarterPulseAttainmentMetric,
    QuarterPulseCoverageMetric,
)
from app.services.fiscal_calendar import get_current_fiscal_period
from app.services.audit_log import write_audit_event
from app.services.quarter_pulse import (
    ERR_MSG_0021,
    QuarterPulseData,
    build_quarter_pulse,
    upsert_seller_quota,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/quarter-pulse", tags=["quarter-pulse"])


def _require_seller_id(seller_id: str | None) -> str:
    if not seller_id or not seller_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="sellerId is required.",
        )
    return seller_id.strip()


def _to_response(data: QuarterPulseData) -> QuarterPulseResponse:
    return QuarterPulseResponse(
        quarter_label=data.quarter_label,
        fiscal_year=data.fiscal_year,
        days_left_in_quarter=data.days_left_in_quarter,
        last_updated_at=data.last_updated_at,
        quota_configured=data.quota_configured,
        quota_target=data.quota_target,
        closed_revenue=data.closed_revenue,
        open_pipeline_value=data.open_pipeline_value,
        open_deal_count=data.open_deal_count,
        quota_attainment=QuarterPulseAttainmentMetric(
            display_value=data.quota_attainment.display_value,
            percent=data.quota_attainment.percent,
            progress_fill_percent=data.quota_attainment.progress_fill_percent,
            band=data.quota_attainment.band,
            bar_color=data.quota_attainment.bar_color,
        ),
        pipeline_coverage=QuarterPulseCoverageMetric(
            display_value=data.pipeline_coverage.display_value,
            ratio=data.pipeline_coverage.ratio,
            progress_fill_percent=data.pipeline_coverage.progress_fill_percent,
            band=data.pipeline_coverage.band,
            bar_color=data.pipeline_coverage.bar_color,
        ),
        prompt=data.prompt,
    )


@router.get(
    "/period",
    summary="Current fiscal period metadata — label, week-of-quarter, phase (no auth required)",
)
def get_fiscal_period() -> dict:
    """Return the current fiscal period for the app-wide header subtitle.

    Phase bands (out of ~13 weeks per quarter):
      weeks 1-4   → Ramp Phase
      weeks 5-9   → Build Phase
      weeks 10-11 → Execute Phase
      weeks 12+   → Closure Phase
    """
    import math
    from datetime import date

    fp = get_current_fiscal_period()
    today = date.today()

    days_elapsed = max((today - fp.start_date).days, 0)
    quarter_total_days = (fp.end_date - fp.start_date).days + 1
    total_weeks = math.ceil(quarter_total_days / 7)
    week_of_quarter = min(days_elapsed // 7 + 1, total_weeks)

    week_pct = week_of_quarter / total_weeks if total_weeks else 1
    if week_pct <= 4 / 13:
        phase = "Ramp Phase"
    elif week_pct <= 9 / 13:
        phase = "Build Phase"
    elif week_pct <= 11 / 13:
        phase = "Execute Phase"
    else:
        phase = "Closure Phase"

    return {
        "label": f"{fp.quarter_label} FY{fp.fiscal_year}",
        "quarterLabel": fp.quarter_label,
        "fiscalYear": fp.fiscal_year,
        "weekOfQuarter": week_of_quarter,
        "totalWeeks": total_weeks,
        "daysLeftInQuarter": fp.days_left,
        "phase": phase,
    }


@router.get(
    "",
    response_model=QuarterPulseResponse,
    summary="Quarter Pulse card — quota attainment, pipeline coverage, days left",
)
def get_quarter_pulse(
    seller_id: str | None = Query(
        default=None,
        alias="sellerId",
        description="Seller UUID — matches opportunity.owninguser.",
    ),
    db: Session = Depends(get_db),
) -> QuarterPulseResponse:
    seller = _require_seller_id(seller_id)
    try:
        return _to_response(build_quarter_pulse(db, seller))
    except Exception:
        logger.exception("Quarter Pulse failed for seller %s", seller)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERR_MSG_0021,
        ) from None


@router.put(
    "/quota",
    response_model=QuarterPulseQuotaUpsertResponse,
    summary="Set or update manual quota for the current fiscal quarter (Phase 1)",
)
def put_quarter_pulse_quota(
    body: QuarterPulseQuotaUpsertRequest,
    seller_id: str | None = Query(
        default=None,
        alias="sellerId",
        description="Seller UUID receiving the quota target.",
    ),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> QuarterPulseQuotaUpsertResponse:
    seller = _require_seller_id(seller_id)
    period = get_current_fiscal_period()
    fiscal_year = body.fiscal_year if body.fiscal_year is not None else period.fiscal_year
    fiscal_quarter = (
        body.fiscal_quarter if body.fiscal_quarter is not None else period.quarter_number
    )

    try:
        before = None
        from sqlalchemy import String, cast, func, select

        from app.models import SellerQuota

        existing = db.execute(
            select(SellerQuota).where(
                func.upper(cast(SellerQuota.seller_id, String)) == seller.upper(),
                SellerQuota.fiscal_year == fiscal_year,
                SellerQuota.fiscal_quarter == fiscal_quarter,
            )
        ).scalar_one_or_none()
        if existing is not None:
            before = {
                "quotaAmount": float(existing.quota_amount),
                "currencyCode": existing.currency_code,
            }

        row = upsert_seller_quota(
            db,
            seller_id=seller,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            quota_amount=body.quota_amount,
            set_by=x_user_id,
            currency_code=body.currency_code.upper(),
        )
        pulse = build_quarter_pulse(db, seller)
        write_audit_event(
            db,
            entity_type="seller_quota",
            entity_id=str(row.lvo_sellerquotaid),
            action="update" if before else "create",
            category="admin_action",
            actor_type="admin",
            changed_by=x_user_id,
            diff={
                "before": before,
                "after": {
                    "quotaAmount": float(row.quota_amount),
                    "currencyCode": row.currency_code,
                    "fiscalYear": fiscal_year,
                    "fiscalQuarter": fiscal_quarter,
                    "sellerId": seller,
                },
            },
        )
        db.commit()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("Quota upsert failed for seller %s", seller)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERR_MSG_0021,
        ) from None

    return QuarterPulseQuotaUpsertResponse(
        seller_id=row.seller_id,
        fiscal_year=row.fiscal_year,
        fiscal_quarter=row.fiscal_quarter,
        quota_amount=float(row.quota_amount),
        currency_code=row.currency_code,
        source=row.source,  # type: ignore[arg-type]
        set_by=row.set_by,
        modified_at=row.modified_at,
        quarter_pulse=_to_response(pulse),
    )

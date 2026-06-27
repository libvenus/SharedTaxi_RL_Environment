"""Sprint 2 — Pre-meeting briefing D365 facts API."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import BriefingContextResponse
from app.services.briefing_context import (
    build_briefing_context,
    BriefingContextData,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/briefing", tags=["briefing"])

ERR_MSG_0024 = "ERR_MSG_0024"


def _require_seller_id(seller_id: str | None) -> str:
    if not seller_id or not seller_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="sellerId is required.",
        )
    return seller_id.strip()


def _require_opportunity_id(opportunity_id: str | None) -> str:
    if not opportunity_id or not opportunity_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="opportunityId is required.",
        )
    return opportunity_id.strip()


def _to_response(data: BriefingContextData) -> BriefingContextResponse:
    from app.schemas import (
        BriefingAccountFactsResponse,
        BriefingCompetitorItemResponse,
        BriefingDealFactsResponse,
        BriefingFactFieldResponse,
        BriefingSignalItemResponse,
        BriefingSourceRefResponse,
    )

    def _src(s):
        if s is None:
            return None
        return BriefingSourceRefResponse(
            source_type=s.source_type,
            source_id=s.source_id,
            label=s.label,
        )

    def _field(f):
        return BriefingFactFieldResponse(
            field_name=f.field_name,
            display_label=f.display_label,
            value=f.value,
            is_missing=f.is_missing,
            is_unverified=f.is_unverified,
            source=_src(f.source),
        )

    competitors = None
    if data.deal.competitor_intel:
        competitors = [
            BriefingCompetitorItemResponse(
                competitor_name=c.competitor_name,
                competitor_type=c.competitor_type,
                reselling_partner=c.reselling_partner,
                primary_risk=c.primary_risk,
                source=_src(c.source),
            )
            for c in data.deal.competitor_intel
        ]

    return BriefingContextResponse(
        seller_id=data.seller_id,
        opportunity_id=data.opportunity_id,
        account_id=data.account_id,
        generated_at=data.generated_at,
        account=BriefingAccountFactsResponse(
            account_id=data.account.account_id,
            account_name=data.account.account_name,
            fields=[_field(f) for f in data.account.fields],
            paragraph=data.account.paragraph,
            word_count=data.account.word_count,
            max_words=data.account.max_words,
            gaps=data.account.gaps,
            unverified_labels=data.account.unverified_labels,
        ),
        deal=BriefingDealFactsResponse(
            opportunity_id=data.deal.opportunity_id,
            opportunity_name=data.deal.opportunity_name,
            stage=data.deal.stage,
            fields=[_field(f) for f in data.deal.fields],
            paragraph=data.deal.paragraph,
            word_count=data.deal.word_count,
            max_words=data.deal.max_words,
            gaps=data.deal.gaps,
            competitor_intel=competitors,
            competitor_message_code=data.deal.competitor_message_code,
        ),
        signals=[
            BriefingSignalItemResponse(
                signal_id=s.signal_id,
                summary=s.summary,
                why_shown=s.why_shown,
                event_at=s.event_at,
                involved_parties=s.involved_parties,
                source=_src(s.source),
            )
            for s in data.signals
        ],
    )


@router.get(
    "/context",
    response_model=BriefingContextResponse,
    summary="D365 facts + signals for pre-meeting briefing generation",
)
def get_briefing_context(
    seller_id: str | None = Query(default=None, alias="sellerId"),
    opportunity_id: str | None = Query(default=None, alias="opportunityId"),
    account_id: str | None = Query(default=None, alias="accountId"),
    max_summary_words: int = Query(
        default=100,
        ge=50,
        le=200,
        alias="maxSummaryWords",
    ),
    db: Session = Depends(get_db),
) -> BriefingContextResponse:
    seller = _require_seller_id(seller_id)
    opp = _require_opportunity_id(opportunity_id)
    try:
        return _to_response(
            build_briefing_context(
                db,
                seller,
                opp,
                account_id=account_id,
                max_summary_words=max_summary_words,
            )
        )
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Seller does not own this opportunity.",
        ) from None
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Opportunity not found.",
        ) from None
    except Exception:
        logger.exception("Briefing context failed for seller %s opp %s", seller, opp)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERR_MSG_0024,
        ) from None

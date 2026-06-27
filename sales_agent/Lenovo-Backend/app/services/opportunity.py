"""Opportunity competitor persistence helpers."""

from __future__ import annotations

import uuid

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Opportunity, OpportunityCompetitor
from app.schemas import (
    CompetitorResponse,
    CompetitorsRequest,
    CompetitorsResponse,
)


def _to_competitor_response(row: OpportunityCompetitor) -> CompetitorResponse:
    name = row.lvo_name or row.lvo_competitorname or ""
    return CompetitorResponse(
        id=row.lvo_opportunitycompetitorid,
        opportunityId=row.lvo_opportunityid or "",
        name=name,
        competitorName=row.lvo_competitorname or "",
        competitorType=row.lvo_competitortype,
        resellingPartnerId=row.lvo_resellingpartner,
    )


def get_opportunity_competitors(
    db: Session,
    opportunity_id: str,
) -> CompetitorsResponse:
    """Return active competitors for one opportunity."""
    rows = (
        db.execute(
            select(OpportunityCompetitor)
            .where(
                func.upper(OpportunityCompetitor.lvo_opportunityid)
                == opportunity_id.strip().upper(),
                OpportunityCompetitor.statecode == "Active",
            )
            .order_by(OpportunityCompetitor.lvo_competitorname)
        )
        .scalars()
        .all()
    )
    return CompetitorsResponse(
        competitors=[_to_competitor_response(row) for row in rows]
    )


def save_competitors(
    db: Session,
    request: CompetitorsRequest,
) -> CompetitorsResponse:
    """Create or update a batch of opportunity competitors."""
    if not request.competitors:
        return CompetitorsResponse(competitors=[])

    saved: list[CompetitorResponse] = []
    try:
        for item in request.competitors:
            opportunity_id = item.opportunityId.strip()
            if not opportunity_id:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="opportunityId is required for each competitor.",
                )

            opp = db.get(Opportunity, opportunity_id)
            if opp is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Opportunity '{opportunity_id}' not found.",
                )

            competitor_name = item.competitorName.strip()
            if not competitor_name:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="competitorName is required.",
                )

            if item.id:
                row = db.get(OpportunityCompetitor, item.id)
                if row is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Competitor '{item.id}' not found.",
                    )
            else:
                row = OpportunityCompetitor(
                    lvo_opportunitycompetitorid=str(uuid.uuid4()),
                    statecode="Active",
                )
                db.add(row)

            row.lvo_opportunityid = opportunity_id.upper()
            row.lvo_competitorname = competitor_name
            row.lvo_name = item.name or competitor_name
            row.lvo_competitortype = item.competitorType
            row.lvo_resellingpartner = item.resellingPartnerId
            row.statecode = "Active"
            saved.append(_to_competitor_response(row))

        db.commit()
        return CompetitorsResponse(competitors=saved)
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

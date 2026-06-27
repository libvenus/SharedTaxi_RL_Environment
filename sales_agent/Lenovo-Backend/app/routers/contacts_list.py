"""GET /api/contacts — portfolio contact roster for a seller."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import SellerContactListResponse
from app.services.seller_contacts import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    list_seller_contacts,
)

router = APIRouter(prefix="/api/contacts", tags=["contacts"])


@router.get(
    "",
    response_model=SellerContactListResponse,
    summary="Paginated contacts across a seller's open opportunity portfolio",
)
def get_seller_contacts(
    seller_id: str | None = Query(
        default=None,
        alias="sellerId",
        description="Seller UUID — matches opportunity.owninguser.",
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(
        DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        alias="pageSize",
    ),
    db: Session = Depends(get_db),
) -> SellerContactListResponse:
    """Return de-duplicated contacts linked to the seller's open deals.

    Contacts are sourced from ``lvo_opportunitycontact`` rows on opportunities
    where ``opportunity.owninguser = sellerId`` and ``statecode = Open``.
    When a contact appears on multiple deals, one representative row is
    returned with ``linkedOpportunityCount`` set accordingly.
    """
    if not seller_id or not seller_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="sellerId is required.",
        )
    seller = seller_id.strip()
    items, total, total_pages = list_seller_contacts(
        db, seller, page=page, page_size=page_size
    )
    return SellerContactListResponse(
        seller_id=seller,
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        items=items,
    )

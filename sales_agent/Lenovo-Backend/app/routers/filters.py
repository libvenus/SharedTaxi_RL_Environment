"""Filter dropdown endpoints used by the Opportunities UI.

Endpoints
---------
- GET /api/filters/regions      (#8)
- GET /api/filters/industries   (#9)
- GET /api/filters/stages       (#10)
- GET /api/filters/products     (#11)
"""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    Account,
    GeographicUnit,
    LegalEntity,
    Opportunity,
    QuoteItem,
    SolutionOffering,
)
from app.normalizers import normalise_stage, slugify
from app.schemas import (
    BusinessGroupOption,
    CountryOption,
    GeographicUnitOption,
    IndustriesResponse,
    IndustryOption,
    ProductOption,
    ProductsResponse,
    RegionsResponse,
    StageOption,
    StagesResponse,
)

router = APIRouter(prefix="/api/filters", tags=["filters"])


# ---------------------------------------------------------------------------
# #8 — Regions
# ---------------------------------------------------------------------------
@router.get(
    "/regions",
    response_model=RegionsResponse,
    summary="Regions filter — business groups, countries and geographic units",
)
def list_regions(db: Session = Depends(get_db)) -> RegionsResponse:
    bg_rows = db.execute(
        select(LegalEntity.lvo_bgid)
        .where(LegalEntity.lvo_bgid.is_not(None))
        .distinct()
        .order_by(LegalEntity.lvo_bgid)
    ).all()
    business_groups = [
        BusinessGroupOption(id=row[0], label=row[0]) for row in bg_rows
    ]

    country_rows = db.execute(
        select(LegalEntity.lvo_countryid, LegalEntity.lvo_bgid)
        .where(LegalEntity.lvo_countryid.is_not(None))
        .distinct()
        .order_by(LegalEntity.lvo_countryid)
    ).all()
    countries = [
        CountryOption(code=code, label=code, business_group=bg)
        for code, bg in country_rows
    ]

    geo_units = (
        db.execute(
            select(GeographicUnit)
            .where(GeographicUnit.statecode == "Active")
            .order_by(GeographicUnit.lvo_name)
        )
        .scalars()
        .all()
    )
    geographic_units = [
        GeographicUnitOption(
            id=g.lvo_geographicunitid,
            name=g.lvo_name,
            code=g.lvo_geounitcode,
            parent_id=g.lvo_parentunitid,
        )
        for g in geo_units
    ]

    return RegionsResponse(
        business_groups=business_groups,
        countries=countries,
        geographic_units=geographic_units,
    )


# ---------------------------------------------------------------------------
# #9 — Industries
# ---------------------------------------------------------------------------
@router.get(
    "/industries",
    response_model=IndustriesResponse,
    summary="Industries filter — distinct values of account.industrycode",
)
def list_industries(db: Session = Depends(get_db)) -> IndustriesResponse:
    rows = db.execute(
        select(Account.industrycode)
        .where(Account.industrycode.is_not(None))
        .distinct()
        .order_by(Account.industrycode)
    ).all()
    items = [IndustryOption(code=code, label=code) for (code,) in rows]
    return IndustriesResponse(total=len(items), items=items)


# ---------------------------------------------------------------------------
# #10 — Stages
# ---------------------------------------------------------------------------
@router.get(
    "/stages",
    response_model=StagesResponse,
    summary="Stage filter — distinct opportunity.stagename with UI labels",
)
def list_stages(db: Session = Depends(get_db)) -> StagesResponse:
    rows = db.execute(
        select(Opportunity.stagename)
        .where(Opportunity.stagename.is_not(None))
        .distinct()
        .order_by(Opportunity.stagename)
    ).all()
    items = [
        StageOption(raw=stage, label=normalise_stage(stage) or stage)
        for (stage,) in rows
    ]
    return StagesResponse(total=len(items), items=items)


# ---------------------------------------------------------------------------
# #11 — Products
# ---------------------------------------------------------------------------
@router.get(
    "/products",
    response_model=ProductsResponse,
    summary="Products filter — product series from quote items + solution offerings",
)
def list_products(db: Session = Depends(get_db)) -> ProductsResponse:
    quote_rows = db.execute(
        select(QuoteItem.lvo_productseries)
        .where(QuoteItem.lvo_productseries.is_not(None))
        .distinct()
    ).all()
    offering_rows = db.execute(
        select(SolutionOffering.lvo_name)
        .where(
            SolutionOffering.lvo_name.is_not(None),
            SolutionOffering.statecode == "Active",
        )
        .distinct()
    ).all()

    items: list[ProductOption] = []
    seen: set[str] = set()

    for (series,) in quote_rows:
        key = series.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(
            ProductOption(id=slugify(series), label=series, source="quoteitem")
        )

    for (offering,) in offering_rows:
        key = offering.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(
            ProductOption(
                id=slugify(offering), label=offering, source="solutionoffering"
            )
        )

    items.sort(key=lambda p: p.label.lower())
    return ProductsResponse(total=len(items), items=items)

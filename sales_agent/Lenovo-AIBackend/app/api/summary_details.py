from fastapi import (
    APIRouter,
    Depends
)

from sqlalchemy.orm import Session

from app.db.database import get_db

from app.schema.summary_details import (
    SummaryDetailsRequest
)

from app.services.summary_details import (
    create_summary_details
)

router = APIRouter(
    prefix="/summary-details",
    tags=["Summary Details"]
)


@router.post("/")
def save_summary_details(
    request: SummaryDetailsRequest,
    db: Session = Depends(get_db)
):
    return create_summary_details(
        db=db,
        payload=request.payload
    )
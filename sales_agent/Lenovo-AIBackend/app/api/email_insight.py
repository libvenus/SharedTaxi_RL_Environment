"""FastAPI router for email outreach insight.

GET /ai-api/emails/insight?opportunityName=<name>

Reads the latest email for the opportunity (matched by name) from tbl_emails,
runs the LLM layer, and returns ``why_now`` + ``latest_activity``.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schema.email_insight import EmailInsightResponse
from app.services.email_insight import generate_latest_email_insight

router = APIRouter(
    prefix="/emails",
    tags=["Email Insight"],
)


@router.get(
    "/insight",
    response_model=EmailInsightResponse,
    summary="Generate why_now + latest_activity from the latest opportunity email",
)
async def get_email_insight(
    opportunity_name: str = Query(..., alias="opportunityName"),
    db: Session = Depends(get_db),
) -> EmailInsightResponse:
    return await generate_latest_email_insight(db, opportunity_name)

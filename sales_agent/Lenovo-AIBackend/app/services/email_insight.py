"""Service layer for the email-insight endpoint.

One call: fetch the latest ``tbl_emails`` row for an opportunity (matched by
name), pull the opportunity value from CRM, summarise the body via the LLM, and
return everything the UI needs. Only ``body`` is sent to the LLM.
"""

from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.generate_ai_data.email_insight import (
    EmailInsightRequest,
    generate_email_insight,
)
from app.models.crm import CrmOpportunity
from app.models.email import TblEmail
from app.schema.email_insight import EmailInsightResponse


async def generate_latest_email_insight(
    db: Session,
    opportunity_name: str,
) -> EmailInsightResponse:
    """Build the combined outreach payload for an opportunity (by name)."""
    try:
        email = (
            db.query(TblEmail)
            .filter(TblEmail.opportunity_name == opportunity_name)
            .order_by(TblEmail.created_at.desc())
            .first()
        )

        if not email:
            raise HTTPException(
                status_code=404,
                detail=f"No email found for opportunity_name '{opportunity_name}'",
            )

        if not (email.body or "").strip():
            raise HTTPException(
                status_code=422,
                detail=f"Email {email.message_id} has no body to summarise",
            )

        opportunity = None
        if email.opportunity_id is not None:
            opportunity = (
                db.query(CrmOpportunity)
                .filter(CrmOpportunity.opportunityid == str(email.opportunity_id))
                .first()
            )

        insight = await generate_email_insight(
            EmailInsightRequest(body=email.body)
        )

        return EmailInsightResponse(
            opportunity_id=email.opportunity_id,
            opportunity_name=email.opportunity_name,
            account_id=email.account_id,
            account_name=email.account_name,
            opportunity_value=opportunity.estimatedvalue if opportunity else None,
            classification=insight["classification"] or email.classification,
            intent_category=email.intent_category,
            why_now=insight["why_now"],
            latest_activity=insight["latest_activity"],
            email_id=email.message_id,
            trace_id=insight["trace_id"],
            latency_ms=insight["latency_ms"],
            timestamp=insight["timestamp"],
            error=insight["error"],
        )

    except HTTPException:
        raise

    except SQLAlchemyError:
        raise HTTPException(
            status_code=500,
            detail="Database error occurred while fetching the email.",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate email insight: {str(e)}",
        )

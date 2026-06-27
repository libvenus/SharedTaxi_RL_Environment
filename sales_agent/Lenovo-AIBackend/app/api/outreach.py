from enum import Enum
import uuid
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
import traceback

from fastapi import APIRouter
from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.models.email_request import EmailRequest
from app.services.email_service import EmailService
from app.services.outreach import get_outreachs,get_outreach_details,create_email_draft_service
from app.schema.outreach import OutreachCategory, CreateEmailDraftRequest

router = APIRouter(
    prefix="/outreach",
    tags=["Outreach"]
)





@router.get("")
def fetch_outreachs(
    category: OutreachCategory | None = Query(None),
    db: AsyncSession = Depends(get_db)
):
    return  get_outreachs(
        db=db,
        category=category
    )






@router.post("/send-email")
async def send_email(
    request: EmailRequest
):

    try:

        EmailService.send_email(
            to_emails=request.to,
            cc_emails=request.cc,
            subject=request.subject,
            body=request.body,
        )

        return {
            "success": True,
            "message": "Email sent successfully"
        }

    except Exception as ex:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )
    


@router.get("/{outreach_id}")
async def get_outreach(
    outreach_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    return get_outreach_details(
        db=db,
        outreach_id=outreach_id
    )

@router.post("/email-draft")
async def create_email_draft(
    request: CreateEmailDraftRequest,
    db: Session = Depends(get_db)
):
    return await create_email_draft_service(db, request)
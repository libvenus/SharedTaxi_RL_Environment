from fastapi import HTTPException
import uuid
import uuid
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.email_template import EmailTemplate
from app.models.outreach import TblOutreach
from app.models.to_do_list import TblToDoList
from app.schema.outreach import DraftEmailRequest, OutreachCategory, CreateEmailDraftRequest,DraftEmailRequest
from app.generate_ai_data.email_draft import draft_email
from app.services.summary_details import get_opportunity_by_id

def get_outreachs(
    db: AsyncSession,
    category: OutreachCategory | None = None
):

    # Counts
    count_query = select(
        func.count(
            case(
                (TblOutreach.outreach_type == "MEETING FOLLOW-UP", 1),
                else_=None
            )
        ).label("meeting_follow_ups"),

        func.count(
            case(
                (TblOutreach.outreach_type == "ACCOUNT", 1),
                else_=None
            )
        ).label("account_outreach"),

        func.count(
            case(
                (TblOutreach.priority_badge == "HIGH PRIORITY", 1),
                else_=None
            )
        ).label("high_priority"),

        func.count(
            case(
                (TblOutreach.outreach_type == "SILENT/AT-RISK", 1),
                else_=None
            )
        ).label("silent_at_risk"),

        func.count(TblOutreach.id).label("total")
    )

    count_result =  db.execute(count_query)
    counts = count_result.one()

    # Data Query
    query = select(TblOutreach)

    if category:

        if category == OutreachCategory.MEETING_FOLLOW_UP:
            query = query.where(
                TblOutreach.outreach_type == "MEETING FOLLOW-UP"
            )

        elif category == OutreachCategory.ACCOUNT:
            query = query.where(
                TblOutreach.outreach_type == "ACCOUNT"
            )

        elif category == OutreachCategory.HIGH_PRIORITY:
            query = query.where(
                TblOutreach.priority_badge == "HIGH PRIORITY"
            )

        elif category == OutreachCategory.SILENT_AT_RISK:
            query = query.where(
                TblOutreach.outreach_type == "SILENT/AT-RISK"
            )

    result = db.execute(query)
    records = result.scalars().all()

    return {
        "counts": {
            "meeting_follow_ups": counts.meeting_follow_ups,
            "account_outreach": counts.account_outreach,
            "silent_at_risk": counts.silent_at_risk,
            "high_priority": counts.high_priority,
            "total": counts.total
        },
        "records": records
    }

def get_outreach_details(db, outreach_id: uuid.UUID):
    outreach = (
        db.query(TblOutreach)
        .filter(TblOutreach.id == outreach_id)
        .first()
    )

    if not outreach:
        raise HTTPException(
            status_code=404,
            detail=f"Outreach with id {outreach_id} not found"
        )

    return {
        "id": outreach.id,
        "deal_name": outreach.deal_name,
        "deal_stage": outreach.deal_stage,
        "deal_value": outreach.deal_value,
        "decision_maker_name": outreach.decision_maker_name,
        "decision_maker_role": outreach.decision_maker_role,
        "why_now_reason": outreach.why_now_reason,
        "outreach_date": outreach.outreach_date,
        "last_activity_type": outreach.last_activity_type,
        "last_activity_datetime": outreach.last_activity_datetime,
        "last_activity_summary": outreach.last_activity_summary,
        "title": outreach.title,
        "engagement_type": outreach.engagement_type,
        "attendees_email": outreach.attendees_email,
        "category": outreach.category,
        "outreach_type": outreach.outreach_type,
        "priority_badge": outreach.priority_badge,
        "account_name": outreach.account_name,
        "company_name": outreach.company_name,
        "email_template": get_email_templates(db)
    }


def get_email_templates(db):
    selected_template='Post-Meeting Follow-up'
    templates = (
        db.query(
            EmailTemplate.template_name,
            EmailTemplate.context_used
        )
        .filter(EmailTemplate.is_active == True)
        .all()
    )

    return [
        {
            "template_name": template.template_name,
            "context_used": template.context_used,
            "default": template.template_name == selected_template
        }
        for template in templates
    ]

async def create_email_draft_service(
    db: Session,
    request: CreateEmailDraftRequest
):
    try:
        todo = (
                db.query(TblToDoList)
                 .filter(TblToDoList.id == request.id)
                 .first()
             )

        if not todo:
                 raise HTTPException(
                     status_code=404,
                     detail=f"No Outreach or ToDo item found for id {request.id}"
                 )
        context_data = " | ".join(filter(None, [todo.task_title or "", todo.notes or "", request.template_name or "", request.context_used or ""]))
        opportunity_data = get_opportunity_by_id(db, todo.linked_opportunity_id)
        
        data = {
                "task_title": todo.task_title,
                "priority": todo.priority,
                "source_label": todo.source_label,
                "notes": todo.notes,
                "status": todo.status,
                "due_date": str(todo.due_date) if todo.due_date else None,
                "opportunity_name": opportunity_data.get("name"),
                "opportunity_deal_stage": opportunity_data.get("stagename"), 
                "opportunity_deal_value": opportunity_data.get("estimatedvalue")  

            }
        # outreach = (
        #     db.query(TblOutreach)
        #     .filter(TblOutreach.id == request.outreach_id)
        #     .first()
        # )

        # if outreach:
        #     data = {
        #         "account_name": outreach.account_name,
        #         "company_name": outreach.company_name,
        #         "contact_name": outreach.decision_maker_name
        #     }
        # else:
        #     todo = (
        #         db.query(TblToDoList)
        #         .filter(TblToDoList.id == request.outreach_id)
        #         .first()
        #     )

        #     if not todo:
        #         raise HTTPException(
        #             status_code=404,
        #             detail=f"No Outreach or ToDo item found for id {request.outreach_id}"
        #         )

            # data = {
            #     "task_title": todo.task_title,
            #     "priority": todo.priority,
            #     "source_label": todo.source_label,
            #     "notes": todo.notes,
            #     "status": todo.status,
            #     "due_date": str(todo.due_date) if todo.due_date else None
            # }

        prompt_payload = DraftEmailRequest(
            data=data,
            template="Hi {{recipient_name}},\n\nI wanted to follow up on {{deal_name}}...\n\nBest regards,\nSales Team",
            placeholders={
    "recipient_name": "John",
    "deal_name": "CRM Implementation"
  },
            context=context_data,
            written_context=request.additional_context or ""
        )

        # AI draft generation
        draft_response = await draft_email(prompt_payload)

        return {
            "success": True,
            "draft": draft_response
        }

    except HTTPException:
        raise

    except SQLAlchemyError as e:
       

        raise HTTPException(
            status_code=500,
            detail="Database error occurred while fetching data."
        )

    except Exception as e:
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create email draft: {str(e)}"
        )
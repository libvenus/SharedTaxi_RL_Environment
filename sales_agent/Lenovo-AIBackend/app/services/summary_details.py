from uuid import UUID
from sqlalchemy.orm import Session

from app.models.summaryDetails import SummaryDetails
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from dateutil import parser
from datetime import date, datetime
from fastapi import HTTPException

from app.models.to_do_list import TblToDoList
from app.models.post_meet import CrmUpdate

def parse_date(value):
    # Handle "no value" cases: None, empty, or non-string junk.
    if value is None:
        return None

    # Already a date/datetime object — return its date directly.
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    # Numeric case: treat as a Unix timestamp (seconds since epoch).
    if isinstance(value, (int, float)):
        try:
            return datetime.utcfromtimestamp(value).date()
        except (ValueError, OverflowError, OSError):
            return None

    # Coerce to a clean string; empty/whitespace means "no date".
    text_value = str(value).strip()
    if not text_value:
        return None

    # Pure-numeric string → also treat as Unix timestamp.
    if text_value.isdigit() or (text_value.startswith("-") and text_value[1:].isdigit()):
        try:
            return datetime.utcfromtimestamp(int(text_value)).date()
        except (ValueError, OverflowError, OSError):
            return None

    # String case: try to parse real dates; relative terms like
    # "tomorrow" or "end of week" that can't be parsed fall back to None.
    try:
        return parser.parse(text_value, dayfirst=True).date()
    except (ValueError, OverflowError, TypeError):
        return None
def parse_time(value):
    if not value:
        return None

    # Times can arrive as a range like "10:00 PM - 11:00 PM EST"; take the
    # start time only (split on hyphen or en/em dash) and strip a trailing
    # timezone abbreviation that dateutil cannot parse on its own.
    text_value = str(value)
    for separator in ("–", "—", "-"):
        if separator in text_value:
            text_value = text_value.split(separator)[0]
            break

    try:
        return parser.parse(text_value, fuzzy=True).time()
    except Exception:
        return None  


def parse_end_time(value):
    if not value:
        return None

    # Times can arrive as a range like "10:00 PM - 11:00 PM EST"; take the
    # end time only (the part after the hyphen / en/em dash). When there is
    # no range separator, there is no distinct end time to capture.
    text_value = str(value)
    for separator in ("–", "—", "-"):
        if separator in text_value:
            text_value = text_value.split(separator, 1)[1]
            break
    else:
        return None

    try:
        return parser.parse(text_value, fuzzy=True).time()
    except Exception:
        return None


def get_attendees_email(attendees):
    emails = [
        attendee.get("email", "").strip()
        for attendee in attendees
        if attendee.get("email", "").strip()
    ]
    return ",".join(emails) if emails else None


def get_opportunity_by_id(db: Session, opportunity_id):
    """Fetch stagename, name and estimatedvalue for a linked opportunity.

    Returns a dict with ``name`` / ``stagename`` / ``estimatedvalue`` (all
    ``None`` when the opportunity id is missing or no row matches).
    """
    if not opportunity_id:
        return {"name": None, "stagename": None, "estimatedvalue": None}

    result = db.execute(
        text("""
            SELECT
                name,
                stagename,
                estimatedvalue
            FROM opportunity
            WHERE LOWER(opportunityid) = LOWER(:opportunity_id)
            LIMIT 1
        """),
        {"opportunity_id": str(opportunity_id)}
    )

    opportunity = result.fetchone()

    if not opportunity:
        return {"name": None, "stagename": None, "estimatedvalue": None}

    return {
        "name": opportunity.name,
        "stagename": opportunity.stagename,
        "estimatedvalue": opportunity.estimatedvalue,
    }
   
def save_todo_list(db, meeting_id, attendees_email,account_id, todo_list):

    for item in todo_list:

        # dueDate may be an explicit date ("2026-06-26") or a relative phrase
        # like "tomorrow" / "end of week" that has no concrete date; store
        # None when it cannot be parsed into a real date.
        due_date = parse_date(item.get("dueDate"))

        todo = TblToDoList(
            meeting_id=meeting_id,
            task_title=item.get("task"),
            type_tag="outreach",
            priority=item.get("confidence"),
            source_label="AI generated post-meet",
            status=item.get("status"),
            due_date=due_date,
            attendees_email=attendees_email,
            linked_account_id=account_id,
            seller_id='055dafe7-9840-451d-8328-5f70a6326c03',
            # notes=f"Owner: {item.get('owner')}, TranscriptRefs: {', '.join(item.get('transcriptRefs', []))}"
        )

        db.add(todo)

    db.commit()    

def save_crm_updates(db, meeting_id, account_id, seller_id, crm_updates):

    for index, item in enumerate(crm_updates, start=1):

        # Derive a numeric seq_id from ids like "DU001"; fall back to the
        # enumeration index when no digits are present.
        raw_id = str(item.get("id", "")).strip()
        digits = "".join(ch for ch in raw_id if ch.isdigit())
        seq_id = int(digits) if digits else index

        crm_update = CrmUpdate(
            meeting_id=meeting_id,
            seller_id=seller_id,
            account_id=account_id,
            seq_id=seq_id,
            entity=item.get("entity"),
            field_name=item.get("field"),
            current_value=item.get("currentValue"),
            suggested_value=item.get("suggestedValue"),
            reasoning=item.get("reasoning"),
            confidence=item.get("confidence"),
            status="open",
        )

        db.add(crm_update)

    db.commit()


def create_summary_details(
    db: Session,
    
    payload: dict
):
    raw_payload = payload
    if isinstance(payload.get("summary"), dict):
        summary_payload = payload["summary"]
    else:
        summary_payload = payload
    meeting_id = raw_payload.get("meeting_id")
    meeting_header = summary_payload.get("meetingHeader", {})
    meeting_summary = summary_payload.get("meetingSummary", {})
    transcript = summary_payload.get("transcript", {})

    crm_updates = meeting_summary.get("crmUpdates", {})

   
    date_value =  parse_date(meeting_header.get("date"))
    time_value =  parse_time(meeting_header.get("time"))
    end_time_value = parse_end_time(meeting_header.get("time"))

    time_since_meeting = None
    meeting_end_time = None

    if date_value and time_value:
        time_since_meeting = datetime.combine(
            date_value, time_value
        ).strftime("%Y-%m-%d %H:%M:%S")
    elif date_value:
        time_since_meeting = date_value.strftime("%Y-%m-%d")

    if date_value and end_time_value:
        meeting_end_time = datetime.combine(
            date_value, end_time_value
        ).strftime("%Y-%m-%d %H:%M:%S")
    print("time_since_meeting", summary_payload.get("crmUpdates", {}).get("accountName", {}).get("value"))
    account_name = meeting_summary.get("crmUpdates", {}).get("accountName", {}).get("value")
    print("account_name", account_name)
    if not account_name:
       raise HTTPException(
         status_code=400,
         detail=f"Required fields missing or empty: account name"
    )
     # Step 1: Fetch accountid
    account_result = db.execute(
        text("""
            SELECT accountid
            FROM account
            WHERE LOWER(name) = LOWER(:account_name)
            LIMIT 1
        """),
        {"account_name": account_name}
    )

    account = account_result.fetchone()

    if not account:
        account_id=None
        opportunity_id = None
        opportunity_name = None
        deal_stage = None
        deal_value = None 
    else:
        account_id = account.accountid

    # Step 2: Fetch opportunity details
        opportunity_result = db.execute(
        text("""
            SELECT
               opportunityid,
                name,
                stagename,
                estimatedvalue
            FROM opportunity
            WHERE LOWER(accountid) = LOWER(:account_id)
            LIMIT 1
        """),
        {"account_id": str(account_id)}
    )

        opportunity = opportunity_result.fetchone()

        if not opportunity:
           opportunity_id = None
           opportunity_name = None
           deal_stage = None
           deal_value = None 
        else:
           opportunity_id = opportunity.opportunityid
           opportunity_name = opportunity.name
           deal_stage = opportunity.stagename
           deal_value = opportunity.estimatedvalue
              
    crm_updates_data= meeting_summary.get("dealUpdatesDetected", {})
    
    
    required_fields = {
    "meeting_id": meeting_id,
        "meeting_title": meeting_header.get("title"),
        "account_name": account_name,
        "customer_sentiment": meeting_summary.get(
            "sentiment", {}
        ).get("customerSentiment"),
        "attendees": meeting_header.get("attendees"),
        "crm_updates_pending_approval": crm_updates,
        "key_points":
            meeting_summary.get("keyPoints", [])
        ,
        "next_steps":
            meeting_summary.get("confirmedNextSteps", [])
        ,
        "summary": meeting_summary.get(
            "summaryQualityNotes", []
        ),
        # Optional fields are intentionally not required:
        # "time_since_meeting", "time", "meeting_time_duration", "meeting_platform"
}

    missing_fields = [
    field_name
    for field_name, value in required_fields.items()
    if value in [None, "", [], {}]
]

    if missing_fields:
       raise HTTPException(
         status_code=400,
         detail=f"Required fields missing or empty: {', '.join(missing_fields)}"
    )
    attendees_email = get_attendees_email(meeting_header.get("attendees", []))
    try:
        save_todo_list(db, meeting_id, attendees_email, account_id, meeting_summary.get("confirmedNextSteps", []))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving to-do list: {str(e)}"
        )
    try:
        save_crm_updates(
            db,
            meeting_id,
            account_id,
            '055dafe7-9840-451d-8328-5f70a6326c03',
            crm_updates_data or [],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving CRM updates: {str(e)}"
        )
    summary_record = SummaryDetails(
        meeting_id=meeting_id,
        meeting_title=meeting_header.get("title"),
        time_since_meeting=time_since_meeting,
        meeting_end_time=meeting_end_time,
        account_name=account_name,
        opportunity_id=opportunity_id,
        meeting_time_duration=meeting_header.get("duration"),
        meeting_platform=meeting_header.get("platform"),
        stagename = deal_stage,
        accountid =account_id,
        estimatedvalue = deal_value,
        customer_sentiment=meeting_summary.get(
            "sentiment", {}
        ).get("customerSentiment"),
        attendees=meeting_header.get("attendees"),
        crm_updates_pending_approval=crm_updates,
        key_points_count=
            meeting_summary.get("keyPoints", [])
        ,
        next_steps_count=
            meeting_summary.get("confirmedNextSteps", [])
        ,
        summary=meeting_summary.get(
            "summaryQualityNotes", []
        ),
        transcript=transcript
    )
    try:
       db.add(summary_record)
       db.commit()
       db.refresh(summary_record)

    except IntegrityError:
       db.rollback()
       raise HTTPException(
        status_code=409,
        detail=f"Meeting id already exist for meeting_id "
    )

    response_payload = raw_payload.copy()
    if "meeting_id" not in response_payload:
        response_payload["meeting_id"] = str(meeting_id)

    return response_payload



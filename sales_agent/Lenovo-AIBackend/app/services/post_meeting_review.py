from click import UUID
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models.summaryDetails import SummaryDetails
from app.models.post_meet import CrmUpdate
from collections import defaultdict

from datetime import date, timedelta
from uuid import UUID
from sqlalchemy import func, case, text
from app.models.to_do_list import TblToDoList

def get_post_meeting_preview(db: Session,seller_id: str):

    seller_uuid = UUID(seller_id)

    meetings = (
        db.query(SummaryDetails)
        .filter(
            SummaryDetails.seller_id == seller_uuid,
            func.lower(func.coalesce(SummaryDetails.status, "")) != "archived",
        )
        .all()
    )
#      meetings = (
#     db.query(SummaryDetails)
#     .filter(
#         or_(
#             and_(
#                 SummaryDetails.status == "review",
#                 SummaryDetails.time_since_meeting >= now - timedelta(days=7)
#             ),
#             and_(
#                 SummaryDetails.status == "completed",
#                 SummaryDetails.updated_at >= now - timedelta(hours=24)
#             ),
#             SummaryDetails.status.notin_(
#                 ["review", "completed", "archived"]
#             )
#         )
#     )
#     .all()
# )

    response = []

    for meeting in meetings:
        print(f"Processing meeting: {meeting.meeting_id}")
        crm_updates_count = (
            db.query(func.count(CrmUpdate.id))
            .filter(
                CrmUpdate.meeting_id == meeting.meeting_id,
                CrmUpdate.seller_id == seller_uuid,
                func.lower(func.coalesce(CrmUpdate.status, "")) != "skip",
            )
            .scalar()
        )

        opportunity_id = None
        if meeting.accountid:
            opportunity = db.execute(
                text("""
                    SELECT opportunityid
                    FROM opportunity
                    WHERE LOWER(accountid) = LOWER(:account_id)
                    LIMIT 1
                """),
                {"account_id": str(meeting.accountid)},
            ).fetchone()
            if opportunity:
                opportunity_id = opportunity.opportunityid

        response.append({
            "meeting_id": str(meeting.meeting_id),
            "title": meeting.meeting_title,
            "meeting_platform": meeting.meeting_platform,
            "meeting_type": meeting.meeting_type,
            "meeting_start_time": meeting.time_since_meeting,
            "customer_sentiment": meeting.customer_sentiment,
            "summary": meeting.summary,
            "attendees": meeting.attendees,
            "attendees_count": len(meeting.attendees or []),
            "opportunity_id": str(opportunity_id) if opportunity_id else None,
            "crm_updates_count": crm_updates_count or 0,
            "key_points_count": len(meeting.key_points_count or []),
            "next_steps_count": len(meeting.next_steps_count or []),
            "status": meeting.status,
        })

    return {
        "weekly_status_counts": get_weekly_status_counts(db, seller_id),
        "meetings": response,
    }



def get_weekly_status_counts(
    db,
    seller_id: str
):
    seller_uuid = UUID(seller_id)

    today = date.today()

    # Monday 00:00:00
    week_start_date = today - timedelta(days=today.weekday())
    week_start = datetime.combine(week_start_date, datetime.min.time())

    # Next Monday 00:00:00
    week_end = week_start + timedelta(days=7)

    result = (
        db.query(
            func.sum(
                case(
                    (
                        func.lower(
                            func.coalesce(SummaryDetails.status, "")
                        ) == "review",
                        1,
                    ),
                    else_=0,
                )
            ).label("review_count"),
            func.sum(
                case(
                    (
                        func.lower(
                            func.coalesce(SummaryDetails.status, "")
                        ) == "approved",
                        1,
                    ),
                    else_=0,
                )
            ).label("approved_count"),
        )
        .filter(
            SummaryDetails.seller_id == seller_uuid,
            SummaryDetails.time_since_meeting >= week_start,
            SummaryDetails.time_since_meeting < week_end,
        )
        .first()
    )

    crm_count = (
        db.query(func.count(CrmUpdate.id))
        .join(
            SummaryDetails,
            SummaryDetails.meeting_id == CrmUpdate.meeting_id
        )
        .filter(
            SummaryDetails.seller_id == seller_uuid,
            SummaryDetails.time_since_meeting >= week_start,
            SummaryDetails.time_since_meeting < week_end,
            func.lower(func.coalesce(SummaryDetails.status, "")).notin_(
                ["archived", "approved"]
            ),
            CrmUpdate.seller_id == seller_uuid,
            func.lower(func.coalesce(CrmUpdate.status, "")) != "skip"
        )
        .scalar()
    )

    return {
        "week_start": week_start.strftime("%Y-%m-%d %H:%M:%S"),
        "week_end": (week_end - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "review_count": result.review_count or 0,
        "approved_count": result.approved_count or 0,
        "crm_count": crm_count or 0
    }

def get_next_steps(
    db: Session,
    seller_id: str,
    meeting_id: str
):
    seller_uuid = UUID(seller_id)
    todos = (
        db.query(TblToDoList)
        .filter(
            TblToDoList.seller_id == seller_uuid,
            TblToDoList.meeting_id == meeting_id,
            TblToDoList.status != "archived"
        )
        .order_by(TblToDoList.created_at.desc())
        .all()
    )

    return [
        {
            "id": todo.id,
            "task_title": todo.task_title,
            "priority": todo.priority,
            "source_label": todo.source_label,
            "status": todo.status,
            "due_date": todo.due_date,
           
        }
        for todo in todos
    ]

def get_postmeeting_review_by_meeting_id(
    db: Session,
    meeting_id: str,
    seller_id: str
):
    print(f"Fetching post-meeting review for meeting_id: {meeting_id}")
    meeting = (
        db.query(SummaryDetails)
        .filter(
            SummaryDetails.meeting_id == meeting_id
        )
        .first()
    )
    print(f"Meeting fetched: {meeting}")
    if not meeting:
        raise HTTPException(
            status_code=404,
            detail="Meeting not found"
        )
    print(f"account_id: {meeting.accountid}")

    return {
        "meeting_id": meeting.meeting_id,
        "title": meeting.meeting_title,
        "meeting_platform": meeting.meeting_platform,
        "meeting_type": meeting.meeting_type,
        "meeting_start_time": meeting.time_since_meeting,
        "meeting_duration": meeting.meeting_time_duration,
        "call_transcript": meeting.transcript,
        "crm_updates": get_crm_updates(db,meeting.meeting_id,meeting.accountid) ,
        "key_points":meeting.key_points_count,
        "next_steps": get_next_steps(db, str(meeting.seller_id), meeting.meeting_id),
        "summary": meeting.summary,
    }

def add_keypoint_id(key_points, new_keypoint):
    # Generate next ID only if not present
    if not new_keypoint.get("id"):
        existing_ids = []

        for kp in key_points:
            if kp.get("id", "").startswith("KP"):
                try:
                    existing_ids.append(
                        int(kp["id"].replace("KP", ""))
                    )
                except ValueError:
                    pass

        next_id = max(existing_ids, default=0) + 1
        new_keypoint["id"] = f"KP{next_id:03d}"

    key_points.append(new_keypoint)

    return key_points


def add_key_point(
    db: Session,
    meeting_id: str,
    payload: dict
):
    meeting = (
        db.query(SummaryDetails)
        .filter(
            SummaryDetails.meeting_id == meeting_id
        )
        .first()
    )

    if not meeting:
        raise HTTPException(
            status_code=404,
            detail="Meeting not found"
        )

    key_points = meeting.key_points_count or []
    
    new_points = add_keypoint_id(key_points, payload)
    print(f"New key point added: {new_points}")
    # key_points.append(new_points)

   

    meeting.key_points_count = new_points
    db.flush()
   

    db.commit()
    db.refresh(meeting)

    return {
        "message": "Key point added successfully",
        "key_points": meeting.key_points_count
    }


def delete_key_point(
    db: Session,
    meeting_id: str,
    key_point_id: str
):
    meeting = (
        db.query(SummaryDetails)
        .filter(
            SummaryDetails.meeting_id == meeting_id
        )
        .first()
    )

    if not meeting:
        raise HTTPException(
            status_code=404,
            detail="Meeting not found"
        )

    key_points = meeting.key_points_count or []

    updated_key_points = [
        kp
        for kp in key_points
        if kp.get("id") != key_point_id
    ]

    if len(updated_key_points) == len(key_points):
        raise HTTPException(
            status_code=404,
            detail="Key point not found"
        )

    meeting.key_points_count = updated_key_points

    db.commit()
    db.refresh(meeting)

    return {
        "message": "Key point deleted successfully",
        "key_points": meeting.key_points_count
    }


def update_key_point(
    db: Session,
    meeting_id: str,
    key_point_id: str,
    payload: dict
):
    meeting = (
        db.query(SummaryDetails)
        .filter(
            SummaryDetails.meeting_id == meeting_id
        )
        .first()
    )

    if not meeting:
        raise HTTPException(
            status_code=404,
            detail="Meeting not found"
        )

    key_points = meeting.key_points_count or []

    updated = False

    for index, kp in enumerate(key_points):
        if kp.get("id") == key_point_id:

            key_points[index] = {
                "id": key_point_id,
                "point": payload["point"],
                "confidence": payload["confidence"],
                "isAmbiguous": payload["isAmbiguous"]
            }

            updated = True
            break

    if not updated:
        raise HTTPException(
            status_code=404,
            detail="Key point not found"
        )

    meeting.key_points_count = key_points

    db.commit()
    db.refresh(meeting)

    return {
        "message": "Key point updated successfully",
        "key_points": meeting.key_points_count
    }

def delete_nextSteps(db: Session, meeting_id: str, nextSteps_id: int):
    todo = (
        db.query(TblToDoList)
        .filter(
            TblToDoList.id == nextSteps_id,
            TblToDoList.meeting_id == meeting_id
        )
        .first()
    )

    if not todo:
        return {
            "success": False,
            "message": "To-do item not found."
        }

    todo.status = "archived"
    todo.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(todo)

    return {
        "success": True,
        "message": "To-do item Deleted successfully.",
        
    }
# def delete_nextSteps(
#     db: Session,
#     meeting_id: str,
#     nextSteps_id: str
# ):
#     meeting = (
#         db.query(SummaryDetails)
#         .filter(
#             SummaryDetails.meeting_id == meeting_id
#         )
#         .first()
#     )

#     if not meeting:
#         raise HTTPException(
#             status_code=404,
#             detail="Meeting not found"
#         )

#     nextSteps = meeting.next_steps_count or []

#     updated_nextSteps = [
#         kp
#         for kp in nextSteps
#         if kp.get("id") != nextSteps_id
#     ]

#     if len(updated_nextSteps) == len(nextSteps):
#         raise HTTPException(
#             status_code=404,
#             detail="Key point not found"
#         )

#     meeting.next_steps_count = updated_nextSteps

#     db.commit()
#     db.refresh(meeting)

#     return {
#         "message": "NextStep deleted successfully",
#         "nextSteps": meeting.next_steps_count
#     }
def update_nextSteps(
    db: Session,
   meeting_id: str,
    nextSteps_id: str,
    payload: dict
):
    todo = (
        db.query(TblToDoList)
        .filter(
            TblToDoList.id == nextSteps_id,
            TblToDoList.meeting_id == meeting_id
        )
        .first()
    )

    if not todo:
        return {
            "success": False,
            "message": "To-do item not found."
        }

    todo.task_title = payload.task
    todo.status = payload.status
    todo.priority = payload.confidence
    todo.owner = payload.owner

    if payload.dueDate:
        todo.due_date = datetime.strptime(
            payload.dueDate,
            "%Y-%m-%d"
        ).date()
    else:
        todo.due_date = None

    todo.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(todo)

    return {
        "success": True,
        "message": "To-do item updated successfully."
    }
# def update_nextSteps(
#     db: Session,
#     meeting_id: str,
#     nextSteps_id: str,
#     payload: dict
# ):
#     meeting = (
#         db.query(SummaryDetails)
#         .filter(
#             SummaryDetails.meeting_id == meeting_id
#         )
#         .first()
#     )

#     if not meeting:
#         raise HTTPException(
#             status_code=404,
#             detail="Meeting not found"
#         )

#     nextSteps = meeting.next_steps_count or []

#     updated = False

#     for index, kp in enumerate(nextSteps):
#         if kp.get("id") == nextSteps_id:

#             nextSteps[index] = {
#                 "id": nextSteps_id,
#                 "task": payload["task"],
#                 "owner": payload["owner"],
#                 "dueDate": payload["dueDate"],
#                 "status": payload["status"],
#                 "confidence": payload["confidence"],
#                 "transcriptRefs": payload["transcriptRefs"]
#             }

#             updated = True
#             break

#     if not updated:
#         raise HTTPException(
#             status_code=404,
#             detail="next step not found"
#         )

#     meeting.next_steps_count = nextSteps

#     db.commit()
#     db.refresh(meeting)

#     return {
#         "message": "next steps updated successfully",
#         "key_points": meeting.next_steps_count
#     }


def get_crm_updates(
    db: Session,
    meeting_id: str,
    account_id: str
):
    records = (
        db.query(CrmUpdate)
        .filter(
            CrmUpdate.meeting_id == meeting_id,
            # CrmUpdate.account_id == account_id,
            CrmUpdate.status.notin_(["skip"])
        )
        .all()
    )
    print("record",records)

    grouped_updates = defaultdict(list)

    for row in records:
        grouped_updates[row.entity].append(
            {   "id":row.seq_id,
                "Field": row.field_name,
                "current": row.current_value,
                "reasoning": row.reasoning,
                "suggested": row.suggested_value,
                "confidence":row.confidence
            }
        )

    return {
        "updates": dict(grouped_updates)
    }



def update_crm_update(
    db,
    meeting_id,
    seq_id,
    payload
):
    record = (
        db.query(CrmUpdate)
        .filter(
            CrmUpdate.meeting_id == meeting_id,
            CrmUpdate.seq_id == seq_id
        )
        .first()
    )

    if not record:
        raise HTTPException(
            status_code=404,
            detail="CRM update not found"
        )

    # Validate status
    allowed_statuses = ["skip", "update"]

    if payload.status.lower() not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail="Status must be either 'skip' or 'update'"
        )

    # Update CRM record
    record.current_value = payload.current_value
    record.suggested_value = payload.suggested_value
    record.reasoning = payload.reasoning
    record.confidence = payload.confidence
    record.status = payload.status.lower()

    db.commit()

    # Check all CRM records for this meeting
    crm_records = (
        db.query(CrmUpdate)
        .filter(CrmUpdate.meeting_id == meeting_id)
        .all()
    )

    all_completed = all(
        crm.status is not None and crm.status.lower() in ["skip", "update"]
        for crm in crm_records
    )

    if all_completed:
        print("yes")
        summary_record = (
            db.query(SummaryDetails)
            .filter(SummaryDetails.meeting_id == meeting_id)
            .first()
        )

        if summary_record:
            summary_record.status = "approved"
            db.commit()

    db.refresh(record)

    return {
        "message": "CRM update updated successfully",
        "meeting_id": str(meeting_id),
        "seq_id": seq_id,
        "status": record.status
    }




def complete_keypoints(db, meeting_id: str):
    
    meetings = (
        db.query(SummaryDetails)
        .filter(SummaryDetails.meeting_id == meeting_id)
        .all()
    )

    if not meetings:
        raise HTTPException(
            status_code=404,
            detail="Meeting not found"
        )

    updated_count = (
        db.query(SummaryDetails)
        .filter(SummaryDetails.meeting_id == meeting_id)
        .update(
            {
                SummaryDetails.keypoint_status: "completed",
                SummaryDetails.updated_at: datetime.utcnow()
            },
            synchronize_session=False
        )
    )

    db.commit()

    return {
        "message": "Keypoints marked as completed",
        "meeting_id": meeting_id,
        "updated_records": updated_count
    }

from datetime import datetime
from fastapi import HTTPException

def complete_nextstep(db, meeting_id: str):

    updated_count = (
        db.query(SummaryDetails)
        .filter(SummaryDetails.meeting_id == meeting_id)
        .update(
            {
                SummaryDetails.nextstep_status: "completed",
                SummaryDetails.updated_at: datetime.utcnow()
            },
            synchronize_session=False
        )
    )

    if updated_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Meeting not found"
        )

    todo_updated_count = (
        db.query(TblToDoList)
        .filter(
            TblToDoList.meeting_id == meeting_id,
            TblToDoList.status != "archived"
        )
        .update(
            {
                TblToDoList.status: "completed",
                TblToDoList.updated_at: datetime.utcnow()
            },
            synchronize_session=False
        )
    )

    db.commit()

    return {
        "message": "Next steps marked as completed",
        "meeting_id": meeting_id,
        "summary_records_updated": updated_count,
        "todo_records_updated": todo_updated_count
    }

def update_summary(
    db,
    meeting_id,
    payload
):
    summary_record = (
        db.query(SummaryDetails)
        .filter(
            SummaryDetails.meeting_id == meeting_id
        )
        .first()
    )

    if not summary_record:
        raise HTTPException(
            status_code=404,
            detail="Meeting summary not found"
        )
    allowed_statuses = ["skip", "update"]

    if payload.status.lower() not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail="Status must be either 'skip' or 'update'"
        )

    summary_record.summary = payload.summary
    summary_record.summary_status = payload.status.lower()
    summary_record.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(summary_record)

    return {
        "message": "Summary updated successfully"
        
    }


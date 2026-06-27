from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import String, case, cast, func, select, update
from sqlalchemy.orm import Session

from app.models.crm import CrmAccount, CrmOpportunity
from app.models.email import TblEmail
from app.models.email_template import EmailTemplate
from app.models.to_do_list import TblToDoList
from app.schema.to_do_list import UpdateTodoRequest, UpdateTodoStatusRequest
from app.services.compliance_audit import write_compliance_audit

OPEN_STATUS_EXCLUSIONS = ("completed", "archived")


@dataclass(frozen=True)
class TodoPendingSummaryData:
    seller_id: str
    count: int
    overdue_count: int
    due_today_count: int
    has_overdue: bool
    badge_color: Literal["red", "default"]
    label: str
    last_updated_at: datetime
    source: Literal["ai"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def build_pending_label(count: int) -> str:
    noun = "task" if count == 1 else "tasks"
    return f"{count} {noun} pending"


def badge_color_for(has_overdue: bool) -> Literal["red", "default"]:
    return "red" if has_overdue else "default"


def _open_status_condition():
    return TblToDoList.status.notin_(OPEN_STATUS_EXCLUSIONS)


def _parse_seller_uuid(seller_id: str) -> UUID:
    return UUID(seller_id.strip())


def _seller_condition(seller_id: str):
    return TblToDoList.seller_id == _parse_seller_uuid(seller_id)


def _due_sort_bucket(today: date):
    return case(
        (TblToDoList.due_date < today, 0),
        (TblToDoList.due_date == today, 1),
        (TblToDoList.due_date > today, 2),
        else_=3,
    )


def _upper_id(column):
    """Case-insensitive UUID/TEXT id match — Postgres has no upper(uuid)."""
    return func.upper(cast(column, String))


def _load_crm_context_maps(
    db: Session,
    tasks: list[TblToDoList],
) -> tuple[dict[str, str], dict[str, str], dict[str, float | None], dict[str, str]]:
    account_ids = {
        str(task.linked_account_id).upper()
        for task in tasks
        if task.linked_account_id
    }
    opportunity_ids = {
        str(task.linked_opportunity_id).upper()
        for task in tasks
        if task.linked_opportunity_id
    }

    account_names: dict[str, str] = {}
    opportunity_names: dict[str, str] = {}
    deal_values: dict[str, float | None] = {}
    stage_names: dict[str, str] = {}

    if account_ids:
        rows = db.execute(
            select(CrmAccount.accountid, CrmAccount.name).where(
                _upper_id(CrmAccount.accountid).in_(account_ids)
            )
        ).all()
        account_names = {
            str(account_id).upper(): name
            for account_id, name in rows
            if account_id and name
        }

    if opportunity_ids:
        rows = db.execute(
            select(
                CrmOpportunity.opportunityid,
                CrmOpportunity.name,
                CrmOpportunity.estimatedvalue,
                CrmOpportunity.stagename,
            ).where(_upper_id(CrmOpportunity.opportunityid).in_(opportunity_ids))
        ).all()
        for opportunity_id, name, estimated_value, stage_name in rows:
            if not opportunity_id:
                continue
            key = str(opportunity_id).upper()
            if name:
                opportunity_names[key] = name
            if stage_name:
                stage_names[key] = stage_name
            deal_values[key] = (
                float(estimated_value) if estimated_value is not None else None
            )

    return account_names, opportunity_names, deal_values, stage_names


def _load_email_opportunity_maps(
    db: Session,
    emails: list[TblEmail],
) -> tuple[dict[str, float | None], dict[str, str], dict[str, str], dict[str, str]]:
    opportunity_ids = {
        str(email.opportunity_id).upper()
        for email in emails
        if email.opportunity_id
    }
    account_ids = {
        str(email.account_id).upper()
        for email in emails
        if email.account_id
    }

    deal_values: dict[str, float | None] = {}
    stage_names: dict[str, str] = {}
    opportunity_names: dict[str, str] = {}
    account_names: dict[str, str] = {}

    if opportunity_ids:
        rows = db.execute(
            select(
                CrmOpportunity.opportunityid,
                CrmOpportunity.name,
                CrmOpportunity.estimatedvalue,
                CrmOpportunity.stagename,
            ).where(_upper_id(CrmOpportunity.opportunityid).in_(opportunity_ids))
        ).all()
        for opportunity_id, name, estimated_value, stage_name in rows:
            if not opportunity_id:
                continue
            key = str(opportunity_id).upper()
            if name:
                opportunity_names[key] = name
            if stage_name:
                stage_names[key] = stage_name
            deal_values[key] = (
                float(estimated_value) if estimated_value is not None else None
            )

    if account_ids:
        rows = db.execute(
            select(CrmAccount.accountid, CrmAccount.name).where(
                _upper_id(CrmAccount.accountid).in_(account_ids)
            )
        ).all()
        account_names = {
            str(account_id).upper(): name
            for account_id, name in rows
            if account_id and name
        }

    return deal_values, stage_names, opportunity_names, account_names


def _serialize_todo(
    todo: TblToDoList,
    account_names: dict[str, str],
    opportunity_names: dict[str, str],
    deal_values: dict[str, float | None],
    stage_names: dict[str, str],
) -> dict:
    account_key = (
        str(todo.linked_account_id).upper() if todo.linked_account_id else None
    )
    opportunity_key = (
        str(todo.linked_opportunity_id).upper()
        if todo.linked_opportunity_id
        else None
    )

    return {
        "id": todo.id,
        "meeting_id": todo.meeting_id,
        "opportunity_id": todo.opportunity_id,
        "task_title": todo.task_title,
        "type_tag": todo.type_tag,
        "priority": todo.priority,
        "source_label": todo.source_label,
        "linked_account_id": todo.linked_account_id,
        "linked_opportunity_id": todo.linked_opportunity_id,
        "account_name": account_names.get(account_key) if account_key else None,
        "opportunity_name": (
            opportunity_names.get(opportunity_key) if opportunity_key else None
        ),
        "stage_name": stage_names.get(opportunity_key) if opportunity_key else None,
        "deal_value": deal_values.get(opportunity_key) if opportunity_key else None,
        "attendees_email": todo.attendees_email,
        "notes": todo.notes,
        "why_now": todo.notes,
        "status": todo.status,
        "due_date": todo.due_date,
        "seller_id": todo.seller_id,
        "created_at": todo.created_at,
        "updated_at": todo.updated_at,
    }


def _serialize_email(
    email: TblEmail,
    deal_values: dict[str, float | None],
    stage_names: dict[str, str],
    opportunity_names: dict[str, str],
    account_names: dict[str, str],
) -> dict:
    opportunity_key = (
        str(email.opportunity_id).upper() if email.opportunity_id else None
    )
    account_key = str(email.account_id).upper() if email.account_id else None
    return {
        "message_id": email.message_id,
        "from_add": email.from_add,
        "to_add": email.to_add,
        "subject": email.subject,
        "body": email.body,
        "sender_type": email.sender_type,
        "classification": email.classification,
        "intent_category": email.intent_category,
        "opportunity_name": (
            opportunity_names.get(opportunity_key)
            if opportunity_key
            else None
        ),
        "account_name": account_names.get(account_key) if account_key else None,
        "opportunity_id": email.opportunity_id,
        "account_id": email.account_id,
        "stage_name": stage_names.get(opportunity_key) if opportunity_key else None,
        "deal_value": deal_values.get(opportunity_key) if opportunity_key else None,
        "why_now": email.subject,
        "received_datetime": email.received_datetime,
        "send_datetime": email.send_datetime,
        "type_tag": email.type_tag,
        "priority": email.priority,
        "source_label": email.source_label,
        "status": email.status,
        "seller_id": email.seller_id,
        "created_at": email.created_at,
        "updated_at": email.updated_at,
    }


def build_todo_pending_summary(
    db: Session,
    seller_id: str,
    *,
    today: date | None = None,
) -> TodoPendingSummaryData:
    today = today or date.today()
    seller = seller_id.strip()
    base = (_open_status_condition(), _seller_condition(seller))

    count = int(
        db.scalar(select(func.count()).select_from(TblToDoList).where(*base)) or 0
    )
    overdue_count = int(
        db.scalar(
            select(func.count())
            .select_from(TblToDoList)
            .where(
                *base,
                TblToDoList.due_date.is_not(None),
                TblToDoList.due_date < today,
            )
        )
        or 0
    )
    due_today_count = int(
        db.scalar(
            select(func.count())
            .select_from(TblToDoList)
            .where(*base, TblToDoList.due_date == today)
        )
        or 0
    )
    has_overdue = overdue_count > 0

    return TodoPendingSummaryData(
        seller_id=seller,
        count=count,
        overdue_count=overdue_count,
        due_today_count=due_today_count,
        has_overdue=has_overdue,
        badge_color=badge_color_for(has_overdue),
        label=build_pending_label(count),
        last_updated_at=_utc_now(),
        source="ai",
    )


def get_todos(
    db: Session,
    filter_type: str = "all",
    seller_id: str | None = None,
    show_completed: bool = False,
):
    today = date.today()
    if show_completed:
        status_condition = TblToDoList.status == "completed"
        email_status_condition = TblEmail.status == "completed"
    else:
        status_condition = _open_status_condition()
        email_status_condition = TblEmail.status.notin_(OPEN_STATUS_EXCLUSIONS)
    scope = [status_condition]
    if seller_id and seller_id.strip():
        scope.append(_seller_condition(seller_id))

    email_scope = [email_status_condition]
    if seller_id and seller_id.strip():
        email_scope.append(TblEmail.seller_id == _parse_seller_uuid(seller_id))

    # -------------------------
    # Summary Counts (open tasks only)
    # -------------------------

    overdue_counts_query = select(
        func.count().label("all"),
        func.count().filter(TblToDoList.status == "Completed").label("completed"),
    ).where(
        *scope,
        TblToDoList.due_date.is_not(None),
        TblToDoList.due_date < today,
    )
    overdue_counts = db.execute(overdue_counts_query).one()

    today_counts_query = select(
        func.count().label("all"),
        func.count().filter(TblToDoList.status == "completed").label("completed"),
    ).where(
        *scope,
        TblToDoList.due_date == today,
    )
    today_counts = db.execute(today_counts_query).one()

    upcoming_counts_query = select(
        func.count().label("all"),
        func.count().filter(TblToDoList.status == "completed").label("completed"),
    ).where(
        *scope,
        TblToDoList.due_date > today,
    )
    upcoming_counts = db.execute(upcoming_counts_query).one()

    no_due_counts_query = select(
        func.count().label("all"),
        func.count().filter(TblToDoList.status == "completed").label("completed"),
    ).where(
        *scope,
        TblToDoList.due_date.is_(None),
    )
    no_due_counts = db.execute(no_due_counts_query).one()

    # -------------------------
    # Filter Counts (open tasks across all due buckets)
    # -------------------------

    filter_counts_query = select(
        func.count().label("all"),
        func.count().filter(TblToDoList.type_tag == "outreach").label("outreach"),
        func.count().filter(TblToDoList.type_tag == "document").label("document"),
        func.count().filter(TblToDoList.type_tag == "action").label("action"),
    ).where(*scope)

    filter_counts = db.execute(filter_counts_query).one()

    # -------------------------
    # Email counts — bucket by due_date (fall back to "no due date" when absent)
    # -------------------------

    email_counts_query = select(
        func.count().label("all"),
        func.count()
        .filter(TblEmail.due_date.is_not(None), TblEmail.due_date < today)
        .label("overdue"),
        func.count().filter(TblEmail.due_date == today).label("today"),
        func.count().filter(TblEmail.due_date > today).label("upcoming"),
        func.count().filter(TblEmail.due_date.is_(None)).label("no_due"),
        func.count().filter(TblEmail.type_tag == "outreach").label("outreach"),
        func.count().filter(TblEmail.type_tag == "document").label("document"),
        func.count().filter(TblEmail.type_tag == "action").label("action"),
    ).where(*email_scope)
    email_counts = db.execute(email_counts_query).one()

    # -------------------------
    # Task List — overdue first, then today, then future, then no due date
    # -------------------------

    task_query = select(TblToDoList).where(*scope)

    if filter_type.lower() != "all":
        task_query = task_query.where(
            func.lower(TblToDoList.type_tag) == filter_type.lower()
        )

    task_query = task_query.order_by(
        _due_sort_bucket(today),
        TblToDoList.due_date.asc(),
        TblToDoList.created_at.asc(),
    )

    tasks = db.execute(task_query).scalars().all()
    account_names, opportunity_names, deal_values, stage_names = _load_crm_context_maps(
        db, tasks
    )

    # -------------------------
    # Email List — same scope, optionally filtered by type_tag
    # -------------------------

    email_query = select(TblEmail).where(*email_scope)

    if filter_type.lower() != "all":
        email_query = email_query.where(
            func.lower(TblEmail.type_tag) == filter_type.lower()
        )

    email_query = email_query.order_by(TblEmail.created_at.desc())
    emails = db.execute(email_query).scalars().all()
    (
        email_deal_values,
        email_stage_names,
        email_opportunity_names,
        email_account_names,
    ) = _load_email_opportunity_maps(db, emails)

    return {
        "summary": {
            "overdue": overdue_counts.all + email_counts.overdue,
            "today": today_counts.all + email_counts.today,
            "upcoming": upcoming_counts.all + email_counts.upcoming,
            "no_due_date": no_due_counts.all + email_counts.no_due,
            "completed_today": today_counts.completed,
            "completed_upcoming": upcoming_counts.completed,
        },
        "filters": {
            "all": filter_counts.all + email_counts.all,
            "outreach": filter_counts.outreach + email_counts.outreach,
            "document": filter_counts.document + email_counts.document,
            "action": filter_counts.action + email_counts.action,
        },
        "tasks": [
            _serialize_todo(task, account_names, opportunity_names, deal_values, stage_names)
            for task in tasks
        ],
        "emails": [
            _serialize_email(
                email,
                email_deal_values,
                email_stage_names,
                email_opportunity_names,
                email_account_names,
            )
            for email in emails
        ],
    }


def _update_email_record(
    db,
    message_id: str,
    payload: UpdateTodoRequest,
):
    email = (
        db.query(TblEmail)
        .filter(TblEmail.message_id == message_id)
        .first()
    )

    if not email:
        raise HTTPException(
            status_code=404,
            detail="Email not found",
        )

    update_data = payload.model_dump(exclude_unset=True)

    # Emails can only update these three fields
    for field in ("type_tag", "priority", "due_date"):
        if field in update_data:
            setattr(email, field, update_data[field])

    email.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(email)

    (
        deal_values,
        stage_names,
        opportunity_names,
        account_names,
    ) = _load_email_opportunity_maps(db, [email])

    return {
        "message": "Email updated successfully",
        "email": _serialize_email(
            email, deal_values, stage_names, opportunity_names, account_names
        ),
    }


def update_todo(
    db,
    todo_id: str,
    payload: UpdateTodoRequest,
):
    if payload.source_label and payload.source_label.lower() == "email":
        return _update_email_record(db, str(todo_id), payload)

    todo = (
        db.query(TblToDoList)
        .filter(TblToDoList.id == int(todo_id))
        .first()
    )

    if not todo:
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )

    update_data = payload.model_dump(exclude_unset=True)
    update_data.pop("source_label", None)

    for field, value in update_data.items():
        setattr(todo, field, value)

    todo.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(todo)

    account_names, opportunity_names, deal_values, stage_names = _load_crm_context_maps(
        db, [todo]
    )

    return {
        "message": "Task updated successfully",
        "task": _serialize_todo(
            todo, account_names, opportunity_names, deal_values, stage_names
        ),
    }


def update_todo_status(
    db,
    todo_id: str,
    payload: UpdateTodoStatusRequest,
    filter_type: str = "all",
    seller_id: str | None = None,
):
    allowed_statuses = ["completed", "archived"]

    if payload.status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Status must be one of {allowed_statuses}",
        )

    if payload.source_label and payload.source_label.lower() == "email":
        email = (
            db.query(TblEmail)
            .filter(TblEmail.message_id == str(todo_id))
            .first()
        )
        if email is None:
            raise HTTPException(status_code=404, detail="Email not found")

        email.status = payload.status
        email.updated_at = datetime.utcnow()
        db.commit()

        return get_todos(
            db=db,
            filter_type=filter_type,
            seller_id=seller_id,
        )

    todo = db.get(TblToDoList, int(todo_id))
    if todo is None:
        raise HTTPException(status_code=404, detail="Task not found")

    before_status = todo.status
    result = db.execute(
        update(TblToDoList)
        .where(TblToDoList.id == int(todo_id))
        .values(
            status=payload.status,
            updated_at=datetime.utcnow(),
        )
    )

    if result.rowcount == 0:
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )

    write_compliance_audit(
        db,
        entity_type="todo",
        entity_id=str(todo_id),
        action="update",
        category="seller_action",
        actor_type="seller",
        changed_by=seller_id,
        opportunity_id=str(todo.linked_opportunity_id) if todo.linked_opportunity_id else None,
        diff={
            "before": {"status": before_status},
            "after": {"status": payload.status},
        },
    )
    db.commit()

    return get_todos(
        db=db,
        filter_type=filter_type,
        seller_id=seller_id,
    )


def create_todo(
    db,
    payload,
):
    if not payload.title:
        raise HTTPException(
            status_code=400,
            detail="Title is required",
        )

    if not payload.type_tag:
        raise HTTPException(
            status_code=400,
            detail="Type is required",
        )

    if not payload.priority:
        raise HTTPException(
            status_code=400,
            detail="Priority is required",
        )

    todo = TblToDoList(
        task_title=payload.title,
        type_tag=payload.type_tag,
        priority=payload.priority,
        linked_account_id=payload.linked_account_id,
        linked_opportunity_id=payload.linked_opportunity_id,
        due_date=payload.due_date,
        notes=payload.notes,
        seller_id=payload.seller_id,
        status="Open",
    )

    db.add(todo)
    db.commit()
    db.refresh(todo)

    return {
        "message": "To-do created successfully",
        "id": todo.id
    }

def get_todo_by_id(db, todo_id: int):
    todo = (
        db.query(TblToDoList)
        .filter(TblToDoList.id == todo_id)
        .first()
    )

    if not todo:
        raise HTTPException(
            status_code=404,
            detail=f"Todo item with id {todo_id} not found"
        )

    account_names, opportunity_names, deal_values, stage_names = _load_crm_context_maps(
        db, [todo]
    )
    serialized = _serialize_todo(
        todo, account_names, opportunity_names, deal_values, stage_names
    )

    return {
        "id": serialized["id"],
        "task_title": serialized["task_title"],
        "type_tag": serialized["type_tag"],
        "priority": serialized["priority"],
        "source_label": serialized["source_label"],
        "linked_account_id": serialized["linked_account_id"],
        "linked_opportunity_id": serialized["linked_opportunity_id"],
        "account_name": serialized["account_name"],
        "opportunity_name": serialized["opportunity_name"],
        "stage_name": serialized["stage_name"],
        "deal_value": serialized["deal_value"],
        "attendees_email": serialized["attendees_email"],
        "notes": serialized["notes"],
        "status": serialized["status"],
        "due_date": serialized["due_date"],
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

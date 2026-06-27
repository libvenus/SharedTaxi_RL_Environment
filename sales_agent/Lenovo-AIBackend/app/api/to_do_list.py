from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db

from app.schema.to_do_list import (
    CreateTodoRequest,
    TodoPendingSummaryResponse,
    UpdateTodoRequest,
    UpdateTodoStatusRequest,
)
from app.services.to_do_list import (
    build_todo_pending_summary,
    create_todo,
    get_todos,
    update_todo,
    update_todo_status,get_todo_by_id,
)

router = APIRouter(prefix="/todos", tags=["ToDo"])


def _require_seller_id(seller_id: str | None) -> str:
    if not seller_id or not seller_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="sellerId is required.",
        )
    return seller_id.strip()


@router.get(
    "/summary",
    response_model=TodoPendingSummaryResponse,
    summary="Task Pending badge — open Execute To-Do count for the Home header",
)
def fetch_todo_summary(
    seller_id: str | None = Query(
        default=None,
        alias="sellerId",
        description="Seller UUID — scopes tbl_to_do_list rows.",
    ),
    db: Session = Depends(get_db),
) -> TodoPendingSummaryResponse:
    seller = _require_seller_id(seller_id)
    data = build_todo_pending_summary(db, seller)
    return TodoPendingSummaryResponse(
        seller_id=data.seller_id,
        count=data.count,
        overdue_count=data.overdue_count,
        due_today_count=data.due_today_count,
        has_overdue=data.has_overdue,
        badge_color=data.badge_color,
        label=data.label,
        last_updated_at=data.last_updated_at,
        source=data.source,
    )


@router.get("/")
def fetch_todos(
    filter_type: str = Query(
        default="all",
        description="all, outreach, document, action",
    ),
    seller_id: str | None = Query(
        default=None,
        alias="sellerId",
        description="Seller UUID — required for seller-scoped Execute To-Do list.",
    ),
    show_completed: bool = Query(
        default=False,
        alias="showCompleted",
        description="When true, returns completed tasks instead of open tasks.",
    ),
    db: Session = Depends(get_db),
):
    return get_todos(
        db=db,
        filter_type=filter_type,
        seller_id=seller_id,
        show_completed=show_completed,
    )


@router.put("/{todo_id}")
def edit_todo(
    todo_id: str,
    payload: UpdateTodoRequest,
    db: Session = Depends(get_db),
):
    return update_todo(
        db=db,
        todo_id=todo_id,
        payload=payload,
    )


@router.patch("/{todo_id}/status")
def change_todo_status(
    todo_id: str,
    payload: UpdateTodoStatusRequest,
    filter_type: str = Query(default="all"),
    seller_id: str | None = Query(default=None, alias="sellerId"),
    db: Session = Depends(get_db),
):
    return update_todo_status(
        db=db,
        todo_id=todo_id,
        payload=payload,
        filter_type=filter_type,
        seller_id=seller_id,
    )


@router.post("/create")
def create_todo_endpoint(
    payload: CreateTodoRequest,
    db: Session = Depends(get_db),
):
    return create_todo(
        db=db,
        payload=payload
    )

@router.get("/to_do_outreach/{todo_id}")
def get_todo(
    todo_id: int,
    db: Session = Depends(get_db)
):
    return get_todo_by_id(db, todo_id)

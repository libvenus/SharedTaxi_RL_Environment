from datetime import date, datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from uuid import UUID

class TodoResponse(BaseModel):
    id: int
    task_title: str
    type_tag: Optional[str]
    priority: Optional[str]
    source_label: Optional[str]
    status: Optional[str]
    due_date: Optional[date]
    linked_account_id: Optional[UUID] = None
    linked_opportunity_id: Optional[UUID] = None
    account_name: Optional[str] = None
    opportunity_name: Optional[str] = None
    deal_value: Optional[float] = None

    class Config:
        from_attributes = True


class TodoSummaryResponse(BaseModel):
    summary: dict
    filters: dict
    tasks: List[TodoResponse]


class TodoPendingSummaryResponse(BaseModel):
    seller_id: str
    count: int
    overdue_count: int
    due_today_count: int
    has_overdue: bool
    badge_color: Literal["red", "default"]
    label: str
    last_updated_at: datetime
    source: Literal["ai"] = "ai"

class UpdateTodoRequest(BaseModel):
    task_title: Optional[str] = None
    type_tag: Optional[str] = None
    priority: Optional[str] = None
    notes: Optional[str] = None
    source_label: Optional[str] = None

    due_date: Optional[date] = None   



class UpdateTodoStatusRequest(BaseModel):
    status: str
    source_label: Optional[str] = None     

class CreateTodoRequest(BaseModel):
    title: str
    type_tag: str
    priority: str

    linked_account_id: Optional[str] = None
    linked_opportunity_id: Optional[str] = None
    due_date: Optional[date] = None
    notes: Optional[str] = None
    seller_id: Optional[UUID] = Field(
        default=None,
        description="Seller UUID — scopes the task to the authenticated seller.",
    )

class CreateTodoResponse(BaseModel):
    id: int
    title: str
    type_tag: str
    priority: str
    linked_account_id: Optional[UUID]
    linked_opportunity_id: Optional[UUID]
    due_date: Optional[date]
    notes: Optional[str]    
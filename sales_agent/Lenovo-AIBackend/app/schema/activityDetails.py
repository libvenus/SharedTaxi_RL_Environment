from datetime import datetime
from decimal import Decimal
from uuid import UUID
from typing import Literal
from pydantic import BaseModel


class Attendee(BaseModel):
    name: str
    role: str


class ActivityDetailsCreate(BaseModel):
    time_since_meeting: str | None = None

    meeting_time: datetime

    duration_minutes: int

    meeting_platform: str

    customer_sentiment: Literal[
        "Positive",
        "Neutral",
        "Negative",
        "Mixed"
    ]


    meeting_title: str

    account_name: str

    deal_stage: str

    deal_value: Decimal

    attendees: list[Attendee]

    crm_updates_pending_approval: int = 0

    key_points_count: int = 0

    next_steps_count: int = 0

    review_url: str | None = None


class ActivityDetailsResponse(
    ActivityDetailsCreate
):
    meeting_id: UUID

    class Config:
        from_attributes = True
from pydantic import BaseModel
from datetime import datetime, date
from uuid import UUID
from typing import Any, Literal, Optional, List
from pydantic import BaseModel,field_validator

from sqlalchemy import TEXT

from typing import Dict, Any
BotStatus = Literal[
    "pending",
    "scheduled",
    "joining",
    "joined",
    "lobby_waiting",
    "cancelled",
    "rescheduled",
    "failed",
]
class MeetingDetailsCreate(BaseModel):
    payload: Dict[str, Any]

# class MeetingDetailsCreate(BaseModel):
#     meeting_id: str
#     meeting_start_time: datetime
#     meeting_end_time: datetime

#     platform: Optional[str] = None
#     title: Optional[str] = None
#     account_name: Optional[str] = None

#     attendees: Optional[str] = None

#     organiser_name: Optional[str] = None

#     action: Optional[str] = None
#     body: Optional[str] = None

#     recurrence_pattern: Optional[str] = None
#     recurrence_interval: Optional[int] = None

#     recurrence_start_date: Optional[date] = None
#     recurrence_end_date: Optional[date] = None

#     opportunity_id: Optional[UUID] = None
#     account_id: Optional[UUID] = None
#     opportunity: Optional[str] = None
#     meeting_url: Optional[str] = None
#     prep_notes: Optional[str] = None


class MeetingDetailsResponse(BaseModel):
    meeting_id: str

    meeting_start_time: datetime
    meeting_end_time: datetime

    platform: Optional[str]
    title: Optional[str]
    account_name: Optional[str]

    attendees: Optional[str]

    organiser_name: Optional[str]

    action: Optional[str]
    body: Optional[str]

    recurrence_pattern: Optional[str]
    recurrence_interval: Optional[int]

    recurrence_start_date: Optional[date]
    recurrence_end_date: Optional[date]

    class Config:
        from_attributes = True    


class Attendee(BaseModel):
    name: str
    role: Optional[str] = None
    email: Optional[str] = None


class MeetingSearchRequest(BaseModel):
    organiser_name: str
    meeting_start_time: datetime
    attendees_emails: str
    title: Optional[str] = None
    @field_validator("attendees_emails")
    @classmethod
    def validate_attendees(cls, value):
        if value:
            emails = []

            for item in value.replace(";", ",").split(","):
                email = item.strip()
                if email:
                    emails.append(email)

            for email in emails:
                if "@" not in email:
                    raise ValueError(f"Invalid email: {email}")

        return value


class MeetingStatusUpdate(BaseModel):
    bot_status: BotStatus
    reason: Optional[str] = None


class MeetingStatusResponse(BaseModel):
    meeting_id: UUID
    bot_status: BotStatus
    bot_status_reason: Optional[str] = None
    bot_last_event_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
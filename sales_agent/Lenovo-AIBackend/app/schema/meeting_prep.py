from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, HttpUrl, field_validator

class Attendee(BaseModel):
    name: str
    role: Optional[str] = None
    email: Optional[str] = None

class MeetingDetailsCreate(BaseModel):
    attendees: list[Attendee]

class CreateMeetingRequest(BaseModel):
    meeting_start_time: datetime
    meeting_end_time: Optional[datetime] = None
    platform: str
    title: str
    account_name: str
    attendees_emails: Optional[str] = None
    opportunity: Optional[str] = None
    meeting_url: Optional[str] = None
    body: Optional[str] = None
    prep_notes: Optional[str] = None
    seller_id: Optional[UUID] = None
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
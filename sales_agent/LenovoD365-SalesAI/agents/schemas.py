from typing import Literal

from pydantic import BaseModel


class RootPayload(BaseModel):
    target: Literal["sales", "orchestrator"] = "sales"
    message: str = ""


class SalesPayload(BaseModel):
    target: Literal["sales", "scheduler", "email"] = "sales"
    reply: str = ""
    message: str = ""


class SchedulerApiPayload(BaseModel):
    meeting_id: str = ""
    meeting_start_time: str = ""
    meeting_end_time: str = ""
    meeting_start_time_utc: str = ""
    meeting_end_time_utc: str = ""
    platform: Literal["Teams"] = "Teams"
    title: str = ""
    account_name: str = ""
    attendees: str = ""
    organiser_name: str = ""
    action: Literal["Schedule", "Reschedule", "Cancel"] = "Schedule"
    body: str = ""
    recurrence_pattern: str = ""
    recurrence_interval: int = 0
    recurrence_start_date: str = ""
    recurrence_end_date: str = ""


class SchedulerOutput(BaseModel):
    agent_name: str = "SchedulerAgent"
    action: Literal["schedule", "reschedule", "cancel", "none"] = "none"
    status: Literal["success", "error", "pending"] = "pending"
    ai_reply: str = ""
    api_payload: SchedulerApiPayload | None = None
    # Set only when the request is out of scope and must go back to sales.
    handoff: Literal["sales"] | None = None
    message: str = ""

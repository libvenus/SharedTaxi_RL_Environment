from enum import Enum
from typing import Optional

from uuid import UUID

from pydantic import BaseModel

class OutreachCategory(str, Enum):
    MEETING_FOLLOW_UP = "MEETING_FOLLOW_UP"
    ACCOUNT = "ACCOUNT"
    SILENT_AT_RISK = "SILENT_AT_RISK"
    HIGH_PRIORITY = "HIGH_PRIORITY"

class CreateEmailDraftRequest(BaseModel):
    id: int | None = None
    template_name: str
    context_used: str
    additional_context: Optional[str] = None

class DraftEmailRequest(BaseModel):
    context: str
    written_context: str
    template: str
    data: dict  
    placeholders: Optional[dict] = None

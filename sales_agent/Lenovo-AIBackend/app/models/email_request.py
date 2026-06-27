from typing import List, Optional

from pydantic import BaseModel, Field


class EmailRequest(BaseModel):
    to: List[str] = Field(..., min_items=1)

    cc: Optional[List[str]] = []

    subject: str

    body: str
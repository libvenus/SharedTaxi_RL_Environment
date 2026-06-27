from pydantic import BaseModel
from typing import Dict, Any
from uuid import UUID
from typing import Optional, List
from datetime import datetime

class SummaryDetailsRequest(BaseModel):
    payload: Dict[str, Any]


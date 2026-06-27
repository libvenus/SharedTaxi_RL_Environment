from pydantic import BaseModel
from typing import List
from typing import Optional

class KeyPointRequest(BaseModel):
    point: str
    confidence: Optional[str] = None 
    isAmbiguous: Optional[bool] = None 

class UpdateKeyPointRequest(BaseModel):
    point: str
    confidence: Optional[str] = None
    isAmbiguous: Optional[bool] = None    

class UpdateNextStepRequest(BaseModel):
    task: str
    owner: str
    dueDate: str
    status: str
    confidence: str 
   

class CrmUpdateEditRequest(BaseModel):
    current_value: str
    suggested_value: str    

class UpdateCrmRequest(BaseModel):
    field : Optional[str] = None
    current_value: Optional[str] = None
    suggested_value: Optional[str] = None
    status: str   

class UpdateSummaryRequest(BaseModel):
    summary: str 
    status: str     


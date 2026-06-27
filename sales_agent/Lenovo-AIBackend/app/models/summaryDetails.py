import uuid

from sqlalchemy import (
    Column,
    String,
    Integer,
    Numeric,
    DateTime,
    Text
)

from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from typing import Literal
from app.db.database import Base
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy import TEXT

class SummaryDetails(Base):
    __tablename__ = "tbl_summary_details"

    meeting_id = Column(
    TEXT,
    primary_key=True
)
    accountid =Column(
        UUID(as_uuid=True)
       
    )
    opportunity_id =Column(
        UUID(as_uuid=True)
       
    )
    seller_id = Column(
        UUID(as_uuid=True)
    )
    time_since_meeting = Column(DateTime)

    meeting_end_time = Column(DateTime)

    meeting_time_duration = Column(String(100))

    meeting_platform = Column(String(100))
    status = Column(String(100))
    keypoint_status = Column(String(100))
    nextstep_status = Column(String(100))
    crmupdate_status = Column(String(100))
    summary_status = Column(String(100))

    customer_sentiment = Column(String(20))

    meeting_title = Column(String(500))

    account_name = Column(String(255))

    stagename = Column(String(100))

    estimatedvalue = Column(Numeric(15, 2))

    attendees = Column(JSONB)

    crm_updates_pending_approval = Column(JSONB)


    key_points_count = Column(
    MutableList.as_mutable(JSONB)
)

    next_steps_count = Column(
    MutableList.as_mutable(JSONB)
)

    cta_to_review = Column(
    MutableList.as_mutable(JSONB)
)

    created_at = Column(
        DateTime,
        server_default=func.now()
    )

    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now()
    )

    meeting_type =Column(String(100))

    summary = Column(String(100))

    transcript = Column(JSONB)

from sqlalchemy import (
    Column,
    String,
    Text,
    Date,
    DateTime,
    Numeric
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from app.db.database import Base


class TblOutreach(Base):
    __tablename__ = "tbl_outreach"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Category
    outreach_type = Column(
        String(100),
        nullable=False
    )  # MEETING FOLLOW-UP, ACCOUNT, SILENT/AT-RISK

    # Priority
    priority_badge = Column(
        String(50),
        nullable=True
    )  # HIGH PRIORITY, MED PRIORITY

    # Account Details
    account_name = Column(
        String(255),
        nullable=True
    )

    company_name = Column(
        String(255),
        nullable=True
    )

    title = Column(Text, nullable=True)
    engagement_type =Column(Text, nullable=True)

    # Deal Details
    deal_name = Column(
        String(500),
        nullable=True
    )

    deal_stage = Column(
        String(100),
        nullable=True
    )  # Negotiation, Proposal, Discovery

    deal_value = Column(
        Numeric(18, 2),
        nullable=True
    )

    # Decision Maker
    decision_maker_name = Column(
        String(255),
        nullable=True
    )

    decision_maker_role = Column(
        String(255),
        nullable=True
    )

    # AI Generated Reason
    why_now_reason = Column(
        Text,
        nullable=True
    )
    attendees_email = Column(
        Text,
        nullable=True
    )
    category = Column(
        String(100),
        nullable=True
    )

    # Outreach Date
    outreach_date = Column(
        Date,
        nullable=True
    )

    # Last Activity
    last_activity_type = Column(
        String(100),
        nullable=True
    )  # Meeting, Email, Call

    last_activity_datetime = Column(
        DateTime(timezone=True),
        nullable=True
    )

    last_activity_summary = Column(
        Text,
        nullable=True
    )

    # Audit Columns
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    account_id = Column(UUID(as_uuid=True), nullable=True)
    opportunity_id = Column(UUID(as_uuid=True), nullable=True)
    contact_id = Column(UUID(as_uuid=True), nullable=True)
    meeting_id = Column(UUID(as_uuid=True), nullable=True)
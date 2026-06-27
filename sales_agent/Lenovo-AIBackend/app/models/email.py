from sqlalchemy import Column, String, DateTime, Date
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.database import Base


class TblEmail(Base):
    __tablename__ = "tbl_emails"

    message_id = Column(String(500), primary_key=True)
    from_add = Column(String(500), nullable=True)
    to_add = Column(String(500), nullable=True)
    subject = Column(String(500), nullable=True)
    body = Column(String(5000), nullable=True)
    sender_type = Column(String(100), nullable=True)
    classification = Column(String(100), nullable=True)
    intent_category = Column(String(255), nullable=True)
    opportunity_name = Column(String(500), nullable=True)
    account_name = Column(String(500), nullable=True)
    received_datetime = Column(String(100), nullable=True)
    send_datetime = Column(String(100), nullable=True)
    opportunity_id = Column(UUID(as_uuid=True), nullable=True)
    account_id = Column(UUID(as_uuid=True), nullable=True)
    type_tag = Column(String(50), nullable=True)
    priority = Column(String(50), nullable=True)
    source_label = Column(String(50), nullable=True)
    status = Column(String(50), nullable=True)
    due_date = Column(Date, nullable=True)
    seller_id = Column(UUID(as_uuid=True), nullable=True)

    created_at = Column(
        DateTime,
        server_default=func.now()
    )

    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now()
    )

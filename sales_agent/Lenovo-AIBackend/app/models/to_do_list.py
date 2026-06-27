from sqlalchemy import Column, Integer, String, Text, Date, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.database import Base


class TblToDoList(Base):
    __tablename__ = "tbl_to_do_list"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Text, nullable=True)
    opportunity_id = Column(UUID(as_uuid=True), nullable=True)

    task_title = Column(Text, nullable=False)
    type_tag = Column(String(100), nullable=True)
    priority = Column(String(100), nullable=True)
    source_label = Column(String(100), nullable=True)
    linked_account_id =Column(Text, nullable=False)
    linked_opportunity_id = Column(Text, nullable=False)
    attendees_email = Column(Text, nullable=True)
    notes = Column(Text)

    status = Column(String(50), default="Open")

    due_date = Column(Date, nullable=True)

    seller_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    created_at = Column(
        DateTime,
        server_default=func.now()
    )

    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now()
    )



from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.db.database import Base


class TblMeetingBriefing(Base):
    __tablename__ = "tbl_meeting_briefing"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Text, nullable=False, index=True)
    seller_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    payload = Column(JSONB, nullable=False)
    generated_at = Column(DateTime, server_default=func.now(), nullable=False)
    payload_version = Column(String(16), nullable=False, server_default="v1")


class TblMeetingPrepTask(Base):
    __tablename__ = "tbl_meeting_prep_task"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Text, nullable=False, index=True)
    seller_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    briefing_id = Column(Integer, ForeignKey("tbl_meeting_briefing.id"), nullable=True)
    description = Column(Text, nullable=False)
    priority = Column(String(16), nullable=False)
    evidence = Column(Text, nullable=False)
    confidence = Column(String(16), nullable=False, server_default="high")
    status = Column(String(16), nullable=False, server_default="open")
    sort_order = Column(Integer, nullable=False, server_default="0")
    source_type = Column(String(64), nullable=True)
    source_id = Column(String(128), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)


class TblMeetingPrepNote(Base):
    __tablename__ = "tbl_meeting_prep_note"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Text, nullable=False, index=True)
    seller_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    note_type = Column(String(32), nullable=False)
    body = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

import uuid

from sqlalchemy import (
    Column,
    BigInteger,
    Integer,
    String,
    Text,
    DateTime,
    UniqueConstraint
)

from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.database import Base


class CrmUpdate(Base):
    __tablename__ = "tbl_crm_updates"

    id = Column(
        BigInteger,
        primary_key=True,
        nullable=False
    )
    seller_id = Column(
        UUID(as_uuid=True)
        
    )
    meeting_id = Column(
        Text,
        nullable=False
    )
   
    seq_id = Column(
        Integer,
       
        nullable=False
    )

    account_id = Column(
        UUID(as_uuid=True),
        nullable=True
    )

    entity = Column(
        String(100),
        nullable=True
    )

    field_name = Column(
        String(255),
        nullable=True
    )
   
    current_value = Column(
        Text,
        nullable=True
    )

    suggested_value = Column(
        Text,
        nullable=True
    )
    confidence = Column(
        String(100),
        nullable=True
    )
    updated_by = Column(
        String(255),
        nullable=True
    )
    status = Column(
        String,
        nullable=True
    )
    reasoning = Column(
        Text,
        nullable=True
    )
    created_at = Column(
        DateTime,
        server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "account_id",
            "seq_id",
            name="uq_meeting_update_seq"
        ),
    )

   
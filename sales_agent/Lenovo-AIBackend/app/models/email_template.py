from sqlalchemy import Column, Integer, String, Text, Boolean

from app.db.database import Base

class EmailTemplate(Base):
    __tablename__ = "tbl_email_templates"

    id = Column(Integer, primary_key=True, index=True)
    template_name = Column(String(255), nullable=False)
    outreach_type = Column(String(100), nullable=False)
    context_used = Column(Text)
    is_active = Column(Boolean, default=True)
  
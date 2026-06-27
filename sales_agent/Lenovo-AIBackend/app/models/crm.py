"""Read-only CRM tables (shared Postgres with D365 Sales)."""

from sqlalchemy import Column, Numeric, String

from app.db.database import Base


class CrmAccount(Base):
    __tablename__ = "account"

    accountid = Column(String, primary_key=True)
    name = Column(String)


class CrmOpportunity(Base):
    __tablename__ = "opportunity"

    opportunityid = Column(String, primary_key=True)
    name = Column(String)
    estimatedvalue = Column(Numeric)
    stagename = Column(String)

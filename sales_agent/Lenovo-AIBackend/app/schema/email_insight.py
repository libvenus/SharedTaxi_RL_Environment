"""Schemas for the email-insight endpoint (single combined payload)."""

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class EmailInsightResponse(BaseModel):
    # The 4 resolved items
    opportunity_id: Optional[UUID] = Field(default=None, description="CRM opportunity id")
    opportunity_name: Optional[str] = Field(default=None, description="CRM opportunity name")
    account_id: Optional[UUID] = Field(default=None, description="CRM account id")
    account_name: Optional[str] = Field(default=None, description="CRM account name")

    # Deal facts (from CRM opportunity table)
    opportunity_value: Optional[float] = Field(default=None, description="Opportunity estimated value")

    # AI classification (already stored on the email row)
    classification: Optional[str] = Field(default=None, description="outreach / action / document")
    intent_category: Optional[str] = Field(default=None, description="Finer intent label")

    # AI-generated insight (from email body)
    why_now: str = Field(..., description="One-line reason we are drafting a new email")
    latest_activity: str = Field(..., description="Short summary of the previous email")

    # Source + telemetry
    email_id: str = Field(..., description="Source message_id from tbl_emails")
    trace_id: str = Field(..., description="Correlation id for this LLM call")
    latency_ms: int = Field(..., description="LLM round-trip latency in milliseconds")
    timestamp: str = Field(..., description="UTC ISO timestamp of generation")
    error: Optional[str] = Field(default=None, description="Set when the LLM could not produce insight")

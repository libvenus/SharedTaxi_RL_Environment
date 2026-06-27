"""Pydantic schemas for pre-meeting briefing card API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class BriefingSourceRef(BaseModel):
    source_type: str
    source_id: str
    label: str


class BriefingAttendee(BaseModel):
    name: str
    role: Optional[str] = None
    email: Optional[str] = None


class BriefingHeader(BaseModel):
    title: str
    start_at: datetime
    end_at: datetime
    duration_minutes: int
    platform: str
    join_url: Optional[str] = None
    attendees: List[BriefingAttendee] = Field(default_factory=list)


class BriefingAccountSection(BaseModel):
    paragraph: str
    word_count: int
    max_words: int
    gaps: List[str] = Field(default_factory=list)
    unverified_labels: List[str] = Field(default_factory=list)
    sources: List[BriefingSourceRef] = Field(default_factory=list)


class BriefingCompetitorIntel(BaseModel):
    items: List[dict] = Field(default_factory=list)
    message_code: Optional[str] = None


class BriefingDealSection(BaseModel):
    paragraph: str
    word_count: int
    max_words: int
    stage: Optional[str] = None
    gaps: List[str] = Field(default_factory=list)
    competitor_intel: BriefingCompetitorIntel
    sources: List[BriefingSourceRef] = Field(default_factory=list)


class BriefingSignal(BaseModel):
    signal_id: str
    summary: str
    why_shown: str
    event_at: datetime
    involved_parties: List[str] = Field(default_factory=list)
    source: BriefingSourceRef


class BriefingPrepTask(BaseModel):
    id: int
    description: str
    priority: Literal["HIGH", "MEDIUM", "LOW"]
    evidence: str
    confidence: Literal["high", "low"] = "high"
    done: bool = False
    source: Optional[BriefingSourceRef] = None


class BriefingTalkingPoint(BaseModel):
    talking_point: str
    why_shown: str
    sort_order: int
    source: BriefingSourceRef


class BriefingWatchOut(BaseModel):
    consideration: str
    why_shown: str
    source: BriefingSourceRef


class BriefingPrepNote(BaseModel):
    id: int
    note_type: Literal["typed", "voice_transcript"]
    body: str
    created_at: datetime
    updated_at: datetime
    is_seller_added: bool = True


class MeetingBriefingResponse(BaseModel):
    meeting_id: str
    seller_id: str
    generated_at: datetime
    is_ai_generated: bool = True
    header: BriefingHeader
    account_summary: BriefingAccountSection
    deal_summary: BriefingDealSection
    recent_signals: List[BriefingSignal] = Field(default_factory=list)
    prep_tasks: List[BriefingPrepTask] = Field(default_factory=list)
    talking_points: List[BriefingTalkingPoint] = Field(default_factory=list)
    talking_points_message_code: Optional[str] = None
    watch_out_for: Optional[List[BriefingWatchOut]] = None
    my_prep_notes: List[BriefingPrepNote] = Field(default_factory=list)


class PrepTaskStatusUpdate(BaseModel):
    done: bool


class PrepNoteCreate(BaseModel):
    body: str
    note_type: Literal["typed", "voice_transcript"] = "typed"


class PrepNoteUpdate(BaseModel):
    body: str


class VoiceNoteCreate(BaseModel):
    """Voice pipeline integration point — FE/AI sends transcript text."""

    transcript: str

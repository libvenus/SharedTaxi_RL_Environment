"""Pydantic request/response schemas for the data-hygiene task queue.

Sprint 1A · US04 — Data Hygiene, Validation & Intelligent Alerts

Casing convention is **snake_case** throughout (matches the rest of this
service — `/meeting-details/`, `/transcripts/`, `/consent-emails/`).

Notes
-----
* ``task_kind`` is intentionally a free-form ``str`` (not a Literal) so
  the AI team can introduce new transcript signals without a backend
  release. Canonical strings are listed in
  ``app/models/data_task.py`` and pinned in
  ``US04_BACKEND_HANDOFF_FOR_AI_TEAM.md``.
* ``current_value`` / ``suggested_value`` / ``resolved_value`` are stored
  as TEXT — the FE chooses how to render (strikethrough / typed input)
  based on ``field_name``. We don't want to leak D365 column types into
  this layer.
"""

from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Lifecycle whitelists — kept in sync with the CHECK constraints in
# sql/2026_06_us04_data_task.sql AND with the *_VALUES tuples in
# app/models/data_task.py. Edit ALL THREE together when adding a new value.
# ---------------------------------------------------------------------------
EntityKind = Literal["account", "contact", "opportunity"]
Severity = Literal["high", "medium", "low"]
TaskStatus = Literal["open", "resolved", "dismissed"]
Confidence = Literal["high", "medium", "low"]
CreatedBySource = Literal["transcript", "scan", "inline", "manual"]


# ---------------------------------------------------------------------------
# POST /api/data-tasks — generic task creation
# ---------------------------------------------------------------------------


class DataTaskCreateRequest(BaseModel):
    """Body for POST /api/data-tasks.

    Used by THREE distinct callers — they all converge on this one shape:

    1. The AI team's transcript-signal pipeline (``created_by_source='transcript'``)
    2. The daily-scan CLI job (``created_by_source='scan'``)
    3. FE inline validators (``created_by_source='inline'``, S1B)

    Idempotency contract: if there's already an OPEN task with the same
    ``(entity_kind, entity_id, task_kind)``, this endpoint returns that
    existing row (status 200) instead of creating a duplicate. If there's
    a DISMISSED task with the same key, the endpoint ALSO returns the
    existing dismissed row — this is how "dismissals suppress future
    alerts for that pair" (AC #5) is implemented.
    """

    owner_id: UUID = Field(
        description=(
            "The seller whose To-Do List this lands in. AC #11 portfolio "
            "scoping is enforced on read, but we still record it here."
        ),
    )
    entity_kind: EntityKind = Field(
        description="Which D365 entity type the issue is on.",
    )
    entity_id: UUID = Field(
        description="D365 entity UUID. Cross-repo FK-by-convention; not a Postgres FK.",
    )
    task_kind: str = Field(
        min_length=1,
        max_length=128,
        description=(
            "Canonical machine string identifying the rule that fired "
            "(e.g. 'past_close_date', 'transcript_signal_close_date_different'). "
            "See app/models/data_task.py for the AI-team-facing constants."
        ),
    )

    severity: Severity = Field(
        default="medium",
        description="Drives ordering in GET. Default 'medium'.",
    )
    confidence: Optional[Confidence] = Field(
        default=None,
        description=(
            "Set by the AI team for transcript signals. Deterministic "
            "scan detectors leave this NULL."
        ),
    )

    field_name: Optional[str] = Field(
        default=None,
        max_length=128,
        description=(
            "Which D365 field needs the seller's attention. Optional — "
            "many tasks are about a record-level issue (e.g. duplicate)."
        ),
    )
    current_value: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="Current display value, rendered with strikethrough on the FE.",
    )
    suggested_value: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="What the system / AI suggests as the corrected value.",
    )

    evidence_ref: Optional[str] = Field(
        default=None,
        max_length=512,
        description=(
            "Stable handle for the source — e.g. 'transcript_segment_id=<uuid>' "
            "or 'scan_run=2026-06-09T02:00:00Z'. Used by the FE to deep-link."
        ),
    )
    evidence_text: str = Field(
        min_length=1,
        max_length=2048,
        description=(
            "REQUIRED. The plain-language 'why' shown to the seller. "
            "AC #3: 'no alert is generated without a grounding reference.'"
        ),
    )

    created_by_source: CreatedBySource = Field(
        description="Which subsystem created this task (for metrics + debugging).",
    )


# ---------------------------------------------------------------------------
# Read responses
# ---------------------------------------------------------------------------


class DataTaskOut(BaseModel):
    """Full record as returned by GET endpoints.

    All audit fields (resolved_at, dismissed_at, dismissal_note,
    resolved_value, actor_id) are nullable — they're populated only on
    state transition. ``status='open'`` rows have all five as NULL.
    """

    model_config = ConfigDict(from_attributes=True)

    task_id: UUID
    owner_id: UUID
    entity_kind: EntityKind
    entity_id: UUID
    task_kind: str

    severity: Severity
    status: TaskStatus
    confidence: Optional[Confidence] = None

    field_name: Optional[str] = None
    current_value: Optional[str] = None
    suggested_value: Optional[str] = None

    evidence_ref: Optional[str] = None
    evidence_text: str

    created_by_source: CreatedBySource

    dismissal_note: Optional[str] = None
    resolved_value: Optional[str] = None
    actor_id: Optional[UUID] = None
    resolved_at: Optional[datetime] = None
    dismissed_at: Optional[datetime] = None

    created_at: datetime
    updated_at: datetime


class DataTaskCreateResponse(BaseModel):
    """Wrapper so the FE / AI team can tell whether the task was newly
    created or returned because of idempotency / suppression.

    ``was_existing=True`` means we found a matching row by
    ``(entity_kind, entity_id, task_kind)`` and returned it untouched.
    The caller should NOT treat this as an error — it's the documented
    happy-path for re-detection.
    """

    task: DataTaskOut
    was_existing: bool = Field(
        description=(
            "False = newly inserted; True = returned because an open or "
            "dismissed task with the same (entity, kind) already existed."
        ),
    )


class DataTaskListResponse(BaseModel):
    """Paginated list. Items are ordered:
    confidence DESC NULLS LAST, severity DESC, created_at ASC.
    """

    items: List[DataTaskOut]
    total: int = Field(
        ge=0,
        description="Total matching records (unpaginated count).",
    )


# ---------------------------------------------------------------------------
# POST /api/data-tasks/{task_id}/resolve
# ---------------------------------------------------------------------------


class DataTaskResolveRequest(BaseModel):
    """Body for POST /api/data-tasks/{task_id}/resolve.

    The seller has applied (or chosen to apply) the suggested fix. We
    record what they wrote back as ``resolved_value`` for audit. The
    actual write to D365 is done by the FE / form layer in v1 — backend
    only logs the resolution metadata. (See open question Q4 in the plan.)
    """

    actor_id: UUID = Field(
        description="The seller who resolved this task. Stored on the row for audit.",
    )
    resolved_value: Optional[str] = Field(
        default=None,
        max_length=2048,
        description=(
            "The value the seller wrote back to D365 (or empty if the "
            "resolution is something other than a field edit, e.g. linking "
            "an account to a parent)."
        ),
    )


class DataTaskResolveResponse(BaseModel):
    """Echo of the now-resolved task, plus an explicit was_already_resolved
    flag so idempotent re-resolves are visibly distinguishable from the
    first resolve.
    """

    task: DataTaskOut
    was_already_resolved: bool


# ---------------------------------------------------------------------------
# POST /api/data-tasks/{task_id}/dismiss
# ---------------------------------------------------------------------------


class DataTaskDismissRequest(BaseModel):
    """Body for POST /api/data-tasks/{task_id}/dismiss.

    AC #4: dismissals MUST come with a note. AC #5: a dismissal blocks
    future re-creation of the same (entity, kind) pair — that's enforced
    by the partial UNIQUE index in the SQL migration, not in this schema.
    """

    actor_id: UUID = Field(
        description="The seller who dismissed this task. Stored on the row for audit.",
    )
    note: str = Field(
        min_length=1,
        max_length=2048,
        description=(
            "REQUIRED dismissal reason. Empty / whitespace-only notes "
            "are rejected at the schema layer (and by the DB CHECK)."
        ),
    )


class DataTaskDismissResponse(BaseModel):
    """Echo of the now-dismissed task, plus an explicit was_already_dismissed
    flag for idempotency."""

    task: DataTaskOut
    was_already_dismissed: bool


# ---------------------------------------------------------------------------
# POST /api/data-tasks/scan — manual trigger for the daily scan
# ---------------------------------------------------------------------------


class ScanRunRequest(BaseModel):
    """Body for POST /api/data-tasks/scan.

    All fields optional — defaults to a full active-opportunity scan
    using the configured stale-days threshold. Mostly useful for ops /
    debugging; the cron-driven CLI invocation
    (``python -m app.jobs.scan_data_tasks``) does the same thing.
    """

    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=10_000,
        description=(
            "If set, scan at most this many opportunities. Useful in dev "
            "to avoid hitting D365 for the entire portfolio."
        ),
    )
    dry_run: bool = Field(
        default=False,
        description=(
            "If true, run all detectors but write NOTHING. The response "
            "still reports what WOULD have been created."
        ),
    )


class ScanRunResponse(BaseModel):
    """Summary of a scan run.

    Designed to be glanceable in a daily Slack / email digest. The
    invariants:

      total_scanned == tasks_created + tasks_skipped_existing +
                       tasks_skipped_dismissed + opportunities_with_errors
    """

    total_scanned: int = Field(ge=0)
    tasks_created: int = Field(ge=0)
    tasks_skipped_existing: int = Field(
        ge=0,
        description=(
            "Already had an OPEN task for the same (entity, kind) — "
            "scan was idempotent."
        ),
    )
    tasks_skipped_dismissed: int = Field(
        ge=0,
        description=(
            "Already had a DISMISSED task for the same (entity, kind) — "
            "AC #5 suppression in action."
        ),
    )
    opportunities_with_errors: int = Field(
        ge=0,
        description=(
            "Opps where the D365 risk-fetch (or detector) raised — "
            "scan continues past these, error count surfaced here."
        ),
    )
    dry_run: bool

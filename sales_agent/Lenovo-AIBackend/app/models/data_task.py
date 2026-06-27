"""ORM model for the data-hygiene task queue.

Sprint 1A · US04 — Data Hygiene, Validation & Intelligent Alerts

One table:
  - tbl_data_task — one row per detected data-quality issue, regardless of
                    whether it came from the AI team's transcript pipeline,
                    the daily-scan CLI job, or an FE inline validator.

See sql/2026_06_us04_data_task.sql for the migration + idempotency contract.
"""

import uuid

from sqlalchemy import Column, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.database import Base


# ---------------------------------------------------------------------------
# Lifecycle whitelists — kept in sync with the CHECK constraints in
# sql/2026_06_us04_data_task.sql AND with the Pydantic Literals in
# app/schema/data_task.py. Edit ALL THREE together when adding a new value.
# ---------------------------------------------------------------------------
ENTITY_KIND_VALUES = (
    "account",
    "contact",
    "opportunity",
)

SEVERITY_VALUES = (
    "high",
    "medium",
    "low",
)

# Order matters — task_status_to_terminal() and the suppression-on-dismissal
# logic both rely on 'open' being the only non-terminal value.
TASK_STATUS_VALUES = (
    "open",
    "resolved",
    "dismissed",
)

# Confidence is OPTIONAL — deterministic detectors leave it NULL.
CONFIDENCE_VALUES = (
    "high",
    "medium",
    "low",
)

CREATED_BY_SOURCE_VALUES = (
    "transcript",   # AI team's NLP pipeline (POST /api/data-tasks)
    "scan",         # daily-scan CLI job
    "inline",       # FE inline validator (S1B)
    "manual",       # admin / debug
)

# Canonical task_kind strings the AI team and the daily scan write. The DB
# column is free-form TEXT (no CHECK enum) so the AI team can introduce
# new transcript signals without a backend release — but if you're
# implementing a detector or wiring up an AI signal, USE these constants
# so the FE can group + i18n consistently.
#
# Deterministic detectors (Sprint 1A scan job):
TASK_KIND_PAST_CLOSE_DATE = "past_close_date"
TASK_KIND_ZERO_OR_MISSING_VALUE = "zero_or_missing_value"
TASK_KIND_STALE_ACTIVITY = "stale_activity"
TASK_KIND_RISK_FLAG = "risk_flag"  # one per row from D365 /opportunities/{id}/risks

# Transcript-signal kinds (AI team POSTs these — see handoff doc):
TASK_KIND_TRANSCRIPT_LOCATION_CHANGE = "transcript_signal_location_change"
TASK_KIND_TRANSCRIPT_HEADCOUNT_CHANGE = "transcript_signal_headcount_change"
TASK_KIND_TRANSCRIPT_NEW_DECISION_MAKER = "transcript_signal_new_decision_maker"
TASK_KIND_TRANSCRIPT_ACQUISITION = "transcript_signal_acquisition"
TASK_KIND_TRANSCRIPT_BUDGET_CYCLE_CHANGE = "transcript_signal_budget_cycle_change"
TASK_KIND_TRANSCRIPT_CLOSE_DATE_DIFFERENT = "transcript_signal_close_date_different"
TASK_KIND_TRANSCRIPT_QUANTITY_DIFFERENT = "transcript_signal_quantity_different"
TASK_KIND_TRANSCRIPT_BUDGET_DIFFERENT = "transcript_signal_budget_different"
TASK_KIND_TRANSCRIPT_REQUIREMENT_CHANGE = "transcript_signal_requirement_change"
TASK_KIND_TRANSCRIPT_UNLOGGED_COMPETITOR = "transcript_signal_unlogged_competitor"


class DataTask(Base):
    """One row per detected data-hygiene issue.

    A task is uniquely identified by ``(entity_kind, entity_id, task_kind)``
    *while it is open or dismissed* — the partial UNIQUE index in the SQL
    migration enforces this. That gives us two important guarantees:

    1. The daily scan can be re-run safely: redetecting the same issue on
       the same record is a silent no-op.
    2. A dismissed task continues to occupy the slot, so an intentionally
       dismissed mismatch will NOT regenerate (AC #5).

    Once a task is ``resolved``, the slot frees up and a fresh occurrence
    (e.g. the seller closes the deal then a new transcript flags a stale
    update) can produce a new task. That's intentional — resolution means
    the data was good *at that time*; later drift is a new event.
    """

    __tablename__ = "tbl_data_task"

    task_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    owner_id = Column(UUID(as_uuid=True), nullable=False)

    entity_kind = Column(Text, nullable=False)
    entity_id = Column(UUID(as_uuid=True), nullable=False)

    task_kind = Column(Text, nullable=False)

    severity = Column(Text, nullable=False, server_default="medium")
    status = Column(Text, nullable=False, server_default="open")

    field_name = Column(Text)
    current_value = Column(Text)
    suggested_value = Column(Text)

    confidence = Column(Text)

    evidence_ref = Column(Text)
    evidence_text = Column(Text, nullable=False)

    created_by_source = Column(Text, nullable=False)

    dismissal_note = Column(Text)
    resolved_value = Column(Text)
    actor_id = Column(UUID(as_uuid=True))
    resolved_at = Column(DateTime(timezone=True))
    dismissed_at = Column(DateTime(timezone=True))

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

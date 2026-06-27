"""Pure detector functions for the data-hygiene daily scan.

Sprint 1A · US04 — Data Hygiene, Validation & Intelligent Alerts

Each detector is a pure function:

    (opportunity_data, ...config) -> DataTaskCreateRequest | None

No DB writes, no D365 calls. The orchestrator
(``app/jobs/scan_data_tasks.py``) feeds the data in, collects the
``DataTaskCreateRequest`` objects, and calls ``data_task_service.create_task``
for each. This separation makes detectors trivially unit-testable and
keeps the scan-job loop readable.

S1A scope (4 detectors):
  D1. detect_past_close_date         — close_date < today AND still active
  D2. detect_zero_or_missing_value   — estimated_value IS NULL OR == 0
  D3. detect_stale_activity          — last_activity older than threshold
  D4. detect_risk_flags              — one task per row from D365 risks API

Deferred to S1B (per the plan doc §1):
  L5  territory mismatch
  L6  contact/account email-domain mismatch
  L7  stage dwell-time anomaly
  L8  parent-child account suggestion
  L9  duplicate-contact (fuzzy)
  multi-signal duplicate-opportunity
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional
from uuid import UUID

from app.clients.d365_client import OpportunityRisk, OpportunityScanRow
from app.models.data_task import (
    TASK_KIND_PAST_CLOSE_DATE,
    TASK_KIND_RISK_FLAG,
    TASK_KIND_STALE_ACTIVITY,
    TASK_KIND_ZERO_OR_MISSING_VALUE,
)
from app.schema.data_task import DataTaskCreateRequest


# ---------------------------------------------------------------------------
# Active-stage helper
# ---------------------------------------------------------------------------


def _is_active(opp: OpportunityScanRow) -> bool:
    """True iff the opp is in an active stage (not Closed Won / Closed Lost).

    The D365 client already filters Closed* opps out of
    ``list_active_opportunities``, but we re-check here so unit tests
    that build OpportunityScanRow directly don't accidentally bypass
    the rule.
    """
    statecode = opp.statecode
    if isinstance(statecode, str) and statecode.lower().startswith("closed"):
        return False
    return True


def _evidence_ref_for_scan(scan_run_at: datetime) -> str:
    """Stable evidence-ref string for tasks born from a scan run.

    Format: ``scan_run=2026-06-09T02:00:00+00:00``. The FE doesn't
    deep-link to scan runs (there's nothing to deep-link to), but this
    string is useful when grepping audit data to ask "which scan
    produced this task?"
    """
    if scan_run_at.tzinfo is None:
        scan_run_at = scan_run_at.replace(tzinfo=timezone.utc)
    return f"scan_run={scan_run_at.isoformat()}"


# ---------------------------------------------------------------------------
# D1. Past close date but opportunity still open
# ---------------------------------------------------------------------------


def detect_past_close_date(
    opp: OpportunityScanRow,
    *,
    today: date,
    scan_run_at: datetime,
) -> Optional[DataTaskCreateRequest]:
    """Fires when close_date < today on a still-active opp.

    Severity 'high' — a missed close date is the single strongest signal
    the deal is stale and the seller needs to act NOW.

    Returns None when:
      - close_date is missing (we can't say it's in the past)
      - close_date >= today (the deal is on time)
      - opp is closed (already ruled out at the client layer, double-checked)
      - owner_id is missing (we can't route a task without an assignee)
    """
    if not _is_active(opp):
        return None
    if opp.close_date is None or opp.close_date >= today:
        return None
    if opp.owner_id is None:
        # Without an owner we can't route a task. Log the orphan opp
        # via the scan-job summary; surface it through some other means.
        return None

    days_overdue = (today - opp.close_date).days
    return DataTaskCreateRequest(
        owner_id=opp.owner_id,
        entity_kind="opportunity",
        entity_id=opp.opportunity_id,
        task_kind=TASK_KIND_PAST_CLOSE_DATE,
        severity="high",
        confidence=None,  # Deterministic
        field_name="close_date",
        current_value=opp.close_date.isoformat(),
        suggested_value=None,
        evidence_ref=_evidence_ref_for_scan(scan_run_at),
        evidence_text=(
            f"Close date {opp.close_date.isoformat()} has passed "
            f"({days_overdue} day{'s' if days_overdue != 1 else ''} ago) "
            f"but the opportunity is still active. Update the close date "
            f"or move the deal to its terminal stage."
        ),
        created_by_source="scan",
    )


# ---------------------------------------------------------------------------
# D2. Zero or missing deal value
# ---------------------------------------------------------------------------


def detect_zero_or_missing_value(
    opp: OpportunityScanRow,
    *,
    scan_run_at: datetime,
) -> Optional[DataTaskCreateRequest]:
    """Fires when estimated_value IS NULL OR == 0 on an active opp.

    Severity 'medium' — an empty deal value distorts pipeline reporting
    but doesn't necessarily mean the deal itself is stale. The inline
    validator on save (D365 Sales' ``deals_write.py`` ERR_MSG_0012) blocks
    NEW zero-value opps; this detector catches LEGACY rows that pre-date
    the validator.
    """
    if not _is_active(opp):
        return None
    if opp.estimated_value not in (None, 0, 0.0):
        return None
    if opp.owner_id is None:
        return None

    return DataTaskCreateRequest(
        owner_id=opp.owner_id,
        entity_kind="opportunity",
        entity_id=opp.opportunity_id,
        task_kind=TASK_KIND_ZERO_OR_MISSING_VALUE,
        severity="medium",
        confidence=None,
        field_name="estimated_value",
        current_value=(
            "0" if opp.estimated_value == 0 else None
        ),
        suggested_value=None,
        evidence_ref=_evidence_ref_for_scan(scan_run_at),
        evidence_text=(
            "Opportunity has no deal value set. Pipeline forecasts and "
            "deal-health calculations both depend on a non-zero estimated "
            "value — please enter the expected deal amount."
        ),
        created_by_source="scan",
    )


# ---------------------------------------------------------------------------
# D3. Stale activity (no logged activity for >N days)
# ---------------------------------------------------------------------------


def detect_stale_activity(
    opp: OpportunityScanRow,
    *,
    today: date,
    stale_days: int,
    scan_run_at: datetime,
) -> Optional[DataTaskCreateRequest]:
    """Fires when there's been no activity logged for ``stale_days`` days.

    Severity 'medium' for stale, 'high' if it's been more than 2x the
    threshold (so a 90-day-quiet deal screams louder than a 31-day-quiet
    one).

    NOTE: ``last_activity`` may legitimately be NULL on a brand-new opp
    that was just created; in that case we fall back to comparing
    ``today`` against the opp's creation date — but our scan row doesn't
    carry creation date today, so we conservatively SKIP no-activity-ever
    opps. (A separate detector for "newly-created opp with no activity
    logged in N days" can land in S1B.)
    """
    if not _is_active(opp):
        return None
    if opp.last_activity is None:
        return None
    if opp.owner_id is None:
        return None

    last = opp.last_activity
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    today_dt = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
    age_days = (today_dt - last).days
    if age_days < stale_days:
        return None

    severity = "high" if age_days >= stale_days * 2 else "medium"

    return DataTaskCreateRequest(
        owner_id=opp.owner_id,
        entity_kind="opportunity",
        entity_id=opp.opportunity_id,
        task_kind=TASK_KIND_STALE_ACTIVITY,
        severity=severity,
        confidence=None,
        field_name=None,
        current_value=last.date().isoformat(),
        suggested_value=None,
        evidence_ref=_evidence_ref_for_scan(scan_run_at),
        evidence_text=(
            f"No activity logged in {age_days} days "
            f"(last activity {last.date().isoformat()}; threshold {stale_days} days). "
            f"Log a meeting, call, or email — or close the deal if it's gone cold."
        ),
        created_by_source="scan",
    )


# ---------------------------------------------------------------------------
# D4. Risk-flag → task materialisation
# ---------------------------------------------------------------------------


# Risk-category → severity mapping. D365's deal_risks.py categorises
# every risk into one of four buckets; we map them to our severity scale
# so the To-Do queue ordering stays consistent.
_RISK_CATEGORY_SEVERITY = {
    "Activity & Engagement": "medium",
    "Stakeholder": "medium",
    "Deal Execution": "high",      # Missing critical data → block deal progression
    "Timeline & Forecast": "high",  # Slipping close-date / wrong stage
}


def detect_risk_flags(
    opp_id: UUID,
    risks: list[OpportunityRisk],
    *,
    owner_id: UUID,
    scan_run_at: datetime,
) -> list[DataTaskCreateRequest]:
    """One ``DataTaskCreateRequest`` per risk row from D365.

    The user-story mandate "Risk and warning flags from the Opportunity
    details screen creates To-Do tasks" maps directly here — D365
    already runs ``deal_risks.compute_risks()`` and persists the result
    into ``lvo_dealrisk``; we just expose those rows in the seller's
    To-Do queue alongside our other detectors.

    The ``task_kind`` is the constant ``'risk_flag'`` for ALL risks,
    NOT a per-risk-name string — the FE can drill into ``field_name``
    (which carries the risk's stable name like 'Low Activity') if it
    wants to group by risk type. This keeps the partial UNIQUE index
    selective: at most ONE 'risk_flag' task per opportunity at a time,
    not one per risk-category-name combination.

    NOTE: that means if D365 reports two risks for the same opp, only
    the FIRST one gets persisted as a task in this scan. The seller
    sees one task with the most-severe risk's text; subsequent risks
    surface in later scans once the first is resolved. We can revisit
    in S1B if PM wants per-risk task fan-out.
    """
    if not risks:
        return []

    # Pick the most-severe risk for the single 'risk_flag' task — sort
    # by mapped severity (high > medium > low) then by the risk's name
    # alphabetically for determinism.
    severity_rank = {"high": 3, "medium": 2, "low": 1}

    def _score(r: OpportunityRisk) -> tuple[int, str]:
        sev = _RISK_CATEGORY_SEVERITY.get(r.category, "medium")
        return (severity_rank.get(sev, 2), r.name)

    primary = max(risks, key=_score)
    severity = _RISK_CATEGORY_SEVERITY.get(primary.category, "medium")

    return [
        DataTaskCreateRequest(
            owner_id=owner_id,
            entity_kind="opportunity",
            entity_id=opp_id,
            task_kind=TASK_KIND_RISK_FLAG,
            severity=severity,
            confidence=None,
            field_name=primary.name,
            current_value=None,
            suggested_value=None,
            evidence_ref=(
                f"d365_risk_id={primary.risk_id}"
                if primary.risk_id
                else _evidence_ref_for_scan(scan_run_at)
            ),
            evidence_text=primary.message,
            created_by_source="scan",
        )
    ]

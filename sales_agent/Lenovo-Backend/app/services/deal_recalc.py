"""Deal Health + Risk recalculation orchestrator.

This is the only place where the pure scoring/risk services meet the
SQLAlchemy session. Routers call it; the FastAPI BackgroundTasks queue
calls it (via the `recalculate_async` wrapper that opens its own session
because the request-scoped session is already closed by then); the batch
job in `app/jobs/recalc_health.py` calls it in a loop.

Responsibilities
----------------
1. Load every input the calculators need from the DB in one efficient pass.
2. Run risks → health (in that order, so risk_count feeds Risk Adjustment).
3. Optionally persist:
     * opportunity.lvo_dealhealthscore        — composite 0–100
     * opportunity.lvo_riskreason             — top-priority risk message (legacy)
     * opportunity.lvo_dealhealthupdatedat    — now()
     * lvo_dealrisk                           — full delete-then-insert per deal
     * lvo_opportunitycontact.lvo_lasttouchdate
                                              — denormalised cache for the
                                              contact panel
4. Return a structured result the recalc endpoint serialises directly.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import (
    Activity,
    DealHealthConfig,
    DealRisk,
    NextAction,
    Opportunity,
    OpportunityContact,
)
from app.services.audit_log import write_audit_event
from app.services.deal_health import (
    DEFAULT_SETTINGS,
    DealHealthBreakdown,
    DealHealthInputs,
    DealHealthSettings,
    compose_deal_health,
)
from app.services.deal_risks import DealRiskInputs, Risk, evaluate_risks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecalculationResult:
    """What the recalculator hands back to the caller."""

    opportunity_id: str
    breakdown: DealHealthBreakdown
    risks: list[Risk]
    persisted_risk_ids: list[str]
    persisted: bool
    updated_at: datetime | None


# ---------------------------------------------------------------------------
# Settings loader (with cache invalidation per request — cheap query)
# ---------------------------------------------------------------------------


def load_settings(db: Session) -> DealHealthSettings:
    """Load tunables from `lvo_dealhealthconfig` (id=1) or fall back to defaults.

    The fallback lets the API survive a fresh DB where the config-seed
    migration hasn't been applied yet — health still calculates with the
    documented defaults.
    """
    row = db.execute(
        select(DealHealthConfig.lvo_settings).where(DealHealthConfig.id == 1)
    ).scalar_one_or_none()
    if row is None:
        return DEFAULT_SETTINGS
    try:
        return DealHealthSettings.from_config_dict(row)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "lvo_dealhealthconfig contains malformed settings; "
            "falling back to defaults. Error: %s",
            exc,
        )
        return DEFAULT_SETTINGS


# ---------------------------------------------------------------------------
# Data loaders — one query per concern, batched per opportunity.
# ---------------------------------------------------------------------------


def _load_opportunity(db: Session, opportunity_id: str) -> Opportunity | None:
    return db.get(Opportunity, opportunity_id)


def _load_last_activity_date(
    db: Session, opportunity_id: str
) -> datetime | None:
    return db.execute(
        select(func.max(Activity.lvo_activitydate)).where(
            func.upper(Activity.lvo_opportunityid) == opportunity_id.upper(),
            Activity.statecode == "Active",
        )
    ).scalar()


def _load_stakeholder_signals(
    db: Session, opportunity_id: str
) -> tuple[int, bool]:
    """Return (active_stakeholder_count, decision_maker_present)."""
    rows = db.execute(
        select(OpportunityContact.lvo_isdecisionmaker).where(
            func.upper(OpportunityContact.lvo_opportunityid)
            == opportunity_id.upper(),
            OpportunityContact.statecode == "Active",
        )
    ).all()
    count = len(rows)
    has_dm = any(bool(r[0]) for r in rows)
    return count, has_dm


def _load_next_action_signals(
    db: Session, opportunity_id: str
) -> tuple[int, int]:
    """Return (open_count, open_count_missing_due_date)."""
    rows = db.execute(
        select(NextAction.lvo_duedate).where(
            func.upper(NextAction.lvo_opportunityid) == opportunity_id.upper(),
            NextAction.statecode == "Active",
            NextAction.lvo_status == "Open",
        )
    ).all()
    open_count = len(rows)
    missing_date = sum(1 for r in rows if r[0] is None)
    return open_count, missing_date


# ---------------------------------------------------------------------------
# Inputs assembly
# ---------------------------------------------------------------------------


def _build_inputs(
    opp: Opportunity,
    *,
    last_activity: datetime | None,
    stakeholder_count: int,
    decision_maker_present: bool,
    open_next_actions: int,
    next_actions_missing_date: int,
    settings: DealHealthSettings,
    today: date,
) -> tuple[DealRiskInputs, DealHealthInputs]:
    """Assemble both input structs from a freshly-loaded opportunity row."""
    tempo = opp.lvo_tempoclass or "Quarterly"
    target_total_days = float(
        settings.tempo_class_target_days.get(_canonicalise_tempo(tempo), 90)
    )

    actual_days_in_stage = 0.0
    if opp.lvo_stageentrydate is not None:
        actual_days_in_stage = float(
            (today - opp.lvo_stageentrydate.date()).days
        )

    last_activity_age_days: float | None = None
    if last_activity is not None:
        last_activity_age_days = float((today - last_activity.date()).days)

    # Approximation: we don't yet wire individual contacts to activities,
    # so DM engagement is "DM is mapped AND deal has *any* recent activity".
    decision_maker_engaged = (
        decision_maker_present
        and last_activity_age_days is not None
        and last_activity_age_days <= float(settings.low_activity_days_threshold)
    )

    is_closed = (opp.statecode or "").lower() in {
        "won",
        "closed won",
        "lost",
        "closed lost",
    } or (opp.stagename or "") in {"Closed Won", "Closed Lost"}

    risk_inputs = DealRiskInputs(
        stage=opp.stagename,
        statecode=opp.statecode,
        estimated_value=float(opp.estimatedvalue) if opp.estimatedvalue is not None else None,
        close_date=opp.estimatedclosedate,
        name=opp.name,
        tempo_class=tempo,
        today=today,
        last_activity_date=last_activity,
        stage_entry_date=opp.lvo_stageentrydate,
        created_at=opp.lvo_createdat,
        active_stakeholders=stakeholder_count,
        decision_maker_present=decision_maker_present,
        decision_maker_engaged=decision_maker_engaged,
        open_next_actions=open_next_actions,
        next_actions_missing_date=next_actions_missing_date,
    )

    health_inputs = DealHealthInputs(
        stage=opp.stagename,
        tempo_class=tempo,
        actual_days_in_stage=actual_days_in_stage,
        target_close_total_days=target_total_days,
        last_activity_age_days=last_activity_age_days,
        active_stakeholders=stakeholder_count,
        created_at=opp.lvo_createdat,
        close_date=opp.estimatedclosedate,
        is_closed=is_closed,
        risk_count=0,  # filled in after risks evaluate
        today=today,
    )

    return risk_inputs, health_inputs


def _canonicalise_tempo(raw: str | None) -> str:
    """Tiny mirror of deal_health._normalise_tempo_class — public-facing."""
    from app.services.deal_health import _normalise_tempo_class

    return _normalise_tempo_class(raw)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist(
    db: Session,
    opp: Opportunity,
    breakdown: DealHealthBreakdown,
    risks: list[Risk],
    *,
    now: datetime,
) -> list[str]:
    """Write the computed results back to the DB.

    Returns the list of newly-inserted lvo_dealrisk ids.
    """
    opp.lvo_dealhealthscore = int(breakdown.score)
    opp.lvo_riskreason = risks[0].message if risks else None
    opp.lvo_dealhealthupdatedat = now

    # `opportunity.opportunityid` is a UUID-typed column → SQLAlchemy hands
    # us back a real `uuid.UUID` object. Stringify before any text ops.
    # `lvo_dealrisk.lvo_opportunityid` is TEXT, and the project convention
    # (mirroring `lvo_opportunitycompetitor` etc.) stores it uppercase.
    opp_id_text = str(opp.opportunityid).upper()

    # Replace the active risks for this deal in one transaction. The
    # ORM-level `delete(DealRisk)` form keeps the unit-of-work consistent
    # so any session-cached DealRisk objects for this deal are evicted.
    db.execute(
        delete(DealRisk).where(
            func.upper(DealRisk.lvo_opportunityid) == opp_id_text,
            DealRisk.statecode == "Active",
        ).execution_options(synchronize_session=False)
    )

    new_ids: list[str] = []
    for r in risks:
        risk_id = str(uuid.uuid4())
        new_ids.append(risk_id)
        db.add(
            DealRisk(
                lvo_dealriskid=risk_id,
                lvo_opportunityid=opp_id_text,
                lvo_riskcategory=r.category,
                lvo_riskname=r.name,
                lvo_message=r.message,
                lvo_detectedat=now,
                statecode="Active",
            )
        )

    db.commit()
    return new_ids


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def recalculate_deal_health(
    db: Session,
    opportunity_id: str,
    *,
    write: bool = True,
    today: date | None = None,
    audit_actor_type: str = "ai",
    audit_category: str = "ai_automated",
    audit_changed_by: str | None = None,
) -> RecalculationResult | None:
    """Run risks + health for one deal.

    Parameters
    ----------
    db
        Active SQLAlchemy session.
    opportunity_id
        opportunity.opportunityid (UUID-as-string, any case).
    write
        When ``True`` (default), persists the results to the DB. The
        synchronous /health/recalculate endpoint passes True; the read-only
        /health and /risks endpoints pass False so they don't mutate state
        on every page load.
    today
        Override "today" — used by tests. Production callers leave this
        unset (UTC today is used).

    Returns
    -------
    The RecalculationResult, or ``None`` if the opportunity was not found.
    """
    opp = _load_opportunity(db, opportunity_id)
    if opp is None:
        return None

    settings = load_settings(db)
    last_activity = _load_last_activity_date(db, opportunity_id)
    stakeholder_count, has_dm = _load_stakeholder_signals(db, opportunity_id)
    open_next_actions, na_missing_date = _load_next_action_signals(db, opportunity_id)

    today = today or datetime.now(timezone.utc).date()

    risk_inputs, health_inputs = _build_inputs(
        opp,
        last_activity=last_activity,
        stakeholder_count=stakeholder_count,
        decision_maker_present=has_dm,
        open_next_actions=open_next_actions,
        next_actions_missing_date=na_missing_date,
        settings=settings,
        today=today,
    )

    risks = evaluate_risks(risk_inputs, settings)

    health_inputs = DealHealthInputs(
        stage=health_inputs.stage,
        tempo_class=health_inputs.tempo_class,
        actual_days_in_stage=health_inputs.actual_days_in_stage,
        target_close_total_days=health_inputs.target_close_total_days,
        last_activity_age_days=health_inputs.last_activity_age_days,
        active_stakeholders=health_inputs.active_stakeholders,
        created_at=health_inputs.created_at,
        close_date=health_inputs.close_date,
        is_closed=health_inputs.is_closed,
        risk_count=len(risks),
        today=today,
    )
    breakdown = compose_deal_health(health_inputs, settings)

    persisted_ids: list[str] = []
    updated_at: datetime | None = None
    if write:
        now = datetime.now(timezone.utc)
        persisted_ids = _persist(db, opp, breakdown, risks, now=now)
        updated_at = now
        try:
            write_audit_event(
                db,
                entity_type="deal_health",
                entity_id=opportunity_id,
                opportunity_id=opportunity_id,
                action="recalculate",
                category=audit_category,  # type: ignore[arg-type]
                actor_type=audit_actor_type,  # type: ignore[arg-type]
                changed_by=audit_changed_by,
                diff={
                    "score": int(breakdown.score),
                    "band": breakdown.band,
                    "riskCount": len(risks),
                },
            )
            db.commit()
        except Exception:
            logger.warning(
                "Deal health persisted but audit log write failed for %s",
                opportunity_id,
                exc_info=True,
            )
            db.rollback()
    else:
        # Refresh-only (read-mode): do NOT touch the DB; just bubble up the
        # last-known timestamp so the caller can render "as of".
        updated_at = opp.lvo_dealhealthupdatedat

    return RecalculationResult(
        opportunity_id=str(opp.opportunityid),
        breakdown=breakdown,
        risks=risks,
        persisted_risk_ids=persisted_ids,
        persisted=write,
        updated_at=updated_at,
    )


def recalculate_async(opportunity_id: str) -> None:
    """Background-task wrapper — opens its own session.

    Wire this into FastAPI's BackgroundTasks instead of the request-scoped
    session because the request session is already closed by the time
    the task runs.

    After the deal recalc finishes successfully, also enqueues an account
    recalc for the owning account — deal-health changes feed the account's
    "At-Risk" derivation, so the two must stay in sync.
    """
    db = SessionLocal()
    account_id: str | None = None
    try:
        try:
            recalculate_deal_health(db, opportunity_id, write=True)
            opp = db.get(Opportunity, opportunity_id)
            if opp is not None and opp.accountid:
                account_id = str(opp.accountid)
        except Exception:
            logger.exception(
                "Failed to recalculate health for opportunity %s",
                opportunity_id,
            )
            db.rollback()
    finally:
        db.close()

    if account_id:
        # Imported here to avoid a circular import at module load time.
        from app.services.account_recalc import recalculate_async as recalc_account
        recalc_account(account_id)


def recalculate_many(
    db: Session, opportunity_ids: Iterable[str], *, write: bool = True
) -> list[RecalculationResult]:
    """Batch helper used by `app/jobs/recalc_health.py`."""
    results: list[RecalculationResult] = []
    for oid in opportunity_ids:
        result = recalculate_deal_health(db, oid, write=write)
        if result is not None:
            results.append(result)
    return results


__all__ = [
    "RecalculationResult",
    "load_settings",
    "recalculate_async",
    "recalculate_deal_health",
    "recalculate_many",
]

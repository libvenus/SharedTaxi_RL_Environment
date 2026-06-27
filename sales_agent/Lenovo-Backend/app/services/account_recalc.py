"""Account-level recalculator.

Bridges the pure ``account_status`` rules with the database. Pattern is the
same as ``deal_recalc.py``:

* one orchestrator function (``recalculate_account``) used by routers,
* a ``recalculate_async`` wrapper that opens its own session for FastAPI
  BackgroundTasks,
* a batch helper (``recalculate_many``) used by ``app/jobs/recalc_accounts.py``.

Persists three columns on ``account``:

* ``lvo_accounttype``         — Prospect / Customer
* ``lvo_accountstatus``       — Active / Inactive / At-Risk
* ``lvo_lastinteractiondate`` — denormalised cache of MAX(lvo_activitydate)

Loaders defensively introspect the schema so a partially-applied migration
(or a fresh DB before the optional columns exist) doesn't break recalc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import String, cast, func, inspect, select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Account, Activity, DealHealthConfig, Opportunity
from app.services.account_status import (
    DEFAULT_ACCOUNT_STATUS_SETTINGS,
    AccountStatusInputs,
    AccountStatusSettings,
    AccountStatusValue,
    AccountTypeValue,
    derive_account_status,
    derive_account_type,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountRecalcResult:
    """Snapshot the routers + batch job hand back to callers."""

    account_id: str
    account_type: AccountTypeValue
    status: AccountStatusValue
    last_interaction: datetime | None
    persisted: bool


# ---------------------------------------------------------------------------
# Settings loader (delegates to the shared lvo_dealhealthconfig row)
# ---------------------------------------------------------------------------


def load_account_status_settings(db: Session) -> AccountStatusSettings:
    """Read ``account_status`` from ``lvo_dealhealthconfig.lvo_settings``.

    Falls back to defaults when the table or sub-key is missing — keeps the
    API alive on a fresh DB before the migrations have been applied.
    """
    if not _has_table(db, "lvo_dealhealthconfig"):
        return DEFAULT_ACCOUNT_STATUS_SETTINGS
    row = db.execute(
        select(DealHealthConfig.lvo_settings).where(DealHealthConfig.id == 1)
    ).scalar_one_or_none()
    if not isinstance(row, dict):
        return DEFAULT_ACCOUNT_STATUS_SETTINGS
    return AccountStatusSettings.from_config(row.get("account_status"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_table(db: Session, table_name: str) -> bool:
    return inspect(db.bind).has_table(table_name)


def _has_account_view_columns(db: Session) -> bool:
    """True only when sql/2026_06_account_view_schema.sql has been applied."""
    cols = {c["name"] for c in inspect(db.bind).get_columns("account")}
    return {
        "lvo_accounttype",
        "lvo_accountstatus",
        "lvo_lastinteractiondate",
    }.issubset(cols)


def _has_dealhealth_column(db: Session) -> bool:
    cols = {c["name"] for c in inspect(db.bind).get_columns("opportunity")}
    return "lvo_dealhealthscore" in cols


# ---------------------------------------------------------------------------
# Loaders — one query per concern
# ---------------------------------------------------------------------------


def _load_account(db: Session, account_id: str) -> Account | None:
    # Case-insensitive — see _ensure_account in accounts.py for the full
    # rationale (lowercase DB text vs uppercase URL).
    return db.execute(
        select(Account).where(
            func.upper(cast(Account.accountid, String)) == account_id.upper()
        )
    ).scalar_one_or_none()


def _load_deal_signals(
    db: Session, account_id: str, *, at_risk_threshold: int
) -> tuple[bool, int, int]:
    """Return (has_won_deal, open_deals_count, open_deals_with_low_health).

    Counts only non-Canceled deals. Low health is determined against the
    persisted ``lvo_dealhealthscore`` column when present; if the column
    isn't there yet we treat every open deal as healthy (so the account
    cannot flip to At-Risk before the deal-health migration ships).
    """
    has_dealhealth = _has_dealhealth_column(db)

    has_won = bool(
        db.execute(
            select(func.count())
            .select_from(Opportunity)
            .where(
                func.upper(Opportunity.accountid) == account_id.upper(),
                (
                    Opportunity.statecode.in_(("Won", "Closed Won"))
                    | Opportunity.stagename.in_(("Closed Won",))
                ),
            )
        ).scalar_one()
    )

    open_count = int(
        db.execute(
            select(func.count())
            .select_from(Opportunity)
            .where(
                func.upper(Opportunity.accountid) == account_id.upper(),
                Opportunity.statecode == "Open",
            )
        ).scalar_one()
        or 0
    )

    low_health_count = 0
    if has_dealhealth:
        low_health_count = int(
            db.execute(
                select(func.count())
                .select_from(Opportunity)
                .where(
                    func.upper(Opportunity.accountid) == account_id.upper(),
                    Opportunity.statecode == "Open",
                    Opportunity.lvo_dealhealthscore.is_not(None),
                    Opportunity.lvo_dealhealthscore < at_risk_threshold,
                )
            ).scalar_one()
            or 0
        )

    return has_won, open_count, low_health_count


def _load_last_interaction(db: Session, account_id: str) -> datetime | None:
    """MAX(lvo_activity.lvo_activitydate) joined through opportunity.

    Returns None when the activity table is missing or no activities exist.
    """
    if not _has_table(db, "lvo_activity"):
        return None

    return db.execute(
        select(func.max(Activity.lvo_activitydate))
        .join(
            Opportunity,
            func.upper(Activity.lvo_opportunityid)
            == func.upper(cast(Opportunity.opportunityid, String)),
        )
        .where(
            func.upper(Opportunity.accountid) == account_id.upper(),
            Activity.statecode == "Active",
        )
    ).scalar()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def recalculate_account(
    db: Session,
    account_id: str,
    *,
    write: bool = True,
) -> AccountRecalcResult | None:
    """Recompute the three derived account fields and (optionally) persist.

    Parameters
    ----------
    db
        Active SQLAlchemy session.
    account_id
        ``account.accountid`` (UUID-as-string, any case).
    write
        When True (default) the result is persisted on the account row.
        Read-only callers (e.g. the GET endpoints) pass False so the page
        stays idempotent.
    """
    account = _load_account(db, account_id)
    if account is None:
        return None

    settings = load_account_status_settings(db)
    has_won, open_count, low_health_count = _load_deal_signals(
        db, account_id, at_risk_threshold=settings.at_risk_health_threshold
    )
    last_interaction = _load_last_interaction(db, account_id)

    account_type = derive_account_type(has_won_deal=has_won)
    status = derive_account_status(
        AccountStatusInputs(
            statecode=account.statecode,
            open_deals_count=open_count,
            open_deals_with_low_health=low_health_count,
            has_won_deal=has_won,
            last_interaction_date=last_interaction,
            today=datetime.now(timezone.utc).date(),
        ),
        settings,
    )

    persisted = False
    if write and _has_account_view_columns(db):
        account.lvo_accounttype = account_type
        account.lvo_accountstatus = status
        account.lvo_lastinteractiondate = last_interaction
        db.commit()
        persisted = True

    return AccountRecalcResult(
        account_id=str(account.accountid),
        account_type=account_type,
        status=status,
        last_interaction=last_interaction,
        persisted=persisted,
    )


def recalculate_async(account_id: str) -> None:
    """Background-task wrapper — opens its own session.

    Use with FastAPI BackgroundTasks: ``add_task(recalculate_async, account_id)``.
    Failures are logged but never re-raised so a transient blip cannot
    surface as a 500 to the originating request.
    """
    db = SessionLocal()
    try:
        try:
            recalculate_account(db, account_id, write=True)
        except Exception:
            logger.exception(
                "Failed to recalculate account %s", account_id
            )
            db.rollback()
    finally:
        db.close()


def recalculate_many(
    db: Session, account_ids: Iterable[str], *, write: bool = True
) -> list[AccountRecalcResult]:
    """Batch helper used by ``app/jobs/recalc_accounts.py``."""
    results: list[AccountRecalcResult] = []
    for aid in account_ids:
        result = recalculate_account(db, aid, write=write)
        if result is not None:
            results.append(result)
    return results


__all__ = [
    "AccountRecalcResult",
    "load_account_status_settings",
    "recalculate_account",
    "recalculate_async",
    "recalculate_many",
]

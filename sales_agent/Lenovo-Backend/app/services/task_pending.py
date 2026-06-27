"""Sprint 2 US 1.3 — Task Pending badge summary for the Home dashboard header."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal

from sqlalchemy import String, cast, func, select
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session

from app.models import NextAction
from app.services.what_changed import _load_seller_opportunities

ERR_MSG_0022 = "ERR_MSG_0022"

BadgeColor = Literal["red", "default"]


@dataclass(frozen=True)
class TaskPendingSummaryData:
    seller_id: str
    count: int
    overdue_count: int
    due_today_count: int
    has_overdue: bool
    badge_color: BadgeColor
    label: str
    last_updated_at: datetime
    source: Literal["d365"]


def _has_table(db: Session, table_name: str) -> bool:
    bind = db.get_bind()
    if bind is None:
        return False
    return inspect(bind).has_table(table_name)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def build_pending_label(count: int) -> str:
    noun = "task" if count == 1 else "tasks"
    return f"{count} {noun} pending"


def badge_color_for(has_overdue: bool) -> BadgeColor:
    return "red" if has_overdue else "default"


def build_task_pending_summary(
    db: Session,
    seller_id: str,
    *,
    today: date | None = None,
) -> TaskPendingSummaryData:
    """Count open ``lvo_nextaction`` rows on the seller's open opportunities."""
    today = today or date.today()
    seller = seller_id.strip()
    empty = TaskPendingSummaryData(
        seller_id=seller,
        count=0,
        overdue_count=0,
        due_today_count=0,
        has_overdue=False,
        badge_color="default",
        label=build_pending_label(0),
        last_updated_at=_utc_now(),
        source="d365",
    )

    if not _has_table(db, "lvo_nextaction"):
        return empty

    opp_map = _load_seller_opportunities(db, seller)
    if not opp_map:
        return empty

    opp_ids_upper = list(opp_map.keys())
    base = (
        func.upper(cast(NextAction.lvo_opportunityid, String)).in_(opp_ids_upper),
        NextAction.statecode == "Active",
        NextAction.lvo_status == "Open",
    )

    count = int(
        db.scalar(select(func.count()).select_from(NextAction).where(*base)) or 0
    )
    overdue_count = int(
        db.scalar(
            select(func.count())
            .select_from(NextAction)
            .where(
                *base,
                NextAction.lvo_duedate.is_not(None),
                NextAction.lvo_duedate < today,
            )
        )
        or 0
    )
    due_today_count = int(
        db.scalar(
            select(func.count())
            .select_from(NextAction)
            .where(*base, NextAction.lvo_duedate == today)
        )
        or 0
    )
    has_overdue = overdue_count > 0

    return TaskPendingSummaryData(
        seller_id=seller,
        count=count,
        overdue_count=overdue_count,
        due_today_count=due_today_count,
        has_overdue=has_overdue,
        badge_color=badge_color_for(has_overdue),
        label=build_pending_label(count),
        last_updated_at=_utc_now(),
        source="d365",
    )

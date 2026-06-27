"""Daily data-hygiene scan — entry point for cron.

Sprint 1A · US04 — Data Hygiene, Validation & Intelligent Alerts

Run:

    python -m app.jobs.scan_data_tasks                   # full scan
    python -m app.jobs.scan_data_tasks --limit 50        # only first 50 opps
    python -m app.jobs.scan_data_tasks --dry-run         # report what would be created

The scan is idempotent — re-running is safe at any time. Each detected
issue collides with the partial UNIQUE index on
``(entity_kind, entity_id, task_kind)`` and the service layer's
``create_task`` returns the existing row instead of duplicating.

Algorithm
---------
1. ``d365_client.list_active_opportunities()`` streams Closed=False opps
   from D365's ``GET /api/opportunities`` (paginated server-side).
2. For each opp:
     a. Run D1, D2, D3 detectors directly on the row.
     b. Call ``d365_client.fetch_opportunity_risks(opp.id)`` and run D4
        on the result.
     c. Persist each ``DataTaskCreateRequest`` via
        ``data_task_service.create_task``.
3. Catch per-opp errors so one bad opp can't kill the whole run; the
   final summary surfaces ``opportunities_with_errors`` so an operator
   knows whether to investigate.

Output
------
The CLI prints a one-line JSON summary on stdout — easy to pipe into
log aggregators / Slack:

    {"total_scanned": 312, "tasks_created": 47, ...}

The function ``run_scan`` is also called directly from the
``POST /api/data-tasks/scan`` endpoint (see ``app/api/data_tasks.py``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.clients.d365_client import (
    D365ClientError,
    OpportunityScanRow,
    fetch_opportunity_risks,
    list_active_opportunities,
)
from app.core.config import DATA_TASK_SCAN_PAGE_SIZE, DATA_TASK_STALE_DAYS
from app.schema.data_task import DataTaskCreateRequest, ScanRunResponse
from app.services import data_task_service
from app.services.data_task_detectors import (
    detect_past_close_date,
    detect_risk_flags,
    detect_stale_activity,
    detect_zero_or_missing_value,
)

logger = logging.getLogger(__name__)


def _run_deterministic_detectors(
    opp: OpportunityScanRow,
    *,
    today: date,
    stale_days: int,
    scan_run_at: datetime,
) -> list[DataTaskCreateRequest]:
    """Run D1/D2/D3 against a single opp. Returns the (possibly empty)
    list of ``DataTaskCreateRequest`` objects for that opp.

    Any detector that returns ``None`` is silently filtered out.
    """
    candidates = [
        detect_past_close_date(opp, today=today, scan_run_at=scan_run_at),
        detect_zero_or_missing_value(opp, scan_run_at=scan_run_at),
        detect_stale_activity(
            opp, today=today, stale_days=stale_days, scan_run_at=scan_run_at
        ),
    ]
    return [c for c in candidates if c is not None]


def _persist_or_count(
    db: Session,
    payload: DataTaskCreateRequest,
    *,
    dry_run: bool,
    summary: dict[str, int],
) -> None:
    """Either persist the task or just bump the appropriate dry-run counter.

    Updates ``summary`` in-place. Distinguishes:
      - tasks_created               — new row inserted
      - tasks_skipped_existing      — open task with same key already there
      - tasks_skipped_dismissed     — dismissed task with same key (AC #5 suppression)
    """
    if dry_run:
        # In dry-run we don't know without a query whether the task
        # would be new or existing. Cheapest correct behaviour: do a
        # read-only check.
        existing = (
            db.query(data_task_service.DataTask)
            .filter(
                data_task_service.DataTask.entity_kind == payload.entity_kind,
                data_task_service.DataTask.entity_id == payload.entity_id,
                data_task_service.DataTask.task_kind == payload.task_kind,
                data_task_service.DataTask.status.in_(("open", "dismissed")),
            )
            .one_or_none()
        )
        if existing is None:
            summary["tasks_created"] += 1
        elif existing.status == "dismissed":
            summary["tasks_skipped_dismissed"] += 1
        else:
            summary["tasks_skipped_existing"] += 1
        return

    task, was_existing = data_task_service.create_task(db, payload)
    if not was_existing:
        summary["tasks_created"] += 1
    elif task.status == "dismissed":
        summary["tasks_skipped_dismissed"] += 1
    else:
        summary["tasks_skipped_existing"] += 1


def run_scan(
    db: Session,
    *,
    limit: Optional[int] = None,
    dry_run: bool = False,
    stale_days: Optional[int] = None,
    page_size: Optional[int] = None,
) -> ScanRunResponse:
    """Execute one scan pass against the live D365 backend.

    Parameters
    ----------
    db
        SQLAlchemy session — the function commits per-task via the
        service layer (so a mid-scan crash leaves the partial set of
        tasks already persisted, which is the desired behaviour).
    limit
        Stop after scanning this many opps. ``None`` = full portfolio.
    dry_run
        If True, no rows are written; the summary still reports what
        WOULD have been created vs already-existing vs suppressed.
    stale_days
        Override the configured ``DATA_TASK_STALE_DAYS`` threshold.
    page_size
        Override the configured page size used when paginating D365.

    Returns
    -------
    ScanRunResponse
        Glanceable summary suitable for a daily Slack digest.
    """
    effective_stale_days = (
        stale_days if stale_days is not None else DATA_TASK_STALE_DAYS
    )
    effective_page_size = (
        page_size if page_size is not None else DATA_TASK_SCAN_PAGE_SIZE
    )
    scan_run_at = datetime.now(timezone.utc)
    today = scan_run_at.date()

    summary = {
        "total_scanned": 0,
        "tasks_created": 0,
        "tasks_skipped_existing": 0,
        "tasks_skipped_dismissed": 0,
        "opportunities_with_errors": 0,
    }

    opportunity_iter = list_active_opportunities(
        page_size=effective_page_size,
        max_records=limit,
    )

    for opp in opportunity_iter:
        summary["total_scanned"] += 1

        # ---- Deterministic detectors (D1/D2/D3) -----------------------
        try:
            for payload in _run_deterministic_detectors(
                opp,
                today=today,
                stale_days=effective_stale_days,
                scan_run_at=scan_run_at,
            ):
                _persist_or_count(db, payload, dry_run=dry_run, summary=summary)
        except Exception:  # noqa: BLE001  — scan must not abort on one bad opp
            logger.exception(
                "Deterministic detectors raised on opp %s — continuing",
                opp.opportunity_id,
            )
            summary["opportunities_with_errors"] += 1
            # Don't skip the risk fetch — even if a deterministic
            # detector raised, the risk-flag fan-out is independent.

        # ---- D4: fetch risks + materialise --------------------------------
        if opp.owner_id is None:
            # detect_risk_flags requires owner_id; with no assignee we
            # have nothing to do here. (Same orphan rule as D1/D2/D3.)
            continue
        try:
            risks = fetch_opportunity_risks(opp.opportunity_id)
        except D365ClientError:
            logger.warning(
                "fetch_opportunity_risks failed for %s — continuing",
                opp.opportunity_id,
            )
            summary["opportunities_with_errors"] += 1
            continue

        try:
            for payload in detect_risk_flags(
                opp.opportunity_id,
                risks,
                owner_id=opp.owner_id,
                scan_run_at=scan_run_at,
            ):
                _persist_or_count(db, payload, dry_run=dry_run, summary=summary)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Risk-flag detector raised on opp %s — continuing",
                opp.opportunity_id,
            )
            summary["opportunities_with_errors"] += 1

    return ScanRunResponse(
        total_scanned=summary["total_scanned"],
        tasks_created=summary["tasks_created"],
        tasks_skipped_existing=summary["tasks_skipped_existing"],
        tasks_skipped_dismissed=summary["tasks_skipped_dismissed"],
        opportunities_with_errors=summary["opportunities_with_errors"],
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.jobs.scan_data_tasks",
        description=(
            "Run the data-hygiene daily scan. Pulls active opportunities "
            "from the D365 Sales backend and creates tasks in tbl_data_task "
            "for any detected issues. Idempotent — safe to re-run."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many opps (default: full portfolio).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write tasks — report what would happen.",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=None,
        help=(
            "Override the configured stale-activity threshold "
            "(default: DATA_TASK_STALE_DAYS env var)."
        ),
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=None,
        help=(
            "Override the configured D365 list page size "
            "(default: DATA_TASK_SCAN_PAGE_SIZE env var)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns Unix exit code."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Lazy DB import keeps test-time imports cheap.
    from app.db.database import SessionLocal

    db = SessionLocal()
    try:
        result = run_scan(
            db,
            limit=args.limit,
            dry_run=args.dry_run,
            stale_days=args.stale_days,
            page_size=args.page_size,
        )
    finally:
        db.close()

    print(json.dumps(result.model_dump()))
    # Non-zero exit if ANY opp errored — gives the cron the option to
    # alert on a non-zero return without parsing JSON.
    return 1 if result.opportunities_with_errors > 0 else 0


if __name__ == "__main__":  # pragma: no cover — exercised by CLI
    sys.exit(main())

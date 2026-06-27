"""Account-KPI bucket snapshot job.

Run nightly to capture the day's four account-side bucket aggregates into
``lvo_accountsnapshot``. The accounts KPI summary endpoint
(``GET /api/accounts/kpi-summary``) reads those rows to emit
period-over-period ``TrendInfo`` payloads.

Usage
-----
    # Snapshot today
    python -m app.jobs.snapshot_account_kpis

    # Snapshot a specific date (idempotent — overwrites that date's row)
    python -m app.jobs.snapshot_account_kpis --date 2026-06-10

    # Backfill so /accounts/kpi-summary shows non-null trends out of the box.
    # Generates rows for today, today-7, today-30 and today-90 using
    # ``account.createdon`` (when present) and ``opportunity.lvo_createdat``
    # as the existed-by filters. ``active`` and ``at_risk`` buckets are
    # approximate during backfill — see the service module docstring.
    python -m app.jobs.snapshot_account_kpis --backfill

    # Backfill to custom dates (comma-separated YYYY-MM-DD)
    python -m app.jobs.snapshot_account_kpis --backfill-dates 2026-03-01,2026-04-01,2026-05-01

Exit code is 0 on success, 1 on any unhandled exception.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime

from app.database import SessionLocal
from app.services.account_kpi_snapshots import backfill, take_snapshot

logger = logging.getLogger("snapshot_account_kpis")


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_dates_csv(value: str) -> list[date]:
    return [_parse_date(part.strip()) for part in value.split(",") if part.strip()]


def run(
    *,
    snapshot_date: date | None = None,
    do_backfill: bool = False,
    backfill_dates: list[date] | None = None,
) -> int:
    """Run the job. Returns the number of snapshots written."""
    db = SessionLocal()
    try:
        if do_backfill or backfill_dates:
            results = backfill(db, dates=backfill_dates)
            for d, buckets in sorted(results.items()):
                logger.info(
                    "Account snapshot %s — %s",
                    d.isoformat(),
                    ", ".join(
                        f"{b}={agg.count}" for b, agg in buckets.items()
                    ),
                )
            return len(results)

        buckets = take_snapshot(db, snapshot_date=snapshot_date)
        target = snapshot_date or datetime.utcnow().date()
        logger.info(
            "Account snapshot %s — %s",
            target.isoformat(),
            ", ".join(f"{b}={agg.count}" for b, agg in buckets.items()),
        )
        return 1
    finally:
        db.close()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture account-KPI bucket aggregates "
            "(total / acv / active / at_risk) into lvo_accountsnapshot."
        ),
    )
    parser.add_argument(
        "--date",
        type=_parse_date,
        default=None,
        help="Snapshot a specific calendar date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help=(
            "Generate snapshots for today, today-7, today-30 and today-90 "
            "using account.createdon / opportunity.lvo_createdat as the "
            "existed-by filters."
        ),
    )
    parser.add_argument(
        "--backfill-dates",
        type=_parse_dates_csv,
        default=None,
        help=(
            "Override the default backfill window with a comma-separated "
            "list of YYYY-MM-DD dates."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.date and (args.backfill or args.backfill_dates):
        parser.error("Use either --date or --backfill / --backfill-dates, not both.")
    try:
        run(
            snapshot_date=args.date,
            do_backfill=args.backfill,
            backfill_dates=args.backfill_dates,
        )
    except Exception:
        logger.exception("Fatal error in snapshot_account_kpis")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

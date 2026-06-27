"""Batch deal-health recalculation job.

Run periodically (cron / scheduler / CI) to make sure every deal's
``lvo_dealhealthscore``, ``lvo_riskreason`` and ``lvo_dealrisk`` rows
are fresh, even for deals that haven't been touched recently. The async
trigger from individual write endpoints handles edited deals; this job
is for the long tail.

Usage
-----
    python -m app.jobs.recalc_health [--limit N] [--include-canceled] [--seller-id UUID] [--verbose]

Exit code is 0 on success, 1 on any unhandled exception.

Examples
--------
    # Refresh every Open deal:
    python -m app.jobs.recalc_health

    # Quick smoke-test against 5 deals:
    python -m app.jobs.recalc_health --limit 5 --verbose

    # Include Canceled deals (rare; useful for back-population):
    python -m app.jobs.recalc_health --include-canceled
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Iterable

from sqlalchemy import String, cast, func, select

from app.database import SessionLocal
from app.models import Opportunity
from app.services.deal_recalc import recalculate_deal_health

logger = logging.getLogger("recalc_health")


def _select_opportunity_ids(
    *,
    include_canceled: bool,
    limit: int | None,
    seller_id: str | None = None,
) -> list[str]:
    """Pull the list of opportunity ids to recalc from the DB.

    Defaults to non-Canceled deals so we don't waste work refreshing
    soft-deleted rows. Pass ``include_canceled=True`` only when explicitly
    backfilling.
    """
    db = SessionLocal()
    try:
        stmt = select(Opportunity.opportunityid)
        if not include_canceled:
            stmt = stmt.where(Opportunity.statecode != "Canceled")
        if seller_id and seller_id.strip():
            stmt = stmt.where(
                func.upper(cast(Opportunity.owninguser, String))
                == seller_id.strip().upper()
            )
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = db.execute(stmt).all()
        return [str(r[0]) for r in rows]
    finally:
        db.close()


def _recalc_one(opportunity_id: str) -> bool:
    """Recalculate one deal in its own session.

    Returns True on success, False on a recoverable failure (already logged).
    Raises on unrecoverable errors (caller decides whether to abort the run).
    """
    db = SessionLocal()
    try:
        result = recalculate_deal_health(db, opportunity_id, write=True)
        if result is None:
            logger.warning("Opportunity %s vanished mid-run", opportunity_id)
            return False
        logger.debug(
            "Recalc OK: %s score=%s band=%s risks=%d",
            opportunity_id,
            result.breakdown.score,
            result.breakdown.band,
            len(result.risks),
        )
        return True
    except Exception:
        logger.exception("Recalc FAILED for %s", opportunity_id)
        db.rollback()
        return False
    finally:
        db.close()


def run(
    *,
    limit: int | None = None,
    include_canceled: bool = False,
    seller_id: str | None = None,
) -> int:
    """Run the batch. Returns the number of successful recalculations.

    Each deal gets its own session so a single failure can't poison the
    whole run.
    """
    opportunity_ids: Iterable[str] = _select_opportunity_ids(
        include_canceled=include_canceled,
        limit=limit,
        seller_id=seller_id,
    )
    opportunity_ids = list(opportunity_ids)

    if not opportunity_ids:
        logger.info("No opportunities to recalc; nothing to do.")
        return 0

    logger.info("Recalculating health for %d opportunities …", len(opportunity_ids))
    started = time.monotonic()
    successes = 0
    for idx, oid in enumerate(opportunity_ids, start=1):
        if _recalc_one(oid):
            successes += 1
        if idx % 25 == 0:
            logger.info("Progress: %d / %d", idx, len(opportunity_ids))
    elapsed = time.monotonic() - started
    logger.info(
        "Done. %d/%d succeeded in %.2fs (%.2f deals/s).",
        successes,
        len(opportunity_ids),
        elapsed,
        (successes / elapsed) if elapsed > 0 else 0,
    )
    return successes


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recalculate deal-health and risks for every opportunity.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of deals processed (handy for smoke tests).",
    )
    parser.add_argument(
        "--include-canceled",
        action="store_true",
        help="Also process soft-deleted (Canceled) deals.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    parser.add_argument(
        "--seller-id",
        default=None,
        help="Only recalc opportunities owned by this seller UUID.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        run(
            limit=args.limit,
            include_canceled=args.include_canceled,
            seller_id=args.seller_id,
        )
    except Exception:
        logger.exception("Fatal error in recalc_health")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

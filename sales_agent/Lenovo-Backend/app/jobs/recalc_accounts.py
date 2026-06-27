"""Batch account recalculation job.

Companion to ``app/jobs/recalc_health.py``. Run periodically to make sure
every account's ``lvo_accounttype``, ``lvo_accountstatus`` and
``lvo_lastinteractiondate`` are fresh — even for accounts whose deals
haven't been touched recently.

Usage
-----
    python -m app.jobs.recalc_accounts [--limit N] [--verbose]

The script processes every account in its own session so a single failure
can't poison the rest of the run. Order of operations matters:

1. Run ``recalc_health`` first (the at-risk derivation depends on
   ``opportunity.lvo_dealhealthscore``).
2. Then run ``recalc_accounts``.

Examples
--------
    # Refresh every account:
    python -m app.jobs.recalc_accounts

    # Quick smoke-test against 5 accounts:
    python -m app.jobs.recalc_accounts --limit 5 --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Iterable

from sqlalchemy import select

from app.database import SessionLocal
from app.models import Account
from app.services.account_recalc import recalculate_account

logger = logging.getLogger("recalc_accounts")


def _select_account_ids(*, limit: int | None) -> list[str]:
    db = SessionLocal()
    try:
        stmt = select(Account.accountid)
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = db.execute(stmt).all()
        return [str(r[0]) for r in rows]
    finally:
        db.close()


def _recalc_one(account_id: str) -> bool:
    db = SessionLocal()
    try:
        result = recalculate_account(db, account_id, write=True)
        if result is None:
            logger.warning("Account %s vanished mid-run", account_id)
            return False
        logger.debug(
            "Recalc OK: %s type=%s status=%s lastInteraction=%s",
            account_id,
            result.account_type,
            result.status,
            result.last_interaction.isoformat() if result.last_interaction else "—",
        )
        return True
    except Exception:
        logger.exception("Recalc FAILED for %s", account_id)
        db.rollback()
        return False
    finally:
        db.close()


def run(*, limit: int | None = None) -> int:
    account_ids: Iterable[str] = _select_account_ids(limit=limit)
    account_ids = list(account_ids)

    if not account_ids:
        logger.info("No accounts to recalc; nothing to do.")
        return 0

    logger.info("Recalculating status for %d accounts …", len(account_ids))
    started = time.monotonic()
    successes = 0
    for idx, aid in enumerate(account_ids, start=1):
        if _recalc_one(aid):
            successes += 1
        if idx % 25 == 0:
            logger.info("Progress: %d / %d", idx, len(account_ids))
    elapsed = time.monotonic() - started
    logger.info(
        "Done. %d/%d succeeded in %.2fs (%.2f accounts/s).",
        successes,
        len(account_ids),
        elapsed,
        (successes / elapsed) if elapsed > 0 else 0,
    )
    return successes


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recalculate type / status / last-interaction for every account.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of accounts processed (handy for smoke tests).",
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
    try:
        run(limit=args.limit)
    except Exception:
        logger.exception("Fatal error in recalc_accounts")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

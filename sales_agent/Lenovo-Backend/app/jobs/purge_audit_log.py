"""Purge audit log rows past the configured retention window.

Run daily via cron / Azure Container Apps job:

    python -m app.jobs.purge_audit_log

Requires ``sql/2026_14_audit_compliance.sql`` (immutability trigger + config).
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.database import SessionLocal
from app.services.audit_log import get_audit_config, purge_expired_audit_rows

logger = logging.getLogger("purge_audit_log")


def run(*, retention_days: int | None = None) -> int:
    db = SessionLocal()
    try:
        cfg = get_audit_config(db)
        days = retention_days if retention_days is not None else cfg.retention_days
        deleted = purge_expired_audit_rows(db, retention_days=days)
        logger.info("Purged %d audit rows older than %d days", deleted, days)
        return deleted
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Purge expired audit log rows.")
    parser.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="Override DB config retention (default: lvo_audit_config.retention_days).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        run(retention_days=args.retention_days)
    except Exception:
        logger.exception("Audit purge failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

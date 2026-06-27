"""Runtime phone-column resolver for the ``contact`` table.

Why this exists
---------------
D365 contact tables disagree on the phone column name:

* Classic on-prem schema  → ``telephone1`` / ``mobilephone``
* Custom Lenovo dumps     → may strip those entirely
* Lenovo fallback (added by ``sql/2026_06_contact_extension.sql``)
                          → ``lvo_phone``

Mapping all three columns on the ``Contact`` ORM would crash any eager-load on
dumps where one of them is missing (the same trap we fell into with
``parentcustomerid``). Instead, this module:

1. Inspects ``information_schema.columns`` once per process.
2. Picks the first column that exists in this priority order:
   ``telephone1`` → ``mobilephone`` → ``lvo_phone``.
3. Uses raw SQL (``text``) for read/write, so SQLAlchemy never emits a SELECT
   against a column that does not exist.

Public API
----------
* ``get_phone_column(db)``               — cached lookup, returns ``str | None``
* ``bulk_read_phones(db, contact_ids)``  — ``{upper(id): phone}`` dict
* ``write_phone(db, contact_id, phone)`` — UPDATE in the resolved column
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Priority order used when picking which column to read/write. Earlier entries
# win — keeps the on-prem D365 schema first so an existing telephone1 value
# isn't shadowed by an empty lvo_phone fallback.
PHONE_COLUMN_PRIORITY: tuple[str, ...] = ("telephone1", "mobilephone", "lvo_phone")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_cached_column: str | None | object = ...  # sentinel: "not yet resolved"


def _resolve_phone_column(db: Session) -> str | None:
    cols = {c["name"] for c in inspect(db.bind).get_columns("contact")}
    for candidate in PHONE_COLUMN_PRIORITY:
        if candidate in cols:
            return candidate
    return None


def get_phone_column(db: Session) -> str | None:
    """Return the resolved phone column name (or ``None`` if no candidate exists).

    Cached for the lifetime of the process. Call ``reset_cache()`` from tests
    if you need to flip the resolver during a single test run.
    """
    global _cached_column
    if _cached_column is not ...:
        return _cached_column  # type: ignore[return-value]
    with _lock:
        if _cached_column is ...:
            resolved = _resolve_phone_column(db)
            _cached_column = resolved
            if resolved is None:
                logger.warning(
                    "No phone column found on contact table — phone reads/writes "
                    "will be silently skipped. Run "
                    "sql/2026_06_contact_extension.sql to add lvo_phone fallback."
                )
            else:
                logger.info("Resolved contact phone column: %s", resolved)
    return _cached_column  # type: ignore[return-value]


def reset_cache() -> None:
    """Test helper — forget the cached resolution."""
    global _cached_column
    with _lock:
        _cached_column = ...


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


def bulk_read_phones(
    db: Session, contact_ids: Iterable[str]
) -> dict[str, str | None]:
    """Return ``{upper(contact_id): phone_value}`` for the given contact IDs.

    IDs are upper-cased on the way in to match the project-wide UUID-as-string
    convention. Missing IDs simply do not appear in the result; the caller can
    treat that as ``None``.

    Inputs are coerced via ``str(...)`` first because Postgres uuid columns
    surface as ``uuid.UUID`` from the ORM despite a String declaration.
    """
    upper_ids = [str(cid).upper() for cid in contact_ids if cid]
    if not upper_ids:
        return {}
    column = get_phone_column(db)
    if column is None:
        return {cid: None for cid in upper_ids}

    sql = text(
        f"SELECT UPPER(contactid::TEXT) AS cid, {column} AS phone "
        "FROM contact "
        "WHERE UPPER(contactid::TEXT) = ANY(:ids)"
    )
    rows = db.execute(sql, {"ids": upper_ids}).all()
    out: dict[str, str | None] = {cid: None for cid in upper_ids}
    for row in rows:
        out[row.cid] = row.phone
    return out


def read_phone(db: Session, contact_id: str) -> str | None:
    """Convenience single-id read."""
    if not contact_id:
        return None
    return bulk_read_phones(db, [contact_id]).get(str(contact_id).upper())


def write_phone(db: Session, contact_id: str, phone: str | None) -> bool:
    """UPDATE the resolved phone column for one contact row.

    Returns ``True`` when a column was available and the UPDATE executed
    (regardless of whether it actually matched a row). Caller is responsible
    for committing the surrounding transaction.
    """
    column = get_phone_column(db)
    if column is None:
        return False
    sql = text(
        f"UPDATE contact SET {column} = :phone "
        "WHERE UPPER(contactid::TEXT) = :cid"
    )
    db.execute(sql, {"phone": phone, "cid": str(contact_id).upper()})
    return True


__all__ = [
    "PHONE_COLUMN_PRIORITY",
    "bulk_read_phones",
    "get_phone_column",
    "read_phone",
    "reset_cache",
    "write_phone",
]

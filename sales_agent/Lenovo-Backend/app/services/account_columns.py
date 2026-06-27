"""Runtime introspection for the ``account`` table.

The Customer-Information endpoint reads ~40 columns across 6 sections.
A vanilla D365 dump usually ships every standard column (``address1_*``,
``websiteurl``, ``telephone1`` and friends), but stripped or older dumps
may be missing some — and the Lenovo-custom ``lvo_*`` columns added by
``sql/2026_06_account_customer_info_schema.sql`` may not be migrated yet
on every environment.

Rather than fail the whole tab when a single column is absent, the router
asks this helper which columns exist and feeds the missing ones into
``sqlalchemy.orm.defer`` so the eager-load just skips them.

Usage
-----
::

    cols = get_account_columns(db)
    if "address1_line1" in cols:
        # safe to read account.address1_line1
        ...

The result is cached per ``Engine`` because the schema only changes when
a migration runs and we don't want every request to round-trip to
``information_schema.columns``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import FrozenSet

from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# Static lists — kept here so the router and the docs can both reference
# the canonical set of columns the Customer-Information tab depends on.
# ---------------------------------------------------------------------------

#: Standard D365 columns the endpoint reads. Should be present on every
#: vanilla dump; ``defer``-loaded for safety.
STANDARD_D365_COLUMNS: tuple[str, ...] = (
    "address1_line1",
    "address1_line2",
    "address1_city",
    "address1_stateorprovince",
    "address1_postalcode",
    "address1_country",
    "address2_line1",
    "address2_line2",
    "address2_city",
    "address2_stateorprovince",
    "address2_postalcode",
    "address2_country",
    "telephone1",
    "websiteurl",
    "tickersymbol",
    "paymenttermscode",
    "defaultpricelevelid",
    "territoryid",
    "createdby",
    "description",
)

#: Lenovo-custom columns added by sql/2026_06_account_customer_info_schema.sql.
LENOVO_CUSTOM_COLUMNS: tuple[str, ...] = (
    "lvo_subsegment",
    "lvo_gtmsegment",
    "lvo_sellerknownas",
    "lvo_legalnamelocal",
    "lvo_locallanguage",
    "lvo_alias",
    "lvo_taxvatnumber",
    "lvo_legalentity",
    "lvo_linkedinurl",
    "lvo_dealsignconfig",
    "lvo_salesterritory",
    "lvo_futureterritory",
    "lvo_salesorg",
    "lvo_territorymovereason",
    "lvo_geographicunit",
    "lvo_salesoffice",
)

#: Convenience: the union — every column the Customer-Information tab cares
#: about beyond the View-Account fields already on the ORM.
CUSTOMER_INFO_COLUMNS: tuple[str, ...] = (
    STANDARD_D365_COLUMNS + LENOVO_CUSTOM_COLUMNS
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_account_columns(db: Session) -> FrozenSet[str]:
    """Return the set of columns currently present on ``account``.

    Cached per ``Engine`` — the schema changes only when a migration runs
    so a dev-time reload is enough to invalidate the cache.
    """
    bind = db.bind
    if isinstance(bind, Engine):
        return _columns_for_engine(bind)
    # Connection (e.g. inside a transaction) — fall back to a direct call.
    return frozenset(c["name"] for c in inspect(bind).get_columns("account"))


def missing_account_columns(
    db: Session, candidates: tuple[str, ...] = CUSTOMER_INFO_COLUMNS
) -> tuple[str, ...]:
    """Return the subset of ``candidates`` that are NOT on the account table."""
    present = get_account_columns(db)
    return tuple(c for c in candidates if c not in present)


@lru_cache(maxsize=8)
def _columns_for_engine(engine: Engine) -> FrozenSet[str]:
    return frozenset(c["name"] for c in inspect(engine).get_columns("account"))

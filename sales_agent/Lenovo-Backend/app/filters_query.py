"""Shared filter helpers for the opportunity grid AND the account grid.

`/api/opportunities/kpi-summary`, `/api/opportunities`, and `/api/accounts`
all accept similar filter sets. Centralising the WHERE-clause builders here
keeps every list endpoint consistent and easy to extend.
"""

from collections.abc import Sequence

from sqlalchemy import ColumnElement, Select, String, cast, exists, func, or_, select
from sqlalchemy.sql import Subquery

from app.models import Account, Opportunity, Quote, QuoteItem
from app.normalizers import slugify

# Soft-deleted deals (DELETE /api/opportunities/{id} sets this) must never
# appear on the grid, KPI cards, or any filter dropdown — but the rows are
# kept in place for audit + account-rollup history.
CANCELED_STATECODE = "Canceled"


def apply_opportunity_filters(
    stmt: Select,
    *,
    search: str | None = None,
    regions: Sequence[str] | None = None,
    industries: Sequence[str] | None = None,
    stages: Sequence[str] | None = None,
    products: Sequence[str] | None = None,
    owner_id: str | None = None,
    account_already_joined: bool = False,
    include_canceled: bool = False,
) -> Select:
    """Attach all opportunity-grid filters to `stmt`.

    `stmt` must already select from / join `Opportunity`.

    Parameters
    ----------
    account_already_joined
        Set ``True`` when the caller has already joined `account` (e.g. to
        pull account columns into the SELECT). Stops the helper adding a
        second join, which Postgres rejects with `DuplicateAlias`.

    Filter semantics
    ----------------
    - ``search``    — ILIKE match against `opportunity.name` or `account.name`
                      (the helper adds the account JOIN automatically when needed).
    - ``regions``   — matches `opportunity.lvo_businessgroup` OR
                      `opportunity.lvo_country` (case-insensitive).
    - ``industries``— `account.industrycode` IN (...).
    - ``stages``    — `opportunity.stagename` IN (...).
    - ``products``  — EXISTS over quote + lvo_quoteitem on
                      `lvo_productseries` (case-insensitive).
    - ``include_canceled`` — by default, soft-deleted deals
                      (``statecode='Canceled'``) are excluded from the
                      result. Set this True only when you specifically
                      want to read them back (e.g. a future "trash" view).
    """
    conditions: list[ColumnElement[bool]] = []

    if not include_canceled:
        conditions.append(
            func.coalesce(Opportunity.statecode, "") != CANCELED_STATECODE
        )

    needs_account_join = (bool(search) or bool(industries)) and not account_already_joined
    if needs_account_join:
        # account.accountid is UUID in the DDL while opportunity.accountid is
        # TEXT — Postgres won't compare them without an explicit cast.
        stmt = stmt.join(
            Account,
            cast(Account.accountid, String) == Opportunity.accountid,
            isouter=True,
        )

    if search:
        like = f"%{search.lower()}%"
        conditions.append(
            or_(
                func.lower(Opportunity.name).like(like),
                func.lower(Account.name).like(like),
            )
        )

    if regions:
        regions_lower = [r.lower() for r in regions if r]
        if regions_lower:
            conditions.append(
                or_(
                    func.lower(Opportunity.lvo_businessgroup).in_(regions_lower),
                    func.lower(Opportunity.lvo_country).in_(regions_lower),
                )
            )

    if industries:
        conditions.append(Account.industrycode.in_(industries))

    if stages:
        conditions.append(Opportunity.stagename.in_(stages))

    if products:
        # /api/filters/products returns slug ids ("thinkpad", "daas-managed-...")
        # while the column stores the original label ("ThinkPad"). Build the
        # equivalent slug in SQL so that whatever the dropdown sends (slug or
        # raw label) is matched after normalisation.
        product_slugs = [s for s in (slugify(p) for p in products if p) if s]
        if product_slugs:
            product_series_slug = func.btrim(
                func.regexp_replace(
                    func.lower(QuoteItem.lvo_productseries),
                    r"[^a-z0-9]",
                    "-",
                    "g",
                ),
                "-",
            )
            product_exists = (
                select(Quote.quoteid)
                .join(
                    QuoteItem,
                    # quote.quoteid is UUID; lvo_quoteitem.lvo_quoteid is TEXT.
                    # Cast Quote.quoteid -> text or Postgres raises
                    # 'operator does not exist: text = uuid'. Use UPPER on
                    # both sides because the sample data stores quoteids
                    # in uppercase while uuid::text emits lowercase.
                    func.upper(QuoteItem.lvo_quoteid)
                    == func.upper(cast(Quote.quoteid, String)),
                )
                .where(
                    # quote.opportunityid is TEXT; opportunity.opportunityid is UUID.
                    # Same lowercase-vs-uppercase trap as the quoteid join.
                    func.upper(Quote.opportunityid)
                    == func.upper(cast(Opportunity.opportunityid, String)),
                    product_series_slug.in_(product_slugs),
                )
                .correlate(Opportunity)
            )
            conditions.append(exists(product_exists))

    if owner_id:
        conditions.append(
            func.upper(Opportunity.owninguser) == owner_id.upper()
        )

    if conditions:
        stmt = stmt.where(*conditions)

    return stmt


# ============================================================================
# Account-side filters — used by the View Account grid
# ============================================================================

# When the user doesn't pick an explicit status, hide soft-deactivated rows
# the same way the deals grid hides Canceled deals.
DEFAULT_ACCOUNT_STATUSES = ("Active", "At-Risk")


def apply_account_filters(
    stmt: Select,
    *,
    search: str | None = None,
    account_types: Sequence[str] | None = None,
    account_statuses: Sequence[str] | None = None,
    segments: Sequence[str] | None = None,
    regions: Sequence[str] | None = None,
    industries: Sequence[str] | None = None,
    owner_id: str | None = None,
    value_min: float | None = None,
    value_max: float | None = None,
    total_account_value_col: ColumnElement | None = None,
    bucket: str | None = None,
) -> Select:
    """Attach the View-Account grid filters to ``stmt``.

    ``stmt`` must already select from / join ``Account``.

    Filter semantics
    ----------------
    - ``search``           — ILIKE on ``account.name`` or ``account.accountnumber``.
    - ``account_types``    — ``account.lvo_accounttype`` IN (Prospect | Customer).
    - ``account_statuses`` — ``account.lvo_accountstatus`` IN (...). Defaults
                             to (Active, At-Risk) when the caller passes None
                             so soft-deactivated rows don't appear unless
                             explicitly requested. ``bucket='total'``
                             additionally bypasses this default so the Total
                             card's drill-down can include Inactive rows.
    - ``segments``         — ``account.lvo_segment`` IN (SMB | Mid-Market | …).
    - ``regions``          — matches ``account.lvo_businessgroupid``,
                             ``account.lvo_countryid`` OR ``account.lvo_territory``
                             (case-insensitive).
    - ``industries``       — ``account.industrycode`` IN (...).
    - ``owner_id``         — ``account.owninguser`` (used by the future
                             "only my accounts" mode; the current dev build
                             always passes None per the locked-down decision).
    - ``value_min`` / ``value_max`` — applied against ``total_account_value_col``
                             when provided. The list endpoint computes that
                             column via a correlated SUM subquery.
    - ``bucket``           — drill-down from a card on the Accounts KPI strip.
                             One of ``total`` / ``acv`` / ``active`` /
                             ``at_risk``. Each maps to the same predicate
                             ``/api/accounts/kpi-summary`` uses, so clicking a
                             card and reading the resulting list is internally
                             consistent (counts in the strip line up with the
                             grid below). ``acv`` requires
                             ``total_account_value_col`` to be passed in;
                             without it the bucket is a no-op.
    """
    conditions: list[ColumnElement[bool]] = []

    if search:
        like = f"%{search.lower()}%"
        conditions.append(
            or_(
                func.lower(Account.name).like(like),
                func.lower(Account.accountnumber).like(like),
            )
        )

    if account_types:
        conditions.append(Account.lvo_accounttype.in_(list(account_types)))

    # Status filter — bucket=total deliberately bypasses the default so
    # Inactive accounts appear in the Total drill-down.
    if account_statuses:
        conditions.append(Account.lvo_accountstatus.in_(list(account_statuses)))
    elif bucket == "total":
        pass
    else:
        conditions.append(
            Account.lvo_accountstatus.in_(list(DEFAULT_ACCOUNT_STATUSES))
        )

    # Bucket-specific predicates layered on top of the user filters.
    if bucket == "active":
        conditions.append(Account.lvo_accountstatus == "Active")
    elif bucket == "at_risk":
        conditions.append(Account.lvo_accountstatus == "At-Risk")
    elif bucket == "acv" and total_account_value_col is not None:
        # "Accounts contributing to ACV" = those with at least one
        # non-canceled opportunity. ``total_account_value_col`` already
        # excludes Canceled deals via ``total_account_value_subquery``.
        conditions.append(total_account_value_col > 0)

    if segments:
        conditions.append(Account.lvo_segment.in_(list(segments)))

    if regions:
        regions_lower = [r.lower() for r in regions if r]
        if regions_lower:
            conditions.append(
                or_(
                    func.lower(Account.lvo_businessgroupid).in_(regions_lower),
                    func.lower(Account.lvo_countryid).in_(regions_lower),
                    func.lower(Account.lvo_territory).in_(regions_lower),
                )
            )

    if industries:
        conditions.append(Account.industrycode.in_(list(industries)))

    if owner_id:
        conditions.append(
            func.upper(Account.owninguser) == owner_id.upper()
        )

    if (value_min is not None or value_max is not None) and total_account_value_col is not None:
        if value_min is not None:
            conditions.append(total_account_value_col >= float(value_min))
        if value_max is not None:
            conditions.append(total_account_value_col <= float(value_max))

    if conditions:
        stmt = stmt.where(*conditions)

    return stmt


def total_account_value_subquery() -> Subquery:
    """Correlated subquery — sum of estimatedvalue per account (Canceled excluded).

    Returned shape: a SELECT statement with two columns the caller joins on:
        ``account_key``  — UPPER(opportunity.accountid)
        ``total_value``  — SUM(estimatedvalue)
        ``open_count``   — number of Open deals
        ``last_won_date``— MAX(estimatedclosedate) for Won deals
                           (used to flip Prospect → Customer in the recalc
                           service). Exposed here too so the listing endpoint
                           can avoid a second roundtrip.
    """
    return (
        select(
            func.upper(Opportunity.accountid).label("account_key"),
            func.coalesce(func.sum(Opportunity.estimatedvalue), 0).label("total_value"),
            func.count().filter(Opportunity.statecode == "Open").label("open_count"),
            func.max(Opportunity.estimatedclosedate)
                .filter(
                    Opportunity.statecode.in_(("Won", "Closed Won"))
                    | Opportunity.stagename.in_(("Closed Won",))
                )
                .label("last_won_date"),
        )
        .where(
            Opportunity.accountid.is_not(None),
            func.coalesce(Opportunity.statecode, "") != CANCELED_STATECODE,
        )
        .group_by(func.upper(Opportunity.accountid))
        .subquery()
    )

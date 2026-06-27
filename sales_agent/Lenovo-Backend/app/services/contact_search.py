"""Fuzzy contact lookup by name (+ optional account name) for AI chat flows.

Used when a seller says things like "schedule a call with John at Deutsche
Bank" — the bot only has a first name (and maybe an account hint), not an
email or contact UUID.
"""

from __future__ import annotations

from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session

from app.models import Account, AccountContact, Contact, Opportunity, OpportunityContact
from app.schemas import ContactSearchItem
from app.services.contact_phone import bulk_read_phones
from app.services.seller_contacts import contact_display_name

DEFAULT_LIMIT = 25
MAX_LIMIT = 50


def _has_table(db: Session, table_name: str) -> bool:
    return inspect(db.bind).has_table(table_name)


def _normalise_term(value: str) -> str:
    return value.strip().lower()


def name_match_rank(term: str, contact: Contact) -> tuple[int, str]:
    """Lower rank = better match. Tie-break alphabetically on display name."""
    first = (contact.firstname or "").strip().lower()
    full = (contact.fullname or "").strip().lower()
    display = contact_display_name(contact) or ""

    if first == term:
        return (0, display.lower())
    if first.startswith(term):
        return (1, display.lower())
    if term in first:
        return (2, display.lower())
    if full == term:
        return (3, display.lower())
    if full.startswith(term):
        return (4, display.lower())
    if term in full:
        return (5, display.lower())
    return (6, display.lower())


def _contact_ids_for_account_name(db: Session, account_term: str) -> set[str]:
    """Contact IDs linked to an account whose name contains ``account_term``."""
    pattern = f"%{account_term}%"
    ids: set[str] = set()

    if _has_table(db, "lvo_accountcontact"):
        ac_rows = db.execute(
            select(AccountContact.lvo_contactid)
            .join(
                Account,
                func.upper(cast(Account.accountid, String))
                == func.upper(AccountContact.lvo_accountid),
            )
            .where(
                AccountContact.statecode == "Active",
                Account.name.ilike(pattern),
            )
        ).scalars().all()
        ids.update(str(cid).upper() for cid in ac_rows if cid)

    if _has_table(db, "lvo_opportunitycontact"):
        oc_rows = db.execute(
            select(OpportunityContact.lvo_contactid)
            .join(
                Opportunity,
                func.upper(cast(Opportunity.opportunityid, String))
                == func.upper(OpportunityContact.lvo_opportunityid),
            )
            .join(
                Account,
                func.upper(cast(Account.accountid, String))
                == func.upper(cast(Opportunity.accountid, String)),
                isouter=True,
            )
            .where(
                OpportunityContact.statecode == "Active",
                Account.name.ilike(pattern),
            )
        ).scalars().all()
        ids.update(str(cid).upper() for cid in oc_rows if cid)

    return ids


def _load_primary_accounts(
    db: Session, contact_ids: list[str]
) -> dict[str, tuple[str | None, str | None]]:
    """Best-effort account (id, name) per contact — primary link preferred."""
    if not contact_ids:
        return {}

    upper_ids = [cid.upper() for cid in contact_ids]
    result: dict[str, tuple[str | None, str | None]] = {}

    if _has_table(db, "lvo_accountcontact"):
        ac_rows = db.execute(
            select(
                AccountContact.lvo_contactid,
                AccountContact.lvo_isprimary,
                Account.accountid,
                Account.name,
            )
            .join(
                Account,
                func.upper(cast(Account.accountid, String))
                == func.upper(AccountContact.lvo_accountid),
            )
            .where(
                func.upper(AccountContact.lvo_contactid).in_(upper_ids),
                AccountContact.statecode == "Active",
            )
            .order_by(AccountContact.lvo_isprimary.desc(), Account.name)
        ).all()
        for row in ac_rows:
            cid_key = str(row.lvo_contactid).upper()
            if cid_key not in result or row.lvo_isprimary:
                result[cid_key] = (
                    str(row.accountid) if row.accountid else None,
                    row.name,
                )

    missing = [cid for cid in upper_ids if cid not in result]
    if missing and _has_table(db, "lvo_opportunitycontact"):
        oc_rows = db.execute(
            select(
                OpportunityContact.lvo_contactid,
                Account.accountid,
                Account.name,
            )
            .join(
                Opportunity,
                func.upper(cast(Opportunity.opportunityid, String))
                == func.upper(OpportunityContact.lvo_opportunityid),
            )
            .join(
                Account,
                func.upper(cast(Account.accountid, String))
                == func.upper(cast(Opportunity.accountid, String)),
                isouter=True,
            )
            .where(
                func.upper(OpportunityContact.lvo_contactid).in_(missing),
                OpportunityContact.statecode == "Active",
                or_(Opportunity.statecode.is_(None), Opportunity.statecode == "Open"),
            )
            .order_by(Account.name)
        ).all()
        for row in oc_rows:
            cid_key = str(row.lvo_contactid).upper()
            if cid_key not in result and row.accountid:
                result[cid_key] = (str(row.accountid), row.name)

    return result


def search_contacts_by_name(
    db: Session,
    name: str,
    account: str | None = None,
    *,
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[ContactSearchItem], str, str | None]:
    """Return ranked contact matches for an AI-extracted name (+ optional account).

    Returns ``(items, normalised_name, normalised_account_or_none)``.
    """
    term = _normalise_term(name)
    account_term = _normalise_term(account) if account and account.strip() else None
    cap = min(max(limit, 1), MAX_LIMIT)

    pattern = f"%{term}%"
    stmt = select(Contact).where(
        or_(Contact.statecode.is_(None), Contact.statecode == "Active"),
        or_(
            func.lower(Contact.firstname).like(pattern),
            func.lower(Contact.fullname).like(pattern),
        ),
    )

    if account_term:
        linked_ids = _contact_ids_for_account_name(db, account_term)
        if not linked_ids:
            return [], term, account_term
        stmt = stmt.where(
            func.upper(cast(Contact.contactid, String)).in_(linked_ids)
        )

    contacts = list(db.execute(stmt).scalars().all())
    contacts.sort(key=lambda c: name_match_rank(term, c))

    contacts = contacts[:cap]
    if not contacts:
        return [], term, account_term

    contact_ids = [str(c.contactid) for c in contacts]
    phones = bulk_read_phones(db, contact_ids)
    accounts = _load_primary_accounts(db, contact_ids)

    items: list[ContactSearchItem] = []
    for contact in contacts:
        cid_upper = str(contact.contactid).upper()
        account_id, account_name = accounts.get(cid_upper, (None, None))
        items.append(
            ContactSearchItem(
                contact_id=str(contact.contactid),
                name=contact_display_name(contact),
                first_name=contact.firstname,
                last_name=contact.lastname,
                email=contact.emailaddress1,
                job_title=contact.jobtitle,
                phone=phones.get(cid_upper),
                account_id=account_id,
                account_name=account_name,
            )
        )

    return items, term, account_term

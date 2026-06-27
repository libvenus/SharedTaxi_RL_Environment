"""Portfolio contact roster for a seller (owninguser).

Returns de-duplicated contacts linked to the seller's open opportunities
via ``lvo_opportunitycontact``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session

from app.models import Account, Contact, Opportunity, OpportunityContact
from app.schemas import SellerContactItem
from app.services.contact_phone import bulk_read_phones

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100


@dataclass
class _ContactLinkRow:
    contact_id: str
    contact: Contact | None
    link: OpportunityContact
    opportunity_id: str
    opportunity_name: str | None
    account_id: str | None
    account_name: str | None


def _has_table(db: Session, table_name: str) -> bool:
    return inspect(db.bind).has_table(table_name)


def contact_display_name(contact: Contact | None) -> str | None:
    if contact is None:
        return None
    if contact.fullname and contact.fullname.strip():
        return contact.fullname.strip()
    parts = [p for p in (contact.firstname, contact.lastname) if p and str(p).strip()]
    return " ".join(parts) if parts else None


def pick_primary_link(rows: list[_ContactLinkRow]) -> _ContactLinkRow:
    """Choose the representative link when a contact appears on multiple deals."""

    def _sort_key(row: _ContactLinkRow) -> tuple:
        touch = row.link.lvo_lasttouchdate
        touch_ts = touch.timestamp() if touch is not None else 0.0
        name = contact_display_name(row.contact) or ""
        return (
            0 if row.link.lvo_isdecisionmaker else 1,
            -touch_ts,
            name.lower(),
        )

    return sorted(rows, key=_sort_key)[0]


def build_seller_contact_items(
    rows: list[_ContactLinkRow],
    phones: dict[str, str | None],
) -> list[SellerContactItem]:
    """De-dupe link rows to one ``SellerContactItem`` per ``contact_id``."""
    grouped: dict[str, list[_ContactLinkRow]] = {}
    for row in rows:
        grouped.setdefault(row.contact_id.upper(), []).append(row)

    items: list[SellerContactItem] = []
    for group in grouped.values():
        primary = pick_primary_link(group)
        contact = primary.contact
        cid_upper = primary.contact_id.upper()
        touch_dates = [
            r.link.lvo_lasttouchdate
            for r in group
            if r.link.lvo_lasttouchdate is not None
        ]
        last_touch = max(touch_dates) if touch_dates else primary.link.lvo_lasttouchdate

        items.append(
            SellerContactItem(
                contact_id=primary.contact_id,
                name=contact_display_name(contact),
                first_name=contact.firstname if contact else None,
                last_name=contact.lastname if contact else None,
                email=contact.emailaddress1 if contact else None,
                job_title=contact.jobtitle if contact else None,
                phone=phones.get(cid_upper),
                account_id=primary.account_id,
                account_name=primary.account_name,
                opportunity_id=primary.opportunity_id,
                opportunity_name=primary.opportunity_name,
                role=primary.link.lvo_role,
                is_decision_maker=bool(primary.link.lvo_isdecisionmaker),
                linked_opportunity_count=len(group),
                last_touch_date=last_touch,
            )
        )

    items.sort(key=lambda i: (i.name or "").lower())
    return items


def _load_seller_contact_rows(db: Session, seller_id: str) -> list[_ContactLinkRow]:
    if not _has_table(db, "lvo_opportunitycontact"):
        return []

    stmt = (
        select(OpportunityContact, Contact, Opportunity, Account)
        .join(
            Opportunity,
            func.upper(cast(Opportunity.opportunityid, String))
            == func.upper(OpportunityContact.lvo_opportunityid),
        )
        .join(
            Contact,
            func.upper(cast(Contact.contactid, String))
            == func.upper(OpportunityContact.lvo_contactid),
            isouter=True,
        )
        .join(
            Account,
            func.upper(cast(Account.accountid, String))
            == func.upper(cast(Opportunity.accountid, String)),
            isouter=True,
        )
        .where(
            func.upper(cast(Opportunity.owninguser, String)) == seller_id.strip().upper(),
            or_(Opportunity.statecode.is_(None), Opportunity.statecode == "Open"),
            OpportunityContact.statecode == "Active",
            or_(Contact.statecode.is_(None), Contact.statecode == "Active"),
        )
    )

    out: list[_ContactLinkRow] = []
    for link, contact, opp, account in db.execute(stmt).all():
        if not link.lvo_contactid:
            continue
        out.append(
            _ContactLinkRow(
                contact_id=str(link.lvo_contactid),
                contact=contact,
                link=link,
                opportunity_id=str(opp.opportunityid),
                opportunity_name=opp.name,
                account_id=str(opp.accountid) if opp.accountid else None,
                account_name=account.name if account else None,
            )
        )
    return out


def list_seller_contacts(
    db: Session,
    seller_id: str,
    *,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> tuple[list[SellerContactItem], int, int]:
    """Return paginated de-duplicated contacts for a seller's open portfolio."""
    rows = _load_seller_contact_rows(db, seller_id)
    if not rows:
        return [], 0, 0

    contact_ids = list({r.contact_id for r in rows})
    phones = bulk_read_phones(db, contact_ids)
    items = build_seller_contact_items(rows, phones)

    total = len(items)
    total_pages = math.ceil(total / page_size) if page_size else 0
    start = (page - 1) * page_size
    end = start + page_size
    return items[start:end], total, total_pages

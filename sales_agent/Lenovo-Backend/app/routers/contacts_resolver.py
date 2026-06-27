"""Contact-by-email resolver — Sprint 1A · US02 (Consent & Recording).

Called by the Note-Taking Agent (Lenovo-AIBackend) once at the start of
each meeting to enrich attendees with CRM context. The bot then tags
each transcript utterance with the result locally — no per-utterance
round-trip back to D365.

Endpoints
---------
POST /api/contacts/resolve-by-emails
GET  /api/contacts/search

Algorithm
---------
1. Lower-case + de-dupe the incoming email list.
2. Lookup ``contact`` rows where ``LOWER(emailaddress1)`` matches AND
   ``statecode = 'Active'``.
3. For each matched contact, find the most senior opportunity-contact
   link:
     - prefer ``lvo_isdecisionmaker = true``
     - tie-break alphabetic by ``lvo_role``
     - exclude links to Cancelled/Lost deals (consistent with the
       meeting-resolver's "active deals only" rule)
4. Hydrate the account name in a single follow-up query.
5. Return one entry per supplied email — including unmatched emails
   (with all fields NULL) so the bot has complete coverage.
"""

from __future__ import annotations

import logging
from typing import Iterable

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy import String, cast, func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Account, Contact, Opportunity, OpportunityContact
from app.schemas import (
    ContactResolveRequest,
    ContactResolveResponse,
    ContactResolveResult,
    ContactSearchResponse,
)
from app.services.contact_search import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    search_contacts_by_name,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/contacts", tags=["contacts-resolver"])


# Cancelled / Lost deals are excluded from role-resolution (consistent
# with /api/meetings/resolve-opportunity).
_EXCLUDED_OPP_STATECODES = ("Canceled", "Cancelled", "Lost")


def _normalise_emails(emails: Iterable[str]) -> list[str]:
    """Lower-case, strip, de-dupe — preserving original order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in emails:
        if not raw:
            continue
        cleaned = raw.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def _pick_senior_role(links: list) -> str | None:
    """Pick the most-senior role from a list of opportunity-contact links.

    Rules (in order):
      1. ``lvo_isdecisionmaker = True`` wins
      2. Alphabetic by ``lvo_role`` for tie-break
      3. Returns None if all roles are blank
    """
    if not links:
        return None

    # Sort: decision-makers first, then alphabetic by role (None last).
    ranked = sorted(
        links,
        key=lambda link: (
            0 if getattr(link, "lvo_isdecisionmaker", False) else 1,
            (getattr(link, "lvo_role", None) or "\uffff").lower(),
        ),
    )
    return ranked[0].lvo_role if ranked else None


@router.post(
    "/resolve-by-emails",
    response_model=ContactResolveResponse,
    summary="Bulk-resolve attendee emails to CRM contact + account context",
    responses={
        422: {"description": "emails list is empty or all-blank"},
    },
)
def resolve_contacts_by_emails(
    payload: ContactResolveRequest = Body(...),
    db: Session = Depends(get_db),
) -> ContactResolveResponse:
    """Lookup CRM context for a batch of attendee emails."""
    emails = _normalise_emails(payload.emails)
    if not emails:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "EMPTY_EMAILS",
                "message": "emails contains no usable addresses.",
            },
        )

    # ---- 1. Pull every Active contact whose email matches ------------------
    contact_rows = db.execute(
        select(
            Contact.contactid,
            Contact.emailaddress1,
            Contact.firstname,
            Contact.lastname,
            Contact.fullname,
            Contact.jobtitle,
        ).where(
            func.lower(Contact.emailaddress1).in_(emails),
            Contact.statecode == "Active",
        )
    ).all()

    # Map normalised-email -> contact row for O(1) join with opp-contact links.
    by_email = {row.emailaddress1.lower(): row for row in contact_rows if row.emailaddress1}

    # ---- 2. For matched contacts, walk to opportunity-contact links --------
    contact_id_strs = [str(row.contactid) for row in contact_rows]
    role_links: dict[str, list] = {}    # contact_id_upper -> list of OpportunityContact
    contact_account: dict[str, tuple[str | None, str | None]] = {}  # contact_id_upper -> (account_id, account_name)

    if contact_id_strs:
        link_rows = db.execute(
            select(
                OpportunityContact.lvo_contactid,
                OpportunityContact.lvo_role,
                OpportunityContact.lvo_isdecisionmaker,
                Opportunity.accountid,
            )
            .join(
                Opportunity,
                func.upper(cast(Opportunity.opportunityid, String))
                == func.upper(OpportunityContact.lvo_opportunityid),
            )
            .where(
                func.upper(OpportunityContact.lvo_contactid).in_(
                    [cid.upper() for cid in contact_id_strs]
                ),
                OpportunityContact.statecode == "Active",
                (Opportunity.statecode.notin_(_EXCLUDED_OPP_STATECODES))
                | (Opportunity.statecode.is_(None)),
            )
        ).all()

        # Bucket links by contact_id (uppercased) for senior-role picking.
        for link in link_rows:
            cid_key = str(link.lvo_contactid).upper()
            role_links.setdefault(cid_key, []).append(link)

        # Best-effort capture of the contact's primary account from any link.
        # Keep the first non-NULL accountid we see — they usually agree.
        for link in link_rows:
            cid_key = str(link.lvo_contactid).upper()
            if cid_key not in contact_account and link.accountid:
                contact_account[cid_key] = (str(link.accountid), None)

        # ---- 3. Hydrate account names for the IDs we collected -------------
        account_ids = list({acct_id for acct_id, _ in contact_account.values() if acct_id})
        if account_ids:
            account_rows = db.execute(
                select(Account.accountid, Account.name).where(
                    func.upper(cast(Account.accountid, String)).in_(
                        [aid.upper() for aid in account_ids]
                    )
                )
            ).all()
            account_name_map = {
                str(row.accountid).upper(): row.name for row in account_rows
            }
            # Patch the (account_id, account_name) tuples with the names.
            for cid_key, (acct_id, _) in list(contact_account.items()):
                contact_account[cid_key] = (
                    acct_id,
                    account_name_map.get(str(acct_id).upper()),
                )

    # ---- 4. Build per-email response (one entry per SUPPLIED email) --------
    results: list[ContactResolveResult] = []
    for email in emails:
        contact_row = by_email.get(email)
        if contact_row is None:
            results.append(
                ContactResolveResult(
                    email=email,
                    contact_id=None,
                    name=None,
                    job_title=None,
                    account_id=None,
                    account_name=None,
                    role=None,
                )
            )
            continue

        cid_key = str(contact_row.contactid).upper()
        role = _pick_senior_role(role_links.get(cid_key, []))
        account_id, account_name = contact_account.get(cid_key, (None, None))

        # Prefer fullname; fall back to "first last".
        name = contact_row.fullname
        if not name:
            parts = [
                p for p in (contact_row.firstname, contact_row.lastname) if p
            ]
            name = " ".join(parts) if parts else None

        results.append(
            ContactResolveResult(
                email=email,
                contact_id=str(contact_row.contactid),
                name=name,
                job_title=contact_row.jobtitle,
                account_id=account_id,
                account_name=account_name,
                role=role,
            )
        )

    logger.info(
        "Resolved %d emails: %d matched, %d unknown",
        len(emails),
        sum(1 for r in results if r.contact_id is not None),
        sum(1 for r in results if r.contact_id is None),
    )
    return ContactResolveResponse(results=results)


@router.get(
    "/search",
    response_model=ContactSearchResponse,
    summary="Fuzzy contact lookup by name (+ optional account hint)",
    responses={
        422: {"description": "name is missing or blank"},
    },
)
def search_contacts(
    name: str | None = Query(
        default=None,
        description=(
            "Contact first name or partial name from the chat prompt "
            "(e.g. 'John'). Case-insensitive partial match against "
            "contact.firstname and contact.fullname."
        ),
    ),
    account: str | None = Query(
        default=None,
        description=(
            "Optional account name hint (e.g. 'Deutsche Bank'). "
            "Narrows results to contacts linked to a matching account "
            "via lvo_accountcontact or an open opportunity."
        ),
    ),
    limit: int = Query(
        DEFAULT_LIMIT,
        ge=1,
        le=MAX_LIMIT,
        description="Max candidates to return (default 25, max 50).",
    ),
    db: Session = Depends(get_db),
) -> ContactSearchResponse:
    """Resolve a spoken contact name to CRM rows for scheduling / task flows."""
    if not name or not name.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "MISSING_NAME",
                "message": "name is required.",
            },
        )

    items, normalised_name, normalised_account = search_contacts_by_name(
        db, name, account, limit=limit
    )
    logger.info(
        "Contact search name=%r account=%r → %d hit(s)",
        normalised_name,
        normalised_account,
        len(items),
    )
    return ContactSearchResponse(
        name=normalised_name,
        account=normalised_account,
        total=len(items),
        items=items,
    )

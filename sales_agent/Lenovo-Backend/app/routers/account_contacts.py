"""Account-level contact management endpoints.

Implements the **Manage Contacts Linked to an Account** user story on top of
the View-Account skeleton. Surface area:

* ``GET    /api/accounts/{id}/contacts``                                    — roster + delete-eligibility chips
* ``POST   /api/accounts/{id}/contacts``                                    — attach existing OR create-and-link
* ``PATCH  /api/accounts/{id}/contacts/{linkId}``                           — update link + underlying contact fields
* ``DELETE /api/accounts/{id}/contacts/{linkId}``                           — soft-delete, blocked when primary or deal-linked
* ``GET    /api/accounts/{id}/contacts/{linkId}/delete-eligibility``        — pre-flight for the FE confirmation dialog

Business rules enforced here (mapped to user-story codes):

* ``ERR_MSG_0008`` — primary contacts cannot be removed.
* ``ERR_MSG_0009`` — contacts referenced by an active opportunity cannot be removed.
* ``ERR_MSG_0013`` — invalid email/phone format on add/update (raised by Pydantic validators).
* ``SUCC_MSG_0007/8/9`` — success codes returned in response bodies for FE toasts.

Every C/U/D mutation appends a row to ``lvo_audit_log`` and enqueues an
account-status recalc, since changing contacts shifts the "last interaction"
signal that feeds into ``At-Risk`` derivation.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    Header,
    HTTPException,
    Path,
    status,
)
from sqlalchemy import String, cast, func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    Account,
    AccountContact,
    Contact,
    Opportunity,
    OpportunityContact,
)
from app.services.audit_log import write_audit_event
from app.schemas import (
    AccountContactCreateRequest,
    AccountContactDeleteEligibilityResponse,
    AccountContactDeleteResponse,
    AccountContactListResponse,
    AccountContactRef,
    AccountContactUpdateRequest,
)
from app.services.account_recalc import recalculate_async
from app.services.contact_phone import bulk_read_phones, write_phone
from app.services.contact_validation import (
    CONF_DELETE,
    SUCC_ADD,
    SUCC_DELETE,
    SUCC_UPDATE,
    DeleteEligibility,
    compute_delete_eligibility,
)

router = APIRouter(prefix="/api/accounts", tags=["account-contacts"])


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def _ensure_account(db: Session, account_id: str) -> Account:
    # NOTE: Postgres serialises native ``uuid`` columns to text in lowercase,
    # but the URL path can carry uppercase / mixed case (D365 dumps and most
    # FE links uppercase the hyphen-blocks). Comparing lowercase DB text to
    # raw user input would 404 every uppercase URL — so we normalise both
    # sides to UPPER before comparing. Same pattern used for OpportunityID
    # lookups elsewhere in the codebase.
    account = db.execute(
        select(Account).where(
            func.upper(cast(Account.accountid, String)) == account_id.upper()
        )
    ).scalar_one_or_none()
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account '{account_id}' not found.",
        )
    return account


def _ensure_contact(db: Session, contact_id: str) -> Contact:
    # Defensive: ORM .contactid can come back as uuid.UUID despite the
    # String declaration (psycopg2 returns the native uuid type).
    cid = str(contact_id)
    contact = db.execute(
        select(Contact).where(
            func.upper(cast(Contact.contactid, String)) == cid.upper()
        )
    ).scalar_one_or_none()
    if contact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contact '{cid}' not found.",
        )
    return contact


def _ensure_link(db: Session, account_id: str, link_id: str) -> AccountContact:
    link = db.get(AccountContact, link_id)
    if link is None or link.statecode != "Active":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account-contact link '{link_id}' not found.",
        )
    if str(link.lvo_accountid or "").upper() != account_id.upper():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Account-contact link '{link_id}' does not belong to "
                f"account '{account_id}'."
            ),
        )
    return link


# ---------------------------------------------------------------------------
# Active-deal lookup for ERR_MSG_0009
# ---------------------------------------------------------------------------


def _active_deal_ids_for_contacts(
    db: Session, contact_ids: list[str]
) -> dict[str, list[str]]:
    """Return ``{upper(contact_id): [opportunity_id, …]}``.

    "Active" = ``opportunity.statecode = 'Open'`` AND the link in
    ``lvo_opportunitycontact`` has ``statecode='Active'``. Soft-deleted
    deals (``Canceled``) and closed deals (``Won`` / ``Lost``) do NOT block
    contact removal.
    """
    if not contact_ids:
        return {}
    # Coerce defensively — ORM-loaded uuid columns surface as uuid.UUID.
    upper_ids = list({str(cid).upper() for cid in contact_ids if cid})
    rows = db.execute(
        select(
            func.upper(OpportunityContact.lvo_contactid).label("cid"),
            cast(Opportunity.opportunityid, String).label("oid"),
        )
        .join(
            Opportunity,
            func.upper(cast(Opportunity.opportunityid, String))
            == func.upper(OpportunityContact.lvo_opportunityid),
        )
        .where(
            func.upper(OpportunityContact.lvo_contactid).in_(upper_ids),
            OpportunityContact.statecode == "Active",
            Opportunity.statecode == "Open",
        )
    ).all()
    out: dict[str, list[str]] = {cid: [] for cid in upper_ids}
    for r in rows:
        out.setdefault(r.cid, []).append(r.oid)
    return out


def _eligibility_for(
    *, link: AccountContact, active_deal_ids: list[str]
) -> DeleteEligibility:
    return compute_delete_eligibility(
        is_primary=bool(link.lvo_isprimary),
        active_deal_ids=active_deal_ids,
    )


# ---------------------------------------------------------------------------
# Ref builder
# ---------------------------------------------------------------------------


def _to_ref(
    link: AccountContact,
    contact: Contact | None,
    *,
    phone: str | None = None,
    eligibility: DeleteEligibility | None = None,
) -> AccountContactRef:
    name = None
    first = last = None
    if contact is not None:
        first = contact.firstname
        last = contact.lastname
        name = contact.fullname or (
            f"{(first or '').strip()} {(last or '').strip()}".strip()
            or None
        )
    eligibility = eligibility or DeleteEligibility.ok()
    return AccountContactRef(
        id=link.lvo_accountcontactid,
        account_id=link.lvo_accountid,
        contact_id=link.lvo_contactid,
        name=name,
        first_name=first,
        last_name=last,
        role=link.lvo_role,
        is_primary=bool(link.lvo_isprimary),
        last_touch_date=link.lvo_lasttouchdate,
        job_title=contact.jobtitle if contact else None,
        email=contact.emailaddress1 if contact else None,
        phone=phone,
        can_delete=eligibility.can_delete,
        delete_restriction_code=eligibility.code,
        delete_restriction_message=eligibility.message,
    )


def _demote_existing_primary(
    db: Session, account_id: str, exclude_link_id: str | None = None
) -> None:
    """Set lvo_isprimary=False on every other Active link on the account."""
    rows = (
        db.execute(
            select(AccountContact).where(
                func.upper(AccountContact.lvo_accountid) == account_id.upper(),
                AccountContact.statecode == "Active",
                AccountContact.lvo_isprimary.is_(True),
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        if exclude_link_id and row.lvo_accountcontactid == exclude_link_id:
            continue
        row.lvo_isprimary = False
        row.lvo_updatedat = datetime.now(timezone.utc)


def _write_audit(
    db: Session,
    *,
    entity_id: str,
    account_id: str,
    action: str,
    changed_by: str | None,
    diff: dict | None,
) -> None:
    write_audit_event(
        db,
        entity_type="account_contact",
        entity_id=entity_id,
        action=action,
        category="crm_writeback",
        actor_type="seller",
        changed_by=changed_by,
        diff={"accountId": account_id, **(diff or {})},
    )


# ---------------------------------------------------------------------------
# GET /api/accounts/{id}/contacts
# ---------------------------------------------------------------------------


@router.get(
    "/{account_id}/contacts",
    response_model=AccountContactListResponse,
    summary="Roster of contacts attached to the account",
)
def list_account_contacts(
    account_id: str = Path(...),
    db: Session = Depends(get_db),
) -> AccountContactListResponse:
    _ensure_account(db, account_id)

    rows = db.execute(
        select(AccountContact, Contact)
        .join(
            Contact,
            func.upper(cast(Contact.contactid, String))
            == func.upper(AccountContact.lvo_contactid),
            isouter=True,
        )
        .where(
            func.upper(AccountContact.lvo_accountid) == account_id.upper(),
            AccountContact.statecode == "Active",
        )
        .order_by(
            AccountContact.lvo_isprimary.desc(),
            AccountContact.lvo_role,
            AccountContact.lvo_createdat,
        )
    ).all()

    # Batch-load phones + active-deal references so the per-row build below
    # doesn't trigger N+1 queries.
    contact_ids = [
        str(r[0].lvo_contactid) for r in rows if r[0].lvo_contactid
    ]
    phones = bulk_read_phones(db, contact_ids)
    deal_map = _active_deal_ids_for_contacts(db, contact_ids)

    primary: AccountContactRef | None = None
    others: list[AccountContactRef] = []
    for link, contact in rows:
        upper_cid = str(link.lvo_contactid or "").upper()
        eligibility = _eligibility_for(
            link=link, active_deal_ids=deal_map.get(upper_cid, [])
        )
        ref = _to_ref(
            link, contact, phone=phones.get(upper_cid), eligibility=eligibility
        )
        if link.lvo_isprimary and primary is None:
            primary = ref
        else:
            others.append(ref)

    total = (1 if primary else 0) + len(others)
    return AccountContactListResponse(
        account_id=account_id,
        primary=primary,
        others=others,
        total=total,
    )


# ---------------------------------------------------------------------------
# POST /api/accounts/{id}/contacts — attach existing OR create-and-link
# ---------------------------------------------------------------------------


@router.post(
    "/{account_id}/contacts",
    response_model=AccountContactRef,
    status_code=status.HTTP_201_CREATED,
    summary="Add a contact to the account (attach existing or create-and-link)",
    responses={
        404: {"description": "Account or contact not found"},
        409: {"description": "Contact is already attached to this account"},
        422: {"description": "Validation error (ERR_MSG_0013)"},
    },
)
def add_account_contact(
    background_tasks: BackgroundTasks,
    account_id: str = Path(...),
    body: AccountContactCreateRequest = Body(...),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> AccountContactRef:
    _ensure_account(db, account_id)

    # ---- Decide flow ------------------------------------------------------
    has_create_fields = bool((body.first_name or "").strip()) or bool(
        (body.last_name or "").strip()
    )
    if body.contact_id and has_create_fields:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "ERR_MSG_0013",
                "message": (
                    "Provide either contactId (attach) or firstName + lastName "
                    "(create), not both."
                ),
            },
        )
    if not body.contact_id and not has_create_fields:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "ERR_MSG_0013",
                "message": (
                    "Provide either contactId (to attach an existing contact) "
                    "or firstName + lastName (to create a new contact)."
                ),
            },
        )
    if has_create_fields and (
        not (body.first_name or "").strip() or not (body.last_name or "").strip()
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "ERR_MSG_0013",
                "message": "Both firstName and lastName are required.",
            },
        )

    now = datetime.now(timezone.utc)

    # ---- Resolve / create the contact row --------------------------------
    if body.contact_id:
        contact = _ensure_contact(db, body.contact_id)
        # Postgres uuid columns come back as uuid.UUID via psycopg2 even when
        # the ORM declares String — coerce defensively so .upper() works below.
        contact_id = str(contact.contactid)
        contact_created = False
    else:
        contact_id = str(uuid.uuid4())
        first = (body.first_name or "").strip()
        last = (body.last_name or "").strip()
        contact = Contact(
            contactid=contact_id,
            fullname=f"{first} {last}".strip(),
            firstname=first,
            lastname=last,
            jobtitle=(body.job_title or None),
            emailaddress1=body.email,
            statecode="Active",
        )
        db.add(contact)
        # Need the contact row in place before the phone UPDATE can match on it.
        db.flush()
        # Re-stringify after flush — SQLAlchemy may re-coerce the PK on read-back.
        contact_id = str(contact.contactid)
        if body.phone:
            write_phone(db, contact_id, body.phone)
        contact_created = True

    # ---- Duplicate-link guard --------------------------------------------
    existing_link = db.execute(
        select(AccountContact.lvo_accountcontactid).where(
            func.upper(AccountContact.lvo_accountid) == account_id.upper(),
            func.upper(AccountContact.lvo_contactid) == contact_id.upper(),
            AccountContact.statecode == "Active",
        )
    ).scalar()
    if existing_link:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "DUPLICATE_CONTACT",
                "message": "Contact is already attached to this account.",
            },
        )

    if body.is_primary:
        _demote_existing_primary(db, account_id)

    new_id = str(uuid.uuid4())
    link = AccountContact(
        lvo_accountcontactid=new_id,
        lvo_accountid=account_id.upper(),
        lvo_contactid=contact_id.upper(),
        lvo_role=body.role,
        lvo_isprimary=body.is_primary,
        lvo_lasttouchdate=None,
        lvo_createdat=now,
        lvo_updatedat=now,
        statecode="Active",
    )
    db.add(link)

    _write_audit(
        db,
        entity_id=new_id,
        account_id=account_id,
        action="create",
        changed_by=x_user_id,
        diff={
            "contact_id": contact_id,
            "contact_created": contact_created,
            "role": body.role,
            "is_primary": body.is_primary,
            "first_name": body.first_name,
            "last_name": body.last_name,
            "email": body.email,
            "phone": body.phone,
            "job_title": body.job_title,
            "code": SUCC_ADD,
        },
    )
    db.commit()
    background_tasks.add_task(recalculate_async, account_id)

    eligibility = _eligibility_for(link=link, active_deal_ids=[])
    return _to_ref(link, contact, phone=body.phone, eligibility=eligibility)


# ---------------------------------------------------------------------------
# PATCH /api/accounts/{id}/contacts/{linkId}
# ---------------------------------------------------------------------------


@router.patch(
    "/{account_id}/contacts/{contact_link_id}",
    response_model=AccountContactRef,
    summary="Update contact fields, role, or primary flag",
    responses={
        404: {"description": "Account or link not found"},
        422: {"description": "Validation error (ERR_MSG_0013)"},
    },
)
def update_account_contact(
    background_tasks: BackgroundTasks,
    account_id: str = Path(...),
    contact_link_id: str = Path(...),
    body: AccountContactUpdateRequest = Body(...),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> AccountContactRef:
    _ensure_account(db, account_id)
    link = _ensure_link(db, account_id, contact_link_id)
    link_contact_id = str(link.lvo_contactid or "")
    contact = _ensure_contact(db, link_contact_id)

    provided = body.model_fields_set
    before_link = {
        "role": link.lvo_role,
        "is_primary": bool(link.lvo_isprimary),
    }
    before_contact = {
        "first_name": contact.firstname,
        "last_name": contact.lastname,
        "email": contact.emailaddress1,
        "job_title": contact.jobtitle,
    }

    # ---- Update the link row ---------------------------------------------
    if "role" in provided:
        link.lvo_role = body.role
    if "is_primary" in provided and body.is_primary is not None:
        if body.is_primary and not link.lvo_isprimary:
            _demote_existing_primary(
                db, account_id, exclude_link_id=contact_link_id
            )
        link.lvo_isprimary = body.is_primary
    link.lvo_updatedat = datetime.now(timezone.utc)

    # ---- Update the underlying contact row -------------------------------
    contact_touched = False
    if "first_name" in provided:
        contact.firstname = (body.first_name or "").strip() or None
        contact_touched = True
    if "last_name" in provided:
        contact.lastname = (body.last_name or "").strip() or None
        contact_touched = True
    if contact_touched:
        contact.fullname = (
            f"{(contact.firstname or '').strip()} "
            f"{(contact.lastname or '').strip()}".strip()
            or None
        )
    if "email" in provided:
        contact.emailaddress1 = body.email
        contact_touched = True
    if "job_title" in provided:
        contact.jobtitle = body.job_title
        contact_touched = True

    new_phone: str | None = None
    if "phone" in provided:
        new_phone = body.phone
        write_phone(db, link_contact_id, new_phone)
        contact_touched = True

    _write_audit(
        db,
        entity_id=contact_link_id,
        account_id=account_id,
        action="update",
        changed_by=x_user_id,
        diff={
            "before": {**before_link, **before_contact},
            "after": {
                "role": link.lvo_role,
                "is_primary": bool(link.lvo_isprimary),
                "first_name": contact.firstname,
                "last_name": contact.lastname,
                "email": contact.emailaddress1,
                "job_title": contact.jobtitle,
                **({"phone": new_phone} if "phone" in provided else {}),
            },
            "code": SUCC_UPDATE,
        },
    )
    db.commit()
    background_tasks.add_task(recalculate_async, account_id)

    # Re-read the live phone (covers both newly-written and unchanged values).
    phone_map = bulk_read_phones(db, [link_contact_id])
    phone_value = phone_map.get(link_contact_id.upper())

    deals = _active_deal_ids_for_contacts(db, [link_contact_id])
    eligibility = _eligibility_for(
        link=link,
        active_deal_ids=deals.get(link_contact_id.upper(), []),
    )
    return _to_ref(link, contact, phone=phone_value, eligibility=eligibility)


# ---------------------------------------------------------------------------
# DELETE /api/accounts/{id}/contacts/{linkId}
# ---------------------------------------------------------------------------


@router.delete(
    "/{account_id}/contacts/{contact_link_id}",
    response_model=AccountContactDeleteResponse,
    summary="Remove a contact from the account (soft-delete)",
    responses={
        404: {"description": "Account or link not found"},
        409: {
            "description": (
                "ERR_MSG_0008 — primary contacts cannot be removed; "
                "ERR_MSG_0009 — contact is referenced by an active deal"
            )
        },
    },
)
def delete_account_contact(
    background_tasks: BackgroundTasks,
    account_id: str = Path(...),
    contact_link_id: str = Path(...),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> AccountContactDeleteResponse:
    _ensure_account(db, account_id)
    link = _ensure_link(db, account_id, contact_link_id)
    link_contact_id = str(link.lvo_contactid or "")

    # ---- Pre-flight gate -------------------------------------------------
    deals = _active_deal_ids_for_contacts(db, [link_contact_id])
    eligibility = _eligibility_for(
        link=link,
        active_deal_ids=deals.get(link_contact_id.upper(), []),
    )
    if not eligibility.can_delete:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": eligibility.code,
                "message": eligibility.message,
                "affectedDealIds": eligibility.affected_deal_ids,
            },
        )

    # ---- Soft-delete the link (contact row stays — disassociate only) ----
    link.statecode = "Inactive"
    link.lvo_updatedat = datetime.now(timezone.utc)

    _write_audit(
        db,
        entity_id=contact_link_id,
        account_id=account_id,
        action="delete",
        changed_by=x_user_id,
        diff={
            "contact_id": link_contact_id,
            "role": link.lvo_role,
            "code": SUCC_DELETE,
        },
    )
    db.commit()
    background_tasks.add_task(recalculate_async, account_id)

    return AccountContactDeleteResponse(
        account_id=account_id,
        contact_link_id=contact_link_id,
        code=SUCC_DELETE,
        message="Contact has been removed from the account.",
    )


# ---------------------------------------------------------------------------
# GET /api/accounts/{id}/contacts/{linkId}/delete-eligibility
# ---------------------------------------------------------------------------


@router.get(
    "/{account_id}/contacts/{contact_link_id}/delete-eligibility",
    response_model=AccountContactDeleteEligibilityResponse,
    summary=(
        "Pre-flight check for the FE delete dialog (CONF_MSG_0003). "
        "Returns the same restriction shape embedded in /contacts."
    ),
)
def check_delete_eligibility(
    account_id: str = Path(...),
    contact_link_id: str = Path(...),
    db: Session = Depends(get_db),
) -> AccountContactDeleteEligibilityResponse:
    _ensure_account(db, account_id)
    link = _ensure_link(db, account_id, contact_link_id)
    link_contact_id = str(link.lvo_contactid or "")

    deals = _active_deal_ids_for_contacts(db, [link_contact_id])
    eligibility = _eligibility_for(
        link=link,
        active_deal_ids=deals.get(link_contact_id.upper(), []),
    )

    # Surface the confirmation-message code to the FE so it can render
    # CONF_MSG_0003 verbatim when can_delete is True.
    if eligibility.can_delete:
        return AccountContactDeleteEligibilityResponse(
            account_id=account_id,
            contact_link_id=contact_link_id,
            can_delete=True,
            code=CONF_DELETE,
            message=(
                "Are you sure you want to remove this contact from the "
                "account? This action may impact associated deals."
            ),
        )
    return AccountContactDeleteEligibilityResponse(
        account_id=account_id,
        contact_link_id=contact_link_id,
        can_delete=False,
        code=eligibility.code,
        message=eligibility.message,
        affected_deal_ids=eligibility.affected_deal_ids,
    )

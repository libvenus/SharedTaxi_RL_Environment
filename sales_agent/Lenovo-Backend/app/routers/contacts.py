"""Contact-link write endpoints (Decision Maker + additional stakeholders).

The Deal Detailed View user story exposes contacts as a panel on the
detail page; this router lets the frontend manage which contacts are
attached to a deal and which one is the decision maker.

Endpoints
---------
POST   /api/opportunities/{id}/contacts                Link a contact to the deal.
PATCH  /api/opportunities/{id}/contacts/{contactLinkId} Change role / DM flag.
DELETE /api/opportunities/{id}/contacts/{contactLinkId} Soft-delete the link.

All mutations write to ``lvo_audit_log`` and enqueue a deal-health recalc
because stakeholder count is one of the five health components.
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
from app.models import Contact, Opportunity, OpportunityContact
from app.schemas import (
    ContactLinkCreateRequest,
    ContactLinkDeleteResponse,
    ContactLinkUpdateRequest,
    ContactRef,
)
from app.services.audit_log import write_audit_event
from app.services.contact_phone import read_phone, write_phone
from app.services.deal_recalc import recalculate_async

router = APIRouter(prefix="/api/opportunities", tags=["opportunity-contacts"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_opportunity(db: Session, opportunity_id: str) -> Opportunity:
    opp = db.get(Opportunity, opportunity_id)
    if opp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Opportunity '{opportunity_id}' not found.",
        )
    return opp


def _ensure_contact(db: Session, contact_id: str) -> Contact:
    contact = db.execute(
        select(Contact).where(
            func.upper(cast(Contact.contactid, String)) == contact_id.upper()
        )
    ).scalar_one_or_none()
    if contact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contact '{contact_id}' not found.",
        )
    return contact


def _ensure_link(
    db: Session, opportunity_id: str, link_id: str
) -> OpportunityContact:
    link = db.get(OpportunityContact, link_id)
    if link is None or link.statecode != "Active":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contact link '{link_id}' not found.",
        )
    if (link.lvo_opportunityid or "").upper() != opportunity_id.upper():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Contact link '{link_id}' does not belong to opportunity "
                f"'{opportunity_id}'."
            ),
        )
    return link


def _write_audit(
    db: Session,
    *,
    entity_id: str,
    opportunity_id: str,
    action: str,
    changed_by: str | None,
    diff: dict | None,
) -> None:
    write_audit_event(
        db,
        entity_type="opportunity_contact",
        entity_id=entity_id,
        opportunity_id=opportunity_id,
        action=action,
        category="crm_writeback",
        actor_type="seller",
        changed_by=changed_by,
        diff=diff,
    )


def _to_contact_ref(
    link: OpportunityContact,
    contact: Contact | None,
    *,
    phone: str | None = None,
) -> ContactRef:
    """Build the ``ContactRef`` payload returned by every contact endpoint.

    ``phone`` is accepted as a parameter (rather than queried in here) so the
    LIST flow can batch the lookup via
    ``app.services.contact_phone.bulk_read_phones`` instead of issuing one
    query per row. Single-contact callers (POST / PATCH responses) read the
    value via ``read_phone`` before invoking this helper — see those handlers.
    """
    name = None
    first_name = None
    last_name = None
    if contact is not None:
        first_name = contact.firstname
        last_name = contact.lastname
        name = contact.fullname or (
            f"{(first_name or '').strip()} {(last_name or '').strip()}".strip()
        )
    return ContactRef(
        id=link.lvo_opportunitycontactid,
        contact_id=link.lvo_contactid,
        name=name,
        first_name=first_name,
        last_name=last_name,
        role=link.lvo_role,
        is_decision_maker=bool(link.lvo_isdecisionmaker),
        last_touch_date=link.lvo_lasttouchdate,
        job_title=contact.jobtitle if contact else None,
        email=contact.emailaddress1 if contact else None,
        phone=phone,
    )


def _demote_existing_decision_maker(
    db: Session, opportunity_id: str, exclude_link_id: str | None = None
) -> None:
    """Set lvo_isdecisionmaker=False on every other active link on the deal.

    We allow at most one decision maker per deal — POST/PATCH that promote
    a contact to DM call this first to keep the invariant true.
    """
    rows = (
        db.execute(
            select(OpportunityContact).where(
                func.upper(OpportunityContact.lvo_opportunityid)
                == opportunity_id.upper(),
                OpportunityContact.statecode == "Active",
                OpportunityContact.lvo_isdecisionmaker.is_(True),
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        if exclude_link_id and row.lvo_opportunitycontactid == exclude_link_id:
            continue
        row.lvo_isdecisionmaker = False
        row.lvo_updatedat = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# POST /api/opportunities/{id}/contacts
# ---------------------------------------------------------------------------


@router.post(
    "/{opportunity_id}/contacts",
    response_model=ContactRef,
    status_code=status.HTTP_201_CREATED,
    summary="Attach a contact to the deal",
    responses={
        404: {"description": "Opportunity or contact not found"},
        409: {"description": "Contact is already attached to this deal"},
    },
)
def add_contact_link(
    background_tasks: BackgroundTasks,
    opportunity_id: str = Path(...),
    body: ContactLinkCreateRequest = Body(...),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> ContactRef:
    """Link an existing contact to the deal.

    * `is_decision_maker=True` automatically demotes any contact currently
      flagged as the decision maker on this deal — only one DM at a time.
    * Re-attaching a contact that already has an Active link returns 409.
    """
    _ensure_opportunity(db, opportunity_id)
    contact = _ensure_contact(db, body.contact_id)

    existing = db.execute(
        select(OpportunityContact.lvo_opportunitycontactid).where(
            func.upper(OpportunityContact.lvo_opportunityid) == opportunity_id.upper(),
            func.upper(OpportunityContact.lvo_contactid) == body.contact_id.upper(),
            OpportunityContact.statecode == "Active",
        )
    ).scalar()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "DUPLICATE_CONTACT",
                "message": "Contact is already attached to this deal.",
            },
        )

    if body.is_decision_maker:
        _demote_existing_decision_maker(db, opportunity_id)

    now = datetime.now(timezone.utc)
    new_id = str(uuid.uuid4())
    link = OpportunityContact(
        lvo_opportunitycontactid=new_id,
        lvo_opportunityid=opportunity_id.upper(),
        lvo_contactid=body.contact_id.upper(),
        lvo_role=body.role,
        lvo_isdecisionmaker=body.is_decision_maker,
        lvo_lasttouchdate=None,
        lvo_createdat=now,
        lvo_updatedat=now,
        statecode="Active",
    )
    db.add(link)

    _write_audit(
        db,
        entity_id=new_id,
        opportunity_id=opportunity_id,
        action="create",
        changed_by=x_user_id,
        diff={
            "contact_id": body.contact_id,
            "role": body.role,
            "is_decision_maker": body.is_decision_maker,
        },
    )
    db.commit()
    background_tasks.add_task(recalculate_async, opportunity_id)

    return _to_contact_ref(link, contact, phone=read_phone(db, body.contact_id))


# ---------------------------------------------------------------------------
# PATCH /api/opportunities/{id}/contacts/{contactLinkId}
# ---------------------------------------------------------------------------


@router.patch(
    "/{opportunity_id}/contacts/{contact_link_id}",
    response_model=ContactRef,
    summary="Update role / decision-maker flag / phone on a contact link",
)
def update_contact_link(
    background_tasks: BackgroundTasks,
    opportunity_id: str = Path(...),
    contact_link_id: str = Path(...),
    body: ContactLinkUpdateRequest = Body(...),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> ContactRef:
    """Update role / DM flag / phone. All fields optional (true PATCH).

    ``phone`` is unusual — it doesn't live on the link, it lives on the
    underlying ``contact`` row. Editing it via this endpoint mutates the
    contact and is therefore visible on every other deal that contact is
    attached to and on the Account Contacts tab. The audit log captures
    the before/after so the change is traceable.
    """
    _ensure_opportunity(db, opportunity_id)
    link = _ensure_link(db, opportunity_id, contact_link_id)

    provided = body.model_fields_set
    phone_before = read_phone(db, link.lvo_contactid) if "phone" in provided else None
    before = {
        "role": link.lvo_role,
        "is_decision_maker": bool(link.lvo_isdecisionmaker),
    }
    if "phone" in provided:
        before["phone"] = phone_before

    if "role" in provided:
        link.lvo_role = body.role

    if "is_decision_maker" in provided and body.is_decision_maker is not None:
        if body.is_decision_maker and not link.lvo_isdecisionmaker:
            _demote_existing_decision_maker(
                db, opportunity_id, exclude_link_id=contact_link_id
            )
        link.lvo_isdecisionmaker = body.is_decision_maker

    if "phone" in provided:
        # Empty string ⇒ clear, mirroring the Account-Contacts write path
        # so both screens behave consistently.
        normalised_phone = (body.phone or "").strip() or None
        write_phone(db, link.lvo_contactid, normalised_phone)

    link.lvo_updatedat = datetime.now(timezone.utc)

    after = {
        "role": link.lvo_role,
        "is_decision_maker": bool(link.lvo_isdecisionmaker),
    }
    if "phone" in provided:
        after["phone"] = read_phone(db, link.lvo_contactid)

    _write_audit(
        db,
        entity_id=contact_link_id,
        opportunity_id=opportunity_id,
        action="update",
        changed_by=x_user_id,
        diff={"before": before, "after": after},
    )
    db.commit()
    background_tasks.add_task(recalculate_async, opportunity_id)

    contact = db.execute(
        select(Contact).where(
            func.upper(cast(Contact.contactid, String)) == link.lvo_contactid.upper()
        )
    ).scalar_one_or_none()
    return _to_contact_ref(
        link, contact, phone=read_phone(db, link.lvo_contactid)
    )


# ---------------------------------------------------------------------------
# DELETE /api/opportunities/{id}/contacts/{contactLinkId}
# ---------------------------------------------------------------------------


@router.delete(
    "/{opportunity_id}/contacts/{contact_link_id}",
    response_model=ContactLinkDeleteResponse,
    summary="Remove a contact from the deal (soft-delete)",
)
def delete_contact_link(
    background_tasks: BackgroundTasks,
    opportunity_id: str = Path(...),
    contact_link_id: str = Path(...),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> ContactLinkDeleteResponse:
    """Soft-delete the link by setting statecode='Inactive'.

    The contact row itself is never deleted — only the link is removed.
    """
    _ensure_opportunity(db, opportunity_id)
    link = _ensure_link(db, opportunity_id, contact_link_id)

    link.statecode = "Inactive"
    link.lvo_updatedat = datetime.now(timezone.utc)

    _write_audit(
        db,
        entity_id=contact_link_id,
        opportunity_id=opportunity_id,
        action="delete",
        changed_by=x_user_id,
        diff={"contact_id": link.lvo_contactid, "role": link.lvo_role},
    )
    db.commit()
    background_tasks.add_task(recalculate_async, opportunity_id)

    return ContactLinkDeleteResponse(
        opportunity_id=opportunity_id,
        contact_link_id=contact_link_id,
        message="Contact has been removed from the deal.",
    )

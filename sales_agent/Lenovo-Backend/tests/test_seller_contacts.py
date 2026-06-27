"""Unit tests for app/services/seller_contacts.py."""

from __future__ import annotations

from datetime import datetime, timezone

from app.models import Contact, OpportunityContact
from app.services.seller_contacts import (
    _ContactLinkRow,
    build_seller_contact_items,
    contact_display_name,
    pick_primary_link,
)


def _link(
    *,
    contact_id: str = "C1",
    is_dm: bool = False,
    role: str | None = "Champion",
    touch: datetime | None = None,
    opp_id: str = "O1",
    opp_name: str = "Deal A",
) -> _ContactLinkRow:
    return _ContactLinkRow(
        contact_id=contact_id,
        contact=Contact(
            contactid=contact_id,
            fullname="Jane Doe",
            firstname="Jane",
            lastname="Doe",
            emailaddress1="jane@example.com",
            jobtitle="CTO",
            statecode="Active",
        ),
        link=OpportunityContact(
            lvo_opportunitycontactid=f"link-{opp_id}",
            lvo_opportunityid=opp_id,
            lvo_contactid=contact_id,
            lvo_role=role,
            lvo_isdecisionmaker=is_dm,
            lvo_lasttouchdate=touch,
            lvo_createdat=datetime(2026, 1, 1, tzinfo=timezone.utc),
            lvo_updatedat=datetime(2026, 1, 1, tzinfo=timezone.utc),
            statecode="Active",
        ),
        opportunity_id=opp_id,
        opportunity_name=opp_name,
        account_id="A1",
        account_name="Acme Corp",
    )


def test_contact_display_name_prefers_fullname() -> None:
    c = Contact(contactid="1", fullname="Full Name", firstname="A", lastname="B")
    assert contact_display_name(c) == "Full Name"


def test_pick_primary_link_prefers_decision_maker() -> None:
    rows = [
        _link(is_dm=False, opp_id="O1"),
        _link(is_dm=True, opp_id="O2", opp_name="Deal B"),
    ]
    primary = pick_primary_link(rows)
    assert primary.opportunity_id == "O2"
    assert primary.link.lvo_isdecisionmaker is True


def test_build_seller_contact_items_dedupes_and_counts() -> None:
    rows = [
        _link(opp_id="O1"),
        _link(opp_id="O2", opp_name="Deal B"),
    ]
    items = build_seller_contact_items(rows, phones={"C1": "+1-555-0100"})
    assert len(items) == 1
    assert items[0].linked_opportunity_count == 2
    assert items[0].phone == "+1-555-0100"
    assert items[0].email == "jane@example.com"

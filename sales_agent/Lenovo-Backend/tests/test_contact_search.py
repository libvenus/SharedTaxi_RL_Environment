"""Unit tests for app/services/contact_search.py."""

from __future__ import annotations

from app.models import Contact
from app.services.contact_search import name_match_rank


def test_name_match_rank_prefers_exact_first_name() -> None:
    exact = Contact(
        contactid="1",
        firstname="John",
        lastname="Smith",
        fullname="John Smith",
        statecode="Active",
    )
    partial = Contact(
        contactid="2",
        firstname="Johnny",
        lastname="Doe",
        fullname="Johnny Doe",
        statecode="Active",
    )
    assert name_match_rank("john", exact)[0] < name_match_rank("john", partial)[0]


def test_name_match_rank_prefers_first_name_prefix_over_fullname_only() -> None:
    first = Contact(
        contactid="1",
        firstname="Raj",
        lastname="Kumar",
        fullname="Raj Kumar",
        statecode="Active",
    )
    full_only = Contact(
        contactid="2",
        firstname="Suresh",
        lastname="Patel",
        fullname="Rajesh Kumar",
        statecode="Active",
    )
    assert name_match_rank("raj", first)[0] < name_match_rank("raj", full_only)[0]

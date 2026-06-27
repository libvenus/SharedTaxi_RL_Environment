"""Pure-function tests for app/services/contact_validation.py.

Focuses on the three concerns the user-story acceptance criteria pin down:

* Email validation surfaces ERR_MSG_0013 on bad input.
* Phone validation accepts the formats the UI mockup uses.
* compute_delete_eligibility encodes the precedence rule between
  ERR_MSG_0008 (primary) and ERR_MSG_0009 (active-deal references).
"""

from __future__ import annotations

import pytest

from app.services.contact_validation import (
    ERR_ACTIVE_DEAL_REFERENCES,
    ERR_PRIMARY_CONTACT,
    DeleteEligibility,
    compute_delete_eligibility,
    normalise_email,
    normalise_phone,
    validate_email,
    validate_phone,
)


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "rajesh.kumar@infosys.com",
        "Rajesh.Kumar@INFOSYS.COM",
        "  rajesh@example.co.in  ",
        "user+tag@example.com",
        "first.last@sub.domain.example",
        None,        # absence is treated as valid
        "",          # whitespace becomes None — also valid
        "   ",
    ],
)
def test_validate_email_accepts(value):
    assert validate_email(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "no-at-sign.example.com",
        "missing-domain@",
        "@missing-local.com",
        "two@@signs.com",
        "spaces in@local.com",
        "trailing-dot@domain.",
        "tld-too-short@a.b",
    ],
)
def test_validate_email_rejects(value):
    assert validate_email(value) is False


def test_validate_email_rejects_overlong_address():
    long_local = "a" * 250
    assert validate_email(f"{long_local}@example.com") is False


def test_normalise_email_lowercases_and_trims():
    assert normalise_email("  Rajesh.Kumar@Infosys.COM ") == "rajesh.kumar@infosys.com"
    assert normalise_email("") is None
    assert normalise_email(None) is None


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "+91 98765 00000",
        "+1-555-555-1234",
        "(212) 555-1234",
        "5551234567",
        "+44 20 7946 0958",
        "  +91 98765 00000  ",
        None,
        "",
    ],
)
def test_validate_phone_accepts(value):
    assert validate_phone(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "98765",                 # too short
        "abc 9876543210",        # contains alpha
        "9876_543_210",          # contains underscore
        "+91 9876543210" * 3,    # too long
        "+91-call-now",
    ],
)
def test_validate_phone_rejects(value):
    assert validate_phone(value) is False


def test_normalise_phone_preserves_formatting():
    """User-typed format should round-trip back to the UI verbatim."""
    raw = "  +91 98765 00000  "
    assert normalise_phone(raw) == "+91 98765 00000"


# ---------------------------------------------------------------------------
# Delete eligibility
# ---------------------------------------------------------------------------


def test_eligibility_ok_when_neither_blocker_present():
    result = compute_delete_eligibility(is_primary=False, active_deal_ids=[])
    assert result.can_delete is True
    assert result.code is None
    assert result.message is None
    assert result.affected_deal_ids == []


def test_eligibility_blocked_by_primary_flag():
    result = compute_delete_eligibility(is_primary=True, active_deal_ids=[])
    assert result.can_delete is False
    assert result.code == ERR_PRIMARY_CONTACT
    assert "primary" in (result.message or "").lower()


def test_eligibility_blocked_by_active_deal_references():
    deal_ids = ["d-1", "d-2"]
    result = compute_delete_eligibility(is_primary=False, active_deal_ids=deal_ids)
    assert result.can_delete is False
    assert result.code == ERR_ACTIVE_DEAL_REFERENCES
    assert result.affected_deal_ids == deal_ids
    assert "active deals" in (result.message or "").lower()


def test_primary_takes_precedence_over_active_deals():
    """The user story says demote first, then re-check — surface the primary
    error first so the FE walks the user through the resolution in order."""
    result = compute_delete_eligibility(is_primary=True, active_deal_ids=["d-1"])
    assert result.can_delete is False
    assert result.code == ERR_PRIMARY_CONTACT
    # Affected deals are intentionally NOT surfaced in the primary case.
    assert result.affected_deal_ids == []


def test_eligibility_ok_factory_returns_clean_value():
    ok = DeleteEligibility.ok()
    assert ok.can_delete is True
    assert ok.code is None
    assert ok.message is None
    assert ok.affected_deal_ids == []

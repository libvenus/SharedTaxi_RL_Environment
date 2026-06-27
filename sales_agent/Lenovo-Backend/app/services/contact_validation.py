"""Pure validators + delete-eligibility helpers for the account-contacts API.

Kept intentionally DB-free so they're trivially unit-testable; callers (the
account_contacts router) collect the necessary signals from the database and
hand them in.

Public surface
--------------
* ``normalise_email`` / ``validate_email`` — RFC-pragmatic email check.
* ``normalise_phone`` / ``validate_phone`` — international-friendly phone check.
* ``DeleteEligibility``                    — value object returned by
  ``compute_delete_eligibility``; encodes the user-story error codes
  ERR_MSG_0008 (primary contact) and ERR_MSG_0009 (active deal references).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

# Pragmatic RFC-5322-lite: requires a local part, an "@", and a dotted domain.
# We deliberately avoid pinning to Pydantic's `EmailStr` to stay dependency-free
# in the unit tests AND to give a single, predictable error code surface.
_EMAIL_RE = re.compile(
    r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
)


def normalise_email(value: str | None) -> str | None:
    """Lower-case + trim. Returns ``None`` for empty / whitespace-only input."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    return cleaned or None


def validate_email(value: str | None) -> bool:
    """``True`` when ``value`` looks like a usable email address.

    ``None`` / empty input is treated as **valid** so optional email fields
    can be skipped — the router decides whether absence is acceptable.
    """
    cleaned = normalise_email(value)
    if cleaned is None:
        return True
    if len(cleaned) > 254:
        return False
    return bool(_EMAIL_RE.match(cleaned))


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------

# Permits the formats actually used in the UI mockup:
#   "+91 98765 00000", "+1-555-555-1234", "(212) 555-1234", "5551234567".
# Strips separators before counting digits; requires 7-15 digits, optional
# leading "+". 7 is the minimum for short national numbers; 15 is the E.164
# upper bound.
_PHONE_DIGITS = re.compile(r"\d")
_PHONE_ALLOWED_CHARS = re.compile(r"^[\d+\-\s().]+$")
_MIN_PHONE_DIGITS = 7
_MAX_PHONE_DIGITS = 15


def normalise_phone(value: str | None) -> str | None:
    """Trim outer whitespace; returns ``None`` for empty input.

    We intentionally keep separators so the original UI formatting round-trips
    back to the user — ``"+91 98765 00000"`` stays ``"+91 98765 00000"`` rather
    than becoming an opaque ``"+919876500000"``.
    """
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def validate_phone(value: str | None) -> bool:
    """``True`` when ``value`` is a plausible phone number."""
    cleaned = normalise_phone(value)
    if cleaned is None:
        return True
    if not _PHONE_ALLOWED_CHARS.match(cleaned):
        return False
    digits = _PHONE_DIGITS.findall(cleaned)
    return _MIN_PHONE_DIGITS <= len(digits) <= _MAX_PHONE_DIGITS


# ---------------------------------------------------------------------------
# Delete eligibility
# ---------------------------------------------------------------------------


# Error codes from the user story.
#  - ERR_MSG_0008  →  contact is primary (FE shows the locked-icon hover text)
#  - ERR_MSG_0009  →  contact is referenced by an active deal
#  - ERR_MSG_0013  →  invalid email/phone format on add or update
ERR_PRIMARY_CONTACT = "ERR_MSG_0008"
ERR_ACTIVE_DEAL_REFERENCES = "ERR_MSG_0009"
ERR_INVALID_FIELD = "ERR_MSG_0013"

SUCC_DELETE = "SUCC_MSG_0007"
SUCC_ADD = "SUCC_MSG_0008"
SUCC_UPDATE = "SUCC_MSG_0009"

CONF_DELETE = "CONF_MSG_0003"


@dataclass(frozen=True)
class DeleteEligibility:
    """Whether a given account-contact link can be removed right now.

    ``can_delete=True`` means the DELETE endpoint will succeed; ``False`` means
    the FE should grey-out the button and show ``message`` on hover. The
    ``code`` field maps to the user-story enum so the FE can localise.
    """

    can_delete: bool
    code: str | None = None
    message: str | None = None
    affected_deal_ids: list[str] = field(default_factory=list)

    @classmethod
    def ok(cls) -> "DeleteEligibility":
        return cls(can_delete=True)


def compute_delete_eligibility(
    *,
    is_primary: bool,
    active_deal_ids: Sequence[str] | None = None,
) -> DeleteEligibility:
    """Pure rule: derive the delete decision from two pre-fetched signals.

    Precedence — primary check beats active-deal check, because once a contact
    is demoted from primary the FE will re-issue this query and likely surface
    the active-deal error next. Showing both at once would be noisier than the
    user story asks for.
    """
    if is_primary:
        return DeleteEligibility(
            can_delete=False,
            code=ERR_PRIMARY_CONTACT,
            message=(
                "This contact is the primary contact for the account. "
                "Demote them or assign a new primary contact before removing."
            ),
        )
    deal_ids = list(active_deal_ids or [])
    if deal_ids:
        return DeleteEligibility(
            can_delete=False,
            code=ERR_ACTIVE_DEAL_REFERENCES,
            message=(
                "This contact is associated with one or more active deals. "
                "Remove the contact from those deals before deleting them "
                "from the account."
            ),
            affected_deal_ids=deal_ids,
        )
    return DeleteEligibility.ok()


__all__ = [
    "CONF_DELETE",
    "DeleteEligibility",
    "ERR_ACTIVE_DEAL_REFERENCES",
    "ERR_INVALID_FIELD",
    "ERR_PRIMARY_CONTACT",
    "SUCC_ADD",
    "SUCC_DELETE",
    "SUCC_UPDATE",
    "compute_delete_eligibility",
    "normalise_email",
    "normalise_phone",
    "validate_email",
    "validate_phone",
]

"""Pure functions that derive Account Type and Account Status.

Mirrors the same architectural style as ``deal_health.py``: no SQLAlchemy
imports, no DB calls, no I/O. The orchestrator (``account_recalc.py``)
loads the inputs, calls these functions, and persists the results.

Two derived attributes are produced:

* **Account Type** — ``Prospect`` until the account books at least one
  Closed-Won deal; flips to ``Customer`` from then on.
* **Account Status** — ``Active`` / ``Inactive`` / ``At-Risk``.
    * ``Inactive`` when the underlying ``account.statecode`` is Inactive,
      or when the account hasn't had any interaction for a long idle
      period (defaults to 180 days).
    * ``At-Risk`` when the account has at least one OPEN deal whose
      ``dealHealth < at_risk_health_threshold`` (default 50).
    * ``Active`` otherwise.

The thresholds come from the existing ``lvo_dealhealthconfig`` JSONB row
(see ``sql/2026_06_account_view_schema.sql`` which adds the
``account_status`` sub-key) so they can be tuned without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal


AccountTypeValue = Literal["Prospect", "Customer"]
AccountStatusValue = Literal["Active", "Inactive", "At-Risk"]


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountStatusSettings:
    """Tunables — usually loaded from lvo_dealhealthconfig.account_status."""

    at_risk_health_threshold: int = 50
    """Open deals scoring strictly below this value flip the account to At-Risk."""

    inactive_idle_days: int = 180
    """When >0, an account with no interaction for this many days becomes Inactive.

    Set to 0 (or any non-positive number) to disable the idle-based rule —
    Inactive then strictly follows ``account.statecode``.
    """

    @classmethod
    def from_config(cls, config: dict | None) -> "AccountStatusSettings":
        """Build from the ``lvo_settings.account_status`` JSON sub-tree.

        Falls back to defaults when keys are missing — the caller can pass
        ``None`` to get the entire default profile.
        """
        if not config:
            return cls()
        return cls(
            at_risk_health_threshold=int(
                config.get("at_risk_health_threshold", 50) or 50
            ),
            inactive_idle_days=int(config.get("inactive_idle_days", 180) or 180),
        )


DEFAULT_ACCOUNT_STATUS_SETTINGS = AccountStatusSettings()


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountStatusInputs:
    """Everything the derivation rules need, decoupled from SQLAlchemy.

    All numeric fields default to neutral values so calling code can keep
    the dataclass small when a signal is missing (e.g. an account with no
    deals).
    """

    statecode: str | None
    """Raw account.statecode — Active / Inactive."""

    open_deals_count: int = 0
    open_deals_with_low_health: int = 0
    """Open deals whose lvo_dealhealthscore < at_risk_health_threshold."""

    has_won_deal: bool = False
    """True when at least one opportunity reached Closed Won (or statecode='Won')."""

    last_interaction_date: datetime | None = None
    today: date | None = None


# ----------------------------------------------------------------------------
# Pure derivations
# ----------------------------------------------------------------------------


def derive_account_type(*, has_won_deal: bool) -> AccountTypeValue:
    """Customer the moment the account has any Closed Won deal."""
    return "Customer" if has_won_deal else "Prospect"


def derive_account_status(
    inputs: AccountStatusInputs,
    settings: AccountStatusSettings = DEFAULT_ACCOUNT_STATUS_SETTINGS,
) -> AccountStatusValue:
    """Apply the three-tier rule set:

    1. Inactive — explicit statecode='Inactive' OR (idle > inactive_idle_days
       and ``inactive_idle_days > 0``).
    2. At-Risk — at least one open deal with health below the threshold.
    3. Active — fallback.
    """
    statecode = (inputs.statecode or "Active")
    if statecode == "Inactive":
        return "Inactive"

    today = inputs.today or datetime.now(timezone.utc).date()

    if (
        settings.inactive_idle_days > 0
        and inputs.last_interaction_date is not None
    ):
        idle_days = (today - inputs.last_interaction_date.date()).days
        if idle_days > settings.inactive_idle_days:
            return "Inactive"

    if inputs.open_deals_with_low_health > 0:
        return "At-Risk"

    return "Active"


__all__ = [
    "AccountStatusInputs",
    "AccountStatusSettings",
    "AccountStatusValue",
    "AccountTypeValue",
    "DEFAULT_ACCOUNT_STATUS_SETTINGS",
    "derive_account_status",
    "derive_account_type",
]

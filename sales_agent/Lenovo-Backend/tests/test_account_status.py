"""Unit tests for app.services.account_status.

These tests cover every branch of the derivation rules so the View Account
grid behaviour can be verified independent of Postgres / FastAPI.
"""

from __future__ import annotations

from datetime import datetime, timezone, date

from app.services.account_status import (
    AccountStatusInputs,
    AccountStatusSettings,
    derive_account_status,
    derive_account_type,
)


SETTINGS = AccountStatusSettings(at_risk_health_threshold=50, inactive_idle_days=180)


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


# ----------------------------------------------------------------------------
# derive_account_type
# ----------------------------------------------------------------------------


def test_account_type_is_prospect_without_won_deal() -> None:
    assert derive_account_type(has_won_deal=False) == "Prospect"


def test_account_type_is_customer_with_won_deal() -> None:
    assert derive_account_type(has_won_deal=True) == "Customer"


# ----------------------------------------------------------------------------
# derive_account_status — Inactive paths
# ----------------------------------------------------------------------------


def test_inactive_when_statecode_says_inactive() -> None:
    inputs = AccountStatusInputs(statecode="Inactive", open_deals_count=2)
    assert derive_account_status(inputs, SETTINGS) == "Inactive"


def test_inactive_when_idle_beyond_threshold() -> None:
    inputs = AccountStatusInputs(
        statecode="Active",
        last_interaction_date=_ts("2025-01-01"),
        today=date(2026, 6, 1),  # 17 months idle
    )
    assert derive_account_status(inputs, SETTINGS) == "Inactive"


def test_active_when_idle_within_threshold() -> None:
    inputs = AccountStatusInputs(
        statecode="Active",
        last_interaction_date=_ts("2026-05-15"),
        today=date(2026, 6, 1),
    )
    assert derive_account_status(inputs, SETTINGS) == "Active"


def test_idle_rule_disabled_when_threshold_zero() -> None:
    settings = AccountStatusSettings(
        at_risk_health_threshold=50, inactive_idle_days=0
    )
    inputs = AccountStatusInputs(
        statecode="Active",
        last_interaction_date=_ts("2020-01-01"),
        today=date(2026, 6, 1),
    )
    assert derive_account_status(inputs, settings) == "Active"


# ----------------------------------------------------------------------------
# derive_account_status — At-Risk paths
# ----------------------------------------------------------------------------


def test_at_risk_when_at_least_one_low_health_open_deal() -> None:
    inputs = AccountStatusInputs(
        statecode="Active",
        open_deals_count=3,
        open_deals_with_low_health=1,
        last_interaction_date=_ts("2026-05-25"),
        today=date(2026, 6, 1),
    )
    assert derive_account_status(inputs, SETTINGS) == "At-Risk"


def test_active_when_no_low_health_open_deal() -> None:
    inputs = AccountStatusInputs(
        statecode="Active",
        open_deals_count=3,
        open_deals_with_low_health=0,
        last_interaction_date=_ts("2026-05-25"),
        today=date(2026, 6, 1),
    )
    assert derive_account_status(inputs, SETTINGS) == "Active"


def test_inactive_takes_precedence_over_at_risk() -> None:
    """An account that's Inactive in CRM should not be flagged At-Risk."""
    inputs = AccountStatusInputs(
        statecode="Inactive",
        open_deals_with_low_health=5,
    )
    assert derive_account_status(inputs, SETTINGS) == "Inactive"


def test_settings_from_config_uses_defaults_when_blank() -> None:
    settings = AccountStatusSettings.from_config(None)
    assert settings.at_risk_health_threshold == 50
    assert settings.inactive_idle_days == 180


def test_settings_from_config_merges_overrides() -> None:
    settings = AccountStatusSettings.from_config(
        {"at_risk_health_threshold": 40, "inactive_idle_days": 365}
    )
    assert settings.at_risk_health_threshold == 40
    assert settings.inactive_idle_days == 365

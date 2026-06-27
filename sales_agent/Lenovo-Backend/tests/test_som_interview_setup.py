"""Unit tests for Sales Operating Model interview helpers."""

from __future__ import annotations

import pytest

from app.services.sales_operating_model import (
    ERR_MSG_0021,
    SCOPE_LABELS,
    count_captured_responses,
    validate_role,
    verify_enabled,
)


def test_validate_role_accepts_known_roles() -> None:
    assert validate_role("national_manager") == "national_manager"
    assert validate_role("Regional_Manager") == "regional_manager"


def test_validate_role_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="role must be one of"):
        validate_role("executive")


def test_count_captured_responses_ignores_whitespace() -> None:
    responses = {
        "q1": "Growth focus",
        "q2": "   ",
        "q3": "",
        "q4": "x",
    }
    assert count_captured_responses(responses) == 2


def test_verify_enabled_requires_at_least_one() -> None:
    assert verify_enabled(0) is False
    assert verify_enabled(1) is True
    assert verify_enabled(5) is True


def test_scope_labels_cover_all_roles() -> None:
    assert len(SCOPE_LABELS) == 3
    assert "Org-level intent" in SCOPE_LABELS["national_manager"]
    assert "Region-specific" in SCOPE_LABELS["regional_manager"]
    assert "Team-level" in SCOPE_LABELS["seller_manager"]


def test_err_msg_0021_constant() -> None:
    assert ERR_MSG_0021 == "ERR_MSG_0021"

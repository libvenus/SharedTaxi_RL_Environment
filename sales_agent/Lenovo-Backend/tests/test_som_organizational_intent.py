"""Unit tests for organizational intent validation (US 3.2.2)."""

from __future__ import annotations

import pytest

from app.services.som_organizational_intent import (
    CUSTOM_METRICS_KEY,
    ERR_MSG_0022,
    ERR_MSG_0023,
    SUCC_MSG_0017,
    SomFieldValidationError,
    _build_field_preview,
    _card_fields_are_complete,
    _extract_card_fields,
    _merge_stored_fields,
    _normalize_bulk_cards,
    _parse_custom_metrics,
    _resolve_custom_metrics_for_save,
    _split_stored_fields,
    _validate_fields,
    validate_intent_type,
)


def _outcome_fields(**overrides: object) -> dict:
    base = {
        "revenueAndQuality": "70% revenue + 30% margin",
        "predictability": "Forecast within 5%",
        "progressionExpectation": "Advance 2 stages by week 8",
        "riskPosture": "Low discount tolerance",
    }
    base.update(overrides)
    return base


def test_validate_intent_type() -> None:
    assert validate_intent_type("Outcome") == "outcome"
    with pytest.raises(ValueError):
        validate_intent_type("invalid")


def test_outcome_requires_all_fields() -> None:
    with pytest.raises(SomFieldValidationError) as exc:
        _validate_fields("outcome", {"revenueAndQuality": "only one"})
    assert exc.value.code == ERR_MSG_0023


def test_margin_floor_out_of_range() -> None:
    fields = {
        "marginFloors": [{"dealType": "hardware", "minPercent": 150}],
        "complianceGates": "CISO checklist",
        "dealDeskTriggers": ">10% discount",
        "pricingAuthority": "Seller up to 8%",
    }
    with pytest.raises(SomFieldValidationError) as exc:
        _validate_fields("constraint", fields)
    assert exc.value.code == ERR_MSG_0022
    assert exc.value.field == "marginFloors"


def test_tradeoff_requires_priority_rank() -> None:
    fields = {
        "priorityRank": [],
        "revenueVsMargin": "Margin wins",
        "newLogoVsExpansion": "Expansion wins",
        "commitVsUpside": "Commit protected",
    }
    with pytest.raises(SomFieldValidationError) as exc:
        _validate_fields("tradeoff", fields)
    assert exc.value.code == ERR_MSG_0023


def test_focus_requires_expiry_date() -> None:
    fields = {
        "quarterType": "Harvest quarter",
        "priorityFocus": "Strategic accounts",
        "temporaryDeprioritisation": "SMB segment paused",
    }
    with pytest.raises(SomFieldValidationError) as exc:
        _validate_fields("focus", fields)
    assert exc.value.field == "expiryDate"


def test_outcome_valid_payload() -> None:
    cleaned = _validate_fields("outcome", _outcome_fields())
    assert cleaned["revenueAndQuality"].startswith("70%")


def test_field_preview_includes_additional_context() -> None:
    fields = _outcome_fields(
        additionalContext="Mayank-quarter terms preferred over one-off discount wins"
    )
    cleaned = _validate_fields("outcome", fields)
    preview = _build_field_preview(cleaned)
    assert preview["additionalContext"] == cleaned["additionalContext"]
    assert len(preview) == 5


def test_field_preview_omits_blank_optional() -> None:
    preview = _build_field_preview(_outcome_fields())
    assert "additionalContext" not in preview
    assert len(preview) == 4


def test_extract_card_fields_accepts_wrapped_or_flat() -> None:
    wrapped = _extract_card_fields({"fields": _outcome_fields()})
    flat = _extract_card_fields(_outcome_fields())
    assert wrapped == flat


def test_normalize_bulk_cards_rejects_empty() -> None:
    with pytest.raises(ValueError, match="At least one"):
        _normalize_bulk_cards({})


def test_normalize_bulk_cards_validates_intent_type() -> None:
    with pytest.raises(ValueError):
        _normalize_bulk_cards({"not_a_card": {"fields": {}}})


def test_card_complete_when_all_required_present() -> None:
    assert _card_fields_are_complete("outcome", _validate_fields("outcome", _outcome_fields()))


def test_card_incomplete_after_required_field_removed() -> None:
    fields = _validate_fields("outcome", _outcome_fields())
    del fields["revenueAndQuality"]
    assert not _card_fields_are_complete("outcome", fields)


def test_card_still_complete_after_optional_removed() -> None:
    fields = _validate_fields(
        "outcome",
        _outcome_fields(additionalContext="Quarter terms preferred"),
    )
    del fields["additionalContext"]
    assert _card_fields_are_complete("outcome", fields)


def test_succ_msg_constant() -> None:
    assert SUCC_MSG_0017 == "SUCC_MSG_0017"


def test_delete_conflict_is_distinct_error() -> None:
    from app.services.som_organizational_intent import SomDeleteConflictError

    err = SomDeleteConflictError("no config")
    assert isinstance(err, ValueError)


def test_custom_metric_round_trip_in_stored_fields() -> None:
    metrics = [
        {
            "id": "m1",
            "label": "4. Risk Posture",
            "description": (
                "Leadership mode = balanced lean defense for current quarter; "
                "discount tolerance low (max 12% without executive approval)."
            ),
            "sortOrder": 4,
        }
    ]
    merged = _merge_stored_fields(_outcome_fields(), metrics)
    standard, parsed = _split_stored_fields(merged)
    assert CUSTOM_METRICS_KEY not in standard
    assert parsed[0]["label"] == "4. Risk Posture"
    preview = _build_field_preview(standard, parsed)
    assert preview["4. Risk Posture"] == metrics[0]["description"]


def test_resolve_custom_metrics_preserves_existing_on_save() -> None:
    existing = _parse_custom_metrics(
        [{"id": "m1", "label": "4. Risk Posture", "description": "Low discount", "sortOrder": 4}]
    )
    resolved = _resolve_custom_metrics_for_save(_outcome_fields(), existing)
    assert resolved == existing


def test_parse_custom_metrics_sorts_by_order() -> None:
    parsed = _parse_custom_metrics(
        [
            {"id": "b", "label": "B", "description": "two", "sortOrder": 2},
            {"id": "a", "label": "A", "description": "one", "sortOrder": 1},
        ]
    )
    assert [m["label"] for m in parsed] == ["A", "B"]

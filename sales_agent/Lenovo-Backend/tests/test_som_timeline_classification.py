"""Unit tests for timeline classification validation (US 3.4.1)."""

from __future__ import annotations

import copy

import pytest

from app.services.som_timeline_classification import (
    DEFAULT_FIELDS,
    ERR_MSG_0025,
    ERR_MSG_0026,
    ERR_MSG_0028,
    INFO_MSG_0006,
    SUCC_MSG_0018,
    SomFieldValidationError,
    _validate_fields,
    validate_card_type,
)


def test_validate_card_type() -> None:
    assert validate_card_type("Tempo_Classes") == "tempo_classes"
    with pytest.raises(ValueError):
        validate_card_type("invalid")


def test_tempo_classes_valid_defaults() -> None:
    cleaned = _validate_fields("tempo_classes", DEFAULT_FIELDS["tempo_classes"])
    assert len(cleaned["classesDeclared"]) == 4
    assert "Quarterly/Enterprise" in cleaned["defaultTempoClassRule"]


def test_anchor_definitions_requires_pause_events() -> None:
    fields = copy.deepcopy(DEFAULT_FIELDS["anchor_definitions"])
    fields["clockPauseEvents"] = []
    with pytest.raises(SomFieldValidationError) as exc:
        _validate_fields("anchor_definitions", fields)
    assert exc.value.code == ERR_MSG_0028


def test_signal_band_requires_silence_days() -> None:
    fields = copy.deepcopy(DEFAULT_FIELDS["signal_expectations_time_band"])
    fields["band0to30"]["acceptableSilenceDays"] = 0
    with pytest.raises(SomFieldValidationError) as exc:
        _validate_fields("signal_expectations_time_band", fields)
    assert exc.value.code == ERR_MSG_0025


def test_acceleration_decay_threshold_positive() -> None:
    fields = copy.deepcopy(DEFAULT_FIELDS["acceleration_decay"])
    fields["decayReviewThresholds"]["fastDays"] = -1
    with pytest.raises(SomFieldValidationError) as exc:
        _validate_fields("acceleration_decay", fields)
    assert exc.value.code == ERR_MSG_0025


def test_exit_recycle_cadence_positive() -> None:
    fields = copy.deepcopy(DEFAULT_FIELDS["exit_recycle_kill"])
    fields["evidenceNoteCadenceDays"] = 0
    with pytest.raises(SomFieldValidationError) as exc:
        _validate_fields("exit_recycle_kill", fields)
    assert exc.value.code == ERR_MSG_0026


def test_canonical_timeline_valid() -> None:
    cleaned = _validate_fields(
        "canonical_timeline", DEFAULT_FIELDS["canonical_timeline"]
    )
    assert cleaned["week1to4"]["dominantActivity"] == "Pipeline creation"
    assert cleaned["q4"]["quarterCharacter"].startswith("Closure")


def test_message_constants() -> None:
    assert SUCC_MSG_0018 == "SUCC_MSG_0018"
    assert INFO_MSG_0006 == "INFO_MSG_0006"

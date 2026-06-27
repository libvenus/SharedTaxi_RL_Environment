"""Unit tests for briefing context helpers (no database)."""

from __future__ import annotations

from app.services.briefing_context import _trim_words, _word_count, INF_MSG_0004


def test_word_count_basic() -> None:
    assert _word_count("one two three") == 3


def test_trim_words_respects_limit() -> None:
    text = " ".join(f"word{i}" for i in range(20))
    trimmed = _trim_words(text, 10)
    assert _word_count(trimmed) == 10
    assert trimmed.endswith("…")


def test_inf_msg_0004_constant() -> None:
    assert INF_MSG_0004 == "INF_MSG_0004"

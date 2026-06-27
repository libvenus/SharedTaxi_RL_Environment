"""Unit tests for app/services/what_changed.py."""

from __future__ import annotations

from datetime import datetime, timezone

from app.services.what_changed import (
    activity_category_label,
    build_activity_titles,
    build_crm_update_titles,
    classify_activity,
    explode_audit_diff,
    feed_key_activity,
    feed_key_audit,
    is_seller_actor,
    normalize_direction,
    normalize_event_at,
    paginate_feed,
    parse_activity_type_filter,
)
from app.schemas import WhatChangedItem


def test_classify_inbound_email() -> None:
    assert classify_activity("email", "inbound") == "email"


def test_classify_outbound_email_excluded() -> None:
    assert classify_activity("email", "outbound") is None


def test_classify_meeting() -> None:
    assert classify_activity("meeting", None) == "meeting"


def test_is_seller_actor_matches_seller_uuid() -> None:
    seller = "A1B2C3D4-0000-0000-0000-000000000001"
    assert is_seller_actor(seller, seller) is True
    assert is_seller_actor(seller.lower(), seller) is True


def test_is_seller_actor_matches_viewer_header() -> None:
    assert is_seller_actor(
        "seller@lenovo.com",
        "OTHER-UUID",
        viewer_id="seller@lenovo.com",
    )


def test_explode_audit_diff_stage_change() -> None:
    diff = (
        '{"before": {"stagename": "Qualify"}, '
        '"after": {"stagename": "Develop"}}'
    )
    fields = explode_audit_diff(diff)
    assert fields == [("stagename", "'Qualify' → 'Develop'")]


def test_build_crm_update_titles_stage() -> None:
    title, summary = build_crm_update_titles("stagename", "Qualify → Develop")
    assert title == "Stage changed"
    assert summary == "Qualify → Develop"


def test_build_activity_titles_email() -> None:
    title, summary = build_activity_titles("email", "Re: pricing", "Please review")
    assert title == "Re: pricing"
    assert summary == "Please review"


def test_feed_keys_are_stable() -> None:
    assert feed_key_activity("abc") == "activity:abc"
    assert feed_key_audit("log1", "stagename") == "audit:log1:stagename"


def test_parse_activity_type_filter_aliases() -> None:
    parsed = parse_activity_type_filter("email, crm, risk")
    assert parsed == {"email", "crm_update", "risk"}


def test_normalize_direction() -> None:
    assert normalize_direction("Inbound") == "inbound"
    assert normalize_direction("outbound") == "outbound"
    assert normalize_direction(None) is None


def test_normalize_event_at_strips_tz_for_sort() -> None:
    aware = datetime(2026, 6, 17, 10, 0, tzinfo=timezone.utc)
    naive = datetime(2026, 6, 17, 9, 0)
    assert normalize_event_at(aware) == datetime(2026, 6, 17, 10, 0)
    # naive + aware normalised to naive — sort must not raise
    items = sorted([normalize_event_at(aware), normalize_event_at(naive)], reverse=True)
    assert items[0] == datetime(2026, 6, 17, 10, 0)


def test_activity_category_label_stage() -> None:
    assert (
        activity_category_label("crm_update", changed_field="stagename")
        == "Stage progression tracked"
    )


def test_activity_category_label_email() -> None:
    assert activity_category_label("email") == "Email received"


def test_paginate_feed_second_page() -> None:
    items = [
        WhatChangedItem(
            id=f"activity:{i}",
            activity_type="email",
            title="t",
            summary="s",
            opportunity_id="opp",
            event_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            link_type="outreach",
            link_id="opp",
            direction="inbound",
            category_label="Email received",
        )
        for i in range(5)
    ]
    page_items, total, total_pages = paginate_feed(items, page=2, page_size=2)
    assert total == 5
    assert total_pages == 3
    assert len(page_items) == 2
    assert page_items[0].id == "activity:2"

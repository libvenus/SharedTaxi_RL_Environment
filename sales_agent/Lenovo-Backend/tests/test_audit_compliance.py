"""Unit tests for compliance audit helpers."""

from app.services.audit_log import (
    build_field_changes,
    infer_actor_type_from_path,
    should_log_read,
)
from app.services.audit_log import AuditConfigSnapshot


def test_build_field_changes_detects_differences() -> None:
    changes = build_field_changes(
        {"estimatedvalue": "100", "stagename": "Qualify"},
        {"estimatedvalue": "200", "stagename": "Qualify"},
    )
    assert len(changes) == 1
    assert changes[0]["field"] == "estimatedvalue"
    assert changes[0]["before"] == "100"
    assert changes[0]["after"] == "200"


def test_infer_actor_type_admin_for_som() -> None:
    assert infer_actor_type_from_path("/api/sales-operating-model/outcome", None) == "admin"


def test_should_log_read_respects_config() -> None:
    cfg = AuditConfigSnapshot(
        retention_days=90,
        log_seller_reads=True,
        log_admin_reads=False,
        log_ai_output_reads=False,
        updated_at=None,
        updated_by=None,
    )
    assert should_log_read(cfg, actor_type="seller", path="/api/opportunities/abc")
    assert not should_log_read(cfg, actor_type="admin", path="/api/opportunities/abc")

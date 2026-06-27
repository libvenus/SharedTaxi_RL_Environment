"""Sales Operating Model — Organizational Intent cards (US 3.2.2)."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Literal

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from app.models import SomOrganizationalIntent
from app.services.audit_log import write_audit_event
from app.services.som_context_lake import rebuild_context_lake

logger = logging.getLogger(__name__)

ERR_MSG_0022 = "ERR_MSG_0022"
ERR_MSG_0023 = "ERR_MSG_0023"
ERR_MSG_0024 = "ERR_MSG_0024"
SUCC_MSG_0017 = "SUCC_MSG_0017"

OrgIntentType = Literal[
    "outcome", "motion", "focus", "behavioral", "constraint", "tradeoff"
]

VALID_INTENT_TYPES: frozenset[str] = frozenset(
    {"outcome", "motion", "focus", "behavioral", "constraint", "tradeoff"}
)

INTENT_ORDER: tuple[str, ...] = (
    "outcome",
    "motion",
    "focus",
    "behavioral",
    "constraint",
    "tradeoff",
)

# field_key -> display label for ERR_MSG_0023
FIELD_LABELS: dict[str, dict[str, str]] = {
    "outcome": {
        "revenueAndQuality": "Revenue & quality",
        "predictability": "Predictability",
        "progressionExpectation": "Progression expectation",
        "riskPosture": "Risk posture",
    },
    "motion": {
        "primaryGrowthLever": "Primary growth lever",
        "sellingMotionMix": "Selling motion mix",
        "routeToMarket": "Route-to-market",
        "salesCyclePolicy": "Sales cycle policy",
        "attachExpectation": "Attach expectation",
    },
    "focus": {
        "quarterType": "Quarter type",
        "priorityFocus": "Priority focus",
        "temporaryDeprioritisation": "Temporary deprioritisation",
        "expiryDate": "Expiry controls",
    },
    "behavioral": {
        "multithreadingNorm": "Multithreading norm",
        "followUpCadence": "Follow-up cadence",
        "walkAwayRule": "Walk-away rule",
        "coachingLens": "Coaching lens",
    },
    "constraint": {
        "marginFloors": "Margin floors",
        "complianceGates": "Compliance gates",
        "dealDeskTriggers": "Deal desk triggers",
        "pricingAuthority": "Pricing authority",
    },
    "tradeoff": {
        "priorityRank": "Trade-off priority rank",
        "revenueVsMargin": "Revenue vs margin",
        "newLogoVsExpansion": "New logo vs expansion",
        "commitVsUpside": "Commit vs upside",
    },
}

OPTIONAL_FIELDS = frozenset({"additionalContext"})
CUSTOM_METRICS_KEY = "customMetrics"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class SomFieldValidationError(ValueError):
    def __init__(
        self,
        code: str,
        field: str,
        message: str,
        *,
        intent_type: str | None = None,
    ) -> None:
        self.code = code
        self.field = field
        self.message = message
        self.intent_type = intent_type
        super().__init__(message)


@dataclass(frozen=True)
class OrgIntentSummary:
    intent_type: str
    display_name: str
    status: str
    is_timeboxed: bool
    is_guardrail: bool
    last_synced_at: datetime | None
    field_preview: dict[str, Any]


@dataclass(frozen=True)
class OrgIntentDetail:
    intent_type: str
    display_name: str
    status: str
    is_timeboxed: bool
    is_guardrail: bool
    guardrail_warning: str | None
    last_synced_at: datetime | None
    expiry_date: date | None
    fields: dict[str, Any]
    field_labels: dict[str, str]
    custom_metrics: tuple["OrgIntentCustomMetric", ...]


@dataclass(frozen=True)
class OrgIntentCustomMetric:
    id: str
    label: str
    description: str
    sort_order: int


@dataclass(frozen=True)
class OrgIntentSaveResult:
    intent_type: str
    display_name: str
    status: str
    is_timeboxed: bool
    is_guardrail: bool
    last_synced_at: datetime
    field_preview: dict[str, Any]
    all_configured: bool
    success_code: str | None


@dataclass(frozen=True)
class OrgIntentBulkSaveResult:
    items: tuple[OrgIntentSaveResult, ...]
    all_configured: bool
    configured_count: int
    total_count: int
    success_code: str | None


@dataclass(frozen=True)
class ConfigurationStatus:
    all_configured: bool
    configured_count: int
    total_count: int
    success_code: str | None


@dataclass(frozen=True)
class OrgIntentDeleteResult:
    intent_type: str
    display_name: str
    status: str
    deleted_at: datetime
    configured_count: int
    total_count: int


@dataclass(frozen=True)
class OrgIntentFieldDeleteResult:
    intent_type: str
    display_name: str
    status: str
    deleted_field: str
    is_timeboxed: bool
    is_guardrail: bool
    last_synced_at: datetime | None
    fields: dict[str, Any]
    custom_metrics: tuple[OrgIntentCustomMetric, ...]
    field_preview: dict[str, Any]
    all_configured: bool
    configured_count: int
    total_count: int
    success_code: str | None


class SomDeleteConflictError(ValueError):
    """Raised when delete is requested on a card that is not CONFIGURED."""


class SomFieldNotFoundError(LookupError):
    """Raised when the requested field key is not present on the card."""

    def __init__(
        self,
        intent_type: str,
        field_key: str,
        available_fields: list[str],
    ) -> None:
        self.intent_type = intent_type
        self.field_key = field_key
        self.available_fields = available_fields
        keys = ", ".join(available_fields) if available_fields else "(none)"
        super().__init__(
            f"Field '{field_key}' is not set on intent card '{intent_type}'. "
            f"Saved fields: {keys}."
        )


class SomMetricNotFoundError(LookupError):
    def __init__(self, intent_type: str, metric_id: str) -> None:
        self.intent_type = intent_type
        self.metric_id = metric_id
        super().__init__(
            f"Custom metric '{metric_id}' not found on intent card '{intent_type}'."
        )


def validate_intent_type(intent_type: str) -> OrgIntentType:
    normalized = intent_type.strip().lower()
    if normalized not in VALID_INTENT_TYPES:
        raise ValueError(
            f"intentType must be one of: {', '.join(INTENT_ORDER)}"
        )
    return normalized  # type: ignore[return-value]


def _has_org_table(db: Session) -> bool:
    return inspect(db.bind).has_table("lvo_som_organizational_intent")


def _require_row(db: Session, intent_type: str) -> SomOrganizationalIntent:
    row = db.get(SomOrganizationalIntent, intent_type)
    if row is None:
        raise LookupError(f"Organizational intent not found: {intent_type}")
    return row


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    return False


def _validate_margin_floors(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise SomFieldValidationError(
            ERR_MSG_0023,
            "marginFloors",
            "Margin floors is required.",
        )
    for item in value:
        if not isinstance(item, dict):
            raise SomFieldValidationError(
                ERR_MSG_0022,
                "marginFloors",
                "Margin floor must be a valid percentage between 0 and 100.",
            )
        pct = item.get("minPercent")
        if pct is None:
            raise SomFieldValidationError(
                ERR_MSG_0023,
                "marginFloors",
                "Margin floors is required.",
            )
        try:
            pct_f = float(pct)
        except (TypeError, ValueError) as exc:
            raise SomFieldValidationError(
                ERR_MSG_0022,
                "marginFloors",
                "Margin floor must be a valid percentage between 0 and 100.",
            ) from exc
        if pct_f < 0 or pct_f > 100:
            raise SomFieldValidationError(
                ERR_MSG_0022,
                "marginFloors",
                "Margin floor must be a valid percentage between 0 and 100.",
            )


def _validate_pricing_authority(value: Any) -> None:
    if isinstance(value, str):
        if not value.strip():
            raise SomFieldValidationError(
                ERR_MSG_0023,
                "pricingAuthority",
                "Pricing authority is required.",
            )
        return
    if not isinstance(value, list) or not value:
        raise SomFieldValidationError(
            ERR_MSG_0023,
            "pricingAuthority",
            "Pricing authority is required.",
        )
    for item in value:
        if isinstance(item, dict) and "maxDiscountPercent" in item:
            try:
                pct_f = float(item["maxDiscountPercent"])
            except (TypeError, ValueError) as exc:
                raise SomFieldValidationError(
                    ERR_MSG_0022,
                    "pricingAuthority",
                    "Pricing authority discount must be between 0 and 100.",
                ) from exc
            if pct_f < 0 or pct_f > 100:
                raise SomFieldValidationError(
                    ERR_MSG_0022,
                    "pricingAuthority",
                    "Pricing authority discount must be between 0 and 100.",
                )


def _validation_error(
    code: str,
    field: str,
    message: str,
    *,
    intent_type: str | None = None,
) -> SomFieldValidationError:
    return SomFieldValidationError(
        code, field, message, intent_type=intent_type
    )


def _validate_fields(intent_type: str, fields: dict[str, Any]) -> dict[str, Any]:
    labels = FIELD_LABELS[intent_type]
    cleaned: dict[str, Any] = {}

    for key, label in labels.items():
        raw = fields.get(key)
        if key == "marginFloors":
            try:
                _validate_margin_floors(raw)
            except SomFieldValidationError as exc:
                raise _validation_error(
                    exc.code, exc.field, exc.message, intent_type=intent_type
                ) from exc
            cleaned[key] = raw
            continue
        if key == "pricingAuthority":
            try:
                _validate_pricing_authority(raw)
            except SomFieldValidationError as exc:
                raise _validation_error(
                    exc.code, exc.field, exc.message, intent_type=intent_type
                ) from exc
            cleaned[key] = raw
            continue
        if key == "priorityRank":
            if not isinstance(raw, list) or len(raw) == 0:
                raise _validation_error(
                    ERR_MSG_0023,
                    key,
                    f"{label} is required.",
                    intent_type=intent_type,
                )
            cleaned[key] = [str(x).strip() for x in raw if str(x).strip()]
            if not cleaned[key]:
                raise _validation_error(
                    ERR_MSG_0023,
                    key,
                    f"{label} is required.",
                    intent_type=intent_type,
                )
            continue
        if key == "expiryDate":
            if _is_blank(raw):
                raise _validation_error(
                    ERR_MSG_0023,
                    key,
                    f"{label} is required.",
                    intent_type=intent_type,
                )
            if isinstance(raw, date):
                cleaned[key] = raw.isoformat()
            else:
                cleaned[key] = str(raw).strip()
            continue
        if _is_blank(raw):
            raise _validation_error(
                ERR_MSG_0023,
                key,
                f"{label} is required.",
                intent_type=intent_type,
            )
        cleaned[key] = str(raw).strip() if not isinstance(raw, (list, dict)) else raw

    extra = fields.get("additionalContext")
    if extra is not None and str(extra).strip():
        cleaned["additionalContext"] = str(extra).strip()

    return cleaned


def _parse_expiry(fields: dict[str, Any]) -> date | None:
    raw = fields.get("expiryDate")
    if not raw:
        return None
    if isinstance(raw, date):
        return raw
    return date.fromisoformat(str(raw)[:10])


def _parse_custom_metrics(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    parsed: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        metric_id = str(item.get("id") or "").strip()
        label = str(item.get("label") or "").strip()
        if not metric_id or not label:
            continue
        parsed.append(
            {
                "id": metric_id,
                "label": label,
                "description": str(item.get("description") or "").strip(),
                "sortOrder": int(item.get("sortOrder") or 0),
            }
        )
    return sorted(parsed, key=lambda m: (m["sortOrder"], m["label"].lower()))


def _split_stored_fields(raw: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = dict(raw or {})
    custom = _parse_custom_metrics(data.pop(CUSTOM_METRICS_KEY, None))
    return data, custom


def _merge_stored_fields(
    standard: dict[str, Any], custom: list[dict[str, Any]]
) -> dict[str, Any]:
    merged = dict(standard)
    if custom:
        merged[CUSTOM_METRICS_KEY] = custom
    return merged


def _custom_metrics_to_dataclass(
    metrics: list[dict[str, Any]],
) -> tuple[OrgIntentCustomMetric, ...]:
    return tuple(
        OrgIntentCustomMetric(
            id=m["id"],
            label=m["label"],
            description=m["description"],
            sort_order=int(m["sortOrder"]),
        )
        for m in metrics
    )


def _resolve_custom_metrics_for_save(
    fields: dict[str, Any], existing: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if CUSTOM_METRICS_KEY in fields:
        return _parse_custom_metrics(fields.get(CUSTOM_METRICS_KEY))
    return list(existing)


def _apply_row_field_state(
    row: SomOrganizationalIntent,
    intent_type: str,
    standard_fields: dict[str, Any],
    custom_metrics: list[dict[str, Any]],
    *,
    saved_by: str | None,
    now: datetime,
) -> None:
    row.lvo_fields = _merge_stored_fields(standard_fields, custom_metrics)
    if standard_fields and _card_fields_are_complete(intent_type, standard_fields):
        row.lvo_status = "CONFIGURED"
    else:
        row.lvo_status = "NOT_CONFIGURED"
    row.lvo_last_synced_at = now
    row.lvo_configured_by = saved_by
    if intent_type == "focus" and standard_fields:
        row.lvo_expiry_date = _parse_expiry(standard_fields)
    elif intent_type == "focus" and not standard_fields:
        row.lvo_expiry_date = None
    elif intent_type != "focus":
        row.lvo_expiry_date = None


def _build_field_preview(
    standard_fields: dict[str, Any],
    custom_metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Grid subtitle map — standard fields plus custom metric labels."""
    preview = {k: v for k, v in standard_fields.items() if not _is_blank(v)}
    for metric in custom_metrics or []:
        if metric.get("description"):
            preview[metric["label"]] = metric["description"]
    return preview


def _card_fields_are_complete(intent_type: str, fields: dict[str, Any]) -> bool:
    """True when every required field for the card type is present and valid."""
    try:
        _validate_fields(intent_type, fields)
        return True
    except SomFieldValidationError:
        return False


def _normalize_field_key(field_key: str) -> str:
    return field_key.strip()


def list_organizational_intents(db: Session) -> list[OrgIntentSummary]:
    if not _has_org_table(db):
        raise RuntimeError("Organizational intent tables are not migrated.")

    by_type = {
        row.lvo_intenttype: row
        for row in db.execute(select(SomOrganizationalIntent)).scalars()
    }

    summaries: list[OrgIntentSummary] = []
    for intent_type in INTENT_ORDER:
        row = by_type.get(intent_type)
        if row is None:
            continue
        fields = dict(row.lvo_fields or {})
        standard, custom = _split_stored_fields(fields)
        preview = _build_field_preview(standard, custom)
        summaries.append(
            OrgIntentSummary(
                intent_type=row.lvo_intenttype,
                display_name=row.lvo_displayname,
                status=row.lvo_status,
                is_timeboxed=row.lvo_is_timeboxed,
                is_guardrail=row.lvo_is_guardrail,
                last_synced_at=row.lvo_last_synced_at,
                field_preview=preview,
            )
        )
    return summaries


def get_organizational_intent(db: Session, intent_type: str) -> OrgIntentDetail:
    intent_type = validate_intent_type(intent_type)
    if not _has_org_table(db):
        raise RuntimeError("Organizational intent tables are not migrated.")

    row = _require_row(db, intent_type)
    guardrail_warning = None
    if row.lvo_is_guardrail:
        guardrail_warning = (
            "Values configured here are non-overridable by AI agents "
            "regardless of deal context."
        )

    standard, custom = _split_stored_fields(dict(row.lvo_fields or {}))
    return OrgIntentDetail(
        intent_type=row.lvo_intenttype,
        display_name=row.lvo_displayname,
        status=row.lvo_status,
        is_timeboxed=row.lvo_is_timeboxed,
        is_guardrail=row.lvo_is_guardrail,
        guardrail_warning=guardrail_warning,
        last_synced_at=row.lvo_last_synced_at,
        expiry_date=row.lvo_expiry_date,
        fields=standard,
        field_labels=dict(FIELD_LABELS.get(intent_type, {})),
        custom_metrics=_custom_metrics_to_dataclass(custom),
    )


def _extract_card_fields(payload: Any) -> dict[str, Any]:
    """Accept `{ fields: {...} }` or a flat field map per card."""
    if not isinstance(payload, dict):
        raise ValueError("Each card entry must be an object.")
    if "fields" in payload:
        inner = payload["fields"]
        if not isinstance(inner, dict):
            raise ValueError("Card fields must be an object.")
        return dict(inner)
    return dict(payload)


def _normalize_bulk_cards(
    cards: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    if not cards:
        raise ValueError("At least one intent card must be provided in cards.")
    normalized: list[tuple[str, dict[str, Any]]] = []
    for raw_type, payload in cards.items():
        intent_type = validate_intent_type(raw_type)
        normalized.append((intent_type, _extract_card_fields(payload)))
    return normalized


def _stage_intent_save(
    db: Session,
    intent_type: str,
    fields: dict[str, Any],
) -> tuple[
    SomOrganizationalIntent,
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    intent_type = validate_intent_type(intent_type)
    row = _require_row(db, intent_type)
    before_standard, before_custom = _split_stored_fields(dict(row.lvo_fields or {}))
    cleaned = _validate_fields(intent_type, fields)
    custom = _resolve_custom_metrics_for_save(fields, before_custom)
    return row, cleaned, before_standard, before_custom, custom


def _apply_intent_save(
    row: SomOrganizationalIntent,
    intent_type: str,
    standard_fields: dict[str, Any],
    custom_metrics: list[dict[str, Any]],
    *,
    saved_by: str | None,
    now: datetime,
) -> None:
    _apply_row_field_state(
        row,
        intent_type,
        standard_fields,
        custom_metrics,
        saved_by=saved_by,
        now=now,
    )


def _save_result_for_row(
    db: Session,
    intent_type: str,
    *,
    status: ConfigurationStatus,
) -> OrgIntentSaveResult:
    row = _require_row(db, intent_type)
    saved_standard, saved_custom = _split_stored_fields(dict(row.lvo_fields or {}))
    return OrgIntentSaveResult(
        intent_type=intent_type,
        display_name=row.lvo_displayname,
        status=row.lvo_status,
        is_timeboxed=row.lvo_is_timeboxed,
        is_guardrail=row.lvo_is_guardrail,
        last_synced_at=row.lvo_last_synced_at or _utc_now(),
        field_preview=_build_field_preview(saved_standard, saved_custom),
        all_configured=status.all_configured,
        success_code=status.success_code,
    )


def _persist_intent_saves(
    db: Session,
    staged: list[
        tuple[
            str,
            SomOrganizationalIntent,
            dict[str, Any],
            dict[str, Any],
            list[dict[str, Any]],
            list[dict[str, Any]],
        ]
    ],
    *,
    saved_by: str | None,
    now: datetime,
) -> None:
    for intent_type, row, cleaned, before_standard, before_custom, custom in staged:
        _apply_intent_save(
            row, intent_type, cleaned, custom, saved_by=saved_by, now=now
        )

    rebuild_context_lake(db, saved_by)

    for intent_type, _row, cleaned, before_standard, before_custom, custom in staged:
        write_audit_event(
            db,
            entity_type="som_organizational_intent",
            entity_id=intent_type,
            action="update",
            category="admin_action",
            actor_type="admin",
            changed_by=saved_by,
            diff={
                "before": _merge_stored_fields(before_standard, before_custom),
                "after": _merge_stored_fields(cleaned, custom),
            },
        )

    db.commit()


def save_organizational_intent(
    db: Session,
    intent_type: str,
    fields: dict[str, Any],
    saved_by: str | None,
) -> OrgIntentSaveResult:
    intent_type = validate_intent_type(intent_type)
    if not _has_org_table(db):
        raise RuntimeError("Organizational intent tables are not migrated.")

    try:
        now = _utc_now()
        row, cleaned, before_standard, before_custom, custom = _stage_intent_save(
            db, intent_type, fields
        )
        _persist_intent_saves(
            db,
            [(intent_type, row, cleaned, before_standard, before_custom, custom)],
            saved_by=saved_by,
            now=now,
        )
    except SomFieldValidationError:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        logger.exception("Organizational intent save failed for %s", intent_type)
        raise RuntimeError(ERR_MSG_0024) from None

    status = get_configuration_status(db)
    return _save_result_for_row(db, intent_type, status=status)


def save_organizational_intents_bulk(
    db: Session,
    cards: dict[str, Any],
    saved_by: str | None,
) -> OrgIntentBulkSaveResult:
    """Save one or more intent cards atomically (single Context Lake rebuild)."""
    if not _has_org_table(db):
        raise RuntimeError("Organizational intent tables are not migrated.")

    try:
        normalized = _normalize_bulk_cards(cards)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    saved_types = {intent_type for intent_type, _ in normalized}
    try:
        now = _utc_now()
        staged: list[
            tuple[
                str,
                SomOrganizationalIntent,
                dict[str, Any],
                dict[str, Any],
                list[dict[str, Any]],
                list[dict[str, Any]],
            ]
        ] = []
        for intent_type, fields in normalized:
            row, cleaned, before_standard, before_custom, custom = _stage_intent_save(
                db, intent_type, fields
            )
            staged.append(
                (intent_type, row, cleaned, before_standard, before_custom, custom)
            )

        _persist_intent_saves(db, staged, saved_by=saved_by, now=now)
    except SomFieldValidationError:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        logger.exception("Organizational intent bulk save failed")
        raise RuntimeError(ERR_MSG_0024) from None

    status = get_configuration_status(db)
    items = tuple(
        _save_result_for_row(db, intent_type, status=status)
        for intent_type in INTENT_ORDER
        if intent_type in saved_types
    )
    return OrgIntentBulkSaveResult(
        items=items,
        all_configured=status.all_configured,
        configured_count=status.configured_count,
        total_count=status.total_count,
        success_code=status.success_code,
    )


def get_configuration_status(db: Session) -> ConfigurationStatus:
    if not _has_org_table(db):
        raise RuntimeError("Organizational intent tables are not migrated.")

    rows = db.execute(select(SomOrganizationalIntent)).scalars().all()
    by_type = {r.lvo_intenttype: r for r in rows}
    configured = sum(
        1
        for t in INTENT_ORDER
        if (by_type.get(t) and by_type[t].lvo_status == "CONFIGURED")
    )
    total = len(INTENT_ORDER)
    all_ok = configured == total
    return ConfigurationStatus(
        all_configured=all_ok,
        configured_count=configured,
        total_count=total,
        success_code=SUCC_MSG_0017 if all_ok else None,
    )


def delete_organizational_intent(
    db: Session,
    intent_type: str,
    deleted_by: str | None,
) -> OrgIntentDeleteResult:
    """Clear a CONFIGURED intent card and remove it from the Context Lake."""
    intent_type = validate_intent_type(intent_type)
    if not _has_org_table(db):
        raise RuntimeError("Organizational intent tables are not migrated.")

    row = _require_row(db, intent_type)
    if row.lvo_status != "CONFIGURED":
        raise SomDeleteConflictError(
            f"Intent card '{intent_type}' has no saved configuration to delete."
        )

    try:
        now = _utc_now()
        before_fields = dict(row.lvo_fields or {})
        row.lvo_fields = {}
        row.lvo_status = "NOT_CONFIGURED"
        row.lvo_last_synced_at = None
        row.lvo_configured_by = None
        row.lvo_expiry_date = None

        rebuild_context_lake(db, deleted_by)
        write_audit_event(
            db,
            entity_type="som_organizational_intent",
            entity_id=intent_type,
            action="delete",
            category="admin_action",
            actor_type="admin",
            changed_by=deleted_by,
            diff={"before": before_fields, "after": {}},
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Organizational intent delete failed for %s", intent_type)
        raise RuntimeError(ERR_MSG_0024) from None

    status = get_configuration_status(db)
    row = _require_row(db, intent_type)
    return OrgIntentDeleteResult(
        intent_type=intent_type,
        display_name=row.lvo_displayname,
        status=row.lvo_status,
        deleted_at=now,
        configured_count=status.configured_count,
        total_count=status.total_count,
    )


def delete_organizational_intent_field(
    db: Session,
    intent_type: str,
    field_key: str,
    deleted_by: str | None,
) -> OrgIntentFieldDeleteResult:
    """Remove one field row from a card (e.g. additionalContext or a required metric).

    Required metrics can be removed; if the card is no longer complete it
    reverts to ``NOT_CONFIGURED`` while keeping the remaining field values.
    """
    intent_type = validate_intent_type(intent_type)
    field_key = _normalize_field_key(field_key)
    if not field_key:
        raise ValueError("fieldKey is required.")

    if not _has_org_table(db):
        raise RuntimeError("Organizational intent tables are not migrated.")

    row = _require_row(db, intent_type)
    before_standard, before_custom = _split_stored_fields(dict(row.lvo_fields or {}))
    if field_key not in before_standard:
        raise SomFieldNotFoundError(
            intent_type,
            field_key,
            sorted(before_standard.keys()),
        )

    try:
        now = _utc_now()
        after_standard = dict(before_standard)
        del after_standard[field_key]

        _apply_row_field_state(
            row,
            intent_type,
            after_standard,
            before_custom,
            saved_by=deleted_by,
            now=now,
        )

        rebuild_context_lake(db, deleted_by)
        write_audit_event(
            db,
            entity_type="som_organizational_intent",
            entity_id=intent_type,
            action="update",
            category="admin_action",
            actor_type="admin",
            changed_by=deleted_by,
            diff={
                "before": _merge_stored_fields(before_standard, before_custom),
                "after": _merge_stored_fields(after_standard, before_custom),
                "deletedField": field_key,
            },
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception(
            "Organizational intent field delete failed for %s.%s",
            intent_type,
            field_key,
        )
        raise RuntimeError(ERR_MSG_0024) from None

    status = get_configuration_status(db)
    row = _require_row(db, intent_type)
    remaining_standard, remaining_custom = _split_stored_fields(dict(row.lvo_fields or {}))
    return OrgIntentFieldDeleteResult(
        intent_type=intent_type,
        display_name=row.lvo_displayname,
        status=row.lvo_status,
        deleted_field=field_key,
        is_timeboxed=row.lvo_is_timeboxed,
        is_guardrail=row.lvo_is_guardrail,
        last_synced_at=row.lvo_last_synced_at,
        fields=remaining_standard,
        custom_metrics=_custom_metrics_to_dataclass(remaining_custom),
        field_preview=_build_field_preview(remaining_standard, remaining_custom),
        all_configured=status.all_configured,
        configured_count=status.configured_count,
        total_count=status.total_count,
        success_code=status.success_code,
    )


def _commit_metric_mutation(
    db: Session,
    row: SomOrganizationalIntent,
    intent_type: str,
    before_standard: dict[str, Any],
    before_custom: list[dict[str, Any]],
    after_standard: dict[str, Any],
    after_custom: list[dict[str, Any]],
    *,
    saved_by: str | None,
    now: datetime,
    audit_extra: dict[str, Any] | None = None,
) -> None:
    _apply_row_field_state(
        row,
        intent_type,
        after_standard,
        after_custom,
        saved_by=saved_by,
        now=now,
    )
    rebuild_context_lake(db, saved_by)
    diff: dict[str, Any] = {
        "before": _merge_stored_fields(before_standard, before_custom),
        "after": _merge_stored_fields(after_standard, after_custom),
    }
    if audit_extra:
        diff.update(audit_extra)
    write_audit_event(
        db,
        entity_type="som_organizational_intent",
        entity_id=intent_type,
        action="update",
        category="admin_action",
        actor_type="admin",
        changed_by=saved_by,
        diff=diff,
    )
    db.commit()


def add_organizational_intent_metric(
    db: Session,
    intent_type: str,
    label: str,
    description: str,
    sort_order: int | None,
    saved_by: str | None,
) -> OrgIntentCustomMetric:
    """Add a custom metric row (label + description), like a new interview question."""
    intent_type = validate_intent_type(intent_type)
    label = label.strip()
    description = description.strip()
    if not label:
        raise ValueError("label is required.")
    if not description:
        raise ValueError("description is required.")
    if not _has_org_table(db):
        raise RuntimeError("Organizational intent tables are not migrated.")

    row = _require_row(db, intent_type)
    before_standard, before_custom = _split_stored_fields(dict(row.lvo_fields or {}))
    if sort_order is None:
        base = len(FIELD_LABELS[intent_type])
        sort_order = max((m["sortOrder"] for m in before_custom), default=base) + 1

    metric = {
        "id": str(uuid.uuid4()),
        "label": label,
        "description": description,
        "sortOrder": int(sort_order),
    }
    after_custom = _parse_custom_metrics(before_custom + [metric])

    try:
        now = _utc_now()
        _commit_metric_mutation(
            db,
            row,
            intent_type,
            before_standard,
            before_custom,
            before_standard,
            after_custom,
            saved_by=saved_by,
            now=now,
            audit_extra={"addedMetric": metric},
        )
    except Exception:
        db.rollback()
        logger.exception("Add organizational intent metric failed for %s", intent_type)
        raise RuntimeError(ERR_MSG_0024) from None

    return OrgIntentCustomMetric(
        id=metric["id"],
        label=metric["label"],
        description=metric["description"],
        sort_order=metric["sortOrder"],
    )


def update_organizational_intent_metric(
    db: Session,
    intent_type: str,
    metric_id: str,
    *,
    label: str | None,
    description: str | None,
    sort_order: int | None,
    saved_by: str | None,
) -> OrgIntentCustomMetric:
    intent_type = validate_intent_type(intent_type)
    metric_id = metric_id.strip()
    if not metric_id:
        raise ValueError("metricId is required.")
    if not _has_org_table(db):
        raise RuntimeError("Organizational intent tables are not migrated.")

    row = _require_row(db, intent_type)
    before_standard, before_custom = _split_stored_fields(dict(row.lvo_fields or {}))
    index = next((i for i, m in enumerate(before_custom) if m["id"] == metric_id), None)
    if index is None:
        raise SomMetricNotFoundError(intent_type, metric_id)

    updated = dict(before_custom[index])
    if label is not None:
        label = label.strip()
        if not label:
            raise ValueError("label cannot be empty.")
        updated["label"] = label
    if description is not None:
        description = description.strip()
        if not description:
            raise ValueError("description cannot be empty.")
        updated["description"] = description
    if sort_order is not None:
        updated["sortOrder"] = int(sort_order)

    after_custom = list(before_custom)
    after_custom[index] = updated
    after_custom = _parse_custom_metrics(after_custom)

    try:
        now = _utc_now()
        _commit_metric_mutation(
            db,
            row,
            intent_type,
            before_standard,
            before_custom,
            before_standard,
            after_custom,
            saved_by=saved_by,
            now=now,
            audit_extra={"updatedMetricId": metric_id},
        )
    except Exception:
        db.rollback()
        logger.exception(
            "Update organizational intent metric failed for %s.%s",
            intent_type,
            metric_id,
        )
        raise RuntimeError(ERR_MSG_0024) from None

    saved = next(m for m in after_custom if m["id"] == metric_id)
    return OrgIntentCustomMetric(
        id=saved["id"],
        label=saved["label"],
        description=saved["description"],
        sort_order=saved["sortOrder"],
    )


def delete_organizational_intent_metric(
    db: Session,
    intent_type: str,
    metric_id: str,
    deleted_by: str | None,
) -> OrgIntentFieldDeleteResult:
    intent_type = validate_intent_type(intent_type)
    metric_id = metric_id.strip()
    if not _has_org_table(db):
        raise RuntimeError("Organizational intent tables are not migrated.")

    row = _require_row(db, intent_type)
    before_standard, before_custom = _split_stored_fields(dict(row.lvo_fields or {}))
    after_custom = [m for m in before_custom if m["id"] != metric_id]
    if len(after_custom) == len(before_custom):
        raise SomMetricNotFoundError(intent_type, metric_id)

    try:
        now = _utc_now()
        _commit_metric_mutation(
            db,
            row,
            intent_type,
            before_standard,
            before_custom,
            before_standard,
            after_custom,
            saved_by=deleted_by,
            now=now,
            audit_extra={"deletedMetricId": metric_id},
        )
    except Exception:
        db.rollback()
        logger.exception(
            "Delete organizational intent metric failed for %s.%s",
            intent_type,
            metric_id,
        )
        raise RuntimeError(ERR_MSG_0024) from None

    status = get_configuration_status(db)
    row = _require_row(db, intent_type)
    remaining_standard, remaining_custom = _split_stored_fields(dict(row.lvo_fields or {}))
    return OrgIntentFieldDeleteResult(
        intent_type=intent_type,
        display_name=row.lvo_displayname,
        status=row.lvo_status,
        deleted_field=metric_id,
        is_timeboxed=row.lvo_is_timeboxed,
        is_guardrail=row.lvo_is_guardrail,
        last_synced_at=row.lvo_last_synced_at,
        fields=remaining_standard,
        custom_metrics=_custom_metrics_to_dataclass(remaining_custom),
        field_preview=_build_field_preview(remaining_standard, remaining_custom),
        all_configured=status.all_configured,
        configured_count=status.configured_count,
        total_count=status.total_count,
        success_code=status.success_code,
    )

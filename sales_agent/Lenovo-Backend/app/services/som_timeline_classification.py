"""Sales Operating Model — Timeline classification cards (US 3.4.1)."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from app.models import SomOrganizationalIntent, SomTimelineClassification
from app.services.som_context_lake import rebuild_context_lake
from app.services.som_organizational_intent import get_configuration_status

logger = logging.getLogger(__name__)

INFO_MSG_0006 = "INFO_MSG_0006"
SUCC_MSG_0018 = "SUCC_MSG_0018"
ERR_MSG_0025 = "ERR_MSG_0025"
ERR_MSG_0026 = "ERR_MSG_0026"
ERR_MSG_0027 = "ERR_MSG_0027"
ERR_MSG_0028 = "ERR_MSG_0028"
ERR_MSG_0029 = "ERR_MSG_0029"
ERR_MSG_0030 = "ERR_MSG_0030"

TimelineCardType = Literal[
    "tempo_classes",
    "anchor_definitions",
    "signal_expectations_time_band",
    "seasonal_delayed_activation",
    "acceleration_decay",
    "multiyear_programmatic",
    "exit_recycle_kill",
    "canonical_timeline",
]

CARD_ORDER: tuple[str, ...] = (
    "tempo_classes",
    "anchor_definitions",
    "signal_expectations_time_band",
    "seasonal_delayed_activation",
    "acceleration_decay",
    "multiyear_programmatic",
    "exit_recycle_kill",
    "canonical_timeline",
)

VALID_CARD_TYPES: frozenset[str] = frozenset(CARD_ORDER)

DISPLAY_NAMES: dict[str, str] = {
    "tempo_classes": "Tempo classes",
    "anchor_definitions": "Anchor definitions",
    "signal_expectations_time_band": "Signal expectations by time band",
    "seasonal_delayed_activation": "Seasonal and delayed activation",
    "acceleration_decay": "Acceleration and decay",
    "multiyear_programmatic": "Multi-year and programmatic deals",
    "exit_recycle_kill": "Exit, recycle, and kill",
    "canonical_timeline": "Canonical (quarter/yearly) timeline",
}

DEFAULT_FIELDS: dict[str, dict[str, Any]] = {
    "tempo_classes": {
        "classesDeclared": [
            {
                "className": "Fast/Transactional",
                "scopeDefinition": "Accessories, SMB laptop refresh",
            },
            {
                "className": "Quarterly/Enterprise",
                "scopeDefinition": "Mid-size infrastructure deals",
            },
            {
                "className": "Programmatic/Annual",
                "scopeDefinition": "Education and public procurement cycles",
            },
            {
                "className": "Strategic/Multiyear",
                "scopeDefinition": "Global framework deals",
            },
        ],
        "defaultTempoClassRule": (
            "Quarterly/Enterprise; exceptions = Strategic and Programmatic"
        ),
    },
    "anchor_definitions": {
        "fastClassAnchor": "First customer meeting",
        "enterpriseAnchor": "Architecture validation",
        "programmaticAnchor": "RFP issuance",
        "strategicAnchor": "Steering committee sign-off",
        "reAnchorPolicy": "Allowed once per deal with manager reason code",
        "clockPauseEvents": [
            "legal freeze",
            "procurement blackout",
            "government election code",
        ],
        "noAnchorStateRule": "Valid for seeded opportunities only",
        "customFields": [],
    },
    "signal_expectations_time_band": {
        "band0to30": {
            "expectedSignals": "Discovery meeting and named champion",
            "acceptableSilenceDays": 10,
        },
        "band30to90": {
            "expectedSignals": "Solution validation and pricing discussion",
            "flatPeriodException": (
                "Flat periods acceptable if procurement calendar is pending"
            ),
        },
        "band90to180": {
            "expectedSignals": "Commercial/legal movement",
            "strategicClassException": (
                "No risk flag for missing technical deep-dive in strategic class"
            ),
        },
        "minimumAliveEvidenceRule": (
            "Stakeholder reply or scheduled checkpoint in the last 21 days"
        ),
        "customFields": [],
    },
    "seasonal_delayed_activation": {
        "activationWindows": [
            {
                "quarterLabel": "Q1",
                "description": "Education procurement",
            },
            {
                "quarterLabel": "Q3",
                "description": "Enterprise refresh budget release",
            },
        ],
        "fiscalYearEndLowActivityRule": (
            "Final 3 weeks of fiscal year close — low activity is normal"
        ),
        "preActivationActivityClassification": (
            "Tagged as seeding/education and not flagged as risk"
        ),
        "noActivationEscalationRule": (
            "If activation never occurs within 2 cycles, auto-route to nurture queue"
        ),
        "authorityRules": "",
        "customFields": [],
    },
    "acceleration_decay": {
        "accelerationMarkers": [
            "New economic buyer engaged",
            "Legal review started",
            "Budget code confirmed",
        ],
        "decaySignals": [
            "21+ days of one-sided activity",
            "Repeated reschedules",
            "Scope contraction without timeline reset",
        ],
        "decayReviewThresholds": {
            "fastDays": 30,
            "enterpriseDays": 45,
            "strategicDays": 75,
        },
        "extendedToleranceRule": (
            "Strategic and partner-led classes tolerate longer intentional waiting"
        ),
        "customFields": [],
    },
    "multiyear_programmatic": {
        "programmaticLifecycleUnit": "Fiscal year",
        "strategicLifecycleUnit": "Multi-year program milestone",
        "formalCheckpoints": [
            "Annual budget approval",
            "Board capex cycle",
            "Partner framework renewal",
        ],
        "strategicAliveEvidence": (
            "Sponsor continuity and roadmap checkpoint"
        ),
        "phasedSuccessRule": (
            "Success may be phased as pilot, phase-1 rollout, and scale, "
            "or deferred by design"
        ),
        "customFields": [],
    },
    "exit_recycle_kill": {
        "pauseCondition": (
            "External dependency blocks progress for more than 30 days "
            "with a confirmed re-entry date"
        ),
        "recycleToNurtureCondition": (
            "Intent exists but budget window is more than 1 quarter out"
        ),
        "closeNoDealCondition": (
            "No champion, no budget, and no engagement for the threshold period"
        ),
        "authorityRule": {
            "proposeRole": "Seller",
            "approveRole": "Manager",
            "recommendRole": "System",
        },
        "evidenceNoteCadenceDays": 14,
    },
    "canonical_timeline": {
        "week1to4": {
            "dominantActivity": "Pipeline creation",
            "kpiTarget": "45% of new quarterly opportunities created",
            "agentInterpretationNote": (
                "Early-quarter prospecting signals are expected and positive"
            ),
        },
        "week5to8": {
            "dominantActivity": "Qualification and solutioning",
            "kpiTarget": "60% stage progression",
            "agentInterpretationNote": (
                "Mid-quarter validation activity should dominate the portfolio"
            ),
        },
        "week9to12": {
            "dominantActivity": "Negotiation and closure",
            "kpiTarget": "75% commit protection",
            "agentInterpretationNote": (
                "Late-quarter closure motion outweighs new pipeline creation"
            ),
        },
        "globalAgentInterpretationRule": (
            "Agents interpret identical signals differently by week context"
        ),
        "q1": {
            "quarterCharacter": (
                "Foundation quarter — new fiscal year planning, account "
                "re-segmentation, and early pipeline seeding"
            ),
            "annualKpiContribution": "20% of annual new logo target",
            "agentInterpretationNote": (
                "Lenient interpretation of early pipeline build activity"
            ),
        },
        "q2": {
            "quarterCharacter": (
                "Growth quarter — expansion motion emphasis following Q1 pipeline build"
            ),
            "annualKpiContribution": "25% of annual revenue target",
            "agentInterpretationNote": (
                "Expect progression from seeded Q1 pipeline"
            ),
        },
        "q3": {
            "quarterCharacter": (
                "Harvest quarter — highest revenue concentration, "
                "enterprise refresh cycles activate"
            ),
            "annualKpiContribution": "30% of annual revenue target",
            "agentInterpretationNote": (
                "Closure and commit protection are primary signals"
            ),
        },
        "q4": {
            "quarterCharacter": (
                "Closure and reset quarter — final commit protection plus "
                "early seeding for next fiscal year Q1"
            ),
            "annualKpiContribution": "25% of annual revenue target",
            "agentInterpretationNote": (
                "Renewed prospecting in Q4 is a recovery signal, not off-pattern"
            ),
        },
        "crossQuarterCarryoverRule": (
            "Deals in Negotiation at quarter-end carry forward into the next "
            "quarter's closure-equivalent week band without restarting the "
            "canonical week clock, unless a clock pause event applies"
        ),
        "globalYearlyInterpretationRule": (
            "Agents compare current-quarter pace against the same week-band "
            "benchmark from the prior quarter, not a fixed annual average"
        ),
    },
}

FIELD_LABELS: dict[str, dict[str, str]] = {
    "tempo_classes": {
        "classesDeclared": "Classes declared",
        "defaultTempoClassRule": "Default tempo class assignment rule",
    },
    "anchor_definitions": {
        "fastClassAnchor": "Fast class anchor",
        "enterpriseAnchor": "Enterprise anchor",
        "programmaticAnchor": "Programmatic anchor",
        "strategicAnchor": "Strategic anchor",
        "reAnchorPolicy": "Re-anchor policy",
        "clockPauseEvents": "Clock pause events",
        "noAnchorStateRule": "No-anchor state rule",
    },
    "signal_expectations_time_band": {
        "band0to30": "0–30d band",
        "band30to90": "30–90d band",
        "band90to180": "90–180d band",
        "minimumAliveEvidenceRule": "Minimum alive evidence rule",
    },
    "seasonal_delayed_activation": {
        "activationWindows": "Activation windows",
        "fiscalYearEndLowActivityRule": "Fiscal year-end low activity rule",
        "preActivationActivityClassification": "Pre-activation activity classification",
        "noActivationEscalationRule": "No-activation escalation rule",
    },
    "acceleration_decay": {
        "accelerationMarkers": "Acceleration markers",
        "decaySignals": "Decay signals",
        "decayReviewThresholds": "Decay review thresholds",
        "extendedToleranceRule": "Extended tolerance rule",
    },
    "multiyear_programmatic": {
        "programmaticLifecycleUnit": "Programmatic lifecycle unit",
        "strategicLifecycleUnit": "Strategic lifecycle unit",
        "formalCheckpoints": "Formal checkpoints",
        "strategicAliveEvidence": "Strategic alive evidence",
        "phasedSuccessRule": "Phased success rule",
    },
    "exit_recycle_kill": {
        "pauseCondition": "Pause condition",
        "recycleToNurtureCondition": "Recycle-to-nurture condition",
        "closeNoDealCondition": "Close-no-deal condition",
        "authorityRule": "Authority rule",
        "evidenceNoteCadenceDays": "Evidence note cadence (days)",
    },
    "canonical_timeline": {
        "week1to4": "Week 1–4 band",
        "week5to8": "Week 5–8 band",
        "week9to12": "Week 9–12 band",
        "globalAgentInterpretationRule": "Global agent interpretation rule",
        "q1": "Q1 band",
        "q2": "Q2 band",
        "q3": "Q3 band",
        "q4": "Q4 band",
        "crossQuarterCarryoverRule": "Cross-quarter carryover rule",
        "globalYearlyInterpretationRule": "Global yearly interpretation rule",
    },
}

OPTIONAL_FIELDS = frozenset(
    {
        "customFields",
        "authorityRules",
        "evidenceNoteCadenceDays",
    }
)

INFO_MSG_0006_MESSAGE = (
    "This change will update agent behaviour from the next recommendation cycle. "
    "Existing in-flight recommendations will not be retroactively changed."
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class SomFieldValidationError(ValueError):
    def __init__(self, code: str, field: str, message: str) -> None:
        self.code = code
        self.field = field
        self.message = message
        super().__init__(message)


class SomAgentImpactRequired(Exception):
    """PUT must include confirmAgentImpact=true after FE shows INFO_MSG_0006."""


class SomSectionLockedError(Exception):
    """Organizational intent is not fully configured."""


def validate_card_type(card_type: str) -> TimelineCardType:
    normalized = card_type.strip().lower()
    if normalized not in VALID_CARD_TYPES:
        raise ValueError(
            f"cardType must be one of: {', '.join(CARD_ORDER)}"
        )
    return normalized  # type: ignore[return-value]


def _has_timeline_table(db: Session) -> bool:
    return inspect(db.bind).has_table("lvo_som_timeline_classification")


def _is_org_intent_complete(db: Session) -> bool:
    try:
        return get_configuration_status(db).all_configured
    except RuntimeError:
        return False


def assert_section_unlocked(db: Session) -> None:
    if not _is_org_intent_complete(db):
        raise SomSectionLockedError(ERR_MSG_0030)


def _require_row(db: Session, card_type: str) -> SomTimelineClassification:
    row = db.get(SomTimelineClassification, card_type)
    if row is None:
        raise LookupError(f"Timeline card not found: {card_type}")
    return row


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    if isinstance(value, dict):
        return len(value) == 0
    return False


def _positive_int(value: Any, field: str, code: str, label: str) -> int:
    if value is None:
        raise SomFieldValidationError(code, field, f"{label} is required.")
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise SomFieldValidationError(
            code, field, f"{label} must be a positive number of days."
        ) from exc
    if n <= 0:
        raise SomFieldValidationError(
            code, field, f"{label} must be a positive number of days."
        )
    return n


def _require_string(fields: dict[str, Any], key: str, label: str) -> str:
    raw = fields.get(key)
    if _is_blank(raw):
        raise SomFieldValidationError(
            ERR_MSG_0028, key, f"{label} is required."
        )
    return str(raw).strip()


def _require_string_list(fields: dict[str, Any], key: str, label: str) -> list[str]:
    raw = fields.get(key)
    if not isinstance(raw, list) or not raw:
        raise SomFieldValidationError(
            ERR_MSG_0028, key, f"{label} is required."
        )
    out = [str(x).strip() for x in raw if str(x).strip()]
    if not out:
        raise SomFieldValidationError(
            ERR_MSG_0028, key, f"{label} is required."
        )
    return out


def _require_classes_declared(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise SomFieldValidationError(
            ERR_MSG_0028,
            "classesDeclared",
            "Classes declared is required.",
        )
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("className") or "").strip()
        scope = str(item.get("scopeDefinition") or "").strip()
        if not name or not scope:
            raise SomFieldValidationError(
                ERR_MSG_0028,
                "classesDeclared",
                "Each tempo class requires a class name and scope definition.",
            )
        out.append({"className": name, "scopeDefinition": scope})
    if not out:
        raise SomFieldValidationError(
            ERR_MSG_0028,
            "classesDeclared",
            "Classes declared is required.",
        )
    return out


def _validate_decay_thresholds(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise SomFieldValidationError(
            ERR_MSG_0028,
            "decayReviewThresholds",
            "Decay review thresholds is required.",
        )
    return {
        "fastDays": _positive_int(
            raw.get("fastDays"),
            "decayReviewThresholds.fastDays",
            ERR_MSG_0025,
            "Fast class decay review threshold",
        ),
        "enterpriseDays": _positive_int(
            raw.get("enterpriseDays"),
            "decayReviewThresholds.enterpriseDays",
            ERR_MSG_0025,
            "Enterprise class decay review threshold",
        ),
        "strategicDays": _positive_int(
            raw.get("strategicDays"),
            "decayReviewThresholds.strategicDays",
            ERR_MSG_0025,
            "Strategic class decay review threshold",
        ),
    }


def _validate_authority_rule(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise SomFieldValidationError(
            ERR_MSG_0028, "authorityRule", "Authority rule is required."
        )
    out: dict[str, str] = {}
    for key, label in (
        ("proposeRole", "Propose role"),
        ("approveRole", "Approve role"),
        ("recommendRole", "Recommend role"),
    ):
        val = raw.get(key)
        if _is_blank(val):
            raise SomFieldValidationError(
                ERR_MSG_0028, f"authorityRule.{key}", f"{label} is required."
            )
        out[key] = str(val).strip()
    return out


def _validate_band(
    raw: Any,
    field_key: str,
    label: str,
    sub_keys: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SomFieldValidationError(
            ERR_MSG_0028, field_key, f"{label} is required."
        )
    result: dict[str, Any] = {}
    for sub in sub_keys:
        if sub == "acceptableSilenceDays":
            result[sub] = _positive_int(
                raw.get(sub),
                f"{field_key}.{sub}",
                ERR_MSG_0025,
                "Acceptable silence duration",
            )
        else:
            val = raw.get(sub)
            if _is_blank(val):
                raise SomFieldValidationError(
                    ERR_MSG_0028,
                    f"{field_key}.{sub}",
                    f"{label} — {sub} is required.",
                )
            result[sub] = str(val).strip()
    return result


def _validate_week_or_quarter_band(
    raw: Any,
    field_key: str,
    label: str,
    sub_keys: tuple[str, ...],
) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise SomFieldValidationError(
            ERR_MSG_0028, field_key, f"{label} is required."
        )
    out: dict[str, str] = {}
    for sub in sub_keys:
        val = raw.get(sub)
        if _is_blank(val):
            raise SomFieldValidationError(
                ERR_MSG_0028,
                f"{field_key}.{sub}",
                f"{label} — {sub} is required.",
            )
        out[sub] = str(val).strip()
    return out


def _validate_activation_windows(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise SomFieldValidationError(
            ERR_MSG_0028,
            "activationWindows",
            "Activation windows is required.",
        )
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        ql = str(item.get("quarterLabel") or "").strip()
        desc = str(item.get("description") or "").strip()
        if not ql or not desc:
            raise SomFieldValidationError(
                ERR_MSG_0028,
                "activationWindows",
                "Each activation window requires a quarter label and description.",
            )
        out.append({"quarterLabel": ql, "description": desc})
    return out


def _validate_fields(card_type: str, fields: dict[str, Any]) -> dict[str, Any]:
    labels = FIELD_LABELS[card_type]
    cleaned: dict[str, Any] = {}

    if card_type == "tempo_classes":
        cleaned["classesDeclared"] = _require_classes_declared(
            fields.get("classesDeclared")
        )
        cleaned["defaultTempoClassRule"] = _require_string(
            fields, "defaultTempoClassRule", labels["defaultTempoClassRule"]
        )
        return cleaned

    if card_type == "anchor_definitions":
        for key in (
            "fastClassAnchor",
            "enterpriseAnchor",
            "programmaticAnchor",
            "strategicAnchor",
            "reAnchorPolicy",
            "noAnchorStateRule",
        ):
            cleaned[key] = _require_string(fields, key, labels[key])
        cleaned["clockPauseEvents"] = _require_string_list(
            fields, "clockPauseEvents", labels["clockPauseEvents"]
        )
        custom = fields.get("customFields")
        if isinstance(custom, list):
            cleaned["customFields"] = custom
        return cleaned

    if card_type == "signal_expectations_time_band":
        cleaned["band0to30"] = _validate_band(
            fields.get("band0to30"),
            "band0to30",
            labels["band0to30"],
            ("expectedSignals", "acceptableSilenceDays"),
        )
        cleaned["band30to90"] = _validate_band(
            fields.get("band30to90"),
            "band30to90",
            labels["band30to90"],
            ("expectedSignals", "flatPeriodException"),
        )
        cleaned["band90to180"] = _validate_band(
            fields.get("band90to180"),
            "band90to180",
            labels["band90to180"],
            ("expectedSignals", "strategicClassException"),
        )
        cleaned["minimumAliveEvidenceRule"] = _require_string(
            fields,
            "minimumAliveEvidenceRule",
            labels["minimumAliveEvidenceRule"],
        )
        custom = fields.get("customFields")
        if isinstance(custom, list):
            cleaned["customFields"] = custom
        return cleaned

    if card_type == "seasonal_delayed_activation":
        cleaned["activationWindows"] = _validate_activation_windows(
            fields.get("activationWindows")
        )
        for key in (
            "fiscalYearEndLowActivityRule",
            "preActivationActivityClassification",
            "noActivationEscalationRule",
        ):
            cleaned[key] = _require_string(fields, key, labels[key])
        auth = fields.get("authorityRules")
        if auth is not None and str(auth).strip():
            cleaned["authorityRules"] = str(auth).strip()
        cadence = fields.get("evidenceNoteCadenceDays")
        if cadence is not None and cadence != "":
            cleaned["evidenceNoteCadenceDays"] = _positive_int(
                cadence,
                "evidenceNoteCadenceDays",
                ERR_MSG_0026,
                "Evidence note cadence",
            )
        custom = fields.get("customFields")
        if isinstance(custom, list):
            cleaned["customFields"] = custom
        return cleaned

    if card_type == "acceleration_decay":
        cleaned["accelerationMarkers"] = _require_string_list(
            fields, "accelerationMarkers", labels["accelerationMarkers"]
        )
        cleaned["decaySignals"] = _require_string_list(
            fields, "decaySignals", labels["decaySignals"]
        )
        cleaned["decayReviewThresholds"] = _validate_decay_thresholds(
            fields.get("decayReviewThresholds")
        )
        cleaned["extendedToleranceRule"] = _require_string(
            fields, "extendedToleranceRule", labels["extendedToleranceRule"]
        )
        custom = fields.get("customFields")
        if isinstance(custom, list):
            cleaned["customFields"] = custom
        return cleaned

    if card_type == "multiyear_programmatic":
        for key in (
            "programmaticLifecycleUnit",
            "strategicLifecycleUnit",
            "strategicAliveEvidence",
            "phasedSuccessRule",
        ):
            cleaned[key] = _require_string(fields, key, labels[key])
        cleaned["formalCheckpoints"] = _require_string_list(
            fields, "formalCheckpoints", labels["formalCheckpoints"]
        )
        custom = fields.get("customFields")
        if isinstance(custom, list):
            cleaned["customFields"] = custom
        return cleaned

    if card_type == "exit_recycle_kill":
        for key in (
            "pauseCondition",
            "recycleToNurtureCondition",
            "closeNoDealCondition",
        ):
            cleaned[key] = _require_string(fields, key, labels[key])
        cleaned["authorityRule"] = _validate_authority_rule(
            fields.get("authorityRule")
        )
        cleaned["evidenceNoteCadenceDays"] = _positive_int(
            fields.get("evidenceNoteCadenceDays"),
            "evidenceNoteCadenceDays",
            ERR_MSG_0026,
            "Evidence note cadence",
        )
        return cleaned

    if card_type == "canonical_timeline":
        week_keys = ("week1to4", "week5to8", "week9to12")
        week_subs = (
            "dominantActivity",
            "kpiTarget",
            "agentInterpretationNote",
        )
        for wk in week_keys:
            cleaned[wk] = _validate_week_or_quarter_band(
                fields.get(wk),
                wk,
                labels[wk],
                week_subs,
            )
        cleaned["globalAgentInterpretationRule"] = _require_string(
            fields,
            "globalAgentInterpretationRule",
            labels["globalAgentInterpretationRule"],
        )
        for qk in ("q1", "q2", "q3", "q4"):
            cleaned[qk] = _validate_week_or_quarter_band(
                fields.get(qk),
                qk,
                labels[qk],
                ("quarterCharacter", "annualKpiContribution", "agentInterpretationNote"),
            )
        for key in ("crossQuarterCarryoverRule", "globalYearlyInterpretationRule"):
            cleaned[key] = _require_string(fields, key, labels[key])
        return cleaned

    raise ValueError(f"Unknown card type: {card_type}")


def _validate_guardrails(db: Session, cleaned: dict[str, Any]) -> None:
    """Block values that violate configured Constraint organizational intent."""
    if not inspect(db.bind).has_table("lvo_som_organizational_intent"):
        return
    row = db.get(SomOrganizationalIntent, "constraint")
    if row is None or row.lvo_status != "CONFIGURED":
        return
    floors = (row.lvo_fields or {}).get("marginFloors") or []
    if not isinstance(floors, list) or not floors:
        return
    min_allowed: float | None = None
    for item in floors:
        if isinstance(item, dict) and item.get("minPercent") is not None:
            try:
                pct = float(item["minPercent"])
            except (TypeError, ValueError):
                continue
            min_allowed = pct if min_allowed is None else min(min_allowed, pct)
    if min_allowed is None:
        return
    candidate = cleaned.get("minimumMarginPercent")
    if candidate is None:
        return
    try:
        val = float(candidate)
    except (TypeError, ValueError) as exc:
        raise SomFieldValidationError(
            ERR_MSG_0027,
            "minimumMarginPercent",
            "This value cannot be set below the declared hard guardrail. "
            "Agents cannot violate hard guardrails.",
        ) from exc
    if val < min_allowed:
        raise SomFieldValidationError(
            ERR_MSG_0027,
            "minimumMarginPercent",
            "This value cannot be set below the declared hard guardrail. "
            "Agents cannot violate hard guardrails.",
        )


@dataclass(frozen=True)
class TimelineSectionStatus:
    section_unlocked: bool
    organizational_intent_configured: bool
    timeline_configured_count: int
    timeline_total_count: int
    all_timeline_configured: bool
    success_code: str | None
    message_code: str | None


@dataclass(frozen=True)
class TimelineCardSummary:
    card_type: str
    display_name: str
    status: str
    last_synced_at: datetime | None
    field_preview: dict[str, Any]


@dataclass(frozen=True)
class TimelineCardDetail:
    card_type: str
    display_name: str
    status: str
    last_synced_at: datetime | None
    fields: dict[str, Any]
    field_labels: dict[str, str]
    defaults: dict[str, Any]
    required_fields: list[str]


@dataclass(frozen=True)
class TimelineSaveResult:
    card_type: str
    display_name: str
    status: str
    last_synced_at: datetime
    all_configured: bool
    success_code: str | None


@dataclass(frozen=True)
class TimelineConfigurationStatus:
    all_configured: bool
    configured_count: int
    total_count: int
    success_code: str | None


def get_timeline_section_status(db: Session) -> TimelineSectionStatus:
    org_ok = _is_org_intent_complete(db)
    timeline_count = 0
    total = len(CARD_ORDER)
    if _has_timeline_table(db):
        rows = db.execute(select(SomTimelineClassification)).scalars().all()
        timeline_count = sum(1 for r in rows if r.lvo_status == "CONFIGURED")
    all_tl = timeline_count == total and org_ok
    return TimelineSectionStatus(
        section_unlocked=org_ok,
        organizational_intent_configured=org_ok,
        timeline_configured_count=timeline_count,
        timeline_total_count=total,
        all_timeline_configured=all_tl,
        success_code=SUCC_MSG_0018 if all_tl else None,
        message_code=None if org_ok else "INF_MSG_0005",
    )


def get_timeline_configuration_status(db: Session) -> TimelineConfigurationStatus:
    if not _has_timeline_table(db):
        raise RuntimeError("Timeline classification tables are not migrated.")
    assert_section_unlocked(db)
    rows = db.execute(select(SomTimelineClassification)).scalars().all()
    by_type = {r.lvo_cardtype: r for r in rows}
    configured = sum(
        1
        for t in CARD_ORDER
        if by_type.get(t) and by_type[t].lvo_status == "CONFIGURED"
    )
    total = len(CARD_ORDER)
    all_ok = configured == total
    return TimelineConfigurationStatus(
        all_configured=all_ok,
        configured_count=configured,
        total_count=total,
        success_code=SUCC_MSG_0018 if all_ok else None,
    )


def list_timeline_cards(db: Session) -> list[TimelineCardSummary]:
    if not _has_timeline_table(db):
        raise RuntimeError("Timeline classification tables are not migrated.")
    assert_section_unlocked(db)
    by_type = {
        row.lvo_cardtype: row
        for row in db.execute(select(SomTimelineClassification)).scalars()
    }
    summaries: list[TimelineCardSummary] = []
    for card_type in CARD_ORDER:
        row = by_type.get(card_type)
        if row is None:
            continue
        fields = dict(row.lvo_fields or {})
        if row.lvo_status != "CONFIGURED":
            fields = copy.deepcopy(DEFAULT_FIELDS.get(card_type, {}))
        preview = {
            k: v
            for k, v in fields.items()
            if k not in OPTIONAL_FIELDS and not _is_blank(v)
        }
        summaries.append(
            TimelineCardSummary(
                card_type=row.lvo_cardtype,
                display_name=row.lvo_displayname,
                status=row.lvo_status,
                last_synced_at=row.lvo_last_synced_at,
                field_preview=preview,
            )
        )
    return summaries


def get_timeline_card(db: Session, card_type: str) -> TimelineCardDetail:
    card_type = validate_card_type(card_type)
    if not _has_timeline_table(db):
        raise RuntimeError("Timeline classification tables are not migrated.")
    assert_section_unlocked(db)
    row = _require_row(db, card_type)
    defaults = copy.deepcopy(DEFAULT_FIELDS[card_type])
    if row.lvo_status == "CONFIGURED":
        fields = dict(row.lvo_fields or {})
    else:
        fields = defaults
    required = [
        k
        for k in FIELD_LABELS.get(card_type, {})
        if k not in OPTIONAL_FIELDS
    ]
    return TimelineCardDetail(
        card_type=row.lvo_cardtype,
        display_name=row.lvo_displayname,
        status=row.lvo_status,
        last_synced_at=row.lvo_last_synced_at,
        fields=fields,
        field_labels=dict(FIELD_LABELS.get(card_type, {})),
        defaults=defaults,
        required_fields=required,
    )


def save_timeline_card(
    db: Session,
    card_type: str,
    fields: dict[str, Any],
    *,
    confirm_agent_impact: bool,
    saved_by: str | None,
) -> TimelineSaveResult:
    card_type = validate_card_type(card_type)
    if not _has_timeline_table(db):
        raise RuntimeError("Timeline classification tables are not migrated.")
    assert_section_unlocked(db)
    if not confirm_agent_impact:
        raise SomAgentImpactRequired()

    try:
        cleaned = _validate_fields(card_type, fields)
        _validate_guardrails(db, cleaned)
        row = _require_row(db, card_type)
        now = _utc_now()
        row.lvo_fields = cleaned
        row.lvo_status = "CONFIGURED"
        row.lvo_last_synced_at = now
        row.lvo_configured_by = saved_by
        rebuild_context_lake(db, saved_by)
        db.commit()
    except (SomFieldValidationError, SomAgentImpactRequired, SomSectionLockedError):
        db.rollback()
        raise
    except Exception:
        db.rollback()
        logger.exception("Timeline card save failed for %s", card_type)
        raise RuntimeError(ERR_MSG_0029) from None

    status_data = get_timeline_configuration_status(db)
    row = _require_row(db, card_type)
    return TimelineSaveResult(
        card_type=card_type,
        display_name=row.lvo_displayname,
        status=row.lvo_status,
        last_synced_at=row.lvo_last_synced_at or _utc_now(),
        all_configured=status_data.all_configured,
        success_code=status_data.success_code,
    )

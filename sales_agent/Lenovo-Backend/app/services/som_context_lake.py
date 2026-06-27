"""Sales Operating Model Context Lake — unified rebuild (US 3.2.1 + 3.2.2)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from app.models import (
    SomConfigurationCycle,
    SomContextLake,
    SomIntentCard,
    SomInterviewQuestion,
    SomInterviewResponse,
    SomOrganizationalIntent,
    SomTimelineClassification,
)

CONTEXT_LAKE_VERSION = 3

_VALID_ROLES = frozenset(
    {"national_manager", "regional_manager", "seller_manager"}
)
_ROLE_DISPLAY = {
    "national_manager": "National Manager",
    "regional_manager": "Regional Manager",
    "seller_manager": "Seller Manager",
}
_SCOPE_LABELS = {
    "national_manager": "Scope: Org-level intent and global guardrails",
    "regional_manager": "Scope: Region-specific behaviour and constraints",
    "seller_manager": "Scope: Team-level behavioural and execution rules",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _new_id() -> str:
    return str(uuid.uuid4())


def _get_active_cycle(db: Session) -> SomConfigurationCycle:
    row = db.execute(
        select(SomConfigurationCycle)
        .order_by(SomConfigurationCycle.lvo_createdat.desc())
        .limit(1)
    ).scalar_one_or_none()
    if row is None:
        row = SomConfigurationCycle(
            lvo_cycleid=_new_id(),
            lvo_status="in_progress",
            lvo_createdat=_utc_now(),
        )
        db.add(row)
        db.flush()
    return row


def _load_questions(db: Session, role: str) -> list[SomInterviewQuestion]:
    return list(
        db.execute(
            select(SomInterviewQuestion)
            .where(
                SomInterviewQuestion.lvo_role == role,
                SomInterviewQuestion.lvo_isactive.is_(True),
            )
            .order_by(SomInterviewQuestion.lvo_sortorder.asc())
        ).scalars()
    )


def _load_saved_map(
    db: Session,
    cycle_id: str,
    role: str,
    question_ids: list[str],
) -> dict[str, str]:
    if not question_ids:
        return {}
    rows = db.execute(
        select(SomInterviewResponse).where(
            SomInterviewResponse.lvo_cycleid == cycle_id,
            SomInterviewResponse.lvo_role == role,
            SomInterviewResponse.lvo_status == "saved",
            SomInterviewResponse.lvo_questionid.in_(question_ids),
        )
    ).scalars()
    return {r.lvo_questionid: r.lvo_responsetext or "" for r in rows}


def _build_interview_block(db: Session, cycle: SomConfigurationCycle) -> dict[str, Any]:
    roles_block: dict[str, Any] = {}
    for role in sorted(_VALID_ROLES):
        questions = _load_questions(db, role)
        qids = [q.lvo_questionid for q in questions]
        saved = _load_saved_map(db, cycle.lvo_cycleid, role, qids)
        if not saved:
            continue
        intent = db.get(SomIntentCard, role)
        entries = []
        for q in questions:
            text = saved.get(q.lvo_questionid, "").strip()
            if not text:
                continue
            entries.append(
                {
                    "questionId": q.lvo_questionid,
                    "sortOrder": q.lvo_sortorder,
                    "question": q.lvo_questiontext,
                    "response": text,
                }
            )
        if entries:
            roles_block[role] = {
                "roleDisplay": _ROLE_DISPLAY[role],
                "scopeLabel": _SCOPE_LABELS[role],
                "status": intent.lvo_status if intent else "NOT_CONFIGURED",
                "configuredAt": (
                    intent.lvo_configuredat.isoformat()
                    if intent and intent.lvo_configuredat
                    else None
                ),
                "interviewResponses": entries,
            }
    return roles_block


def _build_organizational_block(db: Session) -> dict[str, Any]:
    if not inspect(db.bind).has_table("lvo_som_organizational_intent"):
        return {}

    rows = db.execute(
        select(SomOrganizationalIntent).order_by(
            SomOrganizationalIntent.lvo_intenttype.asc()
        )
    ).scalars()

    block: dict[str, Any] = {}
    for row in rows:
        if row.lvo_status != "CONFIGURED":
            continue
        block[row.lvo_intenttype] = {
            "displayName": row.lvo_displayname,
            "status": row.lvo_status,
            "lastSyncedAt": (
                row.lvo_last_synced_at.isoformat()
                if row.lvo_last_synced_at
                else None
            ),
            "isTimeboxed": row.lvo_is_timeboxed,
            "isGuardrail": row.lvo_is_guardrail,
            "expiryDate": (
                row.lvo_expiry_date.isoformat() if row.lvo_expiry_date else None
            ),
            "fields": dict(row.lvo_fields or {}),
        }
    return block


def _build_timeline_block(db: Session) -> dict[str, Any]:
    if not inspect(db.bind).has_table("lvo_som_timeline_classification"):
        return {}

    rows = db.execute(
        select(SomTimelineClassification).order_by(
            SomTimelineClassification.lvo_cardtype.asc()
        )
    ).scalars()

    block: dict[str, Any] = {}
    for row in rows:
        if row.lvo_status != "CONFIGURED":
            continue
        block[row.lvo_cardtype] = {
            "displayName": row.lvo_displayname,
            "status": row.lvo_status,
            "lastSyncedAt": (
                row.lvo_last_synced_at.isoformat()
                if row.lvo_last_synced_at
                else None
            ),
            "fields": dict(row.lvo_fields or {}),
        }
    return block


def rebuild_context_lake(db: Session, updated_by: str | None) -> dict[str, Any]:
    """Rebuild and persist the full Context Lake snapshot."""
    cycle = _get_active_cycle(db)
    org_block = _build_organizational_block(db)
    timeline_block = _build_timeline_block(db)
    version = CONTEXT_LAKE_VERSION if timeline_block or inspect(db.bind).has_table(
        "lvo_som_timeline_classification"
    ) else 2
    payload: dict[str, Any] = {
        "version": version,
        "cycleId": cycle.lvo_cycleid,
        "updatedAt": _utc_now().isoformat(),
        "interview": {"roles": _build_interview_block(db, cycle)},
        "organizationalIntents": org_block,
    }
    if version >= 3:
        payload["timelineClassification"] = timeline_block

    row = db.get(SomContextLake, 1)
    now = _utc_now()
    if row is None:
        row = SomContextLake(
            id=1,
            lvo_payload=payload,
            lvo_updatedat=now,
            lvo_updatedby=updated_by,
        )
        db.add(row)
    else:
        row.lvo_payload = payload
        row.lvo_updatedat = now
        row.lvo_updatedby = updated_by

    return payload


def normalize_context_lake_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Map v1 payloads (top-level roles) to v2 shape for API responses."""
    if int(raw.get("version") or 1) >= 2:
        result = {
            "version": int(raw.get("version") or 2),
            "cycleId": raw.get("cycleId"),
            "updatedAt": raw.get("updatedAt"),
            "interview": {"roles": raw.get("interview", {}).get("roles") or raw.get("roles") or {}},
            "organizationalIntents": raw.get("organizationalIntents") or {},
        }
        if int(raw.get("version") or 0) >= 3:
            result["timelineClassification"] = raw.get("timelineClassification") or {}
        return result
    return {
        "version": 1,
        "cycleId": raw.get("cycleId"),
        "updatedAt": raw.get("updatedAt"),
        "interview": {"roles": raw.get("roles") or {}},
        "organizationalIntents": raw.get("organizationalIntents") or {},
    }

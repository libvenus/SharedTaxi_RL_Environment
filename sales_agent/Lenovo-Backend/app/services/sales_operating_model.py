"""Sales Operating Model — Interview-First Setup (US 3.2.1)."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from sqlalchemy import delete, inspect, select
from sqlalchemy.orm import Session

from app.models import (
    SomConfigurationCycle,
    SomIntentCard,
    SomInterviewQuestion,
    SomInterviewResponse,
)
from app.services.som_context_lake import normalize_context_lake_payload, rebuild_context_lake

logger = logging.getLogger(__name__)

ERR_MSG_0021 = "ERR_MSG_0021"

SomRole = Literal["national_manager", "regional_manager", "seller_manager"]

VALID_ROLES: frozenset[str] = frozenset(
    {"national_manager", "regional_manager", "seller_manager"}
)

SCOPE_LABELS: dict[str, str] = {
    "national_manager": "Scope: Org-level intent and global guardrails",
    "regional_manager": "Scope: Region-specific behaviour and constraints",
    "seller_manager": "Scope: Team-level behavioural and execution rules",
}

ROLE_DISPLAY: dict[str, str] = {
    "national_manager": "National Manager",
    "regional_manager": "Regional Manager",
    "seller_manager": "Seller Manager",
}

DEFAULT_CYCLE_ID = "00000000-0000-4000-8000-000000000001"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _new_id() -> str:
    return str(uuid.uuid4())


def _has_som_tables(db: Session) -> bool:
    inspector = inspect(db.bind)
    return inspector.has_table("lvo_som_interview_question")


def validate_role(role: str) -> SomRole:
    normalized = role.strip().lower()
    if normalized not in VALID_ROLES:
        raise ValueError(
            f"role must be one of: {', '.join(sorted(VALID_ROLES))}"
        )
    return normalized  # type: ignore[return-value]


def count_captured_responses(responses: dict[str, str]) -> int:
    """Questions with at least one non-whitespace character."""
    return sum(1 for text in responses.values() if text and text.strip())


def verify_enabled(captured_count: int) -> bool:
    return captured_count >= 1


@dataclass(frozen=True)
class InterviewQuestionRow:
    question_id: str
    sort_order: int
    text: str
    role: str | None = None


@dataclass(frozen=True)
class InterviewSetupData:
    role: str
    scope_label: str
    role_display: str
    questions: list[InterviewQuestionRow]
    draft_responses: dict[str, str]
    captured_count: int
    total_questions: int
    verify_enabled: bool
    intent_card_status: str
    configured_at: datetime | None
    saved_responses: dict[str, str]


@dataclass(frozen=True)
class IntentCardData:
    role: str
    role_display: str
    scope_label: str
    status: str
    configured_at: datetime | None


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


def _load_draft_map(
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
            SomInterviewResponse.lvo_status == "draft",
            SomInterviewResponse.lvo_questionid.in_(question_ids),
        )
    ).scalars()
    return {r.lvo_questionid: r.lvo_responsetext or "" for r in rows}


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


def build_interview_setup(db: Session, role: str) -> InterviewSetupData:
    role = validate_role(role)
    if not _has_som_tables(db):
        raise RuntimeError("Sales Operating Model tables are not migrated.")

    cycle = _get_active_cycle(db)
    questions = _load_questions(db, role)
    question_rows = [
        InterviewQuestionRow(
            question_id=q.lvo_questionid,
            sort_order=q.lvo_sortorder,
            text=q.lvo_questiontext,
        )
        for q in questions
    ]
    qids = [q.question_id for q in question_rows]

    drafts = _load_draft_map(db, cycle.lvo_cycleid, role, qids)
    saved = _load_saved_map(db, cycle.lvo_cycleid, role, qids)

    responses: dict[str, str] = {qid: drafts.get(qid, "") for qid in qids}
    captured = count_captured_responses(responses)

    intent = db.get(SomIntentCard, role)
    intent_status = intent.lvo_status if intent else "NOT_CONFIGURED"
    configured_at = intent.lvo_configuredat if intent else None

    return InterviewSetupData(
        role=role,
        scope_label=SCOPE_LABELS[role],
        role_display=ROLE_DISPLAY[role],
        questions=question_rows,
        draft_responses=responses,
        captured_count=captured,
        total_questions=len(question_rows),
        verify_enabled=verify_enabled(captured),
        intent_card_status=intent_status,
        configured_at=configured_at,
        saved_responses=saved,
    )


def save_draft_responses(
    db: Session,
    role: str,
    responses: list[dict[str, str]],
    captured_by: str | None,
) -> InterviewSetupData:
    role = validate_role(role)
    cycle = _get_active_cycle(db)
    questions = _load_questions(db, role)
    valid_ids = {q.lvo_questionid for q in questions}

    now = _utc_now()
    for item in responses:
        qid = item["question_id"]
        if qid not in valid_ids:
            raise ValueError(f"Unknown questionId for role {role}: {qid}")
        text = item.get("text") or ""
        existing = db.execute(
            select(SomInterviewResponse).where(
                SomInterviewResponse.lvo_cycleid == cycle.lvo_cycleid,
                SomInterviewResponse.lvo_questionid == qid,
                SomInterviewResponse.lvo_status == "draft",
            )
        ).scalar_one_or_none()
        if existing:
            existing.lvo_responsetext = text
            existing.lvo_capturedby = captured_by
            existing.lvo_capturedat = now
        else:
            db.add(
                SomInterviewResponse(
                    lvo_responseid=_new_id(),
                    lvo_cycleid=cycle.lvo_cycleid,
                    lvo_questionid=qid,
                    lvo_role=role,
                    lvo_responsetext=text,
                    lvo_status="draft",
                    lvo_capturedby=captured_by,
                    lvo_capturedat=now,
                )
            )

    db.commit()
    return build_interview_setup(db, role)


def save_interview_responses(
    db: Session,
    role: str,
    responses: list[dict[str, str]],
    saved_by: str | None,
) -> InterviewSetupData:
    """Atomic save from Verify & Edit review panel."""
    role = validate_role(role)
    if not responses:
        raise ValueError("At least one response is required to save.")

    try:
        cycle = _get_active_cycle(db)
        questions = _load_questions(db, role)
        valid_ids = {q.lvo_questionid for q in questions}
        now = _utc_now()

        submitted: dict[str, str] = {}
        for item in responses:
            qid = item["question_id"]
            if qid not in valid_ids:
                raise ValueError(f"Unknown questionId for role {role}: {qid}")
            text = (item.get("text") or "").strip()
            if text:
                submitted[qid] = text

        if not submitted:
            raise ValueError("At least one non-empty response is required to save.")

        for qid, text in submitted.items():
            existing_saved = db.execute(
                select(SomInterviewResponse).where(
                    SomInterviewResponse.lvo_cycleid == cycle.lvo_cycleid,
                    SomInterviewResponse.lvo_questionid == qid,
                    SomInterviewResponse.lvo_status == "saved",
                )
            ).scalar_one_or_none()
            if existing_saved:
                existing_saved.lvo_responsetext = text
                existing_saved.lvo_savedat = now
                existing_saved.lvo_capturedby = saved_by
            else:
                db.add(
                    SomInterviewResponse(
                        lvo_responseid=_new_id(),
                        lvo_cycleid=cycle.lvo_cycleid,
                        lvo_questionid=qid,
                        lvo_role=role,
                        lvo_responsetext=text,
                        lvo_status="saved",
                        lvo_capturedby=saved_by,
                        lvo_capturedat=now,
                        lvo_savedat=now,
                    )
                )

        db.execute(
            delete(SomInterviewResponse).where(
                SomInterviewResponse.lvo_cycleid == cycle.lvo_cycleid,
                SomInterviewResponse.lvo_role == role,
                SomInterviewResponse.lvo_status == "draft",
            )
        )

        intent = db.get(SomIntentCard, role)
        if intent is None:
            intent = SomIntentCard(
                lvo_role=role,
                lvo_status="CONFIGURED",
                lvo_configuredat=now,
                lvo_configuredby=saved_by,
                lvo_cycleid=cycle.lvo_cycleid,
            )
            db.add(intent)
        else:
            intent.lvo_status = "CONFIGURED"
            intent.lvo_configuredat = now
            intent.lvo_configuredby = saved_by
            intent.lvo_cycleid = cycle.lvo_cycleid

        rebuild_context_lake(db, saved_by)
        db.commit()
    except ValueError:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        logger.exception("SOM interview save failed for role %s", role)
        raise RuntimeError(ERR_MSG_0021) from None

    return build_interview_setup(db, role)


def list_intent_cards(db: Session) -> list[IntentCardData]:
    if not _has_som_tables(db):
        raise RuntimeError("Sales Operating Model tables are not migrated.")

    cards: list[IntentCardData] = []
    for role in ("national_manager", "regional_manager", "seller_manager"):
        row = db.get(SomIntentCard, role)
        cards.append(
            IntentCardData(
                role=role,
                role_display=ROLE_DISPLAY[role],
                scope_label=SCOPE_LABELS[role],
                status=row.lvo_status if row else "NOT_CONFIGURED",
                configured_at=row.lvo_configuredat if row else None,
            )
        )
    return cards


def get_context_lake(db: Session) -> dict[str, Any]:
    if not _has_som_tables(db):
        raise RuntimeError("Sales Operating Model tables are not migrated.")

    from app.models import SomContextLake

    row = db.get(SomContextLake, 1)
    if row is None or not row.lvo_payload:
        return normalize_context_lake_payload(
            {
                "version": 2,
                "cycleId": None,
                "updatedAt": None,
                "interview": {"roles": {}},
                "organizationalIntents": {},
            }
        )
    return normalize_context_lake_payload(dict(row.lvo_payload))


def list_interview_questions(
    db: Session,
    role: str | None = None,
) -> list[InterviewQuestionRow]:
    if not _has_som_tables(db):
        raise RuntimeError("Sales Operating Model tables are not migrated.")

    stmt = select(SomInterviewQuestion).where(
        SomInterviewQuestion.lvo_isactive.is_(True)
    )
    if role:
        role = validate_role(role)
        stmt = stmt.where(SomInterviewQuestion.lvo_role == role)
    stmt = stmt.order_by(
        SomInterviewQuestion.lvo_role.asc(),
        SomInterviewQuestion.lvo_sortorder.asc(),
    )
    rows = db.execute(stmt).scalars()
    return [
        InterviewQuestionRow(
            question_id=r.lvo_questionid,
            sort_order=r.lvo_sortorder,
            text=r.lvo_questiontext,
            role=r.lvo_role,
        )
        for r in rows
    ]


def create_interview_question(
    db: Session,
    role: str,
    question_text: str,
    sort_order: int | None,
) -> InterviewQuestionRow:
    role = validate_role(role)
    if not question_text.strip():
        raise ValueError("questionText is required.")

    if sort_order is None:
        max_order = db.execute(
            select(SomInterviewQuestion.lvo_sortorder)
            .where(
                SomInterviewQuestion.lvo_role == role,
                SomInterviewQuestion.lvo_isactive.is_(True),
            )
            .order_by(SomInterviewQuestion.lvo_sortorder.desc())
            .limit(1)
        ).scalar_one_or_none()
        sort_order = (max_order or 0) + 1

    now = _utc_now()
    row = SomInterviewQuestion(
        lvo_questionid=_new_id(),
        lvo_role=role,
        lvo_sortorder=sort_order,
        lvo_questiontext=question_text.strip(),
        lvo_isactive=True,
        lvo_createdat=now,
        lvo_updatedat=now,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return InterviewQuestionRow(
        question_id=row.lvo_questionid,
        sort_order=row.lvo_sortorder,
        text=row.lvo_questiontext,
        role=row.lvo_role,
    )


def update_interview_question(
    db: Session,
    question_id: str,
    question_text: str | None,
    sort_order: int | None,
) -> InterviewQuestionRow:
    row = db.get(SomInterviewQuestion, question_id)
    if row is None or not row.lvo_isactive:
        raise LookupError("Interview question not found.")

    if question_text is not None:
        if not question_text.strip():
            raise ValueError("questionText cannot be empty.")
        row.lvo_questiontext = question_text.strip()
    if sort_order is not None:
        row.lvo_sortorder = sort_order
    row.lvo_updatedat = _utc_now()
    db.commit()
    db.refresh(row)
    return InterviewQuestionRow(
        question_id=row.lvo_questionid,
        sort_order=row.lvo_sortorder,
        text=row.lvo_questiontext,
        role=row.lvo_role,
    )


def delete_interview_question(db: Session, question_id: str) -> None:
    row = db.get(SomInterviewQuestion, question_id)
    if row is None or not row.lvo_isactive:
        raise LookupError("Interview question not found.")
    row.lvo_isactive = False
    row.lvo_updatedat = _utc_now()
    db.commit()

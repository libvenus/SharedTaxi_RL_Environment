"""Pre-meeting briefing card — generate, cache, and serve."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.clients.d365_client import (
    D365ClientError,
    fetch_briefing_context,
    fetch_opportunity_risks,
    resolve_contacts_by_email,
)
from app.models.meeting_briefing import (
    TblMeetingBriefing,
    TblMeetingPrepNote,
    TblMeetingPrepTask,
)
from app.models.schedulemeeting import MeetingDetails
from app.services.compliance_audit import write_compliance_audit

INF_MSG_0004 = "INF_MSG_0004"
ERR_MSG_0023 = "ERR_MSG_0023"
ERR_MSG_0025 = "ERR_MSG_0025"

_PRIORITY_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _cache_ttl_hours() -> int:
    return int(os.getenv("BRIEFING_CACHE_TTL_HOURS", "4"))


def _max_summary_words() -> int:
    return int(os.getenv("BRIEFING_MAX_SUMMARY_WORDS", "100"))


def _parse_uuid(value: str) -> UUID:
    return UUID(value.strip())


def _load_meeting(db: Session, meeting_id: str) -> MeetingDetails:
    meeting = (
        db.query(MeetingDetails)
        .filter(MeetingDetails.meeting_id == meeting_id)
        .first()
    )
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found.")
    return meeting


def _verify_seller(meeting: MeetingDetails, seller_id: str) -> UUID:
    seller_uuid = _parse_uuid(seller_id)
    if meeting.seller_id is not None and meeting.seller_id != seller_uuid:
        raise HTTPException(status_code=403, detail="Seller cannot access this meeting.")
    return seller_uuid


def _duration_minutes(start: datetime, end: datetime) -> int:
    return max(int((end - start).total_seconds() // 60), 1)


def _parse_attendee_emails(meeting: MeetingDetails) -> list[str]:
    if not meeting.attendees_emails:
        return []
    return [
        e.strip()
        for e in meeting.attendees_emails.replace(";", ",").split(",")
        if e.strip()
    ]


def _build_attendees(meeting: MeetingDetails, seller_id: str) -> list[dict]:
    emails = _parse_attendee_emails(meeting)
    if not emails:
        return []
    try:
        resolved = resolve_contacts_by_email(emails)
    except D365ClientError:
        resolved = []
    by_email = {r.email.lower(): r for r in resolved}
    out: list[dict] = []
    for email in emails:
        row = by_email.get(email.lower())
        out.append(
            {
                "name": (row.name if row else None) or email,
                "role": row.role if row else None,
                "email": email,
            }
        )
    return out


def _source_from_dict(data: dict | None) -> dict | None:
    if not data:
        return None
    return {
        "source_type": data.get("sourceType") or data.get("source_type"),
        "source_id": data.get("sourceId") or data.get("source_id"),
        "label": data.get("label") or "",
    }


def _derive_prep_tasks(signals: list[dict]) -> list[dict]:
    tasks: list[dict] = []
    for sig in signals:
        summary = (sig.get("summary") or "").lower()
        src = _source_from_dict(sig.get("source"))
        signal_id = sig.get("signalId") or sig.get("signal_id") or ""
        if "inbound email" in (sig.get("whyShown") or sig.get("why_shown") or "").lower():
            tasks.append(
                {
                    "description": "Review and prepare a response for the latest inbound customer email.",
                    "priority": "HIGH",
                    "evidence": sig.get("summary") or "",
                    "confidence": "high",
                    "source": src,
                    "signal_id": signal_id,
                }
            )
        if "pricing" in summary or "revised" in summary:
            tasks.append(
                {
                    "description": "Prepare revised pricing materials referenced in recent customer communication.",
                    "priority": "HIGH",
                    "evidence": sig.get("summary") or "",
                    "confidence": "high",
                    "source": src,
                    "signal_id": signal_id,
                }
            )
        if "activity gap" in (sig.get("whyShown") or sig.get("why_shown") or "").lower():
            tasks.append(
                {
                    "description": "Log outreach activity and confirm next steps before this meeting.",
                    "priority": "MEDIUM",
                    "evidence": sig.get("summary") or "",
                    "confidence": "high",
                    "source": src,
                    "signal_id": signal_id,
                }
            )
        if "close date" in summary or "estimatedclosedate" in summary:
            tasks.append(
                {
                    "description": "Validate close-date assumptions and timeline with the customer.",
                    "priority": "MEDIUM",
                    "evidence": sig.get("summary") or "",
                    "confidence": "high",
                    "source": src,
                    "signal_id": signal_id,
                }
            )
    return tasks


def _derive_talking_points(
    signals: list[dict],
    deal: dict,
    competitors: list[dict] | None,
) -> list[dict]:
    points: list[dict] = []
    order = 0
    stage = deal.get("stage")

    for sig in signals:
        summary = sig.get("summary") or ""
        src = _source_from_dict(sig.get("source"))
        if not src:
            continue
        lower = summary.lower()
        if "email" in lower or "replied" in lower or "pricing" in lower:
            points.append(
                {
                    "talking_point": (
                        f"Open by acknowledging the customer's latest request: {summary[:120]}"
                    ),
                    "why_shown": "Derived from a traceable inbound communication signal.",
                    "sort_order": order,
                    "source": src,
                }
            )
            order += 1
        if "close date" in lower:
            points.append(
                {
                    "talking_point": (
                        f"Confirm timeline alignment against the updated close date noted in CRM."
                    ),
                    "why_shown": summary,
                    "sort_order": order,
                    "source": src,
                }
            )
            order += 1

    if competitors:
        comp = competitors[0]
        src = _source_from_dict(comp.get("source"))
        if src:
            points.append(
                {
                    "talking_point": (
                        f"Differentiate against {comp.get('competitorName') or comp.get('competitor_name')} "
                        f"using verified CRM competitor intelligence."
                    ),
                    "why_shown": "Competitor recorded on this opportunity in D365.",
                    "sort_order": order,
                    "source": src,
                }
            )
            order += 1

    if stage and order < 3:
        for field in deal.get("fields") or []:
            if field.get("fieldName") == "close_date" and field.get("value"):
                src = _source_from_dict(field.get("source"))
                if src:
                    points.append(
                        {
                            "talking_point": (
                                f"Close with a clear path to decision while the deal remains in {stage}."
                            ),
                            "why_shown": f"Close date in D365: {field.get('value')}",
                            "sort_order": order,
                            "source": src,
                        }
                    )
                    order += 1
                break

    return points


def _derive_watch_outs(deal: dict, risks: list) -> list[dict]:
    items: list[dict] = []
    close_date = None
    budget = None
    for field in deal.get("fields") or []:
        if field.get("fieldName") == "close_date":
            close_date = field.get("value")
        if field.get("fieldName") == "budget_confirmed":
            budget = field.get("value")
        src = _source_from_dict(field.get("source"))
        if field.get("fieldName") == "close_date" and close_date and not budget and src:
            items.append(
                {
                    "consideration": (
                        "Close date is approaching but budget approval is not confirmed in D365."
                    ),
                    "why_shown": f"Close date {close_date} with no budget confirmation on record.",
                    "source": src,
                }
            )

    for risk in risks:
        items.append(
            {
                "consideration": risk.message,
                "why_shown": f"D365 deal risk: {risk.category}",
                "source": {
                    "source_type": "d365_deal_risk",
                    "source_id": risk.risk_id or risk.name,
                    "label": risk.name,
                },
            }
        )

    return items


def _persist_prep_tasks(
    db: Session,
    meeting_id: str,
    seller_uuid: UUID,
    briefing_id: int,
    task_defs: list[dict],
) -> None:
    db.query(TblMeetingPrepTask).filter(
        TblMeetingPrepTask.meeting_id == meeting_id,
        TblMeetingPrepTask.seller_id == seller_uuid,
        TblMeetingPrepTask.status == "open",
    ).delete(synchronize_session=False)

    for idx, task in enumerate(
        sorted(task_defs, key=lambda t: _PRIORITY_RANK.get(t["priority"], 9))
    ):
        src = task.get("source") or {}
        db.add(
            TblMeetingPrepTask(
                meeting_id=meeting_id,
                seller_id=seller_uuid,
                briefing_id=briefing_id,
                description=task["description"],
                priority=task["priority"],
                evidence=task["evidence"],
                confidence=task.get("confidence", "high"),
                status="open",
                sort_order=idx,
                source_type=src.get("source_type"),
                source_id=src.get("source_id"),
            )
        )
    db.commit()


def _load_prep_tasks(db: Session, meeting_id: str, seller_uuid: UUID) -> list[dict]:
    rows = (
        db.query(TblMeetingPrepTask)
        .filter(
            TblMeetingPrepTask.meeting_id == meeting_id,
            TblMeetingPrepTask.seller_id == seller_uuid,
        )
        .order_by(TblMeetingPrepTask.sort_order.asc())
        .all()
    )
    return [
        {
            "id": row.id,
            "description": row.description,
            "priority": row.priority,
            "evidence": row.evidence,
            "confidence": row.confidence,
            "done": row.status == "done",
            "source": (
                {
                    "source_type": row.source_type,
                    "source_id": row.source_id,
                    "label": row.description[:80],
                }
                if row.source_type and row.source_id
                else None
            ),
        }
        for row in rows
    ]


def count_open_prep_tasks(db: Session, meeting_id: str, seller_uuid: UUID) -> int:
    return int(
        db.query(func.count())
        .select_from(TblMeetingPrepTask)
        .filter(
            TblMeetingPrepTask.meeting_id == meeting_id,
            TblMeetingPrepTask.seller_id == seller_uuid,
            TblMeetingPrepTask.status == "open",
        )
        .scalar()
        or 0
    )


def _load_prep_notes(db: Session, meeting_id: str, seller_uuid: UUID) -> list[dict]:
    rows = (
        db.query(TblMeetingPrepNote)
        .filter(
            TblMeetingPrepNote.meeting_id == meeting_id,
            TblMeetingPrepNote.seller_id == seller_uuid,
        )
        .order_by(TblMeetingPrepNote.created_at.asc())
        .all()
    )
    return [
        {
            "id": row.id,
            "note_type": row.note_type,
            "body": row.body,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "is_seller_added": True,
        }
        for row in rows
    ]


def _latest_briefing(
    db: Session,
    meeting_id: str,
    seller_uuid: UUID,
) -> TblMeetingBriefing | None:
    return (
        db.query(TblMeetingBriefing)
        .filter(
            TblMeetingBriefing.meeting_id == meeting_id,
            TblMeetingBriefing.seller_id == seller_uuid,
        )
        .order_by(TblMeetingBriefing.generated_at.desc())
        .first()
    )


def _is_fresh(row: TblMeetingBriefing) -> bool:
    if row.generated_at is None:
        return False
    age = _utc_now() - row.generated_at.replace(tzinfo=None)
    return age < timedelta(hours=_cache_ttl_hours())


def generate_briefing(
    db: Session,
    meeting_id: str,
    seller_id: str,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    meeting = _load_meeting(db, meeting_id)
    seller_uuid = _verify_seller(meeting, seller_id)

    if not refresh:
        cached = _latest_briefing(db, meeting_id, seller_uuid)
        if cached and _is_fresh(cached):
            payload = dict(cached.payload)
            payload["my_prep_notes"] = _load_prep_notes(db, meeting_id, seller_uuid)
            payload["prep_tasks"] = _load_prep_tasks(db, meeting_id, seller_uuid)
            return payload

    if not meeting.opportunity_id:
        raise HTTPException(
            status_code=422,
            detail="Meeting is not linked to a CRM opportunity.",
        )

    opp_id = str(meeting.opportunity_id)
    account_id = str(meeting.account_id) if meeting.account_id else None

    try:
        ctx = fetch_briefing_context(
            seller_id,
            opp_id,
            account_id=account_id,
            max_summary_words=_max_summary_words(),
        )
    except D365ClientError as exc:
        raise HTTPException(status_code=502, detail=ERR_MSG_0025) from exc

    account = ctx.get("account") or {}
    deal = ctx.get("deal") or {}
    signals = ctx.get("signals") or []
    competitors = deal.get("competitorIntel") or deal.get("competitor_intel")

    risks = []
    try:
        risks = fetch_opportunity_risks(meeting.opportunity_id)
    except D365ClientError:
        risks = []

    prep_task_defs = _derive_prep_tasks(signals)
    talking_points = _derive_talking_points(signals, deal, competitors)
    watch_outs = _derive_watch_outs(deal, risks)

    generated_at = _utc_now()
    header = {
        "title": meeting.title or "Meeting",
        "start_at": meeting.meeting_start_time,
        "end_at": meeting.meeting_end_time,
        "duration_minutes": _duration_minutes(
            meeting.meeting_start_time,
            meeting.meeting_end_time,
        ),
        "platform": meeting.platform or "In Person",
        "join_url": meeting.meeting_url if (meeting.platform or "").lower().find("team") >= 0 else None,
        "attendees": _build_attendees(meeting, seller_id),
    }

    account_sources = [
        _source_from_dict(f.get("source"))
        for f in account.get("fields") or []
        if f.get("source")
    ]
    account_sources = [s for s in account_sources if s]

    deal_sources = [
        _source_from_dict(f.get("source"))
        for f in deal.get("fields") or []
        if f.get("source")
    ]
    deal_sources = [s for s in deal_sources if s]

    payload: dict[str, Any] = {
        "meeting_id": meeting_id,
        "seller_id": seller_id,
        "generated_at": generated_at,
        "is_ai_generated": True,
        "header": header,
        "account_summary": {
            "paragraph": account.get("paragraph") or "",
            "word_count": account.get("wordCount") or account.get("word_count") or 0,
            "max_words": account.get("maxWords") or account.get("max_words") or _max_summary_words(),
            "gaps": account.get("gaps") or [],
            "unverified_labels": account.get("unverifiedLabels") or account.get("unverified_labels") or [],
            "sources": account_sources,
        },
        "deal_summary": {
            "paragraph": deal.get("paragraph") or "",
            "word_count": deal.get("wordCount") or deal.get("word_count") or 0,
            "max_words": deal.get("maxWords") or deal.get("max_words") or _max_summary_words(),
            "stage": deal.get("stage"),
            "gaps": deal.get("gaps") or [],
            "competitor_intel": {
                "items": competitors or [],
                "message_code": deal.get("competitorMessageCode")
                or deal.get("competitor_message_code"),
            },
            "sources": deal_sources,
        },
        "recent_signals": [
            {
                "signal_id": s.get("signalId") or s.get("signal_id"),
                "summary": s.get("summary"),
                "why_shown": s.get("whyShown") or s.get("why_shown"),
                "event_at": s.get("eventAt") or s.get("event_at"),
                "involved_parties": s.get("involvedParties") or s.get("involved_parties") or [],
                "source": _source_from_dict(s.get("source")),
            }
            for s in signals
            if _source_from_dict(s.get("source"))
        ],
        "prep_tasks": [],
        "talking_points": talking_points,
        "talking_points_message_code": None if talking_points else ERR_MSG_0023,
        "watch_out_for": watch_outs or None,
        "my_prep_notes": _load_prep_notes(db, meeting_id, seller_uuid),
    }

    row = TblMeetingBriefing(
        meeting_id=meeting_id,
        seller_id=seller_uuid,
        payload=payload,
        generated_at=generated_at,
        payload_version="v1",
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    _persist_prep_tasks(db, meeting_id, seller_uuid, row.id, prep_task_defs)
    payload["prep_tasks"] = _load_prep_tasks(db, meeting_id, seller_uuid)
    payload["my_prep_notes"] = _load_prep_notes(db, meeting_id, seller_uuid)

    row.payload = payload
    write_compliance_audit(
        db,
        entity_type="meeting_briefing",
        entity_id=meeting_id,
        action="generate",
        category="ai_automated",
        actor_type="ai",
        changed_by=seller_id,
        diff={"prepTaskCount": len(payload.get("prep_tasks") or [])},
    )
    db.commit()
    return payload


def update_prep_task_status(
    db: Session,
    meeting_id: str,
    seller_id: str,
    task_id: int,
    *,
    done: bool,
) -> dict:
    seller_uuid = _verify_seller(_load_meeting(db, meeting_id), seller_id)
    task = (
        db.query(TblMeetingPrepTask)
        .filter(
            TblMeetingPrepTask.id == task_id,
            TblMeetingPrepTask.meeting_id == meeting_id,
            TblMeetingPrepTask.seller_id == seller_uuid,
        )
        .first()
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Prep task not found.")
    before_status = task.status
    task.status = "done" if done else "open"
    task.completed_at = _utc_now() if done else None
    write_compliance_audit(
        db,
        entity_type="meeting_prep_task",
        entity_id=str(task.id),
        action="update",
        category="seller_action",
        actor_type="seller",
        changed_by=seller_id,
        diff={
            "before": {"status": before_status, "done": before_status == "done"},
            "after": {"status": task.status, "done": done},
            "description": task.description,
        },
    )
    db.commit()
    return {"id": task.id, "done": done}


def create_prep_note(
    db: Session,
    meeting_id: str,
    seller_id: str,
    *,
    body: str,
    note_type: str = "typed",
) -> dict:
    seller_uuid = _verify_seller(_load_meeting(db, meeting_id), seller_id)
    now = _utc_now()
    row = TblMeetingPrepNote(
        meeting_id=meeting_id,
        seller_id=seller_uuid,
        note_type=note_type,
        body=body.strip(),
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {
        "id": row.id,
        "note_type": row.note_type,
        "body": row.body,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "is_seller_added": True,
    }


def update_prep_note(
    db: Session,
    meeting_id: str,
    seller_id: str,
    note_id: int,
    *,
    body: str,
) -> dict:
    seller_uuid = _verify_seller(_load_meeting(db, meeting_id), seller_id)
    row = (
        db.query(TblMeetingPrepNote)
        .filter(
            TblMeetingPrepNote.id == note_id,
            TblMeetingPrepNote.meeting_id == meeting_id,
            TblMeetingPrepNote.seller_id == seller_uuid,
        )
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Prep note not found.")
    row.body = body.strip()
    row.updated_at = _utc_now()
    db.commit()
    db.refresh(row)
    return {
        "id": row.id,
        "note_type": row.note_type,
        "body": row.body,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "is_seller_added": True,
    }


def delete_prep_note(
    db: Session,
    meeting_id: str,
    seller_id: str,
    note_id: int,
) -> None:
    seller_uuid = _verify_seller(_load_meeting(db, meeting_id), seller_id)
    row = (
        db.query(TblMeetingPrepNote)
        .filter(
            TblMeetingPrepNote.id == note_id,
            TblMeetingPrepNote.meeting_id == meeting_id,
            TblMeetingPrepNote.seller_id == seller_uuid,
        )
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Prep note not found.")
    db.delete(row)
    db.commit()

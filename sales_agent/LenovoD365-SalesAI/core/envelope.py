import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel


class Display(BaseModel):
    text: str = ""
    needs_user_input: bool = False


class Action(BaseModel):
    type: str
    status: str = "pending"
    payload: dict[str, Any] = {}


class Meta(BaseModel):
    agent: str = "Orchestrator"
    conversation_id: str = ""
    trace_id: str = ""


class Envelope(BaseModel):
    type: str = "orchestrator_response"
    display: Display
    actions: list[Action] = []
    meta: Meta


_ACTION_TYPE = {
    "schedule": "schedule_meeting",
    "reschedule": "reschedule_meeting",
    "cancel": "cancel_meeting",
}

_MEETING_SEARCH_URL = "http://10.245.240.33/ai-api/meeting-details/meetings/search"
_DEFAULT_ORGANISER = "Lenovo_D365_PoC@sutherlandglobal.com"
_LOGGER = logging.getLogger("orchestrator.envelope")
_IST = timezone(timedelta(hours=5, minutes=30))


def _setup_file_logger() -> None:
    logs_dir = Path(__file__).resolve().parents[1] / "logs"
    logs_dir.mkdir(exist_ok=True)
    path = logs_dir / f"scheduler_debug_{datetime.now().strftime('%Y-%m-%d')}.log"
    if any(isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == path for h in _LOGGER.handlers):
        return
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = True


_setup_file_logger()


def _pick_str(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _canonical_organiser(raw: str) -> str:
    """Normalize organiser email to the canonical Lenovo_D365 form.

    Microsoft Graph returns the mailbox as `LenovoD365_PoC@...` (no underscore
    after 'Lenovo'); the business-facing value must be `Lenovo_D365_PoC@...`.
    """
    value = (raw or "").strip()
    if not value:
        return _DEFAULT_ORGANISER
    if value.lower() == "lenovod365_poc@sutherlandglobal.com":
        return _DEFAULT_ORGANISER
    return value


def _normalize_email_list(raw: str) -> str:
    if not raw:
        return ""
    # Accept values like "John <john@x.com>, jane@x.com" and keep only valid email tokens.
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", raw)
    if not emails:
        return raw.strip()
    seen: set[str] = set()
    normalized: list[str] = []
    for email in emails:
        lowered = email.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(lowered)
    return ",".join(normalized)


def _to_second_z(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    if value.endswith("Z"):
        value = value[:-1]
    try:
        dt = datetime.fromisoformat(value)
        return dt.isoformat(timespec="seconds") + "Z"
    except ValueError:
        return raw


def _to_second_ist(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    try:
        if value.endswith("Z"):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.astimezone(_IST).isoformat(timespec="seconds")

        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_IST)
        else:
            dt = dt.astimezone(_IST)
        return dt.isoformat(timespec="seconds")
    except ValueError:
        return raw


def _to_millis_z(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    if value.endswith("Z"):
        value = value[:-1]
    try:
        dt = datetime.fromisoformat(value)
        return dt.isoformat(timespec="milliseconds") + "Z"
    except ValueError:
        return raw


def _extract_meeting_row(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        for key in ("meeting", "data", "result"):
            child = raw.get(key)
            if isinstance(child, dict):
                return child
            if isinstance(child, list) and child and isinstance(child[0], dict):
                return child[0]
        for key in ("meetings", "items", "results"):
            child = raw.get(key)
            if isinstance(child, list) and child and isinstance(child[0], dict):
                return child[0]
        if any(k in raw for k in ("meeting_id", "event_id", "id")):
            return raw
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw[0]
    return {}


def _search_meeting(payload: dict[str, str]) -> dict[str, Any]:
    _LOGGER.info("scheduler_lookup.start url=%s payload=%s", _MEETING_SEARCH_URL, payload)
    try:
        response = httpx.post(
            _MEETING_SEARCH_URL,
            json=payload,
            headers={"Content-Type": "application/json", "accept": "application/json"},
            timeout=30.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        raw_json = response.json()
        row = _extract_meeting_row(raw_json)
        _LOGGER.info(
            "scheduler_lookup.raw_response status=%s full_response=%s",
            response.status_code,
            raw_json,   # full JSON printed
        )
        _LOGGER.info(
            "scheduler_lookup.extracted_row row=%s",
            row,
        )
        if not row:
            _LOGGER.warning("scheduler_lookup.empty_row full_response=%s", raw_json)
        return row
    except (httpx.HTTPError, ValueError) as exc:
        _LOGGER.exception("scheduler_lookup.error url=%s error=%s", _MEETING_SEARCH_URL, exc)
        return {}


def _normalized_scheduler_payload(raw_action: str, api_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(api_payload)
    canonical = {
        "schedule": "Schedule",
        "reschedule": "Reschedule",
        "cancel": "Cancel",
    }

    if raw_action in {"reschedule", "cancel"}:
        organiser_for_search = _pick_str(
            payload,
            "organiser_name",
            "organizer_name",
            "organiser_email",
            "organizer_email",
        ) or _DEFAULT_ORGANISER
        attendees_for_search = _normalize_email_list(
            _pick_str(
                payload,
                "attendees",
                "attendees_emails",
                "attendee_email",
                "required_attendees",
                "recipient_email",
            )
        )
        meeting_start_for_search = _to_millis_z(
            _pick_str(
                payload,
                "meeting_start_time",
                "meeting_start",
                "start_time",
                "start",
                "meetingStartTime",
                "current_meeting_start",
            )
        )
        search_request = {
            "organiser_name": organiser_for_search,
            "meeting_start_time": meeting_start_for_search,
            "attendees_emails": attendees_for_search,
            "title": _pick_str(payload, "title", "meeting_subject", "subject"),
        }
        _LOGGER.info("scheduler_payload.normalize action=%s search_request=%s", raw_action, search_request)
        if search_request["organiser_name"] and search_request["meeting_start_time"] and search_request["attendees_emails"]:
            source = _search_meeting(search_request)
            if source:
                payload["meeting_id"] = payload.get("meeting_id") or _pick_str(source, "meeting_id", "event_id", "id", "meetingId")
                payload["meeting_start_time"] = payload.get("meeting_start_time") or _pick_str(source, "meeting_start_time", "start_time", "start", "meetingStartTime")
                payload["meeting_end_time"] = payload.get("meeting_end_time") or _pick_str(source, "meeting_end_time", "end_time", "end", "meetingEndTime")
                payload["platform"] = payload.get("platform") or _pick_str(source, "platform") or "Teams"
                payload["title"] = payload.get("title") or _pick_str(source, "title", "subject") or "Meeting"
                payload["account_name"] = payload.get("account_name") or _pick_str(source, "account_name", "accountName")
                payload["attendees"] = payload.get("attendees") or _pick_str(source, "attendees", "attendees_emails", "required_attendees", "attendee_email")
                payload["organiser_name"] = payload.get("organiser_name") or _pick_str(source, "organiser_name", "organizer_name", "organiser_email", "organizer_email")
                payload["body"] = payload.get("body") or _pick_str(source, "body", "agenda")
                payload["recurrence_pattern"] = payload.get("recurrence_pattern") or _pick_str(source, "recurrence_pattern")
                payload["recurrence_interval"] = payload.get("recurrence_interval") or source.get("recurrence_interval", 0)
                payload["recurrence_start_date"] = payload.get("recurrence_start_date") or _pick_str(source, "recurrence_start_date")
                payload["recurrence_end_date"] = payload.get("recurrence_end_date") or _pick_str(source, "recurrence_end_date")
            else:
                _LOGGER.warning("scheduler_payload.lookup_empty action=%s search_request=%s", raw_action, search_request)
        else:
            _LOGGER.warning("scheduler_payload.lookup_skipped action=%s reason=missing_required_search_fields", raw_action)

    raw_start = _pick_str(payload, "meeting_start_time")
    raw_end = _pick_str(payload, "meeting_end_time")
    payload["meeting_start_time_utc"] = _to_second_z(raw_start)
    payload["meeting_end_time_utc"] = _to_second_z(raw_end)
    payload["meeting_start_time"] = _to_second_ist(raw_start)
    payload["meeting_end_time"] = _to_second_ist(raw_end)
    payload["platform"] = _pick_str(payload, "platform") or "Teams"
    payload["title"] = _pick_str(payload, "title") or "Meeting"
    payload["account_name"] = _pick_str(payload, "account_name")
    payload["attendees"] = _normalize_email_list(_pick_str(payload, "attendees", "attendees_emails", "attendee_email", "required_attendees", "recipient_email"))
    payload["organiser_name"] = _canonical_organiser(
        _pick_str(payload, "organiser_name", "organizer_name", "organiser_email", "organizer_email")
    )
    payload["body"] = _pick_str(payload, "body")
    payload["recurrence_pattern"] = _pick_str(payload, "recurrence_pattern")
    payload["recurrence_interval"] = int(payload.get("recurrence_interval") or 0)

    start_date = payload["meeting_start_time"][:10] if payload["meeting_start_time"] else ""
    payload["recurrence_start_date"] = _pick_str(payload, "recurrence_start_date") or start_date
    payload["recurrence_end_date"] = _pick_str(payload, "recurrence_end_date") or start_date
    payload["action"] = canonical.get(raw_action, _pick_str(payload, "action") or "Schedule")
    _LOGGER.info(
        "scheduler_payload.normalized action=%s meeting_id=%s start=%s attendees=%s",
        payload["action"],
        _pick_str(payload, "meeting_id"),
        payload["meeting_start_time"],
        payload["attendees"],
    )
    return payload


def _looks_like_scheduler(data: Any) -> bool:
    return isinstance(data, dict) and (
        data.get("agent_name") == "SchedulerAgent"
        or "api_payload" in data
        or "ai_reply" in data
    )


def _should_block_event_id_prompt(data: dict[str, Any]) -> bool:
    status = str(data.get("status") or "").lower()
    action = str(data.get("action") or "").lower()
    api_payload = data.get("api_payload")
    ai_reply = str(data.get("ai_reply") or "").lower()
    return (
        status == "pending"
        and action == "none"
        and not isinstance(api_payload, dict)
        and "event id" in ai_reply
    )


def _scheduler_envelope(data: dict, meta: Meta) -> Envelope:
    meta.agent = data.get("agent_name") or "SchedulerAgent"
    status = str(data.get("status") or "pending")
    display_text = str(data.get("ai_reply") or "")

    if _should_block_event_id_prompt(data):
        _LOGGER.warning(
            "scheduler_guard.blocked_event_id_prompt conversation_id=%s trace_id=%s",
            meta.conversation_id,
            meta.trace_id,
        )
        status = "error"
        display_text = (
            "I could not auto-resolve the meeting from search results, so I cannot proceed yet. "
            "Please confirm organiser email, attendee email, and exact meeting start time (ISO), then retry."
        )

    display = Display(text=display_text, needs_user_input=status == "pending")

    actions: list[Action] = []
    api_payload = data.get("api_payload")
    if isinstance(api_payload, dict):
        raw_action = str(data.get("action") or api_payload.get("action") or "").lower()
        normalized_payload = _normalized_scheduler_payload(raw_action, api_payload)
        actions.append(
            Action(
                type=_ACTION_TYPE.get(raw_action, raw_action or "scheduler_action"),
                status=status,
                payload=normalized_payload,
            )
        )

    return Envelope(display=display, actions=actions, meta=meta)


def build_envelope(raw_text: str, conversation_id: str = "", trace_id: str = "") -> Envelope:
    """Map a final orchestrator output string into the wire envelope."""
    meta = Meta(conversation_id=conversation_id, trace_id=trace_id)
    text = (raw_text or "").strip()

    data: Any = None
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            data = None

    if _looks_like_scheduler(data):
        return _scheduler_envelope(data, meta)

    # Any other JSON object that leaked through (e.g. a sales fallback): pull a
    # readable field instead of dumping raw JSON into the chat bubble.
    if isinstance(data, dict):
        for key in ("reply", "message", "text", "ai_reply"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return Envelope(display=Display(text=value), meta=meta)

    return Envelope(display=Display(text=text), meta=meta)

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import httpx
from agent_framework import tool
from pydantic import Field

_DB_URL = "http://10.245.240.33/ai-api/meeting-details/"
_MEETING_SEARCH_URL = "http://10.245.240.33/ai-api/meeting-details/meetings/search"
_DEFAULT_ORGANISER = "Lenovo_D365_PoC@sutherlandglobal.com"
_LOGGER = logging.getLogger("scheduler.db_tool")


def _setup_file_logger() -> None:
    logs_dir = Path(__file__).resolve().parents[2] / "logs"
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


def _missing_search_fields(payload: dict) -> list[str]:
    missing: list[str] = []
    for key in ("meeting_start_time", "attendees_emails"):
        value = payload.get(key)
        if value is None or not str(value).strip():
            missing.append(key)
    return missing


def _to_z_datetime(raw: str) -> str:
    """Normalize datetimes to YYYY-MM-DDTHH:MM:SSZ."""
    if not raw:
        return ""
    value = raw.strip()
    if value.endswith("Z"):
        return value
    try:
        dt = datetime.fromisoformat(value)
        return dt.isoformat(timespec="seconds") + "Z"
    except ValueError:
        return value


def _extract_meeting_rows(raw_text: str) -> list[dict]:
    try:
        raw = json.loads(raw_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []

    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]

    if isinstance(raw, dict):
        for key in ("meetings", "items", "results", "data"):
            value = raw.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
        if any(k in raw for k in ("meeting_start_time", "meeting_id", "event_id", "id")):
            return [raw]

    return []


def _pick_start_time(row: dict) -> str:
    for key in ("meeting_start_time", "start_time", "start", "meetingStartTime"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _score_row_for_target_start(row_start: str, target_start: str) -> float:
    if not row_start:
        return 10**18
    try:
        a = datetime.fromisoformat(row_start.replace("Z", "+00:00"))
        b = datetime.fromisoformat(target_start.replace("Z", "+00:00"))
        return abs((a - b).total_seconds())
    except ValueError:
        # Keep deterministic fallback ordering for non-ISO rows.
        return 10**12


def _infer_meeting_start_time(
    organiser_name: str,
    attendees_emails: str,
    title: str,
    target_start_time: str,
) -> str:
    params = {
        "organiser_name": organiser_name,
        "attendees": attendees_emails,
        "title": title,
        "action": "schedule",
    }
    status, resp = _db_get(params)
    if status is None or status >= 400:
        return ""

    rows = _extract_meeting_rows(resp)
    if not rows:
        return ""

    if target_start_time:
        rows = sorted(rows, key=lambda row: _score_row_for_target_start(_pick_start_time(row), target_start_time))

    candidate = _pick_start_time(rows[0])
    return _to_z_datetime(candidate)


def _db_post(payload: dict) -> tuple[int | None, str]:
    _LOGGER.info("db_post.start url=%s payload=%s", _DB_URL, payload)
    try:
        r = httpx.post(
            _DB_URL,
            json=payload,
            headers={"Content-Type": "application/json", "accept": "application/json"},
            timeout=30.0,
            follow_redirects=True,
        )
        text = r.text or "(empty)"
        _LOGGER.info("db_post.done url=%s status=%s response_preview=%s", _DB_URL, r.status_code, text[:300])
        return r.status_code, text
    except httpx.HTTPError as exc:
        _LOGGER.exception("db_post.error url=%s error=%s", _DB_URL, exc)
        return None, f"DB POST failed: {exc!r}"


def _db_get(params: dict) -> tuple[int | None, str]:
    filtered = {k: v for k, v in params.items() if v not in (None, "")}
    _LOGGER.info("db_get.start url=%s params=%s", _DB_URL, filtered)
    try:
        r = httpx.get(
            _DB_URL,
            params=filtered,
            headers={"accept": "application/json"},
            timeout=30.0,
            follow_redirects=True,
        )
        text = r.text or "(empty)"
        _LOGGER.info("db_get.done url=%s status=%s response_preview=%s", _DB_URL, r.status_code, text[:300])
        return r.status_code, text
    except httpx.HTTPError as exc:
        _LOGGER.exception("db_get.error url=%s error=%s", _DB_URL, exc)
        return None, f"DB GET failed: {exc!r}"


def _meeting_search_post(payload: dict) -> tuple[int | None, str]:
    payload = dict(payload)
    payload["organiser_name"] = str(payload.get("organiser_name") or "").strip() or _DEFAULT_ORGANISER
    missing = _missing_search_fields(payload)
    if missing:
        msg = f"Meeting search skipped: missing required fields: {', '.join(missing)}"
        _LOGGER.warning("meeting_search.skipped reason=missing_fields missing=%s payload=%s", missing, payload)
        return None, msg

    _LOGGER.info("meeting_search.start url=%s payload=%s", _MEETING_SEARCH_URL, payload)
    try:
        r = httpx.post(
            _MEETING_SEARCH_URL,
            json=payload,
            headers={"Content-Type": "application/json", "accept": "application/json"},
            timeout=30.0,
            follow_redirects=True,
        )
        text = r.text or "(empty)"
        _LOGGER.info(
            "meeting_search.done url=%s status=%s full_response=%s",
            _MEETING_SEARCH_URL,
            r.status_code,
            text,  # full response printed
        )
        return r.status_code, text
    except httpx.HTTPError as exc:
        _LOGGER.exception("meeting_search.error url=%s error=%s", _MEETING_SEARCH_URL, exc)
        return None, f"Meeting search failed: {exc!r}"


@tool(approval_mode="never_require")
def post_meeting_details_to_db(
    meeting_start_time: Annotated[str, Field(description="Start datetime in ISO format")],
    meeting_end_time: Annotated[str, Field(description="End datetime in ISO format")],
    platform: Annotated[str, Field(description="Meeting platform, e.g. Teams")],
    title: Annotated[str, Field(description="Meeting title")],
    account_name: Annotated[str, Field(description="Account name")] = "",
    attendees: Annotated[str, Field(description="Comma-separated attendee emails")] = "",
    organiser_name: Annotated[str, Field(description="Organizer email")] = "nazeerahmed.m@sutherlandglobal.com",
    action: Annotated[str, Field(description="schedule | reschedule | delete")] = "schedule",
    body: Annotated[str, Field(description="Agenda/body text")] = "",
    recurrence_pattern: Annotated[str, Field(description="'Daily' | 'Weekly' | 'Monthly' | '' for one-off")] = "",
    recurrence_interval: Annotated[int, Field(description="Repeat every N units of recurrence_pattern. 0 for non-recurring.")] = 0,
    recurrence_start_date: Annotated[str, Field(description="Start date of recurrence series. Format: YYYY-MM-DD.")] = "",
    recurrence_end_date: Annotated[str, Field(description="End date of recurrence series. Format: YYYY-MM-DD.")] = "",
) -> str:
    """Persist meeting details to DB for cross-chat lookup."""
    payload = {
        "meeting_start_time": _to_z_datetime(meeting_start_time),
        "meeting_end_time": _to_z_datetime(meeting_end_time),
        "platform": platform or "Teams",
        "title": title or "Meeting",
        "account_name": account_name,
        "attendees": attendees,
        "organiser_name": organiser_name,
        "action": action,
        "body": body or title or "Meeting",
        "recurrence_pattern": recurrence_pattern or "none",
        "recurrence_interval": recurrence_interval,
        "recurrence_start_date": recurrence_start_date,
        "recurrence_end_date": recurrence_end_date,
    }
    status, resp = _db_post(payload)
    if status is None:
        return f"DB error: {resp}"
    if status >= 400:
        return f"DB returned {status}: {resp}"
    return f"Saved to DB (HTTP {status}): {resp}"


@tool(approval_mode="never_require")
def get_meeting_details_from_db(
    organiser_name: Annotated[str, Field(description="Organizer email to filter")],
    attendees: Annotated[str, Field(description="Attendee email/name filter")],
    title: Annotated[str, Field(description="Title filter")] = "",
    action: Annotated[str, Field(description="schedule | reschedule | delete")] = "",
) -> str:
    """Query meetings from DB endpoint."""
    params = {
        "organiser_name": organiser_name,
        "attendees": attendees,
        "title": title,
        "action": action,
    }
    status, resp = _db_get(params)
    if status is None:
        return f"DB error: {resp}"
    if status >= 400:
        return f"DB returned {status}: {resp}"
    return resp


@tool(approval_mode="never_require")
def search_meeting_details(
    organiser_name: Annotated[str, Field(description="Organizer email")] = "",
    meeting_start_time: Annotated[str, Field(description="Meeting start datetime in ISO format, e.g. 2026-06-23T21:30:00.000Z")] = "",
    attendees_emails: Annotated[str, Field(description="Comma-separated attendee email(s)")] = "",
    title: Annotated[str, Field(description="Meeting title filter. Empty is allowed")] = "",
) -> str:
    """Find a previously scheduled meeting and return raw API response with event/meeting id."""
    normalized_attendees = ",".join(
        part.strip().lower()
        for part in str(attendees_emails or "").split(",")
        if part and part.strip()
    )
    normalized_start = _to_z_datetime(meeting_start_time)
    organiser_primary = str(organiser_name or "").strip() or _DEFAULT_ORGANISER

    organisers_to_try = [organiser_primary]
    if "LenovoD365" in organiser_primary:
        organisers_to_try.append(organiser_primary.replace("LenovoD365", "Lenovo_D365"))
    elif "Lenovo_D365" in organiser_primary:
        organisers_to_try.append(organiser_primary.replace("Lenovo_D365", "LenovoD365"))

    for organiser in dict.fromkeys(organisers_to_try):
        payload = {
            "organiser_name": organiser,
            "meeting_start_time": normalized_start,
            "attendees_emails": normalized_attendees,
            "title": title,
        }
        status, resp = _meeting_search_post(payload)
        if status is not None and status < 400:
            return resp

    inferred_start = _infer_meeting_start_time(
        organiser_name=organiser_primary,
        attendees_emails=normalized_attendees,
        title=title,
        target_start_time=normalized_start,
    )
    if inferred_start and inferred_start != normalized_start:
        _LOGGER.info(
            "meeting_search.retry_inferred_start original_start=%s inferred_start=%s",
            normalized_start,
            inferred_start,
        )
        for organiser in dict.fromkeys(organisers_to_try):
            payload = {
                "organiser_name": organiser,
                "meeting_start_time": inferred_start,
                "attendees_emails": normalized_attendees,
                "title": title,
            }
            status, resp = _meeting_search_post(payload)
            if status is not None and status < 400:
                return resp

    if status is None:
        return f"Meeting search error: {resp}"
    return f"Meeting search returned {status}: {resp}"



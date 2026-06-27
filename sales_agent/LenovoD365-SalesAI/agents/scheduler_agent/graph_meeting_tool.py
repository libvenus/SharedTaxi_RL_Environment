import os
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import httpx
from agent_framework import tool
from dotenv import load_dotenv
from pydantic import Field

# Load environment variables from .env or .github/.env
_ROOT = Path(__file__).resolve().parents[2]  # project root (c:\work\bot)
load_dotenv(_ROOT / ".env", override=False)
load_dotenv(_ROOT / ".github" / ".env", override=False)


GRAPH_SCOPE = "https://graph.microsoft.com/.default"
GRAPH_TOKEN_URL_TMPL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
GRAPH_EVENTS_URL_TMPL = "https://graph.microsoft.com/v1.0/users/{user_id}/events"
_AIBACKEND_BASE = os.getenv("AIBACKEND_BASE_URL", "http://aibackend:9101").rstrip("/")
MEETING_SEARCH_URL = f"{_AIBACKEND_BASE}/ai-api/meeting-details/meetings/search"
MEETING_DETAILS_URL = f"{_AIBACKEND_BASE}/ai-api/meeting-details/"
DEFAULT_ORGANISER = "Lenovo_D365_PoC@sutherlandglobal.com"
_LOGGER = logging.getLogger("scheduler.graph")


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

# Microsoft Graph requires Windows timezone identifiers, not IANA.
_IANA_TO_WINDOWS = {
    "Asia/Kolkata": "India Standard Time",
    "Asia/Calcutta": "India Standard Time",
    "US/Eastern": "Eastern Standard Time",
    "America/New_York": "Eastern Standard Time",
    "US/Central": "Central Standard Time",
    "America/Chicago": "Central Standard Time",
    "US/Mountain": "Mountain Standard Time",
    "America/Denver": "Mountain Standard Time",
    "US/Pacific": "Pacific Standard Time",
    "America/Los_Angeles": "Pacific Standard Time",
    "Europe/London": "GMT Standard Time",
    "Europe/Berlin": "W. Europe Standard Time",
    "Europe/Paris": "Romance Standard Time",
    "Asia/Tokyo": "Tokyo Standard Time",
    "Asia/Shanghai": "China Standard Time",
    "Asia/Singapore": "Singapore Standard Time",
    "Australia/Sydney": "AUS Eastern Standard Time",
    "Pacific/Auckland": "New Zealand Standard Time",
    "UTC": "UTC",
}


def _to_windows_tz(iana_tz: str) -> str:
    return _IANA_TO_WINDOWS.get(iana_tz, iana_tz)


def _required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _ssl_verify() -> bool:
    raw = (os.getenv("GRAPH_VERIFY_SSL") or "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_user_guid(token: str, upn_or_guid: str) -> str:
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', upn_or_guid, re.I):
        return upn_or_guid
    r = httpx.get(
        f"https://graph.microsoft.com/v1.0/users/{upn_or_guid}?$select=id",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0, verify=_ssl_verify(),
    )
    r.raise_for_status()
    return r.json().get("id", upn_or_guid)


def _apply_lobby_bypass(token: str, user_id: str, join_url: str) -> bool:
    try:
        user_guid = _resolve_user_guid(token, user_id)
        search = httpx.get(
            f"https://graph.microsoft.com/v1.0/users/{user_guid}/onlineMeetings",
            params={"$filter": f"joinWebUrl eq '{join_url}'"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0, verify=_ssl_verify(),
        )
        meetings = (search.json() or {}).get("value", [])
        if not meetings:
            return False
        om_id = meetings[0].get("id", "")
        if not om_id:
            return False
        patch = httpx.patch(
            f"https://graph.microsoft.com/v1.0/users/{user_guid}/onlineMeetings/{om_id}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"lobbyBypassSettings": {"scope": "everyone", "isDialInBypassEnabled": True},
                  "allowedPresenters": "everyone"},
            timeout=15.0, verify=_ssl_verify(),
        )
        return patch.status_code == 200
    except Exception as exc:
        _LOGGER.warning("graph.lobby_bypass_failed error=%s", exc)
        return False


def _get_access_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    token_url = GRAPH_TOKEN_URL_TMPL.format(tenant_id=tenant_id)
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": GRAPH_SCOPE,
    }
    resp = httpx.post(token_url, data=payload, timeout=30.0, verify=_ssl_verify())
    resp.raise_for_status()
    token = (resp.json() or {}).get("access_token", "")
    if not token:
        raise RuntimeError("Graph token response did not include access_token")
    return token


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
        if any(k in raw for k in ("meeting_id", "event_id", "id", "meetingId")):
            return raw
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw[0]
    return {}


def _pick_str(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_attendees(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if not emails:
        return ",".join(part.strip().lower() for part in text.split(",") if part and part.strip())
    return ",".join(dict.fromkeys(email.lower() for email in emails))


def _to_second_z(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    if value.endswith("Z"):
        value = value[:-1]
    try:
        dt = datetime.fromisoformat(value)
        return dt.isoformat(timespec="seconds") + "Z"
    except ValueError:
        return str(raw or "")


def _build_scheduler_envelope(action_type: str, status: str, display_text: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "orchestrator_response",
        "display": {
            "text": display_text,
            "needs_user_input": status != "success",
        },
        "actions": [
            {
                "type": action_type,
                "status": status,
                "payload": payload,
            }
        ],
        "meta": {
            "agent": "SchedulerAgent",
            "conversation_id": "conv_default",
            "trace_id": "",
        },
    }


def _persist_scheduler_envelope(envelope: dict[str, Any]) -> None:
    """Best-effort persistence for scheduler actions to meeting-details endpoint."""
    try:
        _LOGGER.info("scheduler.persist.start url=%s payload=%s", MEETING_DETAILS_URL, envelope)
        response = httpx.post(
            MEETING_DETAILS_URL,
            json=envelope,
            headers={"Content-Type": "application/json", "accept": "application/json"},
            timeout=30.0,
            follow_redirects=True,
        )
        preview = (response.text or "")[:300]
        _LOGGER.info("scheduler.persist.done url=%s status=%s response_preview=%s", MEETING_DETAILS_URL, response.status_code, preview)
    except httpx.HTTPError as exc:
        _LOGGER.exception("scheduler.persist.error url=%s error=%s", MEETING_DETAILS_URL, exc)


def _extract_rows(raw_text: str) -> list[dict[str, Any]]:
    try:
        raw = json.loads(raw_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []

    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]

    if isinstance(raw, dict):
        for key in ("meetings", "items", "results", "data"):
            value = raw.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if any(k in raw for k in ("meeting_start_time", "meeting_id", "event_id", "id", "meetingId")):
            return [raw]

    return []


def _pick_start_time_from_row(row: dict[str, Any]) -> str:
    return _pick_str(row, "meeting_start_time", "start_time", "start", "meetingStartTime")


def _score_start(row_start: str, requested_start: str) -> float:
    if not row_start:
        return 10**18
    try:
        a = datetime.fromisoformat(row_start.replace("Z", "+00:00"))
        b = datetime.fromisoformat(requested_start.replace("Z", "+00:00"))
        return abs((a - b).total_seconds())
    except ValueError:
        return 10**12


def _infer_start_time_from_details(
    organiser_name: str,
    attendees_emails: str,
    title: str,
    requested_start: str,
) -> str:
    params = {
        "organiser_name": organiser_name,
        "attendees": attendees_emails,
        "title": title,
        "action": "schedule",
    }
    _LOGGER.info("meeting_details_lookup.start url=%s params=%s", MEETING_DETAILS_URL, params)
    try:
        response = httpx.get(
            MEETING_DETAILS_URL,
            params={k: v for k, v in params.items() if str(v or "").strip()},
            headers={"accept": "application/json"},
            timeout=30.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        rows = _extract_rows(response.text or "")
        _LOGGER.info("meeting_details_lookup.done status=%s rows=%s", response.status_code, len(rows))
        if not rows:
            return ""

        if requested_start:
            rows = sorted(rows, key=lambda row: _score_start(_pick_start_time_from_row(row), requested_start))

        inferred = _to_second_z(_pick_start_time_from_row(rows[0]))
        _LOGGER.info("meeting_details_lookup.inferred_start requested=%s inferred=%s", requested_start, inferred)
        return inferred
    except (httpx.HTTPError, ValueError) as exc:
        _LOGGER.exception("meeting_details_lookup.error url=%s error=%s", MEETING_DETAILS_URL, exc)
        return ""


def _lookup_event_id(
    organiser_name: str,
    meeting_start_time: str,
    attendees_emails: str,
    title: str,
) -> tuple[str, dict[str, Any], str]:
    resolved_organiser = organiser_name.strip() or DEFAULT_ORGANISER
    normalized_attendees = _normalize_attendees(attendees_emails)
    normalized_start = _to_second_z(meeting_start_time)
    request_payload = {
        "organiser_name": resolved_organiser,
        "meeting_start_time": normalized_start,
        "attendees_emails": normalized_attendees,
        "title": title.strip(),
    }

    missing = [
        key
        for key in ("meeting_start_time", "attendees_emails")
        if not request_payload.get(key)
    ]
    if missing:
        _LOGGER.warning("meeting_lookup.skipped reason=missing_fields missing=%s payload=%s", missing, request_payload)
        return "", {}, f"Meeting search skipped: missing required fields: {', '.join(missing)}"

    organisers_to_try = [resolved_organiser]
    if "LenovoD365" in resolved_organiser:
        organisers_to_try.append(resolved_organiser.replace("LenovoD365", "Lenovo_D365"))
    elif "Lenovo_D365" in resolved_organiser:
        organisers_to_try.append(resolved_organiser.replace("Lenovo_D365", "LenovoD365"))

    last_row: dict[str, Any] = {}
    last_error = "Meeting found but event id is missing in response."

    def _search_with_start(candidate_start: str, attempt_seed: int = 0) -> tuple[str, dict[str, Any], str]:
        local_last_row: dict[str, Any] = {}
        for idx, organiser in enumerate(dict.fromkeys(organisers_to_try), start=1):
            attempt_payload = dict(request_payload)
            attempt_payload["organiser_name"] = organiser
            attempt_payload["meeting_start_time"] = candidate_start
            _LOGGER.info(
                "meeting_lookup.start attempt=%s url=%s payload=%s",
                attempt_seed + idx,
                MEETING_SEARCH_URL,
                attempt_payload,
            )
            try:
                response = httpx.post(
                    MEETING_SEARCH_URL,
                    json=attempt_payload,
                    headers={"Content-Type": "application/json", "accept": "application/json"},
                    timeout=30.0,
                    follow_redirects=True,
                )
                response.raise_for_status()
                raw_json = response.json()
                row = _extract_meeting_row(raw_json)
                local_last_row = row
                event_id = _pick_str(row, "meeting_id", "event_id", "id", "meetingId")
                _LOGGER.info(
                    "meeting_lookup.raw_response attempt=%s status=%s full_response=%s",
                    attempt_seed + idx,
                    response.status_code,
                    raw_json,
                )
                _LOGGER.info("meeting_lookup.extracted_row attempt=%s row=%s event_id=%s", attempt_seed + idx, row, event_id)
                if event_id:
                    return event_id, row, ""
            except (httpx.HTTPError, ValueError) as exc:
                _LOGGER.exception(
                    "meeting_lookup.error attempt=%s url=%s error=%s",
                    attempt_seed + idx,
                    MEETING_SEARCH_URL,
                    exc,
                )
        return "", local_last_row, "Meeting found but event id is missing in response."

    event_id, last_row, last_error = _search_with_start(normalized_start, attempt_seed=0)
    if event_id:
        return event_id, last_row, ""

    inferred_start = _infer_start_time_from_details(
        organiser_name=resolved_organiser,
        attendees_emails=normalized_attendees,
        title=title,
        requested_start=normalized_start,
    )
    if inferred_start and inferred_start != normalized_start:
        _LOGGER.info("meeting_lookup.retry_with_inferred_start original_start=%s inferred_start=%s", normalized_start, inferred_start)
        event_id, last_row, last_error = _search_with_start(inferred_start, attempt_seed=100)
        if event_id:
            return event_id, last_row, ""

    _LOGGER.warning("meeting_lookup.no_event_id tried_organisers=%s last_row=%s", organisers_to_try, last_row)
    return "", last_row, last_error


@tool(approval_mode="never_require")
def create_teams_meeting_invite(
    recipient_email: Annotated[str, Field(description="One or more recipient email addresses, comma-separated. Example: 'alice@example.com' or 'alice@example.com,bob@example.com'.")],
    meeting_start: Annotated[str, Field(description="Meeting start in format YYYY-MM-DD HH:MM. Example: '2026-06-14 14:00'.")],
    meeting_end: Annotated[str, Field(description="Meeting end in format YYYY-MM-DD HH:MM. Example: '2026-06-14 15:00'.")],
    meeting_subject: Annotated[str, Field(description="Meeting subject/title.")] = "Teams Meeting: Calendar Invitation",
    timezone: Annotated[str, Field(description="IANA timezone name. Example: 'Asia/Kolkata'.")] = "Asia/Kolkata",
    recipient_name: Annotated[str, Field(description="Recipient display name(s), comma-separated matching recipient_email order. Used when only one attendee or as fallback.")] = "Required Attendee",
    body_html: Annotated[str, Field(description="Optional HTML body. If empty, a default body is generated.")] = "",
) -> str:
    """Create a Teams-enabled calendar event via Microsoft Graph and send invites to one or more recipients."""
    try:
        tenant_id = _required_env("TENANT_ID")
        client_id = _required_env("CLIENT_ID")
        client_secret = _required_env("CLIENT_SECRET")
        user_id = _required_env("USER_ID")

        start_time = datetime.strptime(meeting_start, "%Y-%m-%d %H:%M")
        end_time = datetime.strptime(meeting_end, "%Y-%m-%d %H:%M")
        if end_time <= start_time:
            return "Invalid input: meeting_end must be after meeting_start."

        token = _get_access_token(tenant_id, client_id, client_secret)

        start_dt = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        end_dt = end_time.strftime("%Y-%m-%dT%H:%M:%S")

        win_tz = _to_windows_tz(timezone)

        content = body_html.strip() or (
            "<p>Teams Meeting Scheduled</p>"
            f"<p><b>Subject:</b> {meeting_subject}</p>"
            f"<p><b>Date:</b> {start_time.strftime('%B %d, %Y')}</p>"
            f"<p><b>Time:</b> {start_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')} ({timezone})</p>"
        )

        # Build attendee list — support comma-separated emails for multiple attendees
        email_list = [e.strip() for e in recipient_email.split(",") if e.strip()]
        name_list = [n.strip() for n in recipient_name.split(",") if n.strip()]
        attendees = [
            {
                "emailAddress": {
                    "address": email,
                    "name": name_list[i] if i < len(name_list) else email.split("@")[0],
                },
                "type": "required",
            }
            for i, email in enumerate(email_list)
        ]
        if not attendees:
            return "Invalid input: recipient_email must contain at least one valid email address."

        event_body = {
            "subject": meeting_subject,
            "start": {"dateTime": start_dt, "timeZone": win_tz},
            "end": {"dateTime": end_dt, "timeZone": win_tz},
            "attendees": attendees,
            "isOnlineMeeting": True,
            "onlineMeetingProvider": "teamsForBusiness",
            "isReminderOn": True,
            "reminderMinutesBeforeStart": 15,
            "body": {
                "contentType": "HTML",
                "content": content,
            },
        }

        events_url = GRAPH_EVENTS_URL_TMPL.format(user_id=user_id)
        _LOGGER.info(
            "graph.schedule.start url=%s recipients=%s meeting_start=%s meeting_end=%s",
            events_url,
            recipient_email,
            meeting_start,
            meeting_end,
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        response = httpx.post(
            events_url,
            headers=headers,
            json=event_body,
            timeout=45.0,
            verify=_ssl_verify(),
        )
        response.raise_for_status()
        _LOGGER.info("graph.schedule.done url=%s status=%s", events_url, response.status_code)
        event = response.json() or {}

        event_id = event.get("id", "")
        online = event.get("onlineMeeting", {}) if isinstance(event.get("onlineMeeting", {}), dict) else {}
        join_url = online.get("joinUrl", "")

        organizer = event.get("organizer", {}) if isinstance(event.get("organizer", {}), dict) else {}
        organizer_email_obj = organizer.get("emailAddress", {}) if isinstance(organizer.get("emailAddress", {}), dict) else {}
        organiser_name = organizer_email_obj.get("address", "") or organizer_email_obj.get("name", "")

        # Best-effort: set lobby bypass so bots and guests can join without waiting
        if join_url:
            ok = _apply_lobby_bypass(token, user_id, join_url)
            _LOGGER.info("graph.schedule.lobby_bypass %s", "set" if ok else "failed")

        normalized_start = _to_second_z(start_dt)
        normalized_end = _to_second_z(end_dt)
        start_date = normalized_start[:10] if normalized_start else ""
        all_emails = ",".join(email_list)
        persist_payload = {
            "meeting_id": event_id,
            "meeting_start_time": normalized_start,
            "meeting_end_time": normalized_end,
            "platform": "Teams",
            "title": meeting_subject or "Meeting",
            "account_name": "",
            "attendees": all_emails,
            "organiser_name": organiser_name or DEFAULT_ORGANISER,
            "action": "Schedule",
            "body": join_url,
            "recurrence_pattern": "",
            "recurrence_interval": 0,
            "recurrence_start_date": start_date,
            "recurrence_end_date": start_date,
        }
        _persist_scheduler_envelope(
            _build_scheduler_envelope(
                action_type="schedule_meeting",
                status="success",
                display_text="All set-the Teams invite has been sent.",
                payload=persist_payload,
            )
        )

        return (
            "Teams meeting created and invite sent.\n"
            f"  Subject    : {meeting_subject}\n"
            f"  Recipients : {all_emails}\n"
            f"  Time       : {meeting_start} -> {meeting_end} ({timezone})\n"
            f"  Organiser  : {organiser_name or '(not returned)'}\n"
            f"  Event ID   : {event_id or '(not returned)'}\n"
            f"  Join URL   : {join_url or '(not returned)'}"
        )
    except Exception as exc:
        _LOGGER.exception("graph.schedule.error error=%s", exc)
        return f"Failed to create Teams meeting invite: {exc}"


@tool(approval_mode="never_require")
def reschedule_teams_meeting_invite(
    event_id: Annotated[str, Field(description="Microsoft Graph event id to reschedule.")],
    meeting_start: Annotated[str, Field(description="New start in format YYYY-MM-DD HH:MM.")],
    meeting_end: Annotated[str, Field(description="New end in format YYYY-MM-DD HH:MM.")],
    attendees_emails: Annotated[str, Field(description="Attendee email(s), comma-separated if multiple")] = "",
    timezone: Annotated[str, Field(description="IANA timezone name. Example: 'Asia/Kolkata'.")] = "Asia/Kolkata",
    meeting_subject: Annotated[str, Field(description="Optional updated subject. Leave empty to keep current subject.")] = "",
    body_html: Annotated[str, Field(description="Optional updated HTML body. Leave empty to keep current body.")] = "",
) -> str:
    """Reschedule an existing Teams meeting event via Microsoft Graph."""
    try:
        tenant_id = _required_env("TENANT_ID")
        client_id = _required_env("CLIENT_ID")
        client_secret = _required_env("CLIENT_SECRET")
        user_id = _required_env("USER_ID")

        start_time = datetime.strptime(meeting_start, "%Y-%m-%d %H:%M")
        end_time = datetime.strptime(meeting_end, "%Y-%m-%d %H:%M")
        if end_time <= start_time:
            return "Invalid input: meeting_end must be after meeting_start."

        token = _get_access_token(tenant_id, client_id, client_secret)
        win_tz = _to_windows_tz(timezone)

        patch_body: dict[str, Any] = {
            "start": {"dateTime": start_time.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": win_tz},
            "end": {"dateTime": end_time.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": win_tz},
        }
        if meeting_subject.strip():
            patch_body["subject"] = meeting_subject.strip()
        if body_html.strip():
            patch_body["body"] = {
                "contentType": "HTML",
                "content": body_html.strip(),
            }
        if attendees_emails.strip():
            email_list = [e.strip() for e in attendees_emails.split(",") if e.strip()]
            patch_body["attendees"] = [
                {
                    "emailAddress": {"address": email, "name": email.split("@")[0]},
                    "type": "required",
                }
                for email in email_list
            ]

        event_url = f"{GRAPH_EVENTS_URL_TMPL.format(user_id=user_id)}/{event_id}"
        _LOGGER.info(
            "graph.reschedule.start url=%s event_id=%s meeting_start=%s meeting_end=%s",
            event_url,
            event_id,
            meeting_start,
            meeting_end,
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        response = httpx.patch(
            event_url,
            headers=headers,
            json=patch_body,
            timeout=45.0,
            verify=_ssl_verify(),
        )
        response.raise_for_status()
        _LOGGER.info("graph.reschedule.done url=%s status=%s event_id=%s", event_url, response.status_code, event_id)

        event = response.json() or {}
        online = event.get("onlineMeeting", {}) if isinstance(event.get("onlineMeeting", {}), dict) else {}
        join_url = online.get("joinUrl", "")

        normalized_start = _to_second_z(start_time.strftime("%Y-%m-%dT%H:%M:%S"))
        normalized_end = _to_second_z(end_time.strftime("%Y-%m-%dT%H:%M:%S"))
        start_date = normalized_start[:10] if normalized_start else ""
        persist_payload = {
            "meeting_id": event_id,
            "meeting_start_time": normalized_start,
            "meeting_end_time": normalized_end,
            "platform": "Teams",
            "title": meeting_subject.strip() or "Meeting",
            "account_name": "",
            "attendees": attendees_emails,
            "organiser_name": DEFAULT_ORGANISER,
            "action": "Reschedule",
            "body": join_url,
            "recurrence_pattern": "",
            "recurrence_interval": 0,
            "recurrence_start_date": start_date,
            "recurrence_end_date": start_date,
        }
        _persist_scheduler_envelope(
            _build_scheduler_envelope(
                action_type="reschedule_meeting",
                status="success",
                display_text="All set-the Teams meeting has been rescheduled.",
                payload=persist_payload,
            )
        )

        return (
            "Teams meeting rescheduled successfully.\n"
            f"  Event ID   : {event_id}\n"
            f"  New Time   : {meeting_start} -> {meeting_end} ({timezone})"
        )
    except Exception as exc:
        _LOGGER.exception("graph.reschedule.error event_id=%s error=%s", event_id, exc)
        return f"Failed to reschedule Teams meeting: {exc}"


@tool(approval_mode="never_require")
def cancel_teams_meeting_invite(
    event_id: Annotated[str, Field(description="Microsoft Graph event id to cancel/delete.")],
    attendees_emails: Annotated[str, Field(description="Attendee email(s), comma-separated if multiple")] = "",
    meeting_start_time: Annotated[str, Field(description="Meeting start time in ISO format if known")] = "",
    meeting_end_time: Annotated[str, Field(description="Meeting end time in ISO format if known")] = "",
    meeting_title: Annotated[str, Field(description="Meeting subject/title if known")] = "",
    organiser_name: Annotated[str, Field(description="Organizer email if known")] = "",
) -> str:
    """Cancel/delete an existing Teams meeting event via Microsoft Graph."""
    try:
        tenant_id = _required_env("TENANT_ID")
        client_id = _required_env("CLIENT_ID")
        client_secret = _required_env("CLIENT_SECRET")
        user_id = _required_env("USER_ID")

        token = _get_access_token(tenant_id, client_id, client_secret)

        event_url = f"{GRAPH_EVENTS_URL_TMPL.format(user_id=user_id)}/{event_id}"
        _LOGGER.info("graph.cancel.start url=%s event_id=%s", event_url, event_id)
        headers = {"Authorization": f"Bearer {token}"}
        response = httpx.delete(
            event_url,
            headers=headers,
            timeout=45.0,
            verify=_ssl_verify(),
        )
        response.raise_for_status()
        _LOGGER.info("graph.cancel.done url=%s status=%s event_id=%s", event_url, response.status_code, event_id)

        normalized_start = _to_second_z(meeting_start_time)
        normalized_end = _to_second_z(meeting_end_time)
        start_date = normalized_start[:10] if normalized_start else ""

        _persist_scheduler_envelope(
            _build_scheduler_envelope(
                action_type="cancel_meeting",
                status="success",
                display_text="Done-the Teams invite has been cancelled.",
                payload={
                    "meeting_id": event_id,
                    "meeting_start_time": normalized_start,
                    "meeting_end_time": normalized_end,
                    "platform": "Teams",
                    "title": meeting_title.strip() or "Meeting",
                    "account_name": "",
                    "attendees": attendees_emails,
                    "organiser_name": organiser_name.strip() or DEFAULT_ORGANISER,
                    "action": "Cancel",
                    "body": "",
                    "recurrence_pattern": "",
                    "recurrence_interval": 0,
                    "recurrence_start_date": start_date,
                    "recurrence_end_date": start_date,
                },
            )
        )

        return (
            "Teams meeting cancelled successfully.\n"
            f"  Event ID   : {event_id}"
        )
    except Exception as exc:
        _LOGGER.exception("graph.cancel.error event_id=%s error=%s", event_id, exc)
        return f"Failed to cancel Teams meeting: {exc}"


@tool(approval_mode="never_require")
def reschedule_teams_meeting_from_search(
    organiser_name: Annotated[str, Field(description="Organizer email, e.g. Lenovo_D365_PoC@sutherlandglobal.com")],
    meeting_start_time: Annotated[str, Field(description="Current meeting start in ISO format, e.g. 2026-06-23T21:30:00.000Z")],
    attendees_emails: Annotated[str, Field(description="Attendee email(s), comma-separated if multiple")],
    new_meeting_start: Annotated[str, Field(description="New start in format YYYY-MM-DD HH:MM")],
    new_meeting_end: Annotated[str, Field(description="New end in format YYYY-MM-DD HH:MM")],
    title: Annotated[str, Field(description="Optional title filter used in meeting search")] = "",
    timezone: Annotated[str, Field(description="IANA timezone name. Example: 'Asia/Kolkata'.")] = "Asia/Kolkata",
    meeting_subject: Annotated[str, Field(description="Optional updated subject. Leave empty to keep current subject.")] = "",
    body_html: Annotated[str, Field(description="Optional updated HTML body. Leave empty to keep current body.")] = "",
) -> str:
    """Reschedule a Teams event by first resolving event_id from the meeting search API."""
    event_id, row, err = _lookup_event_id(organiser_name, meeting_start_time, attendees_emails, title)
    if not event_id:
        return f"Unable to reschedule because event id could not be resolved. {err}".strip()

    resolved_subject = meeting_subject.strip() or _pick_str(row, "title", "subject")
    result = reschedule_teams_meeting_invite(
        event_id=event_id,
        meeting_start=new_meeting_start,
        meeting_end=new_meeting_end,
        attendees_emails=attendees_emails,
        timezone=timezone,
        meeting_subject=resolved_subject,
        body_html=body_html,
    )
    return result


@tool(approval_mode="never_require")
def cancel_teams_meeting_from_search(
    organiser_name: Annotated[str, Field(description="Organizer email, e.g. Lenovo_D365_PoC@sutherlandglobal.com")],
    meeting_start_time: Annotated[str, Field(description="Meeting start in ISO format, e.g. 2026-06-23T21:30:00.000Z")],
    attendees_emails: Annotated[str, Field(description="Attendee email(s), comma-separated if multiple")],
    title: Annotated[str, Field(description="Optional title filter used in meeting search")] = "",
) -> str:
    """Cancel a Teams event by first resolving event_id from the meeting search API."""
    event_id, row, err = _lookup_event_id(organiser_name, meeting_start_time, attendees_emails, title)
    if not event_id:
        return f"Unable to cancel because event id could not be resolved. {err}".strip()
    resolved_start = _pick_str(row, "meeting_start_time", "start_time", "start", "meetingStartTime") or meeting_start_time
    resolved_end = _pick_str(row, "meeting_end_time", "end_time", "end", "meetingEndTime")
    resolved_title = _pick_str(row, "title", "subject") or title
    resolved_organiser = _pick_str(row, "organiser_name", "organizer_name", "organiser_email", "organizer_email") or organiser_name
    return cancel_teams_meeting_invite(
        event_id=event_id,
        attendees_emails=attendees_emails,
        meeting_start_time=resolved_start,
        meeting_end_time=resolved_end,
        meeting_title=resolved_title,
        organiser_name=resolved_organiser,
    )


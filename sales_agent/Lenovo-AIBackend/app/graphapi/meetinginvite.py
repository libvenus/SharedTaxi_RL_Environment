from dotenv import load_dotenv
from pydantic import Field
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

import httpx

# Load environment variables from .env or .github/.env
_ROOT = Path(__file__).resolve().parents[2]  # project root (c:\work\bot)
load_dotenv(_ROOT / ".env", override=False)
load_dotenv(_ROOT / ".github" / ".env", override=False)


GRAPH_SCOPE = "https://graph.microsoft.com/.default"
GRAPH_TOKEN_URL_TMPL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
GRAPH_EVENTS_URL_TMPL = "https://graph.microsoft.com/v1.0/users/{user_id}/events"

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
    """If upn_or_guid looks like a GUID already, return it. Otherwise resolve via Graph."""
    import re
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', upn_or_guid, re.I):
        return upn_or_guid
    r = httpx.get(
        f"https://graph.microsoft.com/v1.0/users/{upn_or_guid}?$select=id",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0, verify=_ssl_verify(),
    )
    r.raise_for_status()
    return r.json().get("id", upn_or_guid)


def apply_lobby_bypass(token: str, user_id: str, join_url: str) -> bool:
    """Set lobby bypass to 'everyone' on a Teams online meeting identified by its join URL.
    Returns True if the PATCH succeeded, False otherwise.
    The onlineMeetings filter API requires a GUID user ID, not a UPN.
    """
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
    except Exception:
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

def create_teams_meeting_invite(
    recipient_email: Annotated[str, Field(description="Recipient email address that should receive the calendar invite.")],
    meeting_start: Annotated[str, Field(description="Meeting start in format YYYY-MM-DD HH:MM. Example: '2026-06-14 14:00'.")],
    meeting_end: Annotated[str, Field(description="Meeting end in format YYYY-MM-DD HH:MM. Example: '2026-06-14 15:00'.")],
    meeting_subject: Annotated[str, Field(description="Meeting subject/title.")] = "Teams Meeting: Calendar Invitation",
    timezone: Annotated[str, Field(description="IANA timezone name. Example: 'Asia/Kolkata'.")] = "Asia/Kolkata",
    recipient_name: Annotated[str, Field(description="Recipient display name shown in attendee list.")] = "Required Attendee",
    body_html: Annotated[str, Field(description="Optional HTML body. If empty, a default body is generated.")] = "",
    
) -> str:
    """Create a Teams-enabled calendar event via Microsoft Graph and send invite to one recipient."""
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

        event_body = {
            "subject": meeting_subject,
            "start": {"dateTime": start_dt, "timeZone": win_tz},
            "end": {"dateTime": end_dt, "timeZone": win_tz},
            "attendees": [
                {
                    "emailAddress": {
                        "address": recipient_email,
                        "name": recipient_name,
                    },
                    "type": "required",
                }
            ],
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
        event = response.json() or {}

        event_id = event.get("id", "")
        online = event.get("onlineMeeting", {}) if isinstance(event.get("onlineMeeting", {}), dict) else {}
        join_url = online.get("joinUrl", "")

        organizer = event.get("organizer", {}) if isinstance(event.get("organizer", {}), dict) else {}
        organizer_email_obj = organizer.get("emailAddress", {}) if isinstance(organizer.get("emailAddress", {}), dict) else {}
        organiser_name = organizer_email_obj.get("address", "") or organizer_email_obj.get("name", "")

        # Best-effort: set lobby bypass so bots and guests can join without waiting
        if join_url:
            apply_lobby_bypass(token, user_id, join_url)

        return {
            "success": True,
            "event_id": event_id or None,
            "join_url": join_url or None,
            "organiser": organiser_name or None,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "event_id": None, "join_url": None}


def reschedule_teams_meeting_invite(
    event_id: Annotated[str, Field(description="Microsoft Graph event id to reschedule.")],
    meeting_start: Annotated[str, Field(description="New start in format YYYY-MM-DD HH:MM.")],
    meeting_end: Annotated[str, Field(description="New end in format YYYY-MM-DD HH:MM.")],
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

        patch_body = {
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

        event_url = f"{GRAPH_EVENTS_URL_TMPL.format(user_id=user_id)}/{event_id}"
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

        return (
            "Teams meeting rescheduled successfully.\n"
            f"  Event ID   : {event_id}\n"
            f"  New Time   : {meeting_start} -> {meeting_end} ({timezone})"
        )
    except Exception as exc:
        return f"Failed to reschedule Teams meeting: {exc}"



def cancel_teams_meeting_invite(
    event_id: Annotated[str, Field(description="Microsoft Graph event id to cancel/delete.")],
) -> str:
    """Cancel/delete an existing Teams meeting event via Microsoft Graph."""
    try:
        tenant_id = _required_env("TENANT_ID")
        client_id = _required_env("CLIENT_ID")
        client_secret = _required_env("CLIENT_SECRET")
        user_id = _required_env("USER_ID")

        token = _get_access_token(tenant_id, client_id, client_secret)

        event_url = f"{GRAPH_EVENTS_URL_TMPL.format(user_id=user_id)}/{event_id}"
        headers = {"Authorization": f"Bearer {token}"}
        response = httpx.delete(
            event_url,
            headers=headers,
            timeout=45.0,
            verify=_ssl_verify(),
        )
        response.raise_for_status()

        return (
            "Teams meeting cancelled successfully.\n"
            f"  Event ID   : {event_id}"
        )
    except Exception as exc:
        return f"Failed to cancel Teams meeting: {exc}"



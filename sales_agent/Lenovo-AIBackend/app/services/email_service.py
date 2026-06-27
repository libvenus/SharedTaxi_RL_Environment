"""Send mail via Microsoft Graph (client-credentials flow)."""

from __future__ import annotations

import os

import httpx

from app.core.config import settings

GRAPH_SCOPE = "https://graph.microsoft.com/.default"
GRAPH_TOKEN_URL_TMPL = (
    "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
)
GRAPH_SEND_MAIL_URL_TMPL = (
    "https://graph.microsoft.com/v1.0/users/{user_email}/sendMail"
)


def _ssl_verify() -> bool:
    """Corporate proxies often MITM HTTPS; set GRAPH_VERIFY_SSL=false in .env."""
    raw = (os.getenv("GRAPH_VERIFY_SSL") or "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _sender_mailbox() -> str:
    mailbox = (
        (os.getenv("USER_EMAIL") or "").strip()
        or (settings.SENDER_EMAIL or "").strip()
    )
    if not mailbox:
        raise RuntimeError(
            "Missing USER_EMAIL (or SENDER_EMAIL) for Graph sendMail."
        )
    return mailbox


class EmailService:

    @staticmethod
    def get_access_token() -> str:
        tenant_id = (settings.TENANT_ID or "").strip()
        client_id = (settings.CLIENT_ID or "").strip()
        client_secret = (settings.CLIENT_SECRET or "").strip()
        if not all([tenant_id, client_id, client_secret]):
            raise RuntimeError(
                "TENANT_ID, CLIENT_ID, and CLIENT_SECRET must be set."
            )

        token_url = GRAPH_TOKEN_URL_TMPL.format(tenant_id=tenant_id)
        payload = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": GRAPH_SCOPE,
        }
        resp = httpx.post(
            token_url,
            data=payload,
            timeout=30.0,
            verify=_ssl_verify(),
        )
        resp.raise_for_status()
        token = (resp.json() or {}).get("access_token", "")
        if not token:
            raise RuntimeError("Graph token response did not include access_token")
        return token

    @staticmethod
    def send_email(
        to_emails: list,
        cc_emails: list,
        subject: str,
        body: str,
    ) -> bool:
        # When Microsoft Graph credentials are not configured, log and simulate success.
        tenant_id = (settings.TENANT_ID or "").strip()
        client_id = (settings.CLIENT_ID or "").strip()
        client_secret = (settings.CLIENT_SECRET or "").strip()
        if not all([tenant_id, client_id, client_secret]):
            import logging
            logging.getLogger(__name__).info(
                "EMAIL_STUB to=%s subject=%r — Graph creds not configured, skipping send",
                to_emails,
                subject,
            )
            return True

        token = EmailService.get_access_token()
        endpoint = GRAPH_SEND_MAIL_URL_TMPL.format(
            user_email=_sender_mailbox(),
        )

        payload = {
            "message": {
                "subject": subject,
                "body": {
                    "contentType": "Text",
                    "content": body,
                },
                "toRecipients": [
                    {"emailAddress": {"address": email}}
                    for email in to_emails
                ],
                "ccRecipients": [
                    {"emailAddress": {"address": email}}
                    for email in cc_emails
                ],
            },
            "saveToSentItems": True,
        }

        response = httpx.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30.0,
            verify=_ssl_verify(),
        )

        if response.status_code != 202:
            raise RuntimeError(
                f"Graph API Error: {response.status_code} {response.text}"
            )

        return True

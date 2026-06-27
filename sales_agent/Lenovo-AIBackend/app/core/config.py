from dotenv import load_dotenv
import os

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
DATABASE_URL = os.getenv("DATABASE_URL") or DATABASE_URL

# ---------------------------------------------------------------------------
# Sprint 1A · US01 — D365 Sales backend integration
#
# The Note-Taking Agent calls D365 to resolve a meeting (attendee emails +
# subject) to a CRM opportunity_id / account_id. Defaults to localhost so
# local dev works without an .env edit.
# ---------------------------------------------------------------------------
D365_BASE_URL = os.getenv("D365_BASE_URL", "http://localhost:9100")
D365_TIMEOUT_SECONDS = float(os.getenv("D365_TIMEOUT_SECONDS", "5.0"))


# ---------------------------------------------------------------------------
# Sprint 1A · US03 — Pre-meeting consent email pipeline
#
# How long BEFORE the meeting starts is the consent email scheduled? If a
# meeting is detected with a start time inside this window (or already
# started), we skip the email and fall back to US02's in-meeting chat
# announcement.
# ---------------------------------------------------------------------------
CONSENT_WINDOW_MINUTES = int(os.getenv("CONSENT_WINDOW_MINUTES", "60"))

# Internal Lenovo domains. Attendees with these email suffixes are NOT sent
# the consent email (AC #1: "external customer attendees only"). Comma-
# separated, trimmed, lower-cased on parse.
INTERNAL_EMAIL_DOMAINS = tuple(
    d.strip().lower()
    for d in os.getenv(
        "INTERNAL_EMAIL_DOMAINS",
        "lenovo.com,motorola.com",
    ).split(",")
    if d.strip()
)

# Public-facing base URL for the opt-out link embedded in the email. The
# email contains <OPT_OUT_BASE_URL><AIBACKEND_API_PREFIX>/consent-emails/opt-out/<token>. In dev
# this defaults to localhost; in staging/prod set to the AIBackend's
# public hostname (must be HTTPS in prod).
OPT_OUT_BASE_URL = os.getenv("OPT_OUT_BASE_URL", "http://localhost:9101").rstrip("/")

# URL prefix for every AIBackend HTTP router (Nginx proxies /ai-api/ → this
# service). D365 Sales keeps /api/ on its own host — no collision.
AIBACKEND_API_PREFIX = os.getenv("AIBACKEND_API_PREFIX", "/ai-api").rstrip("/")

# The "from" address used by the email service. The display name is
# constructed as "<Seller Name> via Lenovo Sales Assistant" by the bot
# (AC #4); the seller name comes from tbl_schedule_meetings.organiser_name.
SYSTEM_EMAIL_ADDRESS = os.getenv("SYSTEM_EMAIL_ADDRESS", "sales-assistant@lenovo.com")

# Outbound email (SMTP) configuration.
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "0")) if os.getenv("SMTP_PORT") else None
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes")
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() in ("1", "true", "yes")
SMTP_TIMEOUT_SECONDS = int(os.getenv("SMTP_TIMEOUT_SECONDS", "30"))

EMAIL_FROM_ADDRESS = os.getenv("EMAIL_FROM_ADDRESS") or SYSTEM_EMAIL_ADDRESS
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "Lenovo Sales Assistant")
EMAIL_API_KEY = os.getenv("EMAIL_API_KEY")


# Retry delay after a failed send attempt (AC #10: "retry once after
# 10 minutes"). Configurable so we can shrink it in dev / e2e tests.
CONSENT_RETRY_DELAY_MINUTES = int(os.getenv("CONSENT_RETRY_DELAY_MINUTES", "10"))


# ---------------------------------------------------------------------------
# Sprint 1A · US04 — Data Hygiene, Validation & Intelligent Alerts
#
# Drives the "stale activity" detector (D3) and the daily-scan CLI job.
# All three knobs are deliberately env-driven so admins can tune behaviour
# without a code change (AC #9 spirit, even though AC #9 itself — admin
# mandatory-fields-per-stage — is a S1B item).
# ---------------------------------------------------------------------------

# An opportunity with NO activity logged for more than this many days
# triggers a 'stale_activity' data-task. Default 30d per the user story.
DATA_TASK_STALE_DAYS = int(os.getenv("DATA_TASK_STALE_DAYS", "30"))

# How many opps the daily scan loads from D365 per page. Bigger = fewer
# round-trips; smaller = lower memory peak on a multi-thousand-deal tenant.
DATA_TASK_SCAN_PAGE_SIZE = int(os.getenv("DATA_TASK_SCAN_PAGE_SIZE", "100"))

# Severity assigned to a detected task when the source doesn't specify one.
# AI-team transcript signals MAY override this in the POST body; the
# deterministic scan detectors set their own severity per rule.
DATA_TASK_DEFAULT_SEVERITY = os.getenv("DATA_TASK_DEFAULT_SEVERITY", "medium")

# Retry delay after a failed send attempt (AC #10: "retry once after
# 10 minutes"). Configurable so we can shrink it in dev / e2e tests.
CONSENT_RETRY_DELAY_MINUTES = int(os.getenv("CONSENT_RETRY_DELAY_MINUTES", "10"))


# ---------------------------------------------------------------------------
# Sprint 1A · US04 — Data Hygiene, Validation & Intelligent Alerts
#
# Drives the "stale activity" detector (D3) and the daily-scan CLI job.
# All three knobs are deliberately env-driven so admins can tune behaviour
# without a code change (AC #9 spirit, even though AC #9 itself — admin
# mandatory-fields-per-stage — is a S1B item).
# ---------------------------------------------------------------------------

# An opportunity with NO activity logged for more than this many days
# triggers a 'stale_activity' data-task. Default 30d per the user story.
DATA_TASK_STALE_DAYS = int(os.getenv("DATA_TASK_STALE_DAYS", "30"))

# How many opps the daily scan loads from D365 per page. Bigger = fewer
# round-trips; smaller = lower memory peak on a multi-thousand-deal tenant.
DATA_TASK_SCAN_PAGE_SIZE = int(os.getenv("DATA_TASK_SCAN_PAGE_SIZE", "100"))

# Severity assigned to a detected task when the source doesn't specify one.
# AI-team transcript signals MAY override this in the POST body; the
# deterministic scan detectors set their own severity per rule.
DATA_TASK_DEFAULT_SEVERITY = os.getenv("DATA_TASK_DEFAULT_SEVERITY", "medium")


import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    TENANT_ID = os.getenv("TENANT_ID")
    CLIENT_ID = os.getenv("CLIENT_ID")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET")

    SENDER_EMAIL = os.getenv("SENDER_EMAIL")


settings = Settings()

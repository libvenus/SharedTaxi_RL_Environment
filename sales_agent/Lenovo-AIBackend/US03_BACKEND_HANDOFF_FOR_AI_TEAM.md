# US03 — Consent Capture · Backend Handoff for AI Team

**Sprint:** 1A
**User Story:** Consent Capture (pre-meeting consent email)
**Backend status:** All 6 endpoints implemented + tested + documented
**Backend repo:** `Lenovo-AIBackend`  (single-repo story — no D365 changes)
**Backend contact:** Sanmay
**Document audience:** AI / Bot integration team (Note-Taking Agent owners)

---

## TL;DR — what backend gives you for US03

| # | Endpoint | When the bot calls it |
|---|---|---|
| 1 | `POST /consent-emails/schedule` | After a meeting becomes eligible (US01 resolved an opportunity) |
| 2 | `PATCH /consent-emails/{consent_id}/delivery` | After every send attempt — success or failure |
| 3 | `GET  /consent-emails/opt-out/{token}` | **You don't call this — the participant's browser does**, after they click the link in your email |
| 4 | `GET  /consent-emails/due-for-retry` | Poll periodically (1–5 min cadence) to pick up failed sends ready for retry |
| 5 | `GET  /consent-emails/{meeting_id}` | (Optional) audit / FE list view |
| 6 | `GET  /meetings/{meeting_id}/consent-status` | **Right before joining** — if `any_opted_out`, do NOT join |

One new table: `tbl_meeting_consent_email` — one row per `(meeting_id, recipient_email)`.

---

## 1. The bot's flow with backend (US03 inserts between US01 and US02)

```
                   Outlook event detected
                          │
                          ▼  (US01)
          ┌─────────────────────────────────────────┐
          │ Resolve to D365 opportunity, create     │
          │ tbl_schedule_meetings row,              │
          │ PATCH bot_status='scheduled'            │
          └────────────────┬────────────────────────┘
                           │
              ────── US03 STARTS HERE ──────
                           │
                           ▼
          ┌─────────────────────────────────────────┐
          │ POST /consent-emails/schedule           │   (1)
          │ → response.should_send                  │
          └────────────────┬────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   should_send=true   should_send=false    fallback='meeting_started'
        │              fallback=             │
        │              'in_meeting_chat'      │ (meeting already started)
        ▼                                    │
   ┌──────────────┐         │                │
   │ Send email   │         │                │
   │ via Graph    │         │                │
   │ on seller's  │         │                │
   │ behalf       │         │                │
   └──────┬───────┘         │                │
          │                 │                │
          ▼                 │                │
   ┌──────────────────┐     │                │
   │ PATCH /delivery  │ (2) │                │
   └──────┬───────────┘     │                │
          │ if failed       │                │
          │ + retry_due     │                │
          ▼                 │                │
   ┌──────────────────┐     │                │
   │ Poll             │ (4) │                │
   │ /due-for-retry   │     │                │
   └──────────────────┘     │                │
                            │                │
   At meeting start time, regardless of which path:
                                 │
                                 ▼
          ┌─────────────────────────────────────────┐
          │ GET /meetings/{id}/consent-status       │  (6)
          └─────────┬───────────────────────────────┘
                    │
              ┌─────┴──────────────┐
              ▼                    ▼
   any_opted_out=true       any_opted_out=false
        │                          │
        ▼                          ▼
   STOP — bot does NOT      Proceed to US01 PATCH
   join. Show INF_MSG_0001   bot_status='joining'
   to seller. (The cancel    → US02 (transcript)
   was already done by
   backend on opt-out)
```

---

## 2. Acceptance criteria coverage

| AC | Backend coverage |
|---|---|
| 1. External customers only — no internal | `INTERNAL_EMAIL_DOMAINS` config filters on schedule. Bot just passes the full attendee list; we filter |
| 2. No response = consent approved | Implicit — we never auto-cancel. Backend only acts on actual opt-out clicks |
| 3. Window passed → skip email, use in-meeting | Schedule returns `should_send=false` + `fallback='in_meeting_chat'` if `(start_time - now) < CONSENT_WINDOW_MINUTES` |
| 4. From `<seller> via Lenovo Sales Assistant` | Schedule response includes `seller_name` (from `tbl_schedule_meetings.organiser_name`) and `system_email_address` (from config). Bot constructs the From header |
| 5. Email contains title / time / seller / opt-out link | Schedule response includes everything you need: meeting (via `meeting_id` lookup), seller, and per-recipient `opt_out_url` |
| 6. Opt-out link active until meeting starts | Backend returns `410 Gone` HTML if clicked after `meeting_start_time` |
| 7. Any opt-out → bot doesn't join, no override | Opt-out cascades to `tbl_schedule_meetings.bot_status='cancelled'` with reason `'participant_opted_out'`. Bot reads `/consent-status.any_opted_out` before joining and stops |
| 8. Opt-out participant sees confirmation (SUCC_MSG_0010) | Opt-out endpoint returns server-rendered HTML page |
| 9. Seller notified immediately (INF_MSG_0001) | Backend stamps `seller_notified_at`. **Bot reads this** when polling status / showing the seller UI in a future sprint. AC says "Notifications will be displayed in future sprints" — backend has the data |
| 10. Failed delivery → retry once after 10 min, then fallback | Backend tracks `attempt_count` (0/1/2), `next_retry_at`, and flips to `fallback_to_in_meeting` after 2 failures |

---

## 3. Endpoint reference

### 3.1 `POST /consent-emails/schedule`

**When:** Bot detects a meeting eligible for the agent (US01 resolved opp).

```http
POST /consent-emails/schedule
Content-Type: application/json

{
  "meeting_id": "11111111-1111-1111-1111-111111111111",
  "recipients": [
    { "email": "k.richter@db.com",       "name": "Klaus Richter" },
    { "email": "rajesh.k@infosys.com",   "name": "Rajesh Kumar" },
    { "email": "seller@lenovo.com",      "name": "Maria Hofer" }
  ]
}
```

**Success — window open (200):**

```jsonc
{
  "meeting_id": "11111111-...",
  "should_send": true,
  "fallback": null,
  "recipients": [
    {
      "consent_id":         "a1b2c3d4-...",
      "recipient_email":    "k.richter@db.com",
      "recipient_name":     "Klaus Richter",
      "opt_out_token":      "9pVkM4xN_yQzj…",  // 43-char URL-safe token
      "opt_out_url":        "http://localhost:8001/consent-emails/opt-out/9pVkM4xN_yQzj…",
      "scheduled_send_at":  "2026-06-15T14:00:00Z",
      "delivery_status":    "pending"
    }
    // …rajesh entry…
  ],
  "filtered_internal_count": 1,    // seller@lenovo.com was internal
  "seller_name":             "Maria Hofer",
  "system_email_address":    "sales-assistant@lenovo.com"
}
```

**Success — window passed (200):**

```jsonc
{
  "meeting_id":              "11111111-...",
  "should_send":             false,
  "fallback":                "in_meeting_chat",  // OR "meeting_started"
  "recipients":              [],
  "filtered_internal_count": 0,
  "seller_name":             "Maria Hofer",
  "system_email_address":    "sales-assistant@lenovo.com"
}
```

**Errors:**
- `404` — meeting not found
- `400` — meeting has no `meeting_start_time`
- `422` — empty `recipients`, malformed email

**Idempotent:** re-calling with the same meeting returns the SAME tokens
(don't issue new tokens — the participant's inbox still has the old link).
New recipients added later get fresh rows on the re-call.

---

### 3.2 `PATCH /consent-emails/{consent_id}/delivery`

**When:** After every send attempt — success OR failure.

```http
PATCH /consent-emails/{consent_id}/delivery
Content-Type: application/json

{
  "status":          "sent",       // | "failed"
  "failure_reason":  null,         // optional, max 512 chars; required for "failed"
  "attempted_at":    "2026-06-15T14:00:12+00:00"   // optional
}
```

**Success (200):**

```jsonc
{
  "consent_id":      "a1b2c3d4-...",
  "meeting_id":      "11111111-...",
  "recipient_email": "k.richter@db.com",
  "delivery_status": "sent",                 // OR "failed", "fallback_to_in_meeting"
  "attempt_count":   1,                      // 0..2
  "last_attempt_at": "2026-06-15T14:00:12Z",
  "next_retry_at":   null,                    // populated only on first failure
  "failure_reason":  null
}
```

**Retry semantics:**
- 1st failure → `delivery_status='failed'`, `next_retry_at = now + 10min`
- 2nd failure → `delivery_status='fallback_to_in_meeting'`, `next_retry_at=null`
- Bot picks up retries by polling `/due-for-retry`

**Errors:**
- `400` — already opted-out, OR already in fallback (no more updates accepted)
- `404` — consent record not found
- `422` — invalid status, missing required fields

---

### 3.3 `GET /consent-emails/opt-out/{token}` — public, browser-facing

**You don't call this directly.** It's the URL embedded in the email body.
The participant's browser hits it when they click the link.

- `200 OK` text/html — opt-out recorded, SUCC_MSG_0010 page rendered
- `404` text/html — invalid token
- `410 Gone` text/html — meeting already started; link expired

The backend automatically:
1. Stamps `opted_out_at` + `opt_out_ip`
2. Stamps `seller_notified_at` (for the future seller-notification UI)
3. Flips `tbl_schedule_meetings.bot_status='cancelled'` with reason `'participant_opted_out'`

**Idempotent:** clicking twice renders the same page; doesn't re-stamp.

---

### 3.4 `GET /consent-emails/due-for-retry`

**When:** Periodic poll (recommend every 1–5 minutes during business hours).

```http
GET /consent-emails/due-for-retry
```

**Response (200):**

```jsonc
{
  "items": [
    {
      "consent_id":      "...",
      "meeting_id":      "...",
      "recipient_email": "rajesh.k@infosys.com",
      "recipient_name":  "Rajesh Kumar",
      "opt_out_url":     "http://...",
      "attempt_count":   1,
      "last_attempt_at": "2026-06-15T13:50:00Z",
      "next_retry_at":   "2026-06-15T14:00:00Z",
      "failure_reason":  "SMTP bounce: 550 mailbox unavailable"
    }
  ]
}
```

For each row, re-send the email and PATCH `/delivery` again.

---

### 3.5 `GET /consent-emails/{meeting_id}` — audit / FE

Returns every consent row for a meeting (no filtering). Useful for the
FE Activity tab and for compliance audits.

---

### 3.6 `GET /meetings/{meeting_id}/consent-status` — pre-join check

**When:** Right before the bot transitions to `bot_status='joining'`.

```http
GET /meetings/{meeting_id}/consent-status
```

**Response (200):**

```jsonc
{
  "meeting_id":         "11111111-...",
  "any_opted_out":      false,
  "opt_out_count":      0,
  "total_recipients":   2,
  "consent_mechanism":  "pre_meeting_email",   // | "in_meeting_chat" | "mixed"
  "pending_retries":    0,
  "sent_count":         2,
  "failed_count":       0,
  "fallback_count":     0
}
```

**Bot logic (this is AC #7):**

```python
status = await get_consent_status(meeting_id)

if status.any_opted_out:
    log.info(f"Aborting bot join — {status.opt_out_count}/{status.total_recipients} opted out")
    # The cancel cascade already happened on the opt-out itself; we don't need to do anything else.
    return

# Otherwise proceed: PATCH /meeting-details/{id}/status with bot_status='joining'
# When bot joins:
#   if status.consent_mechanism in ('pre_meeting_email', 'mixed'):
#       still send the in-meeting chat message (US02) — belt and braces
#   if status.consent_mechanism == 'in_meeting_chat':
#       in-meeting chat is the SOLE consent mechanism
```

---

## 4. Schema reference

### `tbl_meeting_consent_email` — one row per `(meeting_id, recipient_email)`

| Column | Type | Notes |
|---|---|---|
| `consent_id` | UUID PK | |
| `meeting_id` | UUID NOT NULL | FK-by-convention to `tbl_schedule_meetings` |
| `recipient_email` | TEXT NOT NULL | Lower-cased on write |
| `recipient_name` | TEXT | From the calendar invite |
| `opt_out_token` | TEXT NOT NULL UNIQUE | `secrets.token_urlsafe(32)` — 256 bits, ~43 chars |
| `scheduled_send_at` | TIMESTAMPTZ NOT NULL | `meeting_start - CONSENT_WINDOW_MINUTES` |
| `attempt_count` | INT NOT NULL CHECK 0..2 | |
| `last_attempt_at` | TIMESTAMPTZ | When bot last tried |
| `next_retry_at` | TIMESTAMPTZ | NULL when no retry pending |
| `delivery_status` | TEXT NOT NULL CHECK | `pending` / `sent` / `failed` / `fallback_to_in_meeting` |
| `failure_reason` | TEXT | Free-form |
| `opted_out_at` | TIMESTAMPTZ | NULL until clicked |
| `opt_out_ip` | TEXT | Browser IP (best-effort, optional) |
| `seller_notified_at` | TIMESTAMPTZ | When INF_MSG_0001 event was raised |
| `created_at` / `updated_at` | TIMESTAMPTZ | Server-managed |

UNIQUE on `(meeting_id, recipient_email)` — schedule is idempotent.
Partial indexes on retry queue + opt-out lookup keep the hot paths fast.

---

## 5. Email template variables — what backend gives you

When `should_send=true`, backend hands you everything needed to render
the email body. **Backend does NOT render HTML** — that's your domain.

| Variable | Source |
|---|---|
| Meeting title | Look up via `meeting_id` → `tbl_schedule_meetings.title` (or pass it through to schedule) |
| Meeting scheduled time | Look up via `meeting_id` → `tbl_schedule_meetings.meeting_start_time` |
| Seller name | `response.seller_name` |
| System "from" address | `response.system_email_address` |
| Opt-out URL (per recipient) | `recipients[].opt_out_url` — already absolute |

---

## 6. Out-of-scope (deferred)

| Item | Why deferred |
|---|---|
| Email rendering (HTML / plain text bodies) | AI team owns; backend exposes data only |
| Microsoft Graph send-as-seller integration | AI team owns; backend records the result |
| Seller-facing notification UI for INF_MSG_0001 | "Notifications will be displayed in future sprints" — AC says so |
| Localised email content (i18n) | Add when tenant rollout requires it |
| `meeting rescheduled` cascade — re-issue tokens? | Q5 below; current behaviour: re-running schedule keeps existing tokens |
| Click tracking (open / read receipts) | Compliance asks for delivered, not opened — out of scope unless requested |
| Rate limiting / DDoS protection on opt-out endpoint | Cloudflare / WAF concern; not v1 backend |
| Bot ↔ AIBackend authn | Internal-only for v1; bearer-token in 1B |

---

## 7. Open questions for AI team / PM

| # | Question | Owner |
|---|---|---|
| Q1 | Where do `SUCC_MSG_0010` and `INF_MSG_0001` canonical texts live? Backend renders SUCC_MSG_0010 server-side; bot displays INF_MSG_0001. Need single source of truth | PM |
| Q2 | Per-region from address, or one global `sales-assistant@lenovo.com`? Affects `SYSTEM_EMAIL_ADDRESS` config | PM |
| Q3 | Do we need to store opt-out IP / user-agent for compliance? Default yes — minimal GDPR consideration since the participant is acting on themselves | Legal / PM |
| Q4 | What's the bot's preferred polling interval for `/due-for-retry`? 1 min? 5 min? | AI team |
| Q5 | When a meeting is rescheduled (US01), invalidate consent records and re-send? Or keep old tokens? | Joint |
| Q6 | `INTERNAL_EMAIL_DOMAINS` global vs tenant-specific? Could vary per Lenovo subsidiary | DevOps + PM |
| Q7 | Soft-cascade on meeting DELETE (US01) or preserve consent rows? | Joint |

---

## 8. How to test on dev

```bash
# 1. Apply all 3 Sprint 1A migrations
psql … -f sql/2026_06_us01_meeting_lifecycle.sql
psql … -f sql/2026_06_us02_meeting_transcript.sql
psql … -f sql/2026_06_us03_consent_email.sql

# 2. Run AIBackend
uvicorn app.main:app --reload --port 8001

# 3. cURL through the flow (see §3 of this doc)

# 4. Or run the smoke tests
pytest -q tests/test_consent_capture_lifecycle.py
```

---

## 9. Contacts

- **Backend:** Sanmay
- **Frontend integration:** Namisha — when the FE Activity tab is ready, point at `GET /consent-emails/{meeting_id}`
- **AI/Bot team:** [your team contact]

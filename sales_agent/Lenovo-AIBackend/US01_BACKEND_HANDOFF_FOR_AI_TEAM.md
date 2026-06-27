# US01 — Joining the Meetings · Backend Handoff for AI Team

**Sprint:** 1A
**Story:** "Note-Taking Agent — Joining the Meetings"
**Backend version:** AIBackend 0.1 · D365 Sales 0.14.0
**Audience:** AI / Note-Taking Agent team (bot, Graph subscription, Teams Bot SDK owners)
**Last updated:** 2026-06-15

---

## 1. What this doc is for

You (the AI / Agent team) own the bot — Graph webhook subscriptions, Teams Bot SDK,
audio capture, transcript generation. We (Backend) own the persistence + the
CRM resolver. This doc lists every endpoint we built so you can wire your bot
to them.

**TL;DR — 4 endpoints.** Three live in `Lenovo-AIBackend`, one in `Lenovo D365 Sales`.

| Method | Path | Repo | When the bot calls it |
|---|---|---|---|
| `POST` | `/meeting-details/` | AIBackend | Whenever a new Outlook event is detected (or after resolver succeeds) |
| `PATCH` | `/meeting-details/{id}/status` | AIBackend | Every time the bot's lifecycle state changes |
| `DELETE` | `/meeting-details/{id}` | AIBackend | When the calendar event is deleted (soft-delete) |
| `POST` | `/api/meetings/resolve-opportunity` | D365 Sales | **Before** scheduling the bot, to find the matching CRM deal |

---

## 2. The bot's typical flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BOT FLOW (per meeting)                          │
└─────────────────────────────────────────────────────────────────────────┘

  Graph webhook fires → bot detects new Outlook event
            │
            ▼
  ┌─────────────────────────────────────────┐
  │ 1. POST /api/meetings/resolve-opportunity│  → D365 Sales backend
  │    body: attendee emails + subject       │     (returns opportunityId/accountId
  │                                          │      or 404 if no match)
  └─────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────┐
  │ 2. POST /meeting-details/                │  → AIBackend
  │    body: meeting metadata                │     bot_status defaults to 'pending'
  │           + opportunity_id / account_id  │
  │             (from step 1, if matched)    │
  └─────────────────────────────────────────┘
            │
            ▼ (bot schedules itself, time arrives)
  ┌─────────────────────────────────────────┐
  │ 3. PATCH /meeting-details/{id}/status    │  → AIBackend
  │    body: { "bot_status": "joining" }     │
  └─────────────────────────────────────────┘
            │
            ▼ (bot is in the meeting)
  ┌─────────────────────────────────────────┐
  │ 4. PATCH /meeting-details/{id}/status    │  → AIBackend
  │    body: { "bot_status": "joined" }      │
  └─────────────────────────────────────────┘
            │
            ▼ (meeting ends, transcript arrives — out of scope of US01)

  Cancellation path (any time before "joined"):
  ┌─────────────────────────────────────────┐
  │ DELETE /meeting-details/{id}             │  → AIBackend
  │ ?reason=Calendar+event+deleted           │     (soft-delete: row preserved)
  └─────────────────────────────────────────┘
```

---

## 3. Bot lifecycle state machine

Every value below is enforced in three places:
- DB: `CHECK` constraint on `tbl_schedule_meetings.bot_status`
- Code: `Literal[...]` in `app/schema/schedulemeeting.py`
- Tests: 6 parametrised PATCH tests in `tests/test_meeting_lifecycle.py`

```
                ┌────────────┐
                │  pending   │  ← default after POST /meeting-details/
                └─────┬──────┘
                      │  bot accepts the meeting
                      ▼
                ┌────────────┐
                │ scheduled  │
                └─────┬──────┘
                      │  meeting time arrives
                      ▼
                ┌────────────┐
                │  joining   │
                └─────┬──────┘
                      │
           ┌──────────┼──────────────┐
           ▼          ▼              ▼
    ┌──────────┐  ┌─────────┐  ┌─────────────────┐
    │  joined  │  │ failed  │  │ lobby_waiting   │
    └──────────┘  └─────────┘  └────────┬────────┘
                                         │ admit
                                         ▼
                                    ┌──────────┐
                                    │  joined  │
                                    └──────────┘

Any state may also transition to:
  • cancelled    ← calendar event deleted by organiser
  • rescheduled  ← meeting time moved (bot re-schedules itself)
```

| Status | Meaning |
|---|---|
| `pending` | Bot has been told about the meeting, not yet scheduled |
| `scheduled` | Bot has scheduled itself to join at start_time |
| `joining` | Bot is actively joining (transient — may or may not be observed) |
| `joined` | Bot is in the meeting and recording |
| `lobby_waiting` | Customer-organised meeting; bot is waiting to be admitted |
| `cancelled` | Meeting cancelled before / during; bot will not / did not join |
| `rescheduled` | Meeting was moved; bot has re-scheduled itself |
| `failed` | Bot tried to join but errored (network, auth, removed by host, etc.) |

---

## 4. Endpoint reference

> **Casing convention:** `Lenovo-AIBackend` uses **snake_case** keys
> (matching its existing `/meeting-details/` and `/activity-details/` endpoints).
> `Lenovo D365 Sales` uses **camelCase** keys (matches the D365 frontend).
> Don't get confused jumping between them.

### 4.1 `POST /meeting-details/` *(AIBackend)*

Insert-or-update a meeting record. Idempotent on `meeting_id`.

The bot calls this twice in the typical flow:
1. Initially — when the Outlook event is detected (minimal payload).
2. After the resolver succeeds — same payload **plus** `opportunity_id` / `account_id`.

`bot_status` is **preserved** across re-POSTs. So re-POSTing a meeting that's already `joined` does NOT silently revert it to `pending`.

**Request:**
```bash
curl -X POST http://AIBACKEND_HOST:8001/meeting-details/ \
  -H "Content-Type: application/json" \
  -d '{
    "meeting_id":         "11111111-1111-1111-1111-111111111111",
    "meeting_start_time": "2026-06-15T15:00:00+00:00",
    "meeting_end_time":   "2026-06-15T16:00:00+00:00",
    "platform":           "Microsoft Teams",
    "title":              "ThinkPad Fleet Review",
    "attendees":          "k.richter@db.com, seller@lenovo.com",
    "organiser_name":     "Maria Hofer",
    "opportunity_id":     "b0000001-0001-0001-0001-000000000001",
    "account_id":         "6dc95c38-9237-4ce9-84d3-f5d1f7431965"
  }'
```

**Response (200 OK):**
```json
{ "message": "Meeting saved successfully" }
```

**Field reference:**

| Field | Required | Notes |
|---|:---:|---|
| `meeting_id` | ✓ | UUID. Idempotency key — same ID = upsert. |
| `meeting_start_time` | ✓ | ISO-8601 with timezone |
| `meeting_end_time` | ✓ | ISO-8601 with timezone |
| `platform` |  | Free-form, e.g. "Microsoft Teams" |
| `title` |  | Meeting subject |
| `attendees` |  | Free-form string of emails — store whatever you have |
| `organiser_name` |  | Display name (not email) |
| `opportunity_id` |  | UUID — set after resolver returns. NULL means "no CRM tag" |
| `account_id` |  | UUID — set after resolver returns |
| `recurrence_*` |  | Existing fields — pre-US01, unchanged |

---

### 4.2 `PATCH /meeting-details/{meeting_id}/status` *(AIBackend)*

Transition the bot lifecycle state.

**Request:**
```bash
# Bot has just joined the meeting
curl -X PATCH \
  http://AIBACKEND_HOST:8001/meeting-details/11111111-1111-1111-1111-111111111111/status \
  -H "Content-Type: application/json" \
  -d '{ "bot_status": "joined" }'

# Lobby admit timed out
curl -X PATCH \
  http://AIBACKEND_HOST:8001/meeting-details/11111111-1111-1111-1111-111111111111/status \
  -H "Content-Type: application/json" \
  -d '{ "bot_status": "failed", "reason": "Lobby admit timeout (90s)" }'
```

**Response (200 OK):**
```json
{
  "meeting_id":        "11111111-1111-1111-1111-111111111111",
  "bot_status":        "failed",
  "bot_status_reason": "Lobby admit timeout (90s)",
  "bot_last_event_at": "2026-06-15T15:01:32.114Z",
  "updated_at":        "2026-06-15T15:01:32.114Z"
}
```

**Errors:**

| Code | Cause |
|---|---|
| `404` | Unknown `meeting_id` — bot has stale state |
| `422` | `bot_status` value not in the whitelist (see §3) |

`bot_last_event_at` is auto-stamped on every transition — you don't need to send a timestamp. `reason` is overwritten on each call (we keep only the most recent).

---

### 4.3 `DELETE /meeting-details/{meeting_id}` *(AIBackend)*

Soft-delete: flips `bot_status` to `cancelled` with the supplied reason. Row is **preserved** so late-arriving transcripts still find their meeting.

**Request:**
```bash
curl -X DELETE \
  "http://AIBACKEND_HOST:8001/meeting-details/11111111-1111-1111-1111-111111111111?reason=Calendar+event+deleted+by+organiser"
```

**Response (200 OK):**
```json
{
  "meeting_id":        "11111111-1111-1111-1111-111111111111",
  "bot_status":        "cancelled",
  "bot_status_reason": "Calendar event deleted by organiser",
  "bot_last_event_at": "2026-06-15T14:45:00.221Z",
  "updated_at":        "2026-06-15T14:45:00.221Z"
}
```

`reason` is optional. If omitted, the row gets `bot_status_reason="Meeting cancelled."`.

---

### 4.4 `POST /api/meetings/resolve-opportunity` *(D365 Sales)*

> **Different repo, different host, different casing.** This lives in
> `Lenovo D365 Sales` (port 8000) and uses **camelCase** keys.

Match a Teams meeting to its CRM opportunity. Returns the deal IDs the bot
should write back via the AIBackend's `opportunity_id` / `account_id` fields.

**Request:**
```bash
curl -X POST http://D365_HOST:8000/api/meetings/resolve-opportunity \
  -H "Content-Type: application/json" \
  -d '{
    "attendeeEmails": ["k.richter@db.com", "rajesh.k@infosys.com"],
    "subject":        "ThinkPad Fleet Review",
    "organiserEmail": "seller@lenovo.com"
  }'
```

**Response (200 OK):**
```json
{
  "opportunityId":       "B0000001-0001-0001-0001-000000000001",
  "accountId":           "6dc95c38-9237-4ce9-84d3-f5d1f7431965",
  "opportunityName":     "JPMorgan – Trader Workstation Refresh",
  "accountName":         "Deutsche Bank AG",
  "matchScore":          0.667,
  "matchedBy":           "contact_email",
  "matchedContactCount": 2
}
```

**Match score scale:** float `[0.0, 1.0]`.
**Recommended threshold for auto-tagging:** `>= 0.5`.

**Errors:**

| Code | Code field | Meaning | What you should do |
|---|---|---|---|
| `404` | `NO_CONTACT_MATCH` | None of the supplied emails match an active contact | Bot joins the meeting **untagged** (don't crash) |
| `404` | `NO_DEAL_MATCH` | Contacts matched but none are on an active deal | Same — join untagged |
| `422` | — | `attendeeEmails` is empty / malformed | Don't send the call in the first place |
| `5xx` | — | Server error | Retry once after 1 s; if still failing, join untagged + log to ops |

`matchedBy` values: `"contact_email"` | `"subject_keyword"` | `"both"`.

---

## 5. HTTP client wrapper (already built)

We already wrote a thin httpx wrapper around the D365 resolver so the bot
can call it as a Python function:

```python
from app.clients.d365_client import resolve_opportunity, D365ClientError

try:
    result = resolve_opportunity(
        attendee_emails=["k.richter@db.com"],
        subject="ThinkPad Fleet Review",
        organiser_email="seller@lenovo.com",
    )
except D365ClientError:
    # 5xx, 422, timeout — log, then join untagged
    result = None

if result is None:
    # 404 from D365 = no matching active deal. Normal — not an error.
    opportunity_id = account_id = None
else:
    opportunity_id = result.opportunity_id
    account_id     = result.account_id
```

It returns:
- `ResolveResult` on 200
- `None` on 404 (treat as normal traffic — bot still joins, just untagged)
- raises `D365ClientError` on 5xx / 422 / timeout / unexpected payload shape

Configure with two env vars:
```
D365_BASE_URL=http://D365_HOST:8000
D365_TIMEOUT_SECONDS=5.0
```

---

## 6. Database changes (`tbl_schedule_meetings`)

5 new columns added by `sql/2026_06_us01_meeting_lifecycle.sql`:

| Column | Type | Default | Purpose |
|---|---|---|---|
| `bot_status` | `text NOT NULL` | `'pending'` | Lifecycle state (CHECK-constrained to whitelist) |
| `bot_status_reason` | `text NULL` | — | Free-form context for the most recent transition |
| `bot_last_event_at` | `timestamptz NULL` | — | Auto-stamped on every PATCH |
| `opportunity_id` | `uuid NULL` | — | Resolved D365 opportunity (NULL means "untagged") |
| `account_id` | `uuid NULL` | — | Resolved D365 account |

Plus three indexes:
- `idx_schedule_meetings_bot_status` — bot's most common query is "give me everything not joined yet"
- `idx_schedule_meetings_opportunity_id` (partial, where IS NOT NULL)
- `idx_schedule_meetings_account_id` (partial, where IS NOT NULL)

Migration is idempotent — re-running it is safe.

---

## 7. End-to-end smoke test (Postman / cURL)

Run these in sequence once both backends are up:

```bash
# 1. Resolve the meeting → get opportunity / account IDs
RESOLVE=$(curl -s -X POST http://D365_HOST:8000/api/meetings/resolve-opportunity \
  -H "Content-Type: application/json" \
  -d '{ "attendeeEmails":["k.richter@db.com"], "subject":"ThinkPad Fleet Review" }')
echo $RESOLVE
# → {"opportunityId":"b0000001-...","accountId":"6dc95c38-...","matchScore":0.5,...}

# 2. Persist the meeting with the resolved IDs
curl -X POST http://AIBACKEND_HOST:8001/meeting-details/ \
  -H "Content-Type: application/json" \
  -d '{
    "meeting_id":         "22222222-2222-2222-2222-222222222222",
    "meeting_start_time": "2026-06-15T15:00:00+00:00",
    "meeting_end_time":   "2026-06-15T16:00:00+00:00",
    "title":              "ThinkPad Fleet Review",
    "attendees":          "k.richter@db.com",
    "platform":           "Microsoft Teams",
    "opportunity_id":     "b0000001-0001-0001-0001-000000000001",
    "account_id":         "6dc95c38-9237-4ce9-84d3-f5d1f7431965"
  }'
# → {"message": "Meeting saved successfully"}

# 3. Mark as joined when bot enters the meeting
curl -X PATCH http://AIBACKEND_HOST:8001/meeting-details/22222222-2222-2222-2222-222222222222/status \
  -H "Content-Type: application/json" \
  -d '{ "bot_status": "joined" }'
# → { "meeting_id": "...", "bot_status": "joined", "bot_last_event_at": "..." }

# 4. (alternative path) Cancel if the calendar event is deleted
curl -X DELETE \
  "http://AIBACKEND_HOST:8001/meeting-details/22222222-2222-2222-2222-222222222222?reason=Calendar+event+deleted"
# → { "meeting_id": "...", "bot_status": "cancelled", ... }
```

---

## 8. What's covered by automated tests

11 smoke tests in `tests/test_meeting_lifecycle.py` (all passing):

1. POST creates row with `bot_status='pending'` (DB default fires)
2. Re-POST preserves `bot_status` (idempotency)
3-8. PATCH transitions through all 6 non-trivial states (`scheduled` / `joining` / `joined` / `lobby_waiting` / `rescheduled` / `failed`)
9. PATCH with garbage status → 422
10. PATCH on unknown meeting → 404
11. DELETE soft-deletes (status flips, row preserved)

Run locally with `pytest -q` from the AIBackend repo root.

---

## 9. What's deliberately NOT in this story (defer to US02 / later)

| Concern | Why deferred |
|---|---|
| Storing the AI summary / transcript / key points | That's US02 (the meeting-intelligence ingest). Design notes already exist at `Lenovo D365 Sales/MEETING_INTELLIGENCE_STORAGE_BACKEND_NOTES.md` |
| Bot identity / authn between bot and AIBackend | AIBackend is internal-only for now. Add bearer-token before public deploy. |
| Manual-link-sharing endpoint (paste a meeting URL) | Story 1A explicitly says "no UI integration" |
| Outlook Graph webhook receiver | **Bot team's responsibility** — not us |
| Audit log of every status transition | Current row stores only the latest; revisit if compliance demands. |
| Retry queue for D365 5xx | Bot just logs and continues |

---

## 10. Open questions for the agent team

These don't block US01 from shipping, but would be great to align on before US02:

1. **Match-score threshold:** is `>= 0.5` an acceptable cutoff? Below it, bot joins untagged.
2. **Reschedule key:** does Outlook keep the same event ID when a meeting is moved, or do you get a new one? If new, the bot should `DELETE` the old `meeting_id` and `POST` the new one with `bot_status="rescheduled"`.
3. **Eager vs lazy resolution:** the plan assumes the bot calls the resolver **before** scheduling. If you'd rather defer to when the transcript arrives, we'll need a separate "patch the IDs in" endpoint.
4. **5xx retry policy:** how many retries before bot joins untagged? Currently the wrapper raises immediately — the bot's caller decides.
5. **Test data for E2E rehearsal:** want a curated set of "this email + this subject = this deal" pairs so the agent team can verify the resolver against known-good data?

---

## 11. Contacts

| Topic | Person | Repo |
|---|---|---|
| AIBackend (meeting persistence + lifecycle) | Sanmay (with Namisha) | `Lenovo-AIBackend` |
| D365 resolver | Sanmay | `Lenovo D365 Sales` |
| Bot / Graph / Teams Bot SDK | **AI / Agent team** | (your repo) |
| Frontend (Activity tab consumption) | Frontend team | `Lenevo-Frontend` |

---

## 12. Quick reference card

```
┌──────────────────────────────────────────────────────────────────────┐
│  POST   /api/meetings/resolve-opportunity   ← D365 (camelCase)      │
│  POST   /meeting-details/                   ← AIBackend (snake_case)│
│  PATCH  /meeting-details/{id}/status        ← AIBackend (snake_case)│
│  DELETE /meeting-details/{id}?reason=...    ← AIBackend (snake_case)│
└──────────────────────────────────────────────────────────────────────┘

  Statuses:  pending → scheduled → joining → joined
                                 ↘  lobby_waiting → joined
             any → cancelled / rescheduled / failed

  Resolver matchedBy: contact_email | subject_keyword | both
  Resolver scoreThreshold (recommended): 0.5
```

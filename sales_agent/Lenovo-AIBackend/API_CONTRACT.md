# Lenovo AI Backend — API Contract

**Base URL:** `http://localhost:8001` (dev) | TBD (staging/prod)
**API prefix:** `/ai-api` — every endpoint below is `{BASE_URL}/ai-api/...` (D365 Sales uses `/api/...` on its own service; no collision)
**Version:** 0.4.0 (Sprint 1A — US01 + US02 + US03 + US04)
**All responses:** JSON, **snake_case** keys (different from Lenovo D365 Sales which uses camelCase — they're two separate services)
**Auth:** None in Sprint 1A. To be added (bearer token) before public deployment.

This is the contract every team integrates against. If something here disagrees with the running code, the running code is the bug — file a PR against this doc OR the router.

---

## Quick reference — every endpoint

| # | Method + Path | User Story | Caller(s) | Purpose |
|---|---|---|---|---|
| 1 | `POST   /ai-api/meeting-details/` | US01 | Bot / AI team | Insert-or-update a meeting row when an Outlook event is detected (or after D365 resolves it to a deal) |
| 2 | `GET    /ai-api/meeting-details/` | US01 | Bot / Internal | Look up meetings by attendees + organiser + recurrence-start-date |
| 3 | `PATCH  /ai-api/meeting-details/{meeting_id}/status` | US01 | Bot | Transition the bot lifecycle state (`pending` / `scheduled` / `joining` / `joined` / `lobby_waiting` / `cancelled` / `rescheduled` / `failed`) |
| 4 | `DELETE /ai-api/meeting-details/{meeting_id}` | US01 | Bot / Internal | Soft-delete (flips `bot_status` to `cancelled`); preserves the row for late transcripts |
| 5 | `POST   /activity-details/` | US01 | Bot | Create the post-meeting activity-summary card the seller sees |
| 6 | `POST   /ai-api/transcripts/` | US02 | Bot | Start a transcript — REQUIRES the consent message text + timestamp |
| 7 | `POST   /ai-api/transcripts/{meeting_id}/segments` | US02 | Bot | Append a batch of speaker utterances during the meeting |
| 8 | `POST   /ai-api/transcripts/{meeting_id}/finalize` | US02 | Bot | Mark the transcript complete (clean meeting end) with overall confidence |
| 9 | `POST   /ai-api/transcripts/{meeting_id}/terminate` | US02 | Bot | Mark transcript as terminated-partial (organiser kicked / all left / bot crashed) |
| 10 | `GET    /ai-api/transcripts/{meeting_id}` | US02 | Frontend / AI team / QA | Fetch the full transcript + every segment, ordered by start_time |
| 11 | `POST   /ai-api/consent-emails/schedule` | US03 | Bot | Create per-recipient consent-email rows; returns `should_send=false` if window passed |
| 12 | `PATCH  /ai-api/consent-emails/{consent_id}/delivery` | US03 | Bot | Record bot's send-attempt outcome (`sent` / `failed`); schedules retry on failure |
| 13 | `GET    /ai-api/consent-emails/opt-out/{token}` | US03 | **Public** (browser) | Opt-out link click — records opt-out, cascades to `bot_status='cancelled'`, returns HTML |
| 14 | `GET    /ai-api/consent-emails/due-for-retry` | US03 | Bot | Polled retry queue — rows where `failed AND attempt<2 AND next_retry_at<=now()` |
| 15 | `GET    /ai-api/consent-emails/{meeting_id}` | US03 | Frontend / Audit | List all consent records for a meeting (delivery history, opt-outs) |
| 16 | `GET    /ai-api/meetings/{meeting_id}/consent-status` | US03 | Bot | Aggregated state — bot reads this RIGHT before joining; `any_opted_out=true` → don't join |
| 17 | `POST   /ai-api/data-tasks` | US04 | AI team / Daily-scan / FE inline (S1B) | Create a data-hygiene task — idempotent on (entity, kind); honours dismissal-suppression |
| 18 | `GET    /ai-api/data-tasks` | US04 | Frontend | Seller's To-Do list — filtered + ordered (confidence DESC, severity DESC, age ASC) |
| 19 | `GET    /ai-api/data-tasks/{task_id}` | US04 | Frontend | Single-task detail |
| 20 | `POST   /ai-api/data-tasks/{task_id}/resolve` | US04 | Frontend | Mark resolved with audit fields (`actor_id`, `resolved_value`, `resolved_at`) |
| 21 | `POST   /ai-api/data-tasks/{task_id}/dismiss` | US04 | Frontend | Dismiss with **required note**; suppresses future re-detection of same (entity, kind) |
| 22 | `POST   /ai-api/data-tasks/scan` | US04 | Admin / DevOps (HTTP) | Manual scan trigger — same code path as the cron-driven CLI |

---

## Endpoint x consumer matrix

| Endpoint | Bot | AI Team | Frontend | Browser | Admin/DevOps |
|---|:-:|:-:|:-:|:-:|:-:|
| `POST /ai-api/meeting-details/` | yes | yes | - | - | - |
| `GET /ai-api/meeting-details/` | yes | - | - | - | yes |
| `PATCH /ai-api/meeting-details/{id}/status` | yes | - | - | - | - |
| `DELETE /ai-api/meeting-details/{id}` | yes | - | - | - | yes |
| `POST /activity-details/` | yes | - | - | - | - |
| `POST /ai-api/transcripts/` | yes | - | - | - | - |
| `POST /ai-api/transcripts/{id}/segments` | yes | - | - | - | - |
| `POST /ai-api/transcripts/{id}/finalize` | yes | - | - | - | - |
| `POST /ai-api/transcripts/{id}/terminate` | yes | - | - | - | - |
| `GET /ai-api/transcripts/{id}` | - | yes | yes | - | yes |
| `POST /ai-api/consent-emails/schedule` | yes | - | - | - | - |
| `PATCH /ai-api/consent-emails/{id}/delivery` | yes | - | - | - | - |
| `GET /ai-api/consent-emails/opt-out/{token}` | - | - | - | yes | - |
| `GET /ai-api/consent-emails/due-for-retry` | yes | - | - | - | - |
| `GET /ai-api/consent-emails/{meeting_id}` | - | - | yes | - | yes |
| `GET /ai-api/meetings/{id}/consent-status` | yes | - | - | - | - |
| `POST /ai-api/data-tasks` | - | yes | yes (S1B) | - | yes (manual) |
| `GET /ai-api/data-tasks` | - | - | yes | - | yes |
| `GET /ai-api/data-tasks/{id}` | - | - | yes | - | - |
| `POST /ai-api/data-tasks/{id}/resolve` | - | - | yes | - | - |
| `POST /ai-api/data-tasks/{id}/dismiss` | - | - | yes | - | - |
| `POST /ai-api/data-tasks/scan` | - | - | - | - | yes |

**Legend:**
- **Bot** = the Note-Taking Agent (Microsoft Teams bot built by a separate team)
- **AI Team** = the post-meeting summary + transcript-signal NLP pipeline
- **Frontend** = the Lenovo Sales seller-facing UI
- **Browser** = an end customer's web browser (opt-out link)
- **Admin / DevOps** = ops tooling (manual triggers, cron jobs)

---

## Common conventions

### HTTP status codes

| Code | Meaning |
|------|---------|
| `200` | Success (GET, PATCH, POST when idempotent return) |
| `201` | Created (POST, first-time creation only — most of our POSTs return 200 because they're idempotent) |
| `400` | Business-rule violation (e.g. dismissing an already-resolved task) |
| `404` | Resource not found |
| `409` | Conflict (state transition not allowed; whitespace-only dismiss note) |
| `410` | Gone (opt-out link clicked after meeting started) |
| `422` | Pydantic validation — bad shape / missing field |
| `500` | Unexpected server error |

### Standard error body

```json
{
  "detail": "Human-readable message"
}
```

For Pydantic validation errors (422), the detail is a list of per-field errors:

```json
{
  "detail": [
    {
      "loc": ["body", "evidence_text"],
      "msg": "ensure this value has at least 1 characters",
      "type": "value_error.any_str.min_length"
    }
  ]
}
```

### Casing

All request bodies, response bodies, and query parameters use **snake_case**. This is intentional and consistent across every endpoint in this service. (The Lenovo D365 Sales backend uses camelCase — don't mix the two when copy-pasting between services.)

### IDs

Every entity ID is a **UUID v4**. We don't accept integer IDs anywhere. Cross-repo references (e.g. `entity_id` on a data-task pointing to a D365 opportunity) are FK-by-convention — there's no Postgres FK because the two services have separate databases.

---

## US01 — Joining the Meetings

The bot's lifecycle: from "Outlook event detected" to "bot left the meeting." Every state transition is recorded.

### 1. POST `/ai-api/meeting-details/`

**Caller:** Bot (typically twice — once on Outlook detection, once after D365 resolves the meeting to an opportunity)
**Purpose:** Insert-or-update a meeting record. Idempotent on `meeting_id`. Re-POST does NOT touch `bot_status` — that's owned by the lifecycle PATCH.

**Request body** — `MeetingDetailsCreate`:

```json
{
  "meeting_id": "f1b62c30-91b0-4b54-8c50-27f9e6cf4f0d",
  "meeting_start_time": "2026-06-20T14:00:00Z",
  "meeting_end_time": "2026-06-20T15:00:00Z",
  "platform": "Microsoft Teams",
  "title": "ThinkPad Fleet Review",
  "account_name": "Infosys",
  "attendees": "k.richter@db.com, seller@lenovo.com",
  "organiser_name": "Maria Hofer",
  "action": "scheduled",
  "body": "Quarterly fleet review.",
  "recurrence_pattern": "weekly",
  "recurrence_interval": 1,
  "recurrence_start_date": "2026-06-20",
  "recurrence_end_date": "2026-12-20",
  "opportunity_id": "d3c2b1a0-...",
  "account_id": "e4f5a6b7-..."
}
```

`opportunity_id` / `account_id` are populated by the bot AFTER calling D365's `POST /api/meetings/resolve-opportunity`. Both are optional.

**Response 200 / 201** — `MeetingDetailsResponse` (same shape + lifecycle fields: `bot_status`, `bot_status_reason`, `bot_last_event_at`).

---

### 2. GET `/ai-api/meeting-details/`

**Caller:** Bot (lookup before scheduling) / Internal admin
**Purpose:** Fetch meetings matching attendee + organiser + recurrence-start-date.

**Query parameters:**

| Name | Type | Required | Description |
|---|---|---|---|
| `attendees` | string | yes | Comma-separated email list (matches stored value) |
| `organiser_name` | string | yes | Display name of the meeting organiser |
| `recurrence_start_date` | date | yes | `YYYY-MM-DD` — narrows recurring-meeting matches |
| `title` | string | no | Optional title filter |

**Response 200** — `list[MeetingDetailsResponse]`.

---

### 3. PATCH `/ai-api/meeting-details/{meeting_id}/status`

**Caller:** Bot
**Purpose:** Record a lifecycle transition. The bot calls this every time its state changes — joining, joined, kicked, etc.

**Path:** `meeting_id` (UUID)

**Request body** — `MeetingStatusUpdate`:

```json
{
  "bot_status": "cancelled",
  "reason": "Calendar event deleted by organiser"
}
```

**Allowed `bot_status` values:** `pending`, `scheduled`, `joining`, `joined`, `lobby_waiting`, `cancelled`, `rescheduled`, `failed`.

**Response 200** — `MeetingStatusResponse`:

```json
{
  "meeting_id": "...",
  "bot_status": "cancelled",
  "bot_status_reason": "Calendar event deleted by organiser",
  "bot_last_event_at": "2026-06-09T10:23:45Z",
  "updated_at": "2026-06-09T10:23:45Z"
}
```

**Errors:** `404` (unknown meeting), `422` (status not in whitelist).

---

### 4. DELETE `/ai-api/meeting-details/{meeting_id}`

**Caller:** Bot / Internal
**Purpose:** Soft-delete (flips `bot_status` to `cancelled`). The row is preserved so late-arriving transcripts don't 404 and audit trails stay intact.

**Path:** `meeting_id` (UUID)
**Query param:** `reason` (optional string)

**Response 200** — `MeetingStatusResponse` (same as PATCH).

---

### 5. POST `/activity-details/`

**Caller:** Bot (after the meeting wraps + AI summary completes)
**Purpose:** Persist the activity-summary card the seller sees in the Activity tab.

**Request body** — `ActivityDetailsCreate`:

```json
{
  "time_since_meeting": "2 hours ago",
  "meeting_time": "2026-06-09T14:00:00Z",
  "duration_minutes": 47,
  "meeting_platform": "Microsoft Teams",
  "customer_sentiment": "Positive",
  "meeting_title": "ThinkPad Fleet Review",
  "account_name": "Infosys",
  "deal_stage": "Develop",
  "deal_value": 195000.00,
  "attendees": [
    {"name": "Klaus Richter", "role": "Procurement Lead"},
    {"name": "Maria Hofer", "role": "Account Executive"}
  ],
  "crm_updates_pending_approval": 3,
  "key_points_count": 7,
  "next_steps_count": 2,
  "review_url": "https://app.lenovo.com/ai-api/meetings/f1b62c30/summary"
}
```

`customer_sentiment` is one of: `Positive`, `Neutral`, `Negative`, `Mixed`.

**Response 201** — `ActivityDetailsResponse` (same shape + `meeting_id`).

---

## US02 — Consent & Recording (Transcript Pipeline)

The bot streams audio → speech-to-text → speaker diarization → batched POST to AIBackend. Every utterance is timestamped and tagged with the speaker's CRM context (resolved earlier via D365's contact resolver).

### 6. POST `/ai-api/transcripts/`

**Caller:** Bot (immediately after sending the consent chat message)
**Purpose:** Start a new transcript for a meeting. The consent message text + timestamp are REQUIRED — this is the server-side enforcement of AC #1 ("consent message sent before audio capture begins").

**Request body** — `TranscriptStartRequest`:

```json
{
  "meeting_id": "f1b62c30-91b0-4b54-8c50-27f9e6cf4f0d",
  "consent_message_text": "This meeting is being transcribed by the Lenovo Sales Assistant...",
  "consent_sent_at": "2026-06-09T14:00:05Z",
  "started_at": "2026-06-09T14:00:08Z"
}
```

`started_at` defaults to `consent_sent_at` if omitted. Must be `>= consent_sent_at` (consent must precede capture).

**Response 201** — `TranscriptResponse`:

```json
{
  "transcript_id": "...",
  "meeting_id": "...",
  "opportunity_id": "...",
  "account_id": "...",
  "status": "in_progress",
  "consent_message_text": "...",
  "consent_sent_at": "2026-06-09T14:00:05Z",
  "overall_confidence_score": null,
  "segment_count": 0,
  "terminated_reason": null,
  "started_at": "2026-06-09T14:00:08Z",
  "finalized_at": null
}
```

`status` is one of: `in_progress`, `finalized`, `terminated_partial`.

**Errors:** `404` (unknown meeting), `409` (transcript already exists for this meeting), `422` (missing consent fields).

---

### 7. POST `/ai-api/transcripts/{meeting_id}/segments`

**Caller:** Bot (continuously during meeting — bot batches at its discretion)
**Purpose:** Append a batch of speaker utterances to a transcript.

**Path:** `meeting_id` (UUID)

**Request body** — `TranscriptSegmentsAppendRequest`:

```json
{
  "segments": [
    {
      "speaker_name": "Klaus Richter",
      "speaker_email": "k.richter@db.com",
      "speaker_role": "Procurement Lead",
      "speaker_contact_id": "c1d2e3f4-...",
      "utterance_text": "We need delivery before June 15th.",
      "start_time": "2026-06-09T14:05:12Z",
      "end_time": "2026-06-09T14:05:15Z",
      "confidence_score": 0.94
    },
    {
      "speaker_name": "Unknown Attendee",
      "speaker_email": null,
      "speaker_role": null,
      "speaker_contact_id": null,
      "utterance_text": "Yes, that aligns with our budget cycle.",
      "start_time": "2026-06-09T14:05:18Z",
      "end_time": "2026-06-09T14:05:22Z",
      "confidence_score": 0.81
    }
  ]
}
```

`speaker_name` is REQUIRED on every segment — for unidentified speakers the bot must default to `"Unknown Attendee"` (AC #3). The other speaker_* fields are nullable for unknowns. `confidence_score` is `0.0`–`1.0` inclusive.

**Response 200** — `TranscriptSegmentsAppendResponse`:

```json
{
  "transcript_id": "...",
  "meeting_id": "...",
  "appended_count": 2,
  "segment_count": 47
}
```

**Errors:** `404` (transcript not found), `409` (transcript already finalized/terminated), `422` (end_time before start_time).

---

### 8. POST `/ai-api/transcripts/{meeting_id}/finalize`

**Caller:** Bot (clean meeting end)
**Purpose:** Mark the transcript as finalised. Only valid from `status='in_progress'`.

**Path:** `meeting_id` (UUID)

**Request body** — `TranscriptFinalizeRequest`:

```json
{
  "overall_confidence_score": 0.89,
  "finalized_at": "2026-06-09T14:47:30Z"
}
```

`finalized_at` defaults to server `now()` if omitted.

**Response 200** — `TranscriptResponse` with `status='finalized'` and `finalized_at` populated.

**Errors:** `404`, `409` (already finalized/terminated).

---

### 9. POST `/ai-api/transcripts/{meeting_id}/terminate`

**Caller:** Bot (early stop — kicked out, all human attendees left, bot crash)
**Purpose:** Mark the transcript as `terminated_partial` so consumers can distinguish it from a clean finalisation. NOT for clean meeting ends — use `/finalize` for those.

**Path:** `meeting_id` (UUID)

**Request body** — `TranscriptTerminateRequest`:

```json
{
  "reason": "organizer_removed",
  "terminated_at": "2026-06-09T14:23:11Z",
  "overall_confidence_score": 0.74
}
```

`reason` is one of: `organizer_removed`, `all_left`, `bot_failure`. (NOT `meeting_ended` — that's `/finalize`'s job.)

**Response 200** — `TranscriptResponse` with `status='terminated_partial'` and `terminated_reason` populated.

---

### 10. GET `/ai-api/transcripts/{meeting_id}`

**Caller:** Frontend (Activity tab — full transcript view) / AI team (post-meeting summary input) / QA / Audit
**Purpose:** Fetch the full transcript + every segment.

**Path:** `meeting_id` (UUID)

**Response 200** — `TranscriptWithSegmentsResponse`:

```json
{
  "transcript": { /* TranscriptResponse */ },
  "segments": [
    { "segment_id": "...", "speaker_name": "...", "utterance_text": "...", "start_time": "...", "end_time": "...", "confidence_score": 0.94 }
  ]
}
```

Segments are returned ordered by `start_time` ascending. No pagination in S1A — for very long meetings (>3000 segments) we'll add it later.

**Errors:** `404` (no transcript for this meeting).

---

## US03 — Pre-meeting Consent Emails

External attendees get an email before the meeting with an opt-out link. If anyone opts out, the bot does NOT join (no override, AC #7). For meetings inside the consent window, the bot falls back to in-meeting chat consent (US02).

### 11. POST `/ai-api/consent-emails/schedule`

**Caller:** Bot (after meeting becomes eligible — i.e. US01 has resolved an opp / account)
**Purpose:** Create per-recipient consent-email rows. Filters internal Lenovo domains automatically using `INTERNAL_EMAIL_DOMAINS` config. Idempotent — re-calling for the same meeting returns the existing tokens.

**Request body** — `ConsentScheduleRequest`:

```json
{
  "meeting_id": "f1b62c30-91b0-4b54-8c50-27f9e6cf4f0d",
  "recipients": [
    {"email": "k.richter@db.com", "name": "Klaus Richter"},
    {"email": "rajesh.k@infosys.com", "name": "Rajesh Kumar"},
    {"email": "seller@lenovo.com", "name": "Maria Hofer"}
  ]
}
```

**Response 200** — `ConsentScheduleResponse`:

Window-open case:

```json
{
  "meeting_id": "...",
  "should_send": true,
  "fallback": null,
  "recipients": [
    {
      "consent_id": "...",
      "recipient_email": "k.richter@db.com",
      "recipient_name": "Klaus Richter",
      "opt_out_token": "...",
      "opt_out_url": "http://localhost:8001/ai-api/consent-emails/opt-out/...",
      "scheduled_send_at": "2026-06-09T13:00:00Z",
      "delivery_status": "pending"
    }
  ],
  "filtered_internal_count": 1,
  "seller_name": "Maria Hofer",
  "system_email_address": "sales-assistant@lenovo.com"
}
```

Window-passed case (bot falls back to US02 in-meeting chat):

```json
{
  "meeting_id": "...",
  "should_send": false,
  "fallback": "in_meeting_chat",
  "recipients": [],
  "filtered_internal_count": 1
}
```

**Errors:** `404` (unknown meeting).

---

### 12. PATCH `/ai-api/consent-emails/{consent_id}/delivery`

**Caller:** Bot (after each send attempt)
**Purpose:** Record the delivery outcome. On `failed`, the service computes `next_retry_at = now + CONSENT_RETRY_DELAY_MINUTES`. On second failure, falls back to `fallback_to_in_meeting`.

**Path:** `consent_id` (UUID)

**Request body** — `ConsentDeliveryUpdateRequest`:

```json
{
  "status": "failed",
  "failure_reason": "SMTP bounce: mailbox does not exist",
  "attempted_at": "2026-06-09T13:00:00Z"
}
```

`status` is `sent` or `failed` (no other values accepted from the bot).

**Response 200** — `ConsentDeliveryUpdateResponse`:

```json
{
  "consent_id": "...",
  "meeting_id": "...",
  "recipient_email": "...",
  "delivery_status": "failed",
  "attempt_count": 1,
  "last_attempt_at": "2026-06-09T13:00:00Z",
  "next_retry_at": "2026-06-09T13:10:00Z",
  "failure_reason": "SMTP bounce: mailbox does not exist"
}
```

If `delivery_status` is `fallback_to_in_meeting`, the bot has exhausted retries and must use US02's chat-consent flow instead.

**Errors:** `404` (unknown consent_id).

---

### 13. GET `/ai-api/consent-emails/opt-out/{token}`

**Caller:** **PUBLIC** — the customer's web browser, when they click the link in the consent email
**Purpose:** Record the opt-out. Cascades to `tbl_schedule_meetings.bot_status='cancelled'` with reason `participant_opted_out`. Returns a server-rendered HTML confirmation page (SUCC_MSG_0010).

**Path:** `token` (URL-safe string, 256 bits of entropy)

**Response 200** — `text/html` (NOT JSON). Idempotent — clicking the same link twice still returns 200 with the same confirmation.

**Errors:**
- `404` — invalid token
- `410 Gone` — meeting has already started; opt-out link expired

This is the only public-facing endpoint in AIBackend. It deliberately doesn't require auth — that's the whole point.

---

### 14. GET `/ai-api/consent-emails/due-for-retry`

**Caller:** Bot (polls periodically — interval is bot's choice)
**Purpose:** Returns rows where `delivery_status='failed' AND attempt_count<2 AND next_retry_at<=now()`.

**Response 200** — `ConsentRetryQueueResponse`:

```json
{
  "items": [
    {
      "consent_id": "...",
      "meeting_id": "...",
      "recipient_email": "...",
      "recipient_name": "...",
      "opt_out_url": "http://localhost:8001/ai-api/consent-emails/opt-out/...",
      "attempt_count": 1,
      "last_attempt_at": "2026-06-09T13:00:00Z",
      "next_retry_at": "2026-06-09T13:10:00Z",
      "failure_reason": "SMTP bounce..."
    }
  ]
}
```

Bot picks each one up, re-sends, and PATCHes `/delivery` again with the new outcome.

---

### 15. GET `/ai-api/consent-emails/{meeting_id}`

**Caller:** Frontend (Activity tab, audit view) / Internal audit
**Purpose:** List all consent-email rows for a meeting — full delivery + opt-out history.

**Path:** `meeting_id` (UUID)

**Response 200** — `ConsentEmailListResponse`:

```json
{
  "meeting_id": "...",
  "items": [
    {
      "consent_id": "...",
      "meeting_id": "...",
      "recipient_email": "...",
      "recipient_name": "...",
      "opt_out_token": "...",
      "opt_out_url": "...",
      "scheduled_send_at": "...",
      "attempt_count": 2,
      "last_attempt_at": "...",
      "next_retry_at": null,
      "delivery_status": "sent",
      "failure_reason": null,
      "opted_out_at": null,
      "opt_out_ip": null,
      "seller_notified_at": null,
      "created_at": "...",
      "updated_at": "..."
    }
  ]
}
```

---

### 16. GET `/ai-api/meetings/{meeting_id}/consent-status`

**Caller:** Bot (RIGHT before joining the meeting)
**Purpose:** Aggregated state. Bot's logic is literally `if response.any_opted_out: return` — AC #7, no exceptions.

**Path:** `meeting_id` (UUID)

**Response 200** — `MeetingConsentStatusResponse`:

```json
{
  "meeting_id": "...",
  "any_opted_out": false,
  "opt_out_count": 0,
  "total_recipients": 2,
  "consent_mechanism": "pre_meeting_email",
  "pending_retries": 0,
  "sent_count": 2,
  "failed_count": 0,
  "fallback_count": 0
}
```

`consent_mechanism` is one of:
- `pre_meeting_email` — at least one row, all sent / opted-out
- `in_meeting_chat` — no rows OR all fell back; bot uses US02 chat
- `mixed` — some sent + some fell back; bot must use chat too

If `any_opted_out=true`, the bot does NOT join. Period. AC #7.

---

## US04 — Data Hygiene, Validation & Intelligent Alerts

The seller's data-quality To-Do queue. Tasks come from three sources: AI team's transcript-signal NLP, the daily-scan CLI job, and (S1B) FE inline validators. Idempotent on `(entity_kind, entity_id, task_kind)` — re-detection never duplicates, and dismissed tasks suppress re-creation (AC #5).

### 17. POST `/ai-api/data-tasks`

**Caller:** AI team (transcript signals) / Daily-scan job (internal) / FE inline validators (S1B)
**Purpose:** Create a task — or return the existing one for the same `(entity_kind, entity_id, task_kind)` if it's open OR dismissed.

**Request body** — `DataTaskCreateRequest`:

```json
{
  "owner_id": "0b2cc1f8-3b6e-4b8a-9e7f-2a8c39b71a4e",
  "entity_kind": "opportunity",
  "entity_id": "f1b62c30-91b0-4b54-8c50-27f9e6cf4f0d",
  "task_kind": "transcript_signal_close_date_different",
  "severity": "high",
  "confidence": "high",
  "field_name": "close_date",
  "current_value": "2026-06-30",
  "suggested_value": "2026-06-15",
  "evidence_ref": "transcript_segment_id=ab12cd34",
  "evidence_text": "Customer mentioned: We need it before June 15th on the call.",
  "created_by_source": "transcript"
}
```

**Field reference:**

| Field | Type | Required | Notes |
|---|---|---|---|
| `owner_id` | UUID | yes | The seller this task lands on |
| `entity_kind` | enum | yes | `account` / `contact` / `opportunity` |
| `entity_id` | UUID | yes | D365 entity UUID (cross-repo, no FK) |
| `task_kind` | string | yes | Free-form text. AI team should use the canonical strings from `app/models/data_task.py` |
| `severity` | enum | no | `high` / `medium` / `low` (default `medium`) |
| `confidence` | enum | no | `high` / `medium` / `low` / `null`. Drives ordering. |
| `field_name` | string | no | Which D365 field needs the seller's attention |
| `current_value` | string | no | Display value (FE renders strikethrough) |
| `suggested_value` | string | no | What the system / AI suggests |
| `evidence_ref` | string | no | Stable handle for the source (e.g. `transcript_segment_id=<uuid>`) |
| `evidence_text` | string | **yes** | Plain-language WHY (AC #3 grounding). Empty is rejected |
| `created_by_source` | enum | yes | `transcript` / `scan` / `inline` / `manual` |

**Response 200** — `DataTaskCreateResponse`:

```json
{
  "task": {
    "task_id": "...",
    "owner_id": "...",
    "entity_kind": "opportunity",
    "entity_id": "...",
    "task_kind": "transcript_signal_close_date_different",
    "severity": "high",
    "status": "open",
    "confidence": "high",
    "field_name": "close_date",
    "current_value": "2026-06-30",
    "suggested_value": "2026-06-15",
    "evidence_ref": "transcript_segment_id=ab12cd34",
    "evidence_text": "Customer mentioned...",
    "created_by_source": "transcript",
    "dismissal_note": null,
    "resolved_value": null,
    "actor_id": null,
    "resolved_at": null,
    "dismissed_at": null,
    "created_at": "2026-06-09T14:50:00Z",
    "updated_at": "2026-06-09T14:50:00Z"
  },
  "was_existing": false
}
```

`was_existing: true` is **success, not an error** — it means we already had an open or dismissed task with the same key. Don't retry.

**Errors:** `422` (empty `evidence_text`, missing required field).

---

### 18. GET `/ai-api/data-tasks`

**Caller:** Frontend (To-Do List view)
**Purpose:** Filtered + ordered list of tasks. Ordering: `confidence DESC NULLS LAST, severity DESC, created_at ASC`.

**Query parameters:**

| Name | Type | Notes |
|---|---|---|
| `ownerId` | UUID | **AC #11 portfolio scoping** — FE always sends this |
| `status` | enum | `open` / `resolved` / `dismissed` |
| `kind` | string | Filter by `task_kind` |
| `entityKind` | enum | `account` / `contact` / `opportunity` |
| `entityId` | UUID | All tasks on one record (deal-detail page) |
| `limit` | int | Default 50, max 500 |
| `offset` | int | Default 0 |

**Response 200** — `DataTaskListResponse`:

```json
{
  "items": [ /* array of DataTaskOut */ ],
  "total": 12
}
```

---

### 19. GET `/ai-api/data-tasks/{task_id}`

**Caller:** Frontend (when seller clicks into a task)

**Path:** `task_id` (UUID)

**Response 200** — `DataTaskOut` (the same shape nested under `task` in the create response).

**Errors:** `404` (unknown).

---

### 20. POST `/ai-api/data-tasks/{task_id}/resolve`

**Caller:** Frontend (when seller approves the suggestion / marks fixed)
**Purpose:** Mark resolved with audit fields. Idempotent.

**Path:** `task_id` (UUID)

**Request body** — `DataTaskResolveRequest`:

```json
{
  "actor_id": "7b4d4e9e-1c2c-4d9c-9f1e-9a6c4b8a2d3f",
  "resolved_value": "2026-06-15"
}
```

**Response 200** — `DataTaskResolveResponse`:

```json
{
  "task": { /* DataTaskOut with status='resolved', resolved_at + actor_id + resolved_value populated */ },
  "was_already_resolved": false
}
```

Second resolve returns `was_already_resolved: true` and does NOT overwrite the original audit trail.

**Errors:** `404`, `409` (task is currently dismissed — re-open via a fresh POST first).

---

### 21. POST `/ai-api/data-tasks/{task_id}/dismiss`

**Caller:** Frontend (when seller clicks Dismiss and types a reason)
**Purpose:** Dismiss with a **required** note. Suppresses future re-detection of the same `(entity, kind)` (AC #5).

**Path:** `task_id` (UUID)

**Request body** — `DataTaskDismissRequest`:

```json
{
  "actor_id": "7b4d4e9e-1c2c-4d9c-9f1e-9a6c4b8a2d3f",
  "note": "Customer confirmed the original date — this transcript line was about a different procurement."
}
```

**Response 200** — `DataTaskDismissResponse`:

```json
{
  "task": { /* DataTaskOut with status='dismissed', dismissed_at + actor_id + dismissal_note populated */ },
  "was_already_dismissed": false
}
```

**Errors:**
- `422` — empty `note` (Pydantic min_length=1)
- `409` — whitespace-only `note` (slips past Pydantic; service-layer rejects), or task is already resolved
- `404` — unknown task

---

### 22. POST `/ai-api/data-tasks/scan`

**Caller:** Admin / DevOps (manual HTTP trigger). The cron uses the CLI: `python -m app.jobs.scan_data_tasks`.
**Purpose:** Run the data-hygiene scan synchronously. Same code path as the CLI.

**Prerequisite:** the **Lenovo D365 Sales backend must be running** — this scan calls `GET /api/opportunities` and `GET /api/opportunities/{id}/risks` over HTTP.

**Request body** — `ScanRunRequest` (all fields optional):

```json
{
  "limit": 50,
  "dry_run": true
}
```

**Response 200** — `ScanRunResponse`:

```json
{
  "total_scanned": 50,
  "tasks_created": 7,
  "tasks_skipped_existing": 3,
  "tasks_skipped_dismissed": 1,
  "opportunities_with_errors": 0,
  "dry_run": true
}
```

Invariant: `total_scanned == tasks_created + tasks_skipped_existing + tasks_skipped_dismissed + opportunities_with_errors`.

**Errors:** `500` if D365 Sales backend is unreachable.

---

## Outbound: this service calls Lenovo D365 Sales

The AIBackend pulls data from the Lenovo D365 Sales backend over HTTP — never touching D365 / Dataverse directly. The endpoints we depend on:

| D365 Sales endpoint | Used by | Purpose |
|---|---|---|
| `POST /api/meetings/resolve-opportunity` | US01 — bot, before scheduling | Map a meeting (attendees + subject) to a CRM opportunity |
| `POST /api/contacts/resolve-by-emails` | US02 — bot, at meeting start | Bulk-resolve attendee emails to CRM contact context (name / role / contact_id) |
| `GET /api/opportunities` (paginated) | US04 — daily-scan job | Stream active opps for detector evaluation |
| `GET /api/opportunities/{id}/risks` | US04 — daily-scan job | Per-opp risk flags (already computed by D365's `deal_risks.py`) |

These are documented in `Lenovo D365 Sales/API_CONTRACT.md`. If the contract changes there, the wrapper in `app/clients/d365_client.py` is the only file in this service that needs updating.

---

## Versioning

| Version | Date | Changes |
|---|---|---|
| 0.1.0 | 2026-06 | US01 — Meeting lifecycle (5 endpoints) |
| 0.2.0 | 2026-06 | US02 — Transcript pipeline (5 endpoints) |
| 0.3.0 | 2026-06 | US03 — Consent emails (6 endpoints) |
| 0.4.0 | 2026-06 | US04 — Data hygiene (6 endpoints) |

When a backwards-incompatible change is needed, bump the major (1.0.0) and document the migration here. For now, single-version contract.

---

## Changelog convention

Every PR that adds / removes / modifies an endpoint MUST update this file in the same commit. Reviewers should reject any router-touching PR that doesn't.

---

## Contacts

- **Backend (this service):** Sanmay
- **Bot / AI team:** Namisha + AI integration leads
- **Frontend:** Lenovo Sales FE team
- **DevOps:** Lenovo Cloud Ops

For questions on a specific endpoint, the per-story handoff docs are the deeper reference:

- `US01_BACKEND_HANDOFF_FOR_AI_TEAM.md`
- `US02_BACKEND_HANDOFF_FOR_AI_TEAM.md`
- `US03_BACKEND_HANDOFF_FOR_AI_TEAM.md`
- `US04_BACKEND_HANDOFF_FOR_AI_TEAM.md`





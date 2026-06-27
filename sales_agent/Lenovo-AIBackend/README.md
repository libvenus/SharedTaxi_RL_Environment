# Lenovo AI Backend

FastAPI service that backs the **Note-Taking Agent** (Sprint 1A onwards) for
the Lenovo D365 Sales programme. It owns:

- The `tbl_schedule_meetings` table — every Teams meeting the bot is asked
  to join, plus its lifecycle state (pending / scheduled / joined /
  cancelled / rescheduled / failed).
- The `tbl_activty_details` table — the per-meeting summary card that the
  Sales rep sees on the Activity tab.
- The `tbl_meeting_transcript` + `tbl_meeting_transcript_segment` tables —
  the structured transcript pipeline that captures consent proof, every
  utterance the bot hears, and the finalisation/termination event for
  each meeting (Sprint 1A · US02).
- The `tbl_meeting_consent_email` table — the pre-meeting consent email
  pipeline: scheduling, delivery audit, retry tracking, and opt-out
  handling (Sprint 1A · US03). Opt-outs cascade to `bot_status='cancelled'`
  on the parent meeting.
- The `tbl_data_task` table — the seller's data-hygiene To-Do queue
  (Sprint 1A · US04). One row per detected data-quality issue, sourced
  from the AI team's transcript-signal NLP pipeline, the daily-scan CLI
  job, or the FE inline validators. Each task carries a grounding
  reference, an audit trail, and dismissal-suppression for AC #5.
- The HTTP client to the **Lenovo D365 Sales** backend, which resolves
  meetings → CRM opportunities via attendee emails (US01) and resolves
  attendee emails → CRM contact context for transcript speaker tagging
  (US02).

The actual bot (Microsoft Graph subscriptions, Teams Bot SDK, audio
capture, speech-to-text, speaker diarization, organiser-only kick
permission) is built by a separate team and calls into this service
over HTTP.

## Repository layout

```
app/
├── api/                       # FastAPI routers (one file per domain)
│   ├── meeting_details.py     # POST / GET / PATCH / DELETE meetings (US01)
│   ├── activity_details.py    # POST activity summary cards
│   ├── transcripts.py         # 5 transcript-pipeline routes        (US02)
│   ├── consent_emails.py      # 6 consent-email + opt-out routes     (US03)
│   └── data_tasks.py          # 6 data-hygiene task routes           (US04)
├── clients/                   # Outbound HTTP clients
│   └── d365_client.py         # → Lenovo D365 Sales backend (resolve_opportunity + resolve_contacts_by_email + list_active_opportunities + fetch_opportunity_risks)
├── core/
│   └── config.py              # .env-driven settings (DB + D365 + consent window + data-hygiene thresholds)
├── db/
│   └── database.py            # SQLAlchemy session + Base
├── jobs/                      # CLI jobs (cron-driven)
│   └── scan_data_tasks.py     # Daily data-hygiene scan              (US04)
├── models/                    # SQLAlchemy ORM models
│   ├── schedulemeeting.py     # tbl_schedule_meetings              (US01)
│   ├── activityDetails.py     # tbl_activty_details
│   ├── transcript.py          # tbl_meeting_transcript[_segment]   (US02)
│   ├── consent_email.py       # tbl_meeting_consent_email          (US03)
│   └── data_task.py           # tbl_data_task                      (US04)
├── schema/                    # Pydantic request/response models
└── services/                  # DB-side business logic (no HTTP)
    ├── meeting_details_service.py
    ├── transcript_service.py  # start / append / finalize / terminate / get
    ├── consent_email_service.py  # schedule / delivery / opt-out / status
    ├── data_task_service.py   # task CRUD + idempotency contract     (US04)
    └── data_task_detectors.py # 4 pure detector functions             (US04)
sql/                           # Idempotent migration scripts (run by hand)
├── 2026_06_us01_meeting_lifecycle.sql
├── 2026_06_us02_meeting_transcript.sql
├── 2026_06_us03_consent_email.sql
└── 2026_06_us04_data_task.sql
tests/                         # pytest smoke tests (run with `pytest -q`)
├── test_meeting_lifecycle.py            # US01 (8 tests)
├── test_transcript_lifecycle.py         # US02 (9 tests)
├── test_consent_capture_lifecycle.py    # US03 (15 tests)
└── test_data_task_lifecycle.py          # US04 (14 tests)
```

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux / macOS
pip install -r requirements.txt
```

Create a `.env` next to this README:

```
DB_USER=...
DB_PASSWORD=...
DB_HOST=...
DB_PORT=5432
DB_NAME=lenovosales

# Sprint 1A · US01 — points the bot at the D365 backend.
# Defaults to localhost:8000 if omitted.
D365_BASE_URL=http://localhost:8000
D365_TIMEOUT_SECONDS=5.0

# Sprint 1A · US03 — pre-meeting consent email pipeline
# How long before meeting start the email is scheduled (default 60 min)
CONSENT_WINDOW_MINUTES=60
# Comma-separated list of internal Lenovo email domains.
# Attendees with these domains are NOT sent the consent email.
INTERNAL_EMAIL_DOMAINS=lenovo.com,motorola.com
# Public-facing base URL for the opt-out link in the email body.
# In dev: http://localhost:8001 (your AIBackend). In prod: https://aibackend.lenovo.com
OPT_OUT_BASE_URL=http://localhost:8001
# The "from" address — bot constructs "<Seller Name> via Lenovo Sales Assistant"
SYSTEM_EMAIL_ADDRESS=sales-assistant@lenovo.com
# Retry delay on send failure (default 10 min — AC #10)
CONSENT_RETRY_DELAY_MINUTES=10
```

Apply the Sprint 1A schema migrations once (each is idempotent — safe to re-run):

```bash
# US01 — meeting lifecycle (tbl_schedule_meetings additions)
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f sql/2026_06_us01_meeting_lifecycle.sql

# US02 — transcript pipeline (two new tables)
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f sql/2026_06_us02_meeting_transcript.sql

# US03 — consent email pipeline (one new table)
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f sql/2026_06_us03_consent_email.sql
```

Run the dev server:

```bash
uvicorn app.main:app --reload --port 8001
```

Swagger UI: <http://localhost:8001/docs>

## Bot lifecycle state machine

Maintained as a single CHECK constraint on `tbl_schedule_meetings.bot_status`
(see `sql/2026_06_us01_meeting_lifecycle.sql`) and a `Literal` in
`app/schema/schedulemeeting.py`.

```
                   ┌────────────┐
                   │  pending   │ ← POST /meeting-details/ creates here
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
    • cancelled   — calendar event deleted by organiser
    • rescheduled — meeting time moved (bot re-schedules itself)
```

## API reference (Sprint 1A)

### `POST /meeting-details/`

Insert-or-update a meeting. Idempotent on `meeting_id`. The bot calls this
twice in the typical flow:

1. Initial: when the Outlook event is detected.
2. Post-resolve: after `d365_client.resolve_opportunity()` returns the IDs.

`bot_status` is **preserved** across re-POSTs — only the lifecycle PATCH
owns it. So re-POSTing a meeting that's already `joined` doesn't revert
it to `pending`.

```bash
curl -X POST http://localhost:8001/meeting-details/ \
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

### `GET /meeting-details/`

Filtered list. Required query params: `attendees`, `organiser_name`,
`recurrence_start_date`. Optional: `title`.

```bash
curl "http://localhost:8001/meeting-details/?attendees=k.richter&organiser_name=Maria&recurrence_start_date=2026-06-15"
```

### `PATCH /meeting-details/{meeting_id}/status`

Transition the bot lifecycle state. Pydantic enforces the `bot_status`
whitelist — passing an unknown value 422s before any DB write happens.

> **Casing:** this service uses **snake_case** keys throughout, matching
> the existing `/meeting-details/` and `/activity-details/` endpoints.
> Don't confuse with the camelCase used by the D365 Sales backend.

```bash
# Bot has just joined the meeting
curl -X PATCH \
  http://localhost:8001/meeting-details/11111111-1111-1111-1111-111111111111/status \
  -H "Content-Type: application/json" \
  -d '{ "bot_status": "joined" }'

# Lobby admit timed out
curl -X PATCH \
  http://localhost:8001/meeting-details/11111111-1111-1111-1111-111111111111/status \
  -H "Content-Type: application/json" \
  -d '{ "bot_status": "failed", "reason": "Lobby admit timeout (90s)" }'
```

Response:

```jsonc
{
  "meeting_id":        "11111111-1111-1111-1111-111111111111",
  "bot_status":        "failed",
  "bot_status_reason": "Lobby admit timeout (90s)",
  "bot_last_event_at": "2026-06-15T15:01:32.114Z",
  "updated_at":        "2026-06-15T15:01:32.114Z"
}
```

### `DELETE /meeting-details/{meeting_id}`

Soft-delete: flips `bot_status` to `cancelled` and stamps the timestamp.
The row is preserved so late-arriving transcripts don't 404.

```bash
curl -X DELETE \
  "http://localhost:8001/meeting-details/11111111-1111-1111-1111-111111111111?reason=Calendar+event+deleted+by+organiser"
```

### `POST /activity-details/`

(Existing endpoint — pre-Sprint-1A — unchanged.) Inserts a per-meeting
summary card (counts of action items, key points, etc.).

## Sprint 1A · US02 — Transcript pipeline

The transcript pipeline persists everything the bot captures during a
meeting: the consent proof, every utterance, and the finalisation /
termination event. Five endpoints, all under `/transcripts`.

### Transcript lifecycle state machine

Maintained as a CHECK constraint on `tbl_meeting_transcript.status`
(see `sql/2026_06_us02_meeting_transcript.sql`) and a `Literal` in
`app/schema/transcript.py`.

```
            POST /transcripts/   (consent_message_text + consent_sent_at REQUIRED)
                       │
                       ▼
              ┌─────────────────┐
              │   in_progress   │ ← bot pushes segments here continuously
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────────┐
        ▼              ▼                  ▼
   /finalize       /terminate         (bot crashed?
   meeting_ended   organizer_removed   sweeper marks
                   all_left            'bot_failure' — 1B)
                   bot_failure
        ▼              ▼
  ┌────────────┐  ┌──────────────────────┐
  │ finalized  │  │ terminated_partial   │
  └────────────┘  └──────────────────────┘
```

- **`in_progress`** — bot is actively capturing; segments accepted
- **`finalized`** — clean meeting end + `overall_confidence_score` set
- **`terminated_partial`** — early stop (organiser kicked / bot failed /
  all-left); the captured segments are preserved

### `POST /transcripts/`

Open a transcript. The bot calls this **after** posting `CONF_MSG_0004`
in the Teams chat — the consent fields are NOT NULL on the row, so a
transcript literally cannot exist without compliance proof.

Pre-condition: `tbl_schedule_meetings.bot_status` must be one of
`{joined, lobby_waiting, joining}` — the bot must actually be in (or
about to enter) the meeting.

```bash
curl -X POST http://localhost:8001/transcripts/ \
  -H "Content-Type: application/json" \
  -d '{
    "meeting_id":           "11111111-1111-1111-1111-111111111111",
    "consent_message_text": "Hi everyone — this meeting is being recorded by the Lenovo Note-Taking Agent for transcript and follow-up purposes. (CONF_MSG_0004)",
    "consent_sent_at":      "2026-06-15T15:00:30+00:00"
  }'
```

Response (201):

```jsonc
{
  "transcript_id":            "a1b2c3d4-...",
  "meeting_id":               "11111111-...",
  "opportunity_id":           "b0000001-...",  // denormalised from meeting
  "account_id":               "6dc95c38-...",
  "status":                   "in_progress",
  "consent_message_text":     "Hi everyone — ...",
  "consent_sent_at":          "2026-06-15T15:00:30Z",
  "overall_confidence_score": null,
  "segment_count":            0,
  "terminated_reason":        null,
  "started_at":               "2026-06-15T15:00:30Z",
  "finalized_at":             null
}
```

Errors:
- `400` — `bot_status` not in `{joined, lobby_waiting, joining}`
- `404` — meeting not found
- `409` — a transcript already exists for this meeting
- `422` — missing consent fields, or `started_at < consent_sent_at`

### `POST /transcripts/{meeting_id}/segments`

Append a batch of utterances. The bot calls this repeatedly during the
meeting — minimum 1 segment per call, no upper bound. Speaker
attribution is the bot's job (use `/api/contacts/resolve-by-emails`
once at meeting start to enrich attendees, then tag locally). Use
`speaker_name = "Unknown Attendee"` for unidentified speakers — never
fabricate a name (AC #3).

```bash
curl -X POST \
  http://localhost:8001/transcripts/11111111-1111-1111-1111-111111111111/segments \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {
        "speaker_name":       "Klaus Richter",
        "speaker_email":      "k.richter@db.com",
        "speaker_role":       "CTO",
        "speaker_contact_id": "e7cf7aaf-cd45-4a9a-9c6b-39ba8d1b6c1a",
        "utterance_text":     "Thanks for joining. Lets start with the migration plan.",
        "start_time":         "2026-06-15T15:01:00+00:00",
        "end_time":           "2026-06-15T15:01:04+00:00",
        "confidence_score":   0.93
      },
      {
        "speaker_name":     "Unknown Attendee",
        "utterance_text":   "Sorry I am late.",
        "start_time":       "2026-06-15T15:01:30+00:00",
        "end_time":         "2026-06-15T15:01:32+00:00",
        "confidence_score": 0.71
      }
    ]
  }'
```

Response:

```jsonc
{
  "transcript_id":   "a1b2c3d4-...",
  "meeting_id":      "11111111-...",
  "appended_count":  2,
  "segment_count":   2     // post-append total
}
```

Errors:
- `400` — transcript not in `in_progress` state
- `404` — no transcript exists for this meeting
- `422` — `confidence_score` outside `[0, 1]`, `end_time < start_time`,
  empty `segments`, etc.

### `POST /transcripts/{meeting_id}/finalize`

Clean meeting end. Call this when the meeting ends naturally (everyone
left, scheduled end time hit). Sets `status='finalized'` and writes
`terminated_reason='meeting_ended'`. Bot computes the overall
confidence (typically a weighted average of segment confidences); we
record what it sends.

```bash
curl -X POST \
  http://localhost:8001/transcripts/11111111-1111-1111-1111-111111111111/finalize \
  -H "Content-Type: application/json" \
  -d '{
    "overall_confidence_score": 0.91,
    "finalized_at":             "2026-06-15T16:00:12+00:00"
  }'
```

### `POST /transcripts/{meeting_id}/terminate`

Early stop. Call this when the bot leaves before the natural end —
organiser kicked the bot, all human participants left, or the bot
crashed. Sets `status='terminated_partial'` with the reason.

> Use `/finalize` for clean ends — `/terminate` deliberately does NOT
> accept `meeting_ended` as a reason, so the audit trail stays
> unambiguous.

```bash
curl -X POST \
  http://localhost:8001/transcripts/11111111-1111-1111-1111-111111111111/terminate \
  -H "Content-Type: application/json" \
  -d '{ "reason": "organizer_removed" }'
```

Valid reasons: `organizer_removed`, `all_left`, `bot_failure`.

### `GET /transcripts/{meeting_id}`

Read-only fetch — for the FE Activity tab to render the transcript.
Returns metadata + every segment ordered by `start_time`. No
pagination in v1.

```bash
curl http://localhost:8001/transcripts/11111111-1111-1111-1111-111111111111
```

Response shape:

```jsonc
{
  "transcript": {
    "transcript_id": "...",
    "status":        "finalized",
    // ...all the fields from POST /transcripts/'s response
  },
  "segments": [
    { "segment_id": "...", "speaker_name": "Klaus Richter", "utterance_text": "...", ... },
    { "segment_id": "...", "speaker_name": "Unknown Attendee", "utterance_text": "...", ... }
  ]
}
```

## Sprint 1A · US03 — Pre-meeting consent email pipeline

The consent email pipeline persists the per-recipient "we're going to
record this meeting — click here to opt out" lifecycle for every
external customer attendee. Six endpoints; one new table.

### Pipeline state diagram

```
   Meeting detected (US01: tbl_schedule_meetings row exists)
                           │
                           ▼
   ┌──────────────────────────────────────────────────────────┐
   │ POST /consent-emails/schedule                            │
   │  → filters internal domains                              │
   │  → checks consent window:                                │
   │    - if open  → creates rows, returns recipients[] +     │
   │                 opt_out_url for each                     │
   │    - if past  → returns should_send=false +              │
   │                 fallback='in_meeting_chat'               │
   └────────────────────┬─────────────────────────────────────┘
                        │
                        ▼   bot renders email + sends via Graph
   ┌──────────────────────────────────────────────────────────┐
   │ PATCH /consent-emails/{consent_id}/delivery              │
   │  body: { status: 'sent' | 'failed', failure_reason }     │
   │  → on 'failed' + attempt<2: schedules retry at +10 min   │
   │  → on 'failed' + attempt>=2: status='fallback_to_in_meeting'│
   └──────────────────────────────────────────────────────────┘

       Some time later — participant clicks email link:

   ┌──────────────────────────────────────────────────────────┐
   │ GET /consent-emails/opt-out/{token}                      │
   │  → validates token, checks meeting_start > now           │
   │  → records opt-out (timestamp, ip)                       │
   │  → cascades bot_status='cancelled' on parent meeting     │
   │     with reason='participant_opted_out'                  │
   │  → renders SUCC_MSG_0010 HTML page (browser-facing)      │
   └──────────────────────────────────────────────────────────┘

       Bot, right before joining:

   ┌──────────────────────────────────────────────────────────┐
   │ GET /meetings/{meeting_id}/consent-status                │
   │  → returns any_opted_out, opt_out_count,                  │
   │    consent_mechanism, pending_retries                     │
   │  → AC #7: any_opted_out=true → bot does NOT join         │
   └──────────────────────────────────────────────────────────┘
```

### `POST /consent-emails/schedule`

Bot calls this when a meeting becomes eligible. Backend filters internal
Lenovo domains (`INTERNAL_EMAIL_DOMAINS`), checks if the consent window
is still open, and either creates per-recipient rows OR signals fallback.

```bash
curl -X POST http://localhost:8001/consent-emails/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "meeting_id": "11111111-1111-1111-1111-111111111111",
    "recipients": [
      { "email": "k.richter@db.com", "name": "Klaus Richter" },
      { "email": "rajesh.k@infosys.com", "name": "Rajesh Kumar" },
      { "email": "seller@lenovo.com", "name": "Maria Hofer" }
    ]
  }'
```

Response (window open):

```jsonc
{
  "meeting_id": "11111111-...",
  "should_send": true,
  "fallback": null,
  "recipients": [
    {
      "consent_id": "a1b2c3d4-...",
      "recipient_email": "k.richter@db.com",
      "recipient_name": "Klaus Richter",
      "opt_out_token": "9pVkM4xN_…43chars",
      "opt_out_url": "http://localhost:8001/consent-emails/opt-out/9pVkM4xN_…",
      "scheduled_send_at": "2026-06-15T14:00:00Z",
      "delivery_status": "pending"
    },
    { "...rajesh entry...": "..." }
  ],
  "filtered_internal_count": 1,                    // seller@lenovo.com filtered
  "seller_name": "Maria Hofer",
  "system_email_address": "sales-assistant@lenovo.com"
}
```

Response (window passed):

```jsonc
{
  "meeting_id": "11111111-...",
  "should_send": false,
  "fallback": "in_meeting_chat",
  "recipients": [],
  "filtered_internal_count": 0,
  "seller_name": "Maria Hofer",
  "system_email_address": "sales-assistant@lenovo.com"
}
```

### `PATCH /consent-emails/{consent_id}/delivery`

Bot reports the send result.

```bash
# Successful send
curl -X PATCH \
  http://localhost:8001/consent-emails/<consent_id>/delivery \
  -H "Content-Type: application/json" \
  -d '{ "status": "sent" }'

# Failed send — backend schedules retry at +10 min
curl -X PATCH \
  http://localhost:8001/consent-emails/<consent_id>/delivery \
  -H "Content-Type: application/json" \
  -d '{ "status": "failed", "failure_reason": "SMTP bounce: 550 mailbox unavailable" }'
```

After two failures the row flips to `delivery_status='fallback_to_in_meeting'`
and the bot must use US02's in-meeting chat announcement instead.

### `GET /consent-emails/opt-out/{token}` — public, browser-facing

Participant clicks the link in the email. Returns server-rendered
HTML (`SUCC_MSG_0010`). Idempotent — second click renders the same
page without re-stamping. Returns `410 Gone` (HTML) if the meeting
has already started, `404` (HTML) for an unknown token.

```bash
curl http://localhost:8001/consent-emails/opt-out/<token>
# → 200 OK, text/html, "Your preference has been saved."
```

### `GET /consent-emails/due-for-retry`

Bot polls this on whatever cadence it likes (recommend every 1–5 min)
to pick up failed sends ready for retry.

```bash
curl http://localhost:8001/consent-emails/due-for-retry
```

### `GET /consent-emails/{meeting_id}`

Audit / FE list view — every consent row for a meeting.

### `GET /meetings/{meeting_id}/consent-status`

Bot calls this **right before joining**. AC #7: if `any_opted_out` is
`true`, the bot does not join under any circumstances.

```bash
curl http://localhost:8001/meetings/<meeting_id>/consent-status
```

```jsonc
{
  "meeting_id": "11111111-...",
  "any_opted_out": true,
  "opt_out_count": 1,
  "total_recipients": 2,
  "consent_mechanism": "pre_meeting_email",
  "pending_retries": 0,
  "sent_count": 2,
  "failed_count": 0,
  "fallback_count": 0
}
```

## Sprint 1A · US04 — Data Hygiene, Validation & Intelligent Alerts

The data-hygiene subsystem maintains a per-seller To-Do queue of detected
data-quality issues across accounts, contacts, and opportunities.

### Sources of task creation

```
   ┌────────────────────────────────┐    ┌────────────────────────────────┐
   │ AI team's transcript-signal    │    │ Daily-scan CLI job              │
   │ NLP pipeline                   │    │ python -m app.jobs.scan_data_tasks │
   │ POST /api/data-tasks           │    │ (4 deterministic detectors)     │
   │ created_by_source='transcript' │    │ created_by_source='scan'        │
   └──────────────┬─────────────────┘    └──────────────┬─────────────────┘
                  │                                     │
                  └─────────────────┬───────────────────┘
                                    ▼
                ┌──────────────────────────────────────┐
                │ tbl_data_task                        │
                │  partial UNIQUE on                   │
                │  (entity_kind, entity_id, task_kind) │
                │  WHERE status IN ('open','dismissed')│
                │                                      │
                │  → idempotent re-detection           │
                │  → dismissal suppresses re-creation  │
                └──────────────┬───────────────────────┘
                               ▼
                  ┌─────────────────────────────┐
                  │ Seller's To-Do List         │
                  │ GET /api/data-tasks         │
                  │   ?ownerId=...&status=open  │
                  │ ordered by                  │
                  │   confidence DESC,          │
                  │   severity DESC,            │
                  │   created_at ASC            │
                  └────────────┬────────────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
              POST /resolve              POST /dismiss
              writes audit fields        writes note + suppression
```

### S1A detectors

| # | Detector | Severity | Trigger |
|---|---|---|---|
| D1 | `past_close_date` | high | `close_date < today` AND opp still active |
| D2 | `zero_or_missing_value` | medium | `estimated_value IS NULL OR == 0` |
| D3 | `stale_activity` | medium / high | `now − last_activity > DATA_TASK_STALE_DAYS` (high if 2× threshold) |
| D4 | `risk_flag` | maps from D365 risk category | One per opp from `GET /api/opportunities/{id}/risks` (most-severe wins) |

Detectors deferred to S1B: territory mismatch, multi-signal duplicate-opp,
parent-child account suggestion, email-domain mismatch, stage dwell-time
anomaly, fuzzy duplicate-contact, AI-suggested-update tasks. See
`US04_STATUS_AND_DEPENDENCIES.md` for the full list.

### `POST /api/data-tasks` — create / idempotent re-detection

Generic creation endpoint. AI-team transcript-signal pipeline, the daily
scan, and FE inline validators (S1B) all use this single endpoint. The
caller passes `created_by_source` so we can break out metrics later.

```bash
curl -X POST http://localhost:8001/api/data-tasks \
  -H 'content-type: application/json' \
  -d '{
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
    "evidence_text": "Customer mentioned We need it before June 15th on the call.",
    "created_by_source": "transcript"
  }'
```

Response (200):

```json
{
  "task": { "task_id": "...", "status": "open", ... },
  "was_existing": false
}
```

Re-posting the same `(entity_kind, entity_id, task_kind)` returns the
existing task with `was_existing: true` — same row, no duplicate. If
the existing task is **dismissed**, that's also returned (and stays
dismissed). This is how AC #5 ("dismissals suppress future alerts for
that pair") is implemented.

### `GET /api/data-tasks` — seller's To-Do list

Filters: `ownerId` (AC #11 portfolio scoping), `status`, `kind`,
`entityKind`, `entityId`. Ordered by confidence DESC NULLS LAST,
severity DESC, created_at ASC.

```bash
curl 'http://localhost:8001/api/data-tasks?ownerId=0b2cc1f8-...&status=open&limit=50'
```

### `GET /api/data-tasks/{task_id}` — task detail

```bash
curl http://localhost:8001/api/data-tasks/<task_id>
```

### `POST /api/data-tasks/{task_id}/resolve`

Marks the task resolved, records the actor, and stores the
`resolved_value` for audit.

```bash
curl -X POST http://localhost:8001/api/data-tasks/<task_id>/resolve \
  -H 'content-type: application/json' \
  -d '{"actor_id": "<seller_uuid>", "resolved_value": "2026-06-15"}'
```

Idempotent — a second resolve returns `was_already_resolved: true`
without overwriting the original audit trail.

### `POST /api/data-tasks/{task_id}/dismiss`

Dismisses with a **required** note (AC #4). Empty / whitespace-only
notes are rejected (422 / 409). The dismissal blocks future
re-detection of the same `(entity, kind)` pair (AC #5).

```bash
curl -X POST http://localhost:8001/api/data-tasks/<task_id>/dismiss \
  -H 'content-type: application/json' \
  -d '{
    "actor_id": "<seller_uuid>",
    "note": "Confirmed with customer — original date is correct."
  }'
```

### `POST /api/data-tasks/scan` — manual scan trigger

Runs the daily-scan algorithm synchronously. Same code path as the CLI
invocation; useful for ops / debugging.

```bash
# Full scan
curl -X POST http://localhost:8001/api/data-tasks/scan -d '{}'

# Limited dry-run
curl -X POST http://localhost:8001/api/data-tasks/scan \
  -H 'content-type: application/json' \
  -d '{"limit": 25, "dry_run": true}'
```

### Daily-scan CLI

```bash
python -m app.jobs.scan_data_tasks                   # full scan
python -m app.jobs.scan_data_tasks --limit 50        # only first 50 active opps
python -m app.jobs.scan_data_tasks --dry-run         # report-only, no writes
python -m app.jobs.scan_data_tasks --stale-days 45   # override the 30-day threshold
```

The scan prints a one-line JSON summary on stdout. Non-zero exit if any
opp errored — convenient for cron alerting.

```json
{"total_scanned": 312, "tasks_created": 47, "tasks_skipped_existing": 8, "tasks_skipped_dismissed": 3, "opportunities_with_errors": 0, "dry_run": false}
```

## Outbound: D365 Sales backend

`app/clients/d365_client.py` is a thin httpx wrapper around two D365
backend endpoints. Both follow the same error-handling contract:
benign 404 → `None` / NULL fields; everything else (5xx, 422, timeout,
malformed payload) → `D365ClientError`. The bot's pattern is "log and
continue" — D365 hiccups should never crash the bot's flow.

### `resolve_opportunity` (US01) — meeting → CRM deal

Called BEFORE the bot schedules itself, to tag the meeting with its
opportunity / account.

```python
from app.clients.d365_client import resolve_opportunity, D365ClientError

try:
    result = resolve_opportunity(
        attendee_emails=["k.richter@db.com", "rajesh.k@infosys.com"],
        subject="ThinkPad Fleet Review",
        organiser_email="seller@lenovo.com",
    )
except D365ClientError as exc:
    # 5xx, 422, timeout — log and continue (bot still joins, just untagged)
    log.warning("D365 unreachable: %s", exc)
    result = None

if result is None:
    # 404 from D365 = no matching active deal. Normal traffic.
    opportunity_id = account_id = None
else:
    opportunity_id = result.opportunity_id
    account_id     = result.account_id
```

### `resolve_contacts_by_email` (US02) — emails → CRM contact context

Called ONCE at meeting start (after the consent message + transcript
start succeed) so the bot can tag each transcript utterance with the
speaker's CRM name / job title / role / contact_id locally — no
per-utterance round-trip.

```python
from app.clients.d365_client import resolve_contacts_by_email, D365ClientError

attendees = ["k.richter@db.com", "rajesh.k@infosys.com", "stranger@nope.com"]

try:
    contacts = resolve_contacts_by_email(emails=attendees)
except D365ClientError as exc:
    log.warning("D365 contact-resolver unreachable: %s", exc)
    contacts = []  # fall back to "Unknown Attendee" for everyone

# Build a lookup the bot can hit per utterance.
by_email = {c.email: c for c in contacts}

speaker = by_email.get(utterance_email)
if speaker is None or speaker.contact_id is None:
    speaker_name, speaker_role, contact_id = "Unknown Attendee", None, None
else:
    speaker_name = speaker.name or "Unknown Attendee"
    speaker_role = speaker.role        # "Decision Maker" / "CTO" / etc.
    contact_id   = speaker.contact_id
```

### `list_active_opportunities` (US04) — paginated active-opp stream

Generator used by the daily-scan CLI to enumerate opps for the
deterministic detectors. Filters `Closed*` deals client-side.

```python
from app.clients.d365_client import list_active_opportunities

for opp in list_active_opportunities(page_size=100, max_records=500):
    # opp is an OpportunityScanRow dataclass with the columns the
    # detectors actually read (close_date, estimated_value, last_activity, ...)
    ...
```

### `fetch_opportunity_risks` (US04) — per-opp risk passthrough

Wraps `GET /api/opportunities/{id}/risks` so the data-hygiene scan can
materialise one `risk_flag` task per opp.

```python
from app.clients.d365_client import fetch_opportunity_risks

risks = fetch_opportunity_risks(opportunity_id)
# Each risk: category, name, message, optional risk_id + detected_at
```

The D365 endpoints are documented in
`Lenovo D365 Sales/API_CONTRACT.md` and live at:
- `Lenovo D365 Sales/app/routers/meetings_resolver.py` (US01)
- `Lenovo D365 Sales/app/routers/contacts_resolver.py` (US02)
- `Lenovo D365 Sales/app/routers/opportunities.py` — `GET /api/opportunities` (US04)
- `Lenovo D365 Sales/app/routers/deals_read.py` — `GET /api/opportunities/{id}/risks` (US04)

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

The test suite uses an in-memory SQLite DB plus FastAPI's `TestClient`,
so no live Postgres is needed.

## Migrations

For Sprint 1A we use idempotent SQL files (run by hand) plus
`Base.metadata.create_all()` on startup. **This does NOT drop or migrate
existing columns** — it only creates missing tables. For schema changes
on existing tables we hand-write the `ALTER TABLE IF NOT EXISTS` script
in `sql/`.

Tracked tech debt: switch to **Alembic** before Sprint 1B so the bot
team isn't blocked when we need destructive migrations (rename column,
drop column, change type). Until then, every PR that touches a model
must include a matching `sql/<date>_<topic>.sql`.

## Out of scope for Sprint 1A

| Concern | Why deferred |
|---|---|
| Authn / authz between bot and AIBackend | AIBackend is internal-only for now; add bearer-token check before public deployment |
| Audit log of every status transition | The current row stores only the most recent state; revisit if compliance asks |
| Outlook calendar webhook receiver | Bot/agent team's responsibility — they own the Graph subscription |
| Retry queue for D365 5xx | Bot just logs and continues; revisit if D365 outages become common |

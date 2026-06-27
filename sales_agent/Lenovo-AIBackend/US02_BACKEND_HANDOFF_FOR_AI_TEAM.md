# US02 — Consent & Recording · Backend Handoff for AI Team

**Sprint:** 1A
**User Story:** Consent & Recording
**Backend status:** All 6 endpoints implemented + tested + documented
**Backend repo (primary):** `Lenovo-AIBackend`
**Backend repo (cross-cut):** `Lenovo D365 Sales` — one new contact-resolver
**Backend contact:** Sanmay
**Document audience:** AI / Bot integration team (Note-Taking Agent owners)

---

## TL;DR — what backend gives you for US02

| # | Endpoint | Repo | Why you call it |
|---|---|---|---|
| 1 | `POST /transcripts/` | AIBackend | Open the transcript after you send `CONF_MSG_0004` in chat |
| 2 | `POST /transcripts/{meeting_id}/segments` | AIBackend | Append batched utterances during the meeting |
| 3 | `POST /transcripts/{meeting_id}/finalize` | AIBackend | Mark transcript finalised on a clean meeting end |
| 4 | `POST /transcripts/{meeting_id}/terminate` | AIBackend | Mark transcript partial on early stop (organiser kicked / all-left / bot crashed) |
| 5 | `GET  /transcripts/{meeting_id}` | AIBackend | (FE / debug) Read the full transcript |
| 6 | `POST /api/contacts/resolve-by-emails` | D365 Sales | Bulk-enrich attendee emails → CRM context (call once at meeting start) |

Two new tables in AIBackend's database:
- `tbl_meeting_transcript` — one row per meeting (lifecycle, consent, finalisation)
- `tbl_meeting_transcript_segment` — one row per utterance (speaker, text, timestamps, confidence)

---

## 1. The bot's flow with backend

```
                   Outlook event detected
                          │
                          ▼   (US01 — already done)
          ┌─────────────────────────────────────────┐
          │ POST /api/meetings/resolve-opportunity  │  D365 Sales
          │ → opportunity_id, account_id            │
          └────────────────┬────────────────────────┘
                           │
                           ▼
          ┌─────────────────────────────────────────┐
          │ POST /meeting-details/                  │  AIBackend
          │ + attach opportunity_id + account_id    │  (US01 — already done)
          └────────────────┬────────────────────────┘
                           │
                           ▼   meeting time arrives, bot dials in
          ┌─────────────────────────────────────────┐
          │ PATCH /meeting-details/{id}/status      │  AIBackend
          │ → bot_status='joined'                   │  (US01)
          └────────────────┬────────────────────────┘
                           │
              ────── US02 STARTS HERE ──────
                           │
                           ▼   bot sends CONF_MSG_0004 in Teams chat
          ┌─────────────────────────────────────────┐
          │ POST /transcripts/                      │  AIBackend  (1)
          │ + consent_message_text                  │
          │ + consent_sent_at                       │
          │ → transcript_id, status='in_progress'   │
          └────────────────┬────────────────────────┘
                           │
                           ▼   bot calls D365 ONCE to enrich attendees
          ┌─────────────────────────────────────────┐
          │ POST /api/contacts/resolve-by-emails    │  D365 Sales (6)
          │ → name / jobTitle / contactId / role    │
          │   per attendee (NULL fields if unknown) │
          └────────────────┬────────────────────────┘
                           │
                           ▼   meeting in progress…
          ┌─────────────────────────────────────────┐
          │ POST /transcripts/{id}/segments         │  AIBackend  (2)
          │ + array of segments (continuous calls)  │
          │ → appended_count, segment_count         │
          └────────────────┬────────────────────────┘
                           │
                  (one of two terminations)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌────────────────────┐    ┌─────────────────────┐
    │ Meeting ends       │    │ Organiser kicks bot │
    │ POST .../finalize  │ (3)│ POST .../terminate  │ (4)
    │ + overall_score    │    │ + reason            │
    │ → status='finalized'│   │ → 'terminated_partial'│
    └────────────────────┘    └─────────────────────┘
```

---

## 2. Acceptance criteria coverage

| AC | Backend coverage |
|---|---|
| 1. Consent before audio | `consent_message_text` + `consent_sent_at` are NOT NULL on `tbl_meeting_transcript`. `POST /transcripts/` rejects (422) if either is missing. The transcript literally cannot exist without compliance proof. |
| 2. Each segment captures all 7 fields | `speaker_name` (NOT NULL), `speaker_email`, `speaker_role`, `utterance_text` (NOT NULL), `start_time` (NOT NULL), `end_time` (NOT NULL), `confidence_score` (NOT NULL, CHECK 0..1). Plus we store `speaker_contact_id` for CRM linkage. |
| 3. Speakers identified or "Unknown Attendee" | We store whatever the bot sends. Use `POST /api/contacts/resolve-by-emails` (D365) once at meeting start to enrich; for unidentified speakers send `speaker_name="Unknown Attendee"` and leave email/role/contact_id NULL. |
| 4. Continuous segment saves | `POST /transcripts/{id}/segments` is append-friendly — call it as often as you want with batch sizes of your choice. We don't enforce a maximum batch size. |
| 5. Only organiser can remove the agent | Backend records the eventual termination via `POST .../terminate`. The permission check itself (organiser-only) is the bot's job — Teams Bot SDK gives you the role of the user requesting the kick. |
| 6. Finalised transcript linked to D365 opportunity | `tbl_meeting_transcript.opportunity_id` is auto-populated from the parent meeting at start time. `POST .../finalize` sets `status='finalized'` + `overall_confidence_score`. |

---

## 3. Endpoint reference

### 3.1 `POST /transcripts/`

**Repo:** `Lenovo-AIBackend`
**When:** Immediately after you send `CONF_MSG_0004` in the Teams chat.
**Pre-condition:** `bot_status` for this meeting must be one of
`{joined, lobby_waiting, joining}` (i.e. you've already PATCHed the
status via US01).

```http
POST /transcripts/
Content-Type: application/json

{
  "meeting_id":           "11111111-1111-1111-1111-111111111111",
  "consent_message_text": "Hi everyone — this meeting is being recorded by the Lenovo Note-Taking Agent for transcript and follow-up purposes. (CONF_MSG_0004)",
  "consent_sent_at":      "2026-06-15T15:00:30+00:00",
  "started_at":           "2026-06-15T15:00:32+00:00"   // optional; defaults to consent_sent_at
}
```

**Success:** `201 Created` with the transcript metadata (incl. `transcript_id`).
**Errors:**
- `400` — `bot_status` not eligible (e.g. you forgot to PATCH it to `joined` first)
- `404` — meeting not found
- `409` — a transcript already exists for this meeting (use the existing one or terminate it first)
- `422` — missing consent fields, or `started_at < consent_sent_at`

---

### 3.2 `POST /transcripts/{meeting_id}/segments`

**Repo:** `Lenovo-AIBackend`
**When:** Continuously during the meeting. Batch as you see fit — 1 segment per call is fine, 100 is fine. We don't cap.
**Idempotency:** None in v1. If the network drops and you re-POST a batch, you'll get duplicates. Bot is responsible for at-most-once delivery. Tell us if duplicates show up in dev — we can add `(meeting_id, start_time, speaker_email)` dedup if it becomes a real problem.

```http
POST /transcripts/{meeting_id}/segments
Content-Type: application/json

{
  "segments": [
    {
      "speaker_name":       "Klaus Richter",
      "speaker_email":      "k.richter@db.com",
      "speaker_role":       "CTO",
      "speaker_contact_id": "e7cf7aaf-cd45-4a9a-9c6b-39ba8d1b6c1a",
      "utterance_text":     "Thanks for joining. Let's start with the migration plan.",
      "start_time":         "2026-06-15T15:01:00+00:00",
      "end_time":           "2026-06-15T15:01:04+00:00",
      "confidence_score":   0.93
    },
    {
      "speaker_name":       "Unknown Attendee",
      "utterance_text":     "Sorry I am late.",
      "start_time":         "2026-06-15T15:01:30+00:00",
      "end_time":           "2026-06-15T15:01:32+00:00",
      "confidence_score":   0.71
    }
  ]
}
```

**Success:** `200 OK`

```jsonc
{
  "transcript_id":  "a1b2c3d4-...",
  "meeting_id":     "11111111-...",
  "appended_count": 2,
  "segment_count":  47    // post-append running total — useful for your logging
}
```

**Errors:**
- `400` — transcript not in `in_progress` (you've already finalised / terminated)
- `404` — no transcript exists for this meeting (you forgot to call `POST /transcripts/`)
- `422` — `confidence_score` not in `[0, 1]`, `end_time < start_time`, empty `segments`, missing required fields

---

### 3.3 `POST /transcripts/{meeting_id}/finalize`

**Repo:** `Lenovo-AIBackend`
**When:** Meeting ends naturally (everyone left, scheduled end time reached).
**Effect:** `status = 'finalized'` + `terminated_reason = 'meeting_ended'`.

You compute the overall confidence (typically a weighted average of segment confidences); we record what you send.

```http
POST /transcripts/{meeting_id}/finalize
Content-Type: application/json

{
  "overall_confidence_score": 0.91,
  "finalized_at":             "2026-06-15T16:00:12+00:00"   // optional; defaults to server now()
}
```

---

### 3.4 `POST /transcripts/{meeting_id}/terminate`

**Repo:** `Lenovo-AIBackend`
**When:** Bot leaves before natural end — organiser kicked, all human participants left, or bot crashed.
**Effect:** `status = 'terminated_partial'` + `terminated_reason = <your reason>`.

```http
POST /transcripts/{meeting_id}/terminate
Content-Type: application/json

{
  "reason":                   "organizer_removed",   // | "all_left" | "bot_failure"
  "terminated_at":            "2026-06-15T15:45:02+00:00",   // optional
  "overall_confidence_score": 0.88                            // optional, informational only
}
```

> Use `/finalize` for clean ends — `/terminate` deliberately rejects
> `meeting_ended` so the audit trail stays unambiguous.

---

### 3.5 `GET /transcripts/{meeting_id}`

**Repo:** `Lenovo-AIBackend`
**Audience:** FE Activity tab (eventually) + ad-hoc debugging.

Returns the transcript metadata + every segment ordered by `start_time`. No pagination in v1.

```http
GET /transcripts/{meeting_id}
```

```jsonc
{
  "transcript": {
    "transcript_id":            "a1b2c3d4-...",
    "meeting_id":               "11111111-...",
    "opportunity_id":           "b0000001-...",
    "account_id":               "6dc95c38-...",
    "status":                   "finalized",
    "consent_message_text":     "...",
    "consent_sent_at":          "2026-06-15T15:00:30Z",
    "overall_confidence_score": 0.91,
    "segment_count":            382,
    "terminated_reason":        "meeting_ended",
    "started_at":               "2026-06-15T15:00:30Z",
    "finalized_at":             "2026-06-15T16:00:12Z"
  },
  "segments": [
    { "segment_id": "...", "speaker_name": "Klaus Richter", "utterance_text": "...", "start_time": "...", "end_time": "...", "confidence_score": 0.93 },
    { "segment_id": "...", "speaker_name": "Unknown Attendee", ... },
    ...
  ]
}
```

---

### 3.6 `POST /api/contacts/resolve-by-emails` (D365)

**Repo:** `Lenovo D365 Sales`
**When:** ONCE at meeting start (right after `POST /transcripts/` succeeds).
**Why:** So you can tag each utterance with CRM context locally — no per-utterance round-trip.

```http
POST /api/contacts/resolve-by-emails
Content-Type: application/json

{
  "emails": [
    "k.richter@db.com",
    "rajesh.k@infosys.com",
    "seller@lenovo.com",
    "unknown@nope.com"
  ]
}
```

**Success:** `200 OK`

```jsonc
{
  "results": [
    {
      "email":       "k.richter@db.com",
      "contactId":   "E7CF7AAF-CD45-4A9A-9C6B-39BA8D1B6C1A",
      "name":        "Klaus Richter",
      "jobTitle":    "CTO",
      "accountId":   "6dc95c38-9237-4ce9-84d3-f5d1f7431965",
      "accountName": "Deutsche Bank AG",
      "role":        "Decision Maker"
    },
    {
      "email":       "rajesh.k@infosys.com",
      "contactId":   "...",
      "name":        "Rajesh Kumar",
      "jobTitle":    "Engagement Manager",
      "accountId":   "...",
      "accountName": "Infosys",
      "role":        null   // contact exists but isn't on any active opportunity
    },
    {
      "email":     "unknown@nope.com",
      "contactId": null,
      "name":      null,
      // …all other fields null
    }
  ]
}
```

**Note:** D365 backend uses **camelCase** keys (matches the existing
`/api/meetings/resolve-opportunity` endpoint convention). AIBackend uses
snake_case.

**Errors:**
- `422` — `emails` is empty or all-blank

---

## 4. Helpful client wrapper

Already in `app/clients/d365_client.py`:

```python
from app.clients.d365_client import (
    resolve_contacts_by_email,
    D365ClientError,
)

attendees = ["k.richter@db.com", "rajesh.k@infosys.com", "stranger@nope.com"]

try:
    contacts = resolve_contacts_by_email(emails=attendees)
except D365ClientError as exc:
    log.warning("D365 contact-resolver unreachable: %s", exc)
    contacts = []   # graceful fallback: every speaker becomes "Unknown Attendee"

# Build the per-utterance lookup the bot uses while transcribing
by_email = {c.email: c for c in contacts}

# When a new utterance arrives:
speaker = by_email.get(utterance_email_lowercase)
if speaker is None or speaker.contact_id is None:
    name, role, contact_id = "Unknown Attendee", None, None
else:
    name       = speaker.name or "Unknown Attendee"
    role       = speaker.role
    contact_id = speaker.contact_id
```

---

## 5. Schema reference

### `tbl_meeting_transcript`

| Column | Type | Notes |
|---|---|---|
| `transcript_id` | UUID PK | |
| `meeting_id` | UUID UNIQUE NOT NULL | FK-by-convention to `tbl_schedule_meetings` |
| `opportunity_id` | UUID | Denormalised from meeting |
| `account_id` | UUID | Denormalised from meeting |
| `status` | TEXT NOT NULL CHECK | `in_progress` / `finalized` / `terminated_partial` |
| `consent_message_text` | TEXT NOT NULL | The actual message you sent in chat |
| `consent_sent_at` | TIMESTAMPTZ NOT NULL | When you sent it |
| `overall_confidence_score` | NUMERIC(4,3) | NULL until finalised; 0.000–1.000 |
| `segment_count` | INT NOT NULL | Maintained by `/segments` |
| `terminated_reason` | TEXT CHECK | `meeting_ended` / `organizer_removed` / `all_left` / `bot_failure` |
| `started_at` | TIMESTAMPTZ NOT NULL | |
| `finalized_at` | TIMESTAMPTZ | NULL until finalised / terminated |
| `created_at`, `updated_at` | TIMESTAMPTZ | Server-managed |

### `tbl_meeting_transcript_segment`

| Column | Type | Notes |
|---|---|---|
| `segment_id` | UUID PK | |
| `transcript_id` | UUID NOT NULL | |
| `meeting_id` | UUID NOT NULL | Denormalised — primary read path |
| `speaker_name` | TEXT NOT NULL | "Klaus Richter" / "Unknown Attendee" |
| `speaker_email` | TEXT | NULL for unknown speakers |
| `speaker_role` | TEXT | "CTO" / "Decision Maker" / etc. |
| `speaker_contact_id` | UUID | NULL if unidentified |
| `utterance_text` | TEXT NOT NULL | |
| `start_time` | TIMESTAMPTZ NOT NULL | Absolute (not meeting-relative) |
| `end_time` | TIMESTAMPTZ NOT NULL | CHECK `>= start_time` |
| `confidence_score` | NUMERIC(4,3) NOT NULL | CHECK 0..1 |
| `created_at` | TIMESTAMPTZ | Server-managed |

---

## 6. Out-of-scope (your side or future sprints)

| Item | Owner | Notes |
|---|---|---|
| Sending `CONF_MSG_0004` to Teams chat | AI team | Teams Bot SDK / Graph |
| Capturing audio | AI team | Teams Bot SDK |
| Speech-to-text + diarization | AI team | Azure Speech / Whisper / etc. |
| Detecting "remove me" requests + organiser-only check | AI team | Teams Bot SDK + meeting roster |
| Sending the private "you can't kick me" notification | AI team | Teams Bot SDK |
| Detecting meeting end / all-left | AI team | Teams Bot SDK events |
| Computing `overall_confidence_score` | AI team | Backend just records what you send |
| Storing audio file | Out-of-scope | We persist transcripts only; if you need a pointer to your audio blob we can add `audio_blob_url` later |
| Real-time push to FE (WebSocket / SSE) | Future sprint | FE polls on Activity tab load for now |
| Post-meeting AI summary (key points / action items / sentiment) | US03 territory | Separate story; design notes at `Lenovo D365 Sales/MEETING_INTELLIGENCE_STORAGE_BACKEND_NOTES.md` |
| Sweeper to mark stale `in_progress` transcripts (bot crashed) | Sprint 1B | Not blocking US02 |
| Authn between bot ↔ AIBackend | Sprint 1B | Internal-only for now |

---

## 7. Open questions for AI team / PM

| # | Question | Why it matters |
|---|---|---|
| Q1 | Where does the canonical text of `CONF_MSG_0004` live? Are we BOTH referencing the same string, or does each repo hard-code its own copy? | Single source of truth for compliance audit |
| Q2 | What's your preferred batch size for segment POSTs? Per-utterance (~1/sec)? Every N seconds? Every N segments? | Decides backend load profile |
| Q3 | What do you do on network failure mid-meeting — buffer + retry, or drop? | If buffer + retry, we'll add segment dedup in v1.1 |
| Q4 | Should `bot_status='cancelled'` (US01) and `terminated_partial` (US02) be linked? E.g. when you hit `/terminate`, should we auto-flip `bot_status` to `cancelled`? | Decides whether the two endpoints should cascade |
| Q5 | Who owns `overall_confidence_score`? You compute it (current assumption), or do you want us to derive it from segments? | If we derive, you don't need to send it |
| Q6 | Is "Unknown Attendee" rendered as a single bucket per meeting, or per voice (Unknown Attendee 1, 2, 3)? | Affects how you tag `speaker_name` |
| Q7 | Do you need an `audio_blob_url` field on the transcript table now (pointer to where YOU stored the audio blob)? Or defer to later? | Add now if yes — cheap |

---

## 8. How to test this end-to-end on dev

```bash
# 1. Apply both migrations (US01 + US02) to dev Postgres
psql -h $DB_HOST -U $DB_USER -d $DB_NAME \
  -f sql/2026_06_us01_meeting_lifecycle.sql
psql -h $DB_HOST -U $DB_USER -d $DB_NAME \
  -f sql/2026_06_us02_meeting_transcript.sql

# 2. Run the AIBackend
uvicorn app.main:app --reload --port 8001

# 3. Run the D365 Sales backend
uvicorn app.main:app --reload --port 8000   # in the Lenovo D365 Sales repo

# 4. Walk through the bot's flow with cURL — see this doc's §3 for each call
# 5. Or run the smoke tests
pytest -q tests/test_transcript_lifecycle.py
```

---

## 9. Contacts

- **Backend:** Sanmay (this repo + the D365 Sales contact resolver)
- **Frontend integration:** Namisha — when the FE Activity tab is ready to wire up, point it at `GET /transcripts/{meeting_id}`
- **AI/Bot team:** [your team contact]

For backend questions, ping me on the sprint Slack channel or open an
issue against this repo.

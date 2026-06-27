# Sprint 2 · Pre-Meeting Briefing Card — Backend Handoff

**Repos:** `Lenovo D365 Sales` (v0.18.0) + `Lenovo-AIBackend`  
**User story:** Pre-Meeting Briefing Card (Execute → Meeting Prep)  
**Backend contact:** Sanmay  
**Pairing:** Sanmay (backend) ↔ Namisha (FE) ↔ AI team  
**Last updated:** 2026-06-19

**Linked docs:**
- [API_CONTRACT.md](./API_CONTRACT.md) — §19 (D365 briefing context)
- [sql migrations — AIBackend](../Lenovo-AIBackend/sql/2026_09_us06_meeting_briefing.sql)
- **Postman:** [US_PRE_MEETING_BRIEFING.postman_collection.json](../Lenovo-AIBackend/postman/US_PRE_MEETING_BRIEFING.postman_collection.json) (also in `Lenovo D365 Sales/postman/`)

---

## Postman collection

**Import:** `Lenovo-AIBackend/postman/US_PRE_MEETING_BRIEFING.postman_collection.json`

| Collection variable | Set before run | Example |
|-------------------|----------------|---------|
| `d365BaseUrl` | D365 Sales host | `http://localhost:8000` |
| `aiBaseUrl` | AIBackend host | `http://localhost:8001` |
| `sellerId` | Seller UUID (`opportunity.owninguser`) | `055DAFE7-9840-451D-8328-5F70A6326C03` |
| `opportunityId` | Open deal UUID for that seller | from D365 DB |
| `meetingId` | Auto-set by list step, or after create | |
| `prepTaskId` | Auto-set from briefing response | |
| `prepNoteId` | Auto-set from create note response | |

**Run folders in order:** `00` → `07`. Use Collection Runner for full smoke.

**Prereqs:**
1. `uvicorn app.main:app --reload --port 8000` (D365)
2. `uvicorn app.main:app --reload --port 8001` (AIBackend)
3. `psql -f sql/2026_09_us06_meeting_briefing.sql` on AIBackend Postgres
4. Meeting row must have `seller_id` + `opportunity_id` (UUID) for briefing to succeed

---

## Status matrix — Done vs Pending

### Backend (this repo + AIBackend)

| Area | Status | Notes |
|------|--------|-------|
| D365 `GET /api/briefing/context` | ✅ Done | Facts, signals, competitor intel, source refs |
| AIBackend meeting list + `prep_tasks_pending_count` | ✅ Done | Requires `sellerId` |
| AIBackend `GET .../briefing` (auto-generate + cache) | ✅ Done | Template paragraphs; AI can replace |
| Prep tasks generate + PATCH done | ✅ Done | Priority order; strikethrough = FE |
| My Prep Notes CRUD + voice transcript endpoint | ✅ Done | STT not included |
| Seller scoping (`sellerId`) | ✅ Done | 403 when mismatch |
| `INF_MSG_0004` / `ERR_MSG_0023` message codes | ✅ Done | In API responses |
| Unit tests | ✅ Done | `test_briefing_context.py`, `test_briefing_service.py` |
| Postman collection | ✅ Done | This handoff § Postman |
| LLM paragraph generation | ⏳ Pending | **AI team** — hook in `briefing_service.py` |
| Microsoft Graph attendee-change signals | ⏳ Pending | Enterprise Event Spine / Graph |
| Audio upload + server-side STT | ⏳ Pending | **AI/FE** — endpoint accepts transcript text only |
| `opportunity_id` on POST create meeting | ⏳ Partial | May need DB update or scheduler sync |
| Real-time WebSocket badge refresh | ⏳ Pending | **FE** polls or refreshes on navigation |

### Frontend team

| UI / behaviour | Status | API to call |
|----------------|--------|-------------|
| Meeting Prep list screen | ⏳ Pending | `GET /ai-api/meeting-prep/meetings?sellerId=&filter=` |
| Meeting card — pending task count | ⏳ Pending | `prep_tasks_pending_count` from list response |
| Open briefing card | ⏳ Pending | `GET /ai-api/meeting-prep/{meetingId}/briefing?sellerId=` |
| Header — title, time, duration, platform | ⏳ Pending | `response.header` |
| Header — attendees names + roles | ⏳ Pending | `response.header.attendees[]` |
| Join Meeting button (Teams only) | ⏳ Pending | Show when `header.join_url` present |
| AI-generated label + `generated_at` | ⏳ Pending | `is_ai_generated`, `generated_at` |
| Collapsible sections (7 + notes) | ⏳ Pending | Map JSON sections to accordion UI |
| Account Summary paragraph + gaps + unverified | ⏳ Pending | `account_summary.paragraph`, `.gaps`, `.unverified_labels` |
| Deal Summary + stage from D365 | ⏳ Pending | `deal_summary.stage` — display verbatim |
| Competitor Intel block | ⏳ Pending | `deal_summary.competitor_intel.items` OR show `message_code` `INF_MSG_0004` |
| Recent Signals list + timestamps | ⏳ Pending | `recent_signals[]` — sort already chronological |
| Source reference links per AI item | ⏳ Pending | Each item `.source` (`sourceType`, `sourceId`, `label`) |
| Prep tasks — priority order, checkbox | ⏳ Pending | `prep_tasks[]`; PATCH on check |
| Completed task strikethrough | ⏳ Pending | `prep_tasks[].done === true` |
| Talking Points list | ⏳ Pending | `talking_points[]` OR `talking_points_message_code` `ERR_MSG_0023` |
| Watch Out For — hide when null | ⏳ Pending | Only render if `watch_out_for` is non-null array |
| My Prep Notes — typed + voice list | ⏳ Pending | `my_prep_notes[]` in briefing + CRUD endpoints |
| Edit / delete seller notes | ⏳ Pending | PATCH/DELETE prep-notes |
| Voice record UI | ⏳ Pending | Record → STT (AI) → POST `.../prep-notes/voice` |
| Refresh briefing (optional) | ⏳ Pending | `?refresh=true` on GET briefing |
| Error banners `ERR_MSG_0025` | ⏳ Pending | 502/500 on briefing generate |

### AI team

| Responsibility | Status | API / code touchpoint |
|----------------|--------|------------------------|
| Grounded Account/Deal paragraph LLM | ⏳ Pending | Input: `GET /api/briefing/context` JSON only. Hook: `app/services/briefing_service.py` → `generate_briefing()` after `fetch_briefing_context()` |
| Grounded Talking Points LLM | ⏳ Pending | Same facts JSON; must emit `source` per point; never invent figures |
| Voice STT pipeline | ⏳ Pending | Browser or server STT → `POST /ai-api/meeting-prep/{id}/prep-notes/voice` body `{ "transcript": "..." }` |
| Post-meeting summary — consume prep notes | ⏳ Pending | Read `tbl_meeting_prep_note` by `meeting_id` (or GET prep-notes API) before generating post-meeting summary |
| Meeting scheduler — set `seller_id` + `opportunity_id` on meetings | ⏳ Pending | When upserting `tbl_schedule_meetings` from Graph/calendar |
| Graph signals (new attendee on invite) | ⏳ Pending | Future: ingest to signals feed; today D365 activity/audit only |
| Do **not** call briefing card API for facts | ✅ Guidance | Use D365 `/api/briefing/context` for raw facts; AIBackend `/briefing` for assembled card |

---

## All APIs created (this user story)

**11 endpoints** across two services. AIBackend routes are prefixed with `/ai-api` (see `AIBACKEND_API_PREFIX`). D365 routes use `/api`.

### Summary

| # | Service | Method | Endpoint | Primary caller | Purpose |
|---|---------|--------|----------|----------------|---------|
| 1 | D365 Sales | GET | `/api/briefing/context` | AIBackend (internal), AI (debug) | Aggregate CRM facts + signals for briefing generation |
| 2 | AIBackend | GET | `/ai-api/meeting-prep/meetings` | FE | Meeting Prep list with pending task badge count |
| 3 | AIBackend | GET | `/ai-api/meeting-prep/{meetingId}/briefing` | FE | Full pre-meeting briefing card (generate + cache) |
| 4 | AIBackend | PATCH | `/ai-api/meeting-prep/{meetingId}/prep-tasks/{taskId}` | FE | Mark a prep task done / undone |
| 5 | AIBackend | GET | `/ai-api/meeting-prep/{meetingId}/prep-notes` | FE, AI | List seller prep notes for a meeting |
| 6 | AIBackend | POST | `/ai-api/meeting-prep/{meetingId}/prep-notes` | FE | Add a typed prep note |
| 7 | AIBackend | POST | `/ai-api/meeting-prep/{meetingId}/prep-notes/voice` | FE, AI | Add a voice note from STT transcript |
| 8 | AIBackend | PATCH | `/ai-api/meeting-prep/{meetingId}/prep-notes/{noteId}` | FE | Edit a prep note body |
| 9 | AIBackend | DELETE | `/ai-api/meeting-prep/{meetingId}/prep-notes/{noteId}` | FE | Delete a prep note |
| 10 | AIBackend | POST | `/ai-api/meeting-prep` | Scheduler / Postman setup | Create a scheduled meeting row |

### Primary FE flow

```
┌─────────────────────────────────────────────────────────────┐
│  Meeting Prep list                                          │
│  GET /ai-api/meeting-prep/meetings?sellerId=&filter=today   │
└───────────────────────────┬─────────────────────────────────┘
                            │ click card
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Briefing card (auto-generates first time)                   │
│  GET /ai-api/meeting-prep/{meetingId}/briefing?sellerId=    │
└─────────────────────────────────────────────────────────────┘
```

---

### 1 · D365 Sales — `GET /api/briefing/context`

**Host:** D365 Sales (`http://localhost:8000`)  
**File:** `app/routers/briefing.py` → `app/services/briefing_context.py`  
**Called by:** AIBackend `d365_client.fetch_briefing_context()` during briefing generation; AI team for debugging/LLM grounding.

**What it does:** Returns seller-scoped, traceable CRM facts for one opportunity — account fields, deal fields, competitor intel, and recent signals. Does **not** return the assembled briefing card UI payload. Each fact/signal includes a `source` reference for auditability.

| Query param | Required | Description |
|-------------|----------|-------------|
| `sellerId` | yes | Seller UUID (`opportunity.owninguser`) |
| `opportunityId` | yes | Open deal UUID |
| `accountId` | no | Override account; defaults from opportunity |
| `maxSummaryWords` | no | Paragraph word cap (50–200, default `100`) |

**200 response (top-level keys):**

| Key | Description |
|-----|-------------|
| `sellerId`, `opportunityId`, `accountId` | Scope identifiers |
| `generatedAt` | Server timestamp |
| `account` | Account name, structured `fields[]`, template `paragraph`, `gaps[]`, `unverifiedLabels[]` |
| `deal` | Opportunity name, `stage` (verbatim from D365), `fields[]`, `paragraph`, `competitorIntel[]` **or** `competitorMessageCode: INF_MSG_0004` |
| `signals[]` | Recent activity/audit signals: `summary`, `whyShown`, `eventAt`, `involvedParties`, `source` |

**Errors:** `422` missing params · `403` seller does not own opportunity · `404` opportunity not found · `500` `ERR_MSG_0024`

---

### 2 · AIBackend — `GET /ai-api/meeting-prep/meetings`

**Host:** AIBackend (`http://localhost:8001`)  
**File:** `app/api/meeting_prep.py` → `app/services/meeting_prep.py`

**What it does:** Powers the **Meeting Prep list** screen. Returns upcoming meetings for a seller, filtered by time window. Each row includes `prep_tasks_pending_count` for the task-pending badge on the meeting card.

| Query param | Required | Description |
|-------------|----------|-------------|
| `sellerId` | recommended | Filter to seller's meetings; must be valid UUID |
| `filter` | yes | `all_meetings` \| `today` \| `tomorrow` \| `this_week` |

**200 response:** Array of meeting objects:

| Field | Description |
|-------|-------------|
| `meeting_id` | UUID string |
| `meeting_start_time`, `meeting_end_time` | ISO datetimes |
| `platform`, `title`, `account_name` | Display fields |
| `organiser_name` | Organizer display name |
| `attendees_emails`, `attendee_count` | Comma-separated emails + count |
| `opportunity_id`, `account_id` | Linked CRM IDs (needed for briefing) |
| `meeting_url` | Teams join link when present |
| `prep_tasks_pending_count` | Open prep tasks for badge |

**Errors:** `400` invalid `filter` or bad `sellerId` UUID

---

### 3 · AIBackend — `GET /ai-api/meeting-prep/{meetingId}/briefing`

**Host:** AIBackend  
**File:** `app/api/meeting_prep.py` → `app/services/briefing_service.py`

**What it does:** Returns the **full pre-meeting briefing card** JSON. On first call for a meeting, calls D365 `/api/briefing/context`, generates prep tasks / talking points / watch-outs (template-based today; AI team replaces with LLM), persists to `tbl_meeting_briefing`, and caches for **4 hours** (`BRIEFING_CACHE_TTL_HOURS`). Subsequent calls return cached payload unless `refresh=true`.

| Path / query | Required | Description |
|--------------|----------|-------------|
| `meetingId` | yes | Meeting UUID from list or create |
| `sellerId` | yes | Seller UUID — ownership check |
| `refresh` | no | `true` forces regeneration |

**200 response (`MeetingBriefingResponse`):**

| Section | Key | Description |
|---------|-----|-------------|
| Meta | `meetingId`, `sellerId`, `generatedAt`, `isAiGenerated` | Card metadata |
| Header | `header` | `title`, `startAt`, `endAt`, `durationMinutes`, `platform`, `joinUrl`, `attendees[]` (name/role/email from D365 contact lookup) |
| §1 | `accountSummary` | AI paragraph, `gaps[]`, `unverifiedLabels[]`, `sources[]` |
| §2 | `dealSummary` | Paragraph, `stage`, `competitorIntel` (`items` or `messageCode INF_MSG_0004`) |
| §3 | `recentSignals[]` | Chronological signals with `source` |
| §4 | `prepTasks[]` | `id`, `description`, `priority` (HIGH/MEDIUM/LOW), `evidence`, `done` |
| §5 | `talkingPoints[]` | Grounded points with `whyShown` + `source`; or empty + `talkingPointsMessageCode ERR_MSG_0023` |
| §6 | `watchOutFor` | Risk considerations; **`null` = hide section in FE** |
| §7 | `myPrepNotes[]` | Seller notes (`typed` or `voice_transcript`) |

**Errors:** `422` missing `sellerId` · `403` seller mismatch · `404` meeting not found · `500` `ERR_MSG_0025` (generation failure)

**Prerequisite:** Meeting row must have `seller_id` and `opportunity_id` (UUID) set, or briefing cannot link to D365 facts.

---

### 4 · AIBackend — `PATCH /ai-api/meeting-prep/{meetingId}/prep-tasks/{taskId}`

**What it does:** Updates prep task completion when the seller checks/unchecks a checkbox on the briefing card. FE should strikethrough when `done: true`. Re-fetch list to update `prep_tasks_pending_count`.

| Query | Body |
|-------|------|
| `sellerId` (required) | `{ "done": true }` or `{ "done": false }` |

**200:** Updated task object (`id`, `description`, `priority`, `done`, etc.)

**Errors:** `403` / `404` as above

---

### 5 · AIBackend — `GET /ai-api/meeting-prep/{meetingId}/prep-notes`

**What it does:** Returns all seller prep notes for a meeting (standalone list; also embedded in briefing response as `myPrepNotes`).

**200:** Array of `{ id, noteType, body, createdAt, updatedAt, isSellerAdded }`

---

### 6 · AIBackend — `POST /ai-api/meeting-prep/{meetingId}/prep-notes`

**What it does:** Creates a **typed** prep note the seller adds before the meeting.

| Query | Body |
|-------|------|
| `sellerId` (required) | `{ "body": "...", "noteType": "typed" }` |

**200:** Created note object

---

### 7 · AIBackend — `POST /ai-api/meeting-prep/{meetingId}/prep-notes/voice`

**What it does:** Creates a prep note from a **voice transcript**. Does not perform STT — FE/AI records audio, runs speech-to-text, then posts the transcript text here.

| Query | Body |
|-------|------|
| `sellerId` (required) | `{ "transcript": "Customer wants phased delivery in July." }` |

**200:** Created note with `noteType: voice_transcript`

---

### 8 · AIBackend — `PATCH /ai-api/meeting-prep/{meetingId}/prep-notes/{noteId}`

**What it does:** Edits an existing prep note body (seller correction after typing).

| Query | Body |
|-------|------|
| `sellerId` (required) | `{ "body": "updated text" }` |

**200:** Updated note object

---

### 9 · AIBackend — `DELETE /ai-api/meeting-prep/{meetingId}/prep-notes/{noteId}`

**What it does:** Permanently removes a seller prep note.

| Query | Response |
|-------|----------|
| `sellerId` (required) | `204 No Content` |

---

### 10 · AIBackend — `POST /ai-api/meeting-prep`

**What it does:** Creates a new row in `tbl_schedule_meetings` and triggers a Teams calendar invite via Graph. Used by meeting scheduler integration or Postman test setup — **not** the main FE briefing flow.

**Request body (`CreateMeetingRequest`):**

| Field | Required | Description |
|-------|----------|-------------|
| `meeting_start_time` | yes | ISO datetime |
| `meeting_end_time` | no | Defaults to start + 30 min |
| `platform` | yes | e.g. `Teams` |
| `title` | yes | Meeting subject |
| `account_name` | yes | Display name |
| `opportunity` | yes | Opportunity name string |
| `meeting_url` | yes | Join URL |
| `attendees_emails` | no | Comma/semicolon-separated emails |
| `prep_notes` | no | Free-text prep notes on create |
| `seller_id` | no | Seller UUID — **set this for briefing to work** |

**200:** `{ "success": true, "message": "Meeting created successfully" }`

**Note:** `opportunity_id` (UUID column) is not set from this payload today — may need scheduler/DB update for briefing generation.

---

### Who calls what

| Team | APIs to integrate |
|------|-------------------|
| **FE** | #2 list · #3 briefing card · #4 prep tasks · #5–9 prep notes |
| **AI** | #1 D365 facts (grounding) · hook inside #3 generation · #7 voice transcript · #5 notes for post-meeting summary |
| **AIBackend (internal)** | #1 called automatically when #3 runs |
| **Scheduler / DevOps** | #10 create meeting · DB migration `2026_09_us06_meeting_briefing.sql` |

---

## Briefing response sections (FE mapping)

| Order | Section key | Collapsible | Empty behaviour |
|-------|-------------|-------------|-----------------|
| — | `header` | No | Always present |
| 1 | `account_summary` | Yes | Show paragraph + gap chips |
| 2 | `deal_summary` | Yes | Include `competitor_intel` sub-block |
| 3 | `recent_signals` | Yes | Empty array → "No recent signals" |
| 4 | `prep_tasks` | Yes | Empty array allowed |
| 5 | `talking_points` | Yes | If empty, show `talking_points_message_code` |
| 6 | `watch_out_for` | Yes | **`null` → hide entire section** |
| 7 | `my_prep_notes` | Yes | Seller notes only |

---

## curl examples

```bash
# List
curl "http://localhost:8001/ai-api/meeting-prep/meetings?sellerId=<SELLER>&filter=today"

# Briefing card
curl "http://localhost:8001/ai-api/meeting-prep/<MEETING_ID>/briefing?sellerId=<SELLER>"

# D365 facts (AI debug)
curl "http://localhost:8000/api/briefing/context?sellerId=<SELLER>&opportunityId=<OPP>"

# Mark task done
curl -X PATCH "http://localhost:8001/ai-api/meeting-prep/<MEETING_ID>/prep-tasks/1?sellerId=<SELLER>" \
  -H "Content-Type: application/json" -d '{"done":true}'

# Typed note
curl -X POST "http://localhost:8001/ai-api/meeting-prep/<MEETING_ID>/prep-notes?sellerId=<SELLER>" \
  -H "Content-Type: application/json" -d '{"body":"Ask about legal review.","note_type":"typed"}'

# Voice transcript (after STT)
curl -X POST "http://localhost:8001/ai-api/meeting-prep/<MEETING_ID>/prep-notes/voice?sellerId=<SELLER>" \
  -H "Content-Type: application/json" -d '{"transcript":"Customer wants phased delivery in July."}'
```

---

## Migrations & env

| Repo | Action |
|------|--------|
| AIBackend | Run `sql/2026_09_us06_meeting_briefing.sql` |
| AIBackend | `D365_BASE_URL=http://localhost:8000` |
| AIBackend | Optional: `BRIEFING_CACHE_TTL_HOURS=4`, `BRIEFING_MAX_SUMMARY_WORDS=100` |

---

## Message codes

| Code | Where shown | Meaning |
|------|-------------|---------|
| `INF_MSG_0004` | `deal_summary.competitor_intel.message_code` | No competitor rows in D365 |
| `ERR_MSG_0023` | `talking_points_message_code` | Insufficient grounded data for talking points |
| `ERR_MSG_0024` | D365 500 `detail` | Briefing context aggregation failed |
| `ERR_MSG_0025` | AIBackend 500 `detail` | Briefing card generation failed |

---

## Tests

```bash
# D365 Sales
pytest -q tests/test_briefing_context.py

# AIBackend
pytest -q tests/test_briefing_service.py
```

---

## AI team — implementation checklist

1. [ ] Read `GET /api/briefing/context` response shape (Postman folder `01`).
2. [ ] Implement LLM prompt that **only** uses fields from that JSON (no market knowledge).
3. [ ] Replace template strings in `generate_briefing()` with LLM output for `account_summary.paragraph` and `deal_summary.paragraph`.
4. [ ] Enhance `talking_points` generation with LLM, preserving `source` refs per point.
5. [ ] Wire STT → `POST .../prep-notes/voice` with `{ "transcript": "..." }`.
6. [ ] On post-meeting flow: `GET .../prep-notes` and pass to summary agent context.
7. [ ] Ensure calendar/scheduler writes `seller_id` + `opportunity_id` on `tbl_schedule_meetings`.

---

## FE team — implementation checklist

1. [ ] Meeting Prep list — call meetings API with `sellerId` + filter tabs.
2. [ ] Show `prep_tasks_pending_count` on each meeting card.
3. [ ] On card click → `GET .../briefing` — show loading while first generate runs.
4. [ ] Render header + 7 collapsible sections per mapping table above.
5. [ ] Source refs — link or tooltip from each item's `source` object.
6. [ ] Prep task checkbox → PATCH `done: true`; strikethrough when `done`.
7. [ ] Re-fetch list after task complete to update pending count (or optimistic UI).
8. [ ] My Prep Notes — typed form + voice button; call CRUD endpoints.
9. [ ] Hide Watch Out For when `watch_out_for === null`.
10. [ ] Show `INF_MSG_0004` / `ERR_MSG_0023` user-friendly copy when codes present.

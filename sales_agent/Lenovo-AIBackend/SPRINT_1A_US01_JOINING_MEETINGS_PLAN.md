# Sprint 1A · User Story 01 — Joining the Meetings · Backend Plan of Action

**Repo (primary):** `Lenovo-AIBackend`
**Repo (cross-cut):** `Lenovo D365 Sales` — one new endpoint for opportunity resolution
**Owner:** Sanmay (backend) · pairing with Namisha
**Effort:** ~5.5 hrs across both repos
**Status:** Plan locked, ready to implement.

---

## 1. What this story actually needs from backend

Strip away the bot mechanics — Microsoft Graph subscriptions, Teams Bot SDK, lobby handling, "Lenovo Sales Notes Bot" identity, audio capture — those are the **agent team's** scope. The backend's job is only:

1. **Persist** every Teams meeting the bot is asked to join
2. **Track lifecycle** of each meeting (pending → scheduled → joined / cancelled / rescheduled / failed)
3. **Resolve** which CRM opportunity a meeting belongs to (via attendee emails + subject line)

Acceptance-criteria mapping:

| AC from the story | Backend role |
|---|---|
| Bot joins eligible meetings automatically before start | None (bot/agent only) |
| Bot waits in lobby for customer-organized meetings | None (bot/agent only) |
| Bot appears as "Lenovo Sales Notes Bot" | None (bot identity config) |
| Bot does not join cancelled meetings | Backend exposes a status PATCH so the bot can read state |
| Bot auto-adjusts on reschedule | Existing upsert on `meeting_id` already handles this |
| Opportunity ID identified from contacts + subject | **NEW:** `POST /api/meetings/resolve-opportunity` in D365 backend |

---

## 2. Decisions locked

| Decision | Choice | Why |
|---|---|---|
| Where does the opportunity resolver live? | **D365 backend, called by AIBackend over HTTP** | Clean separation — the lookup runs on its own data, AIBackend doesn't reach into D365's tables |
| Include tests? | **Yes** — Phase 6 smoke tests with pytest + httpx | Repo has zero tests today; this is the right time to pin the contract |
| Migrations engine? | **Stay on `create_all()` + idempotent SQL files for this story; flag Alembic as separate tech-debt ticket** | Don't expand US01 scope; revisit before US02 lands |

---

## 3. Phased implementation

### Phase 1 — Schema extension on `tbl_schedule_meetings`

Add five nullable columns. Idempotent — re-runnable.

| Column | Type | Default | Purpose |
|---|---|---|---|
| `bot_status` | `text` | `'pending'` | Lifecycle state |
| `bot_status_reason` | `text NULL` | — | Free-form (e.g. "Lobby admit timeout") |
| `bot_last_event_at` | `timestamptz NULL` | — | Last status transition |
| `opportunity_id` | `uuid NULL` | — | Resolved D365 opportunity |
| `account_id` | `uuid NULL` | — | Resolved D365 account |

Plus a CHECK constraint on `bot_status`:
```sql
ALTER TABLE tbl_schedule_meetings
  ADD CONSTRAINT chk_bot_status CHECK (bot_status IN (
    'pending','scheduled','joining','joined',
    'lobby_waiting','cancelled','rescheduled','failed'
  ));
```

**Files:**
- New: `sql/2026_06_us01_meeting_lifecycle.sql`
- Edit: `app/models/schedulemeeting.py`

**Effort:** ~30 min.

---

### Phase 2 — Status PATCH endpoint

```
PATCH /meeting-details/{meeting_id}/status
  body:  { "botStatus": "cancelled", "reason": "Calendar event deleted" }
  200:   { "meetingId": "...", "botStatus": "cancelled", "updatedAt": "..." }
  404:   { "detail": "Meeting not found" }
  422:   validation error if botStatus not in whitelist
```

Updates `bot_status`, `bot_status_reason`, sets `bot_last_event_at = now()`.

**Files:**
- Edit: `app/schema/schedulemeeting.py` (add `MeetingStatusUpdate` + `MeetingStatusResponse`)
- Edit: `app/services/meeting_details_service.py` (add `update_meeting_status`)
- Edit: `app/api/meeting_details.py` (wire the route)

**Effort:** ~45 min.

---

### Phase 3 — Cancellation convenience endpoint

Sugar over Phase 2 — bot's most common write is "this meeting is cancelled":

```
DELETE /meeting-details/{meeting_id}?reason=...
  200:   { "meetingId": "...", "botStatus": "cancelled" }
```

Soft-delete: doesn't remove the row, just flips `bot_status='cancelled'`.

**Files:** same three as Phase 2.

**Effort:** ~15 min.

---

### Phase 4 — Opportunity / account writeback on upsert

Extend the existing `POST /meeting-details/` body to accept the IDs the bot got back from the D365 resolver:

```jsonc
POST /meeting-details/
{
  "meeting_id": "...",
  "meeting_start_time": "...",
  "meeting_end_time": "...",
  "platform": "Microsoft Teams",
  "title": "ThinkPad Fleet Review",
  "attendees": "...",
  "organiser_name": "...",
  "opportunity_id": "b0000001-...",   // ← NEW (optional)
  "account_id":     "6dc95c38-..."     // ← NEW (optional)
}
```

The existing upsert already does `ON CONFLICT DO UPDATE` — just add the two keys to the SET map.

**Files:**
- Edit: `app/schema/schedulemeeting.py` — two new optional fields on `MeetingDetailsCreate` + `MeetingDetailsResponse`
- Edit: `app/services/meeting_details_service.py` — extend the `set_` dict in `upsert_meeting`

**Effort:** ~15 min.

---

### Phase 5 — Cross-repo: D365 resolver endpoint + AIBackend HTTP client

#### 5a — D365 backend (`Lenovo D365 Sales`)

New endpoint:
```
POST /api/meetings/resolve-opportunity
  body:  {
           "attendeeEmails": ["k.richter@db.com", "rajesh.k@infosys.com"],
           "subject":        "ThinkPad Fleet Review",
           "organiserEmail": "owner@lenovo.com"
         }
  200:   {
           "opportunityId": "b0000001-...",
           "accountId":     "6dc95c38-...",
           "matchScore":    0.92,
           "matchedBy":     "contact_email"   // or "subject_keyword" or "both"
         }
  404:   { "detail": { "code": "NO_MATCH", "message": "No deal found for these contacts" } }
```

Logic:
1. Lookup `attendeeEmails` against `contact.emailaddress1` (case-insensitive)
2. For each matched contact, walk `lvo_opportunitycontact → opportunity` (Active links only, non-Cancelled deals only)
3. Score each candidate:
   - Base score: count of matching contacts on the deal
   - Boost: `+0.5` if subject contains any token from `opportunity.name`
4. Return the highest-scoring deal; 404 if everyone's score is 0

**Files:**
- Edit: `app/routers/opportunities.py` (add the route — or new `meetings_resolver.py` router)
- Edit: `app/schemas.py` — `OpportunityResolveRequest`, `OpportunityResolveResponse`
- Edit: `API_CONTRACT.md` — document the new endpoint

**Effort:** ~2 hrs.

#### 5b — AIBackend HTTP client wrapper

A tiny module the bot will call before scheduling itself:

```
app/clients/d365_client.py:
  resolve_opportunity(attendees, subject, organiser) -> ResolverResult | None
    └─ wraps httpx call to D365's /api/meetings/resolve-opportunity
    └─ handles 404 (returns None), 5xx (raises), timeout (raises)
    └─ reads D365_BASE_URL from env (.env)
```

**Files:**
- New: `app/clients/__init__.py`, `app/clients/d365_client.py`
- Edit: `requirements.txt` — add `httpx`
- Edit: `.env` (and document) — add `D365_BASE_URL=http://localhost:8000`

**Effort:** ~30 min.

---

### Phase 6 — README + cURL smoke tests

Replace the placeholder TODO README with:
- Local-dev setup (`uvicorn app.main:app --reload --port 8001`)
- Each endpoint with copy-pasteable cURL
- A status-state diagram (`pending → scheduled → joined / cancelled / rescheduled / failed`)
- Pointer to the D365 resolver and how to wire `D365_BASE_URL`

**Files:**
- Edit: `README.md`

**Effort:** ~30 min.

---

### Phase 7 — Tests

The repo has zero tests today. Lay the foundation now while scope is small.

Five smoke tests covering the contract:
1. POST a meeting → row appears with `bot_status='pending'`
2. POST same `meeting_id` again → row updated, status preserved
3. PATCH status to `joined` → row updated, `bot_last_event_at` populated
4. PATCH with invalid status → 422
5. DELETE → row's status flips to `cancelled`, row not actually removed

**Files:**
- New: `tests/test_meeting_lifecycle.py`
- New: `tests/conftest.py` — test DB fixture, FastAPI `TestClient`
- Edit: `requirements.txt` — add `pytest`, `httpx`

**Effort:** ~1 hr.

---

## 4. Effort summary

| Phase | Repo | Effort |
|---|---|---|
| 1 — Schema | AIBackend | 30 min |
| 2 — PATCH status | AIBackend | 45 min |
| 3 — DELETE convenience | AIBackend | 15 min |
| 4 — Opportunity / account on upsert | AIBackend | 15 min |
| 5a — D365 resolver endpoint | D365 Sales | 2 hrs |
| 5b — AIBackend HTTP client | AIBackend | 30 min |
| 6 — README + cURL | AIBackend | 30 min |
| 7 — Tests | AIBackend | 1 hr |
| **Total** | | **~5 hr 45 min** |

---

## 5. Out of scope for this story

| Item | Why deferred |
|---|---|
| Manual-link-sharing endpoint | Story explicitly says "in Sprint 1A, the agent operates entirely in the background without UI integration" — no UI to surface a manual entry box |
| Audit log integration (`lvo_audit_log`) | The status column is the audit record for v1; revisit if compliance asks |
| Alembic migrations | Tech-debt ticket separately |
| Webhook receiver for Outlook calendar changes | Bot/agent team's responsibility — they own the Graph subscription |
| Authn/Authz between bot and AIBackend | Add bearer-token check in a follow-up; for now AIBackend is internal-only |

---

## 6. Open questions for PM / agent team

| # | Question | Why it matters |
|---|---|---|
| 1 | If `matchScore` from the resolver is below threshold (say 0.5), should the bot still join (with `opportunity_id=NULL`) or skip entirely? | Decides whether the resolver's 404 means "skip the meeting" or "join anyway" |
| 2 | Reschedule de-duplication key — same Outlook event ID, or new one? | Decides whether the bot upserts (same `meeting_id`) or inserts (new `meeting_id`) |
| 3 | Cancellation source — Outlook calendar webhook only, or can a seller also tell the bot directly? | Decides who calls `PATCH /status` (bot's webhook handler vs. a future UI) |
| 4 | Is opportunity resolution **eager** (compute now and store) or **lazy** (compute when the transcript arrives)? | Eager is faster for the AI summary; lazy survives deal-contact link changes between schedule and execution. Plan as drafted assumes **eager** |
| 5 | What's the threshold for accepting a resolver match — score `>= 0.5` (lenient), `>= 0.8` (strict), or never auto-decide and always 404? | Affects how often bot ends up with NULL `opportunity_id` |

---

## 7. Acceptance criteria for "this story is done from backend's side"

- [ ] `tbl_schedule_meetings` has the 5 new columns + CHECK constraint
- [ ] PATCH `/meeting-details/{id}/status` works for all 8 status values, returns 422 on invalid
- [ ] DELETE `/meeting-details/{id}` flips the row to `cancelled`, doesn't delete
- [ ] POST `/meeting-details/` accepts and writes back `opportunity_id` + `account_id`
- [ ] D365 backend: `POST /api/meetings/resolve-opportunity` returns the right deal for known attendee emails
- [ ] AIBackend `d365_client.resolve_opportunity()` wraps the HTTP call cleanly (returns None on 404, raises on 5xx)
- [ ] All 5 smoke tests pass
- [ ] README has cURL for every new endpoint + the state diagram
- [ ] D365's `API_CONTRACT.md` documents the resolver
- [ ] Both repos lint clean

---

## 8. Order of work (for the implementation pass)

Suggested sequence so we can test as we go:

1. Phase 1 (schema) → can test by reading the row in pgAdmin
2. Phase 2 (PATCH status) → can test with cURL
3. Phase 3 (DELETE) → can test with cURL
4. Phase 4 (upsert with IDs) → can test with cURL
5. Phase 5a (D365 resolver) → switch repos briefly, ship the endpoint, document
6. Phase 5b (AIBackend HTTP client) → switch back, wire the call
7. Phase 7 (tests) → harness everything we just built
8. Phase 6 (README) → finalize, screenshot the state diagram, hand off to Namisha + agent team

---

## 9. TL;DR

- **5 small endpoints, ~6 hrs across two repos.**
- **AIBackend:** schema bump + PATCH status + DELETE + upsert IDs + HTTP client + tests + README.
- **D365 backend:** one new resolver endpoint with attendee-email + subject scoring.
- **No new tables**, no Alembic, no audit-log changes — keeps US01 tight so US02 (transcript ingestion + summary storage) can build cleanly on top.

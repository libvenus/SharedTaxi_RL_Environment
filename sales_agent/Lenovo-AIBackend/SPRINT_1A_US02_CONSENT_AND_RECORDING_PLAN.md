# Sprint 1A · User Story 02 — Consent & Recording · Backend Plan of Action

**Repo (primary):** `Lenovo-AIBackend`
**Repo (cross-cut):** `Lenovo D365 Sales` — one new contact-resolver endpoint
**Owner:** Sanmay (backend) · pairing with Namisha
**Effort:** ~8 hrs across both repos
**Status:** Plan locked, ready to implement.
**Linked docs:**
- [SPRINT_1A_US01_JOINING_MEETINGS_PLAN.md](./SPRINT_1A_US01_JOINING_MEETINGS_PLAN.md) — US01 plan (this builds on top)
- [US01_BACKEND_HANDOFF_FOR_AI_TEAM.md](./US01_BACKEND_HANDOFF_FOR_AI_TEAM.md) — endpoint reference handed to AI team

---

## 1. What this story actually needs from backend

US02 is the AI team's biggest implementation lift (audio capture, diarization, speaker ID, Teams chat write, organiser-only permission check, meeting-end detection). Backend's role is much narrower:

1. **Persist** the consent-message proof (compliance evidence)
2. **Persist** every transcript segment as it streams in (continuously, not at end)
3. **Persist** the finalisation event (overall confidence + termination reason)
4. **Provide** a CRM contact-lookup-by-email endpoint so the bot can enrich speakers once at meeting start

Acceptance-criteria mapping:

| AC | Backend role |
|---|---|
| 1. Consent message sent to chat before audio capture | Record `consent_message_text` + `consent_sent_at`; reject `POST /transcripts/` without them |
| 2. Each segment captures speaker + utterance + timestamps + confidence | Store all 7 fields; enforce `confidence ∈ [0, 1]` |
| 3. Speakers identified from CRM or "Unknown Attendee" | Provide D365 `/api/contacts/resolve-by-emails`; bot enriches locally |
| 4. Segments saved continuously during the meeting | `POST /transcripts/{id}/segments` accepts batched writes; bot calls repeatedly |
| 5. Only organiser can remove the agent | Record termination event with reason — bot owns the permission check |
| 6. When meeting ends, finalised transcript linked to D365 opportunity | `POST /transcripts/{id}/finalize` + denormalised `opportunity_id` on transcript row |

---

## 2. Decisions locked

| Decision | Choice | Why |
|---|---|---|
| Storage location | **AIBackend** | Sits next to `tbl_schedule_meetings`; consistent with US01 persistence boundary |
| Schema shape | **Row-per-segment** | Queryable, append-friendly, supports per-segment confidence and search |
| Audio file | **Transcripts only** | AC mentions transcripts only; bot keeps audio in its own blob storage if needed |
| Speaker enrichment | **Bot pre-enriches once at meeting start** | One round-trip per meeting; bot tags each utterance locally |
| Consent enforcement | **Required on `POST /transcripts/`** | No transcript can exist without compliance proof |
| Duplicate segments | **Trust bot for v1** | Just append; revisit dedup if observed in dev |

---

## 3. Architecture

```
       Bot joins meeting (US01 PATCH .../status to 'joined')
                            │
                            ▼
       Bot sends CONF_MSG_0004 in Teams chat (AI team)
                            │
                            ▼
   ┌────────────────────────────────────────────────────────┐
   │ POST /transcripts/                                     │
   │  body: meeting_id + consent_message_text + consent_at  │
   │  → tbl_meeting_transcript row (status='in_progress')   │
   └────────────────────────────────────────────────────────┘
                            │
                            ▼
       Bot calls D365 once to enrich attendees
   ┌────────────────────────────────────────────────────────┐
   │ POST /api/contacts/resolve-by-emails  (D365 Sales)     │
   │  body: { emails: [...] }                               │
   │  → returns name / jobTitle / contactId / accountId     │
   └────────────────────────────────────────────────────────┘
                            │
                            ▼
       During meeting: bot batches segments and pushes
   ┌────────────────────────────────────────────────────────┐
   │ POST /transcripts/{meeting_id}/segments    (continuous)│
   │  body: { segments: [ {speaker, text, start, end, ...}]}│
   │  → INSERT N rows into tbl_meeting_transcript_segment   │
   └────────────────────────────────────────────────────────┘
                            │
                            ▼
                    (one of two terminations)
                  ┌─────────┴─────────┐
                  ▼                   ▼
   ┌───────────────────┐    ┌──────────────────────────┐
   │ Meeting ends      │    │ Organiser removes bot    │
   │ POST .../finalize │    │ POST .../terminate       │
   │ + overall_score   │    │ + reason='organizer_…'   │
   └───────────────────┘    └──────────────────────────┘
```

---

## 4. Phased implementation

### Phase 1 — Schema (AIBackend)

Two new tables, idempotent migration `sql/2026_06_us02_meeting_transcript.sql`.

#### `tbl_meeting_transcript` — one row per meeting

| Column | Type | Notes |
|---|---|---|
| `transcript_id` | `uuid` PK | |
| `meeting_id` | `uuid` UNIQUE NOT NULL | FK-by-convention to `tbl_schedule_meetings.meeting_id` |
| `opportunity_id` | `uuid` | Denormalised from meeting for fast filtering |
| `account_id` | `uuid` | Same |
| `status` | `text NOT NULL` CHECK | `in_progress` / `finalized` / `terminated_partial` |
| `consent_message_text` | `text NOT NULL` | The actual `CONF_MSG_0004` content (compliance proof) |
| `consent_sent_at` | `timestamptz NOT NULL` | When the bot sent the chat message |
| `overall_confidence_score` | `numeric(4,3)` | NULL until finalised; 0.000–1.000 |
| `segment_count` | `int NOT NULL DEFAULT 0` | Maintained on every append |
| `terminated_reason` | `text` CHECK | `meeting_ended` / `organizer_removed` / `all_left` / `bot_failure` |
| `started_at` | `timestamptz NOT NULL` | When bot started capturing (= consent_sent_at typically) |
| `finalized_at` | `timestamptz` | NULL until finalised / terminated |
| `created_at` / `updated_at` | `timestamptz` | Server defaults |

#### `tbl_meeting_transcript_segment` — one row per utterance

| Column | Type | Notes |
|---|---|---|
| `segment_id` | `uuid` PK | |
| `transcript_id` | `uuid NOT NULL` | FK to `tbl_meeting_transcript` |
| `meeting_id` | `uuid NOT NULL` | Denormalised — most queries filter by meeting |
| `speaker_name` | `text NOT NULL` | "Klaus Richter" or "Unknown Attendee" |
| `speaker_email` | `text` | Nullable for unknown speakers |
| `speaker_role` | `text` | "CTO" / "Decision Maker" — bot resolved from CRM |
| `speaker_contact_id` | `uuid` | NULL if unidentified |
| `utterance_text` | `text NOT NULL` | |
| `start_time` | `timestamptz NOT NULL` | Absolute, not meeting-relative |
| `end_time` | `timestamptz NOT NULL` | |
| `confidence_score` | `numeric(4,3) NOT NULL` CHECK | 0.000–1.000 |
| `created_at` | `timestamptz` | Server default |

Plus indexes:
- `idx_transcript_meeting` on `tbl_meeting_transcript(meeting_id)` UNIQUE
- `idx_transcript_status` on `tbl_meeting_transcript(status)` partial WHERE `status='in_progress'` (sweepers later)
- `idx_segment_meeting_time` on `tbl_meeting_transcript_segment(meeting_id, start_time)` — primary read path
- `idx_segment_transcript` on `tbl_meeting_transcript_segment(transcript_id)`
- `idx_segment_contact` on `tbl_meeting_transcript_segment(speaker_contact_id)` partial WHERE NOT NULL

**Effort:** ~45 min.

---

### Phase 2 — ORM + Pydantic schemas (AIBackend)

**Files:**
- `app/models/transcript.py` — `MeetingTranscript`, `MeetingTranscriptSegment`
- `app/schema/transcript.py` — Create / Append / Finalize / Terminate / Response schemas
- `BotStatus`-style `Literal` for `transcript_status` and `terminated_reason`

**Effort:** ~45 min.

---

### Phase 3 — Service layer (AIBackend)

`app/services/transcript_service.py`:

```python
def start_transcript(db, meeting_id, consent_text, consent_sent_at) -> MeetingTranscript:
    """Validates: meeting exists + bot_status in {'joined', 'lobby_waiting'}.
       Raises 409 if a transcript already exists for the meeting."""

def append_segments(db, meeting_id, segments) -> int:
    """Bulk INSERT; raises 404 if no transcript; raises 400 if status != 'in_progress'.
       Returns the new segment_count."""

def finalize_transcript(db, meeting_id, overall_score) -> MeetingTranscript:
    """Sets status='finalized', stamps finalized_at, terminated_reason='meeting_ended'."""

def terminate_transcript(db, meeting_id, reason) -> MeetingTranscript:
    """Sets status='terminated_partial', stamps finalized_at + reason."""

def get_transcript(db, meeting_id) -> tuple[MeetingTranscript, list[MeetingTranscriptSegment]]:
    """Metadata + ordered segments. Used by FE on the Activity tab later."""
```

**Effort:** ~1 hr.

---

### Phase 4 — API endpoints (AIBackend)

`app/api/transcripts.py`, mounted at `/transcripts`:

```
POST   /transcripts/                        # start (consent required)
POST   /transcripts/{meeting_id}/segments   # append (batched)
POST   /transcripts/{meeting_id}/finalize   # clean end
POST   /transcripts/{meeting_id}/terminate  # organizer removed / partial
GET    /transcripts/{meeting_id}            # FE / read
```

Error surface:
- `404` — unknown meeting / no transcript
- `409` — transcript already exists (on start)
- `400` — bot_status not eligible / transcript already finalised
- `422` — schema validation (invalid confidence, missing consent)

**Effort:** ~1 hr.

---

### Phase 5 — D365 contact-resolver endpoint (D365 Sales)

```
POST /api/contacts/resolve-by-emails
  body: { "emails": ["k.richter@db.com", "rajesh@infosys.com"] }
  200:  {
    "results": [
      {
        "email":          "k.richter@db.com",
        "contactId":      "E7CF7AAF-CD45-...",
        "name":           "Klaus Richter",
        "jobTitle":       "CTO",
        "accountId":      "6dc95c38-...",
        "accountName":    "Deutsche Bank AG",
        "role":           "Decision Maker"
      },
      { "email": "unknown@nope.com", "contactId": null, "name": null, ... }
    ]
  }
```

Role resolution rule: pick the most "senior" link from `lvo_opportunitycontact`:
1. `lvo_isdecisionmaker = true` first
2. Then alphabetic by `lvo_role`

If contact isn't on any opportunity, role is `null`.

**Files:**
- `app/routers/contacts_resolver.py` (new)
- `app/schemas.py` — `ContactResolveRequest`, `ContactResolveResult`, `ContactResolveResponse`
- `app/main.py` — register router, bump version → 0.15.0
- `API_CONTRACT.md` — new §13

**Effort:** ~1.5 hrs.

---

### Phase 6 — `d365_client.py` extension (AIBackend)

Add `resolve_contacts_by_email(emails: list[str]) -> list[ContactResolveResult]`. Same error-handling shape as the existing `resolve_opportunity()`:
- Returns list of results (one per email, with `contactId=None` for unknowns)
- Raises `D365ClientError` on 5xx / timeout / unexpected payload
- 422 from D365 (empty emails) → raise

**Effort:** ~30 min.

---

### Phase 7 — Tests (AIBackend)

8 smoke tests in `tests/test_transcript_lifecycle.py`:

1. `POST /transcripts/` creates row with `status='in_progress'` + consent fields populated
2. `POST /transcripts/` without `consent_message_text` → 422
3. `POST /transcripts/` for a meeting that isn't `joined` → 400
4. `POST /transcripts/` for a meeting that already has a transcript → 409
5. `POST .../segments` appends correctly, increments `segment_count`
6. `POST .../segments` with `confidence_score = 1.5` → 422
7. `POST .../finalize` sets status + `finalized_at` + `overall_confidence_score`
8. `POST .../terminate` with `reason='organizer_removed'` sets `terminated_partial` + reason
9. `GET /transcripts/{meeting_id}` returns metadata + segments **ordered by `start_time`**

**Effort:** ~1.5 hrs.

---

### Phase 8 — Docs + handoff

- README — add transcript section + new state diagram
- `US02_BACKEND_HANDOFF_FOR_AI_TEAM.md` — analogous to US01's handoff doc
- `US02_STATUS_AND_DEPENDENCIES.md` — analogous tracking doc
- D365 `API_CONTRACT.md` §13 — document the contact resolver
- Update `tbl_schedule_meetings` state diagram to show the transcript hand-off

**Effort:** ~1 hr.

---

## 5. Effort summary

| Phase | Repo | Effort |
|---|---|---|
| 1 — Schema | AIBackend | 45 min |
| 2 — ORM + schemas | AIBackend | 45 min |
| 3 — Service layer | AIBackend | 1 hr |
| 4 — API endpoints | AIBackend | 1 hr |
| 5 — Contact-resolver endpoint | D365 Sales | 1.5 hrs |
| 6 — `d365_client.py` extension | AIBackend | 30 min |
| 7 — Tests | AIBackend | 1.5 hrs |
| 8 — Docs + handoff | Both | 1 hr |
| **Total** | | **~8 hrs** |

---

## 6. Out of scope (deferred to later sprints / stories)

| Item | Why deferred |
|---|---|
| Post-meeting AI summary (key points, action items, sentiment) | That's a separate story (US03 territory). Design notes already exist at `Lenovo D365 Sales/MEETING_INTELLIGENCE_STORAGE_BACKEND_NOTES.md` |
| Audio file storage | AC mentions transcripts only; bot keeps audio in its own blob storage |
| Real-time transcript streaming to FE (WebSocket / SSE) | FE wires up Activity tab later as poll-on-load; revisit if real-time becomes a P0 |
| Transcript search / full-text indexing | DB has the data; add `tsvector` index when search becomes a feature |
| Sweeper job to mark stale `in_progress` transcripts (bot crashed) | Add in 1B; not blocking US02 |
| Pagination on `GET /transcripts/{id}` | A 2-hour meeting is ~3000 segments at most; revisit if we hit larger transcripts |
| Authn / authz between bot and AIBackend | AIBackend internal-only for now; bearer-token for v2 |

---

## 7. Open questions for AI team / PM

| # | Question | Why it matters | Owner |
|---|---|---|---|
| Q1 | Where does the canonical text of `CONF_MSG_0004` live? Is it a config string we both reference, or does each repo hard-code its own copy? | Bot sends it; we record what bot sent — should agree on a single source | PM |
| Q2 | Bot's preferred batch size for segment POSTs? (Per-utterance? Every N seconds? Every N segments?) | Decides server load profile | AI team |
| Q3 | What's the bot's behaviour on network failure mid-meeting — retry + buffer? Drop? | Decides whether dedup matters in v1 | AI team |
| Q4 | Should "bot was removed" be detectable by us via `bot_status='cancelled'` (US01) too, or is `terminate` enough? | Decides if the two endpoints should auto-cascade | Joint |
| Q5 | When meeting ends, who computes `overall_confidence_score` — the bot, or do we recompute from segments? | Bot owns it (recommended); we trust the value | AI team |
| Q6 | Is "unknown attendee" rendered as a single bucket per meeting, or per voice (Unknown Attendee 1, 2, 3)? | Affects how bot tags `speaker_name` | AI team |

---

## 8. Acceptance criteria for "this story is done from backend's side"

- [ ] Two new tables created on dev with the migration applied
- [ ] All 5 transcript endpoints work via cURL on dev
- [ ] D365 contact resolver returns the right result for known + unknown emails
- [ ] `d365_client.py` exports `resolve_contacts_by_email()`
- [ ] All 9 smoke tests pass
- [ ] README has cURL for every new endpoint + the transcript state diagram
- [ ] D365's `API_CONTRACT.md` documents the contact resolver
- [ ] AI team has the `US02_BACKEND_HANDOFF_FOR_AI_TEAM.md` doc
- [ ] Both repos lint clean

---

## 9. Order of work (for the implementation pass)

Suggested sequence so we can test as we go:

1. Phase 1 (schema) → confirm via pgAdmin
2. Phase 2 (ORM + schemas) → no DB, just import-time validation
3. Phase 3 (service layer) → unit-testable in isolation
4. Phase 4 (API endpoints) → cURL smoke from Postman
5. Phase 5 (D365 contact resolver) → switch repos, build, document
6. Phase 6 (`d365_client.py` extension) → switch back, wire the call
7. Phase 7 (tests) → harness everything we just built
8. Phase 8 (docs + handoff) → finalise, screenshot, hand off

---

## 10. TL;DR

- **5 endpoints + 2 tables, ~8 hrs across two repos.**
- **AIBackend:** schema + 5 transcript endpoints + service layer + tests + handoff
- **D365 Sales:** one new contact-resolver endpoint
- **No new tables in D365**, no Alembic, no audit-log changes — keeps US02 tight so US03 (post-meeting AI summary) can build cleanly on top

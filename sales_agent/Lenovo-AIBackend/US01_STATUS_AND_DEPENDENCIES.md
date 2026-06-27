# US01 — Joining the Meetings · Status & Dependencies

**Sprint:** 1A
**Story:** "Note-Taking Agent — Joining the Meetings"
**Last updated:** 2026-06-15
**Owner (Backend slice):** Sanmay (with Namisha)
**Owner (Bot/Agent slice):** AI / Agent team
**Linked docs:**
- [US01_BACKEND_HANDOFF_FOR_AI_TEAM.md](./US01_BACKEND_HANDOFF_FOR_AI_TEAM.md) — endpoint reference for AI team
- [SPRINT_1A_US01_JOINING_MEETINGS_PLAN.md](./SPRINT_1A_US01_JOINING_MEETINGS_PLAN.md) — original plan of action
- `Lenovo D365 Sales/MEETING_INTELLIGENCE_STORAGE_BACKEND_NOTES.md` — design notes for US02 (transcript ingest)

---

## 1. TL;DR

| Slice | Status | Owner |
|---|---|---|
| **Backend (persistence + lifecycle + CRM resolver)** | ✅ Done — all 4 endpoints shipped, 11 tests passing, docs handed off | Backend (us) |
| **Bot / Agent (Graph webhook, Teams Bot SDK, identity, audio capture)** | ❌ Not started | AI team |
| **Story-level "done" (seller experience verified end-to-end)** | ❌ Pending — blocked on bot | Joint |

> **Plain-English status for standup:** "Backend is complete and tested. Story moves to ✅ when the AI team's bot is calling our endpoints from production code."

---

## 2. What's DONE — Backend slice

### 2.1 Endpoints shipped

| Method | Path | Repo | Tests | Docs |
|---|---|---|:---:|:---:|
| `POST` | `/meeting-details/` *(extended with `opportunity_id`, `account_id`)* | AIBackend | ✅ | ✅ |
| `PATCH` | `/meeting-details/{id}/status` | AIBackend | ✅ | ✅ |
| `DELETE` | `/meeting-details/{id}` *(soft-delete)* | AIBackend | ✅ | ✅ |
| `POST` | `/api/meetings/resolve-opportunity` | D365 Sales | manual cURL | ✅ |

### 2.2 Code artifacts

**`Lenovo-AIBackend`:**

| Path | What it is |
|---|---|
| `sql/2026_06_us01_meeting_lifecycle.sql` | Idempotent migration — 5 new columns + CHECK + 3 indexes |
| `app/models/schedulemeeting.py` | ORM model extended with lifecycle + CRM-link columns |
| `app/schema/schedulemeeting.py` | `BotStatus` literal, `MeetingStatusUpdate`, `MeetingStatusResponse` |
| `app/services/meeting_details_service.py` | `update_meeting_status()`, `cancel_meeting()` |
| `app/api/meeting_details.py` | PATCH + DELETE routes wired |
| `app/clients/d365_client.py` | httpx wrapper for D365 resolver |
| `app/core/config.py` | `D365_BASE_URL` / `D365_TIMEOUT_SECONDS` env vars |
| `tests/conftest.py` | SQLite + StaticPool fixture |
| `tests/test_meeting_lifecycle.py` | 11 smoke tests (all passing) |
| `README.md` | Setup + cURL + state diagram |

**`Lenovo D365 Sales`:**

| Path | What it is |
|---|---|
| `app/routers/meetings_resolver.py` | Resolver endpoint (attendee email + subject scoring) |
| `app/schemas.py` | `OpportunityResolveRequest`, `OpportunityResolveResponse` |
| `app/main.py` | Router registered, version → 0.14.0 |
| `API_CONTRACT.md` | §12 documents the new endpoint |

### 2.3 Database changes

5 new columns on `tbl_schedule_meetings`:

| Column | Type | Purpose |
|---|---|---|
| `bot_status` | `text NOT NULL` (default `'pending'`) | Lifecycle state |
| `bot_status_reason` | `text` | Most-recent transition context |
| `bot_last_event_at` | `timestamptz` | Auto-stamped on every PATCH |
| `opportunity_id` | `uuid` | Resolved CRM opportunity |
| `account_id` | `uuid` | Resolved CRM account |

Plus CHECK constraint on `bot_status` whitelist + 3 indexes.

### 2.4 Test coverage

11 automated tests (run with `pytest -q`):

1. POST creates row with `bot_status='pending'`
2. Re-POST preserves lifecycle state (idempotency)
3-8. PATCH transitions through 6 non-trivial states
9. PATCH with garbage status → 422
10. PATCH on unknown meeting → 404
11. DELETE soft-deletes (status flips, row preserved)

---

## 3. What's PENDING

### 3.1 Pending — AI / Agent team (BLOCKING story-level done)

| # | Item | Owner | Blocker for |
|---|---|---|---|
| A1 | Microsoft Graph change-notification subscription (calendar events) | AI team | Everything below |
| A2 | Bot scheduler logic (decide when to join, parse Outlook events) | AI team | Story AC #1 |
| A3 | Teams Bot SDK integration (actual join, audio capture) | AI team | Story AC #1, #2 |
| A4 | Bot identity: appears as "Lenovo Sales Notes Bot" in participant list | AI team | Story AC #3 |
| A5 | Lobby-admit handling for customer-organised meetings | AI team | Scenario 3 |
| A6 | Calling our endpoints from inside the bot | AI team | All scenarios |
| A7 | Calling D365 resolver before scheduling | AI team | Scenario 1 (opportunity tagging) |
| A8 | Reschedule detection + re-tracking flow | AI team | Scenario 4 |
| A9 | "Already started" detection within 2 minutes | AI team | Scenario 5 |

### 3.2 Pending — Backend (NON-BLOCKING for story; nice-to-have)

| # | Item | Status | Why deferred |
|---|---|---|---|
| B1 | Authn / authz between bot and AIBackend | Deferred | AIBackend is internal-only for v1; add bearer-token before public deploy |
| B2 | `GET /meeting-details/{id}` (read-by-id) | Not built | Bot doesn't need it for US01 — it owns the IDs. Add when FE wires up Activity tab |
| B3 | Audit log for every status transition | Deferred | Current row stores only the latest; revisit if compliance asks |
| B4 | Alembic migrations | Deferred | Tracked tech debt; idempotent SQL files are fine for 1A |
| B5 | Pydantic v2 cleanup (`class Config` → `model_config = ConfigDict(...)`) | Deferred | 2 deprecation warnings in test output, cosmetic only |
| B6 | Manual link-sharing endpoint (paste a Teams URL) | Deferred | Story explicitly says "no UI integration in 1A" |
| B7 | Sweeper job to auto-fail stale `scheduled` meetings | Deferred | Not in 1A scope; revisit in 1B |
| B8 | E2E integration test against real Postgres (covers DB CHECK constraint) | Deferred | SQLite tests don't observe the CHECK; needs CI Postgres |
| B9 | Mock-server tests for `d365_client.py` | Deferred | Live integration test will catch this; mock with `respx` if needed later |

### 3.3 Pending — Joint (story-level)

| # | Item | Owner |
|---|---|---|
| C1 | Dev-server end-to-end smoke: bot detects meeting → calls our endpoints → row appears with correct state | Both |
| C2 | Sprint review demo: live Teams meeting joined by "Lenovo Sales Notes Bot" with row in DB | Both |
| C3 | Threshold tuning for the resolver `match_score` (currently recommended ≥ 0.5) | Joint, after first real-data run |

---

## 4. Dependencies

### 4.1 Cross-team dependencies (BLOCKING)

```
        Backend (DONE)                    AI / Agent team (NOT STARTED)
        ──────────────                    ──────────────────────────────
                                                 │
   ┌──────────────────────┐                      │ depends on
   │ POST /meeting-       │  ◄───────────────────┤ A1 Graph webhook
   │ details/             │                      │ A2 Bot scheduler
   │ PATCH .../status     │                      │ A6 Calling our endpoints
   │ DELETE .../{id}      │                      │
   └──────────────────────┘                      │
                                                 │
   ┌──────────────────────┐                      │
   │ POST /api/meetings/  │  ◄───────────────────┤ A7 Calling resolver
   │ resolve-opportunity  │                      │
   └──────────────────────┘                      │
                                                 ▼
                                          Bot service exists,
                                          calls all 4 endpoints
                                                 │
                                                 ▼
                                          Story-level DONE
```

**Bottom line:** Nothing on the backend side blocks the AI team from starting **today**. Everything they need (endpoints, schemas, cURL examples, the `d365_client.py` wrapper) is documented in `US01_BACKEND_HANDOFF_FOR_AI_TEAM.md`.

### 4.2 Cross-story dependencies

| Depends on | Depends on | Detail |
|---|---|---|
| **US01** | — | Self-contained. No upstream dependency. |
| **US02 (transcript ingest)** | US01 | US02 reads `tbl_schedule_meetings.opportunity_id` — needs the resolver + lifecycle work US01 ships |
| **Activity tab on FE** | US01 + US02 | FE renders meeting status + transcript snippet from rows we persist |
| **Manual link-sharing scenario** | US01 + future FE story | Needs a UI input box that doesn't exist in 1A |

### 4.3 Infrastructure dependencies

| Item | Owner | Status |
|---|---|---|
| Dev Postgres has the `2026_06_us01_meeting_lifecycle.sql` migration applied | Sanmay | ⚠️ **Pending — apply on dev before AI team starts** |
| `D365_BASE_URL` set on the AI team's bot deployment | AI team | Pending |
| `tbl_schedule_meetings` exists in dev (it does — pre-US01) | — | ✅ Already there |
| D365 Sales backend deployed at known host:port (the AI team can reach) | DevOps | ✅ `http://10.245.240.33:8000` |

### 4.4 Infrastructure / DevOps tasks

| # | Task | Owner | Priority |
|---|---|---|---|
| D1 | Apply migration on dev Postgres | Sanmay | High — blocks AI team |
| D2 | Deploy AIBackend to a host the bot can reach (currently runs locally) | DevOps + Sanmay | High — blocks AI team |
| D3 | Open firewall: bot → D365 backend host:port | DevOps | High |
| D4 | Open firewall: bot → AIBackend host:port | DevOps | High |

---

## 5. Risks & open questions

### 5.1 Open questions for AI team / PM

| # | Question | Why it matters | Owner |
|---|---|---|---|
| Q1 | Match-score threshold for auto-tagging? Recommended ≥ 0.5 — agree? | Decides how often bot joins untagged | AI team + PM |
| Q2 | Reschedule key: same Outlook event ID or new? | Decides whether bot upserts or insert+delete | AI team |
| Q3 | Eager vs lazy CRM resolution? Plan assumes eager (call resolver before scheduling) | Eager is faster; lazy survives contact-link changes | PM |
| Q4 | 5xx retry policy on resolver? | Currently raises immediately; bot's caller decides | AI team |
| Q5 | Test data for E2E rehearsal — curated email/subject/deal pairs | Verifies resolver against real CRM data | Sanmay + AI team |
| Q6 | Cancellation source — Outlook webhook only, or future UI too? | Decides if `DELETE /meeting-details/{id}` needs auth distinguishing bot vs user | PM |

### 5.2 Risks

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | AI team picks a non-Python stack → can't use `d365_client.py` directly | Medium | They re-implement the HTTP call (~30 LoC); contract is documented |
| R2 | Resolver returns wrong deal when one contact is on multiple deals | Medium | Score formula tie-breaks on contact count + name; threshold cushion mitigates |
| R3 | SQLite-only tests miss DB-level constraint failures | Low | Pydantic whitelist catches most; promote test DB to Postgres in CI later |
| R4 | Bot calls our endpoints faster than DB can keep up (sub-second bursts) | Low | Existing connection pool handles it; add rate limiting in 1B if observed |
| R5 | Outlook webhook fires after meeting starts — bot must join "within 2 minutes" | Medium | Backend supports it (`POST` then immediate `PATCH .../status` to `joining`); execution is AI team's |
| R6 | D365 resolver can't reach the AIBackend's Postgres (different DB instances) | Low | They're separate; resolver only reads D365's own tables. Verified during build. |

---

## 6. What unblocks "story-level done"

Three checkboxes:

- [ ] **AI team's bot is calling all 4 endpoints from production code** (not just cURL)
- [ ] **End-to-end dev smoke:** seller creates a Teams meeting → bot joins as "Lenovo Sales Notes Bot" → row appears in `tbl_schedule_meetings` with `bot_status='joined'` and `opportunity_id` populated
- [ ] **PM signs off on resolver match-score threshold** (Q1 above)

Until all three are green, status remains: **"Backend complete, awaiting agent integration."**

---

## 7. Sprint review one-liner

> "Backend slice of US01 is shipped and tested — 4 endpoints, 11 automated tests, full handoff doc shared with the AI team. End-to-end story moves to done when the bot service starts calling these endpoints from production. Backend has no remaining blockers; everything pending is on the AI / Agent side or DevOps."

---

## 8. Change log

| Date | Change | Who |
|---|---|---|
| 2026-06-15 | Initial document — backend slice complete, AI team handoff in flight | Sanmay |

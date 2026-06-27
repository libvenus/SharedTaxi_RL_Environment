# US02 — Consent & Recording · Status & Dependencies

**Sprint:** 1A
**User Story:** Consent & Recording
**Last updated:** 2026-06-15
**Status legend:** ✅ done · 🟡 ready / awaiting trigger · ⏳ blocked · ❌ not started

---

## 1. What's done (backend)

| # | Deliverable | Status |
|---|---|---|
| B1 | Schema: `tbl_meeting_transcript` + `tbl_meeting_transcript_segment` (idempotent migration `sql/2026_06_us02_meeting_transcript.sql`) | ✅ |
| B2 | ORM models: `MeetingTranscript`, `MeetingTranscriptSegment` (`app/models/transcript.py`) | ✅ |
| B3 | Pydantic schemas: start / append / finalize / terminate / response (`app/schema/transcript.py`) | ✅ |
| B4 | Service layer: `start_transcript`, `append_segments`, `finalize_transcript`, `terminate_transcript`, `get_transcript` (`app/services/transcript_service.py`) | ✅ |
| B5 | Router: 5 endpoints under `/transcripts` (`app/api/transcripts.py`) | ✅ |
| B6 | D365 backend: `POST /api/contacts/resolve-by-emails` (Lenovo D365 Sales/`app/routers/contacts_resolver.py`) | ✅ |
| B7 | Schemas + version bump in D365 (0.14.0 → 0.15.0) | ✅ |
| B8 | `app/clients/d365_client.py` extended with `resolve_contacts_by_email()` + `ContactResolveResult` dataclass | ✅ |
| B9 | Pytest smoke suite — 9 tests (`tests/test_transcript_lifecycle.py`) | ✅ |
| B10 | conftest.py extended to register transcripts router + import transcript model | ✅ |
| B11 | README updated — repo layout, lifecycle diagram, all 5 cURL examples, both client wrappers | ✅ |
| B12 | AI-team handoff doc (`US02_BACKEND_HANDOFF_FOR_AI_TEAM.md`) | ✅ |
| B13 | This status doc (`US02_STATUS_AND_DEPENDENCIES.md`) | ✅ |
| B14 | Plan-of-action doc (`SPRINT_1A_US02_CONSENT_AND_RECORDING_PLAN.md`) | ✅ |

---

## 2. Pending — backend, non-blocking

| # | Item | Notes | Trigger |
|---|---|---|---|
| ~~BP1~~ | ~~Run `pytest -q` end-to-end and confirm 9/9 transcript tests pass~~ | ✅ **9 passed, 3 warnings in 0.46s** (2026-06-15) | Done |
| ~~BP2~~ | ~~Update `Lenovo D365 Sales/API_CONTRACT.md` §13 with the new contact-resolver endpoint~~ | ✅ Done — §13 added, version bumped to 0.15.0 | Done |
| BP3 | Sweeper for stale `in_progress` transcripts | Background job that flips long-stuck transcripts to `terminated_partial` with `terminated_reason='bot_failure'` | Sprint 1B |
| BP4 | Add segment dedup if duplicates show up in dev | Q3 in handoff doc; trivial to add `(meeting_id, start_time, speaker_email)` partial unique index | Reactive — only if AI team confirms retry behaviour |
| BP5 | Pagination on `GET /transcripts/{id}` | Becomes interesting only at >3000-segment meetings | Reactive |
| BP6 | Authn between bot ↔ AIBackend | Bearer token / mTLS | Sprint 1B (DevOps + bot team joint) |

---

## 3. Pending — AI / Bot team (blocks "story-level done")

| # | Item | Status | Notes |
|---|---|---|---|
| AI1 | Send `CONF_MSG_0004` in Teams chat on bot join | ❌ | Teams Bot SDK / Graph; backend only records what they sent |
| AI2 | Capture meeting audio | ❌ | Teams Bot SDK |
| AI3 | Speech-to-text + speaker diarization | ❌ | Azure Speech / Whisper / chosen STT engine |
| AI4 | Pre-enrich attendees via `POST /api/contacts/resolve-by-emails` ONCE at meeting start | ❌ | Wrapper ready in `d365_client.py` |
| AI5 | Tag each utterance with speaker info; default to `"Unknown Attendee"` for unidentified | ❌ | AC #3 — bot's local logic |
| AI6 | Batch-POST segments via `POST /transcripts/{id}/segments` continuously | ❌ | Batch size at bot's discretion |
| AI7 | Detect organiser-removes-bot event + send private "you can't kick me" notification to non-organisers | ❌ | Teams Bot SDK roles + DM |
| AI8 | On organiser-removed: call `POST /transcripts/{id}/terminate` with `reason='organizer_removed'` | ❌ | |
| AI9 | Detect natural meeting-end (everyone left / scheduled end) → leave automatically | ❌ | |
| AI10 | On meeting-end: call `POST /transcripts/{id}/finalize` with `overall_confidence_score` | ❌ | |
| AI11 | Compute `overall_confidence_score` (weighted average of segment confidences, suggested) | ❌ | Backend records whatever you send |
| AI12 | Handle backend errors gracefully (log + continue; don't crash bot on D365/AIBackend hiccups) | ❌ | Pattern shown in `d365_client.py` |

---

## 4. Pending — DevOps

| # | Item | Status | Notes |
|---|---|---|---|
| D1 | Apply `sql/2026_06_us02_meeting_transcript.sql` on dev Postgres | ❌ | Idempotent — safe to re-run |
| D2 | Apply `sql/2026_06_us01_meeting_lifecycle.sql` on dev Postgres (if not already) | ❌ | Pre-req for any transcript flow |
| D3 | Deploy AIBackend to a host the bot can reach over HTTP | ❌ | Same blocker as US01 |
| D4 | Deploy D365 Sales backend (v0.15.0+) to the host AIBackend reaches via `D365_BASE_URL` | ❌ | Bumped from 0.14.0 |
| D5 | Open firewall: bot → AIBackend port (for `/transcripts/*` calls) | ❌ | Same as US01 |
| D6 | Open firewall: AIBackend → D365 Sales port (for `/api/contacts/resolve-by-emails`) | ❌ | Same as US01 |
| D7 | Set `D365_BASE_URL` env var on AIBackend's deployment | ❌ | Defaults to `http://localhost:8000` |

---

## 5. Pending — Joint (cross-team)

| # | Item | Owner | Notes |
|---|---|---|---|
| J1 | Sign off on `CONF_MSG_0004` canonical text | AI team + PM | Q1 in handoff |
| J2 | Confirm batch size + retry behaviour for segment POSTs | AI team → backend | Q2 + Q3 in handoff |
| J3 | Decide `terminate` ↔ `bot_status='cancelled'` cascade | AI team + backend | Q4 in handoff |
| J4 | Decide audio-blob-pointer field on transcript table | AI team + backend | Q7 in handoff |
| J5 | E2E dry-run on dev | All | Real Teams meeting + real bot + real D365 + real AIBackend |

---

## 6. Open questions (carried from handoff doc §7)

| # | Question | Status | Owner |
|---|---|---|---|
| Q1 | Where does `CONF_MSG_0004` canonical text live? | open | PM |
| Q2 | Bot's preferred segment batch size? | open | AI team |
| Q3 | Bot's behaviour on network failure mid-meeting? | open | AI team |
| Q4 | Cascade `terminate` ↔ `bot_status='cancelled'`? | open | Joint |
| Q5 | Who computes `overall_confidence_score`? | answered (assumed: bot) | AI team to confirm |
| Q6 | "Unknown Attendee" — single bucket or per-voice? | open | AI team |
| Q7 | Audio-blob-pointer field on transcript table? | open | AI team |

---

## 7. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | High volume of segment POSTs degrades AIBackend latency | Medium | Watch DB write latency on dev; can switch to bulk-insert / partition by month if it bites |
| R2 | Duplicate segments accumulate from bot retries | Low (v1) | Q3 — if confirmed, add `(meeting_id, start_time, speaker_email)` dedup |
| R3 | `Lenovo D365 Sales` deployed at v0.14.0 → AIBackend's contact-resolver wrapper 404s | Medium | Coordinate D365 deploy to v0.15.0+ before AI team starts using `resolve_contacts_by_email` |
| R4 | Meeting → `bot_status='joined'` PATCH gets skipped → `POST /transcripts/` 400s | Medium | Bot's flow MUST PATCH before opening transcript; covered in handoff §1 |
| R5 | Audio file gets too large to keep on the bot's host | Out of scope | Bot's storage problem; we don't store audio |
| R6 | Compliance audit needs the EXACT chat-message timestamp | Low | We store `consent_sent_at` to ms precision via `TIMESTAMPTZ` |

---

## 8. Definition of "story-level done"

US02 is considered fully done when:

- [x] All backend deliverables (B1–B14) complete
- [ ] All DevOps tasks (D1–D7) complete
- [ ] All AI team tasks (AI1–AI12) complete
- [ ] Joint dry-run (J5) succeeds end-to-end on dev
- [ ] Open questions Q1–Q7 answered
- [ ] PM signs off after watching a real Teams meeting flow through the full pipeline:
      `bot joins → consent message in chat → /transcripts/ POSTed → segments stream in →
       /api/contacts/resolve-by-emails called → /finalize on natural end →
       transcript visible via GET endpoint with linked opportunity_id`

---

## 9. Definition of "backend done"

Backend's portion is done. Sanmay is unblocked from US02 work and can pick up:

- US03 (post-meeting AI summary) — design notes already at
  `Lenovo D365 Sales/MEETING_INTELLIGENCE_STORAGE_BACKEND_NOTES.md`
- Pending tasks from earlier conversations (Customer Information Phase 2,
  insight pills on activity cards, etc.)
- Or pair with AI team during their integration if they hit any
  contract-shape questions

---

## 10. Files changed (for the PR)

### `Lenovo-AIBackend`

```
sql/2026_06_us02_meeting_transcript.sql              (new)
app/models/transcript.py                             (new)
app/schema/transcript.py                             (new)
app/services/transcript_service.py                   (new)
app/api/transcripts.py                               (new)
app/clients/d365_client.py                           (extended)
app/main.py                                          (router registered)
tests/conftest.py                                    (transcripts router + model added)
tests/test_transcript_lifecycle.py                   (new — 9 tests)
README.md                                            (US02 sections added)
SPRINT_1A_US02_CONSENT_AND_RECORDING_PLAN.md         (new)
US02_BACKEND_HANDOFF_FOR_AI_TEAM.md                  (new)
US02_STATUS_AND_DEPENDENCIES.md                      (new)
```

### `Lenovo D365 Sales`

```
app/schemas.py                                       (3 new schemas appended)
app/routers/contacts_resolver.py                     (new)
app/main.py                                          (v0.14.0 → v0.15.0; router registered)
```

(Pending) `API_CONTRACT.md` — add §13 for the contact-resolver endpoint (BP2).

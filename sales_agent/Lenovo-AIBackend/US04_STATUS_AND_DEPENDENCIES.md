# US04 — Status & Dependencies

**Story:** Data Hygiene, Validation & Intelligent Alerts
**Sprint:** 1A backend
**Last updated:** Jun 16, 2026

---

## ✅ Done (Sprint 1A backend)

| Item | Where |
|---|---|
| Idempotent schema migration | `sql/2026_06_us04_data_task.sql` |
| ORM model + canonical `task_kind` constants | `app/models/data_task.py` |
| 9 Pydantic schemas | `app/schema/data_task.py` |
| Config additions (`DATA_TASK_STALE_DAYS`, scan page size, default severity) | `app/core/config.py` + `.env.example` |
| D365 client wrappers (`list_active_opportunities`, `fetch_opportunity_risks`) | `app/clients/d365_client.py` |
| Service layer — task CRUD with SELECT-then-INSERT idempotency + IntegrityError fallback | `app/services/data_task_service.py` |
| 4 detectors (`detect_past_close_date`, `detect_zero_or_missing_value`, `detect_stale_activity`, `detect_risk_flags`) | `app/services/data_task_detectors.py` |
| 6 API endpoints under `/api/data-tasks` | `app/api/data_tasks.py` |
| Router registered in `app/main.py` | ✅ |
| Daily-scan CLI (`python -m app.jobs.scan_data_tasks [--limit] [--dry-run]`) | `app/jobs/scan_data_tasks.py` |
| 14 smoke tests | `tests/test_data_task_lifecycle.py` |
| README §4 — state diagram + cURL for every endpoint + scan-job invocation | `README.md` |
| AI team handoff doc | `US04_BACKEND_HANDOFF_FOR_AI_TEAM.md` |
| Plan-of-action doc with locked decisions | `SPRINT_1A_US04_DATA_HYGIENE_PLAN.md` |

---

## ⏳ Pending — backend tasks the user (Sanmay) still owes

| BP# | Task | Priority |
|---|---|---|
| BP1 | Run `pytest -q tests/test_data_task_lifecycle.py` end-to-end and confirm 14/14 pass + total `46/46` (8 + 9 + 15 + 14) across the suite | High |
| BP2 | Apply `sql/2026_06_us04_data_task.sql` to the dev Postgres (`psql -f`) and verify the partial UNIQUE index landed | High |
| BP3 | Smoke-test the daily scan against the real D365 dev instance: `python -m app.jobs.scan_data_tasks --limit 5 --dry-run` | High |
| BP4 | Update `Lenovo D365 Sales/API_CONTRACT.md` if D365's `GET /api/opportunities` response shape changes during integration testing — the field projection in `_parse_*` helpers is the contract surface | Medium |
| BP5 | Hook the scan job into a cron / scheduled task (DevOps will own; we provide the invocation) | Medium (S1B) |
| BP6 | Pagination on `GET /api/data-tasks` once a seller has > 50 open tasks (the query already supports it; just need the FE to send `offset` / `limit`) | Reactive |
| BP7 | Authn between AI team / FE → AIBackend (bearer token) | S1B |
| BP8 | Unified `lvo_audit_log`-style cross-cutting audit table — for now we keep audit fields on the `tbl_data_task` row itself | S1B / Joint |

---

## ❌ Deferred to Sprint 1B+ — explicit list (the L1–L10 from the plan)

The scope cut from Sprint 1A. Each has a one-liner reason; raise a
ticket against the listed owner when ready to schedule.

| L# | Item | Owner | Reason for deferral |
|---|---|---|---|
| L1 | Inline-validation polish — plain-language messages + ERR_MSG_xxxx codes on save (AC #1) | D365 Sales backend (Namisha) | Lives in `deals_write.py`, `account_contacts.py`, `contacts.py`; PM needs to sign off on the copy first |
| L2 | Multi-signal duplicate-opportunity check on save (AC #6, #7, ERR_MSG_0019) | D365 Sales backend | Replaces existing `DUPLICATE_NAME` 409 with 5-signal check + Keep-both/Discard payload; needs FE coordination |
| L3 | Multi-signal duplicate-opportunity daily scan | D365 Sales backend → AIBackend scan | Pairs with L2; scan version sits on top of save-time logic |
| L4 | Mandatory-fields-per-stage admin config + enforcement (AC #8, #9) | D365 Sales backend | Needs new admin-config table + admin UI + PM sign-off on what's mandatory |
| L5 | Territory mismatch detector | AIBackend | Needs verification that `lvo_territoryid` exists on opportunity in the dev dump |
| L6 | Email-domain mismatch detector | AIBackend | Needs allowlist for personal-email providers (gmail / outlook / hotmail / etc.) |
| L7 | Stage-dwell-time anomaly detector | AIBackend | Needs per-stage expected-dwell-time config |
| L8 | Parent-child account suggestion | AIBackend | Needs `pg_trgm` extension + DevOps approval; high false-positive risk |
| L9 | Duplicate contact (fuzzy) on same account | AIBackend / D365 | Inline exact-match exists; fuzzy needs trigram |
| L10 | AI-suggested-update To-Do tasks (entity / field / current / suggested / confidence / why grid display) | AIBackend (we just persist) | Depends on the **Post-Meeting CRM Update Suggestions** user story which doesn't exist yet — when it lands, our `POST /api/data-tasks` already accepts the shape |

---

## 🤝 AI team tasks

See `US04_BACKEND_HANDOFF_FOR_AI_TEAM.md` for the full integration spec.

| AI# | Task | Status |
|---|---|---|
| AI1 | Wire transcript-signal NLP pipeline to `POST /api/data-tasks` | ❌ Pending AI team |
| AI2 | Use canonical `task_kind` strings from §3 of the handoff doc | ❌ |
| AI3 | Populate `evidence_ref` with `transcript_segment_id=<uuid>` | ❌ |
| AI4 | Send calibrated `confidence` per signal | ❌ |
| AI5 | Render `evidence_text` as a complete plain-language sentence | ❌ |
| AI6 | Treat `was_existing: true` as success (not retry) | ❌ |
| AI7 | Answer Q1–Q5 in §12 of the handoff doc | ❌ |

---

## 🛠 DevOps tasks

| D# | Task | Status |
|---|---|---|
| D1 | Apply `sql/2026_06_us04_data_task.sql` to staging + prod Postgres | ❌ |
| D2 | Schedule `python -m app.jobs.scan_data_tasks` daily at 02:00 (or per PM cadence) | ❌ |
| D3 | Wire the scan-job's non-zero exit (any-error) to the alerting channel | ❌ |
| D4 | Configure firewall rule: AIBackend → D365 Sales backend port 8000 (existing US01 rule should already cover this) | ❌ |
| D5 | Set the new env vars on staging + prod (`DATA_TASK_STALE_DAYS`, `DATA_TASK_SCAN_PAGE_SIZE`, `DATA_TASK_DEFAULT_SEVERITY`) | ❌ |

---

## 🤔 Joint / open questions

| Q# | Question | Owner | Why it matters |
|---|---|---|---|
| Q1 | Canonical strings for transcript signals — locked in §3 of handoff doc, but does AI team agree? | AI + Backend | Wrong strings break FE grouping |
| Q2 | Do AI team transcript-signal POSTs need authn / rate-limit? | Joint | Decide before public deploy |
| Q3 | What does `evidence_ref` look like when there is no segment id (cross-meeting aggregation)? | AI | Format consistency |
| Q4 | When the seller resolves a task, should backend write `resolved_value` back to D365, or is that the FE's job? | Joint | v1 = FE writes; backend just logs the metadata |
| Q5 | Cron cadence for the daily scan — daily 02:00? Twice daily? | DevOps + PM | Affects scan-job design (incremental vs full) |
| Q6 | When parent opp is deleted, should open tasks on it auto-close? | Joint | Decides cascade contract |
| Q7 | Multiple signals on the same `(entity, kind)` from one transcript — aggregate into one task or send N? | AI + PM | Today: only the first lands; subsequent are no-ops |

---

## 🧪 Test counts (running tally)

| Story | File | Count | Status |
|---|---|---|---|
| US01 | `test_meeting_lifecycle.py` | 8 | ✅ |
| US02 | `test_transcript_lifecycle.py` | 9 | ✅ |
| US03 | `test_consent_capture_lifecycle.py` | 15 | ✅ |
| **US04** | `test_data_task_lifecycle.py` | **14** | **⏳ awaiting BP1 run** |
| **Total** | | **46** | |

---

## 🔧 Critical risks for US04

| R# | Risk | Mitigation |
|---|---|---|
| R1 | A buggy transcript signal could spam the seller's queue | Idempotency contract (already in place) — same `(entity, kind)` is a no-op |
| R2 | A dismissed task could be incorrectly suppressing a NEW kind of signal | Each signal has its own `task_kind`; suppression is per-kind, so dismissing a `transcript_signal_close_date_different` does NOT suppress a `transcript_signal_quantity_different` for the same opp |
| R3 | The daily scan could create thousands of tasks on a fresh tenant (legacy zero-value opps) | Run with `--limit 50 --dry-run` first; PM may want a one-time bulk-suppress flag |
| R4 | D365's `GET /api/opportunities` payload shape evolves and breaks our wrapper | Wrapper does best-effort parse with `_parse_uuid_or_none` etc., NOT strict validation; bad rows are silently skipped (logged) rather than aborting the scan |
| R5 | Race condition on concurrent POST /api/data-tasks for the same `(entity, kind)` — could create duplicates if SQLite test path replaces production Postgres | The Postgres partial UNIQUE index closes the race; SQLite tests pin the SELECT-then-INSERT happy path; both paths are exercised |

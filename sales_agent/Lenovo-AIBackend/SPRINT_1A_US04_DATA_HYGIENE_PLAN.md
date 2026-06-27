# Sprint 1A · User Story 04 — Data Hygiene, Validation & Intelligent Alerts · Backend Plan of Action

**Repo:** `Lenovo-AIBackend` (primary); `Lenovo D365 Sales` only when truly needed (all S1A scope is AIBackend)
**Owner:** Sanmay (backend) · pairing with Namisha
**Effort:** ~12 hrs (S1A subset)
**Status:** Plan locked, ready to implement.
**Linked docs:**
- [SPRINT_1A_US01_JOINING_MEETINGS_PLAN.md](./SPRINT_1A_US01_JOINING_MEETINGS_PLAN.md) — meeting lifecycle (transcript signal source)
- [SPRINT_1A_US02_CONSENT_AND_RECORDING_PLAN.md](./SPRINT_1A_US02_CONSENT_AND_RECORDING_PLAN.md) — transcript pipeline
- [SPRINT_1A_US03_CONSENT_CAPTURE_PLAN.md](./SPRINT_1A_US03_CONSENT_CAPTURE_PLAN.md) — pre-meeting consent

---

## 1. Reality-check on US04 scope

US04 is the **biggest story we've taken on** so far — 14 distinct scenarios, 11 acceptance criteria. A literal full-scope implementation would be ~30–40 hrs across two sprints.

We're going to ship the **foundation + 4 high-value detectors** in Sprint 1A (~12 hrs) and explicitly defer the rest to Sprint 1B+.

### What's IN Sprint 1A

| # | Scenario | Why it's in S1A |
|---|---|---|
| **F0** | Task-queue infrastructure — `tbl_data_task` table, ORM, schemas, full CRUD, audit trail self-contained on the row | Every other detector / signal needs this. Nothing ships without it. |
| **F1** | Generic transcript-signal entry point (`POST /api/data-tasks`) | AC #3. Lets the AI team start posting alerts the moment their NLP pipeline is ready. Zero NLP work on our side. |
| **D1** | Past close date + still open detector | Cheap deterministic SQL — pure `closeprobability + close_date + statecode` check. |
| **D2** | Zero / missing deal value detector | Daily-scan side of an existing inline check. Already-validated rules; we just persist. |
| **D3** | Stale activity (>30d, configurable) detector | High-value seller signal; reuses `last_activity_date`. |
| **D4** | Risk-flag → To-Do task materialisation | Reuses `GET /api/opportunities/{id}/risks` from D365 Sales (already computes 13 risks). One detector → many task kinds. |
| **F2** | Daily-scan CLI job | Drives D1–D4 idempotently. Mirrors the `recalc_*` job pattern from D365 Sales. |
| **F3** | Resolve / dismiss endpoints with self-contained audit | AC #4, #5, #7, #10. Every action gets timestamped + actor + note. |

### What's DEFERRED to Sprint 1B+

| # | Scenario | Why deferred |
|---|---|---|
| L1 | Inline-validation polish: plain-language messages + ERR_MSG_xxxx codes on save (AC #1) | Lives in **D365 Sales** (`deals_write`, `account_contacts`, `contacts`). PM needs to sign off on copy first. |
| L2 | Multi-signal duplicate-opportunity check on save (AC #6, #7, ERR_MSG_0019) | Lives in D365 Sales `deals_write.py`. Replaces the existing `DUPLICATE_NAME` 409 with a 5-signal check. Non-trivial — needs FE coordination on the new `Keep both / Discard` payload shape. |
| L3 | Multi-signal duplicate-opportunity daily scan | Pairs with L2; scan version is built on top of save-time logic. |
| L4 | Mandatory-fields-per-stage admin config + enforcement (AC #8, #9) | Needs a new admin-config table + admin UI. PM + admin sign-off needed. |
| L5 | Territory mismatch detector | Needs verification that `lvo_territoryid` exists on opportunity in the dev dump. |
| L6 | Email-domain mismatch detector | Needs allowlist for personal-email providers (gmail, outlook, hotmail). |
| L7 | Stage-dwell-time anomaly detector | Needs per-stage expected-dwell-time config. |
| L8 | Parent-child account suggestion | Needs `pg_trgm` extension + DevOps approval. Fuzzy matching with high false-positive risk. |
| L9 | Duplicate contact (fuzzy) on same account | Inline exact-match already exists in `contact_validation.py`. Fuzzy version needs trigram. |
| L10 | AI-suggested-update To-Do tasks (entity / field / current / suggested / confidence / why) | Depends on the **Post-Meeting CRM Update Suggestions** user story, which doesn't exist yet. Once it lands, our `POST /api/data-tasks` already accepts the shape — no extra backend work. |

---

## 2. Decisions locked

| Decision | Choice | Why |
|---|---|---|
| Storage location | **AIBackend** | User directive. Sits next to transcript pipeline, talks to D365 via `d365_client`. |
| Audit log | **Self-contained on `tbl_data_task` row** (resolved_at / dismissed_at / dismissal_note / actor_id) — for S1A | No cross-repo audit table yet. AC #10 satisfied per-task; cross-cutting audit can come later. |
| Daily-scan trigger | **Idempotent CLI job** runnable via `python -m app.jobs.scan_data_tasks` | Mirrors `Lenovo D365 Sales/app/jobs/snapshot_kpis.py` pattern. Cron later in DevOps. |
| Scan idempotency | **Unique index on `(entity_kind, entity_id, task_kind, status='open')`** | Re-running scan on the same data → 0 new tasks. Critical for AC #5 (suppress dismissed pairs). |
| Suppression after dismissal | **Dismissed task + same `(entity, kind)` re-detected → skip create** | AC #5: "intentional mismatches can be dismissed with a note and future alerts for that pair are suppressed." Implemented via the unique index above (dismissed rows still exist, blocking re-create). |
| Transcript signals | **Generic `POST /api/data-tasks`** | AC #3. AI team owns NLP; we just persist. No `transcript_signal_*` enum baked in — `task_kind` is free-form TEXT. |
| Task ordering | **Confidence DESC then severity DESC then created_at ASC** | AC: "ordered by confidence — High first, then Medium, then Low." |
| Portfolio scoping | **`?owner_id=` filter on GET (FE supplies seller's user_id)** | AC #11. We don't have user-auth context here yet, so we trust the query param. Hardening = JWT in S1B. |
| Test scope | **~14 smoke tests** (matches US01/02/03 bar) | Same coverage discipline. |
| Repo footprint | **Single-repo story** (no D365 Sales changes for S1A) | Keeps blast radius small. L1, L2, L4 explicitly defer cross-repo work. |

---

## 3. Architecture

```
   ┌──────────────────────────────────────────────────────────┐
   │ Sources of data-task creation                            │
   ├──────────────────────────────────────────────────────────┤
   │ (1) AI team — post-meeting NLP detects transcript signal │
   │     POST /api/data-tasks  body={entity_kind, entity_id,  │
   │                                  task_kind, evidence_*, │
   │                                  confidence, severity}  │
   │                                                           │
   │ (2) Daily scan (cron, S1B) / manual `/scan` POST          │
   │     Pulls D365 active opps via d365_client, runs 4       │
   │     detectors (D1–D4), POSTs each finding internally.    │
   │                                                           │
   │ (3) FE inline validators (S1B) — same generic endpoint   │
   └──────────────────────────────────────────────────────────┘
                         │
                         ▼
   ┌──────────────────────────────────────────────────────────┐
   │ tbl_data_task (UNIQUE on entity_kind+entity_id+task_kind │
   │                       WHERE status='open')              │
   │  → idempotent: re-detection of an open task is a no-op  │
   │  → idempotent: re-detection of a DISMISSED task is also │
   │                a no-op (suppression via unique index)   │
   └──────────────────────────────────────────────────────────┘
                         │
                         ▼
   ┌──────────────────────────────────────────────────────────┐
   │ Seller's To-Do List (FE — out of scope here)             │
   │   GET /api/data-tasks?ownerId=<seller>&status=open       │
   │   ordered by confidence DESC, severity DESC, created ASC │
   └──────────────────────────────────────────────────────────┘
                         │
                ┌────────┴────────┐
                ▼                 ▼
   POST /resolve   POST /dismiss
   (writes audit fields on the row; cross-repo write to D365 is
    FE's job for v1 — backend only logs the resolution metadata)
```

---

## 4. Phased implementation

### Phase 1 — Schema (idempotent migration) · ~45 min

`sql/2026_06_us04_data_task.sql` — one new table + 1 partial unique index + 4 query indexes.

#### `tbl_data_task`

| Column | Type | Notes |
|---|---|---|
| `task_id` | UUID PK | |
| `owner_id` | UUID NOT NULL | Seller — for portfolio scoping |
| `entity_kind` | TEXT NOT NULL CHECK | `account` / `contact` / `opportunity` |
| `entity_id` | UUID NOT NULL | FK-by-convention to D365 |
| `task_kind` | TEXT NOT NULL | `past_close_date` / `zero_or_missing_value` / `stale_activity` / `risk_flag` / `transcript_signal_*` / etc. (free-form so AI team doesn't need a backend release for new signals) |
| `severity` | TEXT NOT NULL CHECK | `high` / `medium` / `low` (default `medium`) |
| `status` | TEXT NOT NULL CHECK | `open` / `resolved` / `dismissed` (default `open`) |
| `field_name` | TEXT | Optional — which field to fix |
| `current_value` | TEXT | Display value (no implicit type) |
| `suggested_value` | TEXT | What the system / AI suggests |
| `confidence` | TEXT CHECK | `high` / `medium` / `low` / NULL (NULL for deterministic detectors) |
| `evidence_ref` | TEXT | E.g. `transcript_segment_id=<uuid>` or `scan_run=<ts>` |
| `evidence_text` | TEXT NOT NULL | The plain-language "why" — REQUIRED (AC #3 grounding ref) |
| `created_by_source` | TEXT NOT NULL CHECK | `transcript` / `scan` / `inline` / `manual` |
| `dismissal_note` | TEXT | NULL until dismissed |
| `resolved_value` | TEXT | The value the seller wrote back |
| `actor_id` | UUID | Who resolved/dismissed |
| `resolved_at` | TIMESTAMPTZ | |
| `dismissed_at` | TIMESTAMPTZ | |
| `created_at` / `updated_at` | TIMESTAMPTZ | Server defaults |

**Indexes:**
- Partial UNIQUE on `(entity_kind, entity_id, task_kind)` `WHERE status IN ('open','dismissed')` — drives idempotency + dismissal-suppression
- `(owner_id, status)` — FE queue load
- `(status)` — admin / metrics
- `(entity_kind, entity_id)` — show all tasks for one record

---

### Phase 2 — ORM + Pydantic schemas · ~45 min

**Files:**
- `app/models/data_task.py` — `DataTask` ORM + 4 enum tuples (TASK_STATUS_VALUES, ENTITY_KIND_VALUES, SEVERITY_VALUES, CONFIDENCE_VALUES)
- `app/schema/data_task.py` — 9 schemas:
  - `DataTaskCreate` (used by AI team + scan job)
  - `DataTaskOut` (full record)
  - `DataTaskListResponse` (paginated)
  - `DataTaskResolveRequest` / `DataTaskResolveResponse`
  - `DataTaskDismissRequest` / `DataTaskDismissResponse`
  - `ScanRunRequest` / `ScanRunResponse`

---

### Phase 3 — Config additions · ~15 min

`app/core/config.py`:
```python
DATA_TASK_STALE_DAYS = int(os.getenv("DATA_TASK_STALE_DAYS", "30"))
DATA_TASK_SCAN_PAGE_SIZE = int(os.getenv("DATA_TASK_SCAN_PAGE_SIZE", "100"))
DATA_TASK_DEFAULT_SEVERITY = os.getenv("DATA_TASK_DEFAULT_SEVERITY", "medium")
```

`.env.example` extension.

---

### Phase 4 — D365 client extension · ~1 hr

`app/clients/d365_client.py`:

```python
def list_active_opportunities(
    *, page_size: int = 100, base_url=None, timeout=None
) -> Iterable[OpportunityScanRow]:
    """Generator that paginates GET /api/opportunities?status=Open
       and yields a dataclass per row containing close_date,
       estimated_value, last_activity_date, ownerId, accountId."""

def fetch_opportunity_risks(
    opportunity_id: UUID, *, base_url=None, timeout=None
) -> list[OpportunityRisk]:
    """GET /api/opportunities/{id}/risks — passthrough wrapper."""
```

D365 already exposes both endpoints; we just add typed wrappers.

---

### Phase 5 — Service layer · ~3 hrs

`app/services/data_task_service.py`:

```python
def create_task(db, payload) -> DataTask:
    """Idempotent: if open task with same (entity, kind) exists, return it.
       If dismissed task with same (entity, kind) exists, do NOT re-create
       (suppression). Returns existing row in both cases."""

def list_tasks(db, *, owner_id, status, kind, entity_id, limit, offset) -> tuple[list[DataTask], int]:
    """Filtered + ordered (confidence DESC, severity DESC, created ASC)."""

def get_task(db, task_id) -> DataTask | None: ...

def resolve_task(db, task_id, *, resolved_value, actor_id) -> DataTask:
    """Sets status='resolved', resolved_at, resolved_value, actor_id.
       Idempotent — resolving an already-resolved task returns it."""

def dismiss_task(db, task_id, *, note, actor_id) -> DataTask:
    """Sets status='dismissed', dismissed_at, dismissal_note, actor_id.
       Note is REQUIRED (AC #4)."""
```

`app/services/data_task_detectors.py`:

```python
def detect_past_close_date(opp: OpportunityScanRow, today: date) -> DataTaskCreate | None:
    """Active stage AND close_date < today → high severity."""

def detect_zero_or_missing_value(opp: OpportunityScanRow) -> DataTaskCreate | None:
    """estimated_value IS NULL OR == 0 → medium severity."""

def detect_stale_activity(opp, today, stale_days) -> DataTaskCreate | None:
    """now - last_activity_date > stale_days → medium severity."""

def detect_risk_flags(
    opp_id, risks: list[OpportunityRisk], owner_id
) -> list[DataTaskCreate]:
    """One DataTaskCreate per risk, severity derived from category.
       evidence_text = risk.message verbatim."""
```

---

### Phase 6 — API endpoints · ~1.5 hrs

`app/api/data_tasks.py`, mounted at `/api/data-tasks`:

```
POST   /api/data-tasks                     # create (AI team / inline / scan internal)
GET    /api/data-tasks                     # list with filters
GET    /api/data-tasks/{task_id}           # detail
POST   /api/data-tasks/{task_id}/resolve   # mark resolved
POST   /api/data-tasks/{task_id}/dismiss   # dismiss with required note
POST   /api/data-tasks/scan                # admin trigger for the daily scan (auth = none for v1)
```

Error surface:
- `404` — unknown task
- `409` — re-create attempt (returns existing row + 200, not 409 — idempotency)
- `422` — schema validation (esp. missing `dismissal_note` on dismiss)

---

### Phase 7 — Daily-scan CLI job · ~1.5 hrs

`app/jobs/scan_data_tasks.py`:

```
python -m app.jobs.scan_data_tasks [--limit N] [--dry-run]
```

Algorithm (idempotent, resumable):
1. Iterate `d365_client.list_active_opportunities()`
2. For each opp: run D1, D2, D3 detectors → call `create_task` for each finding
3. Then call `d365_client.fetch_opportunity_risks(opp_id)` → run D4 (one task per risk)
4. Print summary: `tasks_created`, `tasks_skipped_existing`, `tasks_skipped_dismissed`, `errors`
5. `--dry-run` flag: log what would be created, write nothing

---

### Phase 8 — Tests · ~2.5 hrs

14 smoke tests in `tests/test_data_task_lifecycle.py`:

1. `POST /api/data-tasks` happy path — creates an open task
2. `POST /api/data-tasks` is idempotent — same `(entity, kind)` → 200 with existing task
3. `POST /api/data-tasks` rejects when matching dismissed task exists (suppression) → 200 with dismissed row
4. `POST /api/data-tasks` rejects missing `evidence_text` → 422
5. `GET /api/data-tasks?ownerId=X` filters by owner
6. `GET /api/data-tasks` ordering — high confidence first, then severity, then age
7. `GET /api/data-tasks/{id}` returns full record
8. `GET /api/data-tasks/{id}` 404 on unknown
9. `POST /resolve` happy path — sets resolved_at, resolved_value, actor
10. `POST /resolve` is idempotent — resolving twice OK
11. `POST /dismiss` rejects empty note (422)
12. `POST /dismiss` happy path — sets dismissed_at, note, actor
13. Detector unit: `detect_past_close_date` + `detect_zero_or_missing_value` + `detect_stale_activity` happy paths
14. Detector unit: `detect_risk_flags` returns one task per risk, severity derived correctly

---

### Phase 9 — Docs + handoff · ~1.5 hrs

- README extension — add §4 "Data Hygiene Tasks" with state diagram + cURL examples for all 6 endpoints + scan-job invocation
- `US04_BACKEND_HANDOFF_FOR_AI_TEAM.md` — full endpoint reference + sample bot/AI flow + exact `task_kind` strings + `evidence_ref` format + open questions
- `US04_STATUS_AND_DEPENDENCIES.md` — done / pending / blocked tracker; explicit list of L1–L10 deferred items
- `.env.example` — new config keys

---

## 5. Effort summary

| Phase | What | Effort |
|---|---|---|
| 1 — Schema | 1 new table + indexes | 45 min |
| 2 — ORM + Pydantic | 9 schemas + 1 model | 45 min |
| 3 — Config | 3 env vars | 15 min |
| 4 — D365 client | 2 new wrappers + dataclasses | 1 hr |
| 5 — Service layer | 5 service fns + 4 detectors | 3 hrs |
| 6 — API | 6 routes | 1.5 hrs |
| 7 — Scan job | CLI + idempotency | 1.5 hrs |
| 8 — Tests | 14 smoke tests | 2.5 hrs |
| 9 — Docs | README + 2 docs + .env | 1.5 hrs |
| **Total** | | **~12 hrs** |

---

## 6. Out-of-scope (for S1A — already accounted in §1 deferral list)

See **L1–L10** in §1. Each has a one-liner reason for deferral so PM / AI-team know what they're getting and what's coming later.

---

## 7. Open questions for AI team / PM

| # | Question | Why | Owner |
|---|---|---|---|
| Q1 | Canonical strings for `task_kind` (esp. transcript signals: `transcript_signal_close_date`, `transcript_signal_headcount`, etc.) | We pin them in the handoff doc so AI team's POSTs go to the right query buckets | AI + Backend |
| Q2 | Does the AI team produce `confidence` per signal, or is everything `medium` until model confidence is plumbed? | Affects ordering in GET — if everything is `medium` we just sort by severity then age | AI |
| Q3 | What goes in `evidence_ref` for transcript signals — raw transcript_segment_id, or an opaque AI-team handle? | Decides how the FE renders the "View grounding" link | AI |
| Q4 | When a user resolves a task, do they also want backend to write the `resolved_value` back to D365, or is that the FE's job? | Cross-repo write coupling. Default v1 = FE writes, we just log. | Joint |
| Q5 | What `owner_id` should the daily-scan job use when it creates a task? Pull `owninguser` from D365 opp? | Confirm field exists in opp scan payload | Joint |
| Q6 | Cron cadence for scan job — daily 02:00? | DevOps | DevOps |
| Q7 | When parent opp is deleted (US01-style cascade later), should open tasks on it auto-close? | Decides cascade contract | Joint |

---

## 8. Acceptance criteria for "this story is done from backend's S1A side"

- [ ] Migration applied to dev Postgres
- [ ] All 6 endpoints work via cURL
- [ ] Idempotency: second call to `POST /api/data-tasks` with same `(entity, kind)` returns existing row, not a duplicate
- [ ] Suppression: `POST` with same `(entity, kind)` as a dismissed task does NOT recreate
- [ ] `POST /dismiss` rejects empty note
- [ ] Daily scan generates tasks for D1–D4 on a seeded dev opp
- [ ] All 14 smoke tests pass
- [ ] README has §4 Data-Hygiene with state diagram + cURL for every new endpoint + scan-job invocation
- [ ] AI team has `US04_BACKEND_HANDOFF_FOR_AI_TEAM.md` with locked `task_kind` enum
- [ ] L1–L10 explicitly listed in `US04_STATUS_AND_DEPENDENCIES.md`
- [ ] Repo lints clean

---

## 9. Order of work (for the implementation pass)

1. Phase 1 (schema)
2. Phase 3 (config) — needed by detectors
3. Phase 2 (ORM + schemas)
4. Phase 4 (D365 client extension)
5. Phase 5 (service layer + detectors)
6. Phase 6 (API endpoints)
7. Phase 7 (scan job)
8. Phase 8 (tests)
9. Phase 9 (docs + handoff)

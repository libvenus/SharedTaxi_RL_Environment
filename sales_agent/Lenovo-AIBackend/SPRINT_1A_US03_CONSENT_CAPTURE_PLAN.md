# Sprint 1A · User Story 03 — Consent Capture · Backend Plan of Action

**Repo:** `Lenovo-AIBackend` (single-repo story — no D365 changes)
**Owner:** Sanmay (backend) · pairing with Namisha
**Effort:** ~8 hrs
**Status:** Plan locked, ready to implement.
**Linked docs:**
- [SPRINT_1A_US01_JOINING_MEETINGS_PLAN.md](./SPRINT_1A_US01_JOINING_MEETINGS_PLAN.md) — meeting lifecycle (cancellation cascade target)
- [SPRINT_1A_US02_CONSENT_AND_RECORDING_PLAN.md](./SPRINT_1A_US02_CONSENT_AND_RECORDING_PLAN.md) — in-meeting consent (the fallback mechanism)

---

## 1. What this story actually needs from backend

The story is about **pre-meeting consent emails** to external customers, with opt-out tracking. Most of the heavy lifting is the bot/AI team's (rendering, sending, displaying notifications). Backend's role:

1. **Schedule** — given a meeting, decide whether the consent window is still open and create per-recipient records
2. **Audit trail** — track every send attempt, delivery status, retry, and opt-out for compliance
3. **Opt-out endpoint** — public-facing URL the participant clicks; we record it + cancel the bot
4. **Bot-cancellation cascade** — when ANY participant opts out, flip `tbl_schedule_meetings.bot_status` to `cancelled` with reason `participant_opted_out`
5. **Status aggregation** — bot calls this right before joining; if any opt-out, bot doesn't join

Acceptance-criteria mapping:

| AC | Backend role |
|---|---|
| 1. External customers only | Filter attendees by `INTERNAL_EMAIL_DOMAINS` config |
| 2. No response = consent approved | Implicit — opt-out is an active action; absence = consent. No code |
| 3. Window passed → skip email | `schedule_consent_emails` returns `should_send=false`, fallback flag |
| 4. From seller's name via Lenovo Sales Assistant | Expose seller name + system address; AI team renders the from header |
| 5. Email contains title / time / seller / opt-out link | Backend exposes all template vars incl. signed opt-out URL |
| 6. Opt-out link active until meeting starts | `record_opt_out` rejects clicks after `meeting_start_time` |
| 7. Any opt-out → bot doesn't join, no override | Cascade `bot_status='cancelled'` + `bot_status_reason='participant_opted_out'` |
| 8. Opt-out participant sees confirmation (SUCC_MSG_0010) | Endpoint returns minimal HTML page |
| 9. Seller notified immediately on opt-out (INF_MSG_0001) | Stamp `seller_notified_at` for future notifications service |
| 10. Failed delivery → retry once after 10 min, then fallback | `next_retry_at` field + `due-for-retry` queue endpoint |

---

## 2. Decisions locked

| Decision | Choice | Why |
|---|---|---|
| Storage location | **AIBackend** | Sits next to `tbl_schedule_meetings`; in-process cascade to `bot_status` |
| Internal vs external | **Config-driven domain list** (`INTERNAL_EMAIL_DOMAINS`) | Simple + per-tenant deployable |
| Email rendering | **AI team renders** | Backend = data; bot = side effects |
| Opt-out response | **Server-rendered HTML page** | Self-contained, no FE dependency |
| Retry mechanism | **Bot polls `/due-for-retry`** | Backend stays stateless, no scheduler infra |
| Token format | **`secrets.token_urlsafe(32)`** | 256 bits, URL-safe, unguessable |
| Test scope | **Match US02 (~11 smoke tests)** | Same coverage bar |

---

## 3. Architecture

```
   Meeting detected (US01: tbl_schedule_meetings row exists)
                           │
                           ▼
   ┌──────────────────────────────────────────────────────────┐
   │ POST /consent-emails/schedule                            │
   │  body: { meeting_id }                                    │
   │  → loads meeting, filters attendees by external domain   │
   │  → IF (start_time - now) >= consent_window_minutes:      │
   │      create one row per external recipient with a fresh  │
   │      opt-out token                                        │
   │      return { should_send: true, recipients: [...] }     │
   │  → ELSE:                                                  │
   │      return { should_send: false, fallback: 'in_meeting' }│
   └────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
              Bot / Email service renders + sends via Graph API
                        │
                        ▼
   ┌──────────────────────────────────────────────────────────┐
   │ PATCH /consent-emails/{consent_id}/delivery              │
   │  body: { status: 'sent' | 'failed', failure_reason? }    │
   │  → on failed + attempt < 2:  next_retry_at = now + 10m   │
   │  → on failed + attempt >= 2: delivery_status =           │
   │                              'fallback_to_in_meeting'    │
   └──────────────────────────────────────────────────────────┘

   Some time later — participant clicks opt-out link:

   ┌──────────────────────────────────────────────────────────┐
   │ GET /consent-emails/opt-out/{token}                      │
   │  → validates token, checks meeting hasn't started        │
   │  → records opt-out (timestamp, ip)                       │
   │  → cascades: tbl_schedule_meetings.bot_status='cancelled'│
   │              with reason='participant_opted_out'          │
   │  → stamps seller_notified_at                             │
   │  → returns HTML confirmation (SUCC_MSG_0010)             │
   └──────────────────────────────────────────────────────────┘

   Bot, right before joining the meeting:

   ┌──────────────────────────────────────────────────────────┐
   │ GET /meetings/{meeting_id}/consent-status                │
   │  → returns aggregated state:                             │
   │      any_opted_out, opt_out_count,                        │
   │      consent_mechanism: 'pre_meeting_email' |             │
   │                          'in_meeting_chat' | 'mixed',     │
   │      pending_retries                                      │
   │  → bot reads this; any_opted_out=true → don't join        │
   └──────────────────────────────────────────────────────────┘

   Periodic polling for retries:

   ┌──────────────────────────────────────────────────────────┐
   │ GET /consent-emails/due-for-retry                        │
   │  → returns rows where next_retry_at <= now() AND         │
   │    delivery_status = 'failed' AND attempt_count < 2      │
   │  → bot picks them up, re-sends, PATCHes /delivery again  │
   └──────────────────────────────────────────────────────────┘
```

---

## 4. Phased implementation

### Phase 1 — Schema (idempotent migration)

`sql/2026_06_us03_consent_email.sql` — one new table.

#### `tbl_meeting_consent_email`

| Column | Type | Notes |
|---|---|---|
| `consent_id` | UUID PK | |
| `meeting_id` | UUID NOT NULL | FK-by-convention to `tbl_schedule_meetings` |
| `recipient_email` | TEXT NOT NULL | Lower-cased on write |
| `recipient_name` | TEXT | Optional — bot may pass display name |
| `opt_out_token` | TEXT NOT NULL UNIQUE | `secrets.token_urlsafe(32)` |
| `scheduled_send_at` | TIMESTAMPTZ NOT NULL | `meeting_start - consent_window_minutes` |
| `attempt_count` | INT NOT NULL DEFAULT 0 | Caps at 2 (initial + 1 retry) |
| `last_attempt_at` | TIMESTAMPTZ | When bot last tried sending |
| `next_retry_at` | TIMESTAMPTZ | NULL when no retry pending |
| `delivery_status` | TEXT NOT NULL CHECK | `pending` / `sent` / `failed` / `fallback_to_in_meeting` |
| `failure_reason` | TEXT | "Invalid email", "SMTP bounce", etc. |
| `opted_out_at` | TIMESTAMPTZ | NULL until clicked |
| `opt_out_ip` | TEXT | Browser IP for audit (optional) |
| `seller_notified_at` | TIMESTAMPTZ | When INF_MSG_0001 event was raised |
| `created_at` / `updated_at` | TIMESTAMPTZ | Server defaults |

Indexes:
- UNIQUE on `(meeting_id, recipient_email)` — re-scheduling is idempotent
- UNIQUE on `opt_out_token` — token lookup
- Partial on `(next_retry_at)` WHERE `delivery_status='failed' AND attempt_count<2` — retry queue
- Partial on `(meeting_id)` WHERE `opted_out_at IS NOT NULL` — fast "any opt-outs?" query

**Effort:** ~45 min.

---

### Phase 2 — ORM + Pydantic schemas

**Files:**
- `app/models/consent_email.py` — `MeetingConsentEmail` + `DELIVERY_STATUS_VALUES` + `INTERNAL_FALLBACK_REASONS`
- `app/schema/consent_email.py` — 8 schemas:
  - `ConsentScheduleRequest`, `ConsentScheduleResponse`, `ConsentRecipientRecord`
  - `ConsentDeliveryUpdateRequest`, `ConsentDeliveryUpdateResponse`
  - `ConsentOptOutResponse` (HTML returned via FastAPI HTMLResponse, not this schema)
  - `MeetingConsentStatusResponse`
  - `ConsentRetryQueueResponse`

**Effort:** ~45 min.

---

### Phase 3 — Service layer

`app/services/consent_email_service.py`:

```python
def schedule_consent_emails(db, meeting_id) -> ScheduleResult:
    """Validates: meeting exists; not already started.
       Filters attendees by INTERNAL_EMAIL_DOMAINS.
       If consent window open: upsert per-recipient rows with fresh tokens.
       If consent window passed: returns should_send=False; bot falls back to US02."""

def record_delivery_status(db, consent_id, status, reason) -> MeetingConsentEmail:
    """Updates row, increments attempt_count.
       If failed + attempt<2: schedules retry at now+10min.
       If failed + attempt>=2: marks 'fallback_to_in_meeting'."""

def record_opt_out(db, token, client_ip) -> tuple[MeetingConsentEmail, MeetingDetails]:
    """Validates: token exists; meeting hasn't started.
       Idempotent — second click returns same SUCC_MSG_0010.
       Cascades bot_status='cancelled' on parent meeting."""

def get_consent_status(db, meeting_id) -> ConsentStatus:
    """Aggregates: any_opted_out, opt_out_count, consent_mechanism, pending_retries.
       Bot reads this right before joining."""

def due_for_retry(db, now) -> list[MeetingConsentEmail]:
    """Returns rows where next_retry_at <= now AND failed AND attempt < 2."""
```

**Effort:** ~1.5 hrs.

---

### Phase 4 — API endpoints

`app/api/consent_emails.py`, mounted at `/consent-emails`:

```
POST   /consent-emails/schedule                # bot calls this when meeting eligible
PATCH  /consent-emails/{consent_id}/delivery   # bot reports send result
GET    /consent-emails/opt-out/{token}         # PUBLIC — browser hits this
GET    /consent-emails/due-for-retry           # bot polls this periodically
GET    /consent-emails/{meeting_id}            # audit / FE — list all records for a meeting
```

Plus on the meetings router (or a thin /meetings router):
```
GET    /meetings/{meeting_id}/consent-status   # bot reads this before joining
```

Error surface:
- `404` — unknown meeting / token
- `409` — token already opted-out (idempotent — return same HTML)
- `410 Gone` — meeting already started (opt-out link expired)
- `400` — meeting not eligible / already cancelled
- `422` — schema validation

**Effort:** ~1 hr.

---

### Phase 5 — Config additions

`app/core/config.py`:

```python
# US03 — Consent Capture
CONSENT_WINDOW_MINUTES = int(os.getenv("CONSENT_WINDOW_MINUTES", "60"))
INTERNAL_EMAIL_DOMAINS = [
    d.strip().lower()
    for d in os.getenv("INTERNAL_EMAIL_DOMAINS", "lenovo.com,motorola.com").split(",")
    if d.strip()
]
OPT_OUT_BASE_URL = os.getenv("OPT_OUT_BASE_URL", "http://localhost:8001")
SYSTEM_EMAIL_ADDRESS = os.getenv("SYSTEM_EMAIL_ADDRESS", "sales-assistant@lenovo.com")
```

`.env.example`:
```
CONSENT_WINDOW_MINUTES=60
INTERNAL_EMAIL_DOMAINS=lenovo.com,motorola.com
OPT_OUT_BASE_URL=https://aibackend.lenovo.com
SYSTEM_EMAIL_ADDRESS=sales-assistant@lenovo.com
```

**Effort:** ~15 min.

---

### Phase 6 — US01 lifecycle integration

`bot_status_reason` on `tbl_schedule_meetings` is already free-form TEXT (no CHECK constraint), so no schema change is needed. We just document the new canonical reason `participant_opted_out` in:

- `app/models/schedulemeeting.py` — add to the comment block listing canonical reasons
- `US01_BACKEND_HANDOFF_FOR_AI_TEAM.md` — note that opt-out cascade uses this reason

The opt-out endpoint calls the existing `meeting_details_service.update_meeting_status()` (US01) so the audit trail is unified.

**Effort:** ~10 min.

---

### Phase 7 — Tests

11 smoke tests in `tests/test_consent_capture_lifecycle.py`:

1. `POST /schedule` happy path — creates rows for external recipients only
2. `POST /schedule` filters out internal Lenovo domains
3. `POST /schedule` when window has passed → returns `should_send=false`
4. `POST /schedule` is idempotent — re-call returns existing tokens
5. `POST /schedule` for a meeting that doesn't exist → 404
6. `PATCH /delivery` with `status='sent'` updates `delivery_status='sent'`
7. `PATCH /delivery` with `status='failed'` schedules retry at +10 min
8. `PATCH /delivery` failed twice → marks `fallback_to_in_meeting`
9. `GET /opt-out/{token}` happy path — 200 HTML, opt-out recorded, bot cancelled
10. `GET /opt-out/{token}` after meeting starts → 410 Gone
11. `GET /opt-out/{token}` is idempotent — second click also 200
12. `GET /meetings/{id}/consent-status` aggregates `any_opted_out=true` correctly
13. `GET /due-for-retry` returns only rows due for retry

(13 actually — slightly above the original 11 estimate; see effort note below.)

**Effort:** ~2 hrs.

---

### Phase 8 — Docs + handoff

- README extension — add §3 "Consent Capture" with state diagram + cURL examples for all 5 endpoints
- `US03_BACKEND_HANDOFF_FOR_AI_TEAM.md` — full endpoint reference + sample bot flow + open questions
- `US03_STATUS_AND_DEPENDENCIES.md` — done / pending / blocked tracker
- `.env.example` — new config keys
- Update `tbl_schedule_meetings` state diagram to show `cancelled / participant_opted_out` reason

**Effort:** ~1.5 hrs.

---

## 5. Effort summary

| Phase | What | Effort |
|---|---|---|
| 1 — Schema | 1 new table + 4 indexes | 45 min |
| 2 — ORM + schemas | 8 Pydantic + 1 model | 45 min |
| 3 — Service layer | 5 service functions | 1.5 hrs |
| 4 — API endpoints | 6 routes (5 consent + 1 status) | 1 hr |
| 5 — Config additions | 4 env vars | 15 min |
| 6 — US01 lifecycle integration | Doc + canonical reason | 10 min |
| 7 — Tests | 13 smoke tests | 2 hrs |
| 8 — Docs + handoff | README + 2 docs | 1.5 hrs |
| **Total** | | **~8 hrs** |

---

## 6. Out-of-scope (deferred / AI team / future sprints)

| Item | Why deferred |
|---|---|
| Email template rendering (HTML / plain) | AI team owns; backend exposes template vars only |
| Microsoft Graph API integration / SMTP send | AI team owns; backend records delivery results |
| Seller-facing notification UI (showing INF_MSG_0001) | "Notifications will be displayed in future sprints" — AC says so |
| Re-scheduling consent on meeting reschedule | Add as US01-side reschedule cascade later if needed |
| Per-recipient opt-out re-schedule (participant changes mind?) | Not in spec — opt-out is one-way for v1 |
| Localised email content (i18n) | Defer until tenant rollout |
| Rate limiting on opt-out endpoint (DDoS) | Cloudflare / WAF concern; not v1 backend |
| Click tracking (did the email get OPENED, not just delivered?) | Compliance asks for delivered, not opened. Out of scope unless requested |

---

## 7. Open questions for AI team / PM

| # | Question | Why it matters | Owner |
|---|---|---|---|
| Q1 | Where does the canonical text of `SUCC_MSG_0010` and `INF_MSG_0001` live? | We render `SUCC_MSG_0010` server-side; bot displays `INF_MSG_0001`. Need single source | PM |
| Q2 | Is the email's "from" address always `<seller> via Lenovo Sales Assistant <sales-assistant@lenovo.com>`, or per-region? | Affects SYSTEM_EMAIL_ADDRESS config | PM |
| Q3 | Do we need to store the opt-out IP / user-agent for compliance? | Adds GDPR considerations; default yes for now | Legal / PM |
| Q4 | What's the bot's polling interval for `/due-for-retry`? | Decides backend load profile | AI team |
| Q5 | When a meeting is rescheduled (US01), should consent records be invalidated and re-scheduled? | Affects how we handle rescheduling cascade | Joint |
| Q6 | Is `INTERNAL_EMAIL_DOMAINS` global or tenant-specific? Could vary per Lenovo subsidiary (Motorola, etc.) | Affects config shape | DevOps + PM |
| Q7 | What happens to consent records when the parent meeting is deleted (US01 DELETE)? Soft-cascade or preserve? | Decides retention policy | Joint |

---

## 8. Acceptance criteria for "this story is done from backend's side"

- [ ] Migration applied to dev Postgres
- [ ] All 6 endpoints work via cURL
- [ ] Schedule endpoint correctly filters internal domains
- [ ] Window-passed scenario returns `should_send=false`
- [ ] Opt-out cascades to `bot_status='cancelled'`
- [ ] Opt-out HTML page renders SUCC_MSG_0010
- [ ] Retry queue returns due-for-retry rows
- [ ] All 13 smoke tests pass
- [ ] README has cURL for every new endpoint + state diagram
- [ ] AI team has the `US03_BACKEND_HANDOFF_FOR_AI_TEAM.md` doc
- [ ] Repo lints clean

---

## 9. Order of work (for the implementation pass)

1. Phase 1 (schema) → confirm via pgAdmin
2. Phase 5 (config additions) — needed by Phase 3 service layer
3. Phase 2 (ORM + schemas)
4. Phase 3 (service layer) — most of the logic
5. Phase 4 (API endpoints) — wire to services
6. Phase 6 (US01 lifecycle integration) — small doc tweaks
7. Phase 7 (tests) — harness everything
8. Phase 8 (docs + handoff) — finalise

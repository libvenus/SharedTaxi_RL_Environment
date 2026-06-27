# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

Four services make up the Lenovo D365 Sales AI platform:

| Directory | Role | Port | Stack |
|---|---|---|---|
| `Lenevo-Frontend/` | React SPA (sales dashboard UI) | 5173 | React 19, Vite, Bootstrap 5 |
| `Lenovo-Backend/` | D365 CRM data layer | 8000 | FastAPI, SQLAlchemy, PostgreSQL |
| `Lenovo-AIBackend/` | Note-Taking Agent / AI features | 8001 | FastAPI, SQLAlchemy, PostgreSQL, httpx |
| `LenovoD365-SalesAI/` | Multi-agent orchestrator | 8091 | Microsoft Agent Framework, FastAPI, `uv` |

> Note the spelling: the frontend directory is **Lenevo** (typo in original), the backends are **Lenovo**.

---

## Running the services

### Lenovo-Backend (D365 Sales CRM — port 8000)

```bash
cd Lenovo-Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in Postgres credentials
# Apply migrations in order (each is idempotent):
psql -f sql/2026_06_create_lvo_activity.sql
psql -f sql/2026_06_add_dealhealth.sql
# ... (see Lenovo-Backend/README.md for the full ordered list)
uvicorn app.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

Optional one-time bootstrap jobs (run after migrations):
```bash
python -m app.jobs.recalc_health
python -m app.jobs.recalc_accounts
python -m app.jobs.snapshot_kpis --backfill
python -m app.jobs.snapshot_account_kpis --backfill
```

### Lenovo-AIBackend (Note-Taking Agent — port 8001)

```bash
cd Lenovo-AIBackend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # see root .env.example for all vars
# Apply Sprint 1A migrations:
psql -f sql/2026_06_us01_meeting_lifecycle.sql
psql -f sql/2026_06_us02_meeting_transcript.sql
psql -f sql/2026_06_us03_consent_email.sql
psql -f sql/2026_06_us04_data_task.sql
uvicorn app.main:app --reload --port 8001
# Swagger UI: http://localhost:8001/docs
```

Daily data-hygiene scan CLI:
```bash
python -m app.jobs.scan_data_tasks [--limit N] [--dry-run] [--stale-days N]
```

### Lenevo-Frontend (React SPA — port 5173)

```bash
cd Lenevo-Frontend
npm install
npm run dev       # http://localhost:5173
npm run build
npm run lint      # eslint
```

The Vite dev server proxies `/api/*` → port 8000 and `/ai-api/*` → port 8001 (see `vite.config.js`). In production, set `VITE_API_BASE_URL`.

### LenovoD365-SalesAI (Multi-agent orchestrator — port 8091)

```bash
cd LenovoD365-SalesAI
python -m pip install uv
uv sync --prerelease=allow --no-install-project
source .venv/Scripts/activate   # or: source .venv/bin/activate on Linux/macOS
cp .env.example .env            # fill in AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL
uvicorn orchestrator_api:app --reload --port 8091
```

Optional DevUI (with `uv sync --extra dev`): run `serve.bat` then open `http://localhost:5173`.

---

## Running tests

### Lenovo-Backend (pure-function tests, no live DB required)

```bash
cd Lenovo-Backend
pip install pytest
pytest tests/ -v
# single test file:
pytest tests/test_deal_health.py -v
```

### Lenovo-AIBackend (TestClient with SQLite, no live Postgres required)

```bash
cd Lenovo-AIBackend
pytest -q
# single test file:
pytest tests/test_meeting_lifecycle.py -q
```

---

## Architecture: how the services connect

```
Lenevo-Frontend
  │  /api/*   ──────────────────────► Lenovo-Backend (:8000)
  │  /ai-api/* ─────────────────────► Lenovo-AIBackend (:8001)
  │                                        │
  │                                        │ d365_client.py (httpx)
  │                                        ▼
  │                                   Lenovo-Backend (:8000)
  │
  └─ /chat  ────────────────────────► LenovoD365-SalesAI orchestrator_api (:8091)
                                           │
                                      Microsoft Agent Framework GroupChat
                                      (OrchestratorAgent → SalesAgent / AnalystAgent / SchedulerAgent)
```

**Key routing rule:** All Lenovo-AIBackend HTTP routes carry the `/ai-api` prefix (configurable via `AIBACKEND_API_PREFIX`) so that Nginx can proxy both services without path collision.

---

## Lenovo-Backend conventions

- **camelCase JSON** everywhere (enforced by `APIModel` in `app/schemas.py` with `alias_generator`). Write operations accept camelCase; `X-User-Id` header is optional but feeds the audit log.
- **Services are pure** — route handlers are thin; DB logic lives in `app/services/`.
- **Background recalc**: every deal write enqueues a deal-health recalc via FastAPI `BackgroundTasks`, which cascades to an account-status recalc. Force-sync via `POST /api/opportunities/{id}/health/recalculate` or `POST /api/accounts/{id}/recompute-status`.
- **Deal health score** (0–100, bands RED/YELLOW/GREEN): five weighted components — Stage Progress 25%, Activity Freshness 25%, Stakeholder 20%, Close-Date Confidence 20%, Risk Adjustment 10%. Thresholds live in `lvo_dealhealthconfig.lvo_settings`.
- **13 risk rules** derived from the same data — see top of `app/services/deal_risks.py`.
- **Migrations** are hand-written idempotent SQL files in `sql/`. Every model change needs a matching `sql/<YYYY_MM>_<topic>.sql`. Alembic is planned for Sprint 1B.

## Lenovo-AIBackend conventions

- **snake_case JSON** throughout (opposite of D365 backend).
- All routes live under `/ai-api` via a shared `APIRouter(prefix=AIBACKEND_API_PREFIX)` in `app/main.py`.
- **D365 client** (`app/clients/d365_client.py`): 404 → `None`; any 5xx/timeout → `D365ClientError`. Bot pattern is "log and continue" — D365 hiccups should never crash bot flow.
- **Bot lifecycle state machine** for `tbl_schedule_meetings.bot_status`: `pending → scheduled → joining → joined / failed / lobby_waiting`. Any state can transition to `cancelled` or `rescheduled`. The lifecycle PATCH owns state transitions; re-POSTing a meeting preserves existing status.
- **Transcript pipeline**: `in_progress → finalized` (clean end) or `terminated_partial` (early stop). Consent fields (`consent_message_text`, `consent_sent_at`) are NOT NULL — a transcript cannot exist without compliance proof.
- **Consent email idempotency**: `POST /consent-emails/schedule` filters internal Lenovo domains (`INTERNAL_EMAIL_DOMAINS`) and checks the consent window before creating rows.
- **Data task idempotency**: partial UNIQUE on `(entity_kind, entity_id, task_kind) WHERE status IN ('open','dismissed')`. Dismissed tasks suppress re-detection (AC #5).

## LenovoD365-SalesAI conventions

- Uses `uv` (not pip) for dependency management — `pyproject.toml` + `uv.lock`.
- Star topology GroupChat: `OrchestratorAgent` selects next speaker between `SalesAgent` and `AnalystAgent`; both see full conversation history.
- Session history is in-memory (`db/chat_history.json` for file-backed persistence); restarting the process clears sessions.
- Email drafting (`api/email_api.py` on `:8093`) is a **standalone** FastAPI service, not an agent. It never sends email — it drafts only.
- All agent tools currently return hard-coded demo data. Swap for Dataverse Web API + Microsoft Graph calls when wiring to the dev tenant.
- Log file: `orchestrator_api.log` in repo root; also writes to `logs/orchestrator_api_<date>.log`.

## Frontend conventions

- Single API client: `src/api/client.js` — all fetch calls go through `request()` which builds URLs and handles errors uniformly.
- Auth is session-storage-only (`isAuthenticated` flag) — stub, not real auth.
- Array query params are serialised as comma-separated strings by the client.
- The `BOT_BASE_URL` and `BOT_API_KEY` in `src/api/client.js` are hardcoded to an external bot service endpoint.

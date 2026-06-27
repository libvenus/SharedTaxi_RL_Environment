# Sprint 2 · Audit Log & Compliance — Backend Handoff

**Repos:** `Lenovo D365 Sales` + `Lenovo-AIBackend` (shared Postgres)  
**Migration:** `sql/2026_14_audit_compliance.sql`  
**Last updated:** 2026-06-19

---

## Summary

Tamper-evident platform audit trail in `lvo_audit_log` with:

- **Who / what** — `lvo_changedby`, `lvo_actortype` (seller | admin | ai | system | event_spine)
- **What changed** — `lvo_diff` JSON with `before` / `after` and auto `fieldChanges[]` for CRM write-backs
- **Outcome** — `lvo_outcome` (success | failure), `lvo_failurereason`
- **Correlation** — `lvo_correlationid` (Event Spine / distributed tracing)
- **Category** — seller_action | admin_action | ai_automated | crm_writeback | event_spine | read_action
- **Immutability** — Postgres trigger blocks UPDATE; DELETE only via retention purge job
- **Retention** — default **90 days**, configurable in `lvo_audit_config` (no code deploy)
- **Read logging** — optional per actor class (seller / admin / AI output views)
- **No UI** — query via `/api/compliance/*` or Azure Monitor log export from structured `audit_event` logger lines

---

## Prerequisites

```bash
# D365 Sales DB (shared)
psql -f sql/2026_06_create_next_actions_audit.sql   # if not already applied
psql -f sql/2026_14_audit_compliance.sql

# Env (both services)
COMPLIANCE_API_KEY=<shared-secret>          # protects /api/compliance/*
AUDIT_READ_LOGGING_ENABLED=true             # D365 middleware master switch
```

---

## APIs (D365 Sales)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/compliance/audit-events` | Ingest from AIBackend / Event Spine |
| GET | `/api/compliance/audit-events` | Compliance query (filters + pagination) |
| GET | `/api/compliance/audit-config` | Read retention + read-logging toggles |
| PATCH | `/api/compliance/audit-config` | Update config without deploy |

**Auth:** `X-Compliance-Api-Key` when `COMPLIANCE_API_KEY` is set.

### Ingest example (Event Spine dead letter)

```bash
curl -X POST "http://localhost:8000/api/compliance/audit-events" \
  -H "X-Compliance-Api-Key: $COMPLIANCE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "event_spine",
    "entity_id": "evt-123",
    "action": "dead_letter",
    "category": "event_spine",
    "actor_type": "event_spine",
    "outcome": "failure",
    "event_type": "meeting.summary.generated",
    "delivery_attempts": 3,
    "failure_reason": "Downstream CRM write timed out after 30s",
    "correlation_id": "evt-123"
  }'
```

### Query example

```bash
curl "http://localhost:8000/api/compliance/audit-events?category=crm_writeback&limit=50" \
  -H "X-Compliance-Api-Key: $COMPLIANCE_API_KEY"
```

### Toggle seller read logging

```bash
curl -X PATCH "http://localhost:8000/api/compliance/audit-config" \
  -H "X-Compliance-Api-Key: $COMPLIANCE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"log_seller_reads": true}'
```

---

## What is audited today

| Category | Source | Examples |
|----------|--------|----------|
| CRM write-back | D365 routers | Deal PATCH, contacts, competitors, next actions |
| Admin action | SOM, quarter pulse | Organizational intent PUT/DELETE, quota upsert |
| AI automated | `deal_recalc` | Background deal-health recalc (`deal_health`) |
| Seller action | AIBackend | Data task resolve/dismiss, todo complete, prep task done |
| Read action | Middleware | GET `/api/*` when toggled in config |
| Event Spine | Ingest API | `record_event_spine()` helper (AIBackend) |

---

## Code layout

| File | Role |
|------|------|
| `app/services/audit_log.py` | Central `write_audit_event`, config, query, purge |
| `app/routers/audit_compliance.py` | Compliance REST API |
| `app/middleware/audit_read_logging.py` | Optional GET logging |
| `app/jobs/purge_audit_log.py` | Daily retention job |
| `Lenovo-AIBackend/app/services/compliance_audit.py` | Shared-table writer + HTTP fallback |

---

## Retention job (cron)

```bash
python -m app.jobs.purge_audit_log
python -m app.jobs.purge_audit_log --retention-days 90 --verbose
```

Uses `SET LOCAL audit.purge_mode = 'on'` to bypass immutability trigger.

---

## Azure Monitor / Log Analytics

Each `write_audit_event` emits a structured log line (`audit_event` / `compliance_audit`) suitable for Diagnostic Settings export. No seller UI — ops wires App Insights / Log Analytics to the container stdout pipeline.

---

## Headers (existing + new)

| Header | Use |
|--------|-----|
| `X-User-Id` | Actor email/id on seller/admin writes |
| `X-Actor-Type` | Override actor inference (seller/admin/ai) |
| `X-Correlation-Id` | Link to originating event |
| `X-Compliance-Api-Key` | Compliance API auth |

---

## Tests

```bash
cd "Lenovo D365 Sales"
pytest -q tests/test_audit_compliance.py
```

---

## Pending (future stories)

- Event Spine producer integration (Graph / enterprise bus)
- Wire all AIBackend paths (transcript finalize, post-meeting summary, outreach send)
- Azure Monitor workbook templates
- Notes (`POST /api/opportunities/notes`) audit
- SOM timeline classification PUT audit

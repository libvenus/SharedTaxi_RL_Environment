# Sprint 2 · US 1.3 — Task Pending Badge · Backend Handoff

**Sprint:** 2  
**User Story:** 1.3 — View and Navigate from the Task Pending Badge  
**Backend status:** Phase 1 (D365) + Phase 2 (AIBackend) shipped  
**Repos:** `Lenovo D365 Sales` + `Lenovo-AIBackend`  
**Last updated:** 2026-06-19

---

## TL;DR — who calls which API

| # | Endpoint | Repo | Purpose |
|---|----------|------|---------|
| 1 | `GET /api/tasks/pending-summary?sellerId=` | D365 Sales | Open `lvo_nextaction` count + overdue flag |
| 2 | `GET /ai-api/todos/summary?sellerId=` | AIBackend | Open Execute To-Do count + overdue flag |
| 3 | `GET /ai-api/todos?sellerId=` | AIBackend | Full To-Do list (overdue first) after badge click |

**FE merge for badge:**

```javascript
const d365 = await fetch(`/api/tasks/pending-summary?sellerId=${sellerId}`);
const ai = await fetch(`/ai-api/todos/summary?sellerId=${sellerId}`);
const count = d365.count + ai.count;
const hasOverdue = d365.hasOverdue || ai.hasOverdue;
const badgeColor = hasOverdue ? 'red' : 'default';
const label = `${count} task${count === 1 ? '' : 's'} pending`;
```

Poll both endpoints every **30–60s** for real-time badge updates (same pattern as notifications).

**Click badge:** navigate to Execute Workspace To-Do → `GET /ai-api/todos?sellerId=` (CRM next actions remain on deal detail).

---

## D365 Sales — `GET /api/tasks/pending-summary`

**Counts:** open `lvo_nextaction` on seller's open opportunities (`owninguser`, `statecode=Open`).

```bash
curl -s "http://localhost:8000/api/tasks/pending-summary?sellerId=055DAFE7-9840-451D-8328-5F70A6326C03"
```

**200:**
```json
{
  "sellerId": "055DAFE7-9840-451D-8328-5F70A6326C03",
  "count": 5,
  "overdueCount": 2,
  "dueTodayCount": 1,
  "hasOverdue": true,
  "badgeColor": "red",
  "label": "5 tasks pending",
  "lastUpdatedAt": "2026-06-19T10:30:00",
  "source": "d365"
}
```

**Files:** `app/services/task_pending.py`, `app/routers/tasks.py`, `app/schemas.py`  
**Version:** API `0.17.0`  
**Errors:** `422` missing sellerId · `500` `ERR_MSG_0022`

---

## AIBackend — Execute To-Do

**Migration (required on dev/prod Postgres):**

```bash
psql -f sql/2026_08_us05_todo_seller_scope.sql
```

Adds `seller_id` to `tbl_to_do_list`.

### `GET /ai-api/todos/summary?sellerId=`

Same JSON shape as D365 with `"source": "ai"`.

```bash
curl -s "http://localhost:8001/ai-api/todos/summary?sellerId=055DAFE7-9840-451D-8328-5F70A6326C03"
```

### `GET /ai-api/todos?sellerId=`

- Includes **overdue** tasks (previously excluded)
- Sort: overdue → due today → future → no due date
- Optional `filter_type=all|outreach|document|action`

```bash
curl -s "http://localhost:8001/ai-api/todos?sellerId=055DAFE7-9840-451D-8328-5F70A6326C03"
```

### `POST /ai-api/todos/create`

Body now accepts optional `seller_id` (UUID) for manual tasks.

---

## DevOps checklist

| Step | Repo | Action |
|------|------|--------|
| 1 | D365 Sales | Deploy v0.17.0 |
| 2 | AIBackend | Run `sql/2026_08_us05_todo_seller_scope.sql` |
| 3 | AIBackend | Deploy service (model + API changes) |
| 4 | FE | Wire Header badge + poll + navigate to Execute |

---

## Out of scope (follow-up)

- Meeting `confirmedNextSteps` → auto-create `tbl_to_do_list` rows
- Priority Actions entity (currently FE mock)
- Single BFF endpoint aggregating D365 + AI counts

---

## Tests

```bash
# D365 Sales
pytest -q tests/test_task_pending_summary.py

# AIBackend
pytest -q tests/test_todo_pending_summary.py
```

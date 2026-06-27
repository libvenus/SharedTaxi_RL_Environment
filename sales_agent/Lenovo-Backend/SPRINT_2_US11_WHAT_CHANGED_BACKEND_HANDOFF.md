# Sprint 2 · US 1.1 — What Changed Notifications · Backend Handoff

**Sprint:** 2
**User Story:** 1.1 — What Changed Feed (in-app notification panel + Activity Timeline)
**Epic:** Home Dashboard — AI Sales Agent landing experience
**Backend status:** MVP read APIs shipped · 11 unit tests passing · migration SQL ready
**Backend repo (primary):** `Lenovo D365 Sales`
**Backend repo (cross-cut):** `Lenovo-AIBackend` (meeting-done events), email/calendar integrations (inbound email)
**Backend contact:** Sanmay
**Pairing:** Sanmay (backend) ↔ Namisha (FE) + integration teams
**Document audience:** Frontend, DevOps, AI/Bot team, email/calendar integration owners
**Linked docs:**
- [API_CONTRACT.md](./API_CONTRACT.md) — §15 (canonical JSON shapes)
- [sql/2026_07_create_lvo_notification_read.sql](./sql/2026_07_create_lvo_notification_read.sql) — read-state migration
- [sql/2026_07_seed_what_changed_local_smoke.sql](./sql/2026_07_seed_what_changed_local_smoke.sql) — local test data (one row per type)
- `app/services/what_changed.py` — feed aggregation logic
- `app/routers/notifications.py` — HTTP routes

**Last updated:** 2026-06-17

---

## UI screens → API mapping (FE mockups)

Two screens share the same `WhatChangedItem` payload. The FE derives display-only fields (relative time, date headers, icons) client-side.

### Screen 1 — Home · "What Changed" panel

```
┌─────────────────────────────────────┐
│ What Changed          View All >    │  ← "View All" → Activity Timeline route
├─────────────────────────────────────┤
│ [icon] Pipeline Value    20 min ago │
│        Total value of all active…   │
├─────────────────────────────────────┤
│ … up to 4–6 items …                 │
└─────────────────────────────────────┘
```

| UI element | API field | FE derivation |
|------------|-----------|---------------|
| Card list (4–6 items) | `GET /api/notifications?limit=6` | Slice is server-side |
| **View All >** | — | Navigate to Activity Timeline page; call `GET /api/activity-timeline` |
| Row icon | `activityType` | Map `email` → envelope, `meeting` → calendar, `crm_update` → CRM, `risk` → alert, `task` → checklist |
| **Title** (bold) | `title` | e.g. email subject, `"Stage changed"`, risk name |
| **Timestamp** ("20 min ago") | `eventAt` | Format relative time in FE |
| **Description** (body) | `summary` | One-line activity summary |
| Account context | `accountName` | Show above/below title if design requires (story AC) |
| Activity type badge | `activityType` | Legend: Email Received, Meeting Done, Stage Changed, etc. |
| Click row | `linkType` + `linkId` | Deep-link per §6 |
| Unread styling | `isRead` | `false` → highlight; call `PATCH .../read` on open |

> **Mock note:** The current Figma sample uses KPI-style titles ("Pipeline Value", "Win Rate"). Production data from this API will be **deal-activity events** (emails, meetings, stage changes, risks, overdue tasks) per the user story — not KPI strip metrics. KPI cards are a separate Home dashboard widget.

### Screen 2 — Activity Timeline (full page)

```
┌──────────────────────────────────────────────┐
│ Today                                        │  ← FE groups by date(eventAt)
│ [icon] Re: Infosys ThinkPad…  [Insight]    │
│        Opportunity pipeline updated · Arjun  │  ← subtitle: FE-built from type + actor
│        Includes 4 deals in negotiation…      │  ← summary
│                              Inbound 10:42 AM│  ← direction + time
├──────────────────────────────────────────────┤
│ Yesterday …                                  │
└──────────────────────────────────────────────┘
```

| UI element | API field | Status |
|------------|-----------|--------|
| Date headers ("Today", "Yesterday") | `eventAt` | **FE only** — group items by calendar day |
| Row icon | `activityType` | **FE** — same mapping as panel |
| **Title** | `title` | ✅ API |
| **Insight badge** ("Expansion signal…") | — | ❌ **Not in API** — AI enrichment; defer or static placeholder in S2 |
| **Subtitle** ("Stage progression tracked · Arjun Shah") | `categoryLabel` + `actorName` | ✅ API — subtitle = ``{categoryLabel} · {actorName}`` |
| **Description** | `summary` | ✅ API |
| **Inbound / Outbound** pill | `direction` | ✅ API — `inbound` / `outbound` / `null` (null for CRM, risk, task) |
| **Time** ("10:42 AM") | `eventAt` | **FE** — format time portion |
| Pagination / infinite scroll | `page`, `pageSize`, `totalPages` | ✅ `GET /api/activity-timeline` |
| Type filter chips | `types` query param | ✅ API |

### API gaps revealed by UI (backlog)

| # | UI needs | Status |
|---|----------|--------|
| G1 | `direction` (Inbound/Outbound) | ✅ Shipped — `direction` on `WhatChangedItem` |
| G2 | Actor display name | ✅ Shipped — `actorName` resolved from `systemuser` |
| G3 | Insight badge text | ❌ AI enrichment — out of scope for D365 read API MVP |
| G4 | Subtitle category label | ✅ Shipped — `categoryLabel` (e.g. `"Stage progression tracked"`) |

---

## TL;DR — who calls which API

| # | Endpoint | Caller | When | Purpose |
|---|----------|--------|------|---------|
| 1 | `GET /api/notifications` | **Frontend** (Home dashboard notification panel) | On Home load + poll every 30–60 s while panel is open | Fetch 4–6 newest portfolio events for the signed-in seller |
| 2 | `GET /api/activity-timeline` | **Frontend** (Activity Timeline page) | On page load + pagination / filter change | Full paginated feed across seller's portfolio |
| 3 | `PATCH /api/notifications/{notificationId}/read` | **Frontend** | User opens/dismisses a notification | Persist read-state so `isRead` flips on next fetch |

**Nobody calls a write/ingest API yet.** Events are **read at query time** from existing D365 tables (`lvo_activity`, `lvo_audit_log`, `lvo_dealrisk`, `lvo_nextaction`). Producers listed in §8 must **write rows into those tables** (or a future ingest endpoint) for new notifications to appear.

**Routing on dev:** Nginx `location /api/` → D365 Sales container (`lenovo-aibackend:8000`). No `/ai-api` prefix for this story.

---

## 1. Story in one paragraph

Sellers see a **What Changed** feed as in-app notifications on the Home dashboard (4–6 recent items) and a full **Activity Timeline** page (paginated). Each item shows account name, activity summary, timestamp, and activity type (Email, Meeting, CRM Update, Risk, Task). Items deep-link to the relevant deal, Outreach Studio, todo, or activity. The seller should **not** be notified for CRM changes they made themselves. Feed load failures surface **`ERR_MSG_0020`** with a retry option (FE responsibility).

---

## 2. Architecture — how the feed is built

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVENT SOURCES (writers)                          │
├─────────────────┬─────────────────┬──────────────────┬────────────────┤
│ Email platform  │ AIBackend / bot │ D365 deal writes │ Deal-health job│
│ (inbound mail)  │ (meeting done)  │ (PATCH opp)      │ (risk persist) │
└────────┬────────┴────────┬────────┴────────┬─────────┴───────┬────────┘
         │                 │                 │                 │
         ▼                 ▼                 ▼                 ▼
   lvo_activity      lvo_activity     lvo_audit_log      lvo_dealrisk
   type=email        type=meeting     entity=opportunity  statecode=Active
   direction=inbound                   action=update
         │                 │                 │                 │
         │                 │                 │          lvo_nextaction
         │                 │                 │          (overdue Open)
         └─────────────────┴─────────────────┴─────────────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │  app/services/what_changed.py           │
              │  build_seller_feed() — query-on-read      │
              │  • seller's OPEN opportunities only       │
              │  • 30-day lookback window                 │
              │  • merge + sort newest first              │
              │  • panel excludes seller's own CRM edits  │
              └────────────────────┬────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
 GET /api/notifications   GET /api/activity-timeline   lvo_notification_read
 (top 6, exclude self)     (paginated, include self)    (read-state only)
         │                         │
         └─────────────┬───────────┘
                       ▼
                 Frontend (Home + Activity Timeline)
```

**Design choice (MVP):** No `tbl_seller_notification` table. Feed items are **synthetic keys** derived from source row IDs. Only **read/dismiss state** is persisted (`lvo_notification_read`).

---

## 2a. How changes are captured

There is **no notification write API** in MVP. Changes are **captured in source tables** by existing flows; `build_seller_feed()` **reads and merges** them when the FE calls GET.

| Source table | Who writes the row | When / how |
|--------------|-------------------|------------|
| **`lvo_activity`** | Email integration (future), AIBackend/bot (future), manual SQL seed | One row per touchpoint: `email`, `meeting`, `crm`, `multiple`. Key columns: `lvo_direction`, `lvo_subject`, `lvo_body`, `lvo_activitydate`, `owninguser`. |
| **`lvo_audit_log`** | **D365 Sales** `deals_write.py` → `_write_audit()` | Auto on `PATCH /api/opportunities/{id}`, competitor CRUD, next-action CRUD. `lvo_changedby` = `X-User-Id` header. |
| **`lvo_dealrisk`** | **D365 Sales** deal-health recalc | `POST /api/opportunities/{id}/health/recalculate` rewrites Active risk rows. |
| **`lvo_nextaction`** | **D365 Sales** next-action APIs | `POST/PATCH .../next-actions`. Feed shows row when overdue. |
| **`lvo_notification_read`** | **FE** via `PATCH .../read` | Read/dismiss state only — not the events. |

**Query-on-read steps:** (1) load seller's open opps → (2) pull event rows (30-day window) → (3) map to `WhatChangedItem` → (4) resolve `actorName` → (5) sort newest first → (6) panel applies exclusion rules.

---

## 2b. What triggers a notification

| Story legend | `activityType` | Source table | Panel | Timeline | How it enters the DB |
|--------------|----------------|--------------|:-----:|:--------:|----------------------|
| Email Received | `email` | `lvo_activity` | ✅ | ✅ | `type=email`, `direction=inbound` |
| Meeting Done | `meeting` | `lvo_activity` | ✅ | ✅ | `type=meeting` |
| Stage Changed / CRM Update | `crm_update` | `lvo_audit_log` | ✅* | ✅ | Opportunity PATCH → audit diff JSON |
| Risk Detected | `risk` | `lvo_dealrisk` | ✅ | ✅ | Health recalc persists rules |
| Task Overdue | `task` | `lvo_nextaction` | ✅ | ✅ | `status=Open`, `dueDate < today` |

\*Panel **excludes** row when `actor` matches `sellerId` or `X-User-Id`.

**Excluded from panel:** outbound emails · seller's own CRM audit edits · closed deals · events >30 days old · inactive rows.

**Not excluded from panel:** inbound emails and meetings on the seller's deals (even when `lvo_activity.owninguser` is the seller).

**Not captured yet:** live email webhook · bot meeting-done hook · AI insight badges.

---

## 3. Acceptance criteria coverage

| AC | Status | How |
|----|--------|-----|
| What Changed events delivered as in-app notifications | 🟡 Partial | Read API ready; FE panel not wired; events only appear if source tables have rows |
| Each notification links to deal / account / Outreach Studio / activity | ✅ Backend | `linkType` + `linkId` on every item (see §6) |
| Activity Timeline opens filterable full feed | 🟡 Partial | `GET /api/activity-timeline` + `types` filter; FE page pending |
| Deal stage changes → CRM Update with old/new stage | ✅ Backend | `lvo_audit_log` diff → title `"Stage changed"`, summary `"before → after"` |
| Inbound emails → Email notification → Outreach Studio | 🟡 Partial | Surfaces inbound `lvo_activity`; email **ingest pipeline** not built |
| Risk signals → Risk notification | 🟡 Partial | Surfaces persisted `lvo_dealrisk`; depends on deal-health recalc job |
| Notifications delivered in real time | ❌ Pending | No SSE/WebSocket; FE must poll |
| Feed failure → ERR_MSG_0020 + retry; rest of Home works | 🟡 Partial | API returns `ERR_MSG_0020` on 500; retry UX is FE |
| Seller not notified for own CRM changes | ✅ Backend | Panel excludes audit rows where `lvo_changedby` matches `sellerId` or `X-User-Id` |
| 4–6 items, newest first | ✅ Backend | `limit` default 6, max 6; sorted `eventAt` desc |

Legend: ✅ Backend done · 🟡 Partial / cross-team · ❌ Not started

---

## 4. Endpoint reference

Base URL (local): `http://localhost:8000`
Base URL (dev via Nginx): `http://<dev-host>/api/...` (prefix already includes `/api`)

All responses use **camelCase** JSON (`APIModel`).

### 4.1 `GET /api/notifications`

**Caller:** Frontend — Home dashboard notification panel
**Purpose:** Return the seller's **4–6 most recent** qualifying portfolio events. Excludes CRM audit changes authored by the seller.

**Query parameters:**

| Param | Required | Default | Notes |
|-------|----------|---------|-------|
| `sellerId` | **yes** | — | UUID matching `opportunity.owninguser` |
| `limit` | no | `6` | Min 1, max 6 |
| `types` | no | all | Comma-separated: `email`, `meeting`, `crm_update`, `risk`, `task` (alias `crm` → `crm_update`) |

**Headers:**

| Header | Required | Notes |
|--------|----------|-------|
| `X-User-Id` | no | Seller email or UUID — used with `sellerId` to exclude self-authored CRM audit rows |

**Example:**

```http
GET /api/notifications?sellerId=A1B2C3D4-0000-0000-0000-000000000001&limit=6
X-User-Id: seller@lenovo.com
```

**200 response:**

```jsonc
{
  "sellerId": "A1B2C3D4-0000-0000-0000-000000000001",
  "limit": 6,
  "items": [
    {
      "id": "audit:LOG-UUID:stagename",
      "activityType": "crm_update",
      "title": "Stage changed",
      "summary": "'Proposal' → 'Negotiation'",
      "accountId": "6dc95c38-9237-4ce9-84d3-f5d1f7431965",
      "accountName": "Infosys Limited",
      "opportunityId": "F1A2B3C4-0000-0000-0000-000000000002",
      "opportunityName": "Infosys — Workstation Refresh",
      "eventAt": "2026-06-17T09:15:00",
      "isRead": false,
      "linkType": "opportunity",
      "linkId": "F1A2B3C4-0000-0000-0000-000000000002",
      "actor": "7D26391E-D020-474E-B1CA-53E6B6C71487",
      "actorName": "Priya Nair",
      "direction": null,
      "categoryLabel": "Stage progression tracked"
    }
  ]
}
```

**Errors:**

| Status | Body | When |
|--------|------|------|
| `422` | `{ "detail": "sellerId is required." }` | Missing `sellerId` |
| `500` | `{ "detail": "ERR_MSG_0020" }` | Aggregation failure — FE shows retry banner |

**FE guidance:**
- Poll every **30–60 seconds** while the notification panel is open (real-time AC).
- Render `eventAt` as relative time ("47 min ago") client-side.
- Map `activityType` to legend icons (§5).
- On click, navigate using `linkType` / `linkId` (§6).
- On `500` + `ERR_MSG_0020`, show retry; **do not block** the rest of the Home dashboard.

---

### 4.2 `GET /api/activity-timeline`

**Caller:** Frontend — Activity Timeline page (side nav)
**Purpose:** Full paginated portfolio feed. **Includes** seller-authored CRM changes (audit trail for history).

**Query parameters:**

| Param | Required | Default | Notes |
|-------|----------|---------|-------|
| `sellerId` | **yes** | — | Same as panel |
| `page` | no | `1` | 1-based |
| `pageSize` | no | `25` | Max `100` |
| `types` | no | all | Same filter as panel |

**Headers:** `X-User-Id` optional (currently unused for exclusion on timeline).

**Example:**

```http
GET /api/activity-timeline?sellerId=A1B2C3D4-...&page=1&pageSize=25&types=email,meeting,risk
```

**200 response:**

```jsonc
{
  "sellerId": "A1B2C3D4-...",
  "page": 1,
  "pageSize": 25,
  "total": 42,
  "totalPages": 2,
  "items": [ /* same shape as notification items */ ]
}
```

**Errors:** Same `422` / `500 ERR_MSG_0020` as panel.

**FE guidance:**
- Items beyond the panel's 6-item cap appear here (story scenario 1.1.1.2.3).
- Provide type filter chips wired to `types` query param.
- BACK button returns to Home — pure navigation, no API.

---

### 4.3 `PATCH /api/notifications/{notificationId}/read`

**Caller:** Frontend — when user views or dismisses a notification
**Purpose:** Idempotently mark one feed item as read for a seller.

**Path parameter:** `notificationId` — the item's `id` field (URL-encode colons).

**Query:** `sellerId` (required)

**Example:**

```http
PATCH /api/notifications/activity%3AE7CF7AAF-.../read?sellerId=A1B2C3D4-...
```

**200 response:**

```jsonc
{
  "sellerId": "A1B2C3D4-...",
  "notificationId": "activity:E7CF7AAF-...",
  "readAt": "2026-06-17T10:00:00"
}
```

**Requires migration:** `lvo_notification_read` table (§10). Without it, PATCH succeeds but `isRead` stays `false` on GET.

---

## 5. Activity types — API values vs UI legend

| `activityType` (API) | UI legend label | Source table | Inclusion rules |
|---------------------|-----------------|--------------|-----------------|
| `email` | Email Received | `lvo_activity` | `lvo_activitytype=email`, `lvo_direction≠outbound` |
| `meeting` | Meeting Done | `lvo_activity` | `lvo_activitytype=meeting` |
| `crm_update` | Stage Changed *(when field=stagename)* or CRM Update | `lvo_audit_log` | Opportunity `update` rows; title `"Stage changed"` for `stagename` |
| `risk` | Risk Detected | `lvo_dealrisk` | `statecode=Active`, within 30-day lookback |
| `task` | Task Overdue | `lvo_nextaction` | `status=Open`, `dueDate < today`, `statecode=Active` |

**Note:** Story lists "Stage Changed" as a distinct legend. Backend uses `activityType: "crm_update"` with `title: "Stage changed"` for stage moves. FE may show a distinct badge when `title === "Stage changed"`.

**Excluded from panel (not from timeline):**
- Outbound emails (`direction=outbound`)
- CRM audit rows where `actor` matches `sellerId` or `X-User-Id`

---

## 6. Deep links — FE routing contract

Each item includes `linkType` and `linkId`. Suggested navigation (confirm with Namisha):

| `linkType` | `linkId` | Navigate to |
|------------|----------|-------------|
| `outreach` | `opportunityId` | Outreach Studio for that deal |
| `opportunity` | `opportunityId` | Deal detail page |
| `activity` | `lvo_activityid` | Deal detail → activity / timeline offcanvas |
| `todo` | `lvo_nextactionid` | Deal detail → Next Actions / To-Do |
| `account` | `accountId` | Account detail *(reserved — not emitted in MVP)* |

`accountId` / `accountName` are always present when the deal has an account — use for display, not necessarily for primary navigation.

---

## 7. Synthetic feed item IDs

Stable keys used for read-state and PATCH path:

| Pattern | Example | Source |
|---------|---------|--------|
| `activity:{lvo_activityid}` | `activity:E7CF7AAF-...` | Activity row |
| `audit:{lvo_auditlogid}:{field}` | `audit:LOG1:stagename` | One per changed field in audit diff |
| `risk:{lvo_dealriskid}` | `risk:RISK-UUID` | Deal risk row |
| `task:{lvo_nextactionid}` | `task:ACTION-UUID` | Overdue next action |

---

## 8. Event producers — who writes data (not HTTP callers)

These teams/services **populate source tables**. The What Changed APIs only **read** them.

| Event | Expected producer | Target table | Status |
|-------|-------------------|--------------|--------|
| **Inbound email** | Email platform integration | `lvo_activity` (`type=email`, `direction=inbound`, linked `lvo_opportunityid`) | ❌ Not built |
| **Meeting done** | AIBackend / Note-Taking Agent (post-meeting) | `lvo_activity` (`type=meeting`) or future ingest | ❌ Not built |
| **CRM stage / field change** | D365 Sales `PATCH /api/opportunities/{id}` (existing) | `lvo_audit_log` | ✅ Already written by deal-update flow |
| **Risk detected** | Deal-health recalc (`POST .../health/recalculate`) | `lvo_dealrisk` | ✅ Exists — surfaces when risks are persisted |
| **Task overdue** | Seller creates next actions via existing APIs | `lvo_nextaction` | ✅ Query-on-read when due date passes |

### Suggested hooks for cross-team integration

**AIBackend (after `POST /transcripts/{id}/finalize`):**
- Write a `lvo_activity` row on the resolved `opportunity_id` **or**
- Call a future `POST /api/notifications/events` on D365 (not implemented in MVP).

**Email integration:**
- On inbound mail matched to account/contact → insert `lvo_activity` with rich `lvo_subject` / `lvo_body` for the summary line.

**AIBackend US04 (optional):**
- Risk tasks in AIBackend To-Do queue are **separate** from D365 `lvo_dealrisk`. Portfolio Risk notifications today come from D365 deal-health persistence only.

---

## 9. What's DONE — backend slice

| Artifact | Path |
|----------|------|
| Read-state migration | `sql/2026_07_create_lvo_notification_read.sql` |
| ORM model | `app/models.py` → `NotificationRead` |
| Pydantic schemas | `app/schemas.py` → `WhatChangedItem`, panel/timeline responses |
| Feed service | `app/services/what_changed.py` |
| HTTP routes | `app/routers/notifications.py` |
| Router registration | `app/main.py` |
| API contract | `API_CONTRACT.md` §15 |
| Unit tests | `tests/test_what_changed.py` — **11 passed** |

**Scope limits (intentional MVP):**
- Seller's **open** opportunities only (`statecode` Open or null)
- **30-day** lookback (`FEED_LOOKBACK_DAYS = 30`)
- No `GET /api/notifications/unread-count`
- No SSE / `?since=` polling cursor
- No dedicated event ingest endpoint

---

## 10. What's PENDING — by owner

### 10.1 Backend (D365 Sales) — Sanmay

| # | Item | Priority |
|---|------|----------|
| B1 | Run `sql/2026_07_create_lvo_notification_read.sql` on dev/staging Postgres | **P0** |
| B2 | Deploy updated D365 Sales image to dev (`:8000`) | **P0** |
| B3 | Integration/smoke tests against seeded Postgres data | P1 |
| B4 | `GET /api/notifications/unread-count` (if FE needs badge count) | P2 |
| B5 | `?since=` query param for efficient polling | P2 |
| B6 | SSE stream endpoint for real-time AC | P3 |
| B7 | Distinct `activityType: "stage_changed"` (vs generic `crm_update`) | P2 |
| B8 | Postman collection for the 3 endpoints | P2 |
| B9 | Optional `POST /api/notifications/events` ingest API | P3 |

~~B10~~ **`direction` + `actorName` + `categoryLabel`** — ✅ shipped on `WhatChangedItem`.

### 10.2 Frontend — Namisha

| # | Item |
|---|------|
| F1 | Notification panel on Home dashboard (4–6 cards) |
| F2 | Activity type badges/icons per §5 |
| F3 | Relative timestamps from `eventAt` |
| F4 | Deep-link navigation per §6 |
| F5 | Activity Timeline page + side-nav entry + pagination |
| F6 | Type filter UI → `types` query param |
| F7 | `ERR_MSG_0020` banner + retry button |
| F8 | Poll `GET /api/notifications` every 30–60 s while panel open |
| F9 | `PATCH .../read` on notification open/dismiss |
| F10 | Pass `sellerId` + `X-User-Id` from auth context |

### 10.3 AIBackend / AI team

| # | Item |
|---|------|
| A1 | Emit **Meeting Done** → `lvo_activity` (or ingest API) after transcript finalize |
| A2 | Confirm `opportunity_id` on meeting row is populated (US01) before writing activity |

### 10.4 Email / calendar integration

| # | Item |
|---|------|
| E1 | Inbound email detection → `lvo_activity` (`email` / `inbound`) |
| E2 | Calendar completion events (if not covered by bot meeting-done hook) |

### 10.5 DevOps

| # | Item |
|---|------|
| D1 | Build + deploy D365 Sales with Sprint 2 US 1.1 routes |
| D2 | Verify Nginx `/api/notifications` and `/api/activity-timeline` proxy to `:8000` |

---

## 11. Related endpoints (do not confuse)

| Endpoint | Scope | Use |
|----------|-------|-----|
| `GET /api/opportunities/{id}/timeline` | **Single deal** | Deal detail page timeline offcanvas |
| `GET /api/notifications` | **Seller portfolio** | Home notification panel |
| `GET /api/activity-timeline` | **Seller portfolio** | Activity Timeline page |

Per-deal timeline and portfolio feed share similar event shapes but different URLs and aggregation scope.

---

## 12. Local setup & smoke test

### Step 1 — Prerequisites

Base D365 seed loaded (`opportunity`, `account`, `systemuser`). Then run migrations (order matters):

```powershell
cd "c:\Users\sanmayan1\Documents\Cursour-Project\Lenovo%20D365%20Sales"

psql $env:DATABASE_URL -f sql/2026_06_create_lvo_activity.sql
psql $env:DATABASE_URL -f sql/2026_06_create_next_actions_audit.sql
psql $env:DATABASE_URL -f sql/2026_06_create_dealrisk.sql
psql $env:DATABASE_URL -f sql/2026_07_create_lvo_notification_read.sql
psql $env:DATABASE_URL -f sql/2026_07_seed_what_changed_local_smoke.sql
```

The smoke seed inserts **7 rows** (one per notification type + exclusion cases) for test seller:

Test sellers:

```
AB3499B1-B088-4F86-B9F2-E458F663ECBF  → Deutsche Bank deal (smoke rows 1–7)
055DAFE7-9840-451D-8328-5F70A6326C03  → Ford Motor deal (smoke rows at end of SQL)
```

### Step 2 — Start API

```powershell
uvicorn app.main:app --reload --port 8000
```

Swagger: http://localhost:8000/docs

### Step 3 — Call the endpoints

```http
### Panel — expect ~5 items (excludes seller's own audit + outbound email)
GET http://localhost:8000/api/notifications?sellerId=AB3499B1-B088-4F86-B9F2-E458F663ECBF&limit=6
X-User-Id: AB3499B1-B088-4F86-B9F2-E458F663ECBF

### Timeline — expect ~6 items (includes seller's own CRM edit)
GET http://localhost:8000/api/activity-timeline?sellerId=AB3499B1-B088-4F86-B9F2-E458F663ECBF&page=1&pageSize=25

### Mark read
PATCH http://localhost:8000/api/notifications/activity%3Ab1010001-0001-4001-8001-000000000001/read?sellerId=AB3499B1-B088-4F86-B9F2-E458F663ECBF
```

### Step 4 — Verify smoke expectations

| Smoke row | Expected in panel | Expected in timeline |
|-----------|:-----------------:|:--------------------:|
| SMOKE-0001 inbound email (`b1010001-…001`) | ✅ | ✅ — `direction: inbound`, `categoryLabel: Email received` |
| SMOKE-0002 meeting (`b1020002-…002`) | ✅ | ✅ — `direction: outbound` |
| SMOKE-0003 stage change (other user) | ✅ | ✅ — `title: Stage changed`, `categoryLabel: Stage progression tracked` |
| SMOKE-0004 risk | ✅ | ✅ |
| SMOKE-0005 overdue task | ✅ | ✅ |
| SMOKE-0006 seller's own CRM edit | ❌ | ✅ |
| SMOKE-0007 outbound email | ❌ | ❌ |

### Step 5 — Trigger a live CRM change (no SQL)

```http
PATCH http://localhost:8000/api/opportunities/CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B
Content-Type: application/json
X-User-Id: 7D26391E-D020-474E-B1CA-53E6B6C71487

{ "stage": "Negotiation" }
```

Re-fetch `/api/notifications` — a new `audit:…:stagename` item should appear (not visible to seller `AB3499B1…` in panel if they PATCH with their own `X-User-Id`).

### Step 6 — Unit tests

```powershell
python -m pytest tests/test_what_changed.py -v
```

Expected: **14 passed** (3 new tests for direction / category label).

---

## 13. Open questions (confirm with FE / product)

| # | Question | Default assumption |
|---|----------|-------------------|
| 1 | How is `sellerId` obtained at runtime? | `opportunity.owninguser` UUID from auth mock / SSO later |
| 2 | Is polling (30–60 s) acceptable for "real time" in Sprint 2? | Yes for MVP |
| 3 | Exact Outreach Studio route for `linkType=outreach`? | `/opportunities/{id}/outreach` (TBD) |
| 4 | Should closed-deal activity appear in portfolio feed? | MVP: open deals only |
| 5 | Unread badge — count all unread or only panel-visible 6? | Needs `unread-count` endpoint if badge required |

---

## 14. Standup one-liner

> **Backend for US 1.1 shipped** — panel + timeline APIs with `direction`, `actorName`, `categoryLabel`; smoke SQL + handoff doc ready. **Next:** migration/deploy, FE wiring, email + meeting-done producers.

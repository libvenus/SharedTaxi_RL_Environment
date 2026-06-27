# Sprint 2 · US 1.2 — Quarter Pulse Summary · Backend Handoff

**Sprint:** 2  
**User Story:** 1.2 — Quarter Pulse Summary (Home dashboard card)  
**Epic:** Home Dashboard — AI Sales Agent landing experience  
**Backend status:** MVP read + manual quota write shipped · unit tests for fiscal calendar + metric bands  
**Backend repo:** `Lenovo D365 Sales`  
**Backend contact:** Sanmay  
**Pairing:** Sanmay (backend) ↔ Namisha (FE)  
**Document audience:** Frontend, DevOps, QA  
**Linked docs:**
- [API_CONTRACT.md](./API_CONTRACT.md) — §16 (canonical JSON shapes)
- [sql/2026_08_create_lvo_seller_quota.sql](./sql/2026_08_create_lvo_seller_quota.sql) — quota migration
- [sql/2026_08_seed_quarter_pulse_local_smoke.sql](./sql/2026_08_seed_quarter_pulse_local_smoke.sql) — optional local quota seed
- `app/services/quarter_pulse.py` — metric aggregation
- `app/services/fiscal_calendar.py` — fiscal quarter boundaries
- `app/routers/quarter_pulse.py` — HTTP routes

**Last updated:** 2026-06-19

---

## UI screen → API mapping (FE mockup)

Home dashboard · right column · **Q3 Pulse** card (`Dashboard.jsx` today is static).

```
┌─────────────────────────────────────┐
│ Q3 Pulse              27 days left  │  ← quarterLabel + daysLeftInQuarter
├─────────────────────────────────────┤
│ Quota Attainment              60%   │  ← quotaAttainment.displayValue
│ [████████████░░░░░░░░]  blue bar    │  ← progressFillPercent + barColor
├─────────────────────────────────────┤
│ Pipeline Coverage            2.4x   │  ← pipelineCoverage.displayValue
│ [████████████████░░░░] yellow bar  │  ← progressFillPercent + barColor
├─────────────────────────────────────┤
│ Last updated: 19 Jun 2026, 10:15    │  ← lastUpdatedAt (FE formats)
└─────────────────────────────────────┘
```

| UI element | API field | FE notes |
|------------|-----------|----------|
| Card title **"Q3 Pulse"** | `quarterLabel` + `fiscalYear` | Suggested: ``{quarterLabel} Pulse`` e.g. `"Q3 Pulse"` |
| **"27 days left"** | `daysLeftInQuarter` | Integer; show `"1 day left"` singular when `=== 1` |
| **Quota Attainment %** | `quotaAttainment.displayValue` | e.g. `"60%"` or `"Not set"` |
| Blue progress bar | `quotaAttainment.progressFillPercent`, `barColor` | Width = `progressFillPercent`; color from `barColor` (`red` / `blue` / `green`) |
| **Pipeline Coverage** | `pipelineCoverage.displayValue` | e.g. `"2.4x"` or `"Not set"` |
| Yellow progress bar | `pipelineCoverage.progressFillPercent`, `barColor` | `yellow` for medium band on coverage |
| **Last updated** | `lastUpdatedAt` | Format relative or absolute client-side |
| Quota-not-set prompt | `prompt` | Show when `quotaConfigured === false` |
| Admin quota edit (Phase 1) | `PUT /api/quarter-pulse/quota` | After save, re-fetch GET or use `quarterPulse` in PUT response |

> **Not the same as `quote` table:** D365 `quote` is a per-opportunity pricing document. Quarter Pulse quota lives in `lvo_seller_quota` (seller fiscal target), not `quote`.

---

## TL;DR — who calls which API

| # | Endpoint | Caller | When | Purpose |
|---|----------|--------|------|---------|
| 1 | `GET /api/quarter-pulse` | **Frontend** (Home dashboard) | On Home load; refresh daily / on focus | Quota attainment, pipeline coverage, days left |
| 2 | `PUT /api/quarter-pulse/quota` | **Frontend** (admin / Phase-1 quota form) | Seller manager sets quota when D365 has none | Upsert manual quota for current fiscal quarter |

**Metrics are computed live** from `opportunity` on every GET (no nightly snapshot job for this card).  
**Routing on dev:** Nginx `location /api/` → D365 Sales container (`:8000`).

---

## 1. Story in one paragraph

Sellers see a compact **Quarter Pulse** card on the Home screen with **quota attainment %**, **pipeline coverage ratio**, and **days remaining** in the fiscal quarter. Data is recalculated from the latest D365 mirror on each load; `lastUpdatedAt` reflects server compute time. When no quota is configured, attainment and coverage show **"Not set"** with a prompt; days left still render. Phase 1 allows manual quota entry via admin UI when D365 has no goal row.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  GET /api/quarter-pulse?sellerId=...                             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
  fiscal_calendar.py   opportunity (D365)   lvo_seller_quota
  (quarter bounds,     owninguser filter    manual / future D365
   days left)          won + open sums       quota target
```

### Metric formulas (MVP)

| Metric | Formula | Source |
|--------|---------|--------|
| **Closed revenue** | `SUM(estimatedvalue)` | Seller's **Won** deals with `estimatedclosedate` in current fiscal quarter |
| **Quota attainment %** | `(closed_revenue / quota_target) × 100` | Requires quota |
| **Open pipeline value** | `SUM(estimatedvalue)` | Seller's deals with `statecode = 'Open'` |
| **Pipeline coverage** | `open_pipeline / remaining_quota` | `remaining_quota = max(quota_target - closed_revenue, 0)`; displayed as `Nx` |
| **Days left** | `fiscal_quarter_end - today` | `fiscal_calendar.py` |

### Progress bar bands

**Quota attainment**

| % | `band` | `barColor` |
|---|--------|------------|
| &lt; 50 | `low` | `red` |
| 50 – 79 | `medium` | `blue` |
| ≥ 80 | `high` | `green` |

**Pipeline coverage**

| Ratio | `band` | `barColor` |
|-------|--------|------------|
| &lt; 1.0x | `low` | `red` |
| 1.0 – 2.0x | `medium` | `yellow` |
| &gt; 2.0x | `high` | `green` |

Coverage bar width: `min(ratio / 3.0 × 100, 100)`.

### Fiscal calendar

- Default **FY starts April** (`FISCAL_YEAR_START_MONTH=4` in `.env`).
- FY naming: **FY2026** = Apr 2025 – Mar 2026.
- Quarters: Q1 Apr–Jun, Q2 Jul–Sep, Q3 Oct–Dec, Q4 Jan–Mar.

---

## 3. Acceptance criteria traceability

| AC | Backend status |
|----|----------------|
| Quota attainment %, pipeline coverage, days left | ✅ `GET /api/quarter-pulse` |
| Progress bar fill + color from bands | ✅ `progressFillPercent`, `barColor`, `band` on each metric |
| Recalculated from live D365 data on each load | ✅ Live SQL aggregates; `lastUpdatedAt` on response |
| Quota not configured → "Not set" + prompt; days left still shown | ✅ `quotaConfigured: false` |
| Manual quota when D365 unset (Phase 1) | ✅ `PUT /api/quarter-pulse/quota` |
| D365 quota entity mirror | ❌ Pending — `source='d365'` reserved |

---

## 4. Endpoint reference

Base URL (local): `http://localhost:8000`  
Base URL (dev): `http://<dev-host>/api/...` (Nginx already prefixes `/api`)

All responses use **camelCase** JSON (`APIModel`).

### 4.1 `GET /api/quarter-pulse`

**Caller:** Frontend — Home dashboard Quarter Pulse card  
**Purpose:** Return seller-scoped quarter metrics for the current fiscal period.

**Query parameters**

| Param | Required | Notes |
|-------|----------|-------|
| `sellerId` | **yes** | UUID — matches `opportunity.owninguser` |

**Example**

```http
GET /api/quarter-pulse?sellerId=055DAFE7-9840-451D-8328-5F70A6326C03
```

**200 — quota configured**

```jsonc
{
  "quarterLabel": "Q1",
  "fiscalYear": 2027,
  "daysLeftInQuarter": 11,
  "lastUpdatedAt": "2026-06-19T10:15:00",
  "quotaConfigured": true,
  "quotaTarget": 1000000.0,
  "closedRevenue": 600000.0,
  "openPipelineValue": 960000.0,
  "openDealCount": 12,
  "quotaAttainment": {
    "displayValue": "60%",
    "percent": 60.0,
    "progressFillPercent": 60.0,
    "band": "medium",
    "barColor": "blue"
  },
  "pipelineCoverage": {
    "displayValue": "2.4x",
    "ratio": 2.4,
    "progressFillPercent": 80.0,
    "band": "high",
    "barColor": "yellow"
  },
  "prompt": null
}
```

**200 — quota not configured**

```jsonc
{
  "quarterLabel": "Q1",
  "fiscalYear": 2027,
  "daysLeftInQuarter": 11,
  "lastUpdatedAt": "2026-06-19T10:15:00",
  "quotaConfigured": false,
  "quotaTarget": null,
  "closedRevenue": 250000.0,
  "openPipelineValue": 500000.0,
  "openDealCount": 5,
  "quotaAttainment": {
    "displayValue": "Not set",
    "percent": null,
    "progressFillPercent": null,
    "band": null,
    "barColor": null
  },
  "pipelineCoverage": {
    "displayValue": "Not set",
    "ratio": null,
    "progressFillPercent": null,
    "band": null,
    "barColor": null
  },
  "prompt": "Set your quota target in D365 to see attainment metrics."
}
```

**Errors**

| Status | Body | When |
|--------|------|------|
| `422` | `{ "detail": "sellerId is required." }` | Missing `sellerId` |
| `500` | `{ "detail": "ERR_MSG_0021" }` | Aggregation failure |

**FE guidance**
- Call on **Home mount** and when user returns to Home (satisfies daily refresh AC).
- Optional: refresh on window focus or once per calendar day client-side.
- Do **not** block Home if this call fails — show card error state + retry.
- Pass `sellerId` from auth context (same as What Changed panel).

---

### 4.2 `PUT /api/quarter-pulse/quota`

**Caller:** Frontend — admin / quota configuration UI (Phase 1)  
**Purpose:** Create or update manual quota for a seller + fiscal period.

**Query parameters**

| Param | Required | Notes |
|-------|----------|-------|
| `sellerId` | **yes** | Seller receiving the quota |

**Headers**

| Header | Required | Notes |
|--------|----------|-------|
| `Content-Type` | **yes** | `application/json` |
| `X-User-Id` | no | Audit trail → `setBy` on stored row |

**Body**

```jsonc
{
  "quotaAmount": 1000000,
  "fiscalYear": 2027,
  "fiscalQuarter": 1,
  "currencyCode": "USD"
}
```

`fiscalYear` and `fiscalQuarter` are optional — default to **current fiscal period** when omitted.

**Example**

```http
PUT /api/quarter-pulse/quota?sellerId=055DAFE7-9840-451D-8328-5F70A6326C03
Content-Type: application/json
X-User-Id: manager@lenovo.com

{
  "quotaAmount": 1000000,
  "currencyCode": "USD"
}
```

**200 response**

```jsonc
{
  "sellerId": "055DAFE7-9840-451D-8328-5F70A6326C03",
  "fiscalYear": 2027,
  "fiscalQuarter": 1,
  "quotaAmount": 1000000.0,
  "currencyCode": "USD",
  "source": "manual",
  "setBy": "manager@lenovo.com",
  "modifiedAt": "2026-06-19T10:20:00",
  "quarterPulse": { /* same shape as GET */ }
}
```

**Errors**

| Status | Body | When |
|--------|------|------|
| `422` | `{ "detail": "sellerId is required." }` | Missing `sellerId` |
| `422` | `{ "detail": "quota_amount must be positive" }` | Invalid body |
| `503` | `{ "detail": "lvo_seller_quota table is not available" }` | Migration not run |
| `500` | `{ "detail": "ERR_MSG_0021" }` | Server error |

---

## 5. Related endpoints (do not confuse)

| Endpoint | Scope | Use |
|----------|-------|-----|
| `GET /api/opportunities/kpi-summary` | **Org-wide** KPI strip | Opportunities page — not seller-scoped |
| `GET /api/notifications` | Seller portfolio feed | What Changed panel (US 1.1) |
| `GET /api/quarter-pulse` | **Seller** quarter metrics | Home Quarter Pulse card (US 1.2) |
| `quote` / `lvo_quoteitem` tables | Per-deal product quotes | Product filters — **not** sales quota |

---

## 6. What's DONE — backend slice

| Artifact | Path |
|----------|------|
| Quota migration | `sql/2026_08_create_lvo_seller_quota.sql` |
| Local quota seed (optional) | `sql/2026_08_seed_quarter_pulse_local_smoke.sql` |
| ORM model | `app/models.py` → `SellerQuota` |
| Pydantic schemas | `app/schemas.py` → `QuarterPulseResponse`, quota upsert models |
| Fiscal calendar | `app/services/fiscal_calendar.py` |
| Metric service | `app/services/quarter_pulse.py` |
| HTTP routes | `app/routers/quarter_pulse.py` |
| Router registration | `app/main.py` (v0.16.0) |
| API contract | `API_CONTRACT.md` §16 |
| Unit tests | `tests/test_fiscal_calendar.py`, `tests/test_quarter_pulse.py` |

**Scope limits (intentional MVP):**
- Quota from **manual table only** — D365 goal entity not wired
- Open pipeline = all `statecode = Open` deals (not forecast-category subset)
- Won revenue uses `estimatedclosedate` (no separate `actualclosedate` in mirror)
- No role-based auth on quota PUT — `sellerId` + optional `X-User-Id` only

---

## 7. What's PENDING — by owner

### 7.1 Backend — Sanmay

| # | Item | Priority |
|---|------|----------|
| B1 | Run `sql/2026_08_create_lvo_seller_quota.sql` on dev/staging Postgres | **P0** |
| B2 | Deploy D365 Sales v0.16.0 to dev | **P0** |
| B3 | Confirm D365 quota entity name with data team; import with `source=d365` | P1 |
| B4 | Integration tests against seeded Postgres | P1 |
| B5 | Postman collection (see §9 curl commands) | P2 |

### 7.2 Frontend — Namisha

| # | Item |
|---|------|
| F1 | Replace static Q3 Pulse mock in `Dashboard.jsx` with `GET /api/quarter-pulse` |
| F2 | Progress bars from `progressFillPercent` + `barColor` |
| F3 | Card title from `quarterLabel`; days left from `daysLeftInQuarter` |
| F4 | Show `prompt` when `quotaConfigured === false` |
| F5 | Display `lastUpdatedAt` on card |
| F6 | Admin quota form → `PUT /api/quarter-pulse/quota` |
| F7 | Pass `sellerId` from auth (same as notifications) |

### 7.3 DevOps

| # | Item |
|---|------|
| D1 | Deploy image with Quarter Pulse routes |
| D2 | Verify Nginx proxies `/api/quarter-pulse` to D365 Sales `:8000` |

---

## 8. Local setup & smoke test

### Step 1 — Run migration

```powershell
cd "c:\Users\sanmayan1\Documents\Cursour-Project\Lenovo%20D365%20Sales"

psql $env:DATABASE_URL -f sql/2026_08_create_lvo_seller_quota.sql
```

### Step 2 — Find a seller UUID

```sql
SELECT DISTINCT owninguser
FROM opportunity
WHERE owninguser IS NOT NULL
LIMIT 10;
```

Smoke-test sellers from What Changed seed:

```
AB3499B1-B088-4F86-B9F2-E458F663ECBF
055DAFE7-9840-451D-8328-5F70A6326C03
```

### Step 3 — Optional quota seed

Edit `sql/2026_08_seed_quarter_pulse_local_smoke.sql` — replace `SELLER-UUID-REPLACE-ME` — then:

```powershell
psql $env:DATABASE_URL -f sql/2026_08_seed_quarter_pulse_local_smoke.sql
```

### Step 4 — Start API

```powershell
uvicorn app.main:app --reload --port 8000
```

Swagger: `http://localhost:8000/docs` → tag **quarter-pulse**

### Step 5 — Smoke sequence

1. `GET /api/quarter-pulse?sellerId=...` → expect `quotaConfigured: false` (no seed)
2. `PUT /api/quarter-pulse/quota?sellerId=...` with `quotaAmount`
3. `GET` again → expect attainment % and coverage ratio populated

---

## 9. cURL commands (Postman-ready)

Set variables (PowerShell):

```powershell
$BASE = "http://localhost:8000"
$SELLER = "055DAFE7-9840-451D-8328-5F70A6326C03"
```

### Health check

```bash
curl -s "%BASE%/health"
```

### GET Quarter Pulse (no quota yet)

```bash
curl -s -X GET "%BASE%/api/quarter-pulse?sellerId=%SELLER%"
```

### PUT manual quota (current fiscal quarter)

```bash
curl -s -X PUT "%BASE%/api/quarter-pulse/quota?sellerId=%SELLER%" ^
  -H "Content-Type: application/json" ^
  -H "X-User-Id: manager@lenovo.com" ^
  -d "{\"quotaAmount\": 1000000, \"currencyCode\": \"USD\"}"
```

### PUT quota (explicit fiscal period)

```bash
curl -s -X PUT "%BASE%/api/quarter-pulse/quota?sellerId=%SELLER%" ^
  -H "Content-Type: application/json" ^
  -H "X-User-Id: manager@lenovo.com" ^
  -d "{\"quotaAmount\": 1500000, \"fiscalYear\": 2027, \"fiscalQuarter\": 1, \"currencyCode\": \"USD\"}"
```

### GET Quarter Pulse (after quota set)

```bash
curl -s -X GET "%BASE%/api/quarter-pulse?sellerId=%SELLER%"
```

### Negative — missing sellerId (expect 422)

```bash
curl -s -X GET "%BASE%/api/quarter-pulse"
```

### Linux / macOS (copy-paste)

```bash
export BASE=http://localhost:8000
export SELLER=055DAFE7-9840-451D-8328-5F70A6326C03

curl -s "$BASE/health" | jq .

curl -s "$BASE/api/quarter-pulse?sellerId=$SELLER" | jq .

curl -s -X PUT "$BASE/api/quarter-pulse/quota?sellerId=$SELLER" \
  -H "Content-Type: application/json" \
  -H "X-User-Id: manager@lenovo.com" \
  -d '{"quotaAmount": 1000000, "currencyCode": "USD"}' | jq .

curl -s "$BASE/api/quarter-pulse?sellerId=$SELLER" | jq .
```

**Postman:** Import → Raw text → paste any curl above, or create a collection with:
- `{{baseUrl}}` = `http://localhost:8000`
- `{{sellerId}}` = your `owninguser` UUID

---

## 10. Error codes

| Code | Meaning | FE action |
|------|---------|-----------|
| `ERR_MSG_0021` | Quarter Pulse aggregation or quota save failed | Show retry on card; log support reference |

---

## 11. Config

| Env var | Default | Purpose |
|---------|---------|---------|
| `FISCAL_YEAR_START_MONTH` | `4` | April FY start |
| `DATABASE_*` | — | Postgres mirror (same as rest of API) |

See `.env.example`.

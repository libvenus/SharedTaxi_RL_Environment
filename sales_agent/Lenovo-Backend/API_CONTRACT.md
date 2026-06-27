# Lenovo D365 Sales — API Contract
**Base URL:** `http://localhost:8000` (dev) | TBD (staging/prod)  
**Version:** 0.19.0  
**All responses:** JSON, camelCase keys  
**Auth header (write endpoints):** `X-User-Id: <user-email>` — optional but recommended for audit trail  

---

## Common Conventions

### HTTP Status Codes
| Code | Meaning |
|------|---------|
| 200 | Success (GET, PATCH) |
| 201 | Created (POST) |
| 400 | Business rule violation (e.g. editing a closed deal) |
| 404 | Resource not found |
| 409 | Conflict (duplicate name / duplicate competitor) |
| 422 | Validation error (bad input) |
| 500 | Unexpected server error |

### Standard Error Body
```json
{
  "detail": {
    "code": "ERROR_CODE",
    "message": "Human-readable message"
  }
}
```

### Error Codes Reference
| Code | Trigger |
|------|---------|
| `ERR_MSG_0012` | `estimatedValue` is exactly `0` |
| `DUPLICATE_NAME` | Another deal already has the same name |
| `DUPLICATE_COMPETITOR` | Competitor already active on this deal |
| `DUPLICATE_CONTACT` | Contact is already attached to this deal **or** to this account |
| `CLOSED_DEAL_LOCKED` | Editing locked fields on a Closed Won/Lost deal |
| `ALREADY_CANCELED` | DELETE on a deal that is already `Canceled` |
| `INVALID_FORECAST_CATEGORY` | Forecast category not valid for the deal stage |
| `INVALID_PARENT_OPPORTUNITY` | `parentOpportunityId` references self / a cancelled deal / a descendant (cycle) / a non-existent deal |
| `VALIDATION_ERROR` | Missing mandatory field or invalid value |
| `ERR_MSG_0008` | DELETE on an account-contact link whose contact is the **primary** contact |
| `ERR_MSG_0009` | DELETE on an account-contact link whose contact is referenced by an **active deal** |
| `ERR_MSG_0010` | Account list returned zero rows after filters were applied (rendered as a **note** in the response, not as an error body) |
| `ERR_MSG_0013` | Invalid email or phone format on add / update of an account contact |
| `MISSING_NAME` | `GET /api/contacts/search` called without a usable `name` query param |
| `EMPTY_EMAILS` | `POST /api/contacts/resolve-by-emails` body has no usable email addresses |

### Success / Confirmation Codes
The account-contacts endpoints embed these in their response bodies so the FE can render the matching toast / dialog without hard-coding strings:

| Code | Surfaced by |
|------|-------------|
| `SUCC_MSG_0007` | Successful `DELETE /api/accounts/{id}/contacts/{linkId}` |
| `SUCC_MSG_0008` | Successful `POST /api/accounts/{id}/contacts` (audit log diff) |
| `SUCC_MSG_0009` | Successful `PATCH /api/accounts/{id}/contacts/{linkId}` (audit log diff) |
| `CONF_MSG_0003` | `GET /api/accounts/{id}/contacts/{linkId}/delete-eligibility` when `canDelete=true` |

---

## 1. Meta

### GET `/health`
Liveness probe.

**Response 200**
```json
{ "status": "ok" }
```

---

## 2. Filters

### GET `/api/filters/regions`
Returns all business groups, countries, and geographic units for the filter dropdowns.

**Response 200**
```json
{
  "businessGroups": [
    { "id": "EMEA", "label": "EMEA" }
  ],
  "countries": [
    { "code": "DE", "label": "DE", "businessGroup": "EMEA" }
  ],
  "geographicUnits": [
    { "id": "uuid", "name": "Central Europe", "code": "CE", "parentId": null }
  ]
}
```

---

### GET `/api/filters/industries`
Returns all distinct industry codes.

**Response 200**
```json
{
  "total": 12,
  "items": [
    { "code": "Technology", "label": "Technology" }
  ]
}
```

---

### GET `/api/filters/stages`
Returns all distinct deal stages.

**Response 200**
```json
{
  "total": 6,
  "items": [
    { "raw": "Qualify", "label": "Qualification" },
    { "raw": "Develop", "label": "Discovery" },
    { "raw": "Propose", "label": "Proposal" },
    { "raw": "Execute", "label": "Negotiation" },
    { "raw": "Closed Won", "label": "Closed Won" },
    { "raw": "Closed Lost", "label": "Closed Lost" }
  ]
}
```

---

### GET `/api/filters/products`
Returns all distinct product series / solution offerings.

**Response 200**
```json
{
  "total": 8,
  "items": [
    { "id": "thinkpad", "label": "ThinkPad", "source": "quoteitem" },
    { "id": "daas-managed-device-bundle", "label": "DaaS Managed Device Bundle", "source": "solutionoffering" }
  ]
}
```

---

## 3. Opportunities — Read

### GET `/api/opportunities/kpi-summary`
KPI cards strip — Open Deals, Pipeline (UI-labelled "Identified"), Best Case,
Commit, Most Likely, Won, Loss.

**Bucket → predicate mapping**

| Bucket field | UI label | Predicate |
|---|---|---|
| `openDeals` | Open Deals | `opportunity.statecode = 'Open'` |
| `pipeline` | **Identified** *(UI relabel)* | `opportunity.lvo_forecastcategory = 'Pipeline'` |
| `bestCase` | Best Case | `opportunity.lvo_forecastcategory = 'Best Case'` |
| `commit` | Commit | `opportunity.lvo_forecastcategory = 'Commit'` |
| `mostLikely` | Most Likely | `opportunity.lvo_forecastcategory = 'Most Likely'` |
| `won` | Won | `statecode IN ('Won','Closed Won')` OR `stagename='Closed Won'` |
| `loss` | Loss | `statecode IN ('Lost','Closed Lost')` OR `stagename='Closed Lost'` |

> The `pipeline` field name is preserved on the wire even though the UI now
> renders it as "Identified" — keeping the field name stable means existing
> snapshot rows, drill-down URLs (`?bucket=pipeline`) and unit tests continue
> to work without migration.

**Query Parameters**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `comparePeriod` | `last_week` \| `past_month` \| `last_quarter` | No (default: `last_week`) | Period for trend comparison |
| `search` | string | No | Free-text search on deal name / account name |
| `regions` | string | No | Comma-separated business groups or country codes |
| `industries` | string | No | Comma-separated industry codes |
| `stages` | string | No | Comma-separated raw stage values |
| `products` | string | No | Comma-separated product slugs |

**Response 200**
```json
{
  "comparePeriod": "last_week",
  "asOf": "2026-06-09T10:00:00Z",
  "currency": "USD",
  "openDeals":  {
    "value": 12500000.0,
    "count": 18,
    "trend": { "direction": "up",   "deltaValue":  450000.0, "deltaCount":  2 }
  },
  "pipeline":   {
    "value": 4200000.0,
    "count": 7,
    "trend": { "direction": "up",   "deltaValue":   80000.0, "deltaCount":  1 }
  },
  "bestCase":   {
    "value": 3100000.0,
    "count": 5,
    "trend": { "direction": "flat", "deltaValue":       0.0, "deltaCount":  0 }
  },
  "commit":     {
    "value": 1800000.0,
    "count": 3,
    "trend": { "direction": "down", "deltaValue": -120000.0, "deltaCount": -1 }
  },
  "mostLikely": {
    "value": 2670000.0,
    "count": 11,
    "trend": { "direction": "up",   "deltaValue":  230000.0, "deltaCount":  1 }
  },
  "won":        {
    "value":  950000.0,
    "count": 4,
    "trend": { "direction": "up",   "deltaValue":  150000.0, "deltaCount":  1 }
  },
  "loss":       {
    "value":  320000.0,
    "count": 2,
    "trend": { "direction": "flat", "deltaValue":       0.0, "deltaCount":  0 }
  },
  "notes": []
}
```

#### Trend semantics (v1)

* `trend` is computed by comparing today's bucket aggregate against the
  most-recent row in `lvo_opportunitysnapshot` whose `lvo_snapshotdate`
  is on or before `today − N days`, where `N` is `7` (`last_week`),
  `30` (`past_month`) or `90` (`last_quarter`).
* `direction` is the **sign of `deltaValue`**. Ties (`deltaValue == 0`)
  fall through to `deltaCount`; `flat` only when both deltas are zero.
* **Trend is `null`** in any of these cases — `notes` will explain which
  applies:
  * Any filter is set on the request (`search`, `regions`, `industries`,
    `stages`, `products`). v1 only stores **global** snapshots; per-
    dimension snapshots are a planned follow-up. Re-issue without
    filters to get a comparison.
  * No snapshot row exists ≤ the lookback date. Run
    `python -m app.jobs.snapshot_kpis --backfill` once and schedule
    `python -m app.jobs.snapshot_kpis` nightly.

#### How snapshots are produced

| Job | Cadence | Effect |
|-----|---------|--------|
| `python -m app.jobs.snapshot_kpis` | Nightly | UPSERTs one row per bucket for today (`lvo_snapshotdate = today`). |
| `python -m app.jobs.snapshot_kpis --date YYYY-MM-DD` | Ad-hoc | Records a snapshot for an arbitrary calendar date. Idempotent. |
| `python -m app.jobs.snapshot_kpis --backfill` | One-time | Generates rows for `today`, `today-7`, `today-30`, `today-90` using `opportunity.lvo_createdat ≤ snapshot_date` as the existed-by filter. **Won/loss buckets are approximate** — the snapshot service has no record of historical state changes. |
| `python -m app.jobs.snapshot_kpis --backfill-dates 2026-03-01,2026-04-01` | One-time | Same as `--backfill` but with custom dates. |

The snapshot table is defined by `sql/2026_06_create_opportunity_snapshot.sql`
and uses a unique index on `(lvo_snapshotdate, lvo_bucket)` so re-running
the job for the same date overwrites rather than duplicates.

---

### GET `/api/opportunities`
Paginated deal list for the main grid.

**Query Parameters**
| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `page` | integer ≥ 1 | No | `1` | Page number |
| `pageSize` | integer 1–100 | No | `10` | Items per page |
| `search` | string | No | — | Free-text on name / account |
| `regions` | string | No | — | Comma-separated BG or country codes |
| `industries` | string | No | — | Comma-separated industry codes |
| `stages` | string | No | — | Comma-separated raw stage values |
| `products` | string | No | — | Comma-separated product slugs |
| `bucket` | `open` \| `pipeline` \| `best_case` \| `commit` \| `most_likely` \| `won` \| `loss` | No | — | KPI card click filter (predicate matches `/kpi-summary`) |
| `sortBy` | `name` \| `value` \| `closeDate` \| `closeProbability` \| `stage` | No | `closeDate` | Sort column |
| `sortOrder` | `asc` \| `desc` | No | `desc` | Sort direction |

**Response 200**
```json
{
  "page": 1,
  "pageSize": 10,
  "total": 47,
  "totalPages": 5,
  "sortBy": "closeDate",
  "sortOrder": "desc",
  "items": [
    {
      "id": "116191c4-ce2f-46cb-8666-4861a6a1ae26",
      "name": "HSBC – DaaS Rollout APAC",
      "accountId": "acc-uuid",
      "accountName": "HSBC",
      "industry": "Banking",
      "country": "HK",
      "region": "APAC BG",
      "stage": { "raw": "Propose", "label": "Proposal" },
      "saleMotion": { "raw": "Renewal", "label": "Renewal" },
      "forecastCategory": "Commit",
      "value": 3200000.0,
      "currency": "USD",
      "closeDate": "2026-08-02",
      "closeProbability": 60.0,
      "competitorCount": 2,
      "competitors": ["Dell Technologies", "HP Inc"],
      "ownerId": "user-uuid",
      "statecode": "Open",
      "risk": "Budget Freeze",
      "riskScore": 3,
      "dealHealth": 72,
      "nextAction": null,
      "lastActivity": "2026-06-01T14:30:00Z",
      "activities": [
        {
          "id": "act-uuid",
          "type": "email",
          "direction": "outbound",
          "subject": "Proposal follow-up",
          "body": null,
          "activityDate": "2026-06-01T14:30:00Z",
          "groupedCount": null
        }
      ]
    }
  ],
  "notes": []
}
```

---

### GET `/api/opportunities/{opportunityId}/competitors`
List all active competitors for a deal.

**Path Parameters**
| Param | Description |
|-------|-------------|
| `opportunityId` | opportunity UUID |

**Response 200**
```json
{
  "opportunityId": "116191c4-ce2f-46cb-8666-4861a6a1ae26",
  "total": 2,
  "items": [
    {
      "id": "comp-uuid",
      "opportunityId": "116191c4-...",
      "name": "Dell Technologies",
      "competitorName": "Dell Technologies",
      "competitorType": "Incumbent",
      "resellingPartnerId": "Tech Data"
    }
  ]
}
```

---

### GET `/api/opportunities/{opportunityId}/sale-motion`
Sale motion pill value for a deal.

**Response 200**
```json
{
  "opportunityId": "116191c4-...",
  "raw": "Renewal",
  "label": "Renewal"
}
```

---

## 4. Deal Update — Write

### PATCH `/api/opportunities/{opportunityId}`
Update editable fields on an existing deal — also serves the **Update**
button on the Complete-Information form (same endpoint, same field names).

> **Rules:**
> - All fields are optional (true PATCH — only sent fields are applied).
> - `stage` is **never** writable via this endpoint.
> - For **Closed Won / Closed Lost** deals: `name`, `estimatedValue`, `estimatedCloseDate`, `closeProbability`, `forecastCategory`, `saleMotion` are **locked** (returns 400). The Complete-Information fields below remain editable post-close.
> - `forecastCategory` for open-stage deals must be one of: `Open`, `Pipeline`, `Best Case`, `Most Likely`, `Commit`.
> - `parentOpportunityId` cannot reference the deal itself, a cancelled deal, or any descendant (would create a cycle) — `INVALID_PARENT_OPPORTUNITY`.

**Headers**
| Header | Required | Description |
|--------|----------|-------------|
| `X-User-Id` | No | Caller identity recorded in audit log |

**Request Body**
```json
{
  "name": "Deutsche Bank Workstation Refresh Q3",
  "estimatedValue": 250000,
  "estimatedCloseDate": "2026-09-30",
  "closeProbability": 75,
  "forecastCategory": "Commit",
  "saleMotion": "Expansion",

  "summary": "Three-year DaaS rollout for HSBC APAC.",
  "priority": "High",
  "leadOrigin": "Partner",
  "partnerInvolved": true,
  "parentOpportunityId": "118c0cb1-1010-4ab3-9aa1-3b9e54df0001",
  "stageEntryDate": "2026-04-04T09:15:00Z",
  "ownerId": "7D26391E-D020-474E-B1CA-53E6B6C71487"
}
```

| Field | Type | Validation |
|-------|------|-----------|
| `name` | string | Optional. Must be unique (case-insensitive). Cannot be empty. |
| `estimatedValue` | number | Optional. Must be > 0 (0 triggers ERR_MSG_0012). Cannot be negative. |
| `estimatedCloseDate` | date (`YYYY-MM-DD`) | Optional. |
| `closeProbability` | number 0–100 | Optional. |
| `forecastCategory` | string | Optional. Must align with deal stage (see rules above). |
| `saleMotion` | `Net-New` \| `Expansion` \| `Renewal` | Optional. |
| `summary` | string | Optional. Free-text. |
| `priority` | `High` \| `Medium` \| `Low` | Optional. Other values ⇒ `VALIDATION_ERROR`. |
| `leadOrigin` | string | Optional. Free-text in v1. Empty string ⇒ stored as `null`. |
| `partnerInvolved` | boolean | Optional. |
| `parentOpportunityId` | string \| null | Optional. `null` clears the link. Self-reference, cancelled-parent, or a chain that would create a cycle ⇒ `INVALID_PARENT_OPPORTUNITY` (max walk depth = 10). |
| `stageEntryDate` | datetime | Optional. Use to override the auto-populated date when correcting drift. |
| `ownerId` | string | Optional. `systemuser.systemuserid`. Not validated against `systemuser` (some dumps don't ship it); the audit log captures the change. |

**Response 200**
```json
{
  "id": "116191c4-ce2f-46cb-8666-4861a6a1ae26",
  "name": "Deutsche Bank Workstation Refresh Q3",
  "stage": { "raw": "Qualify", "label": "Qualification" },
  "statecode": "Open",
  "estimatedValue": 250000.0,
  "estimatedCloseDate": "2026-09-30",
  "closeProbability": 75.0,
  "forecastCategory": "Commit",
  "saleMotion": { "raw": "Expansion", "label": "Expansion" },
  "ownerId": "user-uuid",
  "isStageLocked": true
}
```

> `isStageLocked: true` means stage advancement UI should be disabled.

**Error responses**
```json
// 422 — ERR_MSG_0012
{ "detail": { "code": "ERR_MSG_0012", "message": "Estimated revenue must be greater than zero." } }

// 422 — Duplicate name
{ "detail": { "code": "DUPLICATE_NAME", "message": "A deal with this name already exists." } }

// 400 — Closed deal locked
{ "detail": { "code": "CLOSED_DEAL_LOCKED", "message": "Fields [...] cannot be modified on a Closed Won / Closed Lost deal." } }

// 422 — Bad forecast category
{ "detail": { "code": "INVALID_FORECAST_CATEGORY", "message": "For stage 'Qualify', forecastCategory must be one of [...]." } }
```

---

## 5. Competitors — Write

### POST `/api/opportunities/{opportunityId}/competitors`
Add a competitor to the deal.

**Request Body**
```json
{
  "competitorName": "Dell Technologies",
  "competitorType": "Incumbent",
  "resellingPartner": "Tech Data"
}
```

| Field | Type | Required | Validation |
|-------|------|----------|-----------|
| `competitorName` | string | **Yes** | Cannot be empty. Must be unique (case-insensitive) on this deal. |
| `competitorType` | `Incumbent` \| `Secondary` | No | — |
| `resellingPartner` | string | No | Free text. |

**Response 201**
```json
{
  "id": "new-comp-uuid",
  "opportunityId": "116191c4-...",
  "name": "Dell Technologies",
  "competitorName": "Dell Technologies",
  "competitorType": "Incumbent",
  "resellingPartnerId": "Tech Data"
}
```

**Error responses**
```json
// 409 — Duplicate
{ "detail": { "code": "DUPLICATE_COMPETITOR", "message": "'Dell Technologies' is already associated with this deal." } }

// 422 — Bad competitor type
{ "detail": { "code": "VALIDATION_ERROR", "message": "competitorType must be 'Incumbent' or 'Secondary'." } }
```

---

### PATCH `/api/opportunities/{opportunityId}/competitors/{competitorId}`
Update an existing competitor on the deal.

**Request Body**
```json
{
  "competitorName": "Dell Technologies",
  "competitorType": "Secondary",
  "resellingPartner": "Ingram Micro"
}
```

| Field | Type | Required | Validation |
|-------|------|----------|-----------|
| `competitorName` | string | **Yes** | Must be unique on this deal (excluding the record being updated). |
| `competitorType` | `Incumbent` \| `Secondary` | No | — |
| `resellingPartner` | string | No | Free text. |

**Response 200** — same shape as POST 201 response above.

---

### DELETE `/api/opportunities/{opportunityId}/competitors/{competitorId}`
Remove a competitor from the deal (soft-delete — record retained for history).

**Response 200**
```json
{
  "opportunityId": "116191c4-...",
  "competitorId": "comp-uuid",
  "message": "Competitor 'Dell Technologies' has been removed from the deal."
}
```

---

## 6. Next Actions

### GET `/api/opportunities/{opportunityId}/next-actions`
List all next actions for the deal (both Open and Completed).

**Response 200**
```json
{
  "opportunityId": "116191c4-...",
  "total": 3,
  "items": [
    {
      "id": "action-uuid",
      "opportunityId": "116191c4-...",
      "description": "Follow up with procurement on budget approval",
      "dueDate": "2026-06-20",
      "status": "Open",
      "createdAt": "2026-06-09T10:00:00Z",
      "updatedAt": "2026-06-09T10:00:00Z",
      "createdBy": "seller@lenovo.com"
    }
  ]
}
```

---

### POST `/api/opportunities/{opportunityId}/next-actions`
Add a new next action (status defaults to `Open`).

**Request Body**
```json
{
  "description": "Follow up with procurement team on budget approval",
  "dueDate": "2026-06-20"
}
```

| Field | Type | Required | Validation |
|-------|------|----------|-----------|
| `description` | string | **Yes** | Cannot be empty. |
| `dueDate` | date (`YYYY-MM-DD`) | No | — |

**Response 201** — same shape as a single item in the GET list above.

---

### PATCH `/api/opportunities/{opportunityId}/next-actions/{actionId}`
Update a next action. All fields optional (true PATCH).

**Request Body**
```json
{
  "description": "Schedule executive briefing call",
  "dueDate": "2026-06-25",
  "status": "Completed"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `description` | string | No | Cannot be set to empty string. |
| `dueDate` | date (`YYYY-MM-DD`) | No | — |
| `status` | `Open` \| `Completed` | No | Set to `Completed` to mark done. Action is retained (not deleted). |

**Response 200** — same shape as a single item in the GET list above.

---

## 7. Audit Log
Every POST, PATCH, DELETE automatically writes a record to `lvo_audit_log`.  
Frontend does **not** need to call anything — this is transparent.

Each log entry contains:
- `entityType`: `opportunity` | `competitor` | `next_action`
- `action`: `create` | `update` | `delete`
- `changedBy`: value from `X-User-Id` header
- `diff`: JSON with `before` / `after` snapshots for updates

---

## 8. Shared Types Reference

### `StageRef`
```json
{ "raw": "Qualify", "label": "Qualification" }
```
Stage label mapping:
| raw | label |
|-----|-------|
| `Qualify` | `Qualification` |
| `Develop` | `Discovery` |
| `Propose` | `Proposal` |
| `Execute` | `Negotiation` |
| `Closed Won` | `Closed Won` |
| `Closed Lost` | `Closed Lost` |

### `SaleMotionRef`
```json
{ "raw": "Net-New", "label": "Net new" }
```
| raw | label |
|-----|-------|
| `Net-New` | `Net new` |
| `Expansion` | `Expansion` |
| `Renewal` | `Renewal` |

### `ActivityItem`
```json
{
  "id": "uuid",
  "type": "email",
  "direction": "outbound",
  "subject": "Proposal follow-up",
  "body": null,
  "activityDate": "2026-06-01T14:30:00Z",
  "groupedCount": null
}
```
`type` values: `email` | `meeting` | `crm` | `multiple`  
`direction` values: `inbound` | `outbound`  
`groupedCount` is only set when `type = "multiple"`.

### `KpiCard`
```json
{
  "value": 3200000.0,
  "count": 5,
  "trend": null
}
```
`trend` is `null` until at least one comparison snapshot is older than
the requested `comparePeriod` lookback (and only when no filters are
applied — see [/kpi-summary](#get-apiopportunitieskpi-summary) for the
full semantics). When populated:
```json
{
  "trend": {
    "direction": "up",
    "deltaValue": 450000.0,
    "deltaCount": 2
  }
}
```
* `direction` values: `up` | `down` | `flat`
* `direction` is the sign of `deltaValue`; ties (`deltaValue == 0`) fall
  through to `deltaCount`; `flat` only when both deltas are zero.

---

## 9. Deal Detailed View

The Deal Detailed View user story is served by a small read-router
(`deals_read.py`) plus a contacts CRUD router (`contacts.py`) and the
extension to `deals_write.py` that adds soft-delete.  All endpoints
return camelCase JSON and accept the same `X-User-Id` audit header on
the write paths.

> **Soft delete:** `DELETE /api/opportunities/{id}` sets
> `statecode='Canceled'` instead of removing the row.  Canceled deals are
> excluded from KPIs, the grid, and filter dropdowns automatically.

---

### GET `/api/opportunities/{opportunityId}`
Full payload that hydrates the entire Deal Detail page in one round-trip.

**Response 200**
```json
{
  "id": "116191c4-ce2f-46cb-8666-4861a6a1ae26",
  "name": "HSBC – DaaS Rollout APAC",
  "accountId": "acc-uuid",
  "stage":       { "raw": "Propose", "label": "Proposal" },
  "saleMotion":  { "raw": "Renewal", "label": "Renewal" },
  "forecastCategory": "Commit",
  "value": 3200000.0,
  "currency": "USD",
  "closeDate": "2026-08-02",
  "closeProbability": 60.0,
  "ownerId":   "user-uuid",
  "ownerName": "Amit Ranjan",
  "statecode": "Open",
  "tempoClass": "Quarterly",
  "createdAt":       "2026-01-15T09:00:00Z",
  "stageEntryDate":  "2026-04-04T09:15:00Z",
  "isClosed":        false,
  "isCanceled":      false,
  "isStageLocked":   false,

  "summary": "HSBC's APAC desktop fleet refresh — three-year DaaS rollout.",
  "priority": "High",
  "leadOrigin": "Partner",
  "partnerInvolved": true,
  "parentOpportunityId":   "118c0cb1-1010-4ab3-9aa1-3b9e54df0001",
  "parentOpportunityName": "HSBC – Master Refresh Program",
  "childOpportunities": [
    { "id": "118c0cb1-1010-4ab3-9aa1-3b9e54df0042", "name": "HSBC – Phase 2 (Hong Kong)" }
  ],
  "daysInStage": 32,
  "createdBy":  "Sarah Lee",
  "modifiedAt": "2026-06-09T08:00:00Z",
  "modifiedBy": "Mark Thompson",
  "account": {
    "id": "acc-uuid",
    "name": "HSBC",
    "segment": "Strategic",
    "industry": "Banking",
    "territory": "Asia Pacific",
    "employeeCount": 230000,
    "totalAccountValue": 8450000.0,
    "openDealsCount": 4,
    "businessGroup": "APAC BG",
    "country": "HK"
  },
  "decisionMaker": {
    "id": "OC-DM-...",
    "contactId": "contact-uuid",
    "name": "Jane Lee",
    "role": "Decision Maker",
    "isDecisionMaker": true,
    "lastTouchDate": "2026-06-01T14:30:00Z",
    "jobTitle": "VP IT",
    "email": "jane.lee@example.com"
  },
  "additionalContacts": [
    {
      "id": "OC-...",
      "contactId": "contact-uuid-2",
      "name": "Sam Patel",
      "role": "Influencer",
      "isDecisionMaker": false,
      "lastTouchDate": null,
      "jobTitle": "Procurement Lead",
      "email": "sam.patel@example.com"
    }
  ],
  "competitors":     [],
  "nextActions":     [],
  "activitiesPreview": [],
  "health": {
    "score": 72,
    "band": "YELLOW",
    "updatedAt": "2026-06-09T08:00:00Z",
    "components": {
      "stage_progress":     { "weight": 25, "score": 80, "inputs": { "stage": "Propose" } },
      "activity_freshness": { "weight": 25, "score": 70, "inputs": { "ratio": 1.4 } },
      "stakeholder":        { "weight": 20, "score": 64, "inputs": { "activeStakeholders": 4 } },
      "close_confidence":   { "weight": 20, "score": 100, "inputs": { "gap": 0.05 } },
      "risk_adjustment":    { "weight": 10, "score": 60, "inputs": { "riskCount": 2 } }
    }
  },
  "risks": [
    {
      "id": "risk-uuid",
      "category": "Activity & Engagement",
      "name": "Low Activity",
      "message": "Low activity on deal (18 days since last touch)",
      "detectedAt": "2026-06-09T08:00:00Z"
    }
  ]
}
```

`band` values: `GREEN` (≥75) | `YELLOW` (50–74) | `RED` (<50).

#### Complete-Information (Deal Summary) fields — v0.13.0+

The fields below back the **Complete Information** tab on the Opportunity
Detail page. They are READ via this endpoint and WRITTEN via
`PATCH /api/opportunities/{opportunityId}` — same field names on the
request body (camelCase).

| Field | Type | R/W | Notes |
|---|---|---|---|
| `summary` | string \| null | R/W | Free-text Summary input on the form. |
| `priority` | `"High"` \| `"Medium"` \| `"Low"` \| null | R/W | Constrained server-side; PATCH rejects other values with `VALIDATION_ERROR`. |
| `leadOrigin` | string \| null | R/W | Free-text in v1 (FE governs the dropdown options). Empty string ⇒ null. |
| `partnerInvolved` | boolean | R/W | Default `false`. |
| `parentOpportunityId` | string \| null | R/W | Self-FK. Setting to `null` clears the link. Cycle / self-reference / cancelled-parent attempts ⇒ `INVALID_PARENT_OPPORTUNITY`. |
| `parentOpportunityName` | string \| null | **R only** | Resolved label for `parentOpportunityId`. `null` when no parent or when the parent has been cancelled. |
| `childOpportunities` | `OpportunityRef[]` | **R only** | Every non-cancelled deal whose `parentOpportunityId` points to this one. Sorted alphabetically by name. |
| `daysInStage` | integer \| null | **R only** | Whole days between `stageEntryDate` and today (UTC). Negative values clamped to 0. `null` when `stageEntryDate` is `null`. |
| `ownerName` | string \| null | **R only** | Resolved from `ownerId` via `systemuser.fullname` (or `internalemailaddress` fallback). `null` only when the lookup is genuinely impossible. |
| `createdBy` | string \| null | **R only** | `opportunity.createdby` — `null` if the dump doesn't ship the column. |
| `modifiedAt` | datetime \| null | **R only** | `opportunity.modifiedon`. |
| `modifiedBy` | string \| null | **R only** | `opportunity.modifiedby`. |

`OpportunityRef` shape (used by `childOpportunities` and the search endpoint):
```json
{ "id": "uuid", "name": "Deal Name" }
```

> **Migration:** these fields require `sql/2026_06_opportunity_complete_info_schema.sql`.
> Without it, the writable fields default to `null` and the audit-column
> reads (`createdBy` / `modifiedAt` / `modifiedBy`) silently fall back to
> `null` rather than 500'ing the page.

---

### GET `/api/opportunities/search`
Typeahead behind the **Parent Opportunity** picker on the Complete-
Information form.

**Query parameters**

| Param | Default | Notes |
|---|---|---|
| `q` | *required* | 1–200 chars. ILIKE'd against `opportunity.name` (case-insensitive substring). |
| `excludeId` | — | Optional `opportunityid` to exclude — typically the deal currently being edited so the picker can't pick itself. |
| `limit` | `20` | Range `1..50`. |

**Response 200**
```json
{
  "query": "ThinkPad",
  "total": 3,
  "items": [
    { "id": "118c0cb1-...", "name": "ThinkPad Fleet Refresh" },
    { "id": "27b29d31-...", "name": "ThinkPad Renewal — APAC" },
    { "id": "9d7a2c45-...", "name": "ThinkPad Pro Pilot" }
  ]
}
```

**Ordering** — prefix matches first, then substring matches, then
alphabetical tiebreaker. Cancelled deals are filtered out.

> **Cycle prevention:** the picker may surface a descendant of the deal
> currently being edited; selecting it will fail at PATCH time with
> `INVALID_PARENT_OPPORTUNITY`. The FE should surface that as a toast
> rather than pre-filtering.

---

### DELETE `/api/opportunities/{opportunityId}`
Soft-delete the deal — sets `statecode='Canceled'`.

**Response 200**
```json
{
  "id": "116191c4-...",
  "statecode": "Canceled",
  "message": "Deal 'HSBC – DaaS Rollout APAC' has been canceled."
}
```

**Error responses**
```json
// 409 — Deal already canceled
{ "detail": { "code": "ALREADY_CANCELED", "message": "Deal has already been canceled." } }
```

---

### GET `/api/opportunities/{opportunityId}/timeline`
Paginated chronological feed combining activities + audit-log CRM changes.

**Query Parameters**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | integer ≥ 1 | `1` | Page number |
| `pageSize` | integer 1–100 | `25` | Items per page |
| `types` | csv | — | `email,meeting,crm,multiple,crm_change` — omit to include all |

**Response 200**
```json
{
  "opportunityId": "116191c4-...",
  "page": 1,
  "pageSize": 25,
  "total": 42,
  "totalPages": 2,
  "items": [
    {
      "id": "activity-uuid",
      "source": "activity",
      "type": "email",
      "direction": "outbound",
      "subject": "Proposal follow-up",
      "body": null,
      "eventDate": "2026-06-01T14:30:00Z",
      "groupedCount": null,
      "changedBy": null
    },
    {
      "id": "audit-uuid:estimatedvalue",
      "source": "crm_change",
      "type": "estimatedvalue",
      "direction": null,
      "subject": "estimatedvalue changed",
      "body": "'3000000' → '3200000'",
      "eventDate": "2026-05-28T10:11:00Z",
      "groupedCount": null,
      "changedBy": "seller@lenovo.com"
    }
  ]
}
```

`source` values: `activity` (from `lvo_activity`) | `crm_change` (derived from `lvo_audit_log`).

---

### GET `/api/opportunities/{opportunityId}/contacts`
Decision maker + additional contacts. Powers both the **Contacts tab card grid** and the **Contact detail / edit form** on the Opportunity Detail page.

**Response 200**
```json
{
  "opportunityId": "116191c4-...",
  "decisionMaker": {
    "id": "OC-DM-...",
    "contactId": "5d2a1f2e-...",
    "name": "Rajesh Kumar",
    "firstName": "Rajesh",
    "lastName": "Kumar",
    "role": "Decision Maker",
    "isDecisionMaker": true,
    "lastTouchDate": "2026-06-12T10:00:00Z",
    "jobTitle": "VP of IT Infrastructure",
    "email": "RajeshKumar@infosys.com",
    "phone": "+91 98765 00000"
  },
  "additionalContacts": [
    { "id": "OC-...", "contactId": "...", "name": "Sumit Reddy", "firstName": "Sumit", "lastName": "Reddy", "role": null, "isDecisionMaker": false, "jobTitle": "CTO", "email": "...", "phone": "..." }
  ],
  "total": 2
}
```

`ContactRef` field reference:

| Field | Source | Notes |
|---|---|---|
| `id` | `lvo_opportunitycontact.lvo_opportunitycontactid` | The link row PK — use this for PATCH / DELETE. |
| `contactId` | `lvo_opportunitycontact.lvo_contactid` | The underlying contact row. |
| `name` | `contact.fullname` (fallback: firstname + lastname) | Display name for the card header. |
| `firstName` | `contact.firstname` | Read-only on the FE (the underlying contact owns it). |
| `lastName` | `contact.lastname` | Read-only on the FE. |
| `role` | `lvo_opportunitycontact.lvo_role` | Link-level: `Decision Maker / Champion / Influencer / Procurement / Technical`. **Distinct from `jobTitle`.** |
| `isDecisionMaker` | `lvo_opportunitycontact.lvo_isdecisionmaker` | Drives the "Decision Maker" pill on the card and the toggle on the edit form. |
| `lastTouchDate` | `lvo_opportunitycontact.lvo_lasttouchdate` | Most recent activity touching this contact. Refreshed by the recalc service. |
| `jobTitle` | `contact.jobtitle` | Cards display this as the secondary line ("VP of IT Infrastructure", "CTO"). The detail form shows it in the **Role** input as a read-only attribute. |
| `email` | `contact.emailaddress1` | Read-only. |
| `phone` | resolved at runtime against `telephone1` → `mobilephone` → `lvo_phone` (whichever exists) | Editable via PATCH (see below). `null` if the dump has no phone column or the value is empty. |

> **Note on `phone`:** the value lives on the underlying `contact` row, not on the link. Editing it via the PATCH endpoint mutates the contact and is therefore visible on every other deal that contact is attached to and on the Account Contacts tab. The FE may want to surface a "this affects other deals" toast.

---

### POST `/api/opportunities/{opportunityId}/contacts`
Attach an existing contact to the deal. Powers the "New Contact" picker on the Contacts tab — typical flow:
1. FE fetches `GET /api/accounts/{accountId}/contacts` (the deal's account)
2. Filters out contacts already attached to this deal
3. User picks one
4. FE POSTs here with that `contactId`

**Request Body**
```json
{
  "contactId": "contact-uuid",
  "role": "Champion",
  "isDecisionMaker": false
}
```
Setting `isDecisionMaker=true` automatically demotes any existing DM on the deal.

**Response 201** — `ContactRef` (same shape as one item in the contacts list, including `firstName`, `lastName`, `phone`).
**Errors:** `404` (opportunity / contact not found), `409` (`DUPLICATE_CONTACT`).

---

### PATCH `/api/opportunities/{opportunityId}/contacts/{contactLinkId}`
Edit a contact link. All fields optional (true PATCH semantics — only the keys present in the request body are touched).

```json
{
  "role": "Decision Maker",
  "isDecisionMaker": true,
  "phone": "+91 98765 00000"
}
```

| Field | Type | Writes to |
|---|---|---|
| `role` | string \| null | `lvo_opportunitycontact.lvo_role` (link-level) |
| `isDecisionMaker` | bool \| null | `lvo_opportunitycontact.lvo_isdecisionmaker`. Setting `true` auto-demotes any other DM on the deal. |
| `phone` | string \| null | The underlying `contact` row (`telephone1` / `mobilephone` / `lvo_phone` — whichever exists). Empty string is treated as "clear". **Affects every other deal this contact is on.** |

**Response 200** — refreshed `ContactRef` with the post-update values (re-reads phone after the write).

---

### DELETE `/api/opportunities/{opportunityId}/contacts/{contactLinkId}`
Soft-delete the link (the underlying contact row is retained).

**Response 200**
```json
{
  "opportunityId": "116191c4-...",
  "contactLinkId": "OC-...",
  "message": "Contact has been removed from the deal."
}
```

---

### GET `/api/opportunities/{opportunityId}/health`
Live health breakdown — same shape as the `health` block embedded in the
detail payload. **Does not write** to the DB.

---

### POST `/api/opportunities/{opportunityId}/health/recalculate`
Force a synchronous recalculation. Persists the result. Audit-logged.

**Response 200**
```json
{
  "opportunityId": "116191c4-...",
  "health":  { "score": 72, "band": "YELLOW", "components": { ... } },
  "risks":   [ { "category": "Stakeholder", "name": "...", "message": "..." } ],
  "message": "Deal health recalculated."
}
```

---

### GET `/api/opportunities/{opportunityId}/risks`
Active risks for the deal. Returns persisted rows when available, falls
back to a live evaluation otherwise.

**Response 200**
```json
{
  "opportunityId": "116191c4-...",
  "total": 2,
  "items": [
    {
      "id": "risk-uuid",
      "category": "Activity & Engagement",
      "name": "Low Activity",
      "message": "Low activity on deal",
      "detectedAt": "2026-06-09T08:00:00Z"
    }
  ]
}
```

`category` values: `Activity & Engagement` | `Stakeholder` | `Deal Execution` | `Timeline & Forecast`.

---

> The legacy minimal `GET /api/accounts/{id}` was replaced by the View
> Account user story — see [§ 10 View Account](#10-view-account) below
> for the current `AccountDetail` shape (the same path now returns the
> full payload with rollups, status, owner, and pipeline counters).

---

## 10. View Account

User story: sellers want a paginated, filterable, exportable list of accounts
plus a detail page that surfaces linked deals and stakeholder contacts.

All endpoints below sit behind the `/api/accounts` prefix.

### Vocabulary
| Concept | Values |
|---------|--------|
| `accountType` | `Prospect`, `Customer` (becomes `Customer` after first Closed-Won deal) |
| `accountStatus` | `Active`, `Inactive`, `At-Risk` (At-Risk = ≥1 open deal with `dealHealth < 50`) |
| `segment` | `SMB`, `Mid-Market`, `Enterprise`, `Strategic` |
| Sortable columns | `name`, `totalAccountValue`, `openDealsCount`, `lastInteraction`, `status`, `lvoAccountType` |

### GET `/api/accounts`
Paginated grid. All filters are optional; multi-select fields take a
comma-separated value.

**Query parameters**
| Name | Type | Notes |
|------|------|-------|
| `page` | int ≥ 1 | Default 1 |
| `pageSize` | int 1-100 | Default 25 |
| `search` | string | ILIKE on `account.name` and `account.accountnumber` |
| `accountTypes` | csv | `Prospect,Customer` |
| `accountStatuses` | csv | Defaults to `Active,At-Risk` (Inactive hidden unless requested) |
| `segments` | csv | `SMB,Mid-Market,Enterprise,Strategic` |
| `regions` | csv | Matches `lvo_territory`, `lvo_businessgroupid` or `lvo_countryid` (case-insensitive) |
| `industries` | csv | Matches `account.industrycode` |
| `valueMin` / `valueMax` | float ≥ 0 | Range applied to `totalAccountValue` |
| `bucket` | enum | KPI-strip drill-down. One of `total` / `acv` / `active` / `at_risk`. Predicate matches `/api/accounts/kpi-summary` so card counts and grid counts agree. Composes with every other filter. `bucket=total` additionally bypasses the default `accountStatuses` filter so Inactive rows are included. |
| `sortBy` | enum | One of the sortable columns above (default `name`) |
| `sortOrder` | `asc` \| `desc` | Default `asc` |

**Bucket → predicate mapping**
| `bucket` value | Predicate added to the WHERE clause |
|---|---|
| `total` | _no extra predicate_ — but the default `accountStatuses` filter is bypassed so Inactive accounts also appear |
| `acv` | `totalAccountValue > 0` (account has at least one non-canceled opportunity) |
| `active` | `account.lvo_accountstatus = 'Active'` |
| `at_risk` | `account.lvo_accountstatus = 'At-Risk'` |

**Drill-down examples** (click a card on `/api/accounts/kpi-summary`)
```
GET /api/accounts?bucket=total&page=1&pageSize=10&sortBy=name&sortOrder=asc
GET /api/accounts?bucket=acv&page=1&pageSize=10&sortBy=totalAccountValue&sortOrder=desc
GET /api/accounts?bucket=active&page=1&pageSize=10&sortBy=lastInteraction&sortOrder=desc
GET /api/accounts?bucket=at_risk&page=1&pageSize=10&sortBy=totalAccountValue&sortOrder=desc
```

`bucket` composes with the other filters — e.g. `?bucket=at_risk&regions=APAC&segments=Enterprise` returns the intersection.

**Response 200**
```json
{
  "page": 1,
  "pageSize": 25,
  "total": 65,
  "totalPages": 3,
  "sortBy": "name",
  "sortOrder": "asc",
  "items": [
    {
      "id": "A0000001-AAAA-0001-0001-000000000001",
      "name": "JPMorgan Chase & Co.",
      "accountNumber": "ACC-2025-006",
      "accountType": "Customer",
      "industry": "Financial Services",
      "segment": "Strategic",
      "region": "North America",
      "businessGroup": "Americas BG",
      "country": "US",
      "territory": "North America",
      "status": "Active",
      "statecode": "Active",
      "lastInteraction": "2026-06-04T15:32:00+00:00",
      "activeOpportunitiesCount": 3,
      "totalAccountValue": 12450000,
      "currency": "USD",
      "employeeCount": 300000,
      "revenue": 158000000000,
      "ownerId": "81AADFDB-1817-425C-A5B1-45F383F230CE",
      "ownerName": "Sarah Lee"
    }
  ],
  "notes": []
}
```

When the result-set is empty, `notes` contains `["ERR_MSG_0010"]` so the
frontend can render the canonical no-results banner.

### GET `/api/accounts/filters`
Distinct values for the multi-select pickers.

**Response 200**
```json
{
  "accountTypes": ["Customer", "Prospect"],
  "accountStatuses": ["Active", "At-Risk", "Inactive"],
  "segments": ["Enterprise", "Mid-Market", "SMB", "Strategic"],
  "regions": ["APAC BG", "Americas BG", "Asia Pacific", "EMEA BG", "Western Europe", "..."],
  "industries": ["Aerospace", "Automotive", "Energy", "Financial Services", "..."]
}
```

### GET `/api/accounts/value-range`
Min/max `totalAccountValue` across the entire dataset. Drives the dynamic
range slider.

**Response 200**
```json
{ "min": 0, "max": 18250000, "currency": "USD" }
```

### GET `/api/accounts/kpi-summary`
Top-of-page KPI strip on the Accounts grid. Returns four cards plus
optional period-over-period trend deltas. Mirrors
`/api/opportunities/kpi-summary` — same `comparePeriod` toggle, same
trend payload shape (`TrendInfo`).

**Query parameters** (all optional)

| Param | Default | Description |
|---|---|---|
| `comparePeriod` | `last_week` | One of `last_week` / `past_month` / `last_quarter`. |
| `search` | — | ILIKE on `account.name` + `account.accountnumber`. |
| `accountTypes` | — | Comma-separated `Prospect` / `Customer`. |
| `accountStatuses` | — | Comma-separated `Active` / `Inactive` / `At-Risk`. |
| `segments` | — | Comma-separated `SMB` / `Mid-Market` / `Enterprise` / `Strategic`. |
| `regions` | — | Comma-separated. Matches `lvo_territory`, `lvo_businessgroupid` or `lvo_countryid` (case-insensitive). |
| `industries` | — | Comma-separated industry codes. |
| `valueMin`, `valueMax` | — | Bounds on `totalAccountValue`. |

**Bucket math**

| Card | Bucket id | Predicate | `value` | `count` |
|---|---|---|---|---|
| Total Accounts | `total` | _no predicate_ — every row in `account` | `0` | `count(*)` |
| Account Value (ACV) | `acv` | `opportunity.statecode <> 'Canceled'` | `sum(opportunity.estimatedvalue)` | `count(distinct opportunity.accountid)` |
| Active Accounts | `active` | `account.lvo_accountstatus = 'Active'` | `0` | `count(*)` |
| Accounts at Risk | `at_risk` | `account.lvo_accountstatus = 'At-Risk'` | `0` | `count(*)` |

The page-total `accountValue.value` always equals the sum of the
`totalAccountValue` column on `/api/accounts` (same source SQL).

**Trend semantics (Strategy A on filters)**

`trend` is emitted **only on the unfiltered view**. The moment any
filter is supplied, every card returns `trend: null` and `notes`
explains why — snapshots are global / unfiltered in v1, so a filtered
comparison would lie.

`trend` is also `null` until at least one historical row exists for
the lookback date in `lvo_accountsnapshot`. Run
`python -m app.jobs.snapshot_account_kpis --backfill` once to populate
today, today-7, today-30 and today-90, then schedule
`python -m app.jobs.snapshot_account_kpis` nightly.

**Direction logic** (mirrors the opportunity KPI):
1. `delta_value > 0` → `up`; `< 0` → `down`.
2. Tie on value → `delta_count` decides.
3. Both zero → `flat`.

**Response 200** (unfiltered, snapshots present)
```json
{
  "comparePeriod": "last_week",
  "asOf": "2026-06-11T15:30:00.000Z",
  "currency": "USD",
  "totalAccounts":  { "value": 0,        "count": 52, "trend": { "direction": "down", "deltaValue": 0,        "deltaCount": -2 } },
  "accountValue":   { "value": 19750000, "count": 38, "trend": { "direction": "up",   "deltaValue": 175000.0, "deltaCount": 1 } },
  "activeAccounts": { "value": 0,        "count": 24, "trend": { "direction": "up",   "deltaValue": 0,        "deltaCount": 1 } },
  "accountsAtRisk": { "value": 0,        "count": 7,  "trend": { "direction": "down", "deltaValue": 0,        "deltaCount": -1 } },
  "notes": []
}
```

**Response 200** (filtered — trend suppressed)
```json
{
  "comparePeriod": "last_week",
  "asOf": "2026-06-11T15:30:00.000Z",
  "currency": "USD",
  "totalAccounts":  { "value": 0,       "count": 12, "trend": null },
  "accountValue":   { "value": 4180000, "count": 9,  "trend": null },
  "activeAccounts": { "value": 0,       "count": 11, "trend": null },
  "accountsAtRisk": { "value": 0,       "count": 1,  "trend": null },
  "notes": [
    "Trend is suppressed because filters are applied — v1 snapshots are global only. Re-issue without filters to get period-over-period change."
  ]
}
```

**Backfill caveats** (encoded in the snapshot job docstring too):
- `total` is exact when `account.createdon` is present on the live schema; partial dumps that lack the column fall back to "today" values.
- `acv` is exact whenever `opportunity.lvo_createdat` is populated (set by `sql/2026_06_deal_detail_schema.sql`).
- `active` / `at_risk` are **approximate** during backfill — `lvo_accountstatus` flips aren't journaled, so backfilled historical rows reflect today's status. Trend deltas across periods covered only by backfill will read 0 until nightly snapshots accumulate.

### GET `/api/accounts/export`
Streamed CSV with the same query-string filters as `/api/accounts`,
including the `bucket=total|acv|active|at_risk` drill-down — so clicking
**Export** after clicking a KPI card produces a CSV of exactly those
rows. The response body is plain CSV; `Content-Disposition` is set to
`attachment` so the browser downloads it.

### GET `/api/accounts/{accountId}`
Full account-detail payload powering the Account-detail page header.
Reuses every field on `AccountListItem` and adds pipeline counters
(`wonDealsCount`, `lostDealsCount`, `canceledDealsCount`,
`totalDealsCount`).

The endpoint is read-only but **always re-derives** `accountType`,
`status` and `lastInteraction` on the fly (without persisting), so the
page never shows stale values even if the nightly recalc batch hasn't
run. To force a write, use `POST /api/accounts/{id}/recompute-status`.

**Response 200**
```json
{
  "id": "18c38124-5400-478e-9dc1-9e8c96adb8cc",
  "name": "Infosys",
  "accountNumber": "ID238970",
  "accountType": "Customer",
  "industry": "Technology",
  "segment": "Enterprise",
  "region": "Asia Pacific",
  "businessGroup": "APAC BG",
  "country": "IN",
  "territory": "Asia Pacific",
  "status": "Active",
  "statecode": "Active",
  "lastInteraction": "2026-05-21T09:15:00+00:00",
  "activeOpportunitiesCount": 4,
  "totalAccountValue": 8450000.0,
  "currency": "USD",
  "employeeCount": 230000,
  "revenue": 17000000000.0,
  "ownerId": "system-user-uuid",
  "ownerName": "alice@lenovo.com",
  "wonDealsCount": 3,
  "lostDealsCount": 1,
  "canceledDealsCount": 0,
  "totalDealsCount": 8
}
```

| Field | Source / derivation |
|---|---|
| `accountType` | Cached on `account.lvo_accounttype`; falls back to live derivation when the column is missing. `Customer` once any deal hits Closed Won, else `Prospect`. |
| `status` | Cached on `account.lvo_accountstatus`; live re-derived per request. `Active` / `Inactive` / `At-Risk` (latter when ≥1 open deal has `dealHealth < 50`). |
| `lastInteraction` | `MAX(lvo_activity.lvo_activitydate)` over deals belonging to the account. |
| `totalAccountValue` | `SUM(estimatedvalue)` across non-Canceled deals. |
| `activeOpportunitiesCount` | `COUNT(*)` of deals with `statecode='Open'`. |
| `won/lost/canceled/totalDealsCount` | Single 4-bucket aggregate using `FILTER` clauses. |

Returns `404` when no account with that id exists.

### GET `/api/accounts/{accountId}/opportunities`
Powers the **Opportunities** tab on the Account Detail screen. Same row
shape (`OpportunityListItem`) the main grid uses, but pre-filtered to
`account_id = :id`, with a populated activity timeline (last 90d by
default, same-day events bucketed into a `multiple` marker), a
`riskCount` value driving the `⚠ N` badge, and competitor preview.

**Query parameters**
| Name | Type | Notes |
|------|------|-------|
| `page` | int ≥ 1 | Default 1 |
| `pageSize` | int 1-100 | Default 25 |
| `includeCanceled` | bool | Default `false` |
| `search` | string | ILIKE on `opportunity.name` OR `account.name`. Backs the toolbar **Search Opportunity** box. |
| `regions` | CSV | Matches `opportunity.lvo_businessgroup` OR `lvo_country` (case-insensitive). Vocabulary: `/api/filters/regions`. |
| `industries` | CSV | Matches `account.industrycode`. Vocabulary: `/api/filters/industries`. |
| `stages` | CSV | Matches `opportunity.stagename`. Vocabulary: `/api/filters/stages`. |
| `products` | CSV | EXISTS over `quote` + `lvo_quoteitem.lvo_productseries`. Accepts product slugs (`thinkpad`) or raw labels (`ThinkPad`). Vocabulary: `/api/filters/products`. |
| `sortBy` | enum | `name` \| `value` \| `closeDate` \| `closeProbability` \| `stage`. Default `closeDate`. |
| `sortOrder` | enum | `asc` \| `desc`. Default `desc`. |
| `timelineDays` | int 7-365 | Activity-timeline window length. Default `90` (matches the UI tick marks at 90/60/30/0). |

**Activity-timeline bucketing**

For each row, the response's `activities` array is built from
`lvo_activity` rows in the last `timelineDays`, ordered most-recent-first:

* **1 event on a calendar day** → emitted as-is with its own `type`
  (`email` / `meeting` / `crm`).
* **2+ events on the same calendar day** → collapsed into a single
  `type='multiple'` marker with `groupedCount=N`. The FE renders this as
  the numbered circle ("5") shown in the legend (`Multiple Events`).

Capped at 30 markers per opportunity (the strip can't render more dots
than that).

**Risk-count badge**

`riskCount` is the count of Active rows in `lvo_dealrisk` for the
opportunity. Powers the `⚠ N` badge in the Risk column. `null` when the
`lvo_dealrisk` table isn't on the deployed schema.

**Response 200**
```json
{
  "accountId": "18c38124-5400-478e-9dc1-9e8c96adb8cc",
  "page": 1,
  "pageSize": 25,
  "total": 50,
  "totalPages": 2,
  "items": [
    {
      "id": "f86ec95a-f627-4c12-a3a5-4adbc43c6dba",
      "name": "ThinkStation Workstations",
      "accountId": "18c38124-5400-478e-9dc1-9e8c96adb8cc",
      "accountName": "Infosys",
      "stage": { "raw": "Proposal", "label": "Proposal" },
      "saleMotion": { "raw": "Renewal", "label": "Renewal" },
      "value": 850000.0,
      "currency": "USD",
      "closeDate": "2026-09-30",
      "dealHealth": 25,
      "risk": "Budget Freeze",
      "riskScore": 3,
      "riskCount": 3,
      "competitorCount": 2,
      "competitors": ["HP", "Dell"],
      "lastActivity": "2026-06-08T09:45:00",
      "activities": [
        { "id": "...", "type": "multiple", "groupedCount": 5, "activityDate": "2026-06-08T..." },
        { "id": "...", "type": "email",    "activityDate": "2026-06-05T..." },
        { "id": "...", "type": "meeting",  "activityDate": "2026-05-30T..." }
      ]
    }
  ]
}
```

### GET `/api/accounts/{accountId}/opportunities/export`
CSV stream of the same filtered set the list endpoint returns. Honours
every filter the list endpoint accepts (forward the active query string
to keep the download in sync with the grid).

Streamed in chunks; the unbounded server-side loop keeps memory bounded
and the response starts arriving while the rest is still being assembled.

**Columns**
`Name`, `Account`, `Stage`, `Sale Motion`, `Forecast`, `Value`,
`Currency`, `Close Date`, `Probability`, `Deal Health`, `Risk Reason`,
`Risk Count`, `Competitors`, `Last Activity`, `Statecode`.

**Response 200**
`Content-Type: text/csv` with `Content-Disposition: attachment;
filename="account_<id>_opportunities_<timestamp>.csv"`.

### GET `/api/accounts/{accountId}/customer-information`
Sectioned read-only payload for the **Customer Information** tab
(View Customer Information user story, Phase 1).

Returns six cards in a single response so the FE can render the entire
tab without orchestrating sub-resource calls. Every section is always
present in the response — fields whose source column is missing on the
deployed DB come back as `null` rather than throwing.

**Source columns**

| Section | Column source |
|---------|---------------|
| `basicInformation` | `account.accountid`, `name`, `lvo_accounttype`, `lvo_segment`, `lvo_subsegment` *(new)*, `industrycode` (Industry Segment), `lvo_gtmsegment` *(new)*, `revenue`, `numberofemployees`, `lvo_sellerknownas` *(new)* |
| `billingAddress` | `account.address1_line1` / `address1_line2` / `address1_city` / `address1_stateorprovince` / `address1_postalcode` / `address1_country` |
| `shippingAddress` | `account.address2_line1` / `address2_line2` / `address2_city` / `address2_stateorprovince` / `address2_postalcode` / `address2_country` |
| `identityAndLegal` | `account.lvo_legalnamelocal` *(new)*, `lvo_locallanguage` *(new)*, `lvo_alias` *(new)*, `lvo_taxvatnumber` *(new)*, `lvo_legalentity` *(new)*, `telephone1`, `websiteurl`, `lvo_linkedinurl` *(new)* |
| `commercialTerms` | `account.lvo_defaultcurrency`, `paymenttermscode` (label-resolved), `defaultpricelevelid` (name-resolved when `pricelevel.name` is queryable), `lvo_dealsignconfig` *(new)* |
| `territoryAndOwnership` | Region (precedence `lvo_territory` > `lvo_businessgroupid` > `lvo_countryid`), `lvo_salesterritory` *(new)* with `territoryid` lookup fallback, `lvo_futureterritory` *(new)*, `lvo_salesorg` *(new)*, `lvo_territorymovereason` *(new)*, `lvo_geographicunit` *(new)*, `lvo_salesoffice` *(new)*, `owninguser` (Assigned Owner — name resolved from `systemuser.fullname`), `createdby` (Record Owner — same lookup) |

`(new)` columns are added by `sql/2026_06_account_customer_info_schema.sql`. The
endpoint defers any column missing from the live `account` table so a
partially-migrated DB still serves the page.

**Response 200**
```json
{
  "id": "18c38124-5400-478e-9dc1-9e8c96adb8cc",
  "basicInformation": {
    "accountId": "18c38124-5400-478e-9dc1-9e8c96adb8cc",
    "accountName": "Infosys",
    "accountType": "Customer",
    "segment": "Enterprise",
    "subSegment": "Tech Services",
    "industrySegment": "Technology",
    "gtmSegment": "Direct",
    "annualRevenue": 17000000000.0,
    "employeeCount": 230000,
    "sellerKnownAs": "Infosys Ltd."
  },
  "billingAddress": {
    "line1": "Electronics City",
    "line2": "Phase 1",
    "city": "Bengaluru",
    "stateProvince": "Karnataka",
    "postalCode": "560100",
    "country": "IN"
  },
  "shippingAddress": {
    "line1": null, "line2": null, "city": null,
    "stateProvince": null, "postalCode": null, "country": null
  },
  "identityAndLegal": {
    "legalNameLocal": "इन्फोसिस लिमिटेड",
    "localLanguage": "hi-IN",
    "alias": "INFY",
    "taxVatNumber": "AABCI4798L",
    "legalEntity": "Infosys Limited",
    "mainPhone": "+91-80-2852-0261",
    "website": "https://www.infosys.com",
    "linkedinUrl": "https://www.linkedin.com/company/infosys"
  },
  "commercialTerms": {
    "defaultCurrency": "USD",
    "paymentTerms": "Net 30",
    "priceList": "APAC Standard",
    "dealSignConfig": "DocuSign-APAC"
  },
  "territoryAndOwnership": {
    "region": "Asia Pacific",
    "salesTerritory": "APAC-IN",
    "futureTerritory": null,
    "salesOrg": "Lenovo APAC",
    "territoryMoveReason": null,
    "geographicUnit": "IN-South",
    "salesOffice": "Bengaluru",
    "assignedOwnerId": "system-user-uuid",
    "assignedOwnerName": "alice@lenovo.com",
    "recordOwnerId": "system-user-uuid",
    "recordOwnerName": "alice@lenovo.com"
  },
  "notes": []
}
```

**`notes` semantics**
- Empty list when the schema is fully migrated.
- Otherwise contains a single human-readable diagnostic, e.g.
  `"Lenovo-custom columns missing — run sql/2026_06_account_customer_info_schema.sql (16 column(s) absent: lvo_subsegment, lvo_gtmsegment, …)."`
- The FE can ignore `notes` in production; it's intended for ops diagnosis.

**404** when the account does not exist.

This endpoint is **read-only** in v1. A future user story will add
`PATCH /api/accounts/{accountId}/customer-information` for editable
fields together with audit-log entries (`entityType: account`).

### GET `/api/accounts/{accountId}/contacts`
Account-level roster used by the **Manage Contacts Linked to an Account**
screen. Returns the primary contact and any number of additional
contacts, each enriched with the delete-eligibility chip the FE uses to
grey out the trash-can icon.

`phone` is resolved at runtime from `contact.telephone1`,
`contact.mobilephone`, or `contact.lvo_phone` — whichever exists on the
deployed schema (see `sql/2026_06_contact_extension.sql`).

**Response 200**
```json
{
  "accountId": "A0000001-...-001",
  "primary": {
    "id": "AC-...-A1B2C3D4",
    "accountId": "A0000001-...-001",
    "contactId": "C0000001-...-099",
    "name": "Rajesh Kumar",
    "firstName": "Rajesh",
    "lastName": "Kumar",
    "role": "Primary",
    "isPrimary": true,
    "lastTouchDate": "2026-05-21T09:15:00+00:00",
    "jobTitle": "VP IT Infrastructure",
    "email": "rajesh.kumar@infosys.com",
    "phone": "+91 98765 00000",
    "canDelete": false,
    "deleteRestrictionCode": "ERR_MSG_0008",
    "deleteRestrictionMessage": "This contact is the primary contact for the account. Demote them or assign a new primary contact before removing."
  },
  "others": [
    {
      "id": "AC-...-7E8F9A0B",
      "contactId": "C0000001-...-100",
      "name": "Sumit Reddy",
      "firstName": "Sumit",
      "lastName": "Reddy",
      "role": "CTO",
      "isPrimary": false,
      "email": "sumit.reddy@infosys.com",
      "phone": "+91 98765 00001",
      "canDelete": false,
      "deleteRestrictionCode": "ERR_MSG_0009",
      "deleteRestrictionMessage": "This contact is associated with one or more active deals. Remove the contact from those deals before deleting them from the account."
    }
  ],
  "total": 2
}
```

---

### POST `/api/accounts/{accountId}/contacts`
Two valid request shapes — choose one:

#### A. Create-and-link (used by the UI form)
Creates a new row in `contact` and immediately links it to the account.
`firstName` and `lastName` are required; the rest are optional.

```json
{
  "firstName": "Rajesh",
  "lastName":  "Kumar",
  "email":     "rajesh.kumar@infosys.com",
  "phone":     "+91 98765 00000",
  "jobTitle":  "VP IT Infrastructure",
  "role":      "Primary Decision Maker",
  "isPrimary": true
}
```

#### B. Attach existing
Provide a known `contactId` to link an existing record without mutating it.

```json
{ "contactId": "C0000001-...-099", "role": "Champion", "isPrimary": false }
```

**Validation (raises 422)**

| Trigger | Code |
|---|---|
| Both `contactId` AND `firstName`/`lastName` supplied | `ERR_MSG_0013` |
| Neither path is satisfied | `ERR_MSG_0013` |
| Missing one of `firstName`/`lastName` in flow A | `ERR_MSG_0013` |
| Invalid email format | `ERR_MSG_0013` |
| Invalid phone format (regex/digit-count) | `ERR_MSG_0013` |

**Other failures**

| Status | Code | Trigger |
|---|---|---|
| 404 | — | Account or `contactId` not found |
| 409 | `DUPLICATE_CONTACT` | Same `contactId` already linked to this account |

**Response 201** — same shape as a roster entry:
```json
{
  "id": "AC-...-NEW1",
  "accountId": "A0000001-...-001",
  "contactId": "C0000001-...-NEW",
  "firstName": "Rajesh",
  "lastName":  "Kumar",
  "role": "Primary Decision Maker",
  "isPrimary": true,
  "email": "rajesh.kumar@infosys.com",
  "phone": "+91 98765 00000",
  "canDelete": false,
  "deleteRestrictionCode": "ERR_MSG_0008",
  "deleteRestrictionMessage": "This contact is the primary contact …"
}
```

The audit-log entry attached to this mutation carries `code: SUCC_MSG_0008`
in its `diff` payload — the FE renders the corresponding success toast.

---

### PATCH `/api/accounts/{accountId}/contacts/{contactLinkId}`
Update any subset of fields. Link-row fields (`role`, `isPrimary`) and
contact-row fields (`firstName`, `lastName`, `email`, `phone`, `jobTitle`)
are routed to the right table automatically. Setting `isPrimary=true`
auto-demotes the previous primary.

**Body** (every field optional)
```json
{
  "firstName": "Rajesh",
  "lastName":  "Kumar",
  "email":     "rajesh.k@infosys.com",
  "phone":     "+91 98765 00012",
  "jobTitle":  "SVP IT Infrastructure",
  "role":      "Decision Maker",
  "isPrimary": true
}
```

| Status | Code | Trigger |
|---|---|---|
| 422 | `ERR_MSG_0013` | Invalid email or phone format |
| 404 | — | Account or link not found |

**Response 200** — refreshed `AccountContactRef`. The audit-log diff carries
`code: SUCC_MSG_0009`.

---

### DELETE `/api/accounts/{accountId}/contacts/{contactLinkId}`
Soft-delete the link (sets `statecode='Inactive'` on the
`lvo_accountcontact` row). The underlying `contact` row is **never deleted**
— acceptance criteria says "removing a contact from an account does not
delete the contact from the system (only disassociates)".

**Pre-flight gate (returns 409)**

| Code | Trigger |
|---|---|
| `ERR_MSG_0008` | The link is currently the **primary contact** on the account |
| `ERR_MSG_0009` | The contact is referenced by at least one **active opportunity** (`opportunity.statecode='Open'` AND `lvo_opportunitycontact.statecode='Active'`) |

A 409 response includes the affected deal IDs so the FE can deep-link the
seller to clear the dependency:
```json
// 409 — ERR_MSG_0009
{
  "detail": {
    "code": "ERR_MSG_0009",
    "message": "This contact is associated with one or more active deals. …",
    "affectedDealIds": ["O0000001-...-007", "O0000001-...-013"]
  }
}
```

**Success 200**
```json
{
  "accountId":     "A0000001-...-001",
  "contactLinkId": "AC-...-A1B2C3D4",
  "code":          "SUCC_MSG_0007",
  "message":       "Contact has been removed from the account."
}
```

---

### GET `/api/accounts/{accountId}/contacts/{contactLinkId}/delete-eligibility`
Pre-flight check used by the FE confirmation dialog. Returns the same
`canDelete` block embedded in `/contacts`, but on a single link — useful
when the user has the page open for a while and you want to revalidate
just before showing `CONF_MSG_0003`.

**Response 200 — eligible**
```json
{
  "accountId":     "A0000001-...-001",
  "contactLinkId": "AC-...-7E8F",
  "canDelete":     true,
  "code":          "CONF_MSG_0003",
  "message":       "Are you sure you want to remove this contact from the account? This action may impact associated deals.",
  "affectedDealIds": []
}
```

**Response 200 — blocked (note: still 200 — this is informational, not an error)**
```json
{
  "accountId":     "A0000001-...-001",
  "contactLinkId": "AC-...-7E8F",
  "canDelete":     false,
  "code":          "ERR_MSG_0009",
  "message":       "This contact is associated with one or more active deals. …",
  "affectedDealIds": ["O0000001-...-007"]
}
```

### POST `/api/accounts/{accountId}/recompute-status`
Force a re-derivation of `accountType`, `status` and `lastInteraction`.
The deal-write paths already trigger this asynchronously; this endpoint is
exposed for ops/debug use.

**Response 200**
```json
{
  "id": "A0000001-...-001",
  "accountType": "Customer",
  "status": "At-Risk",
  "lastInteraction": "2026-05-21T09:15:00+00:00",
  "message": "Account derived fields recomputed."
}
```

---

## 11. Background recalculation

Every successful PATCH/POST/DELETE on a deal-related route enqueues a
deal-health recalculation via FastAPI `BackgroundTasks`. The recalc runs
**after** the HTTP response is returned, so writes stay fast.

* The synchronous response you receive reflects the **previous** health
  score (if any). To get the freshly-computed score, re-fetch the deal a
  moment later or call `GET /health`.
* For ops use, `POST /health/recalculate` runs synchronously and returns
  the new score immediately.
* The deal-recalc cascade also enqueues an **account recalc** for the
  owning account so `accountStatus` and `lastInteraction` follow deal
  changes automatically.
* Batch scripts refresh everything from cold:
  ```bash
  # Run deal-health first because account status depends on it.
  python -m app.jobs.recalc_health
  python -m app.jobs.recalc_accounts

  # Smoke-test variants:
  python -m app.jobs.recalc_health   --limit 5 --verbose
  python -m app.jobs.recalc_accounts --limit 5 --verbose
  ```

---

## 12. Sprint 1A · Meetings Resolver

> **New in v0.14.0** — consumed by the Note-Taking Agent (Lenovo-AIBackend)
> before the bot schedules itself to join a Teams meeting.

### `POST /api/meetings/resolve-opportunity`

Match a meeting (attendee emails + subject) to its CRM opportunity.

**Request body:**
```jsonc
{
  "attendeeEmails": ["k.richter@db.com", "rajesh.k@infosys.com"],
  "subject":        "ThinkPad Fleet Review",   // optional, used as score boost
  "organiserEmail": "seller@lenovo.com"        // optional, reserved for future
}
```

**`200 OK`:**
```jsonc
{
  "opportunityId":        "B0000001-0001-0001-0001-000000000001",
  "accountId":            "6dc95c38-9237-4ce9-84d3-f5d1f7431965",
  "opportunityName":      "JPMorgan – Trader Workstation Refresh",
  "accountName":          "Deutsche Bank AG",
  "matchScore":           0.667,
  "matchedBy":            "contact_email",   // 'contact_email' | 'subject_keyword' | 'both'
  "matchedContactCount":  2
}
```

**`404 Not Found`:**
```jsonc
{
  "detail": {
    "code":    "NO_CONTACT_MATCH",  // or "NO_DEAL_MATCH"
    "message": "None of the supplied attendee emails match an active contact in the CRM."
  }
}
```

**`422`:** `attendeeEmails` is empty or contains only blank strings.

### Algorithm

1. Lower-case the supplied emails; lookup against `contact.emailaddress1`
   where `contact.statecode = 'Active'`.
2. Walk `lvo_opportunitycontact` (Active links only) → `opportunity`
   (excluding `Canceled` / `Cancelled` / `Lost` deals).
3. Group by deal; count distinct matched contacts per deal.
4. Score = `matched_contacts / total_supplied_emails` + `0.3` if subject
   tokens (≥ 4 chars) overlap with the deal name (clamped to 1.0).
5. Return the highest-scoring deal. Ties broken by contact count, then
   alphabetic by name.

### Caller guidance (Note-Taking Agent)

* **Threshold:** v1 recommendation is `matchScore ≥ 0.5`. Below that,
  the bot should still join (story says "join the meeting") but persist
  `opportunity_id = NULL` so the meeting isn't tagged to a wrong deal.
* **404 is normal traffic:** customer-organised meetings, internal
  meetings, and meetings with brand-new contacts will all 404. Don't
  log them as errors.
* **5xx / timeouts ARE errors:** retry once after 1 s; if still failing,
  join untagged and surface to the bot's ops dashboard.

---

## 13. Sprint 1A · Contact-by-email Resolver (US02)

> **New in v0.15.0** — consumed by the Note-Taking Agent (Lenovo-AIBackend)
> ONCE at meeting start to enrich attendee emails with CRM context for
> transcript speaker tagging. The bot then tags each utterance locally —
> no per-utterance round-trip.

### `POST /api/contacts/resolve-by-emails`

Bulk-resolve attendee emails to CRM contact + opportunity-role context.

**Request body:**
```jsonc
{
  "emails": [
    "k.richter@db.com",
    "rajesh.k@infosys.com",
    "seller@lenovo.com",
    "unknown@nope.com"
  ]
}
```

**`200 OK`:**
```jsonc
{
  "results": [
    {
      "email":       "k.richter@db.com",
      "contactId":   "E7CF7AAF-CD45-4A9A-9C6B-39BA8D1B6C1A",
      "name":        "Klaus Richter",
      "jobTitle":    "CTO",
      "accountId":   "6dc95c38-9237-4ce9-84d3-f5d1f7431965",
      "accountName": "Deutsche Bank AG",
      "role":        "Decision Maker"
    },
    {
      "email":       "rajesh.k@infosys.com",
      "contactId":   "...",
      "name":        "Rajesh Kumar",
      "jobTitle":    "Engagement Manager",
      "accountId":   "...",
      "accountName": "Infosys",
      "role":        null    // contact exists but isn't on any active opportunity
    },
    {
      "email":       "unknown@nope.com",
      "contactId":   null,
      "name":        null,
      "jobTitle":    null,
      "accountId":   null,
      "accountName": null,
      "role":        null
    }
  ]
}
```

**`422 Unprocessable Entity`:**
```jsonc
{
  "detail": {
    "code":    "EMPTY_EMAILS",
    "message": "emails contains no usable addresses."
  }
}
```

### Algorithm

1. Lower-case + de-dupe the incoming email list (preserves order).
2. Lookup `contact` rows where `LOWER(emailaddress1)` matches AND
   `statecode = 'Active'`.
3. For each matched contact, walk `lvo_opportunitycontact` (Active
   links) → `opportunity` (excluding `Canceled` / `Cancelled` / `Lost`
   deals) and pick the most senior link:
     - `lvo_isdecisionmaker = TRUE` wins
     - Tie-break alphabetic by `lvo_role`
4. Hydrate `account.name` for the IDs collected.
5. Return one entry per supplied email. Unmatched emails get all NULL
   fields — the bot uses that signal to render the speaker as
   `"Unknown Attendee"` in the transcript.

### Caller guidance (Note-Taking Agent)

* **Call ONCE per meeting** — at meeting start, after the bot has
  PATCHed `bot_status` to `joined`, sent `CONF_MSG_0004` in chat, and
  created the transcript via `POST /transcripts/`. The bot caches the
  result locally and tags each utterance from the lookup map.
* **Always 200** — even when zero emails match. Unknown attendees come
  back with `contactId = null`. Don't treat `null` as an error.
* **5xx / timeouts ARE errors:** retry once after 1 s; if still failing,
  fall back to `"Unknown Attendee"` for everyone — the transcript still
  gets recorded, just without CRM tags.
* **`role` semantics:** value reflects the contact's most senior role
  across ALL their opportunity-contact links. Two contacts with the
  same name on different deals will both come back with their per-deal
  role surfaced (the deal isn't part of this lookup; the bot already
  knows it from US01's resolver).

---

## 13b. Contact search by name (AI chat)

> **New in v0.19.0** — consumed by the AI / chat orchestrator (Lenovo-AIBackend)
> when a seller types a natural-language prompt that names a person but does
> **not** supply a CRM UUID or email — e.g. *"schedule a call with John at
> Deutsche Bank"*. The LLM extracts `name` (mandatory) and `account` (optional)
> and this endpoint fuzzy-matches against the D365 `contact` table.

### When to use this vs other contact endpoints

| Endpoint | Lookup key | Scope |
|----------|------------|-------|
| `GET /api/contacts/search` | **Name** (+ optional account name hint) | Global active `contact` rows |
| `POST /api/contacts/resolve-by-emails` | Email address | Global active `contact` rows |
| `GET /api/contacts?sellerId=` | Seller portfolio | Contacts on the seller's **open** deals only |
| `GET /api/accounts/{accountId}/contacts` | Account UUID | Contacts linked to one account |
| `GET /api/opportunities/{opportunityId}/contacts` | Opportunity UUID | Contacts linked to one deal |

---

### `GET /api/contacts/search`

Fuzzy-resolve a spoken or typed contact name to CRM rows for scheduling,
task creation, or follow-up flows.

**Example request**
```
GET /api/contacts/search?name=John&account=Deutsche%20Bank&limit=10
```

| Query | Required | Default | Notes |
|-------|----------|---------|-------|
| `name` | **yes** | — | First name or partial name from the chat prompt. Trimmed and lower-cased server-side before matching. Case-insensitive partial match (`ILIKE '%term%'`) on `contact.firstname` **and** `contact.fullname`. |
| `account` | no | — | Account **name** hint (not a UUID). When supplied, only contacts linked to a matching account are returned (see Algorithm). Trimmed and lower-cased server-side; matched with `ILIKE '%term%'` on `account.name`. |
| `limit` | no | `25` | Maximum candidates returned. Min `1`, max `50`. |

**`200 OK` (one or more matches):**
```jsonc
{
  "name": "john",
  "account": "deutsche bank",
  "total": 1,
  "items": [
    {
      "contactId":   "E7CF7AAF-CD45-4A9A-9C6B-39BA8D1B6C1A",
      "name":        "John Smith",
      "firstName":   "John",
      "lastName":    "Smith",
      "email":       "john.smith@deutschebank.com",
      "jobTitle":    "CTO",
      "phone":       "+49-30-1234567",
      "accountId":   "6dc95c38-9237-4ce9-84d3-f5d1f7431965",
      "accountName": "Deutsche Bank AG"
    }
  ]
}
```

**`200 OK` (no matches):** same envelope with `"total": 0` and `"items": []`.
This is **not** an error — the orchestrator should ask the seller for more
detail or offer to create a new contact.

**`422 Unprocessable Entity`:**
```jsonc
{
  "detail": {
    "code":    "MISSING_NAME",
    "message": "name is required."
  }
}
```

Returned when `name` is omitted, blank, or whitespace-only.

---

### Response field reference (`ContactSearchItem`)

| Field | Source | Notes |
|-------|--------|-------|
| `contactId` | `contact.contactid` | Use this UUID for downstream scheduling / task / deal-contact APIs. |
| `name` | `contact.fullname` (fallback: `firstname` + `lastname`) | Display name. |
| `firstName` | `contact.firstname` | |
| `lastName` | `contact.lastname` | |
| `email` | `contact.emailaddress1` | `null` when absent. |
| `jobTitle` | `contact.jobtitle` | |
| `phone` | `contact.telephone1` → `mobilephone` → `lvo_phone` (first column that exists in the dump) | Resolved at runtime; `null` when no phone column or value. |
| `accountId` | `account.accountid` | Best-effort primary account — see Algorithm step 5. |
| `accountName` | `account.name` | Paired with `accountId`. |

Top-level envelope fields:

| Field | Notes |
|-------|-------|
| `name` | Normalised search term echoed back (trimmed, lower-cased). |
| `account` | Normalised account hint echoed back when supplied; `null` when omitted. |
| `total` | Count of items in this response (≤ `limit`). |
| `items` | Ranked list of contact candidates. |

---

### Data sources

| Purpose | Table(s) |
|---------|----------|
| Contact identity + name / email / title | **`contact`** (primary) |
| Phone | **`contact`** (`telephone1` / `mobilephone` / `lvo_phone`) |
| Account filter (when `account` supplied) | **`lvo_accountcontact`** → **`account`**, or **`lvo_opportunitycontact`** → **`opportunity`** → **`account`** |
| Account enrichment on results | **`lvo_accountcontact`** → **`account`** (primary link preferred), fallback **`lvo_opportunitycontact`** → **`opportunity`** → **`account`** |

Only contacts where `contact.statecode = 'Active'` (or `NULL`) are considered.
Link tables must have `statecode = 'Active'`. Opportunity-backed paths only
include deals where `opportunity.statecode` is `Open` or `NULL`.

> **Not seller-scoped:** unlike `GET /api/contacts?sellerId=`, this endpoint
> searches the full active contact table. Pass `account` when the prompt
> mentions a company to avoid cross-portfolio ambiguity.

---

### Algorithm

1. **Validate** — reject with `422 MISSING_NAME` if `name` is blank.
2. **Normalise** — trim + lower-case `name`; same for `account` when present.
3. **Name query** — `SELECT` from `contact` where:
   - `statecode = 'Active'` (or `NULL`), **and**
   - `LOWER(firstname) LIKE '%{name}%'` **OR** `LOWER(fullname) LIKE '%{name}%'`.
4. **Account filter** (only when `account` is supplied) — collect `contactId`
   values linked to any account whose `name ILIKE '%{account}%'` via:
   - `lvo_accountcontact` (Active) → `account`, **or**
   - `lvo_opportunitycontact` (Active) → `opportunity` (Open) → `account`.
   Intersect with the name query. If the account filter matches zero contacts,
   return `200` with `total: 0` immediately.
5. **Rank** matches (best first):
   1. Exact `firstname` match (case-insensitive)
   2. `firstname` starts with term
   3. `firstname` contains term
   4. Exact `fullname` match
   5. `fullname` starts with term
   6. `fullname` contains term
   7. Alphabetic by display name
6. **Truncate** to `limit` (default 25, max 50).
7. **Hydrate** phone (bulk read) and account (`lvo_accountcontact` primary
   link first; fallback to first open-opportunity account).
8. **Return** envelope with normalised `name`, `account`, `total`, `items`.

---

### Caller guidance (AI / chat orchestrator)

* **Typical flow** — LLM extracts entities from the user message → call this
  endpoint → if `total === 1`, use `contactId` directly → if `total > 1`,
  ask the seller to pick → if `total === 0`, retry without `account` or ask
  for email / full name.
* **Always 200** when `name` is supplied — zero hits is not an error.
* **Pass `account` when the prompt mentions a company** — e.g. *"call John at
  Deutsche Bank"* → `name=John&account=Deutsche Bank`. This sharply reduces
  ambiguity for common first names.
* **First-name-only is expected** — the search is intentionally fuzzy on
  `firstname`; do not require the seller to spell out a full name.
* **Use `contactId`, not `name`** for downstream writes (schedule meeting,
  attach to deal, create task). Never invent UUIDs from the spoken name.
* **5xx / timeouts ARE errors** — retry once after 1 s; on persistent failure,
  ask the seller to provide an email or pick the contact manually.

**Example prompts → query params**

| User says | Suggested call |
|-----------|----------------|
| "Schedule a call with Raj" | `?name=Raj` |
| "Follow up with Klaus at Deutsche Bank" | `?name=Klaus&account=Deutsche Bank` |
| "Email Priya from Infosys" | `?name=Priya&account=Infosys` |

---

## 14. Seller portfolio contacts

> **New in v0.16.0** — de-duplicated contact roster for a seller's open deals.

### `GET /api/contacts`

| Query | Required | Default | Notes |
|-------|----------|---------|-------|
| `sellerId` | **yes** | — | UUID matching `opportunity.owninguser` |
| `page` | no | `1` | 1-based |
| `pageSize` | no | `25` | Max `100` |

Sources active ``lvo_opportunitycontact`` rows on opportunities where
``owninguser = sellerId`` and ``statecode = Open``. Contacts are de-duplicated
by ``contactId``; when a contact is on multiple deals,
``linkedOpportunityCount`` reflects the total.

**200:**
```jsonc
{
  "sellerId": "055DAFE7-9840-451D-8328-5F70A6326C03",
  "page": 1,
  "pageSize": 25,
  "total": 12,
  "totalPages": 1,
  "items": [
    {
      "contactId": "E7CF7AAF-CD45-4A9A-9C6B-39BA8D1B6C1A",
      "name": "Klaus Richter",
      "firstName": "Klaus",
      "lastName": "Richter",
      "email": "k.richter@db.com",
      "jobTitle": "CTO",
      "phone": "+49-30-123456",
      "accountId": "6dc95c38-9237-4ce9-84d3-f5d1f7431965",
      "accountName": "Deutsche Bank AG",
      "opportunityId": "CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B",
      "opportunityName": "Deutsche Bank — Workstation Refresh",
      "role": "Decision Maker",
      "isDecisionMaker": true,
      "linkedOpportunityCount": 2,
      "lastTouchDate": "2026-06-15T14:30:00"
    }
  ]
}
```

**422:** `{ "detail": "sellerId is required." }`

---

## 15. Sprint 2 · What Changed (US 1.1)

> **Full handoff** (who calls which API, pending work, event producers):
> [SPRINT_2_US11_WHAT_CHANGED_BACKEND_HANDOFF.md](./SPRINT_2_US11_WHAT_CHANGED_BACKEND_HANDOFF.md)

Portfolio-level notification panel + activity timeline for the signed-in
seller. Events are aggregated at query time from `lvo_activity`,
`lvo_audit_log`, `lvo_dealrisk`, and overdue `lvo_nextaction` rows on the
seller's open opportunities. Read-state is persisted in
`lvo_notification_read` (see `sql/2026_07_create_lvo_notification_read.sql`).

### `GET /api/notifications`

| Query | Required | Notes |
|-------|----------|-------|
| `sellerId` | yes | UUID — matches `opportunity.owninguser` |
| `limit` | no | Default `6`, max `6` |
| `types` | no | Comma-separated: `email`, `meeting`, `crm_update`, `risk`, `task` |

Header `X-User-Id` (optional) — used to exclude CRM changes authored by the
viewer from the panel feed.

**200:**
```jsonc
{
  "sellerId": "A1B2C3D4-...",
  "limit": 6,
  "items": [
    {
      "id": "activity:E7CF7AAF-...",
      "activityType": "email",
      "title": "Re: pricing",
      "summary": "Please review the updated quote.",
      "accountId": "6dc95c38-...",
      "accountName": "Deutsche Bank AG",
      "opportunityId": "F1A2B3C4-...",
      "opportunityName": "DB Workstation Refresh",
      "eventAt": "2026-06-15T14:30:00",
      "isRead": false,
      "linkType": "outreach",
      "linkId": "F1A2B3C4-...",
      "actor": "7D26391E-D020-474E-B1CA-53E6B6C71487",
      "actorName": "Priya Nair",
      "direction": "inbound",
      "categoryLabel": "Email received"
    }
  ]
}
```

**500:** `{ "detail": "ERR_MSG_0020" }` — feed aggregation failure.

### `PATCH /api/notifications/{notificationId}/read`

| Query | Required |
|-------|----------|
| `sellerId` | yes |

`notificationId` is the synthetic feed key (`activity:…`, `audit:…`, etc.).

### `GET /api/activity-timeline`

Same `sellerId` / `types` filters as the panel. Includes seller-authored CRM
changes. Paginated.

| Query | Default |
|-------|---------|
| `page` | `1` |
| `pageSize` | `25` (max `100`) |

**200:**
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

---

## 16. Sprint 2 · Quarter Pulse (US 1.2)

Home dashboard card — seller-scoped quota attainment, pipeline coverage, and
days remaining in the current **fiscal quarter**. Metrics are computed live
from the D365 Postgres mirror on each request (`opportunity` rows filtered by
`owninguser`). Manual quota overrides live in `lvo_seller_quota` when D365
has no goal (see `sql/2026_08_create_lvo_seller_quota.sql`).

Fiscal year defaults to **April start** (`FISCAL_YEAR_START_MONTH=4`).

### `GET /api/quarter-pulse`

| Query | Required | Notes |
|-------|----------|-------|
| `sellerId` | yes | UUID — matches `opportunity.owninguser` |

**200 (quota configured):**
```jsonc
{
  "quarterLabel": "Q3",
  "fiscalYear": 2026,
  "daysLeftInQuarter": 27,
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

**200 (quota not configured):** `quotaConfigured: false`, attainment/coverage
`displayValue: "Not set"`, `prompt` set to configure-quota message; `daysLeftInQuarter`
still populated.

**422:** `{ "detail": "sellerId is required." }`  
**500:** `{ "detail": "ERR_MSG_0021" }`

### `PUT /api/quarter-pulse/quota`

Phase-1 manual quota entry when D365 has no target.

| Query / header | Required |
|----------------|----------|
| `sellerId` | yes |
| `X-User-Id` | no (audit `setBy`) |

**Body:**
```jsonc
{
  "quotaAmount": 1000000,
  "fiscalYear": 2026,
  "fiscalQuarter": 3,
  "currencyCode": "USD"
}
```

`fiscalYear` / `fiscalQuarter` default to the current fiscal period when omitted.

**200:** upsert metadata + refreshed `quarterPulse` object (same shape as GET).

---

## 17. Sprint 2 · US 1.3 — Task Pending badge (Home header)

Seller-scoped open task counts for the Home dashboard header badge. **Frontend merges**
D365 next actions with Execute Workspace todos:

| Source | Endpoint | `source` field |
|--------|----------|----------------|
| CRM next actions (`lvo_nextaction`) | `GET /api/tasks/pending-summary` | `"d365"` |
| Execute To-Do (`tbl_to_do_list`) | `GET /ai-api/todos/summary` (AIBackend) | `"ai"` |

Combined badge: `count = d365.count + ai.count`; `hasOverdue = d365.hasOverdue || ai.hasOverdue`;
`badgeColor = "red"` when `hasOverdue`.

### `GET /api/tasks/pending-summary`

| Query | Required | Notes |
|-------|----------|-------|
| `sellerId` | yes | UUID — matches `opportunity.owninguser` |

Counts open `lvo_nextaction` rows on the seller's open opportunities (`statecode=Active`,
`lvo_status=Open`).

**200:**
```jsonc
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

**422:** `{ "detail": "sellerId is required." }`  
**500:** `{ "detail": "ERR_MSG_0022" }`

### AIBackend companion — `GET /ai-api/todos/summary`

Same response shape with `source: "ai"`. Requires `sellerId` query param. Run
`sql/2026_08_us05_todo_seller_scope.sql` on AIBackend Postgres before using seller scope.

`GET /ai-api/todos?sellerId=` returns open tasks sorted **overdue first**, then due today,
then future due dates (Execute Workspace To-Do list).

---

## 19. Sprint 2 · Pre-Meeting Briefing — D365 facts

`GET /api/briefing/context` — seller-scoped CRM facts + traceable signals for briefing generation (called by AIBackend).

| Query | Required |
|-------|----------|
| `sellerId` | yes |
| `opportunityId` | yes |
| `accountId` | no |
| `maxSummaryWords` | no (default 100) |

**200:** `account`, `deal` (with `competitorIntel` or `competitorMessageCode: INF_MSG_0004`), `signals[]` each with `source`.

**403:** seller does not own opportunity · **404:** opportunity not found · **500:** `ERR_MSG_0024`

Full briefing card: `GET /ai-api/meeting-prep/{meetingId}/briefing?sellerId=` (AIBackend).

---

## 20. Sprint 2 · US 3.2.1 — Sales Operating Model · Interview-First Setup

Admin Policy Layer — capture organizational intent via structured interview questions.

### GET `/api/sales-operating-model/interview-setup?role=`

| Query | Required | Values |
|-------|----------|--------|
| `role` | no (default `national_manager`) | `national_manager` \| `regional_manager` \| `seller_manager` |

**200:** `roleDisplay`, `scopeLabel`, `questions[]`, `responses` (drafts), `savedResponses`, `capturedCount`, `totalQuestions`, `verifyEnabled`, `intentCardStatus`.

### PUT `/api/sales-operating-model/interview-setup/{role}/draft`

**Body:** `{ "responses": [{ "questionId", "text" }] }`  
**Header:** `X-User-Id` optional.

### POST `/api/sales-operating-model/interview-setup/{role}/save`

Atomic commit from Verify & Edit panel. **500:** `ERR_MSG_0021` (no partial save).

### GET `/api/sales-operating-model/intent-cards`

**200:** `{ "items": [{ "role", "roleDisplay", "scopeLabel", "status", "configuredAt" }] }`  
`status`: `NOT_CONFIGURED` \| `CONFIGURED`.

### GET `/api/sales-operating-model/context-lake`

**AI consumption** — latest saved interview responses per role. Empty `roles` = not configured.

### Interview question CRUD

- `GET /api/sales-operating-model/interview-questions?role=`
- `POST /api/sales-operating-model/interview-questions`
- `PATCH /api/sales-operating-model/interview-questions/{questionId}`
- `DELETE /api/sales-operating-model/interview-questions/{questionId}`

Handoff: `SPRINT_2_US321_SOM_INTERVIEW_SETUP_BACKEND_HANDOFF.md`.

---

## 21. Sprint 2 · US 3.2.2 — Organizational Intent Setup

Six fixed intent cards: `outcome`, `motion`, `focus`, `behavioral`, `constraint`, `tradeoff`.

### GET `/api/sales-operating-model/organizational-intent-cards`

**200:** `{ "items": [{ "intentType", "displayName", "status", "isTimeboxed", "isGuardrail", "lastSyncedAt", "fieldPreview" }] }`

### GET `/api/sales-operating-model/organizational-intent-cards/{intentType}`

**200:** full `fields`, `fieldLabels`, `guardrailWarning` (constraint only), `expiryDate` (focus).

### PUT `/api/sales-operating-model/organizational-intent-cards`

**Bulk save** — persist one or more cards atomically (single Context Lake rebuild).

**Body:**
```json
{
  "cards": {
    "outcome": { "fields": { "revenueAndQuality": "...", "...": "..." } },
    "focus": { "fields": { "quarterType": "...", "expiryDate": "2026-09-30", "...": "..." } }
  }
}
```

Include every card the Save button edits (all 6 for full setup). Omitted cards are left unchanged.

**200:** `{ "items": [ ... per saved card same shape as single PUT ... ], "allConfigured", "configuredCount", "totalCount", "successCode" }`

**422:** `{ "code", "field", "message", "intentType" }` — first validation failure; nothing is committed.

### PUT `/api/sales-operating-model/organizational-intent-cards/{intentType}`

**Single-card save** (unchanged).

**Body:** `{ "fields": { ... } }` — see handoff for per-card field keys.

**200:** `{ "status": "CONFIGURED", "lastSyncedAt", "allConfigured", "successCode": "SUCC_MSG_0017" }` when all 6 configured.

**422:** `{ "code": "ERR_MSG_0023", "field", "message" }` required field · `{ "code": "ERR_MSG_0022", "field": "marginFloors" }` invalid %.

**500:** `{ "detail": "ERR_MSG_0024" }` save failure (no partial commit).

### GET `/api/sales-operating-model/configuration-status`

**200:** `{ "allConfigured", "configuredCount", "totalCount", "successCode" }`

### GET `/api/sales-operating-model/interview-intent-cards`

Replaces deprecated `/intent-cards` (3.2.1 interview role tabs).

### Context Lake v2

`GET /context-lake` returns `{ "version": 2, "interview": { "roles": {} }, "organizationalIntents": {} }`.

Handoff: `SPRINT_2_US322_SOM_ORGANIZATIONAL_INTENT_BACKEND_HANDOFF.md`.

---

## 22. Sprint 2 · US 3.2.3 — Delete Intent Card

`DELETE /api/sales-operating-model/organizational-intent-cards/{intentType}`

Clears a **CONFIGURED** card: resets to `NOT_CONFIGURED`, removes field values, rebuilds Context Lake so AI agents stop consuming that intent.

### Delete one field row inside a card

`DELETE /api/sales-operating-model/organizational-intent-cards/{intentType}/fields/{fieldKey}`

Removes a single metric from a **CONFIGURED** card (trash icon on one line).

| `fieldKey` examples (Outcome) | `revenueAndQuality`, `predictability`, `additionalContext` |
|-------------------------------|--------------------------------------------------------------|

**200:** `{ intentType, status, deletedField, fields, fieldPreview, allConfigured, configuredCount, totalCount, successCode }`

- Optional field removed → card stays `CONFIGURED` when remaining required fields are complete.
- Required field removed → card becomes `NOT_CONFIGURED`; other saved values are kept until re-saved.
- Works on **partial** cards (`NOT_CONFIGURED` with leftover field values) — not only fully configured cards.

**404:** field not present on the card (or card has no saved field values).

### Custom metrics (add / edit / delete)

User-defined metric rows (label + description), same UX pattern as **Add question** in interview setup.

`GET .../organizational-intent-cards/{intentType}` includes `customMetrics[]`: `{ id, label, description, sortOrder }`.

| Method | Path | Body |
|--------|------|------|
| `POST` | `.../organizational-intent-cards/{intentType}/metrics` | `{ "label": "4. Risk Posture", "description": "Leadership mode = ...", "sortOrder": 4 }` |
| `PATCH` | `.../organizational-intent-cards/{intentType}/metrics/{metricId}` | `{ "label"?, "description"?, "sortOrder"? }` |
| `DELETE` | `.../organizational-intent-cards/{intentType}/metrics/{metricId}` | — |

**POST 201:** returns the created metric item. `sortOrder` is optional (auto-assigned after built-in fields).

**DELETE 200:** same shape as field delete, with updated `customMetrics` and `fieldPreview`.

Stored under `lvo_fields.customMetrics` on the card. Custom metrics do not affect `CONFIGURED` / `NOT_CONFIGURED` status (only required standard fields do).

**Full card delete** (`DELETE .../organizational-intent-cards/{intentType}`):

**200:** `{ intentType, displayName, status: NOT_CONFIGURED, deletedAt, configuredCount, totalCount, message }`

**409:** card is already `NOT_CONFIGURED` (nothing to delete).

**500:** `ERR_MSG_0024` on transaction failure.

**Header:** `X-User-Id` optional (audit). Cancel = FE only (no API call).

---

## 24. Sprint 2 · US 3.4.1 — Timeline Classification & Canonical Sales Clock

> **Depends on** US 3.2.2 — section locked until `GET /configuration-status` reports `allConfigured: true` (6/6 organizational intents).  
> **Handoff:** [SPRINT_2_US341_TIMELINE_CLASSIFICATION_BACKEND_HANDOFF.md](./SPRINT_2_US341_TIMELINE_CLASSIFICATION_BACKEND_HANDOFF.md)

Eight fixed cards under **Admin → Sales Operating Model → Timeline classification and canonical sales clock**.

### Who calls these APIs

| Consumer | Endpoints |
|----------|-----------|
| **Admin FE** | All `/timeline-classification-*` routes |
| **Lenovo-AIBackend** | `GET /context-lake` (v3 — `timelineClassification` block) |
| **Sellers / Opportunities UI** | None |

### Section gate

`GET /api/sales-operating-model/timeline-classification-section-status`

Returns `sectionUnlocked` (false until org intent 6/6), `timelineConfiguredCount` / `timelineTotalCount` (8).

### Card CRUD

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/timeline-classification-cards` | Grid — 8 cards, status, preview |
| GET | `/timeline-classification-cards/{cardType}` | Edit panel — fields, labels, defaults |
| PUT | `/timeline-classification-cards/{cardType}` | Save — body `{ "fields": {}, "confirmAgentImpact": true }` |
| GET | `/timeline-classification-configuration-status` | 8/8 + `SUCC_MSG_0018` |

**`cardType` values:** `tempo_classes`, `anchor_definitions`, `signal_expectations_time_band`, `seasonal_delayed_activation`, `acceleration_decay`, `multiyear_programmatic`, `exit_recycle_kill`, `canonical_timeline`

**PUT responses:** `200` with `lastSyncedAt`, `allConfigured`, optional `successCode: SUCC_MSG_0018`.  
**428** if `confirmAgentImpact` is not `true` — `INFO_MSG_0006` (FE shows confirm dialog, then retries).  
**422** — `ERR_MSG_0025` (decay days), `ERR_MSG_0026` (evidence cadence), `ERR_MSG_0027` (guardrail), `ERR_MSG_0028` (required field).  
**403** — `ERR_MSG_0030` section locked.  
**500** — `ERR_MSG_0029` save failure.

Field keys per card: see handoff §8.

### Context Lake v3

`GET /context-lake` adds `timelineClassification` alongside `interview` and `organizationalIntents`. Only `CONFIGURED` cards included. `version` bumps to `3`.

**Phase B (not yet built):** custom Add Card Category, DELETE clear card, AI Concept Assistant API.

**Implemented in v0.23.0:** Phase A — all endpoints above, Context Lake v3, validation, section gate.

---

## 23. Swagger / Interactive Docs
Full interactive documentation is available at:
```
http://localhost:8000/docs
```

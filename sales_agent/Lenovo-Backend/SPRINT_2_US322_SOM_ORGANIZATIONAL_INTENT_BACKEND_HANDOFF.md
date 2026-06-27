# Sprint 2 · US 3.2.2 — Organizational Intent Setup

**Repo:** `Lenovo D365 Sales` (v0.22.0)  
**Depends on:** US 3.2.1 (`sql/2026_10_som_interview_setup.sql`)  
**Migration:** `sql/2026_11_som_organizational_intent.sql`  
**Postman:** [postman/US_SOM_ORGANIZATIONAL_INTENT.postman_collection.json](./postman/US_SOM_ORGANIZATIONAL_INTENT.postman_collection.json) — run folders `00` → `09` in order.

---

## Prerequisites

```bash
psql -f sql/2026_10_som_interview_setup.sql
psql -f sql/2026_11_som_organizational_intent.sql
uvicorn app.main:app --reload --port 8000
```

---

## APIs

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/organizational-intent-cards` | List 6 cards + status + preview |
| GET | `/organizational-intent-cards/{type}` | Edit panel fields |
| PUT | `/organizational-intent-cards/{type}` | Save card → Context Lake |
| DELETE | `/organizational-intent-cards/{type}` | Clear CONFIGURED card (US 3.2.3) |
| GET | `/configuration-status` | 6/6 check + `SUCC_MSG_0017` |
| GET | `/interview-intent-cards` | 3.2.1 role tabs (renamed) |
| GET | `/intent-cards` | **Deprecated** alias |
| GET | `/context-lake` | v2 — interview + organizational |

**Intent types:** `outcome`, `motion`, `focus`, `behavioral`, `constraint`, `tradeoff`

---

## Field keys per card (PUT body `fields`)

### Outcome
`revenueAndQuality`, `predictability`, `progressionExpectation`, `riskPosture`, `additionalContext?`

### Motion
`primaryGrowthLever`, `sellingMotionMix`, `routeToMarket`, `salesCyclePolicy`, `attachExpectation`, `additionalContext?`

### Focus (time-boxed)
`quarterType`, `priorityFocus`, `temporaryDeprioritisation`, `expiryDate` (ISO date), `additionalContext?`

### Behavioral
`multithreadingNorm`, `followUpCadence`, `walkAwayRule`, `coachingLens`, `additionalContext?`

### Constraint (guardrails)
`marginFloors` (array: `{ dealType, minPercent }`), `complianceGates`, `dealDeskTriggers`, `pricingAuthority`, `additionalContext?`

### Trade-off
`priorityRank` (ordered string array), `revenueVsMargin`, `newLogoVsExpansion`, `commitVsUpside`, `additionalContext?`

---

## Errors

| Code | HTTP | When |
|------|------|------|
| `ERR_MSG_0022` | 422 | Margin/pricing % not 0–100 |
| `ERR_MSG_0023` | 422 | Required field empty — `{ code, field, message }` |
| `ERR_MSG_0024` | 500 | Save or delete transaction failed |
| `SUCC_MSG_0017` | 200 | All 6 cards configured (in save response + configuration-status) |

Cancel = FE only (no API).

---

## Context Lake v2 (AI team)

```json
{
  "version": 2,
  "interview": { "roles": { ... } },
  "organizationalIntents": {
    "constraint": {
      "isGuardrail": true,
      "fields": { "marginFloors": [...] }
    },
    "focus": {
      "isTimeboxed": true,
      "expiryDate": "2026-09-30",
      "fields": { ... }
    }
  }
}
```

**AI rules:**
1. Read via `GET /context-lake` or `fetch_som_context_lake()`
2. **Constraint** fields are non-overridable guardrails
3. **Focus** — apply only when `today <= expiryDate`
4. **Trade-off** — use `priorityRank` order for conflicts
5. **Behavioral** — coaching only, not hard enforcement

---

## FE flow

1. `GET /organizational-intent-cards` — grid with CONFIGURED badges
2. Click Configure → `GET /organizational-intent-cards/{type}`
3. Edit fields → `PUT /organizational-intent-cards/{type}`
4. On 422 → highlight `field` from error body
5. When `successCode === SUCC_MSG_0017` → show confirmation banner
6. Use `GET /configuration-status` on page load for global banner

### Delete (US 3.2.3)

1. Show confirmation dialog (FE copy from user story)
2. On **Confirm** → `DELETE /organizational-intent-cards/{type}`
3. On **Cancel** → no API call
4. Refresh list — card shows `NOT_CONFIGURED`, empty `fieldPreview`
5. **409** if already not configured

```bash
curl -X DELETE "http://localhost:8000/api/sales-operating-model/organizational-intent-cards/outcome" \
  -H "X-User-Id: admin@lenovo.com"
```

Card **type** remains in the grid (6 fixed seeds); only **saved values** are removed. Context Lake drops that intent key for AI agents.

---

## Tests

```bash
pytest -q tests/test_som_organizational_intent.py
```

---

## Pending (Phase C)

- `GET .../suggested-values` — rule-based pre-fill from interview Context Lake
- Custom Add Card Category (3.2.2.11a)
- AI Concept Assistant API (3.2.2.12)

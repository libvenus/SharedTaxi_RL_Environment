# Sprint 2 · US 3.2.1 — Interview-First Setup (Sales Operating Model)

**Repo:** `Lenovo D365 Sales` (v0.20.0)  
**User story:** 3.2.1 Interview-First Setup — Capture Operating Model  
**Backend contact:** Sanmay  
**Pairing:** Sanmay (backend) ↔ Namisha (FE) ↔ AI team  
**Last updated:** 2026-06-19

**Linked docs:**
- [API_CONTRACT.md](./API_CONTRACT.md) — §20
- Migration: [sql/2026_10_som_interview_setup.sql](./sql/2026_10_som_interview_setup.sql)
- Postman: [postman/US_SOM_INTERVIEW_SETUP.postman_collection.json](./postman/US_SOM_INTERVIEW_SETUP.postman_collection.json)

---

## Overview

Admins capture organizational intent through **5 structured interview questions per leadership role**. Saved responses update **intent cards** to `CONFIGURED` and populate the **Sales Operating Model Context Lake** for AI agents.

| Role | Scope |
|------|--------|
| `national_manager` | Org-level intent and global guardrails |
| `regional_manager` | Region-specific behaviour and constraints |
| `seller_manager` | Team-level behavioural and execution rules |

---

## Prerequisites

```bash
# 1. Run migration on Postgres (lenovosales)
psql -f sql/2026_10_som_interview_setup.sql

# 2. Start API
uvicorn app.main:app --reload --port 8000
```

---

## All APIs created

| # | Method | Endpoint | Caller | Purpose |
|---|--------|----------|--------|---------|
| 1 | GET | `/api/sales-operating-model/interview-setup?role=` | FE | Load form: questions, drafts, counter, verify flag |
| 2 | PUT | `/api/sales-operating-model/interview-setup/{role}/draft` | FE | Optional autosave while typing |
| 3 | POST | `/api/sales-operating-model/interview-setup/{role}/save` | FE | Commit from Verify & Edit panel |
| 4 | GET | `/api/sales-operating-model/interview-intent-cards` | FE | Section 5.1 interview role CONFIGURED status |
| 4b | GET | `/api/sales-operating-model/intent-cards` | FE | **Deprecated** — use interview-intent-cards |
| 5 | GET | `/api/sales-operating-model/context-lake` | **AI team** | Organizational intent JSON |
| 6 | GET | `/api/sales-operating-model/interview-questions` | Admin | List question bank |
| 7 | POST | `/api/sales-operating-model/interview-questions` | Admin | Add question (2.4a) |
| 8 | PATCH | `/api/sales-operating-model/interview-questions/{questionId}` | Admin | Update question |
| 9 | DELETE | `/api/sales-operating-model/interview-questions/{questionId}` | Admin | Soft-delete question |

---

## FE integration flow

```
1. Admin opens Admin > Sales Operating Model
2. GET /interview-setup?role=national_manager   (default tab)
3. User types → optional PUT .../draft on blur/debounce
4. verifyEnabled === true → enable "Verify & Edit Before Save"
5. Review modal → POST .../save with all responses
6. GET /intent-cards → show CONFIGURED on saved role card
7. capturedCount resets to 0 for next capture cycle on that tab
```

### Response counter (FE or backend)

`capturedCount` = questions with **≥1 non-whitespace character** in draft `responses`.  
Display: `Captured {capturedCount}/{totalQuestions} responses`.

### Save error

**500** `{ "detail": "ERR_MSG_0021" }` — show: *"Configuration could not be saved. Your responses are preserved - please try again."*  
No partial save; review panel stays open with same payload.

---

## API details

### GET `/api/sales-operating-model/interview-setup?role=national_manager`

**200:**
```json
{
  "role": "national_manager",
  "roleDisplay": "National Manager",
  "scopeLabel": "Scope: Org-level intent and global guardrails",
  "questions": [
    { "questionId": "11111111-...", "sortOrder": 1, "text": "What does a successful quarter mean beyond quota?" }
  ],
  "responses": { "11111111-...": "draft text or empty" },
  "savedResponses": { "11111111-...": "last committed text" },
  "capturedCount": 2,
  "totalQuestions": 5,
  "verifyEnabled": true,
  "intentCardStatus": "NOT_CONFIGURED",
  "configuredAt": null
}
```

### POST `/api/sales-operating-model/interview-setup/{role}/save`

**Headers:** `X-User-Id: <admin-id>` (optional audit)

**Body:**
```json
{
  "responses": [
    { "questionId": "11111111-1111-4111-8111-111111111101", "text": "Success means pipeline quality and customer outcomes." }
  ]
}
```

**200:** `intentCardStatus: CONFIGURED`, `capturedCount: 0`, `verifyEnabled: false`

### GET `/api/sales-operating-model/context-lake`

See **AI team integration** below.

---

## Status matrix

### Backend — Done

| Area | Status |
|------|--------|
| Migration + 15 seeded questions | ✅ |
| Interview setup GET per role | ✅ |
| Draft PUT + atomic save POST | ✅ |
| Intent cards GET | ✅ |
| Context Lake GET + rebuild on save | ✅ |
| Question CRUD (soft delete) | ✅ |
| ERR_MSG_0021 on save failure | ✅ |
| AIBackend `fetch_som_context_lake()` client | ✅ |
| Unit tests | ✅ |

### Backend — Future

| Area | Status |
|------|--------|
| Entra ID Admin RBAC enforcement | ⏳ Future (2.2a) |
| Per-manager scoped tabs | ⏳ Future |
| Voice / AI-led interview capture | ⏳ AI team |

### FE — Pending

| UI | API |
|----|-----|
| 3 role tabs + scope label | GET interview-setup |
| 5 question text areas + counter | GET + local state or PUT draft |
| Verify & Edit modal | POST save |
| Intent cards Section 5.1 | GET intent-cards |
| ERR_MSG_0021 banner | POST save 500 |

---

## AI team — integration guide

### What you receive

After an admin saves interview responses, organizational intent is available at:

```
GET http://<D365_HOST>:8000/api/sales-operating-model/context-lake
```

No auth header required in Phase 1 (same as other internal D365 reads). Call from **AIBackend** or any agent service.

### AIBackend client (recommended)

```python
from app.clients.d365_client import fetch_som_context_lake, D365ClientError

try:
    context = fetch_som_context_lake()
except D365ClientError:
    context = {"roles": {}}  # degrade gracefully — agent runs without SOM intent
```

**Env:** `D365_BASE_URL=http://localhost:8000` (already used by briefing client)

### Context Lake JSON shape

```json
{
  "version": 1,
  "cycleId": "00000000-0000-4000-8000-000000000001",
  "updatedAt": "2026-06-19T12:00:00",
  "roles": {
    "national_manager": {
      "roleDisplay": "National Manager",
      "scopeLabel": "Scope: Org-level intent and global guardrails",
      "status": "CONFIGURED",
      "configuredAt": "2026-06-19T12:00:00",
      "interviewResponses": [
        {
          "questionId": "11111111-1111-4111-8111-111111111101",
          "sortOrder": 1,
          "question": "What does a successful quarter mean beyond quota?",
          "response": "Pipeline quality and strategic account growth."
        }
      ]
    }
  }
}
```

Only roles with **saved** responses appear under `roles`. Empty `{}` means operating model not configured yet.

### Where to use Context Lake

Inject into **system prompt context** (not user-visible) for agents that need strategic alignment:

| Agent / feature | How to use |
|-----------------|------------|
| Pre-meeting briefing | Add `national_manager` + `regional_manager` intent to briefing LLM prompt |
| CRM update suggestions | Respect non-negotiable constraints from National Manager Q4 |
| Deal health / coaching | Apply Seller Manager behavioural rules (Q1, Q3, Q5) |
| Outreach drafts | Align tone/motion with National Manager Q3 (net-new vs expansion) |
| To-Do / hygiene | Regional follow-up intensity (Regional Manager Q4) |
| Scheduling assistant | Regional timeline pause/reset rules (Regional Manager Q5) |

### Prompt integration pattern

```python
def build_som_prompt_block(context: dict) -> str:
    roles = context.get("roles") or {}
    if not roles:
        return ""
    lines = ["## Sales Operating Model (organizational intent)"]
    for role_key, block in roles.items():
        lines.append(f"\n### {block.get('roleDisplay', role_key)}")
        lines.append(block.get("scopeLabel", ""))
        for item in block.get("interviewResponses") or []:
            lines.append(f"- Q: {item['question']}")
            lines.append(f"  A: {item['response']}")
    return "\n".join(lines)
```

**Rules for AI team:**
1. **Only use text from Context Lake** — do not invent organizational policy.
2. **Refresh on each agent run** or cache ≤15 min — admins may reconfigure.
3. **If `roles` is empty**, proceed without SOM block (do not fail the agent).
4. **Do not call interview-setup APIs** for runtime — use `/context-lake` only.
5. **Do not expose raw admin interview text to sellers** unless product explicitly asks.

### AI team checklist

- [ ] Wire `fetch_som_context_lake()` in briefing `generate_briefing()` before LLM call
- [ ] Add SOM block to post-meeting summary system prompt
- [ ] Add SOM block to outreach `draft_email()` grounding
- [ ] Add SOM block to data-hygiene / deal-health scoring prompts (when LLM-backed)
- [ ] Log when Context Lake is empty vs populated (observability)
- [ ] Future: filter Context Lake by seller region/team when RBAC lands

---

## curl examples

```bash
# Load National Manager form
curl "http://localhost:8000/api/sales-operating-model/interview-setup?role=national_manager"

# Save after review
curl -X POST "http://localhost:8000/api/sales-operating-model/interview-setup/national_manager/save" \
  -H "Content-Type: application/json" \
  -H "X-User-Id: admin@lenovo.com" \
  -d '{"responses":[{"questionId":"11111111-1111-4111-8111-111111111101","text":"Growth with margin discipline."}]}'

# Intent cards
curl "http://localhost:8000/api/sales-operating-model/intent-cards"

# Context Lake (AI)
curl "http://localhost:8000/api/sales-operating-model/context-lake"
```

---

## Tests

```bash
pytest -q tests/test_som_interview_setup.py
```

---

## Message codes

| Code | When |
|------|------|
| `ERR_MSG_0021` | Save transaction failed — no partial commit |

*(Also used by Quarter Pulse — FE should show context-appropriate copy.)*

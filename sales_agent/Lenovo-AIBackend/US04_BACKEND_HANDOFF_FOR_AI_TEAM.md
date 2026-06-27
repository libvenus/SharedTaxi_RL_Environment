# US04 — Backend Handoff for the AI Team

**Story:** Data Hygiene, Validation & Intelligent Alerts
**Backend repo:** `Lenovo-AIBackend`
**Status:** ✅ Sprint 1A backend complete — endpoints + scan job + tests live
**Pairing:** Sanmay (backend) ↔ Namisha + AI/Bot team

This document is the **single source of truth** for what the AI team
needs to do to integrate against the data-hygiene queue. If something
here is wrong or unclear, raise a PR or ping Sanmay before building
around it.

---

## 1. The big picture

US04 has many scenarios; **only one is your job: transcript-signal alerts.**

The rest are handled by:
- A daily CLI scan job that runs deterministic detectors over D365
  active opportunities (4 detectors live in S1A; more in S1B).
- The FE inline validators (S1B) for save-time validation.
- The seller's manual resolve / dismiss actions in the To-Do List UI.

**Your one integration:** when the post-meeting summary NLP detects a
signal in the transcript that conflicts with D365 data, you POST a
data-task and we persist it to the seller's To-Do queue.

---

## 2. The endpoint you'll call

### `POST /api/data-tasks`

One generic endpoint. You pass `created_by_source: "transcript"`. We
handle idempotency, dismissal-suppression, and ordering. You don't have
to look anything up first.

**Request body** (all fields per the Pydantic schema in
`app/schema/data_task.py`):

```jsonc
{
  "owner_id":          "<UUID — the seller>",
  "entity_kind":       "opportunity",                 // or "account" / "contact"
  "entity_id":         "<UUID — the affected D365 record>",
  "task_kind":         "transcript_signal_close_date_different",
  "severity":          "high",                        // "high" / "medium" / "low"
  "confidence":        "high",                        // your model's confidence
  "field_name":        "close_date",                  // optional but please send when you have it
  "current_value":     "2026-06-30",                  // for the strikethrough display
  "suggested_value":   "2026-06-15",                  // your inferred value
  "evidence_ref":      "transcript_segment_id=ab12",  // see §4
  "evidence_text":     "Customer mentioned: \"We need it before June 15th.\"",
  "created_by_source": "transcript"
}
```

**Response (200)**:

```jsonc
{
  "task": { /* DataTaskOut record */ },
  "was_existing": false                               // false = newly created
                                                      // true  = idempotent / suppressed
}
```

`was_existing: true` is **success, not an error.** It means we already
have an open or dismissed task for the same `(entity_kind, entity_id,
task_kind)`. Don't retry, don't change the payload — just log "already
on the queue" and move on.

---

## 3. Canonical `task_kind` strings

The DB column is free-form TEXT — we WILL accept any string you POST.
But please use these constants so the FE can group + i18n consistently.

### Account-level transcript signals

| Signal in the transcript | `task_kind` |
|---|---|
| "We moved our HQ to Pune" | `transcript_signal_location_change` |
| "We have grown to about 2,000 employees now" | `transcript_signal_headcount_change` |
| "Ashok has left, Priya is now leading this" | `transcript_signal_new_decision_maker` |
| "We're being acquired by Tata Group" | `transcript_signal_acquisition` |
| "Our budget cycle resets in October, not March" | `transcript_signal_budget_cycle_change` |

### Opportunity-level transcript signals

| Signal in the transcript | `task_kind` |
|---|---|
| Different close date than D365 | `transcript_signal_close_date_different` |
| Different quantity / scope | `transcript_signal_quantity_different` |
| Budget figure differs from deal value | `transcript_signal_budget_different` |
| Requirement / scope change | `transcript_signal_requirement_change` |
| Competitor mentioned but not in D365 | `transcript_signal_unlogged_competitor` |

These constants are also exported from
`app/models/data_task.py` so backend tests can import them.

---

## 4. `evidence_ref` format

`evidence_ref` is a **stable handle** that lets the FE deep-link from
the To-Do task to the source. We don't parse it; we just store and
echo. Recommended format:

```
transcript_segment_id=<uuid>
```

…where `<uuid>` is the `tbl_meeting_transcript_segment.segment_id`
from US02. If you don't have a segment id (e.g. you're aggregating
across the whole meeting), use:

```
transcript_meeting_id=<uuid>
```

Either way, keep it parseable as a `key=value` pair so future deep-link
logic can just `evidence_ref.split('=')`.

---

## 5. `evidence_text` requirements (AC #3 grounding)

**REQUIRED.** Empty / whitespace-only is rejected (422 from Pydantic
+ DB CHECK constraint as a final guard).

Should be a **plain-language sentence** the seller can read in their
To-Do list and immediately understand. The signal extraction model has
the transcript context; please render that into something the seller
can act on.

✅ Good: `"Customer said 'we need delivery before June 15' but D365 close date is June 30."`

❌ Bad: `"close_date_mismatch"` (machine string — that's what `task_kind` is for)

❌ Bad: `"D365 mismatch detected"` (no grounding — the seller can't tell what to do)

Length cap: 2048 characters. Anything longer should go in your own
log; just summarise here.

---

## 6. Confidence levels

Set `confidence` to your model's calibrated bucket:

- `"high"` — you're very sure (verbatim quote, unambiguous semantics)
- `"medium"` — likely but worth seller verification
- `"low"` — a hint; the seller should weigh it heavily

**Effect on the FE:** GET `/api/data-tasks` orders by confidence DESC
NULLS LAST, then severity DESC, then created_at ASC. So `confidence:
"high"` tasks appear at the top of the seller's queue.

If you genuinely don't have confidence info, send `null`. NULL sorts
last — same bucket as deterministic detectors which also send NULL.

---

## 7. Severity vs confidence — what's the difference?

- **Severity** — how *urgent* is this issue? Drives the visual badge.
- **Confidence** — how *sure* are we about it? Drives the order.

Examples:

| Signal | Severity | Confidence | Why |
|---|---|---|---|
| Close date verbatim quoted in transcript | high | high | Urgent (timing) + verbatim (high cert) |
| Maybe-acquisition mentioned in passing | high | low | Urgent (M&A) + ambiguous quote |
| Budget cycle mentioned offhand | low | medium | Not urgent + reasonably clear |

---

## 8. What you do NOT need to send

- ❌ Don't POST the same task twice — the endpoint is idempotent. We
  handle de-dup automatically.
- ❌ Don't check first whether a task exists before POSTing. Just POST.
  We do the SELECT internally.
- ❌ Don't try to update D365 yourself. AC #3: "the system never
  updates any field automatically." The seller fixes the data; we
  record the resolution.
- ❌ Don't dismiss tasks programmatically. Dismissal is a seller action
  with a required note.

---

## 9. Error handling

| Status | Meaning | What to do |
|---|---|---|
| `200` | Created OR idempotent return (check `was_existing`) | Both are success — proceed |
| `422` | Schema validation — missing field, empty `evidence_text`, etc. | Fix and retry |
| `500` | Backend bug | Log + alert; don't retry storm |

We deliberately do NOT return `409` for the "already exists" case —
that's `200 + was_existing: true`. So your retry policy is simple:
treat all 5xx as transient, all 4xx as permanent (don't retry).

---

## 10. Reading back tasks (the FE is your audience, not you)

You probably won't call `GET /api/data-tasks` — that's the FE's
domain. But if you want to verify a POST landed correctly during
integration testing:

```bash
curl 'http://localhost:8001/api/data-tasks?ownerId=<seller>&kind=transcript_signal_close_date_different'
```

---

## 11. Sample integration flow

```python
import httpx

AIBACKEND_URL = "http://localhost:8001"

def emit_transcript_alert(
    *,
    owner_id: str,
    opportunity_id: str,
    task_kind: str,
    field_name: str | None,
    current_value: str | None,
    suggested_value: str | None,
    evidence_segment_id: str,
    evidence_text: str,
    confidence: str,         # "high" / "medium" / "low"
    severity: str = "medium" # "high" / "medium" / "low"
) -> None:
    """Post a single transcript-signal alert. Idempotent — safe to call
    again with the same args."""
    payload = {
        "owner_id": owner_id,
        "entity_kind": "opportunity",
        "entity_id": opportunity_id,
        "task_kind": task_kind,
        "severity": severity,
        "confidence": confidence,
        "field_name": field_name,
        "current_value": current_value,
        "suggested_value": suggested_value,
        "evidence_ref": f"transcript_segment_id={evidence_segment_id}",
        "evidence_text": evidence_text,
        "created_by_source": "transcript",
    }
    response = httpx.post(
        f"{AIBACKEND_URL}/api/data-tasks", json=payload, timeout=5.0
    )
    response.raise_for_status()
    body = response.json()
    if body["was_existing"]:
        log.info(
            "Task already on queue (entity=%s/%s kind=%s)",
            payload["entity_kind"], payload["entity_id"], payload["task_kind"],
        )
    else:
        log.info("Created new data-task %s", body["task"]["task_id"])
```

---

## 12. Open questions for you

| # | Question | Why we need it |
|---|---|---|
| Q1 | Will every signal include a `transcript_segment_id`, or only some? | Decides whether `evidence_ref` is mandatory in your pipeline |
| Q2 | Are confidence values calibrated against ground truth, or model-internal? | Affects whether the FE should show the H/M/L label literally or convert to a percentile |
| Q3 | What's the latency target post-meeting? Real-time vs end-of-day batch? | Decides whether we need a backend rate-limit / queue |
| Q4 | When the same signal appears in two consecutive meetings, do you re-POST? | Today the second POST is a no-op. Confirm that's the desired behaviour (vs bumping evidence_ref to the more recent one) |
| Q5 | Multiple signals in one transcript on the same `(entity, kind)` — do you aggregate or send N posts? | Today only the first lands; subsequent ones are idempotent no-ops. Aggregate into one richer evidence_text if multiple matter |

Reply on those before you start wiring; we may need to adjust the
backend if Q4 / Q5 want different semantics.

---

## 13. Where the backend code lives

| File | Purpose |
|---|---|
| `app/api/data_tasks.py` | The 6 routes (POST / GET list / GET / resolve / dismiss / scan) |
| `app/services/data_task_service.py` | CRUD + idempotency contract |
| `app/services/data_task_detectors.py` | The 4 deterministic detectors (D1–D4) |
| `app/jobs/scan_data_tasks.py` | Daily-scan CLI |
| `app/models/data_task.py` | ORM + canonical `task_kind` constants |
| `app/schema/data_task.py` | Pydantic request/response shapes |
| `sql/2026_06_us04_data_task.sql` | Idempotent schema migration |
| `tests/test_data_task_lifecycle.py` | 14 smoke tests |

---

## 14. Status

- [x] Schema + migration
- [x] ORM + Pydantic
- [x] Service layer + 4 detectors
- [x] 6 endpoints
- [x] Daily-scan CLI
- [x] 14 smoke tests
- [x] README + handoff doc
- [ ] AI team integration (you're here)
- [ ] DevOps cron schedule for `python -m app.jobs.scan_data_tasks`

Ping Sanmay when you have a question or hit something this doc doesn't
cover.

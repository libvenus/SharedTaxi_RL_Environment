# Meeting Intelligence Storage — Backend Approach

**Purpose:** capture the AI-generated meeting summaries (key points, next steps, transcript, sentiment, CRM suggestions) and surface them on the Meeting Prep page tabs (Overview / Recent Signals / Talking Points) and the Activity Timeline.

**Status:** design proposal — not yet implemented.
**Backend version at time of writing:** `0.13.2`
**Audience:** AI team, backend / FE engineers, PMs scoping the Meeting Prep story.

---

## 1. The problem

The AI team produces a structured JSON per meeting with seven distinct sections (header, key points, next steps, open questions, CRM-update suggestions, sentiment, transcript). Existing tables in the backend cover only a fraction of this:

| AI dump section | Existing table | Coverage |
|---|---|---|
| `meetingHeader` (title / date / time / platform / attendees) | `lvo_activity` | Partial — has subject/body/date but no platform or attendee list |
| `meetingSummary.keyPoints[]` | none | None |
| `meetingSummary.confirmedNextSteps[]` | `lvo_nextaction` | Structurally similar, but no provenance (no link back to AI source, no confidence, no transcript refs) |
| `meetingSummary.openQuestions[]` | none | None |
| `meetingSummary.crmUpdates.*` | scattered (`lvo_opportunitycontact`, `lvo_opportunitycompetitor`, `opportunity`) | These are *suggestions*, not facts — should not be auto-applied |
| `meetingSummary.sentiment` | none | None |
| `transcript.transcriptSegments[]` | none | None |

Normalizing the entire dump would require **7+ new tables**. The AI output schema is also still evolving — fixing it in normalized columns now would mean a migration every time the AI team adds a field.

---

## 2. Recommended approach — one JSONB table + explicit promotion

The standard pattern for AI-generated artifacts: **store the raw payload as JSONB for fidelity, keep CRM tables pristine, and let users explicitly "promote" actionable items into CRM tables when they're ready.**

```
                AI dump (verbatim JSONB)
                       │
                       ├── Render directly to UI
                       │   (Talking Points / Recent Signals / Overview tabs)
                       │
                       └── User clicks "Add to deal" on a specific item
                                │
                                ├─ confirmedNextSteps[]   → lvo_nextaction
                                ├─ crmUpdates.competitors → lvo_opportunitycompetitor
                                ├─ crmUpdates.decisionMakers → lvo_opportunitycontact (set isDecisionMaker)
                                └─ crmUpdates.dealAmount / closeDate / dealStage → PATCH /api/opportunities/{id}
```

### Why this shape

| Concern | How JSONB-first handles it |
|---|---|
| AI schema evolves | No migration needed when a new field appears — it's just another key in the JSONB blob |
| Read-pattern is "fetch one meeting" | A single-row JSONB read is fast; no joins required to render the UI |
| Future analytics ("all open questions across deals") | Postgres JSONB ops + GIN indexes handle this without re-engineering |
| Audit fidelity | The original AI output is preserved verbatim — confidence scores, transcript refs, ambiguity flags all stay intact |
| Trust / hallucination guard | CRM tables stay clean. The AI never silently writes to them. Every CRM mutation is a deliberate user action, traced through the existing `lvo_audit_log` |
| Reversibility | If the AI hallucinated a competitor, the JSONB is untouched; the user simply doesn't promote it |

---

## 3. Proposed schema — `lvo_meeting_intelligence`

One row per AI processing run. If the AI team re-processes a transcript with a better model, insert another row pointing to the same meeting — latest wins for display, full history is preserved.

| Column | Type | Purpose |
|---|---|---|
| `lvo_meetingintelligenceid` | `uuid` (PK) | Surrogate key |
| `lvo_activityid` | `uuid` (FK → `lvo_activity`) | The meeting this dump describes |
| `lvo_opportunityid` | `uuid` (nullable) | Denormalized from activity for fast filtering on the deal-detail page |
| `lvo_accountid` | `uuid` (nullable) | Denormalized from opportunity for fast filtering on the account page |
| `lvo_payload` | `jsonb` | The entire AI dump verbatim |
| `lvo_payloadversion` | `text` (nullable) | AI team's contract version, e.g. `"v2.0"` — lets us tolerate breaking schema changes |
| `lvo_modelname` | `text` (nullable) | e.g. `"gpt-4o-2024-11"` — for debugging / model A-B comparison |
| `lvo_modelconfidence` | `numeric(4,3)` (nullable) | Header-level confidence if AI provides one |
| `lvo_status` | `text` | One of: `ingested` / `reviewed` / `promoted` / `archived` |
| `lvo_processedat` | `timestamptz` | When the AI emitted the dump |
| `lvo_reviewedat` | `timestamptz` (nullable) | When a human reviewed it |
| `lvo_reviewedby` | `text` (nullable) | User ID of reviewer |
| `statecode` | `text` | Soft-delete: `Active` / `Inactive` |
| `lvo_createdat`, `lvo_updatedat` | `timestamptz` | Standard audit columns |

### Indexes

| Index | Reason |
|---|---|
| `(lvo_activityid)` | Fast lookup by meeting |
| `(lvo_opportunityid)` | All dumps for a deal |
| `(lvo_accountid)` | All dumps for an account |
| GIN on `lvo_payload->'meetingSummary'->'sentiment'` | Sentiment-based queries (optional, add later if analytics demands it) |

### Why one row per processing run, not per meeting

If we use `(lvo_activityid)` as a unique key, we can never re-process. If we allow many rows per meeting and pick the latest, we get free version history without losing anything. The "latest" read becomes:

```sql
SELECT * FROM lvo_meeting_intelligence
WHERE lvo_activityid = :meetingId
  AND statecode = 'Active'
ORDER BY lvo_processedat DESC
LIMIT 1;
```

---

## 4. Endpoints to expose

### Ingest

| Method | Path | Body | Purpose |
|---|---|---|---|
| `POST` | `/api/meetings/{meetingId}/intelligence` | The full AI dump | Store a fresh dump. Idempotent on `(meetingId, payloadVersion, processedAt)` — same dump twice should not create two rows. |

### Read

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/meetings/{meetingId}/intelligence` | Latest dump — drives all four tabs on the Meeting Prep page |
| `GET` | `/api/meetings/{meetingId}/intelligence/history` | All historical dumps for that meeting (optional, for audit / model A-B) |
| `GET` | `/api/opportunities/{id}/meeting-intelligence` | Latest dump across **all** meetings on a deal — drives an optional "Meeting Insights" tab on Deal Detail |
| `GET` | `/api/accounts/{id}/meeting-intelligence` | Same, scoped to an account |

### Lifecycle

| Method | Path | Purpose |
|---|---|---|
| `PATCH` | `/api/meetings/{meetingId}/intelligence/{intelId}` | Mark `reviewed` / attach `reviewedBy` |

### Promotion (the bridge to CRM)

| Method | Path | Body | Purpose |
|---|---|---|---|
| `POST` | `/api/meetings/{meetingId}/intelligence/{intelId}/promote` | `{ "nextSteps": ["NS001","NS003"], "competitors": ["..."], "dealUpdates": {...} }` | Push selected items into the appropriate normalized CRM tables. Returns `{ created: { nextActions: [...], competitors: [...] } }` so the FE can show a confirmation. |

The promotion endpoint **calls existing routers internally** (POST `/api/opportunities/{id}/next-actions`, POST `/api/opportunities/{id}/competitors`, PATCH `/api/opportunities/{id}`, etc.). All existing audit / validation / recalc logic runs unchanged.

---

## 5. Linkage — how a dump finds its deal/account

The AI team's JSON has **empty** `crmUpdates.opportunityName` and `crmUpdates.accountName`, so the dump itself doesn't tell us which deal it belongs to. The linkage comes from the **meeting ID** (`lvo_activityid`) passed in the URL:

```
POST /api/meetings/{meetingId}/intelligence
                     │
                     ▼
    lvo_activity row (the meeting)
                     │
                     ├─ lvo_opportunityid → denormalize into lvo_meeting_intelligence
                     │                          ▼
                     │                   opportunity.accountid → denormalize too
                     │
                     └─ if the activity has no opportunity link → store dump without one
                                                                  (still valid; just no CRM context)
```

**Implication for the AI team:** they need the `lvo_activityid` of the meeting before they can POST. This means:

1. The meeting is created in `lvo_activity` first (by whatever ingests calendar events)
2. The transcript is captured and tagged with that activity ID
3. After AI processing, the dump is POSTed to `/api/meetings/{activityId}/intelligence`

If step 1 is missing — for ad-hoc meetings the AI knows about but the CRM doesn't — we can add a `POST /api/meetings` endpoint that creates an `lvo_activity` row with `lvo_activitytype='meeting'` and a nullable opportunity link.

---

## 6. What lives in JSONB vs. promoted to CRM

Cheat-sheet for the FE / AI / CRM teams:

| AI dump field | Stays in JSONB only | Promotable to CRM | Promotion target table |
|---|---|---|---|
| `meetingHeader.*` | ✅ | — | — |
| `keyPoints[]` | ✅ | — | — |
| `confirmedNextSteps[]` | ✅ | ✅ | `lvo_nextaction` |
| `openQuestions[]` | ✅ | — | — |
| `crmUpdates.dealAmount / closeDate / dealStage / forecastCategory` | ✅ | ✅ | `opportunity` (PATCH) |
| `crmUpdates.competitors[]` | ✅ | ✅ | `lvo_opportunitycompetitor` |
| `crmUpdates.decisionMakers[]` / `champions[]` | ✅ | ✅ | `lvo_opportunitycontact` (toggle `lvo_isdecisionmaker`) |
| `crmUpdates.risks[]` | ✅ | ✅ | feeds into the existing risks endpoint |
| `crmUpdates.customerSentiment` | ✅ | — | (not promoted — display only) |
| `sentiment.*` | ✅ | — | — |
| `summaryQualityNotes[]` | ✅ | — | — |
| `transcript.transcriptSegments[]` | ✅ | — | — |

Every "✅ Promotable" item goes through the existing PATCH/POST endpoints, which already write to `lvo_audit_log`. After promotion, the item gets marked in the JSONB payload so the UI can show "Added to deal ✓" instead of an "Add to deal" button — a cheap `lvo_payload->'meetingSummary'->'confirmedNextSteps'->[i]->'promotedAt'` write.

---

## 7. Open questions (to settle before code)

| # | Question | Why it matters |
|---|---|---|
| 1 | Should the `POST /intelligence` endpoint be **idempotent** (same dump twice = no-op) or **versioned** (every POST creates a new row)? | Affects whether the AI team can safely retry, and whether re-processing creates an audit trail |
| 2 | Should AI-generated **decision-maker / champion** suggestions auto-flag the contact, or always require human approval? | Trust boundary — same AI hallucination concern |
| 3 | Do we expose **transcript text** in the FE, or only the AI-summarized bullets? Some orgs treat raw transcripts as PII / sensitive | Affects access control + storage scope |
| 4 | Is one transcript file per meeting always small enough (~25 segments) to keep in JSONB, or do we anticipate hour-long meetings with hundreds of segments? | Decides whether `transcriptSegments[]` stays in JSONB or splits into a separate `lvo_meeting_transcript` table |
| 5 | What's the **retention policy**? Keep all snapshots forever, or auto-archive after N days? | Affects table size / soft-delete strategy |
| 6 | Who **owns** ingestion authentication? AI team passes a service token, or we accept any internal call? | Security — we don't want random clients spamming AI dumps |

---

## 8. Effort estimate

| Scope | Effort | What you get |
|---|---|---|
| **v1 — minimum** | ~3 hrs | Migration + ORM + 2 endpoints (POST ingest, GET latest). FE can render every tab. No promotion. |
| **v1.5 — promotion** | +3 hrs | Promote endpoint + UI plumbing for "Add to deal" buttons |
| **v2 — history & analytics** | +2 hrs | History endpoints, GIN indexes, cross-meeting queries |

No external dependencies. No new Python packages. The whole thing is one new SQL migration + one new router file + a Pydantic schema for the payload.

---

## 9. TL;DR

- **One new table:** `lvo_meeting_intelligence`, JSONB-heavy.
- **Don't normalize** the AI dump — it's evolving, normalizing now will create churn.
- **Don't auto-write** AI suggestions into CRM — every CRM mutation must be a deliberate user click.
- **Linkage** is via `lvo_activityid` (the meeting). Existing tables tell us which deal/account it belongs to.
- **Effort for v1** (ingest + render) is ~3 hours. Promotion is another ~3 hours.

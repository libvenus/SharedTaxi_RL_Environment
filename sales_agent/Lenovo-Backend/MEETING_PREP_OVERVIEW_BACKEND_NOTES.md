# Meeting Prep — Overview Tab — Backend Mapping

**Page:** `Execute > Meeting Prep > Overview` (per-meeting view)
**Status:** Mapping only — **no code shipped yet for the new Meeting tile.**
**Backend version at time of writing:** `0.13.2`
**Audience:** Backend / FE engineers picking up the Meeting Prep story.

---

## 1. UI screenshot — what we're mapping

The Overview tab on the Meeting Prep page contains five visible blocks:

1. **Meeting tile** (top): time slot · meeting subject · linked account name · `Reschedule` and `Start Meeting` buttons.
2. **Tabs row**: `Overview` (active) / `Recent Signals` / `Talking Points`.
3. **Deal Summary card** (left, large): stage stepper with days-in-stage pill, priority pill (High / Medium / Low), three KPI tiles (Deal Value, Close Date, Forecast %), narrative paragraph, footer chips (Owner / Region / Lead Origin / Partner).
4. **Account Overview card** (right, top): a narrative paragraph describing the account (size, key contacts, open-deal count, revenue potential, executive relationship).
5. **Competitors card** (right, bottom): list of competitor names + reseller + a type pill (e.g. `Incumbent`).

---

## 2. UI block → backend table cheatsheet

| # | UI block | Source table(s) | Specific columns / derivation |
|---|---|---|---|
| 1 | Meeting tile header (`10:00 AM ▸ ThinkPad Fleet Proposal Review ▸ Infosys`) | `lvo_activity` (filtered to `lvo_activitytype IN ('meeting','call')`) joined to `opportunity` joined to `account` | `lvo_activity.lvo_activitydate` (time), `lvo_activity.lvo_subject` (title); via `lvo_opportunityid` → `opportunity.name` and `opportunity.accountid` → `account.name`. |
| 2 | `Reschedule` / `Start Meeting` buttons | `lvo_activity` (write path) | `Reschedule` = PATCH `lvo_activitydate`. `Start Meeting` = PATCH `statecode` (or app-level marker — see §5). |
| 3 | Stage stepper (`Qualify ✓ Develop ✓ Propose (32D) Execute Closed`) | `opportunity.stagename` + `opportunity.lvo_stageentrydate` | Stage = `stagename`, normalized via `app/normalizers.py`. The `(32D)` pill = `now() − lvo_stageentrydate`, in days. Already exposed as `daysInStage` on `OpportunityDetail` (v0.13.0). |
| 4 | `High` priority pill | `opportunity.lvo_priority` | Added in v0.13.0. Allowed values: `High` / `Medium` / `Low`. |
| 5 | Deal Value `$9.75M` | `opportunity.estimatedvalue` | Already exposed on every existing endpoint. |
| 6 | Close Date `12/06/26` | `opportunity.estimatedclosedate` | Already exposed. |
| 7 | Forecast `50% Best Case` | `opportunity.closeprobability` × 100 + `opportunity.lvo_forecastcategory` | "Best Case" is the literal string from `lvo_forecastcategory`. |
| 8 | Narrative paragraph (left card) | `opportunity.lvo_summary` (primary) **or** the synthesizer in `app/services/account_narrative.py` (fallback) | If `lvo_summary` is non-null/non-empty, surface it verbatim. Otherwise re-synthesize from account + opportunity facts the same way the Customer-Information tab does. |
| 9 | Footer chips (Owner / Region / Lead Origin / Partner) | `opportunity.owninguser` (→ `systemuser.fullname`) + `opportunity.lvo_country` / `lvo_geoid` + `opportunity.lvo_leadorigin` + `opportunity.lvo_partnerinvolved` | All four exposed in v0.13.0 Complete-Info tab. The `Partner: Company A` string requires a partner directory; today we only expose the boolean toggle. |
| 10 | Account Overview card (right) | `account` table + the narrative service | Same data the Customer-Information tab returns — `GET /api/accounts/{id}/customer-information`. |
| 11 | Competitors card (right) | `lvo_opportunitycompetitor` | `lvo_competitorname`, `lvo_resellingpartner`, `lvo_competitortype`. Already exposed: `GET /api/opportunities/{id}/competitors`. |

---

## 3. What's already built (no work needed)

For UI rows **3–11** the FE can render the entire Overview tab today by calling these existing endpoints:

| Endpoint | Hydrates rows |
|---|---|
| `GET /api/opportunities/{opportunityId}` | 3, 4, 5, 6, 7, 8 (via `summary`), 9 |
| `GET /api/accounts/{accountId}` | 10 (basic) |
| `GET /api/accounts/{accountId}/customer-information` | 10 (richer narrative) |
| `GET /api/opportunities/{opportunityId}/competitors` | 11 |

The v0.13.0 Complete-Information work already added the exact fields this Overview needs:
`lvo_summary`, `lvo_priority`, `lvo_leadorigin`, `lvo_partnerinvolved`, `daysInStage`, `ownerName`, plus the parent/child opportunity hierarchy.

---

## 4. What's net-new — the Meeting tile

This is the only gap. The Meeting Prep page is scoped per **meeting**, not per deal — so the URL is something like `/execute/meetings/{meetingId}`. We need a meeting-centric router because today `lvo_activity` is only queried via the deal-side timeline (`GET /api/opportunities/{id}/timeline`) and the activity preview embedded in the deal detail.

### Endpoints required

| Endpoint | Purpose | Underlying query |
|---|---|---|
| `GET /api/meetings/{meetingId}` | Hydrate the Meeting tile (time / subject / linked deal / linked account in one shot). | `lvo_activity` JOIN `opportunity` JOIN `account` — filter `lvo_activitytype IN ('meeting','call')` and `lvo_activityid = :id`. |
| `GET /api/meetings?date=YYYY-MM-DD` *(optional)* | Today's meeting list (for a sidebar / picker, if there is one). | `lvo_activity` filtered to `lvo_activitydate::date = :date` and the same activity types. |
| `PATCH /api/meetings/{meetingId}` | Reschedule (write `lvo_activitydate` and/or `lvo_subject`). | UPDATE one `lvo_activity` row + `lvo_audit_log` entry. |
| `POST /api/meetings/{meetingId}/start` *(optional)* | Mark meeting as started. Could equally be a `PATCH` with a started flag. | UPDATE statecode or a new `lvo_meetingstartedat` column — see §5. |

### No schema migration required

`lvo_activity` already exists with everything we need (`lvo_activitytype`, `lvo_subject`, `lvo_body`, `lvo_activitydate`, `lvo_opportunityid`, `statecode`). Adding `Start Meeting` semantics may eventually warrant a dedicated column (`lvo_meetingstartedat` / `lvo_meetingstatus`), but for v1 we can reuse `statecode` transitions.

---

## 5. Open questions for the team

The following items are **not blockers** for v1 but should be answered before we lock the contract:

1. **Activity-type vocabulary.** Do we model meetings as `lvo_activitytype = 'meeting'`, or is there already a distinct meeting entity in D365 we should be reading from instead? Need to confirm with the seed data.
2. **`Start Meeting` semantics.** Is this just a UI marker (FE keeps state), or does the backend persist a "started at" timestamp? If the latter, we'll add `lvo_meetingstartedat` (and probably `lvo_meetingendedat`) to `lvo_activity` in a migration.
3. **Partner directory.** The footer chip shows `Partner: Company A` (a name). Today we only have a `lvo_partnerinvolved` boolean. To render the name we need either (a) a partner-name column added to `opportunity`, or (b) a partner directory table to join against.
4. **Talking Points tab.** Out of scope here, but it'll likely be a deterministic synthesis service over recent activities + risks + competitor moves — no new tables needed.

---

## 6. Recommended path

**v1 — minimum viable Meeting Prep Overview**

Three small pieces of backend work:

1. New `app/routers/meetings.py` with the single `GET /api/meetings/{meetingId}` endpoint. Returns a composite payload:
   ```json
   {
     "meeting": { "id": "...", "subject": "...", "scheduledFor": "...", "type": "meeting", "status": "scheduled" },
     "opportunity": { /* OpportunityDetail-shaped payload */ },
     "account":     { /* AccountDetail-shaped payload */ },
     "competitors": [ /* CompetitorRef[] — already-existing schema */ ]
   }
   ```
2. `PATCH /api/meetings/{meetingId}` for Reschedule (writes `lvo_activitydate` only — minimal scope).
3. Update `app/main.py` description + `API_CONTRACT.md`.

**Estimated effort:** ~2 hours including tests. No SQL migration. No schema changes.

**v1.5 — once partner directory / meeting-status questions are answered**

- Optional `lvo_meetingstartedat` column + `POST /api/meetings/{id}/start`.
- Partner-name resolution for the footer chip.
- Talking Points tab once the synthesis rules are agreed.

---

## 7. TL;DR for the FE engineer

> You can build the entire **Deal Summary**, **Account Overview**, and **Competitors** sections of the Overview tab today — those four endpoints already return everything you need (see §3). The only piece you can't build yet is the **Meeting tile at the very top** because the meeting-centric `GET /api/meetings/{meetingId}` endpoint doesn't exist. That's a one-router add (~2 hours). Ping the backend team to greenlight v1 above and we'll ship it.

---

## Appendix — column reference (most-relevant fields only)

### `opportunity`
| Column | Used for |
|---|---|
| `opportunityid` | Path param. |
| `name` | Meeting tile (deal name). |
| `accountid` | Join to `account`. |
| `stagename` | Stage stepper. |
| `estimatedvalue` | Deal Value tile. |
| `estimatedclosedate` | Close Date tile. |
| `closeprobability` | Forecast `%`. |
| `lvo_forecastcategory` | Forecast category label. |
| `lvo_priority` | Priority pill (v0.13.0). |
| `lvo_summary` | Narrative paragraph (v0.13.0). |
| `lvo_leadorigin` | Footer chip. |
| `lvo_partnerinvolved` | Footer chip (boolean today). |
| `lvo_stageentrydate` | Days-in-stage pill. |
| `lvo_country` / `lvo_geoid` | Region chip. |
| `owninguser` | Owner chip (resolved to `systemuser.fullname`). |

### `account`
| Column | Used for |
|---|---|
| `accountid` | Join target. |
| `name` | Meeting tile (linked account name). |
| `lvo_segment`, `numberofemployees`, `revenue`, `lvo_territory`, `lvo_countryid`, `industrycode` | Account Overview narrative. |

### `lvo_activity`
| Column | Used for |
|---|---|
| `lvo_activityid` | Path param for new meeting endpoints. |
| `lvo_activitytype` | Filter to `'meeting'` / `'call'`. |
| `lvo_subject` | Meeting tile title. |
| `lvo_activitydate` | Meeting tile time + Reschedule writes. |
| `lvo_opportunityid` | Join to `opportunity`. |
| `statecode` | Active vs. cancelled. |

### `lvo_opportunitycompetitor`
| Column | Used for |
|---|---|
| `lvo_competitorname` | Competitor row name. |
| `lvo_competitortype` | "Incumbent" pill. |
| `lvo_resellingpartner` | Reseller string ("Acme Resellers"). |
| `lvo_opportunityid` | Filter to current deal. |

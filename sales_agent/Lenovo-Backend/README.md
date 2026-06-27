# Lenovo D365 Sales — Backend

FastAPI backend that powers the Lenovo Opportunities + Accounts dashboards. Implements:

- **Opportunities grid + KPI strip** (Screen 1)
- **Deal Update** user story (PATCH deal, manage competitors and next actions, full audit log)
- **Deal Detailed View** user story (full deal payload, contacts, paginated timeline,
  configurable Deal Health calculator, automatic risk derivation, soft delete)
- **View Account** user story (paginated accounts grid, multi-select filters,
  dynamic value-range slider, CSV export, derived `accountType` /
  `accountStatus` / `lastInteractionDate`, account-level contact CRUD)
- **View Customer Information** user story (sectioned read-only profile —
  Basic Info, Billing & Shipping Address, Identity & Legal, Commercial
  Terms, Territory & Ownership)
- **Per-account Opportunities tab** (filterable grid scoped to one
  account, activity-timeline dots with same-day bucketing, risk-count
  badge from `lvo_dealrisk`, CSV export)
- **Accounts-page KPI strip** (Total Accounts / Account Value (ACV) /
  Active Accounts / Accounts at Risk cards with `vs. Last Week / Past
  Month / Last Quarter` toggle and snapshot-driven trend deltas)

> Frontend lives in the sibling `Lenevo-Frontend` folder.

---

## Quick start

```bash
# 1. Create venv + install
python -m venv .venv
.venv\Scripts\activate                       # Windows
# source .venv/bin/activate                  # macOS / Linux
pip install -r requirements.txt

# 2. Configure .env (copy from .env.example), pointing at Postgres
copy .env.example .env

# 3. Apply the database migrations (in order, against the same DB)
psql -f sql/2026_06_create_lvo_activity.sql
psql -f sql/2026_06_add_dealhealth.sql
psql -f sql/2026_06_bulk_seed_50_opportunities.sql
psql -f sql/2026_06_create_next_actions_audit.sql
# --- Deal Detailed View additions ---
psql -f sql/2026_06_deal_detail_schema.sql
psql -f sql/2026_06_account_extension.sql
psql -f sql/2026_06_create_opportunity_contact.sql
psql -f sql/2026_06_create_dealrisk.sql
psql -f sql/2026_06_create_dealhealth_config.sql
# --- View Account additions ---
psql -f sql/2026_06_account_view_schema.sql
psql -f sql/2026_06_create_account_contact.sql
# --- KPI trend snapshot table ---
psql -f sql/2026_06_create_opportunity_snapshot.sql
# --- Manage Contacts: phone-column fallback (no-op when telephone1/mobilephone exist) ---
psql -f sql/2026_06_contact_extension.sql
# --- Manage Contacts: widen audit-log CHECK to allow contact entity types ---
#     Required for DBs that have already applied 2026_06_create_next_actions_audit.sql.
#     A no-op for fresh DBs (the source migration already ships the wider whitelist).
psql -f sql/2026_06_widen_audit_log_entitytypes.sql
# --- Customer Information tab: idempotent ADD COLUMN IF NOT EXISTS for the
#     Lenovo-custom columns (sub-segment, GTM segment, legal entity, sales
#     org, etc.). Standard D365 columns (address1_*, websiteurl, telephone1)
#     are NOT touched — they are part of the base schema.
psql -f sql/2026_06_account_customer_info_schema.sql
# --- Accounts-page KPI strip: per-day bucket aggregates for the four cards ---
psql -f sql/2026_06_create_account_snapshot.sql
# --- Most-Likely KPI card: widen the snapshot bucket whitelist (no-op on fresh DBs) ---
psql -f sql/2026_06_widen_kpi_snapshot_buckets.sql
# --- Opportunity Complete-Information tab: editable form fields + audit columns ---
#     Adds lvo_summary, lvo_priority, lvo_leadorigin, lvo_partnerinvolved,
#     lvo_parentopportunityid (self-FK), and createdby/modifiedon/modifiedby.
#     All ADD COLUMN IF NOT EXISTS — safe to re-run.
psql -f sql/2026_06_opportunity_complete_info_schema.sql

# 4. Run the API
uvicorn app.main:app --reload --port 8000

# 5. (Optional) Populate health + account-status for every row.
#    Order matters — account-status reads opportunity.lvo_dealhealthscore.
python -m app.jobs.recalc_health
python -m app.jobs.recalc_accounts

# 6. (Optional) Backfill KPI snapshots so the kpi-summary endpoints show
#    non-null trends out of the box. After this, schedule the no-arg
#    forms nightly.
python -m app.jobs.snapshot_kpis           --backfill   # opportunity buckets
python -m app.jobs.snapshot_account_kpis   --backfill   # account buckets
# nightly cron / scheduled task:
# python -m app.jobs.snapshot_kpis
# python -m app.jobs.snapshot_account_kpis
```

Open <http://localhost:8000/docs> for the live Swagger UI, or read
[API_CONTRACT.md](./API_CONTRACT.md) for the same surface in markdown.

---

## Project layout

```
app/
├── main.py                    FastAPI entrypoint, router wiring
├── config.py                  Settings (DATABASE_URL / CORS / etc.)
├── database.py                SQLAlchemy engine + session factory
├── models.py                  ORM tables
├── schemas.py                 Pydantic request/response models (camelCase)
├── normalizers.py             Vocabulary mapping (stage / sale-motion / slug)
├── filters_query.py           Shared opportunity filter helpers
├── routers/
│   ├── filters.py             /api/filters/*
│   ├── opportunities.py       /api/opportunities (read: list + KPI + competitors)
│   ├── deals_write.py         PATCH/POST/PATCH/DELETE for deal/competitors/next-actions
│   ├── deals_read.py          GET /api/opportunities/{id} + timeline/contacts/health/risks
│   ├── contacts.py            POST/PATCH/DELETE /api/opportunities/{id}/contacts
│   ├── accounts.py            /api/accounts (list, detail, filters, value-range, export, opps-by-account + filters/timeline/risk-count + CSV, customer-information, recompute-status)
│   └── account_contacts.py    /api/accounts/{id}/contacts CRUD (account-level roster)
├── services/
│   ├── deal_health.py         Pure scoring functions (5 components)
│   ├── deal_risks.py          Pure risk-rule evaluators (13 rules)
│   ├── deal_recalc.py         Orchestrator: load → score → persist (cascades to account)
│   ├── account_status.py      Pure account-type / account-status derivation
│   ├── account_recalc.py      Orchestrator for account derived fields
│   ├── kpi_snapshots.py       Opportunity KPI bucket aggregates + period-over-period trend math
│   ├── account_kpi_snapshots.py  Account KPI bucket aggregates (total/acv/active/at_risk)
│   ├── contact_phone.py       Runtime resolver for contact.telephone1 / mobilephone / lvo_phone
│   ├── contact_validation.py  Pure email/phone validators + delete-eligibility (ERR_MSG_0008/9/13)
│   └── account_columns.py     Runtime introspection of `account` columns for the Customer Information tab
└── jobs/
    ├── recalc_health.py            CLI: `python -m app.jobs.recalc_health`
    ├── recalc_accounts.py          CLI: `python -m app.jobs.recalc_accounts`
    ├── snapshot_kpis.py            CLI: `python -m app.jobs.snapshot_kpis [--backfill]`
    └── snapshot_account_kpis.py    CLI: `python -m app.jobs.snapshot_account_kpis [--backfill]`

sql/                            Idempotent migrations — run in numeric order
tests/                          pytest unit tests (calculators)
```

---

## Deal Health at a glance

The score is `0–100` with bands `RED / YELLOW / GREEN`. Five weighted components:

| Component             | Weight | Source signal                        |
|-----------------------|-------:|--------------------------------------|
| Stage Progress        |   25 % | stage position + actual-vs-expected days in stage |
| Activity Freshness    |   25 % | days since last activity vs tempo-class cadence |
| Stakeholder           |   20 % | active stakeholder coverage × threading factor |
| Close-Date Confidence |   20 % | elapsed time vs cumulative expected stage progress |
| Risk Adjustment       |   10 % | `100 − 20 × len(risks)`              |

Every threshold is configurable via `lvo_dealhealthconfig.lvo_settings` —
the seed migration loads the user-story defaults. The calculator falls
back to those defaults if the table is missing or malformed.

13 risks are derived from the same data set; full list and triggers are
documented at the top of `app/services/deal_risks.py`.

---

## Tests

Unit tests live in `tests/` and cover every band/branch of the
calculators. They have **no DB dependency** — pure functions only.

```bash
pip install pytest
pytest tests/ -v
```

The HTTP routes are intentionally not covered by tests yet; add an
integration test using `fastapi.testclient.TestClient` against a
SQLite-or-Postgres fixture when the test suite grows.

---

## Account view at a glance

The Accounts grid shows three derived columns that are persisted on the
`account` row and refreshed automatically:

| Column                     | Source                                                                      |
|----------------------------|-----------------------------------------------------------------------------|
| `accountType`              | `Customer` once any opportunity reaches Closed Won; otherwise `Prospect`.   |
| `accountStatus`            | `Inactive` if `statecode='Inactive'` or idle > 180d. `At-Risk` if any open deal has `dealHealth < 50`. Else `Active`. |
| `lastInteractionDate`      | `MAX(lvo_activity.lvo_activitydate)` joined through `opportunity`.          |

All three knobs (`at_risk_health_threshold`, `inactive_idle_days`) live
under `lvo_dealhealthconfig.lvo_settings.account_status`.

Filters apply across `accountType`, `accountStatus`, `segment`, `region`
(territory/businessGroup/country), `industry`, and a dynamic
`valueMin/valueMax` slider against `totalAccountValue`.

CSV export at `GET /api/accounts/export` honours every active filter so the
download matches what's on the screen.

---

## Manage Contacts (account roster)

The `/api/accounts/{id}/contacts` family of endpoints implements the
**Manage Contacts Linked to an Account** user story. Two flows on the
single `POST` endpoint cover both UI cases:

* **Create-and-link** (UI form) — body has `firstName`, `lastName`,
  optional `email`, `phone`, `jobTitle`, `role`, `isPrimary`. A new row
  is inserted into `contact` and immediately linked via
  `lvo_accountcontact`.
* **Attach existing** — body has `contactId` only. Used for bulk-import
  flows or programmatic linking.

Delete is gated by two pre-flight checks the FE can mirror without an
extra round-trip (`canDelete` / `deleteRestrictionCode` are embedded in
every roster row):

| Block code | Trigger |
|---|---|
| `ERR_MSG_0008` | The link is the **primary** contact on the account. |
| `ERR_MSG_0009` | The contact is referenced by ≥ 1 **active opportunity** (`opportunity.statecode='Open'` AND `lvo_opportunitycontact.statecode='Active'`). The 409 detail includes `affectedDealIds`. |

The contact row is **never deleted** — only the link's `statecode` flips
to `Inactive`, matching the acceptance criterion "removing a contact
from an account does not delete the contact from the system".

### Phone-column compatibility

D365 dumps disagree on which phone column is shipped (`telephone1`,
`mobilephone`, both, or neither). `app/services/contact_phone.py`
introspects `information_schema.columns` once per process and picks the
first available column in priority order
`telephone1 → mobilephone → lvo_phone`. The fallback `lvo_phone` is
added by `sql/2026_06_contact_extension.sql` only when neither D365
column exists, so the migration is a no-op on standard schemas.

---

## Per-account Opportunities tab

`GET /api/accounts/{accountId}/opportunities` powers the **Opportunities**
tab on the Account Detail screen. It returns the same `OpportunityListItem`
shape the main grid uses so the FE re-uses its row component, but adds:

| Concern | How it's served |
|---|---|
| **Filter toolbar** (Search, Regions, Industries, Stage, Products) | Query params forwarded into `apply_opportunity_filters` — the same helper the main grid uses. Vocabulary comes from the global `/api/filters/*` endpoints. |
| **Activity timeline dots** | One batched `SELECT * FROM lvo_activity WHERE lvo_opportunityid = ANY(:ids) AND lvo_activitydate >= now() - interval :timelineDays days`. Python groups by calendar day; days with ≥ 2 events collapse into a single `type='multiple'` marker carrying `groupedCount=N`, which renders as the numbered circle in the legend. Capped at 30 markers per row. |
| **`⚠ N` Risk badge** | `riskCount` field on each row, fed by a `LEFT JOIN (SELECT lvo_opportunityid, COUNT(*) FROM lvo_dealrisk WHERE statecode='Active' GROUP BY 1)` subquery. `null` when the `lvo_dealrisk` table is missing on this dump. |
| **Competitors column** | Existing batched preview (top 3 names + count). |
| **Download icon** | `GET /api/accounts/{accountId}/opportunities/export` — streamed CSV honouring every filter. |

`timelineDays` defaults to **90** (matching the UI tick marks at 90/60/30/0)
but accepts any value 7–365 if the FE wants to extend the strip.

No new migrations are required — the data already lives in `lvo_activity`
and `lvo_dealrisk`.

---

## Customer Information tab

`GET /api/accounts/{accountId}/customer-information` returns a
sectioned read-only payload for the **Customer Information** tab on
the Account Detail screen. Six cards, every field optional:

| Card | Source columns |
|------|----------------|
| `basicInformation` | `lvo_accounttype`, `lvo_segment`, `lvo_subsegment`, `industrycode`, `lvo_gtmsegment`, `revenue`, `numberofemployees`, `lvo_sellerknownas` |
| `billingAddress` | `address1_line1` … `address1_country` (standard D365) |
| `shippingAddress` | `address2_line1` … `address2_country` (standard D365) |
| `identityAndLegal` | `lvo_legalnamelocal`, `lvo_locallanguage`, `lvo_alias`, `lvo_taxvatnumber`, `lvo_legalentity`, `telephone1`, `websiteurl`, `lvo_linkedinurl` |
| `commercialTerms` | `lvo_defaultcurrency`, `paymenttermscode` (label-resolved), `defaultpricelevelid` (name-resolved when `pricelevel.name` is queryable), `lvo_dealsignconfig` |
| `territoryAndOwnership` | Region precedence, `lvo_salesterritory` / `territoryid` fallback, `lvo_futureterritory`, `lvo_salesorg`, `lvo_territorymovereason`, `lvo_geographicunit`, `lvo_salesoffice`, owner names from `systemuser.fullname` |

`app/services/account_columns.py` introspects the live `account` table
once per engine and the router defers any column missing from the
deployed schema, so a partially-migrated DB still serves the page —
the missing fields simply read as `null`. The response includes a
`notes` array that surfaces a diagnostic when the
`sql/2026_06_account_customer_info_schema.sql` migration hasn't been
applied yet.

This endpoint is **read-only** in v1. A future user story will add
`PATCH /api/accounts/{accountId}/customer-information` (with
`entityType='account'` audit-log entries) for the editable form fields
shown in the UI mockup.

---

## How background recalc works

Every PATCH/POST/DELETE on a deal-related route enqueues a deal-health
recalculation via FastAPI's `BackgroundTasks`. The recalc opens its own
SQLAlchemy session (the request-scoped session is already closed by
then), so writes stay snappy.

A successful deal recalc also enqueues an **account recalc** for the
owning account, so `accountStatus` reflects deal-health changes the next
time the page loads.

* For an immediate / synchronous deal recalc, call
  `POST /api/opportunities/{id}/health/recalculate`.
* For an immediate / synchronous account recalc, call
  `POST /api/accounts/{id}/recompute-status`.
* The cron-friendly batch refreshes are
  `python -m app.jobs.recalc_health` followed by
  `python -m app.jobs.recalc_accounts`.

### KPI trends (period-over-period)

`/api/opportunities/kpi-summary` returns a `trend` block on each card —
`{direction, deltaValue, deltaCount}` — by comparing the live aggregate
against the most-recent row in `lvo_opportunitysnapshot` whose
`lvo_snapshotdate ≤ today − N days` (`N = 7 / 30 / 90` for `last_week /
past_month / last_quarter`).

* `python -m app.jobs.snapshot_kpis` — run nightly to record today's
  six bucket totals (UPSERT, idempotent).
* `python -m app.jobs.snapshot_kpis --backfill` — one-time bootstrap
  that synthesises history from `opportunity.lvo_createdat` so the
  KPI strip shows non-null trends right after a fresh deploy.
* Trend is suppressed (`null`) when the request includes any filter —
  v1 only stores **global** snapshots; per-dimension snapshots are a
  planned follow-up.

---

## Contributing

* Follow the snake_case → camelCase pattern set up by `APIModel` in
  `app/schemas.py` — every public field is camelCase in JSON.
* Add a SQL migration for any schema change, prefixed `<YYYY_MM>_…`.
* Keep service modules pure; route handlers stay thin.

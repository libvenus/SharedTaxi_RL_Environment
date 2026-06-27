"""FastAPI entry point.

Run locally:
    uvicorn app.main:app --reload --port 9100

Then visit http://localhost:9100/docs for the interactive Swagger UI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.middleware.audit_read_logging import AuditReadLoggingMiddleware
from app.routers import (
    account_contacts,
    accounts,
    audit_compliance,
    contacts,
    contacts_resolver,
    contacts_list,
    deals_read,
    filters,
    meetings_resolver,
    notifications,
    opportunities,
    quarter_pulse,
    sales_operating_model,
    tasks,
    briefing,
)
from app.routers.deals_write import router as deals_write_router

settings = get_settings()

app = FastAPI(
    title="Lenovo Opportunities API",
    description=(
        "Endpoints backing the Opportunities + Accounts dashboards. "
        "Implements APIs #1, #2, #5, #8, #9, #10, #11 and #13 from the contract, "
        "the Deal Update user story, the Deal Detailed View user story "
        "(full deal payload, contacts, timeline, deal-health calculator + risks), "
        "the View Account user story (paginated grid, multi-select filters, "
        "value-range slider, CSV export, account-status / account-type "
        "derivation, account-level contact CRUD), the View Customer "
        "Information tab (sectioned read-only profile across Basic Info, "
        "Billing/Shipping Address, Identity & Legal, Commercial Terms and "
        "Territory & Ownership), the per-account Opportunities tab "
        "(filtered grid, activity-timeline dots with same-day bucketing, "
        "risk-count badge, CSV export), the Accounts-page KPI strip "
        "(Total / Account Value / Active / At-Risk cards with snapshot-driven "
        "period-over-period trend deltas), a `bucket=` drill-down "
        "parameter on /api/accounts and /api/accounts/export so clicking a "
        "KPI card filters the grid to that card's predicate, and a new "
        "`most_likely` KPI card on the Opportunities strip backed by "
        "`lvo_forecastcategory='Most Likely'` (the existing `pipeline` "
        "bucket is relabelled 'Identified' on the FE — predicate "
        "unchanged), the v0.12.1 forward-compatible accountKpi + "
        "dailyKpiTrend columns on the Accounts grid (today mirrors TAV / "
        "always 0.0 placeholder until the per-account daily-snapshot job "
        "lands), the v0.13.0 Opportunity Complete-Information tab — "
        "full read / update of the Deal Summary form fields (summary, "
        "priority, lead origin, partner-involved toggle, parent/child "
        "opportunity hierarchy with cycle prevention, days-in-stage, "
        "owner display name, and createdby/modifiedon/modifiedby audit "
        "columns) plus GET /api/opportunities/search powering the "
        "Parent-Opportunity typeahead picker, and the v0.13.1 "
        "Opportunity > Contacts tab additions — firstName / lastName / "
        "phone surfaced on every contact-link payload (phone resolved "
        "at runtime against telephone1 / mobilephone / lvo_phone) plus "
        "an editable `phone` field on the contact-link PATCH that "
        "writes back to the underlying contact row, and v0.13.2 "
        "case-insensitive Account-ID lookups (every endpoint that "
        "takes an accountId from the URL — _ensure_account, "
        "Customer-Information loader, deal-detail account-summary "
        "panel and account-recalc loader — now UPPER-normalises both "
        "sides of the comparison so uppercase URLs no longer 404), "
        "and the v0.14.0 Sprint-1A meeting → CRM resolver "
        "(POST /api/meetings/resolve-opportunity) consumed by the "
        "Note-Taking Agent (Lenovo-AIBackend) before the bot schedules "
        "itself to join a Teams meeting — matches attendee emails "
        "against contact.emailaddress1, walks lvo_opportunitycontact "
        "to active deals only, and tie-breaks on subject-keyword "
        "overlap with the deal name, plus the v0.15.0 Sprint-1A US02 "
        "contact-by-email batch resolver "
        "(POST /api/contacts/resolve-by-emails) called by the "
        "Note-Taking Agent once at meeting start to enrich attendees "
        "with name / jobTitle / accountId / role for transcript "
        "speaker tagging — picks the most senior opportunity-contact "
        "link (decision-maker first, then alphabetic by role) and "
        "returns NULL fields for unknown emails so the bot can "
        "render them as 'Unknown Attendee', and v0.16.0 Sprint-2 "
        "US 1.2 Quarter Pulse (GET /api/quarter-pulse) — seller-scoped "
        "quota attainment, pipeline coverage ratio, and days remaining "
        "in the fiscal quarter for the Home dashboard card, plus "
        "PUT /api/quarter-pulse/quota for Phase-1 manual quota entry "
        "when D365 has no goal configured, and v0.17.0 Sprint-2 "
        "US 1.3 Task Pending badge (GET /api/tasks/pending-summary) — "
        "seller-scoped open next-action count and overdue signal for "
        "the Home dashboard header, and v0.18.0 Sprint-2 Pre-Meeting "
        "Briefing facts (GET /api/briefing/context) — seller-scoped D365 "
        "account/deal summaries, competitor intel, and traceable signals "
        "for AI briefing card generation, and v0.19.0 contact search by name "
        "(GET /api/contacts/search) — fuzzy first-name lookup with optional "
        "account-name hint for AI chat flows like scheduling a call, and "
        "v0.20.0 Sprint-2 US 3.2.1 Sales Operating Model Interview-First "
        "Setup (GET/PUT/POST /api/sales-operating-model/interview-setup, "
        "GET /api/sales-operating-model/interview-intent-cards, "
        "GET /api/sales-operating-model/context-lake) for admin policy "
        "capture and AI Context Lake consumption, and v0.21.0 Sprint-2 "
        "US 3.2.2 Organizational Intent Setup (GET/PUT "
        "/api/sales-operating-model/organizational-intent-cards) for "
        "Outcome/Motion/Focus/Behavioral/Constraint/Trade-off cards, and v0.22.0 "
        "US 3.2.3 Delete Intent Card (DELETE "
        "/api/sales-operating-model/organizational-intent-cards/{type}), and v0.23.0 "
        "US 3.4.1 Timeline Classification & Canonical Sales Clock (GET/PUT "
        "/api/sales-operating-model/timeline-classification-cards, Context Lake v3)."
    ),
    version="0.23.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.audit_read_logging_enabled:
    app.add_middleware(AuditReadLoggingMiddleware)

app.include_router(filters.router)
app.include_router(opportunities.router)
app.include_router(deals_write_router)
app.include_router(deals_read.router)
app.include_router(contacts.router)
app.include_router(accounts.router)
app.include_router(account_contacts.router)
app.include_router(meetings_resolver.router)
app.include_router(contacts_resolver.router)
app.include_router(contacts_list.router)
app.include_router(notifications.notifications_router)
app.include_router(notifications.timeline_router)
app.include_router(quarter_pulse.router)
app.include_router(tasks.router)
app.include_router(briefing.router)
app.include_router(sales_operating_model.router)
app.include_router(audit_compliance.router)


@app.get("/health", tags=["meta"], summary="Liveness probe")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", tags=["meta"], summary="Root — quick endpoint index")
def root() -> dict[str, object]:
    return {
        "name": "Lenovo Opportunities API",
        "version": app.version,
        "docs": "/docs",
        "endpoints": [
            "GET    /api/opportunities",
            "GET    /api/opportunities/kpi-summary",
            "GET    /api/opportunities/search",
            "GET    /api/opportunities/{opportunity_id}",
            "PATCH  /api/opportunities/{opportunity_id}",
            "GET    /api/opportunities/{opportunity_id}/competitors",
            "POST   /api/opportunities/{opportunity_id}/competitors",
            "PATCH  /api/opportunities/{opportunity_id}/competitors/{competitor_id}",
            "DELETE /api/opportunities/{opportunity_id}/competitors/{competitor_id}",
            "GET    /api/opportunities/{opportunity_id}/next-actions",
            "POST   /api/opportunities/{opportunity_id}/next-actions",
            "PATCH  /api/opportunities/{opportunity_id}/next-actions/{action_id}",
            "GET    /api/opportunities/{opportunity_id}/sale-motion",
            "DELETE /api/opportunities/{opportunity_id}",
            "GET    /api/opportunities/{opportunity_id}/timeline",
            "GET    /api/opportunities/{opportunity_id}/contacts",
            "POST   /api/opportunities/{opportunity_id}/contacts",
            "PATCH  /api/opportunities/{opportunity_id}/contacts/{contact_link_id}",
            "DELETE /api/opportunities/{opportunity_id}/contacts/{contact_link_id}",
            "GET    /api/opportunities/{opportunity_id}/health",
            "POST   /api/opportunities/{opportunity_id}/health/recalculate",
            "GET    /api/opportunities/{opportunity_id}/risks",
            "GET    /api/accounts",
            "GET    /api/accounts/filters",
            "GET    /api/accounts/value-range",
            "GET    /api/accounts/kpi-summary",
            "GET    /api/accounts/export",
            "GET    /api/accounts/{account_id}",
            "GET    /api/accounts/{account_id}/customer-information",
            "GET    /api/accounts/{account_id}/opportunities",
            "GET    /api/accounts/{account_id}/opportunities/export",
            "POST   /api/accounts/{account_id}/recompute-status",
            "GET    /api/accounts/{account_id}/contacts",
            "POST   /api/accounts/{account_id}/contacts",
            "PATCH  /api/accounts/{account_id}/contacts/{contact_link_id}",
            "DELETE /api/accounts/{account_id}/contacts/{contact_link_id}",
            "GET    /api/accounts/{account_id}/contacts/{contact_link_id}/delete-eligibility",
            "GET    /api/filters/regions", 
            "GET    /api/filters/industries",
            "GET    /api/filters/stages",
            "GET    /api/filters/products",
            "POST   /api/meetings/resolve-opportunity",
            "GET    /api/contacts",
            "GET    /api/contacts/search",
            "POST   /api/contacts/resolve-by-emails",
            "GET    /api/notifications",
            "PATCH  /api/notifications/{notification_id}/read",
            "GET    /api/activity-timeline",
            "GET    /api/quarter-pulse",
            "PUT    /api/quarter-pulse/quota",
            "GET    /api/tasks/pending-summary",
            "GET    /api/briefing/context",
            "GET    /api/sales-operating-model/interview-setup",
            "PUT    /api/sales-operating-model/interview-setup/{role}/draft",
            "POST   /api/sales-operating-model/interview-setup/{role}/save",
            "GET    /api/sales-operating-model/interview-intent-cards",
            "GET    /api/sales-operating-model/intent-cards",
            "GET    /api/sales-operating-model/organizational-intent-cards",
            "PUT    /api/sales-operating-model/organizational-intent-cards",
            "GET    /api/sales-operating-model/organizational-intent-cards/{intent_type}",
            "PUT    /api/sales-operating-model/organizational-intent-cards/{intent_type}",
            "DELETE /api/sales-operating-model/organizational-intent-cards/{intent_type}/fields/{field_key}",
            "POST   /api/sales-operating-model/organizational-intent-cards/{intent_type}/metrics",
            "PATCH  /api/sales-operating-model/organizational-intent-cards/{intent_type}/metrics/{metric_id}",
            "DELETE /api/sales-operating-model/organizational-intent-cards/{intent_type}/metrics/{metric_id}",
            "DELETE /api/sales-operating-model/organizational-intent-cards/{intent_type}",
            "GET    /api/sales-operating-model/configuration-status",
            "GET    /api/sales-operating-model/context-lake",
            "GET    /api/sales-operating-model/interview-questions",
            "POST   /api/sales-operating-model/interview-questions",
            "PATCH  /api/sales-operating-model/interview-questions/{question_id}",
            "DELETE /api/sales-operating-model/interview-questions/{question_id}",
            "GET    /api/sales-operating-model/timeline-classification-section-status",
            "GET    /api/sales-operating-model/timeline-classification-cards",
            "GET    /api/sales-operating-model/timeline-classification-cards/{card_type}",
            "PUT    /api/sales-operating-model/timeline-classification-cards/{card_type}",
            "GET    /api/sales-operating-model/timeline-classification-configuration-status",
        ],
    }

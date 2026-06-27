"""Pydantic response models for the public API."""

from datetime import date, datetime
from typing import List, Literal
from uuid import UUID
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field, field_validator
from decimal import Decimal
from pydantic import BaseModel
from typing import List, Optional

class CreateOpportunityRequest(BaseModel):
   
    opportunityid: UUID
    name: Optional[str] = None
    accountid: Optional[str] = None
    contactid: Optional[str] = None
    stagename: Optional[str] = None
    estimatedvalue: Optional[Decimal] = None
    estimatedclosedate: Optional[date] = None
    lvo_forecastcategory: Optional[str] = None
    lvo_salesmotion: Optional[str] = None
    lvo_dealtempo: Optional[str] = None
    lvo_solutionarea: Optional[str] = None
    lvo_solutioncertificationid: Optional[str] = None
    lvo_dealqualificationreviewid: Optional[str] = None
    lvo_rfpid: Optional[str] = None
    lvo_geoid: Optional[str] = None
    lvo_country: Optional[str] = None
    lvo_businessgroup: Optional[str] = None
    lvo_dealdeskanalystid: Optional[str] = None
    lvo_dealengagementmanagerid: Optional[str] = None
    lvo_solutionarchitectid: Optional[str] = None
    lvo_leasingvendor: Optional[str] = None
    lvo_primary_partner: Optional[str] = None
    owninguser: Optional[str] = None
    msdyn_accountmanagerid: Optional[str] = None
    closeprobability: Optional[int] = None
    statecode: Optional[int] = None
    lvo_dealhealthscore: Optional[int] = None
    lvo_riskscore: Optional[int] = None
    lvo_riskreason: Optional[str] = None
    lvo_tempoclass: Optional[str] = None
    lvo_stageentrydate: Optional[datetime] = None
    lvo_createdat: Optional[datetime] = None
    lvo_dealhealthupdatedat: Optional[datetime] = None
    lvo_summary: Optional[str] = None
    lvo_priority: Optional[str] = None
    lvo_leadorigin: Optional[str] = None
    lvo_partnerinvolved: Optional[bool] = None
    lvo_parentopportunityid: Optional[str] = None
    createdby: Optional[str] = None
    modifiedon: Optional[datetime] = None
    modifiedby: Optional[str] = None
    actual_revenue: Optional[Decimal] = None
    actual_close_date: Optional[date] = None
    close_reason: Optional[str] = None
    sales_order_reference: Optional[str] = None
    won_solution_category: Optional[str] = None
    win_notes_commentary: Optional[str] = None
    invoice_number: Optional[str] = None
    loss_reason: Optional[str] = None
    competitor_won: Optional[str] = None
    lost_solution_category: Optional[str] = None
    lost_revenue_value: Optional[Decimal] = None
    loss_notes_commentary: Optional[str] = None
    deal_appeal: Optional[str] = None
    re_engagement_date: Optional[date] = None
    solution_area: Optional[str] = None
    sub_solution_area: Optional[str] = None
    solution_certifications: Optional[str] = None
    solution_offerings: Optional[str] = None
    leasing_vendor: Optional[str] = None
    sales_model: Optional[str] = None
    service_model: Optional[str] = None
    budget_confirmed: Optional[bool] = None
    quote_reference: Optional[str] = None
    partner_commercial_model: Optional[str] = None
    actual_confirmed_revenue: Optional[Decimal] = None
    reseller_channel_account: Optional[str] = None
    deal_protection_status: Optional[str] = None
    deal_registration_ref: Optional[str] = None
    number_of_countries: Optional[int] = None
    sow_required: Optional[bool] = None
    multi_country_solution_required: Optional[bool] = None
    deal_qualification_review: Optional[str] = None
    solution_handover_artefacts: Optional[str] = None
    solution_service_executive: Optional[str] = None
    solution_service_domain_specialist: Optional[str] = None
    lgfs_sales_representatives: Optional[str] = None
    lgfs_sales_support: Optional[str] = None
    deal_desk_analyst: Optional[str] = None
    deal_engagement_manager: Optional[str] = None
    ssds_channel: Optional[str] = None
    sell_through_week_auto: Optional[int] = None
    competitor_type: Optional[str] = None
    order_date: Optional[date] = None
    shipping_date: Optional[date] = None
    sales_order_reference_po: Optional[str] = None
    created_date: Optional[datetime] = None
    order_number: Optional[str] = None
    days_in_stage: Optional[int] = None

class CompetitorItem(BaseModel):
    id: Optional[str] = None
    opportunityId: str
    name: Optional[str] = None
    competitorName: str
    competitorType: Optional[str] = None
    resellingPartnerId: Optional[str] = None


class CompetitorsRequest(BaseModel):
    competitors: List[CompetitorItem]
class CompetitorResponse(BaseModel):
    id: str
    opportunityId: str
    name: str
    competitorName: str
    competitorType: Optional[str]
    resellingPartnerId: Optional[str]


class CompetitorsResponse(BaseModel):
    competitors: List[CompetitorResponse]

class CompetitorCreateRequest(BaseModel):
    opportunityId: str
    competitorName: str
    competitorType: Optional[str] = None
    resellingPartner: Optional[str] = None

class CompetitorCreateResponse(BaseModel):
    id: str
    opportunityId: str
    competitorName: str
    competitorType: Optional[str]
    resellingPartner: Optional[str]    

class APIModel(BaseModel):
    """Base model — emits camelCase to keep the JSON contract front-end friendly.

    Postgres UUID columns come back from SQLAlchemy as `uuid.UUID` objects,
    so we coerce them to strings here before field validation. Applies to
    every subclass automatically.
    """

    model_config = ConfigDict(
        from_attributes=True,
        alias_generator=lambda field_name: _to_camel(field_name),
        populate_by_name=True,
    )

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_uuid_to_str(cls, value: object) -> object:
        if isinstance(value, UUID):
            return str(value)
        return value


def _to_camel(snake: str) -> str:
    head, *tail = snake.split("_")
    return head + "".join(part.title() for part in tail)


# ----------------------------------------------------------------------------
# #5 — /api/opportunities/{id}/competitors
# ----------------------------------------------------------------------------


class Competitor(APIModel):
    id: str = Field(description="lvo_opportunitycompetitorid")
    opportunity_id: str
    name: str | None = Field(default=None, description="lvo_name — the row label")
    competitor_name: str | None = None
    competitor_type: str | None = Field(
        default=None, description="Direct | Indirect"
    )
    reselling_partner_id: str | None = None


class CompetitorList(APIModel):
    opportunity_id: str
    total: int
    items: list[Competitor]


# ----------------------------------------------------------------------------
# #8 — /api/filters/regions
# ----------------------------------------------------------------------------


class BusinessGroupOption(APIModel):
    id: str
    label: str


class CountryOption(APIModel):
    code: str
    label: str
    business_group: str | None = None


class GeographicUnitOption(APIModel):
    id: str
    name: str | None = None
    code: str | None = None
    parent_id: str | None = None


class RegionsResponse(APIModel):
    business_groups: list[BusinessGroupOption]
    countries: list[CountryOption]
    geographic_units: list[GeographicUnitOption]


# ----------------------------------------------------------------------------
# #9 — /api/filters/industries
# ----------------------------------------------------------------------------


class IndustryOption(APIModel):
    code: str
    label: str


class IndustriesResponse(APIModel):
    total: int
    items: list[IndustryOption]


# ----------------------------------------------------------------------------
# #10 — /api/filters/stages
# ----------------------------------------------------------------------------


class StageOption(APIModel):
    raw: str = Field(description="Original value from opportunity.stagename")
    label: str = Field(description="Normalised label shown in the UI")


class StagesResponse(APIModel):
    total: int
    items: list[StageOption]


# ----------------------------------------------------------------------------
# #11 — /api/filters/products
# ----------------------------------------------------------------------------


class ProductOption(APIModel):
    id: str = Field(description="Slug — lowercase, no spaces")
    label: str
    source: str = Field(description="'quoteitem' | 'solutionoffering'")


class ProductsResponse(APIModel):
    total: int
    items: list[ProductOption]


# ----------------------------------------------------------------------------
# #13 — /api/opportunities/{id}/sale-motion
# ----------------------------------------------------------------------------


class SaleMotionResponse(APIModel):
    opportunity_id: str
    raw: str | None = Field(default=None, description="Stored value in opportunity.lvo_salesmotion")
    label: str | None = Field(default=None, description="Normalised label for the UI pill")


# ----------------------------------------------------------------------------
# #1 — /api/opportunities/kpi-summary
# ----------------------------------------------------------------------------

ComparePeriod = Literal["last_week", "past_month", "last_quarter"]
TrendDirection = Literal["up", "down", "flat"]

# A bucket on the KPI strip. Used both by /kpi-summary (to label its cards)
# and by /opportunities (as a query-string filter when the user clicks a card).
#
# ``pipeline`` is the predicate ``lvo_forecastcategory = 'Pipeline'``. The
# Lenovo UI relabels this card as "Identified" — purely a display-side change,
# the predicate and bucket id stay ``pipeline``.
KpiBucket = Literal[
    "open", "pipeline", "best_case", "commit", "most_likely", "won", "loss"
]


class TrendInfo(APIModel):
    direction: TrendDirection
    delta_value: float = Field(description="Absolute change in monetary value")
    delta_count: int = Field(description="Change in deal count")


class KpiCard(APIModel):
    value: float = Field(description="Sum of estimatedvalue in the bucket")
    count: int = Field(description="Number of opportunities in the bucket")
    trend: TrendInfo | None = Field(
        default=None,
        description=(
            "Period-over-period change. Null until lvo_opportunitysnapshot "
            "is populated."
        ),
    )


class KpiSummaryResponse(APIModel):
    compare_period: ComparePeriod
    as_of: datetime
    currency: str = Field(default="USD")
    open_deals: KpiCard
    pipeline: KpiCard = Field(
        description=(
            "lvo_forecastcategory = 'Pipeline'. The Lenovo UI relabels this "
            "card as 'Identified' — predicate and field name stay `pipeline`."
        ),
    )
    best_case: KpiCard
    commit: KpiCard
    most_likely: KpiCard = Field(
        description="lvo_forecastcategory = 'Most Likely'.",
    )
    won: KpiCard
    loss: KpiCard
    notes: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# /api/accounts/kpi-summary — Account-page KPI strip
#
# Mirrors the Opportunities KPI summary but with four account-side buckets:
#   total       — every row in the account table
#   acv         — sum(opportunity.estimatedvalue) where statecode <> 'Canceled'
#   active      — account.lvo_accountstatus = 'Active'
#   at_risk     — account.lvo_accountstatus = 'At-Risk'
#
# ``ComparePeriod``, ``TrendDirection`` and ``TrendInfo`` are reused from
# the opportunity KPI section above — the trend payload shape is identical.
# ----------------------------------------------------------------------------

AccountKpiBucket = Literal["total", "acv", "active", "at_risk"]


class AccountKpiCard(APIModel):
    value: float = Field(
        description=(
            "Monetary aggregate for the bucket. ``acv`` carries the dollar "
            "sum; ``total`` / ``active`` / ``at_risk`` always emit 0 — the "
            "FE renders ``count`` for those cards."
        )
    )
    count: int = Field(description="Number of accounts in the bucket.")
    trend: TrendInfo | None = Field(
        default=None,
        description=(
            "Period-over-period change. Null until lvo_accountsnapshot is "
            "populated, or whenever any filter parameter is supplied "
            "(snapshots are global / unfiltered in v1)."
        ),
    )


class AccountKpiSummaryResponse(APIModel):
    compare_period: ComparePeriod
    as_of: datetime
    currency: str = Field(default="USD")
    total_accounts: AccountKpiCard
    account_value: AccountKpiCard
    active_accounts: AccountKpiCard
    accounts_at_risk: AccountKpiCard
    notes: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# #2 — /api/opportunities
# ----------------------------------------------------------------------------

OpportunityView = Literal["timeline", "details"]
OpportunitySort = Literal["name", "value", "closeDate", "closeProbability", "stage"]
SortOrder = Literal["asc", "desc"]


class StageRef(APIModel):
    raw: str | None = None
    label: str | None = None


class SaleMotionRef(APIModel):
    raw: str | None = None
    label: str | None = None


ActivityType = Literal["email", "meeting", "crm", "multiple"]
ActivityDirection = Literal["inbound", "outbound"]


class ActivityItem(APIModel):
    """One logged touchpoint — email / meeting / CRM update / multi-event."""

    id: str
    type: ActivityType
    direction: ActivityDirection | None = None
    subject: str | None = None
    body: str | None = None
    activity_date: datetime | None = None
    grouped_count: int | None = Field(
        default=None,
        description="Only set when type='multiple'; how many events on the day.",
    )


class OpportunityListItem(APIModel):
    id: str
    name: str | None = None
    account_id: str | None = None
    account_name: str | None = None
    industry: str | None = None
    country: str | None = None
    region: str | None = Field(default=None, description="Business group (e.g. EMEA BG)")
    stage: StageRef
    sale_motion: SaleMotionRef
    forecast_category: str | None = None
    value: float | None = None
    currency: str = Field(default="USD")
    close_date: date | None = None
    close_probability: float | None = None
    competitor_count: int = 0
    competitors: list[str] = Field(
        default_factory=list,
        description="Up to 3 competitor names for preview in the grid.",
    )
    owner_id: str | None = None
    statecode: str | None = None
    risk: str | None = Field(
        default=None,
        description=(
            "Short risk-reason label rendered in the grid's ⚠ badge "
            "(e.g. 'Budget Freeze'). Sourced from opportunity.lvo_riskreason."
        ),
    )
    risk_score: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description=(
            "Numeric risk severity 1–5 from opportunity.lvo_riskscore. "
            "Exposed for future filtering/sorting; the grid uses `risk`."
        ),
    )
    risk_count: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Count of Active rows in lvo_dealrisk for this opportunity — "
            "drives the '⚠ N' badge on the per-account Opportunities tab. "
            "Null when the lvo_dealrisk table is missing on this dump."
        ),
    )
    deal_health: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Deal Health percentage 0–100 driving the SVG ring on the grid. "
            "Sourced from opportunity.lvo_dealhealthscore."
        ),
    )
    # Placeholder fields still without backing schema (see canvas / API contract):
    next_action: None = Field(default=None, description="Awaits opportunity.lvo_nextaction*.")
    last_activity: datetime | None = Field(
        default=None,
        description="Most recent lvo_activity.lvo_activitydate for the opportunity.",
    )
    activities: list[ActivityItem] = Field(
        default_factory=list,
        description=(
            "Up to 5 most recent activities for this opportunity. Powers the "
            "timeline dots in the grid and the offcanvas panel."
        ),
    )


class OpportunityListResponse(APIModel):
    page: int
    page_size: int
    total: int
    total_pages: int
    sort_by: OpportunitySort
    sort_order: SortOrder
    items: list[OpportunityListItem]
    notes: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Deal Update — PATCH /api/opportunities/{id}
# ----------------------------------------------------------------------------

DealPriority = Literal["High", "Medium", "Low"]


class OpportunityUpdateRequest(APIModel):
    """PATCH body — all fields optional; only explicitly provided fields are applied.

    Stage is intentionally absent: stage advancement is controlled separately
    (it remains disabled for Qualify-stage deals and for all closed deals).

    The Complete-Information form on the Opportunity Detail page also writes
    ``summary``, ``priority``, ``lead_origin``, ``partner_involved``,
    ``parent_opportunity_id``, ``stage_entry_date`` and ``owner_id`` — all
    are appended below to keep the form-side and grid-side write paths on
    a single endpoint.
    """

    name: str | None = Field(default=None, description="Deal name — must be unique.")
    estimated_value: float | None = Field(
        default=None,
        description="Estimated revenue in USD — must be > 0 when provided (ERR_MSG_0012).",
    )
    estimated_close_date: date | None = Field(default=None, description="Target close date.")
    close_probability: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Win probability 0–100.",
    )
    forecast_category: str | None = Field(
        default=None,
        description=(
            "Open / Pipeline / Best Case / Most Likely / Commit for open deals. "
            "Locked for Closed Won / Closed Lost deals."
        ),
    )
    sale_motion: str | None = Field(
        default=None,
        description="Net-New | Expansion | Renewal",
    )

    # ------------------------------------------------------------------
    # Complete-Information (Deal Summary) extras.
    # ------------------------------------------------------------------
    summary: str | None = Field(
        default=None,
        description="Free-text deal summary — surfaced as the Summary input.",
    )
    priority: DealPriority | None = Field(
        default=None,
        description="Deal priority — High / Medium / Low.",
    )
    lead_origin: str | None = Field(
        default=None,
        description=(
            "Lead Origin — free-text dropdown value (e.g. Partner / Direct / "
            "Marketing / Inbound). Enums are governed FE-side for v1."
        ),
    )
    partner_involved: bool | None = Field(
        default=None,
        description="True when a partner is involved in this deal.",
    )
    parent_opportunity_id: str | None = Field(
        default=None,
        description=(
            "opportunityid of the parent deal. Setting to ``null`` clears the "
            "relationship. Self-references and cycles are rejected with "
            "INVALID_PARENT_OPPORTUNITY."
        ),
    )
    stage_entry_date: datetime | None = Field(
        default=None,
        description="Override the date the deal entered its current stage.",
    )
    owner_id: str | None = Field(
        default=None,
        description="systemuser.systemuserid of the new owner.",
    )


class OpportunityUpdateResponse(APIModel):
    id: str
    name: str | None = None
    stage: StageRef
    statecode: str | None = None
    estimated_value: float | None = None
    estimated_close_date: date | None = None
    close_probability: float | None = None
    forecast_category: str | None = None
    sale_motion: SaleMotionRef
    owner_id: str | None = None
    is_stage_locked: bool = Field(
        description=(
            "True when the deal stage cannot be advanced via this endpoint "
            "(Qualify stage or any closed deal)."
        ),
    )


# ----------------------------------------------------------------------------
# Competitor write — POST / PATCH / DELETE competitors
# ----------------------------------------------------------------------------

class CompetitorCreateRequest(APIModel):
    """Body for POST /api/opportunities/{id}/competitors."""

    competitor_name: str = Field(description="Competitor name — mandatory.")
    competitor_type: str | None = Field(
        default=None,
        description="Incumbent | Secondary",
    )
    reselling_partner: str | None = Field(default=None, description="Free-text reselling partner.")


class CompetitorUpdateRequest(APIModel):
    """Body for PATCH /api/opportunities/{id}/competitors/{competitor_id}."""

    competitor_name: str = Field(description="Competitor name — mandatory.")
    competitor_type: str | None = Field(default=None, description="Incumbent | Secondary")
    reselling_partner: str | None = Field(default=None, description="Free-text reselling partner.")


class CompetitorDeleteResponse(APIModel):
    opportunity_id: str
    competitor_id: str
    message: str


# ----------------------------------------------------------------------------
# Next Actions — GET / POST / PATCH /api/opportunities/{id}/next-actions
# ----------------------------------------------------------------------------

NextActionStatus = Literal["Open", "Completed"]


class NextActionItem(APIModel):
    id: str
    opportunity_id: str
    description: str
    due_date: date | None = None
    status: NextActionStatus
    verbal_written_acceptance:str 
    verbal_commit_date: datetime | None = None
    created_at: datetime
    updated_at: datetime
    created_by: str | None = None


class NextActionListResponse(APIModel):
    opportunity_id: str
    total: int
    items: list[NextActionItem]


class NextActionCreateRequest(APIModel):
    """Body for POST /api/opportunities/{id}/next-actions."""

    description: str = Field(description="Action description — mandatory.")
    due_date: date | None = Field(default=None, description="Target due date.")
    verbal_written_acceptance:str = Field(description="Verbal/Written Acceptance details.")
    verbal_commit_date: datetime | None = Field(default=None, description="Verbal Commit date.")


class NextActionUpdateRequest(APIModel):
    """Body for PATCH /api/opportunities/{id}/next-actions/{action_id}.

    All fields optional — only explicitly provided fields are applied.
    Set status to 'Completed' to mark the action done (retained for history).
    """

    description: str | None = None
    due_date: date | None = None
    verbal_written_acceptance:str | None = None
    verbal_commit_date: datetime | None = None
    status: NextActionStatus | None = None

class NoteCreateRequest(BaseModel):
    notes: str

class NoteResponse(BaseModel):
    action: str
    notes: str
class NoteUpdateRequest(BaseModel):
    opportunity_id: str
    notes: str


class Note(BaseModel):
    id: str
    notes: str

 
# ============================================================================
# Deal Detailed View
# ============================================================================
#
# All schemas below back the User Story "Deal Detailed View".
# They are split into:
#   * Account panel              — AccountSummary
#   * Contacts panel             — ContactRef, ContactListResponse, write reqs
#   * Activity timeline          — TimelineEvent, TimelineResponse
#   * Deal health & risks        — DealHealthInfo, RiskInfo, etc.
#   * The headline OpportunityDetail payload
#   * Soft-delete + recalc responses
# ============================================================================


# ----------------------------------------------------------------------------
# Account panel — GET /api/accounts/{id} and inlined into OpportunityDetail
# ----------------------------------------------------------------------------


class AccountSummary(APIModel):
    """Account fields the deal-detail view shows at the top of the page.

    `total_account_value` and `open_deals_count` are aggregated on the fly
    from `opportunity` rows that point at this account; they are never
    persisted on the account row.
    """

    id: str
    name: str | None = None
    segment: str | None = Field(default=None, description="account.lvo_segment")
    industry: str | None = Field(default=None, description="account.industrycode")
    territory: str | None = Field(default=None, description="account.lvo_territory")
    employee_count: int | None = Field(
        default=None, description="account.numberofemployees"
    )
    total_account_value: float = Field(
        default=0.0,
        description=(
            "Sum of estimatedvalue across every active opportunity tied to "
            "this account (open + won + lost). Excludes Canceled deals."
        ),
    )
    open_deals_count: int = Field(
        default=0,
        description="Count of opportunities on this account where statecode='Open'.",
    )
    business_group: str | None = Field(
        default=None, description="account.lvo_businessgroupid"
    )
    country: str | None = Field(default=None, description="account.lvo_countryid")


# ----------------------------------------------------------------------------
# Contacts panel — GET/POST/PATCH/DELETE on /opportunities/{id}/contacts
# ----------------------------------------------------------------------------


class ContactRef(APIModel):
    """One contact attached to a deal.

    Used both inline in OpportunityDetail (decision_maker + additional_contacts)
    and from the standalone contacts list endpoint.
    """

    id: str = Field(description="lvo_opportunitycontactid — the link row's PK")
    contact_id: str = Field(description="contact.contactid")
    name: str | None = None
    role: str | None = Field(
        default=None,
        description="Decision Maker | Champion | Influencer | Procurement | Technical | …",
    )
    is_decision_maker: bool = False
    last_touch_date: datetime | None = Field(
        default=None,
        description=(
            "Most recent activity that involved this contact. "
            "Cached on lvo_opportunitycontact.lvo_lasttouchdate; refreshed by "
            "the recalc service."
        ),
    )
    job_title: str | None = Field(default=None, description="contact.jobtitle")
    email: str | None = Field(default=None, description="contact.emailaddress1")
    # ------------------------------------------------------------------
    # Contact-level extras surfaced for the Opportunity > Contacts tab
    # (both the card grid and the detail / edit form).
    #
    # ``first_name`` / ``last_name`` are split out from ``contact.firstname``
    # / ``contact.lastname`` so the detail form can render them in their own
    # input boxes without parsing ``name``.
    #
    # ``phone`` is resolved at runtime via ``app.services.contact_phone`` —
    # it picks the first column that exists from
    # ``telephone1`` → ``mobilephone`` → ``lvo_phone``. ``None`` either
    # means the contact has no phone OR that the dump ships none of the
    # candidate columns.
    #
    # NOTE: editing ``phone`` (via PATCH on the contact-link) writes back
    # to the underlying ``contact`` row — so the change is visible on every
    # other deal that contact is attached to and on the Account Contacts
    # tab. The FE may want to surface a "this affects other deals" toast.
    # ------------------------------------------------------------------
    first_name: str | None = Field(default=None, description="contact.firstname")
    last_name: str | None = Field(default=None, description="contact.lastname")
    phone: str | None = Field(
        default=None,
        description=(
            "Resolved via ContactPhoneResolver — reads telephone1 / "
            "mobilephone / lvo_phone (whichever exists in the dump)."
        ),
    )


class ContactListResponse(APIModel):
    opportunity_id: str
    decision_maker: ContactRef | None = None
    additional_contacts: list[ContactRef] = Field(default_factory=list)
    total: int


class ContactLinkCreateRequest(APIModel):
    """Body for POST /api/opportunities/{id}/contacts."""

    contact_id: str = Field(description="contact.contactid to link to this deal")
    role: str | None = Field(
        default=None,
        description="Decision Maker | Champion | Influencer | Procurement | Technical",
    )
    is_decision_maker: bool = Field(
        default=False,
        description=(
            "Setting this True auto-demotes any other contact on the deal "
            "currently flagged as the decision maker."
        ),
    )


class ContactLinkUpdateRequest(APIModel):
    """Body for PATCH /api/opportunities/{id}/contacts/{contactLinkId}.

    All three fields are optional (true PATCH semantics). ``phone`` writes
    to the underlying ``contact`` row — see the note on ``ContactRef.phone``.
    """

    role: str | None = None
    is_decision_maker: bool | None = None
    phone: str | None = Field(
        default=None,
        description=(
            "Updates the contact's phone number on the underlying contact row "
            "(via the ContactPhoneResolver — telephone1 / mobilephone / "
            "lvo_phone). Empty string is treated as 'clear'. Visible on every "
            "other deal this contact is attached to."
        ),
    )


class ContactLinkDeleteResponse(APIModel):
    opportunity_id: str
    contact_link_id: str
    message: str


# ----------------------------------------------------------------------------
# Activity timeline — GET /api/opportunities/{id}/timeline
# ----------------------------------------------------------------------------


TimelineEventSource = Literal["activity", "crm_change"]


class TimelineEvent(APIModel):
    """One entry on the detailed timeline.

    Combines:
      * lvo_activity rows                    (source='activity')   email/meeting/CRM/multiple
      * lvo_audit_log opportunity updates    (source='crm_change') field-by-field changes

    Both sources expose the same shape so the frontend can render them
    chronologically without branching.
    """

    id: str
    source: TimelineEventSource
    type: str = Field(
        description=(
            "For source='activity': email | meeting | crm | multiple. "
            "For source='crm_change': the changed-field name (e.g. 'estimatedvalue')."
        ),
    )
    direction: ActivityDirection | None = None
    subject: str | None = None
    body: str | None = None
    event_date: datetime
    grouped_count: int | None = None
    changed_by: str | None = Field(
        default=None,
        description="Only set for source='crm_change' — copied from lvo_audit_log.lvo_changedby.",
    )


class TimelineResponse(APIModel):
    opportunity_id: str
    page: int
    page_size: int
    total: int
    total_pages: int
    items: list[TimelineEvent]


# ----------------------------------------------------------------------------
# Risks — GET /api/opportunities/{id}/risks  +  embedded in detail
# ----------------------------------------------------------------------------


RiskCategory = Literal[
    "Activity & Engagement",
    "Stakeholder",
    "Deal Execution",
    "Timeline & Forecast",
]


class RiskInfo(APIModel):
    id: str | None = Field(
        default=None,
        description=(
            "lvo_dealrisk.lvo_dealriskid when the risk has been persisted. "
            "Omitted on the live recalculation response, which is computed "
            "before the row is written."
        ),
    )
    category: RiskCategory
    name: str = Field(description="Stable identifier — e.g. 'Low Activity'.")
    message: str = Field(description="Human-readable label for the UI badge.")
    detected_at: datetime | None = None


class RiskListResponse(APIModel):
    opportunity_id: str
    total: int
    items: list[RiskInfo]


# ----------------------------------------------------------------------------
# Deal health — GET /api/opportunities/{id}/health  +  embedded in detail
# ----------------------------------------------------------------------------


HealthBand = Literal["GREEN", "YELLOW", "RED"]
HealthComponentKey = Literal[
    "stage_progress",
    "activity_freshness",
    "stakeholder",
    "close_confidence",
    "risk_adjustment",
]


class HealthComponent(APIModel):
    """One of the five weighted contributors to the overall deal health.

    `inputs` is an open-ended diagnostic dict — exactly what the calculator
    fed into the formula (e.g. tempoClass, ageDays, expectedDays). Useful
    when an admin wants to debug why a score landed where it did.
    """

    weight: int = Field(description="Percentage contribution to the final score (0-100).")
    score: float = Field(description="Component score 0-100.")
    inputs: dict = Field(default_factory=dict)


class DealHealthInfo(APIModel):
    score: int = Field(ge=0, le=100, description="Composite health score 0-100.")
    band: HealthBand
    updated_at: datetime | None = Field(
        default=None,
        description="opportunity.lvo_dealhealthupdatedat — last recalculation.",
    )
    components: dict[HealthComponentKey, HealthComponent]


class RecalculateHealthResponse(APIModel):
    opportunity_id: str
    health: DealHealthInfo
    risks: list[RiskInfo]
    message: str = "Deal health recalculated."


# ----------------------------------------------------------------------------
# OpportunityDetail — GET /api/opportunities/{id}
# ----------------------------------------------------------------------------


class OpportunityRef(APIModel):
    """Lightweight reference to an opportunity — used by:

    * The Parent / Child Opportunity pickers on the Complete-Information form.
    * The ``GET /api/opportunities/search`` typeahead endpoint.

    Intentionally minimal so the pickers stay snappy; full hydration happens
    via ``GET /api/opportunities/{id}`` once the user makes a selection.
    """

    id: str
    name: str | None = None


class OpportunityDetail(APIModel):
    """Full payload that hydrates the Deal Detail page in one round-trip.

    Reuses StageRef / SaleMotionRef / ActivityItem from the list response so
    the frontend can share components between the grid and the detail view.
    """

    id: str
    name: str | None = None
    account_id: str | None = None
    stage: StageRef
    sale_motion: SaleMotionRef
    forecast_category: str | None = None
    value: float | None = None
    currency: str = Field(default="USD")
    close_date: date | None = None
    close_probability: float | None = None
    owner_id: str | None = None
    owner_name: str | None = Field(
        default=None,
        description=(
            "Resolved seller display name — pulled from systemuser.fullname "
            "when that table ships with the dump; falls back to the raw UUID "
            "otherwise. Surfaced as the 'Owner' input on the Complete-"
            "Information form and the owner chip on the Overview tab."
        ),
    )
    statecode: str | None = None
    tempo_class: str | None = Field(
        default=None,
        description="Fast | Quarterly | Programmatic | Strategic — drives health math.",
    )
    created_at: datetime | None = None
    stage_entry_date: datetime | None = None
    is_closed: bool = Field(
        description="True when the deal is Closed Won, Closed Lost, or Canceled."
    )
    is_canceled: bool = Field(
        description="True when statecode='Canceled' (soft-deleted via DELETE)."
    )
    is_stage_locked: bool = Field(
        description="Mirrors the flag exposed by the PATCH-update endpoint."
    )

    # ------------------------------------------------------------------
    # Complete-Information (Deal Summary) form fields.
    #
    # Persisted on the opportunity row via the matching PATCH body. All
    # values default to ``None`` so a deal that hasn't been edited via
    # the form yet still serialises cleanly.
    # ------------------------------------------------------------------
    summary: str | None = Field(
        default=None,
        description="Free-text deal summary (Summary input on the form).",
    )
    priority: DealPriority | None = Field(
        default=None,
        description="Deal priority dropdown — High / Medium / Low.",
    )
    lead_origin: str | None = Field(
        default=None,
        description="Lead Origin dropdown value (free-text in v1).",
    )
    partner_involved: bool = Field(
        default=False,
        description="Whether a partner is involved on the deal (toggle).",
    )
    parent_opportunity_id: str | None = Field(
        default=None,
        description="opportunityid of the parent deal (self-FK).",
    )
    parent_opportunity_name: str | None = Field(
        default=None,
        description=(
            "Resolved label for ``parent_opportunity_id`` so the picker can "
            "render the chip without an extra round-trip. ``None`` when no "
            "parent is set or the parent has been cancelled."
        ),
    )
    child_opportunities: list[OpportunityRef] = Field(
        default_factory=list,
        description=(
            "Read-only — every deal whose ``parent_opportunity_id`` points "
            "back to this one. Cancelled children are filtered out."
        ),
    )
    days_in_stage: int | None = Field(
        default=None,
        description=(
            "Whole days between ``stage_entry_date`` and today (UTC). "
            "Negative input dates are clamped to 0. ``None`` when the "
            "deal has no stage_entry_date set."
        ),
    )

    # ------------------------------------------------------------------
    # System audit columns — exposed read-only on the form.
    # Sourced from D365's ``createdby`` / ``modifiedon`` / ``modifiedby``.
    # All three are ``deferred=True`` on the ORM so dumps that don't ship
    # them simply read as ``None`` here.
    # ------------------------------------------------------------------
    created_by: str | None = Field(
        default=None,
        description="opportunity.createdby (D365 user UUID or display name).",
    )
    modified_at: datetime | None = Field(
        default=None,
        description="opportunity.modifiedon — last edit timestamp.",
    )
    modified_by: str | None = Field(
        default=None,
        description="opportunity.modifiedby — last editor (D365 user UUID).",
    )

    # Composite panels
    account: AccountSummary
    decision_maker: ContactRef | None = None
    additional_contacts: list[ContactRef] = Field(default_factory=list)
    competitors: list[Competitor] = Field(default_factory=list)
    next_actions: list[NextActionItem] = Field(default_factory=list)
    activities_preview: list[ActivityItem] = Field(
        default_factory=list,
        description="Up to 10 most recent activities; full history at /timeline.",
    )
    health: DealHealthInfo
    risks: list[RiskInfo] = Field(default_factory=list)
    actual_revenue:str | None 
    actual_close_date:datetime | None 
    close_reason:str | None 
    sales_order_reference:str | None 
    won_solution_category:str | None 
    win_notes_commentary:str | None 
    invoice_number:str | None 
    loss_reason:str | None 
    competitor_won:str | None 
    lost_solution_category:str | None 
    lost_revenue_value:str | None 
    loss_notes_commentary:str | None 
    deal_appeal:str | None 
    re_engagement_date:datetime | None 
    solution_area:str | None 
    sub_solution_area:str | None 
    solution_certifications:str | None 
    solution_offerings:str | None 
    leasing_vendor:str | None 
    sales_model:str | None 
    service_model:str | None 

    budget_confirmed:str | None 

    quote_reference:str | None 
    partner_commercial_model:str | None 
    actual_confirmed_revenue:str | None 
    reseller_channel_account:str | None 

    deal_protection_status:str | None 
    deal_registration_ref:str | None 
    number_of_countries:str | None 



    sow_required:str | None 
    multi_country_solution_required:str | None 
    deal_qualification_review:str | None 
    solution_handover_artefacts:str | None 
    solution_service_executive:str | None 
    solution_service_domain_specialist:str | None 
    lgfs_sales_representatives:str | None 
    lgfs_sales_support:str | None 
    deal_desk_analyst:str | None 
    deal_engagement_manager:str | None 
    ssds_channel:str | None 
    sell_through_week_auto:str | None 
    competitor_type:str | None 

    order_date :datetime | None 
    shipping_date :datetime | None 
    sales_order_reference_po :str | None 
    created_date :datetime | None 
    order_number :str | None 
    


class OpportunitySearchResponse(APIModel):
    """Response shape for ``GET /api/opportunities/search`` — Parent-Opportunity picker.

    Returns a flat list of lightweight refs ordered by relevance (prefix match
    first, then substring match). The result is capped server-side via the
    ``limit`` query parameter (default 20, max 50).
    """

    query: str = Field(description="The search string the caller passed in.")
    total: int = Field(description="Length of ``items`` — clamped to ``limit``.")
    items: list[OpportunityRef] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Soft-delete — DELETE /api/opportunities/{id}
# ----------------------------------------------------------------------------


class OpportunityDeleteResponse(APIModel):
    id: str
    statecode: str = Field(default="Canceled")
    message: str


# ============================================================================
# View Account user story
# ============================================================================
#
# Schemas backing the Accounts grid + detail page. All names follow the same
# camelCase-on-the-wire convention as the rest of the file.
# ============================================================================


AccountType = Literal["Prospect", "Customer"]
AccountStatus = Literal["Active", "Inactive", "At-Risk"]
AccountSegment = Literal["SMB", "Mid-Market", "Enterprise", "Strategic"]
AccountSort = Literal[
    "name",
    "totalAccountValue",
    "openDealsCount",
    "lastInteraction",
    "status",
    "lvoAccountType",
]


class AccountListItem(APIModel):
    """One row on the Accounts grid (Screen 1 of the View Account story)."""

    id: str
    name: str | None = None
    account_number: str | None = Field(default=None, description="account.accountnumber")
    account_type: AccountType | None = None
    industry: str | None = Field(default=None, description="account.industrycode")
    segment: AccountSegment | str | None = Field(
        default=None,
        description="account.lvo_segment — SMB / Mid-Market / Enterprise / Strategic",
    )
    region: str | None = Field(
        default=None,
        description=(
            "Highest-precedence label among lvo_territory > lvo_businessgroupid "
            "> lvo_countryid. The grid renders one column."
        ),
    )
    business_group: str | None = None
    country: str | None = None
    territory: str | None = None
    status: AccountStatus | None = None
    statecode: str | None = Field(default=None, description="Raw account.statecode")
    last_interaction: datetime | None = Field(
        default=None,
        description="account.lvo_lastinteractiondate (cached MAX of lvo_activitydate).",
    )
    active_opportunities_count: int = Field(
        default=0,
        description="opportunities with statecode='Open' on this account.",
    )
    total_account_value: float = Field(
        default=0.0,
        description="Sum of estimatedvalue across non-Canceled opportunities.",
    )
    currency: str = Field(default="USD")
    employee_count: int | None = Field(default=None, description="account.numberofemployees")
    revenue: float | None = Field(default=None, description="account.revenue (annual)")
    owner_id: str | None = Field(default=None, description="account.owninguser (Seller).")
    owner_name: str | None = Field(
        default=None,
        description=(
            "Seller display name resolved via systemuser.fullname when that "
            "table is present in the dump; falls back to owner_id otherwise."
        ),
    )


class AccountListResponse(APIModel):
    page: int
    page_size: int
    total: int
    total_pages: int
    sort_by: AccountSort
    sort_order: SortOrder
    items: list[AccountListItem]
    notes: list[str] = Field(
        default_factory=list,
        description=(
            "Auxiliary diagnostics. When the result-set is empty after "
            "filters are applied, contains 'ERR_MSG_0010' so the FE can "
            "render the canonical no-results message."
        ),
    )


class AccountDetail(AccountListItem):
    """Full account-detail payload (Screen 2 of the View Account story).

    Reuses every field on the row item and adds rollups + descriptive metadata
    that the list view does not need.
    """

    # `id` etc. are inherited.
    account_value_label: str | None = Field(
        default=None,
        description="Formatted version of total_account_value for the header.",
    )
    won_deals_count: int = Field(default=0)
    lost_deals_count: int = Field(default=0)
    canceled_deals_count: int = Field(default=0)
    total_deals_count: int = Field(default=0)


class AccountFiltersResponse(APIModel):
    """Distinct filter options the FE renders into the multi-select pickers."""

    account_types: list[AccountType] = Field(default_factory=list)
    account_statuses: list[AccountStatus] = Field(default_factory=list)
    segments: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)


class AccountValueRangeResponse(APIModel):
    """Drives the dynamic min/max value slider."""

    min: float = Field(default=0.0, description="Smallest totalAccountValue across the dataset.")
    max: float = Field(default=0.0, description="Largest totalAccountValue across the dataset.")
    currency: str = Field(default="USD")


# ----------------------------------------------------------------------------
# Account-level contacts — GET/POST/PATCH/DELETE on /api/accounts/{id}/contacts
# ----------------------------------------------------------------------------


class AccountContactRef(APIModel):
    """One contact attached to an account (lvo_accountcontact link row).

    Adds delete-eligibility fields so the FE can grey-out the trash-can icon
    and surface the user-story tooltip without an extra round-trip
    (``ERR_MSG_0008`` / ``ERR_MSG_0009``).
    """

    id: str = Field(description="lvo_accountcontactid — the link row's PK")
    account_id: str
    contact_id: str = Field(description="contact.contactid")
    name: str | None = None
    first_name: str | None = Field(default=None, description="contact.firstname")
    last_name: str | None = Field(default=None, description="contact.lastname")
    role: str | None = Field(
        default=None,
        description="Primary | Influencer | Procurement | Technical | …",
    )
    is_primary: bool = False
    last_touch_date: datetime | None = None
    job_title: str | None = Field(default=None, description="contact.jobtitle")
    email: str | None = Field(default=None, description="contact.emailaddress1")
    phone: str | None = Field(
        default=None,
        description=(
            "Resolved via ContactPhoneResolver — reads telephone1, "
            "mobilephone, or lvo_phone depending on dump variant."
        ),
    )
    can_delete: bool = Field(
        default=True,
        description=(
            "False when the contact is primary or referenced by an active "
            "deal. FE should grey-out the delete control and show "
            "deleteRestrictionMessage on hover."
        ),
    )
    delete_restriction_code: str | None = Field(
        default=None,
        description="ERR_MSG_0008 (primary) or ERR_MSG_0009 (active deals).",
    )
    delete_restriction_message: str | None = None


class AccountContactListResponse(APIModel):
    account_id: str
    primary: AccountContactRef | None = None
    others: list[AccountContactRef] = Field(default_factory=list)
    total: int


class AccountContactCreateRequest(APIModel):
    """Body for ``POST /api/accounts/{id}/contacts``.

    Two valid shapes share this schema (validated in the router):

    1. **Attach existing contact** — supply ``contactId`` only (legacy flow).
    2. **Create-and-link** — supply ``firstName`` + ``lastName`` (required
       per the UI mockup) plus optional ``email`` / ``phone`` /
       ``jobTitle`` / ``role``. The router creates a fresh ``contact`` row
       then links it.

    Email and phone are validated against the same predicates the router
    uses for ``ERR_MSG_0013``; an empty / whitespace-only value is treated
    as "not supplied".
    """

    contact_id: str | None = Field(
        default=None,
        description="contact.contactid for the attach-existing flow.",
    )
    first_name: str | None = Field(
        default=None, description="Required for the create-and-link flow."
    )
    last_name: str | None = Field(
        default=None, description="Required for the create-and-link flow."
    )
    email: str | None = None
    phone: str | None = None
    job_title: str | None = None
    role: str | None = None
    is_primary: bool = Field(
        default=False,
        description=(
            "Setting True auto-demotes any other contact currently flagged "
            "as primary on the account."
        ),
    )

    @field_validator("email")
    @classmethod
    def _email_format(cls, value: str | None) -> str | None:
        from app.services.contact_validation import normalise_email, validate_email

        if not validate_email(value):
            raise ValueError("ERR_MSG_0013: invalid email address.")
        return normalise_email(value)

    @field_validator("phone")
    @classmethod
    def _phone_format(cls, value: str | None) -> str | None:
        from app.services.contact_validation import normalise_phone, validate_phone

        if not validate_phone(value):
            raise ValueError("ERR_MSG_0013: invalid phone number.")
        return normalise_phone(value)


class AccountContactUpdateRequest(APIModel):
    """Body for ``PATCH /api/accounts/{id}/contacts/{linkId}``.

    All fields optional — only the keys the caller supplies are mutated.
    Updating ``firstName`` / ``lastName`` / ``email`` / ``phone`` /
    ``jobTitle`` writes back to the underlying ``contact`` row; ``role`` and
    ``isPrimary`` mutate the ``lvo_accountcontact`` link row.
    """

    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    phone: str | None = None
    job_title: str | None = None
    role: str | None = None
    is_primary: bool | None = None

    @field_validator("email")
    @classmethod
    def _email_format(cls, value: str | None) -> str | None:
        from app.services.contact_validation import normalise_email, validate_email

        if not validate_email(value):
            raise ValueError("ERR_MSG_0013: invalid email address.")
        return normalise_email(value)

    @field_validator("phone")
    @classmethod
    def _phone_format(cls, value: str | None) -> str | None:
        from app.services.contact_validation import normalise_phone, validate_phone

        if not validate_phone(value):
            raise ValueError("ERR_MSG_0013: invalid phone number.")
        return normalise_phone(value)


class AccountContactDeleteResponse(APIModel):
    """Returned by a successful ``DELETE /api/accounts/{id}/contacts/{linkId}``."""

    account_id: str
    contact_link_id: str
    code: str = Field(
        default="SUCC_MSG_0007",
        description="User-story success code; FE renders the matching toast.",
    )
    message: str


class AccountContactDeleteEligibilityResponse(APIModel):
    """Returned by ``GET /api/accounts/{id}/contacts/{linkId}/delete-eligibility``.

    Mirrors the ``can_delete`` block embedded in each ``AccountContactRef`` so
    the FE can re-check just before showing ``CONF_MSG_0003``.
    """

    account_id: str
    contact_link_id: str
    can_delete: bool
    code: str | None = None
    message: str | None = None
    affected_deal_ids: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Opportunities-by-account — GET /api/accounts/{id}/opportunities
# ----------------------------------------------------------------------------


class AccountOpportunitiesResponse(APIModel):
    """Slim list of deals for the "Linked Opportunities" panel.

    Uses the existing OpportunityListItem shape so the FE can render with
    the same row component the main grid uses.
    """

    account_id: str
    page: int
    page_size: int
    total: int
    total_pages: int
    items: list[OpportunityListItem]


# ----------------------------------------------------------------------------
# Recompute-status — POST /api/accounts/{id}/recompute-status
# ----------------------------------------------------------------------------


class AccountRecomputeResponse(APIModel):
    id: str
    account_type: AccountType | None = None
    status: AccountStatus | None = None
    last_interaction: datetime | None = None
    message: str = "Account derived fields recomputed."


# ----------------------------------------------------------------------------
# Meetings resolver — POST /api/meetings/resolve-opportunity (Sprint 1A · US01)
# ----------------------------------------------------------------------------

OpportunityResolveMatchedBy = Literal["contact_email", "subject_keyword", "both"]


class OpportunityResolveRequest(APIModel):
    attendee_emails: list[str] = Field(min_length=1)
    subject: str | None = None
    organiser_email: str | None = None


class OpportunityResolveResponse(APIModel):
    opportunity_id: str
    account_id: str | None = None
    opportunity_name: str | None = None
    account_name: str | None = None
    match_score: float
    matched_by: OpportunityResolveMatchedBy
    matched_contact_count: int


# ----------------------------------------------------------------------------
# Contact resolver — POST /api/contacts/resolve-by-emails (Sprint 1A · US02)
# Contact search — GET /api/contacts/search (v0.19.0)
# Seller portfolio — GET /api/contacts (v0.16.0)
# ----------------------------------------------------------------------------


class ContactResolveRequest(APIModel):
    emails: list[str]


class ContactResolveResult(APIModel):
    email: str
    contact_id: str | None = None
    name: str | None = None
    job_title: str | None = None
    account_id: str | None = None
    account_name: str | None = None
    role: str | None = None


class ContactResolveResponse(APIModel):
    results: list[ContactResolveResult]


class ContactSearchItem(APIModel):
    contact_id: str
    name: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    job_title: str | None = None
    phone: str | None = None
    account_id: str | None = None
    account_name: str | None = None


class ContactSearchResponse(APIModel):
    name: str
    account: str | None = None
    total: int
    items: list[ContactSearchItem]


class SellerContactItem(APIModel):
    contact_id: str
    name: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    job_title: str | None = None
    phone: str | None = None
    account_id: str | None = None
    account_name: str | None = None
    opportunity_id: str
    opportunity_name: str | None = None
    role: str | None = None
    is_decision_maker: bool = False
    linked_opportunity_count: int = 1
    last_touch_date: datetime | None = None


class SellerContactListResponse(APIModel):
    seller_id: str
    page: int
    page_size: int
    total: int
    total_pages: int
    items: list[SellerContactItem]


# ----------------------------------------------------------------------------
# Customer Information tab — GET /api/accounts/{id}/customer-information
#
# Phase 1 of the "View Customer Information" user story is read-only and
# covers six sections that map to standard D365 columns + a small set of
# Lenovo-custom columns. Each section is a flat object so the FE can
# render the corresponding card on the screen with no client-side joins.
# ----------------------------------------------------------------------------


class CustomerInfoBasicInformation(APIModel):
    """The "Basic Information" card on the Customer Information tab."""

    account_id: str = Field(description="account.accountid")
    account_name: str | None = Field(
        default=None, description="account.name"
    )
    account_type: AccountType | None = Field(
        default=None, description="account.lvo_accounttype"
    )
    segment: AccountSegment | str | None = Field(
        default=None, description="account.lvo_segment"
    )
    sub_segment: str | None = Field(
        default=None, description="account.lvo_subsegment"
    )
    industry_segment: str | None = Field(
        default=None,
        description="account.industrycode (re-purposed as Industry Segment).",
    )
    gtm_segment: str | None = Field(
        default=None, description="account.lvo_gtmsegment"
    )
    annual_revenue: float | None = Field(
        default=None, description="account.revenue"
    )
    employee_count: int | None = Field(
        default=None, description="account.numberofemployees"
    )
    seller_known_as: str | None = Field(
        default=None, description="account.lvo_sellerknownas"
    )


class CustomerInfoAddress(APIModel):
    """One address block — used for both Billing and Shipping cards."""

    line1: str | None = None
    line2: str | None = None
    city: str | None = None
    state_province: str | None = Field(
        default=None,
        description=(
            "address1_stateorprovince / address2_stateorprovince — "
            "the form accepts either State or Province here."
        ),
    )
    postal_code: str | None = None
    country: str | None = None


class CustomerInfoIdentityLegal(APIModel):
    """The "Identity & Legal" card."""

    legal_name_local: str | None = Field(
        default=None, description="account.lvo_legalnamelocal"
    )
    local_language: str | None = Field(
        default=None, description="account.lvo_locallanguage"
    )
    alias: str | None = Field(
        default=None, description="account.lvo_alias"
    )
    tax_vat_number: str | None = Field(
        default=None, description="account.lvo_taxvatnumber"
    )
    legal_entity: str | None = Field(
        default=None, description="account.lvo_legalentity"
    )
    main_phone: str | None = Field(
        default=None, description="account.telephone1"
    )
    website: str | None = Field(
        default=None, description="account.websiteurl"
    )
    linkedin_url: str | None = Field(
        default=None, description="account.lvo_linkedinurl"
    )


class CustomerInfoCommercialTerms(APIModel):
    """The "Commercial Terms" card."""

    default_currency: str | None = Field(
        default=None, description="account.lvo_defaultcurrency"
    )
    payment_terms: str | None = Field(
        default=None,
        description=(
            "account.paymenttermscode — D365 ships this as an int "
            "option-set; surfaced as a string label so the FE can render "
            "the user-readable value."
        ),
    )
    price_list: str | None = Field(
        default=None, description="account.defaultpricelevelid"
    )
    deal_sign_config: str | None = Field(
        default=None, description="account.lvo_dealsignconfig"
    )


class CustomerInfoTerritoryOwnership(APIModel):
    """The "Territory & Ownership" card."""

    region: str | None = Field(
        default=None,
        description=(
            "Highest-precedence label among lvo_territory > "
            "lvo_businessgroupid > lvo_countryid."
        ),
    )
    sales_territory: str | None = Field(
        default=None,
        description=(
            "account.lvo_salesterritory when populated, else "
            "account.territoryid."
        ),
    )
    future_territory: str | None = Field(
        default=None, description="account.lvo_futureterritory"
    )
    sales_org: str | None = Field(
        default=None, description="account.lvo_salesorg"
    )
    territory_move_reason: str | None = Field(
        default=None, description="account.lvo_territorymovereason"
    )
    geographic_unit: str | None = Field(
        default=None, description="account.lvo_geographicunit"
    )
    sales_office: str | None = Field(
        default=None, description="account.lvo_salesoffice"
    )
    assigned_owner_id: str | None = Field(
        default=None, description="account.owninguser (UUID)."
    )
    assigned_owner_name: str | None = Field(
        default=None,
        description=(
            "Resolved from systemuser.fullname when the systemuser "
            "table is present; falls back to the raw UUID."
        ),
    )
    record_owner_id: str | None = Field(
        default=None, description="account.createdby (UUID)."
    )
    record_owner_name: str | None = None


class CustomerInformationResponse(APIModel):
    """Sectioned payload for the Customer Information tab.

    Every section is always returned (even when all its fields are
    ``null``) so the FE can render skeletons / empty-states without
    branching on response shape. ``notes`` carries diagnostics — e.g.
    when the lvo_* migration hasn't been applied yet.
    """

    id: str = Field(description="account.accountid (echoed for convenience).")
    basic_information: CustomerInfoBasicInformation
    billing_address: CustomerInfoAddress
    shipping_address: CustomerInfoAddress
    identity_and_legal: CustomerInfoIdentityLegal
    commercial_terms: CustomerInfoCommercialTerms
    territory_and_ownership: CustomerInfoTerritoryOwnership
    notes: list[str] = Field(
        default_factory=list,
        description=(
            "Diagnostic notes — e.g. 'lvo_subsegment column is missing — "
            "run sql/2026_06_account_customer_info_schema.sql'. Empty on "
            "a fully-migrated DB."
        ),
    )


# ----------------------------------------------------------------------------
# What Changed — GET /api/notifications + GET /api/activity-timeline
# ----------------------------------------------------------------------------


WhatChangedActivityType = Literal["email", "meeting", "crm_update", "risk", "task"]

WhatChangedLinkType = Literal["opportunity", "account", "outreach", "activity", "todo"]

WhatChangedDirection = Literal["inbound", "outbound"]


class WhatChangedItem(APIModel):
    """One row in the seller portfolio activity feed."""

    id: str = Field(
        description=(
            "Stable synthetic key for read-state and deep links — "
            "e.g. 'activity:<uuid>', 'audit:<uuid>:stagename', "
            "'risk:<uuid>', 'task:<uuid>'."
        ),
    )
    activity_type: WhatChangedActivityType
    title: str
    summary: str
    account_id: str | None = None
    account_name: str | None = None
    opportunity_id: str
    opportunity_name: str | None = None
    event_at: datetime
    is_read: bool = False
    link_type: WhatChangedLinkType
    link_id: str
    actor: str | None = Field(
        default=None,
        description="Raw actor id or email from source row (audit log / activity owner).",
    )
    actor_name: str | None = Field(
        default=None,
        description="Human-readable name resolved from systemuser when available.",
    )
    direction: WhatChangedDirection | None = Field(
        default=None,
        description=(
            "Inbound/outbound for activity-sourced items only "
            "(from lvo_activity.lvo_direction). Null for CRM audit, risk, and task rows."
        ),
    )
    category_label: str = Field(
        description=(
            "Subtitle prefix for Activity Timeline — e.g. 'Stage progression tracked', "
            "'Email received'. Pair with actorName: '{categoryLabel} · {actorName}'."
        ),
    )


class NotificationPanelResponse(APIModel):
    seller_id: str
    limit: int
    items: list[WhatChangedItem]


class ActivityTimelineResponse(APIModel):
    seller_id: str
    page: int
    page_size: int
    total: int
    total_pages: int
    items: list[WhatChangedItem]


class NotificationMarkReadResponse(APIModel):
    seller_id: str
    notification_id: str
    read_at: datetime


# ----------------------------------------------------------------------------
# Quarter Pulse — GET /api/quarter-pulse + PUT /api/quarter-pulse/quota
# ----------------------------------------------------------------------------


QuarterPulseBand = Literal["low", "medium", "high"]
QuarterPulseBarColor = Literal["red", "blue", "yellow", "green"]


class QuarterPulseMetric(APIModel):
    display_value: str
    progress_fill_percent: float | None = None
    band: QuarterPulseBand | None = None
    bar_color: QuarterPulseBarColor | None = None


class QuarterPulseAttainmentMetric(QuarterPulseMetric):
    percent: float | None = Field(
        default=None,
        description="Quota attainment percentage when quota is configured.",
    )


class QuarterPulseCoverageMetric(QuarterPulseMetric):
    ratio: float | None = Field(
        default=None,
        description="Pipeline coverage multiplier when quota is configured.",
    )


class QuarterPulseResponse(APIModel):
    quarter_label: str = Field(description="e.g. Q3")
    fiscal_year: int
    days_left_in_quarter: int
    last_updated_at: datetime
    quota_configured: bool
    quota_target: float | None = None
    closed_revenue: float = Field(
        description="Closed-won revenue in the current fiscal quarter.",
    )
    open_pipeline_value: float = Field(
        description="Sum of estimatedvalue for seller open opportunities.",
    )
    open_deal_count: int
    quota_attainment: QuarterPulseAttainmentMetric
    pipeline_coverage: QuarterPulseCoverageMetric
    prompt: str | None = Field(
        default=None,
        description="Shown when quota is not configured.",
    )


class QuarterPulseQuotaUpsertRequest(APIModel):
    quota_amount: float = Field(gt=0, description="Fiscal-quarter revenue target.")
    fiscal_year: int | None = Field(
        default=None,
        description="Defaults to the current fiscal year when omitted.",
    )
    fiscal_quarter: int | None = Field(
        default=None,
        ge=1,
        le=4,
        description="Defaults to the current fiscal quarter when omitted.",
    )
    currency_code: str = Field(default="USD", min_length=3, max_length=3)


class QuarterPulseQuotaUpsertResponse(APIModel):
    seller_id: str
    fiscal_year: int
    fiscal_quarter: int
    quota_amount: float
    currency_code: str
    source: Literal["manual", "d365"]
    set_by: str | None = None
    modified_at: datetime
    quarter_pulse: QuarterPulseResponse


# ---------------------------------------------------------------------------
# Sprint 2 · US 1.3 — Task Pending badge (Home header)
# ---------------------------------------------------------------------------


class TaskPendingSummaryResponse(APIModel):
    seller_id: str
    count: int = Field(description="Open next-action rows on seller opportunities.")
    overdue_count: int
    due_today_count: int
    has_overdue: bool
    badge_color: Literal["red", "default"]
    label: str = Field(description='e.g. "5 tasks pending"')
    last_updated_at: datetime
    source: Literal["d365"] = Field(
        default="d365",
        description="FE merges with AIBackend /ai-api/todos/summary for full badge count.",
    )


# ---------------------------------------------------------------------------
# Sprint 2 · Pre-Meeting Briefing — GET /api/briefing/context
# ---------------------------------------------------------------------------


class BriefingSourceRefResponse(APIModel):
    source_type: str
    source_id: str
    label: str


class BriefingFactFieldResponse(APIModel):
    field_name: str
    display_label: str
    value: str | None = None
    is_missing: bool
    is_unverified: bool
    source: BriefingSourceRefResponse | None = None


class BriefingCompetitorItemResponse(APIModel):
    competitor_name: str
    competitor_type: str | None = None
    reselling_partner: str | None = None
    primary_risk: str | None = None
    source: BriefingSourceRefResponse


class BriefingAccountFactsResponse(APIModel):
    account_id: str | None = None
    account_name: str | None = None
    fields: list[BriefingFactFieldResponse] = Field(default_factory=list)
    paragraph: str
    word_count: int
    max_words: int
    gaps: list[str] = Field(default_factory=list)
    unverified_labels: list[str] = Field(default_factory=list)


class BriefingDealFactsResponse(APIModel):
    opportunity_id: str
    opportunity_name: str | None = None
    stage: str | None = None
    fields: list[BriefingFactFieldResponse] = Field(default_factory=list)
    paragraph: str
    word_count: int
    max_words: int
    gaps: list[str] = Field(default_factory=list)
    competitor_intel: list[BriefingCompetitorItemResponse] | None = None
    competitor_message_code: str | None = Field(
        default=None,
        description='Set to INF_MSG_0004 when no CRM competitor data exists.',
    )


class BriefingSignalItemResponse(APIModel):
    signal_id: str
    summary: str
    why_shown: str
    event_at: datetime
    involved_parties: list[str] = Field(default_factory=list)
    source: BriefingSourceRefResponse


class BriefingContextResponse(APIModel):
    seller_id: str
    opportunity_id: str
    account_id: str | None = None
    generated_at: datetime
    account: BriefingAccountFactsResponse
    deal: BriefingDealFactsResponse
    signals: list[BriefingSignalItemResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sprint 2 · US 3.2.1 — Sales Operating Model · Interview-First Setup
# ---------------------------------------------------------------------------

SomRoleLiteral = Literal[
    "national_manager", "regional_manager", "seller_manager"
]
SomIntentStatusLiteral = Literal["NOT_CONFIGURED", "CONFIGURED"]


class SomInterviewQuestionItem(APIModel):
    question_id: str
    sort_order: int
    text: str


class SomInterviewQuestionAdminItem(SomInterviewQuestionItem):
    role: SomRoleLiteral


class SomInterviewSetupResponse(APIModel):
    role: SomRoleLiteral
    role_display: str
    scope_label: str
    questions: list[SomInterviewQuestionItem]
    responses: dict[str, str] = Field(
        description="Draft responses keyed by questionId for the active capture cycle.",
    )
    saved_responses: dict[str, str] = Field(
        default_factory=dict,
        description="Last committed responses from Verify & Edit save (read-only reference).",
    )
    captured_count: int
    total_questions: int
    verify_enabled: bool
    intent_card_status: SomIntentStatusLiteral
    configured_at: datetime | None = None


class SomInterviewResponseInput(APIModel):
    question_id: str
    text: str = ""


class SomInterviewDraftRequest(APIModel):
    responses: list[SomInterviewResponseInput]


class SomInterviewSaveRequest(APIModel):
    responses: list[SomInterviewResponseInput]


class SomInterviewSaveResponse(APIModel):
    role: SomRoleLiteral
    intent_card_status: SomIntentStatusLiteral
    captured_count: int
    total_questions: int
    verify_enabled: bool
    configured_at: datetime | None = None


class SomIntentCardItem(APIModel):
    role: SomRoleLiteral
    role_display: str
    scope_label: str
    status: SomIntentStatusLiteral
    configured_at: datetime | None = None


class SomIntentCardsResponse(APIModel):
    items: list[SomIntentCardItem]


class SomContextLakeRoleBlock(APIModel):
    role_display: str
    scope_label: str
    status: SomIntentStatusLiteral
    configured_at: str | None = None
    interview_responses: list[dict[str, object]] = Field(default_factory=list)


class SomContextLakeOrgIntentBlock(APIModel):
    display_name: str
    status: SomIntentStatusLiteral
    last_synced_at: str | None = None
    is_timeboxed: bool = False
    is_guardrail: bool = False
    expiry_date: str | None = None
    fields: dict[str, object] = Field(default_factory=dict)


class SomContextLakeTimelineBlock(APIModel):
    display_name: str
    status: SomIntentStatusLiteral
    last_synced_at: str | None = None
    fields: dict[str, object] = Field(default_factory=dict)


class SomContextLakeInterviewBlock(APIModel):
    roles: dict[str, SomContextLakeRoleBlock] = Field(default_factory=dict)


class SomContextLakeResponse(APIModel):
    version: int = 2
    cycle_id: str | None = None
    updated_at: str | None = None
    interview: SomContextLakeInterviewBlock = Field(
        default_factory=SomContextLakeInterviewBlock
    )
    organizational_intents: dict[str, SomContextLakeOrgIntentBlock] = Field(
        default_factory=dict
    )
    timeline_classification: dict[str, SomContextLakeTimelineBlock] = Field(
        default_factory=dict
    )


class SomInterviewQuestionCreateRequest(APIModel):
    role: SomRoleLiteral
    question_text: str
    sort_order: int | None = None


class SomInterviewQuestionUpdateRequest(APIModel):
    question_text: str | None = None
    sort_order: int | None = None


# ---------------------------------------------------------------------------
# Sprint 2 · US 3.2.2 — Organizational Intent Setup
# ---------------------------------------------------------------------------

OrgIntentTypeLiteral = Literal[
    "outcome", "motion", "focus", "behavioral", "constraint", "tradeoff"
]


class SomMarginFloorItem(APIModel):
    deal_type: str
    min_percent: float


class SomPricingAuthorityItem(APIModel):
    role: str
    max_discount_percent: float


class SomOrganizationalIntentSummary(APIModel):
    intent_type: OrgIntentTypeLiteral
    display_name: str
    status: SomIntentStatusLiteral
    is_timeboxed: bool
    is_guardrail: bool
    last_synced_at: datetime | None = None
    field_preview: dict[str, object] = Field(default_factory=dict)


class SomOrganizationalIntentCustomMetricItem(APIModel):
    id: str
    label: str
    description: str
    sort_order: int


class SomOrganizationalIntentDetailResponse(APIModel):
    intent_type: OrgIntentTypeLiteral
    display_name: str
    status: SomIntentStatusLiteral
    is_timeboxed: bool
    is_guardrail: bool
    guardrail_warning: str | None = None
    last_synced_at: datetime | None = None
    expiry_date: date | None = None
    fields: dict[str, object] = Field(default_factory=dict)
    field_labels: dict[str, str] = Field(default_factory=dict)
    custom_metrics: list[SomOrganizationalIntentCustomMetricItem] = Field(
        default_factory=list
    )


class SomOrganizationalIntentMetricCreateRequest(APIModel):
    label: str
    description: str
    sort_order: int | None = None


class SomOrganizationalIntentMetricUpdateRequest(APIModel):
    label: str | None = None
    description: str | None = None
    sort_order: int | None = None


class SomOrganizationalIntentListResponse(APIModel):
    items: list[SomOrganizationalIntentSummary]


class SomOrganizationalIntentSaveRequest(APIModel):
    fields: dict[str, object] = Field(default_factory=dict)


class SomOrganizationalIntentSaveResponse(APIModel):
    intent_type: OrgIntentTypeLiteral
    display_name: str
    status: SomIntentStatusLiteral
    is_timeboxed: bool
    is_guardrail: bool
    last_synced_at: datetime
    field_preview: dict[str, object] = Field(default_factory=dict)
    all_configured: bool
    success_code: str | None = None


class SomOrganizationalIntentBulkSaveRequest(APIModel):
    """Save one or more cards atomically — keys are intent types."""

    cards: dict[str, SomOrganizationalIntentSaveRequest] = Field(
        ...,
        description=(
            "Map of intentType → card payload. Each value may be "
            "`{ fields: {...} }` (same as single-card PUT) or a flat field map."
        ),
    )


class SomOrganizationalIntentBulkSaveResponse(APIModel):
    items: list[SomOrganizationalIntentSaveResponse]
    all_configured: bool
    configured_count: int
    total_count: int
    success_code: str | None = None


class SomConfigurationStatusResponse(APIModel):
    all_configured: bool
    configured_count: int
    total_count: int
    success_code: str | None = None


class SomValidationErrorDetail(APIModel):
    code: str
    field: str
    message: str
    intent_type: str | None = Field(
        default=None,
        description="Present on bulk-save validation errors to identify the card.",
    )


class SomFieldNotFoundDetail(APIModel):
    code: str = "ERR_MSG_0025"
    intent_type: str
    field: str
    message: str
    available_fields: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sprint 2 · US 3.2.3 — Delete Intent Card
# ---------------------------------------------------------------------------


class SomOrganizationalIntentDeleteResponse(APIModel):
    intent_type: OrgIntentTypeLiteral
    display_name: str
    status: SomIntentStatusLiteral
    deleted_at: datetime
    configured_count: int
    total_count: int
    message: str = Field(
        default=(
            "Configuration removed. AI agents will no longer apply "
            "this intent from the next recommendation cycle."
        )
    )


class SomOrganizationalIntentFieldDeleteResponse(APIModel):
    intent_type: OrgIntentTypeLiteral
    display_name: str
    status: SomIntentStatusLiteral
    deleted_field: str
    is_timeboxed: bool
    is_guardrail: bool
    last_synced_at: datetime | None = None
    fields: dict[str, object] = Field(default_factory=dict)
    custom_metrics: list[SomOrganizationalIntentCustomMetricItem] = Field(
        default_factory=list
    )
    field_preview: dict[str, object] = Field(default_factory=dict)
    all_configured: bool
    configured_count: int
    total_count: int
    success_code: str | None = None


# ---------------------------------------------------------------------------
# Sprint 2 · US 3.4.1 — Timeline Classification & Canonical Sales Clock
# ---------------------------------------------------------------------------

TimelineCardTypeLiteral = Literal[
    "tempo_classes",
    "anchor_definitions",
    "signal_expectations_time_band",
    "seasonal_delayed_activation",
    "acceleration_decay",
    "multiyear_programmatic",
    "exit_recycle_kill",
    "canonical_timeline",
]


class SomTimelineSectionStatusResponse(APIModel):
    section_unlocked: bool
    organizational_intent_configured: bool
    timeline_configured_count: int
    timeline_total_count: int
    all_timeline_configured: bool
    success_code: str | None = None
    message_code: str | None = None


class SomTimelineCardSummary(APIModel):
    card_type: TimelineCardTypeLiteral
    display_name: str
    status: SomIntentStatusLiteral
    status_badge: str = Field(
        description="FE maps CONFIGURED → ACTIVE badge label."
    )
    last_synced_at: datetime | None = None
    field_preview: dict[str, object] = Field(default_factory=dict)


class SomTimelineCardListResponse(APIModel):
    section_label: str = "Time-aware expectations"
    items: list[SomTimelineCardSummary]


class SomTimelineCardDetailResponse(APIModel):
    card_type: TimelineCardTypeLiteral
    display_name: str
    status: SomIntentStatusLiteral
    last_synced_at: datetime | None = None
    fields: dict[str, object] = Field(default_factory=dict)
    field_labels: dict[str, str] = Field(default_factory=dict)
    defaults: dict[str, object] = Field(default_factory=dict)
    required_fields: list[str] = Field(default_factory=list)


class SomTimelineCardSaveRequest(APIModel):
    fields: dict[str, object] = Field(default_factory=dict)
    confirm_agent_impact: bool = Field(
        default=False,
        description="Must be true after admin confirms INFO_MSG_0006.",
    )


class SomTimelineCardSaveResponse(APIModel):
    card_type: TimelineCardTypeLiteral
    display_name: str
    status: SomIntentStatusLiteral
    last_synced_at: datetime
    all_configured: bool
    success_code: str | None = None


class SomTimelineConfigurationStatusResponse(APIModel):
    all_configured: bool
    configured_count: int
    total_count: int
    success_code: str | None = None


class SomAgentImpactDetail(APIModel):
    code: str
    message: str

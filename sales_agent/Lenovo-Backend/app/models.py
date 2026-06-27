"""SQLAlchemy ORM models.

Only tables (and columns) that are read by the public API are modelled here.
Columns are intentionally a subset of the DDL — extend as new endpoints are
implemented.
"""

from datetime import date, datetime


from sqlalchemy import (
    JSON,
    UUID,
    Boolean,
    Column,
    Date,
    DateTime,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column
from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime, date
from decimal import Decimal
from app.database import Base



class Opportunity(Base):
    __tablename__ = "opportunity"

    opportunityid: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str | None] = mapped_column(String)
    accountid: Mapped[str | None] = mapped_column(String)
    contactid: Mapped[str | None] = mapped_column(String)
    stagename: Mapped[str | None] = mapped_column(String)
    estimatedvalue: Mapped[float | None] = mapped_column(Numeric)
    estimatedclosedate: Mapped[date | None] = mapped_column(Date)
    closeprobability: Mapped[float | None] = mapped_column(Numeric)
    lvo_forecastcategory: Mapped[str | None] = mapped_column(String)
    lvo_salesmotion: Mapped[str | None] = mapped_column(String)
    lvo_country: Mapped[str | None] = mapped_column(String)
    lvo_geoid: Mapped[str | None] = mapped_column(String)
    lvo_businessgroup: Mapped[str | None] = mapped_column(String)
    owninguser: Mapped[str | None] = mapped_column(String)
    msdyn_accountmanagerid: Mapped[str | None] = mapped_column(String)
    statecode: Mapped[str | None] = mapped_column(String)
    # Added by sql/2026_06_add_dealhealth.sql — populated only after the
    # migration runs. Accessed defensively by the /api/opportunities router.
    lvo_dealhealthscore: Mapped[int | None] = mapped_column(Integer)
    lvo_riskscore: Mapped[int | None] = mapped_column(SmallInteger)
    lvo_riskreason: Mapped[str | None] = mapped_column(String)
    # Added by sql/2026_06_deal_detail_schema.sql — drives the deal-health
    # calculator (tempo class + stage entry date + creation date) and tracks
    # when the score was last refreshed. All four are optional in the DB.
    lvo_tempoclass: Mapped[str | None] = mapped_column(String)
    lvo_stageentrydate: Mapped[datetime | None] = mapped_column(DateTime)
    lvo_createdat: Mapped[datetime | None] = mapped_column(DateTime)
    lvo_dealhealthupdatedat: Mapped[datetime | None] = mapped_column(DateTime)
    # ----------------------------------------------------------------------
    # Added by sql/2026_06_opportunity_complete_info_schema.sql — Complete
    # Information tab on the Opportunity Detail page (Deal Summary section).
    #
    # Five editable fields + three D365 audit columns. The audit columns
    # (``createdby`` / ``modifiedon`` / ``modifiedby``) are marked
    # ``deferred=True`` because the dump-quality varies in dev — most
    # production D365 dumps include them, but stripped dumps don't, and we
    # don't want every code path that ``SELECT *``s an opportunity to 500
    # on a missing column. The Complete-Information GET path explicitly
    # ``undefer()``s them.
    # ----------------------------------------------------------------------
    lvo_summary: Mapped[str | None] = mapped_column(String)
    lvo_priority: Mapped[str | None] = mapped_column(String)
    lvo_leadorigin: Mapped[str | None] = mapped_column(String)
    lvo_partnerinvolved: Mapped[bool] = mapped_column(Boolean, default=False)
    lvo_parentopportunityid: Mapped[str | None] = mapped_column(String)
    createdby: Mapped[str | None] = mapped_column(String, deferred=True)
    modifiedon: Mapped[datetime | None] = mapped_column(DateTime, deferred=True)
    modifiedby: Mapped[str | None] = mapped_column(String, deferred=True)
    actual_revenue = Column(String, nullable=True)
    actual_close_date = Column(Date, nullable=True)
    close_reason = Column(String, nullable=True)
    sales_order_reference = Column(String, nullable=True)
    won_solution_category = Column(String, nullable=True)
    win_notes_commentary = Column(String, nullable=True)
    invoice_number = Column(String, nullable=True)
    loss_reason = Column(String, nullable=True)
    competitor_won = Column(String, nullable=True)
    lost_solution_category = Column(String, nullable=True)
    lost_revenue_value = Column(String, nullable=True)
    loss_notes_commentary = Column(String, nullable=True)
    deal_appeal = Column(String, nullable=True)
    re_engagement_date = Column(Date, nullable=True)

    solution_area = Column(String, nullable=True)
    sub_solution_area = Column(String, nullable=True)
    solution_certifications = Column(String, nullable=True)
    solution_offerings = Column(String, nullable=True)
    leasing_vendor = Column(String, nullable=True)
    sales_model = Column(String, nullable=True)
    service_model = Column(String, nullable=True)

    budget_confirmed = Column(String, nullable=True)

    quote_reference = Column(String, nullable=True)
    partner_commercial_model = Column(String, nullable=True)
    actual_confirmed_revenue = Column(String, nullable=True)
    reseller_channel_account = Column(String, nullable=True)

    deal_protection_status = Column(String, nullable=True)
    deal_registration_ref = Column(String, nullable=True)
    number_of_countries = Column(String, nullable=True)

    sow_required = Column(String, nullable=True)
    multi_country_solution_required = Column(String, nullable=True)
    deal_qualification_review = Column(String, nullable=True)
    solution_handover_artefacts = Column(String, nullable=True)
    solution_service_executive = Column(String, nullable=True)
    solution_service_domain_specialist = Column(String, nullable=True)
    lgfs_sales_representatives = Column(String, nullable=True)
    lgfs_sales_support = Column(String, nullable=True)
    deal_desk_analyst = Column(String, nullable=True)
    deal_engagement_manager = Column(String, nullable=True)
    ssds_channel = Column(String, nullable=True)
    sell_through_week_auto = Column(String, nullable=True)
    competitor_type = Column(String, nullable=True)

    order_date = Column(Date, nullable=True)
    shipping_date = Column(Date, nullable=True)
    sales_order_reference_po = Column(String, nullable=True)
    created_date = Column(Date, nullable=True)
    order_number = Column(String, nullable=True)
    days_in_stage = Column(String, nullable=True)


class Account(Base):
    __tablename__ = "account"

    accountid: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str | None] = mapped_column(String)
    accountnumber: Mapped[str | None] = mapped_column(String)
    industrycode: Mapped[str | None] = mapped_column(String)
    lvo_segment: Mapped[str | None] = mapped_column(String)
    # Already present on the base D365 dump.
    revenue: Mapped[float | None] = mapped_column(Numeric)
    numberofemployees: Mapped[int | None] = mapped_column(Integer)
    lvo_businessgroupid: Mapped[str | None] = mapped_column(String)
    lvo_countryid: Mapped[str | None] = mapped_column(String)
    lvo_defaultcurrency: Mapped[str | None] = mapped_column(String)
    owninguser: Mapped[str | None] = mapped_column(String)
    # Added by sql/2026_06_account_extension.sql.
    lvo_territory: Mapped[str | None] = mapped_column(String)
    # Added by sql/2026_06_account_view_schema.sql — drive the View Account
    # grid filters and the persisted "At-Risk" derivation. All three are
    # nullable; the routers defer them when the migration hasn't run.
    lvo_accounttype: Mapped[str | None] = mapped_column(String)
    lvo_accountstatus: Mapped[str | None] = mapped_column(String)
    lvo_lastinteractiondate: Mapped[datetime | None] = mapped_column(DateTime)
    statecode: Mapped[str | None] = mapped_column(String)

    # ------------------------------------------------------------------
    # Customer-Information tab — Phase 1 read-only fields.
    #
    # Every column below is marked ``deferred=True`` so the default
    # ``SELECT Account`` issued by other routers (``_ensure_account``,
    # the View-Account list/detail, account-contacts CRUD, recompute
    # endpoints, etc.) does NOT include them. Without this, a stripped
    # D365 dump that lacks e.g. ``address1_line1`` would 500 every read
    # path on the account.
    #
    # The Customer-Information endpoint uses ``undefer()`` to pull only
    # the columns it knows are present (via ``account_columns.py``
    # introspection). Code paths that bypass undefer simply read these
    # attributes as ``None``, never triggering a lazy SELECT against a
    # missing column.
    #
    # Powering:
    #   * Billing Address  —  address1_*
    #   * Shipping Address —  address2_*
    #   * Identity & Legal —  telephone1, websiteurl, tickersymbol
    #   * Commercial Terms —  paymenttermscode, defaultpricelevelid
    #   * Territory        —  territoryid, createdby
    # ------------------------------------------------------------------
    address1_line1: Mapped[str | None] = mapped_column(String, deferred=True)
    address1_line2: Mapped[str | None] = mapped_column(String, deferred=True)
    address1_city: Mapped[str | None] = mapped_column(String, deferred=True)
    address1_stateorprovince: Mapped[str | None] = mapped_column(String, deferred=True)
    address1_postalcode: Mapped[str | None] = mapped_column(String, deferred=True)
    address1_country: Mapped[str | None] = mapped_column(String, deferred=True)

    address2_line1: Mapped[str | None] = mapped_column(String, deferred=True)
    address2_line2: Mapped[str | None] = mapped_column(String, deferred=True)
    address2_city: Mapped[str | None] = mapped_column(String, deferred=True)
    address2_stateorprovince: Mapped[str | None] = mapped_column(String, deferred=True)
    address2_postalcode: Mapped[str | None] = mapped_column(String, deferred=True)
    address2_country: Mapped[str | None] = mapped_column(String, deferred=True)

    telephone1: Mapped[str | None] = mapped_column(String, deferred=True)
    websiteurl: Mapped[str | None] = mapped_column(String, deferred=True)
    tickersymbol: Mapped[str | None] = mapped_column(String, deferred=True)
    paymenttermscode: Mapped[int | None] = mapped_column(Integer, deferred=True)
    defaultpricelevelid: Mapped[str | None] = mapped_column(String, deferred=True)
    territoryid: Mapped[str | None] = mapped_column(String, deferred=True)
    createdby: Mapped[str | None] = mapped_column(String, deferred=True)
    description: Mapped[str | None] = mapped_column(String, deferred=True)
    # Standard D365 row-creation timestamp. Used by the account-KPI snapshot
    # service as the "existed-by" filter during backfill so historical
    # ``total`` aggregates reflect the row count as of N days ago.
    # Marked deferred + introspected at the call site — partial dumps that
    # lack the column simply skip the WHERE clause and fall back to "today".
    createdon: Mapped[datetime | None] = mapped_column(DateTime, deferred=True)

    # ------------------------------------------------------------------
    # Lenovo-custom columns — added by
    # sql/2026_06_account_customer_info_schema.sql. Same ``deferred=True``
    # treatment so unmigrated environments don't break the rest of the app.
    # ------------------------------------------------------------------
    lvo_subsegment: Mapped[str | None] = mapped_column(String, deferred=True)
    lvo_gtmsegment: Mapped[str | None] = mapped_column(String, deferred=True)
    lvo_sellerknownas: Mapped[str | None] = mapped_column(String, deferred=True)

    lvo_legalnamelocal: Mapped[str | None] = mapped_column(String, deferred=True)
    lvo_locallanguage: Mapped[str | None] = mapped_column(String, deferred=True)
    lvo_alias: Mapped[str | None] = mapped_column(String, deferred=True)
    lvo_taxvatnumber: Mapped[str | None] = mapped_column(String, deferred=True)
    lvo_legalentity: Mapped[str | None] = mapped_column(String, deferred=True)
    lvo_linkedinurl: Mapped[str | None] = mapped_column(String, deferred=True)

    lvo_dealsignconfig: Mapped[str | None] = mapped_column(String, deferred=True)

    lvo_salesterritory: Mapped[str | None] = mapped_column(String, deferred=True)
    lvo_futureterritory: Mapped[str | None] = mapped_column(String, deferred=True)
    lvo_salesorg: Mapped[str | None] = mapped_column(String, deferred=True)
    lvo_territorymovereason: Mapped[str | None] = mapped_column(String, deferred=True)
    lvo_geographicunit: Mapped[str | None] = mapped_column(String, deferred=True)
    lvo_salesoffice: Mapped[str | None] = mapped_column(String, deferred=True)


class Contact(Base):
    """Standard D365 contact table from lenovo_nitro_d365_postgres.sql.

    Only the columns the Deal-Detail endpoints read are mapped.

    NOTE: ``parentcustomerid`` is intentionally NOT mapped here — D365
    dumps disagree on whether the contact-to-account link column is
    ``parentcustomerid``, ``accountid`` or absent entirely. Mapping it
    would force every eager-load to ``SELECT contact.parentcustomerid``
    and fail on dumps that lack the column. The seed migrations
    (``sql/2026_06_create_*_contact.sql``) handle the variance via
    ``information_schema.columns`` introspection, so the orchestrator
    code never needs the column on the ORM side.
    """

    __tablename__ = "contact"

    contactid: Mapped[str] = mapped_column(String, primary_key=True)
    fullname: Mapped[str | None] = mapped_column(String)
    firstname: Mapped[str | None] = mapped_column(String)
    lastname: Mapped[str | None] = mapped_column(String)
    jobtitle: Mapped[str | None] = mapped_column(String)
    emailaddress1: Mapped[str | None] = mapped_column(String)
    statecode: Mapped[str | None] = mapped_column(String)


class OpportunityContact(Base):
    """Link table between an opportunity and a contact.

    Created by sql/2026_06_create_opportunity_contact.sql. Powers the
    "Decision maker" + "Additional contacts" panels on the Deal Detail view.
    """

    __tablename__ = "lvo_opportunitycontact"

    lvo_opportunitycontactid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_opportunityid: Mapped[str] = mapped_column(String)
    lvo_contactid: Mapped[str] = mapped_column(String)
    lvo_role: Mapped[str | None] = mapped_column(String)
    lvo_isdecisionmaker: Mapped[bool] = mapped_column(Boolean, default=False)
    lvo_lasttouchdate: Mapped[datetime | None] = mapped_column(DateTime)
    lvo_createdat: Mapped[datetime] = mapped_column(DateTime)
    lvo_updatedat: Mapped[datetime] = mapped_column(DateTime)
    statecode: Mapped[str] = mapped_column(String, default="Active")


class AccountContact(Base):
    """Link table between an account and a contact.

    Created by sql/2026_06_create_account_contact.sql. Powers the
    "Account contacts" CRUD panel on the View Account user story
    (Add / Update / Delete contacts independent of any specific deal).
    """

    __tablename__ = "lvo_accountcontact"

    lvo_accountcontactid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_accountid: Mapped[str] = mapped_column(String)
    lvo_contactid: Mapped[str] = mapped_column(String)
    lvo_role: Mapped[str | None] = mapped_column(String)
    lvo_isprimary: Mapped[bool] = mapped_column(Boolean, default=False)
    lvo_lasttouchdate: Mapped[datetime | None] = mapped_column(DateTime)
    lvo_createdat: Mapped[datetime] = mapped_column(DateTime)
    lvo_updatedat: Mapped[datetime] = mapped_column(DateTime)
    statecode: Mapped[str] = mapped_column(String, default="Active")


class OpportunityCompetitor(Base):
    __tablename__ = "lvo_opportunitycompetitor"

    lvo_opportunitycompetitorid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_name: Mapped[str | None] = mapped_column(String)
    lvo_opportunityid: Mapped[str | None] = mapped_column(String)
    lvo_competitorname: Mapped[str | None] = mapped_column(String)
    lvo_competitortype: Mapped[str | None] = mapped_column(String)
    lvo_resellingpartner: Mapped[str | None] = mapped_column(String)
    statecode: Mapped[str | None] = mapped_column(String)


class GeographicUnit(Base):
    __tablename__ = "lvo_geographicunit"

    lvo_geographicunitid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_name: Mapped[str | None] = mapped_column(String)
    lvo_geounitcode: Mapped[str | None] = mapped_column(String)
    lvo_parentunitid: Mapped[str | None] = mapped_column(String)
    statecode: Mapped[str | None] = mapped_column(String)


class LegalEntity(Base):
    __tablename__ = "lvo_legalentity"

    lvo_legalentityid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_name: Mapped[str | None] = mapped_column(String)
    lvo_countryid: Mapped[str | None] = mapped_column(String)
    lvo_bgid: Mapped[str | None] = mapped_column(String)
    statecode: Mapped[str | None] = mapped_column(String)


class Activity(Base):
    """Sales activity log per opportunity (powers the timeline + offcanvas)."""

    __tablename__ = "lvo_activity"

    lvo_activityid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_opportunityid: Mapped[str] = mapped_column(String)
    lvo_activitytype: Mapped[str] = mapped_column(String)
    lvo_direction: Mapped[str | None] = mapped_column(String)
    lvo_subject: Mapped[str | None] = mapped_column(String)
    lvo_body: Mapped[str | None] = mapped_column(String)
    lvo_activitydate: Mapped[datetime] = mapped_column(DateTime)
    lvo_groupedcount: Mapped[int | None] = mapped_column(Integer)
    owninguser: Mapped[str | None] = mapped_column(String)
    statecode: Mapped[str | None] = mapped_column(String)


class Quote(Base):
    __tablename__ = "quote"

    quoteid: Mapped[str] = mapped_column(String, primary_key=True)
    opportunityid: Mapped[str | None] = mapped_column(String)
    statecode: Mapped[str | None] = mapped_column(String)


class QuoteItem(Base):
    __tablename__ = "lvo_quoteitem"

    lvo_quoteitemid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_quoteid: Mapped[str | None] = mapped_column(String)
    lvo_productseries: Mapped[str | None] = mapped_column(String)
    statecode: Mapped[str | None] = mapped_column(String)


class SolutionOffering(Base):
    __tablename__ = "lvo_solutionoffering"

    lvo_solutionofferingid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_name: Mapped[str | None] = mapped_column(String)
    lvo_description: Mapped[str | None] = mapped_column(String)
    statecode: Mapped[str | None] = mapped_column(String)


class NextAction(Base):
    """Scheduled next action associated with an opportunity.

    Created by sql/2026_06_create_next_actions_audit.sql.
    """

    __tablename__ = "lvo_nextaction"

    lvo_nextactionid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_opportunityid: Mapped[str] = mapped_column(String)
    lvo_description: Mapped[str] = mapped_column(String)
    lvo_duedate: Mapped[date | None] = mapped_column(Date)
    verbal_written_acceptance: Mapped[str] = mapped_column(String)
    verbal_commit_date: Mapped[date | None] = mapped_column(Date)
    lvo_status: Mapped[str] = mapped_column(String, default="Open")
    lvo_createdat: Mapped[datetime] = mapped_column(DateTime)
    lvo_updatedat: Mapped[datetime] = mapped_column(DateTime)
    lvo_createdby: Mapped[str | None] = mapped_column(String)
    statecode: Mapped[str] = mapped_column(String, default="Active")

class FileNotes(Base):
    __tablename__ = "lvo_file_notes"

    id = Column(UUID(as_uuid=True), primary_key=True)
    opportunity_id = Column(UUID(as_uuid=True), nullable=False)
    notes = Column(Text)
    statecode = Column(Text)

class AuditLog(Base):
    """Immutable C/U/D event trail for deal-related entities.

    Created by sql/2026_06_create_next_actions_audit.sql; extended by
    sql/2026_14_audit_compliance.sql with actor/category/outcome metadata.
    lvo_diff stores a JSON string — before/after snapshot for updates,
    the created object for creates, and the entity id for deletes.
    """

    __tablename__ = "lvo_audit_log"

    lvo_auditlogid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_entitytype: Mapped[str] = mapped_column(String)
    lvo_entityid: Mapped[str] = mapped_column(String)
    lvo_opportunityid: Mapped[str | None] = mapped_column(String)
    lvo_action: Mapped[str] = mapped_column(String)
    lvo_changedby: Mapped[str | None] = mapped_column(String)
    lvo_changedat: Mapped[datetime] = mapped_column(DateTime)
    lvo_diff: Mapped[str | None] = mapped_column(String)
    lvo_actortype: Mapped[str | None] = mapped_column(String)
    lvo_category: Mapped[str | None] = mapped_column(String)
    lvo_outcome: Mapped[str] = mapped_column(String, default="success")
    lvo_correlationid: Mapped[str | None] = mapped_column(String)
    lvo_failurereason: Mapped[str | None] = mapped_column(String)
    lvo_eventtype: Mapped[str | None] = mapped_column(String)
    lvo_deliveryattempts: Mapped[int | None] = mapped_column(Integer)
    lvo_sourceservice: Mapped[str] = mapped_column(String, default="d365_sales")


class AuditConfig(Base):
    """Singleton runtime config for audit retention and read logging."""

    __tablename__ = "lvo_audit_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    retention_days: Mapped[int] = mapped_column(Integer, default=90)
    log_seller_reads: Mapped[bool] = mapped_column(Boolean, default=False)
    log_admin_reads: Mapped[bool] = mapped_column(Boolean, default=False)
    log_ai_output_reads: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    updated_by: Mapped[str | None] = mapped_column(String)


class DealRisk(Base):
    """Persisted output of the risk-derivation rules.

    Created by sql/2026_06_create_dealrisk.sql.  The deal-health recalc
    service rewrites all rows for a given deal in one transaction (delete
    Active rows, insert fresh ones) so callers always see a consistent set.
    """

    __tablename__ = "lvo_dealrisk"

    lvo_dealriskid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_opportunityid: Mapped[str] = mapped_column(String)
    lvo_riskcategory: Mapped[str] = mapped_column(String)
    lvo_riskname: Mapped[str] = mapped_column(String)
    lvo_message: Mapped[str] = mapped_column(String)
    lvo_detectedat: Mapped[datetime] = mapped_column(DateTime)
    statecode: Mapped[str] = mapped_column(String, default="Active")


class OpportunitySnapshot(Base):
    """Daily KPI-bucket aggregate snapshot.

    Created by sql/2026_06_create_opportunity_snapshot.sql. One row per
    (snapshot_date, bucket). Powers the period-over-period trend math
    on /api/opportunities/kpi-summary — see app/services/kpi_snapshots.py.
    """

    __tablename__ = "lvo_opportunitysnapshot"

    lvo_opportunitysnapshotid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_snapshotdate: Mapped[date] = mapped_column(Date)
    lvo_bucket: Mapped[str] = mapped_column(String)
    lvo_value: Mapped[float] = mapped_column(Numeric(20, 2), default=0)
    lvo_count: Mapped[int] = mapped_column(Integer, default=0)
    lvo_createdat: Mapped[datetime] = mapped_column(DateTime)


class AccountSnapshot(Base):
    """Daily Account-KPI bucket aggregate snapshot.

    Created by sql/2026_06_create_account_snapshot.sql. One row per
    (snapshot_date, bucket). Mirrors OpportunitySnapshot — the table is
    separate so the four account-side buckets (total / acv / active /
    at_risk) live in their own whitelist and never collide with the
    opportunity buckets.

    Powers the period-over-period trend math on
    /api/accounts/kpi-summary — see app/services/account_kpi_snapshots.py.
    """

    __tablename__ = "lvo_accountsnapshot"

    lvo_accountsnapshotid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_snapshotdate: Mapped[date] = mapped_column(Date)
    lvo_bucket: Mapped[str] = mapped_column(String)
    lvo_value: Mapped[float] = mapped_column(Numeric(20, 2), default=0)
    lvo_count: Mapped[int] = mapped_column(Integer, default=0)
    lvo_createdat: Mapped[datetime] = mapped_column(DateTime)


class DealHealthConfig(Base):
    """Single-row JSONB config table used by the deal-health calculator.

    Created (and seeded) by sql/2026_06_create_dealhealth_config.sql.
    The id column is always 1 — the helper in the service layer always
    selects WHERE id = 1.
    """

    __tablename__ = "lvo_dealhealthconfig"

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    lvo_settings: Mapped[dict] = mapped_column(JSON)
    lvo_updatedat: Mapped[datetime] = mapped_column(DateTime)


class NotificationRead(Base):
    """Read-state for Sprint 2 What Changed notification panel items.

    Created by sql/2026_07_create_lvo_notification_read.sql. Feed events are
    aggregated at query time; this table only records which synthetic
    ``feed_item_key`` values a seller has dismissed/marked read.
    """

    __tablename__ = "lvo_notification_read"

    lvo_notificationreadid: Mapped[str] = mapped_column(String, primary_key=True)
    seller_id: Mapped[str] = mapped_column(String)
    feed_item_key: Mapped[str] = mapped_column(String)
    read_at: Mapped[datetime] = mapped_column(DateTime)


class SellerQuota(Base):
    """Per-seller fiscal-quarter quota target for Quarter Pulse (US 1.2).

    Created by sql/2026_08_create_lvo_seller_quota.sql. Holds manual
    overrides when D365 has no goal configured; ``source='d365'`` reserved
    for future mirror imports.
    """

    __tablename__ = "lvo_seller_quota"

    lvo_sellerquotaid: Mapped[str] = mapped_column(String, primary_key=True)
    seller_id: Mapped[str] = mapped_column(String)
    fiscal_year: Mapped[int] = mapped_column(Integer)
    fiscal_quarter: Mapped[int] = mapped_column(Integer)
    quota_amount: Mapped[float] = mapped_column(Numeric(20, 2))
    currency_code: Mapped[str] = mapped_column(String, default="USD")
    source: Mapped[str] = mapped_column(String, default="manual")
    set_by: Mapped[str | None] = mapped_column(String)
    modified_at: Mapped[datetime] = mapped_column(DateTime)


class SomInterviewQuestion(Base):
    """Interview question bank for Sales Operating Model setup (US 3.2.1)."""

    __tablename__ = "lvo_som_interview_question"

    lvo_questionid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_role: Mapped[str] = mapped_column(String)
    lvo_sortorder: Mapped[int] = mapped_column(SmallInteger)
    lvo_questiontext: Mapped[str] = mapped_column(Text)
    lvo_isactive: Mapped[bool] = mapped_column(default=True)
    lvo_createdat: Mapped[datetime] = mapped_column(DateTime)
    lvo_updatedat: Mapped[datetime] = mapped_column(DateTime)


class SomConfigurationCycle(Base):
    __tablename__ = "lvo_som_configuration_cycle"

    lvo_cycleid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_status: Mapped[str] = mapped_column(String, default="in_progress")
    lvo_createdat: Mapped[datetime] = mapped_column(DateTime)
    lvo_configuredat: Mapped[datetime | None] = mapped_column(DateTime)
    lvo_configuredby: Mapped[str | None] = mapped_column(String)


class SomInterviewResponse(Base):
    __tablename__ = "lvo_som_interview_response"

    lvo_responseid: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_cycleid: Mapped[str] = mapped_column(String)
    lvo_questionid: Mapped[str] = mapped_column(String)
    lvo_role: Mapped[str] = mapped_column(String)
    lvo_responsetext: Mapped[str] = mapped_column(Text, default="")
    lvo_status: Mapped[str] = mapped_column(String, default="draft")
    lvo_capturedby: Mapped[str | None] = mapped_column(String)
    lvo_capturedat: Mapped[datetime] = mapped_column(DateTime)
    lvo_savedat: Mapped[datetime | None] = mapped_column(DateTime)


class SomIntentCard(Base):
    __tablename__ = "lvo_som_intent_card"

    lvo_role: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_status: Mapped[str] = mapped_column(String, default="NOT_CONFIGURED")
    lvo_configuredat: Mapped[datetime | None] = mapped_column(DateTime)
    lvo_configuredby: Mapped[str | None] = mapped_column(String)
    lvo_cycleid: Mapped[str | None] = mapped_column(String)


class SomContextLake(Base):
    __tablename__ = "lvo_som_context_lake"

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    lvo_payload: Mapped[dict] = mapped_column(JSON)
    lvo_updatedat: Mapped[datetime] = mapped_column(DateTime)
    lvo_updatedby: Mapped[str | None] = mapped_column(String)


class SomOrganizationalIntent(Base):
    """Organizational intent cards — Outcome, Motion, Focus, etc. (US 3.2.2)."""

    __tablename__ = "lvo_som_organizational_intent"

    lvo_intenttype: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_displayname: Mapped[str] = mapped_column(String)
    lvo_status: Mapped[str] = mapped_column(String, default="NOT_CONFIGURED")
    lvo_fields: Mapped[dict] = mapped_column(JSON, default=dict)
    lvo_is_timeboxed: Mapped[bool] = mapped_column(Boolean, default=False)
    lvo_is_guardrail: Mapped[bool] = mapped_column(Boolean, default=False)
    lvo_expiry_date: Mapped[date | None] = mapped_column(Date)
    lvo_last_synced_at: Mapped[datetime | None] = mapped_column(DateTime)
    lvo_configured_by: Mapped[str | None] = mapped_column(String)
    lvo_cycleid: Mapped[str | None] = mapped_column(String)


class SomTimelineClassification(Base):
    """Timeline classification cards — tempo, anchors, signals, etc. (US 3.4.1)."""

    __tablename__ = "lvo_som_timeline_classification"

    lvo_cardtype: Mapped[str] = mapped_column(String, primary_key=True)
    lvo_displayname: Mapped[str] = mapped_column(String)
    lvo_status: Mapped[str] = mapped_column(String, default="NOT_CONFIGURED")
    lvo_fields: Mapped[dict] = mapped_column(JSON, default=dict)
    lvo_last_synced_at: Mapped[datetime | None] = mapped_column(DateTime)
    lvo_configured_by: Mapped[str | None] = mapped_column(String)
    lvo_cycleid: Mapped[str | None] = mapped_column(String)


__all__ = [
    "Account",
    "AccountContact",
    "AccountSnapshot",
    "Activity",
    "AuditLog",
    "AuditConfig",
    "Contact",
    "DealHealthConfig",
    "DealRisk",
    "GeographicUnit",
    "LegalEntity",
    "NextAction",
    "Opportunity",
    "OpportunityCompetitor",
    "OpportunityContact",
    "OpportunitySnapshot",
    "Quote",
    "QuoteItem",
    "SellerQuota",
    "SomConfigurationCycle",
    "SomContextLake",
    "SomIntentCard",
    "SomOrganizationalIntent",
    "SomTimelineClassification",
    "SomInterviewQuestion",
    "SomInterviewResponse",
    "SolutionOffering",
]

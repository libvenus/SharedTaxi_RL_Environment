-- ============================================================================
-- account customer-information schema — adds the Lenovo-custom columns the
-- "Customer Information" tab needs that are NOT present in a vanilla D365 dump.
--
-- The standard D365 columns the endpoint reads (address1_*, address2_*,
-- telephone1, websiteurl, paymenttermscode, defaultpricelevelid,
-- transactioncurrencyid, territoryid, owninguser, createdby, etc.) are NOT
-- created here — they are part of the base D365 schema. The router uses
-- runtime introspection (information_schema.columns) and SQLAlchemy ``defer``
-- so any of them can be missing on a stripped dump without breaking the page.
--
-- Re-running this migration is safe — every statement is idempotent.
--
-- Phase 1 user story: "View Customer Information" (read-only).
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Basic Information — fields that don't have a standard D365 home.
-- ----------------------------------------------------------------------------
ALTER TABLE account
    ADD COLUMN IF NOT EXISTS lvo_subsegment       TEXT,
    ADD COLUMN IF NOT EXISTS lvo_gtmsegment       TEXT,
    ADD COLUMN IF NOT EXISTS lvo_sellerknownas    TEXT;

-- ----------------------------------------------------------------------------
-- Identity & Legal.
-- ----------------------------------------------------------------------------
ALTER TABLE account
    ADD COLUMN IF NOT EXISTS lvo_legalnamelocal   TEXT,
    ADD COLUMN IF NOT EXISTS lvo_locallanguage    TEXT,
    ADD COLUMN IF NOT EXISTS lvo_alias            TEXT,
    ADD COLUMN IF NOT EXISTS lvo_taxvatnumber     TEXT,
    ADD COLUMN IF NOT EXISTS lvo_legalentity      TEXT,
    ADD COLUMN IF NOT EXISTS lvo_linkedinurl      TEXT;

-- ----------------------------------------------------------------------------
-- Commercial Terms.
--
-- ``defaultcurrency`` already exists (account.lvo_defaultcurrency from the
-- base dump). PaymentTerms and PriceList come from the standard D365 columns
-- ``paymenttermscode`` / ``defaultpricelevelid`` and need no migration.
-- ----------------------------------------------------------------------------
ALTER TABLE account
    ADD COLUMN IF NOT EXISTS lvo_dealsignconfig   TEXT;

-- ----------------------------------------------------------------------------
-- Territory & Ownership — the extra Lenovo-specific dimensions on top of
-- the standard ``territoryid`` / ``owninguser`` columns.
-- ----------------------------------------------------------------------------
ALTER TABLE account
    ADD COLUMN IF NOT EXISTS lvo_salesterritory       TEXT,
    ADD COLUMN IF NOT EXISTS lvo_futureterritory      TEXT,
    ADD COLUMN IF NOT EXISTS lvo_salesorg             TEXT,
    ADD COLUMN IF NOT EXISTS lvo_territorymovereason  TEXT,
    ADD COLUMN IF NOT EXISTS lvo_geographicunit       TEXT,
    ADD COLUMN IF NOT EXISTS lvo_salesoffice          TEXT;

-- ----------------------------------------------------------------------------
-- Lightweight indexes to support filter/search use cases the FE may add later.
-- ----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_account_lvo_subsegment
    ON account (lvo_subsegment);
CREATE INDEX IF NOT EXISTS idx_account_lvo_gtmsegment
    ON account (lvo_gtmsegment);
CREATE INDEX IF NOT EXISTS idx_account_lvo_legalentity
    ON account (lvo_legalentity);
CREATE INDEX IF NOT EXISTS idx_account_lvo_salesterritory
    ON account (lvo_salesterritory);
CREATE INDEX IF NOT EXISTS idx_account_lvo_geographicunit
    ON account (lvo_geographicunit);

-- ----------------------------------------------------------------------------
-- Sanity check helpers (uncomment locally if you want to verify):
-- ----------------------------------------------------------------------------
-- SELECT column_name
--   FROM information_schema.columns
--  WHERE table_name = 'account'
--    AND column_name LIKE 'lvo_%'
--  ORDER BY column_name;

-- ============================================================================
-- Audit Log & Compliance (US — tamper-evident platform audit trail)
--
-- Extends lvo_audit_log with compliance metadata, adds runtime config table,
-- and enforces immutability (UPDATE blocked; DELETE only via retention purge).
--
-- Depends on: sql/2026_06_create_next_actions_audit.sql
-- Re-running is safe.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Extend lvo_audit_log
-- ----------------------------------------------------------------------------
ALTER TABLE lvo_audit_log
    ADD COLUMN IF NOT EXISTS lvo_actortype TEXT,
    ADD COLUMN IF NOT EXISTS lvo_category TEXT,
    ADD COLUMN IF NOT EXISTS lvo_outcome TEXT NOT NULL DEFAULT 'success',
    ADD COLUMN IF NOT EXISTS lvo_correlationid TEXT,
    ADD COLUMN IF NOT EXISTS lvo_failurereason TEXT,
    ADD COLUMN IF NOT EXISTS lvo_eventtype TEXT,
    ADD COLUMN IF NOT EXISTS lvo_deliveryattempts INT,
    ADD COLUMN IF NOT EXISTS lvo_sourceservice TEXT NOT NULL DEFAULT 'd365_sales';

-- Backfill legacy rows
UPDATE lvo_audit_log
   SET lvo_actortype = COALESCE(lvo_actortype, 'seller'),
       lvo_category = COALESCE(lvo_category, 'crm_writeback'),
       lvo_outcome = COALESCE(lvo_outcome, 'success'),
       lvo_sourceservice = COALESCE(lvo_sourceservice, 'd365_sales')
 WHERE lvo_actortype IS NULL
    OR lvo_category IS NULL
    OR lvo_sourceservice IS NULL;

-- Widen action whitelist
ALTER TABLE lvo_audit_log DROP CONSTRAINT IF EXISTS chk_lvo_audit_log_action;
ALTER TABLE lvo_audit_log
    ADD CONSTRAINT chk_lvo_audit_log_action
    CHECK (lvo_action IN (
        'create', 'update', 'delete', 'read',
        'approve', 'dismiss', 'generate', 'deliver',
        'retry', 'dead_letter', 'recalculate', 'resolve'
    ));

-- Widen entity-type whitelist
ALTER TABLE lvo_audit_log DROP CONSTRAINT IF EXISTS chk_lvo_audit_log_entitytype;
ALTER TABLE lvo_audit_log
    ADD CONSTRAINT chk_lvo_audit_log_entitytype
    CHECK (lvo_entitytype IN (
        'opportunity', 'competitor', 'next_action',
        'opportunity_contact', 'account_contact', 'file_notes',
        'som_organizational_intent', 'som_timeline_classification',
        'som_interview_setup', 'seller_quota', 'account',
        'meeting_prep_task', 'meeting_prep_note', 'meeting_briefing', 'data_task', 'todo',
        'meeting_bot', 'transcript', 'post_meeting_summary',
        'notification', 'event_spine', 'audit_config', 'deal_health', 'api_request'
    ));

ALTER TABLE lvo_audit_log DROP CONSTRAINT IF EXISTS chk_lvo_audit_log_actortype;
ALTER TABLE lvo_audit_log
    ADD CONSTRAINT chk_lvo_audit_log_actortype
    CHECK (lvo_actortype IN ('seller', 'admin', 'ai', 'system', 'event_spine'));

ALTER TABLE lvo_audit_log DROP CONSTRAINT IF EXISTS chk_lvo_audit_log_category;
ALTER TABLE lvo_audit_log
    ADD CONSTRAINT chk_lvo_audit_log_category
    CHECK (lvo_category IN (
        'seller_action', 'admin_action', 'ai_automated',
        'crm_writeback', 'event_spine', 'read_action', 'system'
    ));

ALTER TABLE lvo_audit_log DROP CONSTRAINT IF EXISTS chk_lvo_audit_log_outcome;
ALTER TABLE lvo_audit_log
    ADD CONSTRAINT chk_lvo_audit_log_outcome
    CHECK (lvo_outcome IN ('success', 'failure'));

CREATE INDEX IF NOT EXISTS idx_lvo_audit_log_correlationid
    ON lvo_audit_log (lvo_correlationid)
    WHERE lvo_correlationid IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_lvo_audit_log_category_changedat
    ON lvo_audit_log (lvo_category, lvo_changedat DESC);

-- ----------------------------------------------------------------------------
-- Runtime config (toggle read logging + retention without code deploy)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lvo_audit_config (
    id                  INT PRIMARY KEY DEFAULT 1,
    retention_days      INT NOT NULL DEFAULT 90,
    log_seller_reads    BOOLEAN NOT NULL DEFAULT FALSE,
    log_admin_reads     BOOLEAN NOT NULL DEFAULT FALSE,
    log_ai_output_reads BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by          TEXT,
    CONSTRAINT chk_lvo_audit_config_singleton CHECK (id = 1),
    CONSTRAINT chk_lvo_audit_config_retention CHECK (retention_days BETWEEN 1 AND 3650)
);

INSERT INTO lvo_audit_config (id, retention_days)
VALUES (1, 90)
ON CONFLICT (id) DO NOTHING;

-- ----------------------------------------------------------------------------
-- Immutability: block UPDATE; DELETE only when audit.purge_mode = 'on'
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION trg_lvo_audit_log_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION 'lvo_audit_log records are immutable (UPDATE not permitted)';
    ELSIF TG_OP = 'DELETE' THEN
        IF current_setting('audit.purge_mode', true) IS DISTINCT FROM 'on' THEN
            RAISE EXCEPTION
                'lvo_audit_log records cannot be deleted except via retention purge';
        END IF;
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_lvo_audit_log_immutable ON lvo_audit_log;
CREATE TRIGGER trg_lvo_audit_log_immutable
    BEFORE UPDATE OR DELETE ON lvo_audit_log
    FOR EACH ROW
    EXECUTE FUNCTION trg_lvo_audit_log_immutable();

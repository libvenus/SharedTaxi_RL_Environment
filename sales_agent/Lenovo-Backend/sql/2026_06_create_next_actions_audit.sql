-- ============================================================================
-- Next Actions + Audit Log
--
-- lvo_nextaction  — per-opportunity action items with Open / Completed status.
-- lvo_audit_log   — immutable C/U/D event trail for opportunities, competitors
--                   and next actions.
--
-- Run this against the same database that holds `opportunity`.
-- Re-running is safe: all CREATE statements use IF NOT EXISTS.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- lvo_nextaction
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lvo_nextaction (
    lvo_nextactionid    TEXT PRIMARY KEY,
    lvo_opportunityid   TEXT NOT NULL,
    lvo_description     TEXT NOT NULL,
    lvo_duedate         DATE,
    lvo_status          TEXT NOT NULL DEFAULT 'Open',
    lvo_createdat       TIMESTAMPTZ NOT NULL DEFAULT now(),
    lvo_updatedat       TIMESTAMPTZ NOT NULL DEFAULT now(),
    lvo_createdby       TEXT,
    statecode           TEXT NOT NULL DEFAULT 'Active'
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'chk_lvo_nextaction_status'
    ) THEN
        ALTER TABLE lvo_nextaction
            ADD CONSTRAINT chk_lvo_nextaction_status
            CHECK (lvo_status IN ('Open', 'Completed'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_lvo_nextaction_opportunityid
    ON lvo_nextaction (lvo_opportunityid);

-- ----------------------------------------------------------------------------
-- lvo_audit_log
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lvo_audit_log (
    lvo_auditlogid      TEXT PRIMARY KEY,
    lvo_entitytype      TEXT NOT NULL,
    lvo_entityid        TEXT NOT NULL,
    lvo_opportunityid   TEXT,
    lvo_action          TEXT NOT NULL,
    lvo_changedby       TEXT,
    lvo_changedat       TIMESTAMPTZ NOT NULL DEFAULT now(),
    lvo_diff            JSONB
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'chk_lvo_audit_log_action'
    ) THEN
        ALTER TABLE lvo_audit_log
            ADD CONSTRAINT chk_lvo_audit_log_action
            CHECK (lvo_action IN ('create', 'update', 'delete'));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'chk_lvo_audit_log_entitytype'
    ) THEN
        -- Widened whitelist (originally just opportunity/competitor/next_action;
        -- contact entities were added by later stories — see
        -- sql/2026_06_widen_audit_log_entitytypes.sql for the in-place migration
        -- that retrofits this onto already-applied DBs).
        ALTER TABLE lvo_audit_log
            ADD CONSTRAINT chk_lvo_audit_log_entitytype
            CHECK (lvo_entitytype IN (
                'opportunity',
                'competitor',
                'next_action',
                'opportunity_contact',
                'account_contact'
            ));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_lvo_audit_log_entity
    ON lvo_audit_log (lvo_entitytype, lvo_entityid);

CREATE INDEX IF NOT EXISTS idx_lvo_audit_log_opportunityid
    ON lvo_audit_log (lvo_opportunityid);

CREATE INDEX IF NOT EXISTS idx_lvo_audit_log_changedat
    ON lvo_audit_log (lvo_changedat DESC);

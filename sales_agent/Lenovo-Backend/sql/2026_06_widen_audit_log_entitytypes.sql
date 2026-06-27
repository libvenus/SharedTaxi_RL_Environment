-- ============================================================================
-- Widen lvo_audit_log.chk_lvo_audit_log_entitytype to cover contact entities
--
-- The original CHECK constraint, created by
-- ``sql/2026_06_create_next_actions_audit.sql``, only whitelists:
--     'opportunity', 'competitor', 'next_action'
--
-- Subsequent stories started writing two more entity types that were never
-- added to the whitelist:
--     'opportunity_contact'  — Deal Detail View (app/routers/contacts.py)
--     'account_contact'      — View Account / Manage Contacts
--                              (app/routers/account_contacts.py)
--
-- DBs that already have the CHECK reject those inserts with
--     CheckViolation: chk_lvo_audit_log_entitytype
-- and the surrounding API call fails with a 500.
--
-- This migration drops the old constraint (if present) and re-creates it
-- with the widened whitelist. Re-runnable safely.
-- ============================================================================

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'chk_lvo_audit_log_entitytype'
    ) THEN
        ALTER TABLE lvo_audit_log
            DROP CONSTRAINT chk_lvo_audit_log_entitytype;
    END IF;

    ALTER TABLE lvo_audit_log
        ADD CONSTRAINT chk_lvo_audit_log_entitytype
        CHECK (lvo_entitytype IN (
            'opportunity',
            'competitor',
            'next_action',
            'opportunity_contact',
            'account_contact'
        ));
END $$;


-- ----------------------------------------------------------------------------
-- Sanity check (uncomment to verify locally):
-- ----------------------------------------------------------------------------
-- SELECT conname, pg_get_constraintdef(oid)
--   FROM pg_constraint
--  WHERE conname = 'chk_lvo_audit_log_entitytype';

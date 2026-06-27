-- =============================================================================
-- Sprint 1A · User Story 01 — Joining the Meetings
-- =============================================================================
-- Adds bot-lifecycle tracking + CRM-link columns to tbl_schedule_meetings.
--
-- The Note-Taking Agent (other team) needs to record what's happening with
-- each scheduled meeting (pending -> joined / cancelled / rescheduled / etc.)
-- and writes back the opportunity / account it resolved the meeting to (via
-- the D365 backend's POST /api/meetings/resolve-opportunity endpoint).
--
-- All statements are idempotent: this migration is safe to re-run.
-- =============================================================================

BEGIN;

-- 1. Lifecycle state ----------------------------------------------------------
ALTER TABLE tbl_schedule_meetings
    ADD COLUMN IF NOT EXISTS bot_status TEXT NOT NULL DEFAULT 'pending';

ALTER TABLE tbl_schedule_meetings
    ADD COLUMN IF NOT EXISTS bot_status_reason TEXT NULL;

ALTER TABLE tbl_schedule_meetings
    ADD COLUMN IF NOT EXISTS bot_last_event_at TIMESTAMPTZ NULL;

-- 2. CRM linkage --------------------------------------------------------------
-- Resolved by the D365 backend via attendee-email + subject matching.
-- Both nullable: bot may schedule a meeting before resolution succeeds, and
-- the resolver may legitimately fail to find a match (unscoped meeting).
ALTER TABLE tbl_schedule_meetings
    ADD COLUMN IF NOT EXISTS opportunity_id UUID NULL;

ALTER TABLE tbl_schedule_meetings
    ADD COLUMN IF NOT EXISTS account_id UUID NULL;

-- 3. CHECK constraint on bot_status ------------------------------------------
-- Drop-and-recreate pattern keeps the constraint in sync with the whitelist
-- if we ever expand it (just edit this migration and re-run).
ALTER TABLE tbl_schedule_meetings
    DROP CONSTRAINT IF EXISTS chk_bot_status;

ALTER TABLE tbl_schedule_meetings
    ADD CONSTRAINT chk_bot_status CHECK (
        bot_status IN (
            'pending',          -- bot has been told about the meeting, not yet scheduled
            'scheduled',        -- bot has scheduled itself to join at start_time
            'joining',          -- bot is actively joining (transient, may or may not be observed)
            'joined',           -- bot is in the meeting
            'lobby_waiting',    -- customer-organized; bot is waiting for admit
            'cancelled',        -- meeting cancelled before/during; bot will not / did not join
            'rescheduled',      -- meeting was moved; bot has re-scheduled itself
            'failed'            -- bot tried to join but errored (network, auth, removed by host, etc.)
        )
    );

-- 4. Indexes ------------------------------------------------------------------
-- The bot's most common query is "give me everything not yet joined / cancelled".
CREATE INDEX IF NOT EXISTS idx_schedule_meetings_bot_status
    ON tbl_schedule_meetings (bot_status);

-- Lookup by D365 deal — the FE will call /meeting-details for a given opportunity.
CREATE INDEX IF NOT EXISTS idx_schedule_meetings_opportunity_id
    ON tbl_schedule_meetings (opportunity_id)
    WHERE opportunity_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_schedule_meetings_account_id
    ON tbl_schedule_meetings (account_id)
    WHERE account_id IS NOT NULL;

COMMIT;

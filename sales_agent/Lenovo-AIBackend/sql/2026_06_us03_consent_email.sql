-- =============================================================================
-- Sprint 1A · User Story 03 — Consent Capture
-- =============================================================================
-- Adds one new table for the pre-meeting consent email pipeline:
--
--   tbl_meeting_consent_email — one row per (meeting_id, recipient_email)
--                              tracks scheduled send, delivery status, retries,
--                              opt-out events, and seller-notification stamp.
--
-- Linkage:
--   tbl_meeting_consent_email.meeting_id  → tbl_schedule_meetings.meeting_id
--
-- Cascade contract (handled in app/services/consent_email_service.py):
--   On opt-out, the parent tbl_schedule_meetings row is updated via
--   meeting_details_service.update_meeting_status() to:
--     bot_status        = 'cancelled'
--     bot_status_reason = 'participant_opted_out'
--   This keeps the audit trail unified with US01's lifecycle.
--
-- All statements are idempotent: this migration is safe to re-run.
-- =============================================================================

BEGIN;

-- 1. Consent email + opt-out tracking ----------------------------------------
CREATE TABLE IF NOT EXISTS tbl_meeting_consent_email (
    consent_id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id          UUID         NOT NULL,
    recipient_email     TEXT         NOT NULL,
    recipient_name      TEXT         NULL,

    opt_out_token       TEXT         NOT NULL,

    scheduled_send_at   TIMESTAMPTZ  NOT NULL,
    attempt_count       INT          NOT NULL DEFAULT 0,
    last_attempt_at     TIMESTAMPTZ  NULL,
    next_retry_at       TIMESTAMPTZ  NULL,

    delivery_status     TEXT         NOT NULL DEFAULT 'pending',
    failure_reason      TEXT         NULL,

    opted_out_at        TIMESTAMPTZ  NULL,
    opt_out_ip          TEXT         NULL,

    seller_notified_at  TIMESTAMPTZ  NULL,

    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Delivery-status whitelist. Drop-and-recreate so adding a new value later
-- is just an edit-and-re-run of this migration.
ALTER TABLE tbl_meeting_consent_email
    DROP CONSTRAINT IF EXISTS chk_consent_delivery_status;
ALTER TABLE tbl_meeting_consent_email
    ADD CONSTRAINT chk_consent_delivery_status CHECK (
        delivery_status IN (
            'pending',                  -- row created; bot hasn't tried sending yet
            'sent',                     -- delivery succeeded
            'failed',                   -- last attempt failed; may retry
            'fallback_to_in_meeting'    -- gave up after retries; bot uses US02 chat msg
        )
    );

-- attempt_count caps at 2 (initial + 1 retry), per AC #10. Defensive guard
-- so a buggy bot can't put us in attempt_count=99.
ALTER TABLE tbl_meeting_consent_email
    DROP CONSTRAINT IF EXISTS chk_consent_attempt_count;
ALTER TABLE tbl_meeting_consent_email
    ADD CONSTRAINT chk_consent_attempt_count CHECK (
        attempt_count >= 0 AND attempt_count <= 2
    );

-- A recipient_email must be present and roughly email-shaped. We don't enforce
-- full RFC 5322 — Pydantic on the request layer does sane validation; this is
-- just a last-line "no whitespace, has @" guard.
ALTER TABLE tbl_meeting_consent_email
    DROP CONSTRAINT IF EXISTS chk_consent_recipient_email;
ALTER TABLE tbl_meeting_consent_email
    ADD CONSTRAINT chk_consent_recipient_email CHECK (
        position('@' IN recipient_email) > 1
        AND recipient_email NOT LIKE '% %'
    );

-- One consent record per (meeting, recipient). Re-calling /schedule must be
-- idempotent — same recipient must not get two tokens for the same meeting.
CREATE UNIQUE INDEX IF NOT EXISTS uq_consent_meeting_recipient
    ON tbl_meeting_consent_email (meeting_id, recipient_email);

-- Tokens MUST be globally unique — they're the primary auth surface of the
-- public opt-out endpoint. Collisions would leak access between meetings.
CREATE UNIQUE INDEX IF NOT EXISTS uq_consent_opt_out_token
    ON tbl_meeting_consent_email (opt_out_token);

-- Retry queue. Partial index keeps it tiny and fast — only rows currently
-- waiting for their second send attempt land here.
CREATE INDEX IF NOT EXISTS idx_consent_retry_due
    ON tbl_meeting_consent_email (next_retry_at)
    WHERE delivery_status = 'failed' AND attempt_count < 2;

-- "Has anyone opted out of this meeting?" — the bot reads this right before
-- joining. Partial keeps it tight to actually-opted-out rows.
CREATE INDEX IF NOT EXISTS idx_consent_opt_out_meeting
    ON tbl_meeting_consent_email (meeting_id)
    WHERE opted_out_at IS NOT NULL;

-- All-records-for-a-meeting (audit / FE list view).
CREATE INDEX IF NOT EXISTS idx_consent_meeting
    ON tbl_meeting_consent_email (meeting_id);

COMMIT;

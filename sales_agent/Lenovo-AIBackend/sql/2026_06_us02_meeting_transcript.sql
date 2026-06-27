-- =============================================================================
-- Sprint 1A · User Story 02 — Consent & Recording
-- =============================================================================
-- Adds two tables for the Note-Taking Agent's transcript pipeline:
--
--   tbl_meeting_transcript          — one row per meeting (metadata, status,
--                                     consent proof, finalisation info)
--   tbl_meeting_transcript_segment  — one row per utterance (speaker,
--                                     text, timestamps, confidence)
--
-- Linkage:
--   tbl_meeting_transcript.meeting_id      → tbl_schedule_meetings.meeting_id
--   tbl_meeting_transcript.opportunity_id  → D365 opportunity (denormalised
--                                            from the meeting row for fast
--                                            opportunity-scoped reads)
--
-- All statements are idempotent: this migration is safe to re-run.
-- =============================================================================

BEGIN;

-- 1. Transcript metadata table -----------------------------------------------
CREATE TABLE IF NOT EXISTS tbl_meeting_transcript (
    transcript_id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id               UUID         NOT NULL,
    opportunity_id           UUID         NULL,
    account_id               UUID         NULL,

    status                   TEXT         NOT NULL DEFAULT 'in_progress',
    consent_message_text     TEXT         NOT NULL,
    consent_sent_at          TIMESTAMPTZ  NOT NULL,

    overall_confidence_score NUMERIC(4,3) NULL,
    segment_count            INT          NOT NULL DEFAULT 0,

    terminated_reason        TEXT         NULL,
    started_at               TIMESTAMPTZ  NOT NULL,
    finalized_at             TIMESTAMPTZ  NULL,

    created_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Status whitelist — drop-and-recreate so the constraint stays in sync if we
-- expand the set later (just edit this migration and re-run).
ALTER TABLE tbl_meeting_transcript
    DROP CONSTRAINT IF EXISTS chk_transcript_status;
ALTER TABLE tbl_meeting_transcript
    ADD CONSTRAINT chk_transcript_status CHECK (
        status IN (
            'in_progress',          -- bot is actively capturing utterances
            'finalized',            -- meeting ended cleanly; overall_confidence_score set
            'terminated_partial'    -- organiser removed bot, all-left, or bot crashed
        )
    );

-- Termination-reason whitelist (only meaningful when status != 'in_progress').
ALTER TABLE tbl_meeting_transcript
    DROP CONSTRAINT IF EXISTS chk_transcript_terminated_reason;
ALTER TABLE tbl_meeting_transcript
    ADD CONSTRAINT chk_transcript_terminated_reason CHECK (
        terminated_reason IS NULL
        OR terminated_reason IN (
            'meeting_ended',        -- natural end (everyone left / scheduled end time)
            'organizer_removed',    -- organiser kicked the bot
            'all_left',             -- all human participants left, bot remained
            'bot_failure'           -- bot crashed / unrecoverable error
        )
    );

-- Confidence range guard (nullable, but if set must be in [0, 1]).
ALTER TABLE tbl_meeting_transcript
    DROP CONSTRAINT IF EXISTS chk_transcript_overall_confidence;
ALTER TABLE tbl_meeting_transcript
    ADD CONSTRAINT chk_transcript_overall_confidence CHECK (
        overall_confidence_score IS NULL
        OR (overall_confidence_score >= 0 AND overall_confidence_score <= 1)
    );

-- One transcript per meeting — bot can't accidentally create a duplicate.
CREATE UNIQUE INDEX IF NOT EXISTS idx_transcript_meeting
    ON tbl_meeting_transcript (meeting_id);

-- Sweepers / dashboards need fast access to "still in progress".
CREATE INDEX IF NOT EXISTS idx_transcript_status_in_progress
    ON tbl_meeting_transcript (started_at)
    WHERE status = 'in_progress';

-- Opportunity-scoped reads (FE Activity tab eventually).
CREATE INDEX IF NOT EXISTS idx_transcript_opportunity
    ON tbl_meeting_transcript (opportunity_id)
    WHERE opportunity_id IS NOT NULL;


-- 2. Segment table -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS tbl_meeting_transcript_segment (
    segment_id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    transcript_id       UUID         NOT NULL,
    meeting_id          UUID         NOT NULL,

    speaker_name        TEXT         NOT NULL,
    speaker_email       TEXT         NULL,
    speaker_role        TEXT         NULL,
    speaker_contact_id  UUID         NULL,

    utterance_text      TEXT         NOT NULL,
    start_time          TIMESTAMPTZ  NOT NULL,
    end_time            TIMESTAMPTZ  NOT NULL,
    confidence_score    NUMERIC(4,3) NOT NULL,

    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- FK to transcript (application-managed; defer DB-level FK to keep the
-- migration safe to re-run if rows already exist).
ALTER TABLE tbl_meeting_transcript_segment
    DROP CONSTRAINT IF EXISTS chk_segment_confidence;
ALTER TABLE tbl_meeting_transcript_segment
    ADD CONSTRAINT chk_segment_confidence CHECK (
        confidence_score >= 0 AND confidence_score <= 1
    );

-- start_time must precede end_time (sanity check on bot input).
ALTER TABLE tbl_meeting_transcript_segment
    DROP CONSTRAINT IF EXISTS chk_segment_time_order;
ALTER TABLE tbl_meeting_transcript_segment
    ADD CONSTRAINT chk_segment_time_order CHECK (
        start_time <= end_time
    );

-- Primary read path: "give me everything for meeting X, ordered by time".
CREATE INDEX IF NOT EXISTS idx_segment_meeting_time
    ON tbl_meeting_transcript_segment (meeting_id, start_time);

-- Secondary: walk a single transcript directly.
CREATE INDEX IF NOT EXISTS idx_segment_transcript
    ON tbl_meeting_transcript_segment (transcript_id);

-- "What did this CRM contact say across all meetings?" — defer use case but
-- the partial index is cheap.
CREATE INDEX IF NOT EXISTS idx_segment_contact
    ON tbl_meeting_transcript_segment (speaker_contact_id)
    WHERE speaker_contact_id IS NOT NULL;

COMMIT;

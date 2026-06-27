-- Sprint 2 · US — Pre-Meeting Briefing Card (AIBackend)
-- Run on AIBackend Postgres. Idempotent.

BEGIN;

ALTER TABLE tbl_schedule_meetings
    ADD COLUMN IF NOT EXISTS seller_id UUID NULL;

CREATE INDEX IF NOT EXISTS idx_schedule_meetings_seller_id
    ON tbl_schedule_meetings (seller_id);

-- Cached AI briefing payload per meeting (regenerated on open when stale/missing).
CREATE TABLE IF NOT EXISTS tbl_meeting_briefing (
    id              SERIAL PRIMARY KEY,
    meeting_id      TEXT NOT NULL,
    seller_id       UUID NOT NULL,
    payload         JSONB NOT NULL,
    generated_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    payload_version TEXT NOT NULL DEFAULT 'v1'
);

CREATE INDEX IF NOT EXISTS idx_meeting_briefing_meeting_seller
    ON tbl_meeting_briefing (meeting_id, seller_id);

CREATE INDEX IF NOT EXISTS idx_meeting_briefing_generated_at
    ON tbl_meeting_briefing (generated_at DESC);

-- Prep tasks generated from briefing signals (completion tracked per seller).
CREATE TABLE IF NOT EXISTS tbl_meeting_prep_task (
    id              SERIAL PRIMARY KEY,
    meeting_id      TEXT NOT NULL,
    seller_id       UUID NOT NULL,
    briefing_id     INTEGER NULL REFERENCES tbl_meeting_briefing(id) ON DELETE SET NULL,
    description     TEXT NOT NULL,
    priority        TEXT NOT NULL CHECK (priority IN ('HIGH', 'MEDIUM', 'LOW')),
    evidence        TEXT NOT NULL,
    confidence      TEXT NOT NULL DEFAULT 'high'
                        CHECK (confidence IN ('high', 'low')),
    status          TEXT NOT NULL DEFAULT 'open'
                        CHECK (status IN ('open', 'done')),
    sort_order      INTEGER NOT NULL DEFAULT 0,
    source_type     TEXT NULL,
    source_id       TEXT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMP NULL
);

CREATE INDEX IF NOT EXISTS idx_meeting_prep_task_meeting_seller
    ON tbl_meeting_prep_task (meeting_id, seller_id);

CREATE INDEX IF NOT EXISTS idx_meeting_prep_task_status
    ON tbl_meeting_prep_task (meeting_id, seller_id, status);

-- Seller-added prep notes (typed or voice transcript).
CREATE TABLE IF NOT EXISTS tbl_meeting_prep_note (
    id              SERIAL PRIMARY KEY,
    meeting_id      TEXT NOT NULL,
    seller_id       UUID NOT NULL,
    note_type       TEXT NOT NULL CHECK (note_type IN ('typed', 'voice_transcript')),
    body            TEXT NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_meeting_prep_note_meeting_seller
    ON tbl_meeting_prep_note (meeting_id, seller_id);

COMMIT;

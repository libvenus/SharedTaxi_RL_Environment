-- =============================================================================
-- Sprint 1A · User Story 04 — Data Hygiene, Validation & Intelligent Alerts
-- =============================================================================
-- Adds one new table for the seller's data-quality To-Do queue:
--
--   tbl_data_task — one row per detected data-hygiene issue. Sources:
--     1. AI team's transcript-signal NLP pipeline (POST /api/data-tasks)
--     2. The daily-scan CLI job (app/jobs/scan_data_tasks.py)
--     3. FE inline validators (deferred to S1B)
--
-- Linkage:
--   tbl_data_task.entity_id → D365 records (FK-by-convention; cross-repo —
--                              we don't have a Postgres FK to D365's UUIDs).
--   tbl_data_task.owner_id  → seller's user UUID (also cross-repo).
--
-- Idempotency contract — critical for the daily scan:
--   The partial UNIQUE index on (entity_kind, entity_id, task_kind) WHERE
--   status IN ('open','dismissed') means:
--     * Re-running the scan on the same opp → INSERT silently no-ops (the
--       service catches the IntegrityError and returns the existing row)
--     * A DISMISSED task continues to occupy the slot, so re-detection of an
--       intentionally-dismissed mismatch will NOT recreate the alert. This
--       is how AC #5 ("intentional mismatches can be dismissed and future
--       alerts for that pair are suppressed") is implemented.
--   A RESOLVED task does NOT block re-creation — once the seller fixes the
--   issue, a fresh occurrence (new evidence, new transcript) should produce
--   a new task.
--
-- All statements are idempotent: this migration is safe to re-run.
-- =============================================================================

BEGIN;

-- 1. The data-task queue ------------------------------------------------------
CREATE TABLE IF NOT EXISTS tbl_data_task (
    task_id             UUID         PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Whose To-Do List this lands in. AC #11: "view and act on records within
    -- their own portfolio." All GETs filter by this.
    owner_id            UUID         NOT NULL,

    -- Which D365 entity is in trouble. (entity_kind, entity_id) is the join
    -- key the FE uses to navigate to the record.
    entity_kind         TEXT         NOT NULL,
    entity_id           UUID         NOT NULL,

    -- The rule that fired. Free-form TEXT (NOT a CHECK enum) so the AI team
    -- can introduce new transcript signals without a backend release. We
    -- document the canonical strings in US04_BACKEND_HANDOFF_FOR_AI_TEAM.md.
    task_kind           TEXT         NOT NULL,

    severity            TEXT         NOT NULL DEFAULT 'medium',
    status              TEXT         NOT NULL DEFAULT 'open',

    -- Optional UI / write-back hints. The FE renders these in the
    -- correction widget; current_value gets the strikethrough treatment.
    field_name          TEXT         NULL,
    current_value       TEXT         NULL,
    suggested_value     TEXT         NULL,

    -- AC: "ordered by confidence — High first." NULL is allowed for
    -- deterministic detectors where there's no model confidence to report.
    confidence          TEXT         NULL,

    -- AC #3: "no alert is generated without a grounding reference."
    -- evidence_text is the plain-language WHY shown to the seller.
    -- evidence_ref is a stable handle (e.g. transcript_segment_id=<uuid>,
    -- scan_run=<iso8601>) the FE can use to deep-link to the source.
    evidence_ref        TEXT         NULL,
    evidence_text       TEXT         NOT NULL,

    -- Where did this row come from? Useful for metrics + debugging.
    created_by_source   TEXT         NOT NULL,

    -- Resolution / dismissal trail. AC #10: "all corrections, resolutions,
    -- and dismissals are recorded in the audit log." For S1A we keep the
    -- audit on the row itself — a unified audit table is a cross-cutting
    -- concern for a future sprint.
    dismissal_note      TEXT         NULL,
    resolved_value      TEXT         NULL,
    actor_id            UUID         NULL,
    resolved_at         TIMESTAMPTZ  NULL,
    dismissed_at        TIMESTAMPTZ  NULL,

    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- entity_kind whitelist. Drop-and-recreate so new entity types can be
-- added by editing this migration and re-running it.
ALTER TABLE tbl_data_task
    DROP CONSTRAINT IF EXISTS chk_data_task_entity_kind;
ALTER TABLE tbl_data_task
    ADD CONSTRAINT chk_data_task_entity_kind CHECK (
        entity_kind IN (
            'account',
            'contact',
            'opportunity'
        )
    );

-- severity whitelist.
ALTER TABLE tbl_data_task
    DROP CONSTRAINT IF EXISTS chk_data_task_severity;
ALTER TABLE tbl_data_task
    ADD CONSTRAINT chk_data_task_severity CHECK (
        severity IN ('high', 'medium', 'low')
    );

-- status whitelist. 'open' is the only state where the FE shows a
-- pending To-Do; 'resolved' / 'dismissed' are terminal.
ALTER TABLE tbl_data_task
    DROP CONSTRAINT IF EXISTS chk_data_task_status;
ALTER TABLE tbl_data_task
    ADD CONSTRAINT chk_data_task_status CHECK (
        status IN ('open', 'resolved', 'dismissed')
    );

-- confidence whitelist (NULL allowed — deterministic detectors leave it null).
ALTER TABLE tbl_data_task
    DROP CONSTRAINT IF EXISTS chk_data_task_confidence;
ALTER TABLE tbl_data_task
    ADD CONSTRAINT chk_data_task_confidence CHECK (
        confidence IS NULL OR confidence IN ('high', 'medium', 'low')
    );

-- created_by_source whitelist.
ALTER TABLE tbl_data_task
    DROP CONSTRAINT IF EXISTS chk_data_task_source;
ALTER TABLE tbl_data_task
    ADD CONSTRAINT chk_data_task_source CHECK (
        created_by_source IN (
            'transcript',   -- AI team's NLP pipeline
            'scan',         -- daily-scan CLI job
            'inline',       -- FE inline validator (S1B)
            'manual'        -- admin / debug
        )
    );

-- evidence_text must not be empty whitespace (AC #3 grounding ref).
ALTER TABLE tbl_data_task
    DROP CONSTRAINT IF EXISTS chk_data_task_evidence_text;
ALTER TABLE tbl_data_task
    ADD CONSTRAINT chk_data_task_evidence_text CHECK (
        char_length(btrim(evidence_text)) > 0
    );

-- A dismissal MUST come with a note (AC #4 "dismissing with a note —
-- dismissals are recorded"). Defensive guard so a buggy client can't
-- write a noteless dismissal.
ALTER TABLE tbl_data_task
    DROP CONSTRAINT IF EXISTS chk_data_task_dismiss_note;
ALTER TABLE tbl_data_task
    ADD CONSTRAINT chk_data_task_dismiss_note CHECK (
        status <> 'dismissed'
        OR (dismissal_note IS NOT NULL
            AND char_length(btrim(dismissal_note)) > 0)
    );

-- Resolved tasks must record WHEN they were resolved.
ALTER TABLE tbl_data_task
    DROP CONSTRAINT IF EXISTS chk_data_task_resolved_at;
ALTER TABLE tbl_data_task
    ADD CONSTRAINT chk_data_task_resolved_at CHECK (
        status <> 'resolved' OR resolved_at IS NOT NULL
    );

-- Dismissed tasks must record WHEN they were dismissed.
ALTER TABLE tbl_data_task
    DROP CONSTRAINT IF EXISTS chk_data_task_dismissed_at;
ALTER TABLE tbl_data_task
    ADD CONSTRAINT chk_data_task_dismissed_at CHECK (
        status <> 'dismissed' OR dismissed_at IS NOT NULL
    );

-- Idempotency + suppression backbone — see header for the full contract.
-- Resolved tasks are EXCLUDED from the partial index so a fresh
-- occurrence after the seller fixed it can produce a new task.
CREATE UNIQUE INDEX IF NOT EXISTS uq_data_task_active_or_dismissed
    ON tbl_data_task (entity_kind, entity_id, task_kind)
    WHERE status IN ('open', 'dismissed');

-- FE queue load — most common query: "give me all my open tasks."
CREATE INDEX IF NOT EXISTS idx_data_task_owner_status
    ON tbl_data_task (owner_id, status);

-- Admin / metrics: how many open tasks across the system.
CREATE INDEX IF NOT EXISTS idx_data_task_status
    ON tbl_data_task (status);

-- "Show me all tasks for this opportunity" — used by the deal-detail page.
CREATE INDEX IF NOT EXISTS idx_data_task_entity
    ON tbl_data_task (entity_kind, entity_id);

COMMIT;

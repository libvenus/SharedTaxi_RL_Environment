-- ============================================================================
-- 2026-06: Opportunity "Complete Information" tab schema
--
-- Adds five new columns + three D365 audit columns to ``opportunity`` so the
-- Complete-Information form on the Opportunity Detail page can read, update
-- and persist its full field surface.
--
-- Powering (form labels in parens):
--   * lvo_summary             (Summary)
--   * lvo_priority            (Priority — High/Medium/Low)
--   * lvo_leadorigin          (Lead Origin)
--   * lvo_partnerinvolved     (Partner Involved toggle)
--   * lvo_parentopportunityid (Parent Opportunity — TEXT, integrity at app
--                              level; see comment below for rationale)
--   * createdby               (Created By — system audit)
--   * modifiedon              (Modified Date — system audit)
--   * modifiedby              (Modified By — system audit)
--
-- The migration is idempotent (every ``ALTER TABLE`` uses ``IF NOT EXISTS``,
-- the priority CHECK is wrapped in a ``DO $$ ... IF NOT EXISTS`` guard, and
-- the in-flight FK from earlier v0.13.0 attempts is dropped defensively),
-- so it can be re-run safely on dev / staging / prod without side effects.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Editable form fields
-- ---------------------------------------------------------------------------

ALTER TABLE opportunity
    ADD COLUMN IF NOT EXISTS lvo_summary TEXT;

ALTER TABLE opportunity
    ADD COLUMN IF NOT EXISTS lvo_priority TEXT;

DO $$
BEGIN
    -- Whitelist for lvo_priority. Allows NULL so existing rows don't fail.
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint
         WHERE conname = 'chk_opportunity_priority'
    ) THEN
        ALTER TABLE opportunity
            ADD CONSTRAINT chk_opportunity_priority
            CHECK (
                lvo_priority IS NULL
                OR lvo_priority IN ('High', 'Medium', 'Low')
            );
    END IF;
END $$;

ALTER TABLE opportunity
    ADD COLUMN IF NOT EXISTS lvo_leadorigin TEXT;

ALTER TABLE opportunity
    ADD COLUMN IF NOT EXISTS lvo_partnerinvolved BOOLEAN NOT NULL DEFAULT FALSE;

-- Self-referencing parent for the Parent / Child Opportunity hierarchy.
--
-- Note on type + integrity strategy:
--   * Column is plain TEXT to mirror the rest of our ORM, where every
--     UUID-shaped column is mapped as ``Mapped[str | None]``. This keeps
--     the parent/child query helpers (``func.upper(...)`` comparisons in
--     ``deals_read.py`` / ``deals_write.py``) consistent with how we
--     already join on ``accountid`` and ``owninguser``.
--
--   * No DB-level FK to ``opportunity(opportunityid)``. PostgreSQL
--     refuses a FK between TEXT and UUID, and the ``cast(... AS UUID)``
--     workaround isn't supported inside FK definitions. Integrity is
--     enforced at the API layer instead, in ``PATCH /api/opportunities/{id}``:
--       - parent must exist
--       - parent must not be cancelled
--       - no self-reference
--       - no cycle (recursive walk capped at depth 10)
--     Cancelled / hard-deleted parents are also filtered out at READ time
--     so a stale link never surfaces in the UI.
--
--   * Defensive ``DROP CONSTRAINT IF EXISTS`` cleans up after any
--     in-flight v0.13.0 deployment that tried (and failed) to add the
--     incompatible FK, so this migration is fully re-runnable.
ALTER TABLE opportunity
    ADD COLUMN IF NOT EXISTS lvo_parentopportunityid TEXT;

ALTER TABLE opportunity
    DROP CONSTRAINT IF EXISTS fk_opportunity_parent;

CREATE INDEX IF NOT EXISTS ix_opportunity_parent
    ON opportunity (lvo_parentopportunityid)
 WHERE lvo_parentopportunityid IS NOT NULL;

-- ---------------------------------------------------------------------------
-- D365 audit columns
--
-- These usually ship with a D365 dump but are guarded with IF NOT EXISTS so
-- the migration is safe on stripped dumps. The ORM marks all three as
-- ``deferred=True`` so other read paths don't accidentally SELECT them when
-- they're absent (we already do this for the customer-information columns).
-- ---------------------------------------------------------------------------

ALTER TABLE opportunity
    ADD COLUMN IF NOT EXISTS createdby TEXT;

ALTER TABLE opportunity
    ADD COLUMN IF NOT EXISTS modifiedon TIMESTAMPTZ;

ALTER TABLE opportunity
    ADD COLUMN IF NOT EXISTS modifiedby TEXT;

-- ---------------------------------------------------------------------------
-- Search-helper index for the Parent-Opportunity picker
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS ix_opportunity_name_lower
    ON opportunity (LOWER(name))
 WHERE name IS NOT NULL;

-- ============================================================================
-- Widen lvo_opportunitysnapshot.lvo_bucket whitelist to include 'most_likely'.
--
-- The KPI strip on the Opportunities page added a "Most Likely" card backed by
-- a new bucket whose predicate is `lvo_forecastcategory = 'Most Likely'`. The
-- snapshot job cannot UPSERT rows for this bucket until the CHECK constraint
-- on lvo_opportunitysnapshot accepts the new value.
--
-- This migration is **idempotent** — re-running it is a no-op when the wider
-- whitelist is already in place.
--
-- Safe on:
--   * Existing DBs running the original 6-bucket whitelist  → drops + re-adds.
--   * DBs that already have the 7-bucket whitelist          → no-op.
--   * Fresh DBs that have not yet applied
--     sql/2026_06_create_opportunity_snapshot.sql            → prints a NOTICE
--                                                              and bails
--                                                              cleanly.
-- ============================================================================

DO $$
BEGIN
    -- Bail out cleanly when the snapshot table doesn't exist yet — operator
    -- should run sql/2026_06_create_opportunity_snapshot.sql first.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
         WHERE table_name = 'lvo_opportunitysnapshot'
    ) THEN
        RAISE NOTICE 'lvo_opportunitysnapshot does not exist yet — '
                     'run sql/2026_06_create_opportunity_snapshot.sql first. '
                     'Skipping CHECK widening.';
        RETURN;
    END IF;

    -- Drop the old constraint (if present) and re-add with the wider whitelist.
    -- The CHECK expression is inlined as a literal IN-list so it is preserved
    -- verbatim in pg_catalog (a PL/pgSQL local would not be in scope at
    -- constraint-eval time).
    IF EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'chk_lvo_opportunitysnapshot_bucket'
    ) THEN
        ALTER TABLE lvo_opportunitysnapshot
            DROP CONSTRAINT chk_lvo_opportunitysnapshot_bucket;
    END IF;

    ALTER TABLE lvo_opportunitysnapshot
        ADD CONSTRAINT chk_lvo_opportunitysnapshot_bucket
        CHECK (lvo_bucket IN (
            'open', 'pipeline', 'best_case', 'commit', 'most_likely', 'won', 'loss'
        ));
END $$;

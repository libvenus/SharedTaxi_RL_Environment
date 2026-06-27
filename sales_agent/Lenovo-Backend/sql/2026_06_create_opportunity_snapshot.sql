-- ============================================================================
-- lvo_opportunitysnapshot — Daily KPI bucket aggregates (powers trend math)
--
-- Stores one row per (snapshot_date × bucket). The kpi-summary endpoint
-- subtracts a historical row from the live aggregate to compute
-- `TrendInfo.deltaValue` / `deltaCount` / `direction`.
--
-- Buckets mirror the six KPI cards on the Opportunities grid:
--    open       — opportunity.statecode = 'Open'
--    pipeline   — opportunity.lvo_forecastcategory = 'Pipeline'
--    best_case  — opportunity.lvo_forecastcategory = 'Best Case'
--    commit     — opportunity.lvo_forecastcategory = 'Commit'
--    won        — statecode IN ('Won','Closed Won') OR stagename='Closed Won'
--    loss       — statecode IN ('Lost','Closed Lost') OR stagename='Closed Lost'
--
-- The job at app/jobs/snapshot_kpis.py UPSERTs one row per bucket each time
-- it runs; uniqueness is enforced by uq_lvo_opportunitysnapshot_date_bucket.
--
-- Re-running this migration is safe (CREATE IF NOT EXISTS, no INSERTs).
-- ============================================================================

CREATE TABLE IF NOT EXISTS lvo_opportunitysnapshot (
    lvo_opportunitysnapshotid  TEXT PRIMARY KEY,
    lvo_snapshotdate           DATE NOT NULL,
    lvo_bucket                 TEXT NOT NULL,
    lvo_value                  NUMERIC(20, 2) NOT NULL DEFAULT 0,
    lvo_count                  INTEGER       NOT NULL DEFAULT 0,
    lvo_createdat              TIMESTAMPTZ   NOT NULL DEFAULT now()
);


-- ----------------------------------------------------------------------------
-- Bucket whitelist (CHECK constraint, idempotent)
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'chk_lvo_opportunitysnapshot_bucket'
    ) THEN
        ALTER TABLE lvo_opportunitysnapshot
            ADD CONSTRAINT chk_lvo_opportunitysnapshot_bucket
            CHECK (lvo_bucket IN (
                'open', 'pipeline', 'best_case', 'commit', 'most_likely', 'won', 'loss'
            ));
    END IF;
END $$;


-- ----------------------------------------------------------------------------
-- Indexes
--   * uq_*  — guarantees one row per (date, bucket); the upsert in the job
--             targets this index with ON CONFLICT.
--   * idx_* — cheap range lookups for the snapshot-history endpoint.
-- ----------------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS uq_lvo_opportunitysnapshot_date_bucket
    ON lvo_opportunitysnapshot (lvo_snapshotdate, lvo_bucket);

CREATE INDEX IF NOT EXISTS idx_lvo_opportunitysnapshot_date
    ON lvo_opportunitysnapshot (lvo_snapshotdate DESC);


-- Sanity check (optional — uncomment after the snapshot job runs once):
-- SELECT lvo_snapshotdate, lvo_bucket, lvo_value, lvo_count
--   FROM lvo_opportunitysnapshot
--  ORDER BY lvo_snapshotdate DESC, lvo_bucket;

-- ============================================================================
-- lvo_accountsnapshot — Daily Account-KPI bucket aggregates (powers trend math)
--
-- Stores one row per (snapshot_date × bucket). The /api/accounts/kpi-summary
-- endpoint subtracts a historical row from the live aggregate to compute
-- TrendInfo.deltaValue / deltaCount / direction.
--
-- Buckets mirror the four cards on the View-Account page:
--    total      — every row in the account table (no predicate)
--    acv        — sum of opportunity.estimatedvalue WHERE statecode <> 'Canceled'
--                 (matches the per-account totalAccountValue rollup; page total
--                  always equals the column sum on the grid)
--    active     — account.lvo_accountstatus = 'Active'
--    at_risk    — account.lvo_accountstatus = 'At-Risk'
--
-- The job at app/jobs/snapshot_account_kpis.py UPSERTs one row per bucket each
-- time it runs; uniqueness is enforced by uq_lvo_accountsnapshot_date_bucket.
--
-- Re-running this migration is safe (CREATE IF NOT EXISTS, no INSERTs).
-- ============================================================================

CREATE TABLE IF NOT EXISTS lvo_accountsnapshot (
    lvo_accountsnapshotid  TEXT PRIMARY KEY,
    lvo_snapshotdate       DATE NOT NULL,
    lvo_bucket             TEXT NOT NULL,
    lvo_value              NUMERIC(20, 2) NOT NULL DEFAULT 0,
    lvo_count              INTEGER       NOT NULL DEFAULT 0,
    lvo_createdat          TIMESTAMPTZ   NOT NULL DEFAULT now()
);


-- ----------------------------------------------------------------------------
-- Bucket whitelist (CHECK constraint, idempotent)
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'chk_lvo_accountsnapshot_bucket'
    ) THEN
        ALTER TABLE lvo_accountsnapshot
            ADD CONSTRAINT chk_lvo_accountsnapshot_bucket
            CHECK (lvo_bucket IN (
                'total', 'acv', 'active', 'at_risk'
            ));
    END IF;
END $$;


-- ----------------------------------------------------------------------------
-- Indexes
--   * uq_*  — guarantees one row per (date, bucket); the upsert in the job
--             targets this index with ON CONFLICT.
--   * idx_* — cheap range lookups for the snapshot-history view.
-- ----------------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS uq_lvo_accountsnapshot_date_bucket
    ON lvo_accountsnapshot (lvo_snapshotdate, lvo_bucket);

CREATE INDEX IF NOT EXISTS idx_lvo_accountsnapshot_date
    ON lvo_accountsnapshot (lvo_snapshotdate DESC);


-- Sanity check (optional — uncomment after the snapshot job runs once):
-- SELECT lvo_snapshotdate, lvo_bucket, lvo_value, lvo_count
--   FROM lvo_accountsnapshot
--  ORDER BY lvo_snapshotdate DESC, lvo_bucket;

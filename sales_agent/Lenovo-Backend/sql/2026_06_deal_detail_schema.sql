-- ============================================================================
-- Deal Detailed View — opportunity-side schema additions
--
-- Adds the columns needed to power:
--   * Deal Health calculator (tempo class + stage entry date + recalc timestamp)
--   * Stage Velocity score (uses lvo_stageentrydate as 'Last stage change date')
--   * Close Date Confidence score (uses lvo_createdat as 'Opportunity creation date')
--
-- Run this against the same database that holds `opportunity`.
-- Re-running is safe: ALTERs use IF NOT EXISTS, UPDATEs are idempotent.
-- ============================================================================

ALTER TABLE opportunity
    ADD COLUMN IF NOT EXISTS lvo_tempoclass             TEXT,
    ADD COLUMN IF NOT EXISTS lvo_stageentrydate         TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS lvo_createdat              TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS lvo_dealhealthupdatedat    TIMESTAMPTZ;


-- ----------------------------------------------------------------------------
-- Constrain tempo class to the four documented buckets.
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'chk_opportunity_tempoclass'
    ) THEN
        ALTER TABLE opportunity
            ADD CONSTRAINT chk_opportunity_tempoclass
            CHECK (
                lvo_tempoclass IS NULL
                OR lvo_tempoclass IN (
                    'Fast',                  -- Fast / Transactional      (30 days)
                    'Quarterly',             -- Quarterly / Enterprise    (90 days)
                    'Programmatic',          -- Programmatic / Annual     (365 days)
                    'Strategic'              -- Strategic / Multiyear     (730 days)
                )
            );
    END IF;
END $$;


-- ----------------------------------------------------------------------------
-- Default every existing row to 'Quarterly' (per Phase-1 plan decision).
-- New rows can be set explicitly by the create-deal flow when that lands.
-- ----------------------------------------------------------------------------
UPDATE opportunity
   SET lvo_tempoclass = 'Quarterly'
 WHERE lvo_tempoclass IS NULL;


-- ----------------------------------------------------------------------------
-- Backfill lvo_createdat from createdon if that column exists in the base
-- D365 dump; otherwise from the bulk-seed close-date heuristic. We only set
-- rows where the column is currently NULL so re-running the migration won't
-- clobber data the application has written.
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'opportunity' AND column_name = 'createdon'
    ) THEN
        EXECUTE $sql$
            UPDATE opportunity
               SET lvo_createdat = createdon
             WHERE lvo_createdat IS NULL
               AND createdon IS NOT NULL;
        $sql$;
    END IF;
END $$;

-- Final fallback: assume the deal was created 60 days before its close date.
-- Keeps the close-date confidence calculator from blowing up on legacy rows.
UPDATE opportunity
   SET lvo_createdat = (estimatedclosedate - INTERVAL '60 days')::TIMESTAMPTZ
 WHERE lvo_createdat IS NULL
   AND estimatedclosedate IS NOT NULL;


-- ----------------------------------------------------------------------------
-- Backfill lvo_stageentrydate.
--
-- Strategy (per "both" decision in plan):
--   1. Prefer the most recent audit-log entry where the stage actually
--      changed. That's any lvo_audit_log row with lvo_entitytype='opportunity'
--      and lvo_diff -> 'before' ->> 'stagename' != lvo_diff -> 'after' ->> 'stagename'.
--   2. Otherwise fall back to lvo_createdat (treat the whole deal as having
--      sat in its current stage since creation).
--
-- The audit-log query is wrapped in DO so it can be skipped gracefully when
-- the table doesn't exist yet (very-fresh DBs).
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
         WHERE table_name = 'lvo_audit_log'
    ) THEN
        EXECUTE $sql$
            UPDATE opportunity o
               SET lvo_stageentrydate = sub.last_stage_change
              FROM (
                  SELECT lvo_opportunityid,
                         MAX(lvo_changedat) AS last_stage_change
                    FROM lvo_audit_log
                   WHERE lvo_entitytype = 'opportunity'
                     AND lvo_action = 'update'
                     AND lvo_diff IS NOT NULL
                     AND COALESCE(lvo_diff -> 'before' ->> 'stagename', '')
                         <> COALESCE(lvo_diff -> 'after'  ->> 'stagename', '')
                   GROUP BY lvo_opportunityid
              ) sub
             WHERE UPPER(o.opportunityid::TEXT) = UPPER(sub.lvo_opportunityid)
               AND o.lvo_stageentrydate IS NULL;
        $sql$;
    END IF;
END $$;

-- Fallback: use creation date so the velocity calculator always has a value.
UPDATE opportunity
   SET lvo_stageentrydate = lvo_createdat
 WHERE lvo_stageentrydate IS NULL
   AND lvo_createdat IS NOT NULL;


-- ----------------------------------------------------------------------------
-- Helpful index for the recalc batch job (filters Open deals only).
-- ----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_opportunity_statecode
    ON opportunity (statecode);

CREATE INDEX IF NOT EXISTS idx_opportunity_dealhealthupdatedat
    ON opportunity (lvo_dealhealthupdatedat DESC NULLS LAST);


-- Sanity check (optional — uncomment to run):
-- SELECT name, lvo_tempoclass, lvo_stageentrydate, lvo_createdat
--   FROM opportunity
--  ORDER BY lvo_stageentrydate DESC NULLS LAST
--  LIMIT 20;

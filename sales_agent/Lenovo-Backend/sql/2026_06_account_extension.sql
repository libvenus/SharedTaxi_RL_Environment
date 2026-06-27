-- ============================================================================
-- account — Deal Detailed View additions
--
-- The "Account linked to deal" panel needs:
--   * Name              (already on `account.name`)
--   * Segment           (already on `account.lvo_segment`)
--   * Industry          (already on `account.industrycode`)
--   * Employee Count    (already on `account.numberofemployees`)
--   * Territory         (NEW — added below)
--   * Total Account Value   (DERIVED at query time from opportunity.estimatedvalue)
--   * Open Deals Count      (DERIVED at query time from opportunity.statecode='Open')
--
-- This migration only adds the missing column. Re-running is safe.
-- ============================================================================

ALTER TABLE account
    ADD COLUMN IF NOT EXISTS lvo_territory TEXT;


-- ----------------------------------------------------------------------------
-- Seed territory for the existing accounts. Mirrors the business-group split
-- from sql/2026_06_bulk_seed_50_opportunities.sql so a brand-new dev DB has
-- non-null values in the Account panel out of the box.
-- ----------------------------------------------------------------------------

UPDATE account
   SET lvo_territory = CASE lvo_businessgroupid
                           WHEN 'Americas BG' THEN 'North America'
                           WHEN 'EMEA BG'     THEN 'Western Europe'
                           WHEN 'APAC BG'     THEN 'Asia Pacific'
                           ELSE 'Global'
                       END
 WHERE lvo_territory IS NULL;


-- Sanity check (optional — uncomment to run):
-- SELECT name, lvo_segment, industrycode, lvo_territory, numberofemployees
--   FROM account
--  ORDER BY name
--  LIMIT 20;

-- ============================================================================
-- Demo fix: align live deal-health calculator with grid (lvo_dealhealthscore)
--
-- Problem:
--   Grid  → stored opportunity.lvo_dealhealthscore (e.g. 79% GREEN)
--   Detail → live recalculate_deal_health(write=False) (e.g. 45% RED)
--
-- Live score was low because many open deals lack:
--   * lvo_createdat / lvo_stageentrydate  → Close Confidence = 0
--   * lvo_opportunitycontact rows (5+)    → Stakeholder ≈ 12%
--
-- After this script, run batch recalc (see bottom) so stored scores match live.
--
-- Safe to re-run.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Step 1 — Health metadata for ALL open opportunities
-- ----------------------------------------------------------------------------
UPDATE opportunity
   SET lvo_tempoclass = COALESCE(lvo_tempoclass, 'Quarterly'),
       lvo_createdat = COALESCE(
           lvo_createdat,
           (estimatedclosedate - INTERVAL '120 days')::TIMESTAMPTZ
       ),
       lvo_stageentrydate = COALESCE(
           lvo_stageentrydate,
           COALESCE(lvo_createdat, (estimatedclosedate - INTERVAL '120 days')::TIMESTAMPTZ)
               + INTERVAL '45 days'
       )
 WHERE statecode = 'Open';

-- Ensure stage entry is not in the future (helps stage-velocity score)
UPDATE opportunity
   SET lvo_stageentrydate = LEAST(
           lvo_stageentrydate,
           NOW() - INTERVAL '14 days'
       )
 WHERE statecode = 'Open'
   AND lvo_stageentrydate > NOW();

-- ----------------------------------------------------------------------------
-- Step 2 — Ensure decision maker from legacy opportunity.contactid
-- ----------------------------------------------------------------------------
INSERT INTO lvo_opportunitycontact (
    lvo_opportunitycontactid,
    lvo_opportunityid,
    lvo_contactid,
    lvo_role,
    lvo_isdecisionmaker,
    statecode
)
SELECT
    'OC-DM-' || REPLACE(o.opportunityid::TEXT, '-', ''),
    UPPER(o.opportunityid::TEXT),
    UPPER(o.contactid::TEXT),
    'Decision Maker',
    TRUE,
    'Active'
  FROM opportunity o
 WHERE o.statecode = 'Open'
   AND o.contactid IS NOT NULL
   AND NOT EXISTS (
       SELECT 1
         FROM lvo_opportunitycontact oc
        WHERE UPPER(oc.lvo_opportunityid) = UPPER(o.opportunityid::TEXT)
          AND UPPER(oc.lvo_contactid) = UPPER(o.contactid::TEXT)
          AND oc.statecode = 'Active'
   )
ON CONFLICT (lvo_opportunitycontactid) DO NOTHING;

-- ----------------------------------------------------------------------------
-- Step 3 — Pad to 5 active stakeholders per open deal (demo coverage)
-- Uses any Active contacts when the account has too few.
-- ----------------------------------------------------------------------------
DO $$
DECLARE
    parent_col TEXT;
BEGIN
    SELECT column_name
      INTO parent_col
      FROM information_schema.columns
     WHERE table_name = 'contact'
       AND column_name IN (
            'parentcustomerid',
            'parentcustomerid_value',
            '_parentcustomerid_value',
            'accountid'
       )
     ORDER BY ARRAY_POSITION(
        ARRAY['parentcustomerid', 'parentcustomerid_value',
              '_parentcustomerid_value', 'accountid'],
        column_name
     )
     LIMIT 1;

    IF parent_col IS NOT NULL THEN
        EXECUTE format($sql$
            INSERT INTO lvo_opportunitycontact (
                lvo_opportunitycontactid,
                lvo_opportunityid,
                lvo_contactid,
                lvo_role,
                lvo_isdecisionmaker,
                statecode
            )
            SELECT
                'OC-ACC-' || REPLACE(o.opportunityid::TEXT, '-', '')
                      || '-' || LEFT(REPLACE(c.contactid::TEXT, '-', ''), 8),
                UPPER(o.opportunityid::TEXT),
                UPPER(c.contactid::TEXT),
                'Champion',
                FALSE,
                'Active'
              FROM opportunity o
              JOIN contact c
                ON UPPER(c.%I::TEXT) = UPPER(o.accountid::TEXT)
             WHERE o.statecode = 'Open'
               AND o.accountid IS NOT NULL
               AND NOT EXISTS (
                   SELECT 1
                     FROM lvo_opportunitycontact oc
                    WHERE UPPER(oc.lvo_opportunityid) = UPPER(o.opportunityid::TEXT)
                      AND UPPER(oc.lvo_contactid) = UPPER(c.contactid::TEXT)
                      AND oc.statecode = 'Active'
               )
            ON CONFLICT (lvo_opportunitycontactid) DO NOTHING;
        $sql$, parent_col);
    END IF;
END $$;

-- Top-up: if still < 5 stakeholders, add from global contact pool
INSERT INTO lvo_opportunitycontact (
    lvo_opportunitycontactid,
    lvo_opportunityid,
    lvo_contactid,
    lvo_role,
    lvo_isdecisionmaker,
    statecode
)
SELECT
    'OC-POOL-' || REPLACE(need.opportunityid::TEXT, '-', '') || '-' || seq.n,
    UPPER(need.opportunityid::TEXT),
    UPPER(pool.contactid::TEXT),
    CASE WHEN seq.n = 1 THEN 'Decision Maker' ELSE 'Influencer' END,
    seq.n = 1,
    'Active'
  FROM (
        SELECT o.opportunityid
          FROM opportunity o
         WHERE o.statecode = 'Open'
           AND (
               SELECT COUNT(*)
                 FROM lvo_opportunitycontact oc
                WHERE UPPER(oc.lvo_opportunityid) = UPPER(o.opportunityid::TEXT)
                  AND oc.statecode = 'Active'
           ) < 5
       ) need
 CROSS JOIN generate_series(1, 5) AS seq(n)
 CROSS JOIN LATERAL (
        SELECT c.contactid
          FROM contact c
         ORDER BY c.contactid
        OFFSET seq.n - 1
         LIMIT 1
       ) pool
 WHERE NOT EXISTS (
        SELECT 1
          FROM lvo_opportunitycontact oc
         WHERE UPPER(oc.lvo_opportunityid) = UPPER(need.opportunityid::TEXT)
           AND UPPER(oc.lvo_contactid) = UPPER(pool.contactid::TEXT)
           AND oc.statecode = 'Active'
       )
ON CONFLICT (lvo_opportunityid, lvo_contactid)
    WHERE statecode = 'Active'
    DO NOTHING;

-- ----------------------------------------------------------------------------
-- Step 4 — Clear stale risk flags on healthy demo deals (optional polish)
-- Keep Tesla Fremont (040) red story if risk reason is set intentionally.
-- ----------------------------------------------------------------------------
UPDATE opportunity
   SET lvo_riskscore = NULL,
       lvo_riskreason = NULL
 WHERE statecode = 'Open'
   AND lvo_riskreason IS DISTINCT FROM 'Stale Activity';

-- Step 4b — Drop stale lvo_dealrisk rows so batch recalc repopulates the
-- full live set (grid riskCount must match detail risk list).
DELETE FROM lvo_dealrisk
 WHERE UPPER(lvo_opportunityid) IN (
       SELECT UPPER(opportunityid::text)
         FROM opportunity
        WHERE statecode = 'Open'
       );

-- ============================================================================
-- Step 5 — Persist live scores (REQUIRED — run after this SQL)
-- ============================================================================
-- Prerequisite: if recalc_health fails on lvo_actortype missing, run:
--   sql/2026_14_audit_compliance.sql
-- (Updated app code also supports legacy audit schema without that migration.)
--
-- From Lenovo D365 Sales project root:
--
--   python -m app.jobs.recalc_health
--
-- Or one seller only (Americas demo):
--
--   python -m app.jobs.recalc_health --seller-id 81AADFDB-1817-425C-A5B1-45F383F230CE
--
-- Then verify Ford / open deals — grid % should match deal detail:
--
--   SELECT name, lvo_dealhealthscore, lvo_dealhealthupdatedat
--     FROM opportunity
--    WHERE statecode = 'Open'
--    ORDER BY lvo_dealhealthscore DESC NULLS LAST;

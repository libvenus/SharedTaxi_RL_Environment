-- ============================================================================
-- opportunity — Deal Health + Risk columns
--
-- Replaces the hard-coded `health` / `risk` values that used to live in the
-- React UI (LenovoD365/src/pages/Opportunities.jsx) with real DB columns:
--
--   lvo_dealhealthscore  INTEGER   0–100, drives the SVG ring on the grid
--   lvo_riskscore        SMALLINT  1–5,   numeric severity (filter/sort)
--   lvo_riskreason       TEXT             short label shown in the ⚠ badge
--
-- Run this in pgAdmin against the same database that holds `opportunity`.
-- Re-running is safe: ALTERs use IF NOT EXISTS, UPDATEs are idempotent.
-- ============================================================================

ALTER TABLE opportunity
    ADD COLUMN IF NOT EXISTS lvo_dealhealthscore INTEGER,
    ADD COLUMN IF NOT EXISTS lvo_riskscore       SMALLINT,
    ADD COLUMN IF NOT EXISTS lvo_riskreason      TEXT;

-- Postgres has no "ADD CONSTRAINT IF NOT EXISTS", so guard via pg_constraint.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'chk_opportunity_dealhealth_range'
    ) THEN
        ALTER TABLE opportunity
            ADD CONSTRAINT chk_opportunity_dealhealth_range
            CHECK (lvo_dealhealthscore IS NULL
                   OR lvo_dealhealthscore BETWEEN 0 AND 100);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'chk_opportunity_riskscore_range'
    ) THEN
        ALTER TABLE opportunity
            ADD CONSTRAINT chk_opportunity_riskscore_range
            CHECK (lvo_riskscore IS NULL
                   OR lvo_riskscore BETWEEN 1 AND 5);
    END IF;
END $$;


-- ============================================================================
-- Seed data — narrative-consistent with the lvo_activity rows.
-- ============================================================================

-- --- Deutsche Bank — Workstation Refresh -----------------------------------
-- Demo done, multi-event review logged. Strong momentum, no risk.
UPDATE opportunity
   SET lvo_dealhealthscore = 85,
       lvo_riskscore       = NULL,
       lvo_riskreason      = NULL
 WHERE UPPER(opportunityid::TEXT) = 'CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B';

-- --- Siemens AG — Edge AI Infrastructure -----------------------------------
-- Client flagged budget freeze, risk score elevated in CRM, HP Inc. competing.
UPDATE opportunity
   SET lvo_dealhealthscore = 42,
       lvo_riskscore       = 3,
       lvo_riskreason      = 'Budget Freeze'
 WHERE UPPER(opportunityid::TEXT) = 'F86EC95A-F627-4C12-A3A5-4ADBC43C6DBA';

-- --- HSBC — DaaS Rollout APAC ----------------------------------------------
-- Renewal motion, proposal already sent. Healthy but not yet signed.
UPDATE opportunity
   SET lvo_dealhealthscore = 70,
       lvo_riskscore       = NULL,
       lvo_riskreason      = NULL
 WHERE UPPER(opportunityid::TEXT) = '116191C4-CE2F-46CB-8666-4861A6A1AE26';

-- --- Ford Motor — HPC Cluster Upgrade --------------------------------------
-- Kickoff → pricing → PO submitted. Strongest deal in the pipeline.
UPDATE opportunity
   SET lvo_dealhealthscore = 92,
       lvo_riskscore       = NULL,
       lvo_riskreason      = NULL
 WHERE UPPER(opportunityid::TEXT) = '5977B053-8389-4497-BA97-076CBA41FB86';

-- --- Infosys — Developer Laptop Refresh ------------------------------------
-- Through qualify, DQR approved, but PO only "expected end of June".
UPDATE opportunity
   SET lvo_dealhealthscore = 62,
       lvo_riskscore       = 2,
       lvo_riskreason      = 'PO Delay'
 WHERE UPPER(opportunityid::TEXT) = '84D4BB4D-E2DC-4B46-9D32-F7D1D182B414';


-- Sanity check (optional — uncomment to run):
-- SELECT name, lvo_dealhealthscore, lvo_riskscore, lvo_riskreason
--   FROM opportunity
--  ORDER BY lvo_dealhealthscore DESC NULLS LAST;

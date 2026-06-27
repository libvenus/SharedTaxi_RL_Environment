-- ============================================================================
-- lvo_dealrisk — Persisted output of the risk-derivation rules
--
-- One row per active risk per deal. Re-written each time the deal-health
-- recalculator runs (delete-then-insert; the recalc service does it inside a
-- single transaction).
--
-- Risk taxonomy is fixed by the user story:
--
--   Activity & Engagement Risks
--     'Low Activity'                  | last activity > 14 days
--     'No Recent Engagement'          | no activity within tempo cadence
--   Stakeholder Risks
--     'Decision Maker Not Engaged'    | no activity tied to DM, or no DM mapped
--     'Single-Threaded Engagement'    | only one active stakeholder
--     'Low Stakeholder Score'         | stakeholder score < configured limit
--   Deal Execution Risks
--     'No Next Steps Defined'         | no Open next-action
--     'Missing Action Date'           | next action exists but due date is empty
--     'Stale Deal Stage'              | today > expected stage exit date
--     'Incomplete Deal Information'   | one of value/closeDate/stage is missing
--   Timeline & Forecast Risks
--     'No Close Date'                 | estimatedclosedate IS NULL
--     'Close Date Overdue'            | estimatedclosedate < today AND not closed
--     'Unrealistic Close Timeline'    | gap > 40% (close date confidence = 0)
--     'Deal Stuck in Stage'           | velocity > 2.0
--
-- The names here MUST stay in sync with the Risk dataclass in
-- app/services/deal_risks.py — both ends are validated by tests.
-- ============================================================================

CREATE TABLE IF NOT EXISTS lvo_dealrisk (
    lvo_dealriskid       TEXT PRIMARY KEY,
    lvo_opportunityid    TEXT NOT NULL,
    lvo_riskcategory     TEXT NOT NULL,
    lvo_riskname         TEXT NOT NULL,
    lvo_message          TEXT NOT NULL,
    lvo_detectedat       TIMESTAMPTZ NOT NULL DEFAULT now(),
    statecode            TEXT NOT NULL DEFAULT 'Active'
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'chk_lvo_dealrisk_category'
    ) THEN
        ALTER TABLE lvo_dealrisk
            ADD CONSTRAINT chk_lvo_dealrisk_category
            CHECK (lvo_riskcategory IN (
                'Activity & Engagement',
                'Stakeholder',
                'Deal Execution',
                'Timeline & Forecast'
            ));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_lvo_dealrisk_opportunity
    ON lvo_dealrisk (lvo_opportunityid);

CREATE INDEX IF NOT EXISTS idx_lvo_dealrisk_active
    ON lvo_dealrisk (lvo_opportunityid)
 WHERE statecode = 'Active';


-- Sanity check (optional — uncomment after the recalc service runs once):
-- SELECT lvo_opportunityid, COUNT(*) AS active_risks
--   FROM lvo_dealrisk
--  WHERE statecode = 'Active'
--  GROUP BY lvo_opportunityid
--  ORDER BY active_risks DESC;

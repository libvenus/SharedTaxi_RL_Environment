-- ============================================================================
-- account — View Account user-story additions
--
-- Adds the three new columns the accounts grid + detail view need that aren't
-- already on the base D365 dump:
--
--   lvo_accounttype           Prospect | Customer
--                             (derived from opportunity history — Customer if
--                              the account has at least one Closed Won deal)
--
--   lvo_accountstatus         Active | Inactive | At-Risk
--                             (Active/Inactive mirrors statecode by default;
--                              At-Risk is set by the account-recalc service
--                              when the account has any open deal whose
--                              dealHealth < 50)
--
--   lvo_lastinteractiondate   Most recent lvo_activitydate joined through
--                             opportunity. Cached so the grid can sort/filter
--                             cheaply; refreshed by the account-recalc
--                             service when activities or deals change.
--
-- All three are nullable so a partially-applied migration cannot break the
-- existing /api/accounts/{id} endpoint or the deal-detail account panel.
--
-- Re-running is safe: ADD COLUMN uses IF NOT EXISTS, all UPDATEs are idempotent.
-- ============================================================================

ALTER TABLE account
    ADD COLUMN IF NOT EXISTS lvo_accounttype          TEXT,
    ADD COLUMN IF NOT EXISTS lvo_accountstatus        TEXT,
    ADD COLUMN IF NOT EXISTS lvo_lastinteractiondate  TIMESTAMPTZ;


-- ----------------------------------------------------------------------------
-- CHECK constraints — guard against typos getting persisted by the API.
-- Wrapped in DO blocks so re-running is safe (pg_constraint lookup).
-- ----------------------------------------------------------------------------

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_account_lvo_accounttype'
    ) THEN
        ALTER TABLE account
            ADD CONSTRAINT chk_account_lvo_accounttype
            CHECK (lvo_accounttype IS NULL
                   OR lvo_accounttype IN ('Prospect', 'Customer'));
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_account_lvo_accountstatus'
    ) THEN
        ALTER TABLE account
            ADD CONSTRAINT chk_account_lvo_accountstatus
            CHECK (lvo_accountstatus IS NULL
                   OR lvo_accountstatus IN ('Active', 'Inactive', 'At-Risk'));
    END IF;
END $$;


-- ----------------------------------------------------------------------------
-- Indexes — the View Account grid filters/sorts on these columns.
-- ----------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_account_lvo_accounttype
    ON account (lvo_accounttype);

CREATE INDEX IF NOT EXISTS idx_account_lvo_accountstatus
    ON account (lvo_accountstatus);

CREATE INDEX IF NOT EXISTS idx_account_lvo_lastinteractiondate
    ON account (lvo_lastinteractiondate);

CREATE INDEX IF NOT EXISTS idx_account_owninguser
    ON account (owninguser);


-- ============================================================================
-- Backfill 1 — lvo_accounttype  (Prospect / Customer)
--
-- An account is a Customer the moment it has *any* won deal, otherwise it is
-- a Prospect. This is a one-time seed; the account-recalc service keeps it
-- in sync for new wins.
-- ============================================================================
UPDATE account a
   SET lvo_accounttype = CASE
       WHEN EXISTS (
            SELECT 1
              FROM opportunity o
             WHERE UPPER(o.accountid) = UPPER(a.accountid::TEXT)
               AND (
                    o.statecode IN ('Won', 'Closed Won')
                 OR o.stagename IN ('Closed Won')
               )
       ) THEN 'Customer'
       ELSE 'Prospect'
   END
 WHERE lvo_accounttype IS NULL;


-- ============================================================================
-- Backfill 2 — lvo_accountstatus (Active / Inactive / At-Risk)
--
-- Initial seed only sets Active vs. Inactive from statecode. The At-Risk
-- transition is handled by `app/services/account_recalc.py` once Phase A3
-- ships, because it needs the deal-health scores to be present.
-- ============================================================================
UPDATE account
   SET lvo_accountstatus = CASE
       WHEN COALESCE(statecode, 'Active') = 'Inactive' THEN 'Inactive'
       ELSE 'Active'
   END
 WHERE lvo_accountstatus IS NULL;


-- ============================================================================
-- Backfill 3 — lvo_lastinteractiondate
--
-- Most recent activity for ANY opportunity tied to the account.
-- Skipped when sql/2026_06_create_lvo_activity.sql hasn't been applied yet.
-- ============================================================================
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
         WHERE table_name = 'lvo_activity'
    ) THEN
        EXECUTE $sql$
            UPDATE account a
               SET lvo_lastinteractiondate = sub.last_activity
              FROM (
                  SELECT UPPER(o.accountid)               AS account_key,
                         MAX(act.lvo_activitydate)        AS last_activity
                    FROM opportunity o
                    JOIN lvo_activity act
                      ON UPPER(act.lvo_opportunityid)
                         = UPPER(o.opportunityid::TEXT)
                   WHERE act.statecode = 'Active'
                     AND o.accountid IS NOT NULL
                   GROUP BY UPPER(o.accountid)
              ) sub
             WHERE UPPER(a.accountid::TEXT) = sub.account_key
               AND a.lvo_lastinteractiondate IS NULL;
        $sql$;
    END IF;
END $$;


-- ============================================================================
-- Extend lvo_dealhealthconfig with the account-status knobs.
--
-- We re-use the existing single-row JSONB config table so all tunables stay
-- in one place. The merge uses jsonb || jsonb so previously-tuned values are
-- preserved; only missing keys get the new defaults.
-- ============================================================================
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
         WHERE table_name = 'lvo_dealhealthconfig'
    ) THEN
        UPDATE lvo_dealhealthconfig
           SET lvo_settings = lvo_settings ||
               '{
                    "account_status": {
                        "at_risk_health_threshold":  50,
                        "inactive_idle_days":        180
                    }
                }'::JSONB,
               lvo_updatedat = now()
         WHERE id = 1
           AND NOT (lvo_settings ? 'account_status');
    END IF;
END $$;


-- Sanity check (optional — uncomment after running):
-- SELECT name, lvo_accounttype, lvo_accountstatus, lvo_lastinteractiondate
--   FROM account
--  ORDER BY name
--  LIMIT 20;

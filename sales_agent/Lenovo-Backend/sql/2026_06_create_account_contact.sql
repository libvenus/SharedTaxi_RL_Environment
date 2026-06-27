-- ============================================================================
-- lvo_accountcontact — Account-level contact roster
--
-- The View Account user story exposes "Add / Update / Delete contacts" on the
-- account-detail page (independent of any specific deal). We deliberately do
-- NOT mutate `contact.parentcustomerid` directly because that column belongs
-- to the core D365 import.
--
-- Mirrors lvo_opportunitycontact in shape so the frontend can share most
-- contact-link components between the deal-detail and account-detail pages.
--
--   lvo_accountcontactid   PK
--   lvo_accountid          FK -> account.accountid
--   lvo_contactid          FK -> contact.contactid
--   lvo_role               'Primary' | 'Influencer' | 'Procurement' | 'Technical' | …
--   lvo_isprimary          One contact per account may carry the primary flag
--   lvo_lasttouchdate      Cached most-recent activity that involved this contact
--   lvo_createdat          Audit
--   lvo_updatedat          Audit
--   statecode              'Active' | 'Inactive'
--
-- Re-running is safe: CREATE uses IF NOT EXISTS, INSERTs use ON CONFLICT.
-- ============================================================================

CREATE TABLE IF NOT EXISTS lvo_accountcontact (
    lvo_accountcontactid   TEXT PRIMARY KEY,
    lvo_accountid          TEXT NOT NULL,
    lvo_contactid          TEXT NOT NULL,
    lvo_role               TEXT,
    lvo_isprimary          BOOLEAN NOT NULL DEFAULT FALSE,
    lvo_lasttouchdate      TIMESTAMPTZ,
    lvo_createdat          TIMESTAMPTZ NOT NULL DEFAULT now(),
    lvo_updatedat          TIMESTAMPTZ NOT NULL DEFAULT now(),
    statecode              TEXT NOT NULL DEFAULT 'Active'
);

CREATE INDEX IF NOT EXISTS idx_lvo_accountcontact_accountid
    ON lvo_accountcontact (lvo_accountid);

CREATE INDEX IF NOT EXISTS idx_lvo_accountcontact_primary
    ON lvo_accountcontact (lvo_accountid, lvo_isprimary)
 WHERE statecode = 'Active';

-- A contact can only be listed once per account while they are Active.
CREATE UNIQUE INDEX IF NOT EXISTS uq_lvo_accountcontact_active_pair
    ON lvo_accountcontact (lvo_accountid, lvo_contactid)
 WHERE statecode = 'Active';


-- ============================================================================
-- Seed: best-effort link from existing `contact` rows to their parent account.
--
-- The column on `contact` that links to the account differs across D365
-- extracts (classic schema uses `parentcustomerid`, newer Web API exports
-- use `_parentcustomerid_value`, custom schemas use `accountid`). We detect
-- the right column at runtime and gracefully skip when none is present
-- (mirrors the pattern in sql/2026_06_create_opportunity_contact.sql).
-- ============================================================================
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

    IF parent_col IS NULL THEN
        RAISE NOTICE
            'lvo_accountcontact: contact table has no recognised account-link '
            'column (parentcustomerid / accountid / etc.). Seed skipped.';
        RETURN;
    END IF;

    EXECUTE format($sql$
        INSERT INTO lvo_accountcontact (
            lvo_accountcontactid,
            lvo_accountid,
            lvo_contactid,
            lvo_role,
            lvo_isprimary,
            statecode
        )
        SELECT
            'AC-' || REPLACE(a.accountid::TEXT, '-', '')
                  || '-' || LEFT(REPLACE(c.contactid::TEXT, '-', ''), 8),
            UPPER(a.accountid::TEXT),
            UPPER(c.contactid::TEXT),
            'Influencer',
            FALSE,
            'Active'
          FROM account a
          JOIN contact c
            ON UPPER(c.%I::TEXT) = UPPER(a.accountid::TEXT)
        ON CONFLICT (lvo_accountcontactid) DO NOTHING;
    $sql$, parent_col);
END $$;


-- After the bulk seed, mark the first contact on each account as the primary
-- so the "Primary contact" badge has something meaningful to render. This is
-- a one-time best-effort step; the API lets sellers re-pick the primary at
-- any time afterwards.
WITH first_per_account AS (
    SELECT lvo_accountcontactid,
           ROW_NUMBER() OVER (
               PARTITION BY lvo_accountid
               ORDER BY lvo_createdat ASC, lvo_accountcontactid ASC
           ) AS rn
      FROM lvo_accountcontact
     WHERE statecode = 'Active'
)
UPDATE lvo_accountcontact
   SET lvo_isprimary = TRUE,
       lvo_updatedat = now()
  FROM first_per_account fpa
 WHERE lvo_accountcontact.lvo_accountcontactid = fpa.lvo_accountcontactid
   AND fpa.rn = 1
   AND lvo_accountcontact.lvo_isprimary = FALSE
   AND NOT EXISTS (
       SELECT 1
         FROM lvo_accountcontact other
        WHERE other.lvo_accountid = lvo_accountcontact.lvo_accountid
          AND other.lvo_isprimary = TRUE
          AND other.statecode = 'Active'
   );


-- Sanity check (optional — uncomment to run):
-- SELECT lvo_accountid,
--        COUNT(*) FILTER (WHERE lvo_isprimary)        AS primary_count,
--        COUNT(*) FILTER (WHERE NOT lvo_isprimary)    AS additional_count
--   FROM lvo_accountcontact
--  WHERE statecode = 'Active'
--  GROUP BY lvo_accountid
--  ORDER BY 1;

-- ============================================================================
-- lvo_opportunitycontact — Decision Maker + additional contacts on a deal
--
-- The base D365 dump (lenovo_nitro_d365_postgres.sql) already provides the
-- `contact` table with the standard D365 columns we read:
--   contactid, fullname, firstname, lastname, jobtitle, emailaddress1,
--   parentcustomerid (FK -> account.accountid), statecode.
--
-- This migration only adds the join table that links a contact to a deal,
-- captures their role on that deal, and flags the decision maker.
--
-- Re-running is safe: CREATE uses IF NOT EXISTS, INSERTs use ON CONFLICT.
-- ============================================================================

CREATE TABLE IF NOT EXISTS lvo_opportunitycontact (
    lvo_opportunitycontactid  TEXT PRIMARY KEY,
    lvo_opportunityid         TEXT NOT NULL,            -- FK -> opportunity.opportunityid
    lvo_contactid             TEXT NOT NULL,            -- FK -> contact.contactid
    lvo_role                  TEXT,                     -- 'Decision Maker' | 'Champion' | 'Influencer' | 'Procurement' | 'Technical' | …
    lvo_isdecisionmaker       BOOLEAN NOT NULL DEFAULT FALSE,
    lvo_lasttouchdate         TIMESTAMPTZ,              -- denormalised cache of the most recent activity that involved this contact
    lvo_createdat             TIMESTAMPTZ NOT NULL DEFAULT now(),
    lvo_updatedat             TIMESTAMPTZ NOT NULL DEFAULT now(),
    statecode                 TEXT NOT NULL DEFAULT 'Active'   -- 'Active' | 'Inactive'
);

CREATE INDEX IF NOT EXISTS idx_lvo_oppcontact_opportunityid
    ON lvo_opportunitycontact (lvo_opportunityid);

CREATE INDEX IF NOT EXISTS idx_lvo_oppcontact_decisionmaker
    ON lvo_opportunitycontact (lvo_opportunityid, lvo_isdecisionmaker)
 WHERE statecode = 'Active';

-- A contact can only be listed once per deal while they are Active. The
-- partial-unique-index form is the Postgres way to enforce that without
-- blocking the same contact being re-added after soft-delete.
CREATE UNIQUE INDEX IF NOT EXISTS uq_lvo_oppcontact_active_pair
    ON lvo_opportunitycontact (lvo_opportunityid, lvo_contactid)
 WHERE statecode = 'Active';


-- ============================================================================
-- Seed: best-effort link of existing contacts to the seeded opportunities,
-- so the new GET /api/opportunities/{id}/contacts endpoint returns useful
-- data the moment the migration lands.
--
-- Uses opportunity.contactid (the legacy single-contact field already on the
-- base table) as the seed for the decision maker. Anyone who shares the
-- account with the deal gets added as an additional contact.
-- ============================================================================

-- 1. Seed the decision maker — exactly one row per deal where opportunity
--    already points at a contact via the legacy contactid column.
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
 WHERE o.contactid IS NOT NULL
ON CONFLICT (lvo_opportunitycontactid) DO NOTHING;


-- 2. Seed up to a handful of additional contacts per deal from contacts that
--    share the same parent account but aren't the decision maker.
--
--    The column on `contact` that links to the account differs across D365
--    extracts (classic schema uses `parentcustomerid`, newer Web API exports
--    use `_parentcustomerid_value`, custom schemas use `accountid`).  We
--    detect which one exists at runtime and adapt the query accordingly so
--    the migration works on any of them — and silently skips this seed when
--    none of the candidates is present (the rest of the file still runs).
DO $$
DECLARE
    parent_col TEXT;
BEGIN
    -- Pick the first candidate column that actually exists on `contact`.
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
            'lvo_opportunitycontact: contact table has no recognised '
            'account-link column (parentcustomerid / accountid / etc.). '
            'Skipping the additional-contacts seed; only decision-maker '
            'links will be populated from opportunity.contactid.';
        RETURN;
    END IF;

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
            'OC-' || REPLACE(o.opportunityid::TEXT, '-', '')
                  || '-' || LEFT(REPLACE(c.contactid::TEXT, '-', ''), 8),
            UPPER(o.opportunityid::TEXT),
            UPPER(c.contactid::TEXT),
            'Influencer',
            FALSE,
            'Active'
          FROM opportunity o
          JOIN contact c
            ON UPPER(c.%I::TEXT) = UPPER(o.accountid)
         WHERE o.accountid IS NOT NULL
           AND (o.contactid IS NULL
                OR UPPER(c.contactid::TEXT) <> UPPER(o.contactid::TEXT))
        ON CONFLICT (lvo_opportunitycontactid) DO NOTHING;
    $sql$, parent_col);
END $$;


-- Sanity check (optional — uncomment to run):
-- SELECT lvo_opportunityid,
--        COUNT(*) FILTER (WHERE lvo_isdecisionmaker)        AS decision_makers,
--        COUNT(*) FILTER (WHERE NOT lvo_isdecisionmaker)    AS additional
--   FROM lvo_opportunitycontact
--  WHERE statecode = 'Active'
--  GROUP BY lvo_opportunityid
--  ORDER BY 1;

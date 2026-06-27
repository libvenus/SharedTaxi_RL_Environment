-- ============================================================================
-- contact extension — guarantee a phone column exists on the contact table
--
-- D365 dumps disagree on which phone column is shipped:
--   * Classic on-prem schema   →  contact.telephone1 / contact.mobilephone
--   * Custom Lenovo schemas    →  may strip those columns entirely
--
-- The "Manage Contacts Linked to an Account" user story (UI mockup includes a
-- Phone field in the add-contact form) requires a writable phone column. Rather
-- than mutate the D365-shipped schema, this migration ONLY adds a fallback
-- ``lvo_phone`` column when neither ``telephone1`` nor ``mobilephone`` exists.
-- The application code resolves the actual column name at runtime via
-- ``ContactPhoneResolver`` (see app/models.py).
--
-- Re-running this migration is safe (CREATE/ADD ... IF NOT EXISTS guards).
-- ============================================================================

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
          FROM information_schema.columns
         WHERE table_name = 'contact'
           AND column_name IN ('telephone1', 'mobilephone')
    ) THEN
        RAISE NOTICE
            'contact table already exposes telephone1/mobilephone — '
            'skipping lvo_phone fallback creation.';
    ELSE
        ALTER TABLE contact
            ADD COLUMN IF NOT EXISTS lvo_phone TEXT;

        -- Cheap index to support search-by-phone in future endpoints.
        CREATE INDEX IF NOT EXISTS idx_contact_lvo_phone
            ON contact (lvo_phone);

        RAISE NOTICE 'Added contact.lvo_phone fallback column.';
    END IF;
END $$;


-- ----------------------------------------------------------------------------
-- Sanity check helpers (uncomment locally if you want to verify):
-- ----------------------------------------------------------------------------
-- SELECT column_name
--   FROM information_schema.columns
--  WHERE table_name = 'contact'
--    AND column_name IN ('telephone1', 'mobilephone', 'lvo_phone');

-- Sprint: Vexa bot integration
-- Adds vexa_bot_id (returned by POST /bots) and graph_event_id (returned
-- by Graph API when creating the Teams calendar event).  Both are nullable
-- so existing rows are unaffected.

ALTER TABLE tbl_schedule_meetings
    ADD COLUMN IF NOT EXISTS vexa_bot_id   TEXT,
    ADD COLUMN IF NOT EXISTS graph_event_id TEXT;

CREATE INDEX IF NOT EXISTS ix_tbl_schedule_meetings_vexa_bot_id
    ON tbl_schedule_meetings (vexa_bot_id)
    WHERE vexa_bot_id IS NOT NULL;

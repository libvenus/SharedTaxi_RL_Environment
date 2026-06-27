-- ============================================================================
-- lvo_notification_read — read-state for Sprint 2 "What Changed" feed items
--
-- Feed events are aggregated at query time from lvo_activity, lvo_audit_log,
-- lvo_dealrisk, and lvo_nextaction. This table only tracks which synthetic
-- feed keys a seller has marked as read in the notification panel.
--
-- feed_item_key examples:
--   activity:<lvo_activityid>
--   audit:<lvo_auditlogid>:<field>
--   risk:<lvo_dealriskid>
--   task:<lvo_nextactionid>
--
-- Idempotent — safe to re-run.
-- ============================================================================

CREATE TABLE IF NOT EXISTS lvo_notification_read (
    lvo_notificationreadid  UUID PRIMARY KEY,
    seller_id               TEXT NOT NULL,
    feed_item_key           TEXT NOT NULL,
    read_at                 TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_notification_read_seller_key
    ON lvo_notification_read (seller_id, feed_item_key);

CREATE INDEX IF NOT EXISTS idx_notification_read_seller
    ON lvo_notification_read (seller_id);

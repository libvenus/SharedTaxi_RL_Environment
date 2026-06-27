-- Sprint 2 · US 1.3 — seller scope for Execute Workspace To-Do list
-- Run on AIBackend Postgres after deploying the model change.

ALTER TABLE tbl_to_do_list
    ADD COLUMN IF NOT EXISTS seller_id UUID NULL;

COMMENT ON COLUMN tbl_to_do_list.seller_id IS
    'Seller UUID (D365 systemuser) — scopes manual and meeting-promoted todos.';

CREATE INDEX IF NOT EXISTS idx_tbl_to_do_list_seller_id
    ON tbl_to_do_list (seller_id);

CREATE INDEX IF NOT EXISTS idx_tbl_to_do_list_seller_status_due
    ON tbl_to_do_list (seller_id, status, due_date);

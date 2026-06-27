-- ============================================================================
-- lvo_seller_quota — Sprint 2 US 1.2 Quarter Pulse quota targets
--
-- Stores per-seller fiscal-quarter quota when D365 has no goal row yet, or
-- when a seller manager enters a Phase-1 override via the admin UI.
--
-- Idempotent — safe to re-run.
-- ============================================================================

CREATE TABLE IF NOT EXISTS lvo_seller_quota (
    lvo_sellerquotaid  UUID PRIMARY KEY,
    seller_id          TEXT NOT NULL,
    fiscal_year        INTEGER NOT NULL,
    fiscal_quarter     INTEGER NOT NULL CHECK (fiscal_quarter BETWEEN 1 AND 4),
    quota_amount       NUMERIC(20, 2) NOT NULL CHECK (quota_amount > 0),
    currency_code      TEXT NOT NULL DEFAULT 'USD',
    source             TEXT NOT NULL DEFAULT 'manual'
                           CHECK (source IN ('d365', 'manual')),
    set_by             TEXT,
    modified_at        TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_seller_quota_period
    ON lvo_seller_quota (seller_id, fiscal_year, fiscal_quarter);

CREATE INDEX IF NOT EXISTS idx_seller_quota_seller
    ON lvo_seller_quota (seller_id);

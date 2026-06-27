-- Local smoke seed for Quarter Pulse manual quota (optional).
-- Replace SELLER-UUID with a real opportunity.owninguser from your dump.
--
--   psql ... -f sql/2026_08_seed_quarter_pulse_local_smoke.sql

DELETE FROM lvo_seller_quota
 WHERE seller_id = 'SELLER-UUID-REPLACE-ME';

INSERT INTO lvo_seller_quota (
    lvo_sellerquotaid,
    seller_id,
    fiscal_year,
    fiscal_quarter,
    quota_amount,
    currency_code,
    source,
    set_by,
    modified_at
) VALUES (
    '11111111-1111-1111-1111-111111111111',
    'SELLER-UUID-REPLACE-ME',
    2026,
    3,
    1000000.00,
    'USD',
    'manual',
    'local-smoke',
    NOW()
) ON CONFLICT DO NOTHING;

-- Sprint 2 · US 1.2 — Quarter Pulse demo seed (Home "Q1 Pulse" card)
--
-- Seller: Rachel Green — 055DAFE7-9840-451D-8328-5F70A6326C03
-- Fiscal: FY2027 Q1 = 2026-04-01 .. 2026-06-30 (Lenovo FY starts April)
--
-- If pgAdmin shows: "current transaction is aborted" (SQL state 25P02)
--   Run once:  ROLLBACK;
--   Then re-run this entire script.
--
-- Verify:
--   curl "http://10.245.240.33/api/quarter-pulse?sellerId=055DAFE7-9840-451D-8328-5F70A6326C03"

-- 1. Quota — $5M for FY2027 Q1 (fixes UI "Not set")
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
    '11111111-1111-4111-8111-111111111111'::uuid,
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    2027,
    1,
    5000000.00,
    'USD',
    'manual',
    'quarter-pulse-demo-seed',
    NOW()
)
ON CONFLICT (seller_id, fiscal_year, fiscal_quarter)
DO UPDATE SET
    quota_amount = EXCLUDED.quota_amount,
    currency_code = EXCLUDED.currency_code,
    source = EXCLUDED.source,
    set_by = EXCLUDED.set_by,
    modified_at = EXCLUDED.modified_at;

-- 2a. Remove extra Won deals in Q1 for this seller (they inflate attainment above 100%)
UPDATE opportunity
   SET statecode          = 'Open',
       stagename          = 'Propose',
       estimatedclosedate = '2026-09-15'
 WHERE UPPER(owninguser::TEXT) = '055DAFE7-9840-451D-8328-5F70A6326C03'
   AND (
       statecode IN ('Won', 'Closed Won')
       OR stagename IN ('Closed Won', 'Won')
   )
   AND UPPER(opportunityid::TEXT) NOT IN (
       'B0000001-0001-0001-0001-000000000038',
       'B0000001-0001-0001-0001-000000000030'
   );

-- 2b. Closed revenue in Q1: $3.0M total → 60% of $5M quota
UPDATE opportunity
   SET statecode            = 'Won',
       stagename            = 'Closed Won',
       estimatedvalue       = 1800000,
       estimatedclosedate   = '2026-05-15',
       owninguser           = '055DAFE7-9840-451D-8328-5F70A6326C03'
 WHERE UPPER(opportunityid::TEXT) = 'B0000001-0001-0001-0001-000000000038';

UPDATE opportunity
   SET statecode            = 'Won',
       stagename            = 'Closed Won',
       estimatedvalue       = 1200000,
       estimatedclosedate   = '2026-06-10',
       owninguser           = '055DAFE7-9840-451D-8328-5F70A6326C03'
 WHERE UPPER(opportunityid::TEXT) = 'B0000001-0001-0001-0001-000000000030';

-- 3. Open pipeline ~$4.75M → ~2.4× coverage on remaining quota
UPDATE opportunity
   SET owninguser = 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2'
 WHERE UPPER(owninguser::TEXT) = '055DAFE7-9840-451D-8328-5F70A6326C03'
   AND statecode = 'Open'
   AND UPPER(opportunityid::TEXT) NOT IN (
       'B0000001-0001-0001-0001-000000000024',
       'B0000001-0001-0001-0001-000000000032',
       'B0000001-0001-0001-0001-000000000048'
   );

UPDATE opportunity
   SET statecode = 'Open', stagename = 'Qualify',  estimatedvalue = 1450000,
       owninguser = '055DAFE7-9840-451D-8328-5F70A6326C03'
 WHERE UPPER(opportunityid::TEXT) = 'B0000001-0001-0001-0001-000000000024';

UPDATE opportunity
   SET statecode = 'Open', stagename = 'Develop', estimatedvalue = 1850000,
       owninguser = '055DAFE7-9840-451D-8328-5F70A6326C03'
 WHERE UPPER(opportunityid::TEXT) = 'B0000001-0001-0001-0001-000000000032';

UPDATE opportunity
   SET statecode = 'Open', stagename = 'Qualify',  estimatedvalue = 1450000,
       owninguser = '055DAFE7-9840-451D-8328-5F70A6326C03'
 WHERE UPPER(opportunityid::TEXT) = 'B0000001-0001-0001-0001-000000000048';

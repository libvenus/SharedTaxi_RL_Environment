-- ============================================================================
-- Bulk seed: +15 accounts, +50 opportunities, +supporting rows
--
-- Adds enough variety to exercise pagination, KPI cards, every filter and
-- the activity timeline. Everything is keyed by deterministic UUIDs so the
-- script is fully idempotent (ON CONFLICT DO NOTHING for INSERTs).
--
-- Prerequisites:
--   1. lenovo_nitro_d365_postgres.sql          (base DDL + 5 sample opps)
--   2. sql/2026_06_create_lvo_activity.sql     (activity timeline table)
--   3. sql/2026_06_add_dealhealth.sql          (opportunity health / risk columns)
--
-- Re-running is safe.
-- ============================================================================


-- ============================================================================
-- 1. New accounts  (15 rows)
-- ============================================================================
INSERT INTO account (
    accountid, accountnumber, name, industrycode, lvo_segment,
    lvo_countryid, lvo_businessgroupid, lvo_defaultcurrency,
    revenue, numberofemployees, owninguser, statecode
) VALUES
    ('A0000001-AAAA-0001-0001-000000000001', 'ACC-2025-006', 'JPMorgan Chase & Co.',     'Financial Services',      'Strategic',  'US', 'Americas BG', 'USD', 158000000000, 300000, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('A0000001-AAAA-0001-0001-000000000002', 'ACC-2025-007', 'Tesla Inc.',                'Automotive',              'Strategic',  'US', 'Americas BG', 'USD',  96000000000, 140000, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('A0000001-AAAA-0001-0001-000000000003', 'ACC-2025-008', 'Walmart Inc.',              'Retail',                  'Strategic',  'US', 'Americas BG', 'USD', 611000000000, 2100000,'81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('A0000001-AAAA-0001-0001-000000000004', 'ACC-2025-009', 'Bayer AG',                  'Pharmaceuticals',         'Strategic',  'DE', 'EMEA BG',     'EUR',  47000000000, 100000, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('A0000001-AAAA-0001-0001-000000000005', 'ACC-2025-010', 'Volkswagen AG',             'Automotive',              'Strategic',  'DE', 'EMEA BG',     'EUR', 322000000000, 670000, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('A0000001-AAAA-0001-0001-000000000006', 'ACC-2025-011', 'Toyota Motor Corp.',        'Automotive',              'Strategic',  'JP', 'APAC BG',     'JPY', 287000000000, 375000, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('A0000001-AAAA-0001-0001-000000000007', 'ACC-2025-012', 'Samsung Electronics',       'Technology',              'Strategic',  'KR', 'APAC BG',     'USD', 232000000000, 270000, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('A0000001-AAAA-0001-0001-000000000008', 'ACC-2025-013', 'Pfizer Inc.',               'Pharmaceuticals',         'Strategic',  'US', 'Americas BG', 'USD',  58000000000, 88000,  '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('A0000001-AAAA-0001-0001-000000000009', 'ACC-2025-014', 'Shell plc',                 'Energy',                  'Strategic',  'GB', 'EMEA BG',     'USD', 381000000000, 103000, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('A0000001-AAAA-0001-0001-000000000010', 'ACC-2025-015', 'Tata Consultancy Services', 'IT Services',             'Enterprise', 'IN', 'APAC BG',     'USD',  27800000000, 614000, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('A0000001-AAAA-0001-0001-000000000011', 'ACC-2025-016', 'Maersk',                    'Logistics',               'Enterprise', 'DK', 'EMEA BG',     'USD',  51000000000, 110000, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('A0000001-AAAA-0001-0001-000000000012', 'ACC-2025-017', 'Commonwealth Bank',         'Financial Services',      'Enterprise', 'AU', 'APAC BG',     'AUD',  18000000000, 49000,  '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('A0000001-AAAA-0001-0001-000000000013', 'ACC-2025-018', 'BNP Paribas',               'Financial Services',      'Strategic',  'FR', 'EMEA BG',     'EUR',  52000000000, 183000, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('A0000001-AAAA-0001-0001-000000000014', 'ACC-2025-019', 'AT&T Inc.',                 'Telecommunications',      'Strategic',  'US', 'Americas BG', 'USD', 120000000000, 161000, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('A0000001-AAAA-0001-0001-000000000015', 'ACC-2025-020', 'Airbus SE',                 'Aerospace',               'Strategic',  'FR', 'EMEA BG',     'EUR',  65000000000, 134000, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active')
ON CONFLICT (accountid) DO NOTHING;


-- ============================================================================
-- 2. New opportunities  (50 rows)
--
-- Distribution chosen to exercise the UI:
--   Stages    : Qualify=8, Develop=12, Propose=12, Execute=8, Closed Won=7, Closed Lost=3
--   Forecast  : Pipeline=10, Best Case=11, Most Likely=12, Commit=11, NULL=6
--   Motion    : Net-New=25, Expansion=15, Renewal=10
--   Regions   : EMEA=20, APAC=18, Americas=12
--   Countries : DE FR GB ES IT NL SE DK CH | US CA MX BR | IN HK JP KR SG AU
--   Health    : populated for every row; ~30% have a risk badge
-- ============================================================================
INSERT INTO opportunity (
    opportunityid, name, accountid, stagename,
    estimatedvalue, estimatedclosedate,
    lvo_forecastcategory, lvo_salesmotion,
    lvo_country, lvo_businessgroup,
    closeprobability, statecode, owninguser,
    lvo_dealhealthscore, lvo_riskscore, lvo_riskreason
) VALUES
    -- ── EMEA BG (20 rows) ────────────────────────────────────────────────
    ('B0000001-0001-0001-0001-000000000001', 'JPMorgan – Trader Workstation Refresh',     '6DC95C38-9237-4CE9-84D3-F5D1F7431965', 'Qualify',    1850000, '2026-11-15', 'Pipeline',     'Net-New',   'DE', 'EMEA BG',  30, 'Open', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2',  55, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000002', 'Bayer – R&D Compute Cluster',                'A0000001-AAAA-0001-0001-000000000004', 'Develop',    3450000, '2026-09-28', 'Best Case',    'Net-New',   'DE', 'EMEA BG',  45, 'Open', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 72, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000003', 'Volkswagen – Plant Floor Devices',           'A0000001-AAAA-0001-0001-000000000005', 'Propose',    5200000, '2026-08-18', 'Most Likely',  'Expansion', 'DE', 'EMEA BG',  60, 'Open', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 81, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000004', 'Shell – Drilling Site Rugged Laptops',       'A0000001-AAAA-0001-0001-000000000009', 'Develop',    2400000, '2026-10-05', 'Best Case',    'Expansion', 'GB', 'EMEA BG',  50, 'Open', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 48,  3,    'Stale Activity'),
    ('B0000001-0001-0001-0001-000000000005', 'Maersk – Vessel Crew Devices',               'A0000001-AAAA-0001-0001-000000000011', 'Execute',    1750000, '2026-07-10', 'Commit',       'Renewal',   'DK', 'EMEA BG',  85, 'Open', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 88, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000006', 'BNP Paribas – Branch Endpoint Refresh',      'A0000001-AAAA-0001-0001-000000000013', 'Propose',    4100000, '2026-09-04', 'Most Likely',  'Renewal',   'FR', 'EMEA BG',  65, 'Open', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 70, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000007', 'Airbus – Engineering Workstations',          'A0000001-AAAA-0001-0001-000000000015', 'Develop',    6800000, '2026-12-02', 'Pipeline',     'Net-New',   'FR', 'EMEA BG',  35, 'Open', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 60,  2,    'Long Cycle Risk'),
    ('B0000001-0001-0001-0001-000000000008', 'Siemens – Annual Renewal Add-on',            'D4D580CA-EDF7-46A9-9BA7-C4114B83A3A6', 'Qualify',    1200000, '2026-11-22', 'Pipeline',     'Renewal',   'DE', 'EMEA BG',  25, 'Open', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 52, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000009', 'Deutsche Bank – AI Trader Tools',            '6DC95C38-9237-4CE9-84D3-F5D1F7431965', 'Develop',    3200000, '2026-09-15', 'Best Case',    'Expansion', 'DE', 'EMEA BG',  50, 'Open', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 66, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000010', 'Bayer – Lab Tablet Rollout',                 'A0000001-AAAA-0001-0001-000000000004', 'Propose',    1950000, '2026-08-28', 'Commit',       'Net-New',   'DE', 'EMEA BG',  70, 'Open', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 78, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000011', 'Volkswagen – Plant 2 Expansion',             'A0000001-AAAA-0001-0001-000000000005', 'Execute',    4250000, '2026-06-30', 'Commit',       'Expansion', 'DE', 'EMEA BG',  90, 'Open', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 91, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000012', 'Shell – HQ Office Refresh',                  'A0000001-AAAA-0001-0001-000000000009', 'Propose',    2850000, '2026-09-12', 'Most Likely',  'Net-New',   'GB', 'EMEA BG',  60, 'Open', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 73, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000013', 'Maersk – Port Operations Workstations',      'A0000001-AAAA-0001-0001-000000000011', 'Develop',    1620000, '2026-10-22', 'Best Case',    'Net-New',   'NL', 'EMEA BG',  40, 'Open', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 58, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000014', 'BNP Paribas – Trading Floor Upgrade',        'A0000001-AAAA-0001-0001-000000000013', 'Execute',    5650000, '2026-07-22', 'Commit',       'Net-New',   'FR', 'EMEA BG',  85, 'Open', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 87, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000015', 'Airbus – CAD Workstation Refresh',           'A0000001-AAAA-0001-0001-000000000015', 'Closed Won', 8200000, '2026-05-10', NULL,           'Net-New',   'FR', 'EMEA BG', 100, 'Won',  'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 100,NULL, NULL),
    ('B0000001-0001-0001-0001-000000000016', 'Siemens – Edge AI Lab Phase 2',              'D4D580CA-EDF7-46A9-9BA7-C4114B83A3A6', 'Propose',    2950000, '2026-09-29', 'Most Likely',  'Expansion', 'DE', 'EMEA BG',  65, 'Open', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 64,  3,    'Budget Freeze'),
    ('B0000001-0001-0001-0001-000000000017', 'Bayer – Salesforce Endpoint Refresh',        'A0000001-AAAA-0001-0001-000000000004', 'Qualify',     980000, '2026-12-15', 'Pipeline',     'Renewal',   'CH', 'EMEA BG',  20, 'Open', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 50, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000018', 'Shell – Refinery Field Devices',             'A0000001-AAAA-0001-0001-000000000009', 'Closed Lost',3400000, '2026-04-20', NULL,           'Net-New',   'GB', 'EMEA BG',   0, 'Lost', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 25,  5,    'Competitor Won'),
    ('B0000001-0001-0001-0001-000000000019', 'Airbus – Toulouse Site Standardisation',     'A0000001-AAAA-0001-0001-000000000015', 'Develop',    3850000, '2026-10-15', 'Best Case',    'Expansion', 'ES', 'EMEA BG',  45, 'Open', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 67, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000020', 'Volkswagen – Design Studio HW',              'A0000001-AAAA-0001-0001-000000000005', 'Closed Won', 2150000, '2026-04-30', NULL,           'Renewal',   'IT', 'EMEA BG', 100, 'Won',  'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 100,NULL, NULL),

    -- ── APAC BG (18 rows) ────────────────────────────────────────────────
    ('B0000001-0001-0001-0001-000000000021', 'Toyota – Plant Floor IoT Devices',           'A0000001-AAAA-0001-0001-000000000006', 'Develop',    4400000, '2026-10-08', 'Best Case',    'Net-New',   'JP', 'APAC BG',  45, 'Open', '7D26391E-D020-474E-B1CA-53E6B6C71487', 68, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000022', 'Samsung – Internal Developer Workstations',  'A0000001-AAAA-0001-0001-000000000007', 'Propose',    3700000, '2026-08-25', 'Most Likely',  'Expansion', 'KR', 'APAC BG',  60, 'Open', '7D26391E-D020-474E-B1CA-53E6B6C71487', 76, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000023', 'TCS – Pune Office Refresh',                  'A0000001-AAAA-0001-0001-000000000010', 'Execute',    2200000, '2026-07-05', 'Commit',       'Renewal',   'IN', 'APAC BG',  90, 'Open', '7D26391E-D020-474E-B1CA-53E6B6C71487', 89, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000024', 'Commonwealth Bank – Branch Endpoint Refresh','A0000001-AAAA-0001-0001-000000000012', 'Qualify',    1450000, '2026-12-08', 'Pipeline',     'Renewal',   'AU', 'APAC BG',  25, 'Open', '055DAFE7-9840-451D-8328-5F70A6326C03', 48, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000025', 'HSBC – Singapore Trading Floor',             'E04F3BC9-2FCD-42E7-87C2-636B6328EFA8', 'Develop',    3950000, '2026-10-18', 'Best Case',    'Expansion', 'SG', 'APAC BG',  50, 'Open', '7D26391E-D020-474E-B1CA-53E6B6C71487', 71, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000026', 'Infosys – Bangalore HPC Expansion',          '1C603E6A-C4EA-4E97-B73E-345A5A9C2460', 'Propose',    2950000, '2026-09-08', 'Most Likely',  'Expansion', 'IN', 'APAC BG',  65, 'Open', '055DAFE7-9840-451D-8328-5F70A6326C03', 74, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000027', 'Toyota – Kentucky Plant Mirror',             'A0000001-AAAA-0001-0001-000000000006', 'Qualify',    1850000, '2026-11-30', 'Pipeline',     'Net-New',   'JP', 'APAC BG',  30, 'Open', '7D26391E-D020-474E-B1CA-53E6B6C71487', 54, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000028', 'Samsung – Display Plant Tablets',            'A0000001-AAAA-0001-0001-000000000007', 'Execute',    3150000, '2026-06-25', 'Commit',       'Net-New',   'KR', 'APAC BG',  85, 'Open', '7D26391E-D020-474E-B1CA-53E6B6C71487', 86, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000029', 'TCS – Hyderabad Tower Standardisation',      'A0000001-AAAA-0001-0001-000000000010', 'Develop',    2750000, '2026-10-12', 'Best Case',    'Expansion', 'IN', 'APAC BG',  45, 'Open', '055DAFE7-9840-451D-8328-5F70A6326C03', 65, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000030', 'Commonwealth Bank – Workstation Refresh',    'A0000001-AAAA-0001-0001-000000000012', 'Propose',    2400000, '2026-09-18', 'Commit',       'Renewal',   'AU', 'APAC BG',  70, 'Open', '055DAFE7-9840-451D-8328-5F70A6326C03', 80, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000031', 'HSBC – Hong Kong Compute Refresh',           'E04F3BC9-2FCD-42E7-87C2-636B6328EFA8', 'Closed Won', 4250000, '2026-05-05', NULL,           'Renewal',   'HK', 'APAC BG', 100, 'Won',  '7D26391E-D020-474E-B1CA-53E6B6C71487', 100,NULL, NULL),
    ('B0000001-0001-0001-0001-000000000032', 'Infosys – Mysore Campus Devices',            '1C603E6A-C4EA-4E97-B73E-345A5A9C2460', 'Develop',    1850000, '2026-10-28', 'Best Case',    'Expansion', 'IN', 'APAC BG',  40, 'Open', '055DAFE7-9840-451D-8328-5F70A6326C03', 62, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000033', 'TCS – Mumbai Tower Refresh',                 'A0000001-AAAA-0001-0001-000000000010', 'Propose',    3050000, '2026-09-22', 'Most Likely',  'Renewal',   'IN', 'APAC BG',  65, 'Open', '055DAFE7-9840-451D-8328-5F70A6326C03', 75, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000034', 'Samsung – Tokyo R&D Cluster',                'A0000001-AAAA-0001-0001-000000000007', 'Qualify',    2400000, '2026-12-22', 'Pipeline',     'Net-New',   'JP', 'APAC BG',  20, 'Open', '7D26391E-D020-474E-B1CA-53E6B6C71487', 49, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000035', 'Toyota – Thailand Plant Tablets',            'A0000001-AAAA-0001-0001-000000000006', 'Execute',    1980000, '2026-07-12', 'Commit',       'Net-New',   'JP', 'APAC BG',  85, 'Open', '7D26391E-D020-474E-B1CA-53E6B6C71487', 90, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000036', 'Commonwealth Bank – Trader Devices',         'A0000001-AAAA-0001-0001-000000000012', 'Closed Lost',1750000, '2026-04-10', NULL,           'Net-New',   'AU', 'APAC BG',   0, 'Lost', '055DAFE7-9840-451D-8328-5F70A6326C03', 22,  4,    'Pricing Mismatch'),
    ('B0000001-0001-0001-0001-000000000037', 'HSBC – APAC Branch Renewal',                 'E04F3BC9-2FCD-42E7-87C2-636B6328EFA8', 'Develop',    3650000, '2026-10-26', 'Most Likely',  'Renewal',   'SG', 'APAC BG',  55, 'Open', '7D26391E-D020-474E-B1CA-53E6B6C71487', 70, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000038', 'Infosys – Pune Tower Expansion',             '1C603E6A-C4EA-4E97-B73E-345A5A9C2460', 'Closed Won', 2950000, '2026-04-25', NULL,           'Expansion', 'IN', 'APAC BG', 100, 'Won',  '055DAFE7-9840-451D-8328-5F70A6326C03', 100,NULL, NULL),

    -- ── Americas BG (12 rows) ────────────────────────────────────────────
    ('B0000001-0001-0001-0001-000000000039', 'JPMorgan – NY HQ Workstation Refresh',       'A0000001-AAAA-0001-0001-000000000001', 'Propose',    6750000, '2026-08-12', 'Most Likely',  'Net-New',   'US', 'Americas BG',  65, 'Open', '81AADFDB-1817-425C-A5B1-45F383F230CE', 82, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000040', 'Tesla – Fremont Plant Tablets',              'A0000001-AAAA-0001-0001-000000000002', 'Develop',    3650000, '2026-10-22', 'Best Case',    'Net-New',   'US', 'Americas BG',  45, 'Open', '81AADFDB-1817-425C-A5B1-45F383F230CE', 69,  2,    'Stale Activity'),
    ('B0000001-0001-0001-0001-000000000041', 'Walmart – Store Endpoint Refresh',           'A0000001-AAAA-0001-0001-000000000003', 'Execute',    5450000, '2026-06-18', 'Commit',       'Renewal',   'US', 'Americas BG',  90, 'Open', '81AADFDB-1817-425C-A5B1-45F383F230CE', 92, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000042', 'Pfizer – Lab Workstation Refresh',           'A0000001-AAAA-0001-0001-000000000008', 'Propose',    2850000, '2026-09-02', 'Commit',       'Expansion', 'US', 'Americas BG',  75, 'Open', '055DAFE7-9840-451D-8328-5F70A6326C03', 84, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000043', 'AT&T – Call Centre DaaS Pilot',              'A0000001-AAAA-0001-0001-000000000014', 'Qualify',    1750000, '2026-12-05', 'Pipeline',     'Net-New',   'US', 'Americas BG',  30, 'Open', '81AADFDB-1817-425C-A5B1-45F383F230CE', 56, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000044', 'Ford – Mexico Plant Expansion',              '18C38124-5400-478E-9DC1-9E8C96ADB8CC', 'Develop',    2950000, '2026-10-30', 'Best Case',    'Expansion', 'MX', 'Americas BG',  45, 'Open', '81AADFDB-1817-425C-A5B1-45F383F230CE', 66, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000045', 'Tesla – Gigafactory Engineer Workstations',  'A0000001-AAAA-0001-0001-000000000002', 'Propose',    4150000, '2026-09-08', 'Most Likely',  'Net-New',   'US', 'Americas BG',  65, 'Open', '81AADFDB-1817-425C-A5B1-45F383F230CE', 78, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000046', 'Walmart – DC Logistics Devices',             'A0000001-AAAA-0001-0001-000000000003', 'Develop',    2350000, '2026-10-15', 'Best Case',    'Expansion', 'US', 'Americas BG',  50, 'Open', '81AADFDB-1817-425C-A5B1-45F383F230CE', 63, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000047', 'JPMorgan – Toronto Branch Refresh',          'A0000001-AAAA-0001-0001-000000000001', 'Closed Won', 3850000, '2026-05-15', NULL,           'Renewal',   'CA', 'Americas BG', 100, 'Won',  '81AADFDB-1817-425C-A5B1-45F383F230CE', 100,NULL, NULL),
    ('B0000001-0001-0001-0001-000000000048', 'Pfizer – Brazil Lab Devices',                'A0000001-AAAA-0001-0001-000000000008', 'Qualify',    1450000, '2026-11-08', 'Pipeline',     'Net-New',   'BR', 'Americas BG',  25, 'Open', '055DAFE7-9840-451D-8328-5F70A6326C03', 47, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000049', 'AT&T – Field Tech Rugged Tablets',           'A0000001-AAAA-0001-0001-000000000014', 'Execute',    2250000, '2026-07-18', 'Commit',       'Expansion', 'US', 'Americas BG',  85, 'Open', '81AADFDB-1817-425C-A5B1-45F383F230CE', 88, NULL, NULL),
    ('B0000001-0001-0001-0001-000000000050', 'Ford – Detroit HPC Phase 2',                 '18C38124-5400-478E-9DC1-9E8C96ADB8CC', 'Closed Won', 4500000, '2026-05-28', NULL,           'Expansion', 'US', 'Americas BG', 100, 'Won',  '81AADFDB-1817-425C-A5B1-45F383F230CE', 100,NULL, NULL)
ON CONFLICT (opportunityid) DO NOTHING;


-- ============================================================================
-- 3. Competitors  (24 rows — every other opportunity gets a competitor)
--    Keeps the "Competitors" grid column populated with realistic data.
-- ============================================================================
INSERT INTO lvo_opportunitycompetitor (
    lvo_opportunitycompetitorid, lvo_name, lvo_opportunityid,
    lvo_competitorname, lvo_competitortype, statecode
) VALUES
    ('C0000001-0001-0001-0001-000000000001', 'Dell – JPMorgan Trader',              'B0000001-0001-0001-0001-000000000001', 'Dell Technologies',    'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000002', 'HP – Bayer R&D',                      'B0000001-0001-0001-0001-000000000002', 'HP Inc.',              'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000003', 'Dell – Volkswagen Plant',             'B0000001-0001-0001-0001-000000000003', 'Dell Technologies',    'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000004', 'Panasonic – Shell Drilling',          'B0000001-0001-0001-0001-000000000004', 'Panasonic Toughbook',  'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000005', 'HP – BNP Paribas Branch',             'B0000001-0001-0001-0001-000000000006', 'HP Inc.',              'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000006', 'Dell – Airbus Engineering',           'B0000001-0001-0001-0001-000000000007', 'Dell Technologies',    'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000007', 'HP – Siemens Renewal',                'B0000001-0001-0001-0001-000000000008', 'HP Inc.',              'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000008', 'Apple – Deutsche Bank AI',            'B0000001-0001-0001-0001-000000000009', 'Apple',                'Indirect', 'Active'),
    ('C0000001-0001-0001-0001-000000000009', 'Microsoft Surface – Bayer Lab',       'B0000001-0001-0001-0001-000000000010', 'Microsoft Surface',    'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000010', 'HP – Shell HQ',                       'B0000001-0001-0001-0001-000000000012', 'HP Inc.',              'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000011', 'Dell – Maersk Port',                  'B0000001-0001-0001-0001-000000000013', 'Dell Technologies',    'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000012', 'HP – BNP Trading Floor',              'B0000001-0001-0001-0001-000000000014', 'HP Inc.',              'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000013', 'Dell – Siemens Edge AI Phase 2',      'B0000001-0001-0001-0001-000000000016', 'Dell Technologies',    'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000014', 'HP – Shell Refinery',                 'B0000001-0001-0001-0001-000000000018', 'HP Inc.',              'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000015', 'Fujitsu – Toyota Plant',              'B0000001-0001-0001-0001-000000000021', 'Fujitsu',              'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000016', 'HP – Samsung Dev Workstations',       'B0000001-0001-0001-0001-000000000022', 'HP Inc.',              'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000017', 'Dell – TCS Pune',                     'B0000001-0001-0001-0001-000000000023', 'Dell Technologies',    'Indirect', 'Active'),
    ('C0000001-0001-0001-0001-000000000018', 'HP – HSBC Singapore',                 'B0000001-0001-0001-0001-000000000025', 'HP Inc.',              'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000019', 'SGI / HPE – Infosys HPC',             'B0000001-0001-0001-0001-000000000026', 'SGI / HPE',            'Indirect', 'Active'),
    ('C0000001-0001-0001-0001-000000000020', 'Dell – JPMorgan NY HQ',               'B0000001-0001-0001-0001-000000000039', 'Dell Technologies',    'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000021', 'HP – Tesla Fremont',                  'B0000001-0001-0001-0001-000000000040', 'HP Inc.',              'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000022', 'Acer – Walmart Stores',               'B0000001-0001-0001-0001-000000000041', 'Acer',                 'Indirect', 'Active'),
    ('C0000001-0001-0001-0001-000000000023', 'Dell – Pfizer Lab',                   'B0000001-0001-0001-0001-000000000042', 'Dell Technologies',    'Direct',   'Active'),
    ('C0000001-0001-0001-0001-000000000024', 'HP – AT&T Call Centre',               'B0000001-0001-0001-0001-000000000043', 'HP Inc.',              'Direct',   'Active')
ON CONFLICT (lvo_opportunitycompetitorid) DO NOTHING;


-- ============================================================================
-- 4. Activities  (100 rows — 2 per opportunity)
--    Powers the "Last Activity" column + the timeline dots in the grid.
-- ============================================================================
INSERT INTO lvo_activity (
    lvo_activityid, lvo_opportunityid, lvo_activitytype, lvo_direction,
    lvo_subject, lvo_body, lvo_activitydate, lvo_groupedcount,
    owninguser, statecode
) VALUES
    -- 2 activities per opportunity. Dates chosen so they land at varied
    -- positions on the 90-day timeline track.
    ('D0000001-0001-0001-0001-000000000101', 'B0000001-0001-0001-0001-000000000001', 'email',    'outbound', 'Initial Outreach',   'Sent intro deck for trader workstation refresh.',     '2026-04-12 09:30:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000102', 'B0000001-0001-0001-0001-000000000001', 'meeting',  'outbound', 'Discovery Call',     'Mapped current Dell fleet, identified replacement scope.','2026-05-22 14:00:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000103', 'B0000001-0001-0001-0001-000000000002', 'email',    'inbound',  'RFP Requested',      'Client requested formal RFP for compute cluster.',    '2026-04-02 10:15:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000104', 'B0000001-0001-0001-0001-000000000002', 'crm',      'outbound', 'Stage Updated',      'Moved to Develop after positive RFP feedback.',       '2026-05-28 11:30:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000105', 'B0000001-0001-0001-0001-000000000003', 'meeting',  'outbound', 'Site Walkthrough',   'Walked Wolfsburg plant floor with ops team.',         '2026-04-08 13:00:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000106', 'B0000001-0001-0001-0001-000000000003', 'email',    'inbound',  'Pricing Returned',   'Client returned signed pricing memo.',                '2026-05-30 16:00:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000107', 'B0000001-0001-0001-0001-000000000004', 'meeting',  'outbound', 'Rugged Demo',        'Demoed rugged tablets at drilling site.',             '2026-03-28 10:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000108', 'B0000001-0001-0001-0001-000000000004', 'crm',      'outbound', 'Risk Updated',       'Risk elevated — Panasonic pushing hard on pricing.',  '2026-05-15 09:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000109', 'B0000001-0001-0001-0001-000000000005', 'email',    'inbound',  'PO Submitted',       'Maersk procurement submitted PO.',                    '2026-05-26 11:00:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000110', 'B0000001-0001-0001-0001-000000000005', 'meeting',  'outbound', 'Contract Review',    'Closed contract negotiation, awaiting countersign.',  '2026-06-02 15:00:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000111', 'B0000001-0001-0001-0001-000000000006', 'email',    'outbound', 'Proposal Sent',      'Sent branch endpoint refresh proposal.',              '2026-04-10 09:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000112', 'B0000001-0001-0001-0001-000000000006', 'multiple', 'inbound',  'Multiple Events',    'Pricing call + technical review + CRM update.',       '2026-05-24 13:00:00', 3,    'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000113', 'B0000001-0001-0001-0001-000000000007', 'meeting',  'outbound', 'Engineering Demo',   'Demo of ThinkStation P-series for CAD workloads.',    '2026-04-15 14:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000114', 'B0000001-0001-0001-0001-000000000007', 'crm',      'inbound',  'Note Added',         'No outbound activity in 14 days — flagged stale.',    '2026-05-18 10:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000115', 'B0000001-0001-0001-0001-000000000008', 'email',    'inbound',  'Renewal Trigger',    'Annual renewal window opened.',                       '2026-05-02 09:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000116', 'B0000001-0001-0001-0001-000000000008', 'email',    'outbound', 'Renewal Proposal',   'Sent renewal proposal pack.',                         '2026-05-29 11:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000117', 'B0000001-0001-0001-0001-000000000009', 'meeting',  'outbound', 'AI Strategy Workshop','Co-presented AI trader use cases with Microsoft.',    '2026-04-22 10:00:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000118', 'B0000001-0001-0001-0001-000000000009', 'crm',      'outbound', 'Stage Updated',      'Moved to Develop, BANT confirmed.',                   '2026-05-31 09:30:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000119', 'B0000001-0001-0001-0001-000000000010', 'email',    'outbound', 'Tablet Pilot Offer', 'Offered 30-day pilot for 50 lab tablets.',            '2026-04-18 09:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000120', 'B0000001-0001-0001-0001-000000000010', 'meeting',  'inbound',  'Pilot Review',       'Pilot feedback workshop with lab managers.',          '2026-06-01 14:30:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000121', 'B0000001-0001-0001-0001-000000000011', 'multiple', 'outbound', 'Negotiation Day',    'Final pricing + contract + legal review same day.',   '2026-05-20 16:00:00', 3,    'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000122', 'B0000001-0001-0001-0001-000000000011', 'email',    'inbound',  'Signed Order',       'Volkswagen procurement signed off.',                  '2026-05-30 11:00:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000123', 'B0000001-0001-0001-0001-000000000012', 'email',    'outbound', 'HQ Refresh Pitch',   'Sent corporate refresh deck to HQ procurement.',      '2026-04-25 09:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000124', 'B0000001-0001-0001-0001-000000000012', 'meeting',  'outbound', 'Executive Review',   'Presented to Shell CIO and procurement VP.',          '2026-05-25 13:30:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000125', 'B0000001-0001-0001-0001-000000000013', 'email',    'inbound',  'Initial Inquiry',    'Port ops asked about ruggedised workstation specs.',  '2026-04-05 10:00:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000126', 'B0000001-0001-0001-0001-000000000013', 'meeting',  'outbound', 'Site Survey',        'Walked Rotterdam port operations centre.',            '2026-05-19 11:30:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000127', 'B0000001-0001-0001-0001-000000000014', 'meeting',  'outbound', 'Trading Floor Demo', 'Demoed ThinkStation for low-latency trading.',        '2026-04-30 09:30:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000128', 'B0000001-0001-0001-0001-000000000014', 'crm',      'inbound',  'PO Received',        'Purchase order received from BNP procurement.',       '2026-05-31 14:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000129', 'B0000001-0001-0001-0001-000000000015', 'crm',      'outbound', 'Closed Won',         'Order signed. Stage moved to Closed Won.',            '2026-05-08 16:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000130', 'B0000001-0001-0001-0001-000000000015', 'email',    'inbound',  'Thank-you Note',     'Customer thanked AE for smooth procurement.',         '2026-05-12 10:30:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000131', 'B0000001-0001-0001-0001-000000000016', 'email',    'inbound',  'Budget Update',      'Phase 2 budget approved, but with reduced ceiling.',  '2026-04-19 09:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000132', 'B0000001-0001-0001-0001-000000000016', 'crm',      'outbound', 'Risk Score Updated', 'Risk elevated to 3 — budget freeze concern.',         '2026-05-28 11:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000133', 'B0000001-0001-0001-0001-000000000017', 'email',    'outbound', 'Refresh Pitch',      'Sent Salesforce-bundle refresh proposal.',            '2026-05-05 13:00:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000134', 'B0000001-0001-0001-0001-000000000017', 'crm',      'outbound', 'Stage Update',       'Moved to Qualify after CRM confirmation.',            '2026-05-29 10:00:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000135', 'B0000001-0001-0001-0001-000000000018', 'crm',      'outbound', 'Lost to Competitor', 'Lost to HP — pricing edge on rugged.',                '2026-04-18 09:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000136', 'B0000001-0001-0001-0001-000000000018', 'email',    'inbound',  'Notification',       'Client notified of vendor decision.',                 '2026-04-20 08:30:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000137', 'B0000001-0001-0001-0001-000000000019', 'meeting',  'outbound', 'Toulouse Site Visit','On-site review of standardisation gaps.',             '2026-04-28 11:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000138', 'B0000001-0001-0001-0001-000000000019', 'email',    'outbound', 'BOM Sent',           'Sent BOM + pricing for engineering laptops.',         '2026-05-26 14:00:00', NULL, 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('D0000001-0001-0001-0001-000000000139', 'B0000001-0001-0001-0001-000000000020', 'crm',      'outbound', 'Closed Won',         'Renewal closed, stage moved to Closed Won.',          '2026-04-22 15:00:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000140', 'B0000001-0001-0001-0001-000000000020', 'email',    'inbound',  'Renewal Confirmed',  'Procurement confirmed 3-year renewal.',               '2026-04-26 09:00:00', NULL, 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('D0000001-0001-0001-0001-000000000141', 'B0000001-0001-0001-0001-000000000021', 'email',    'outbound', 'IoT Strategy Doc',   'Shared Toyota Production System device strategy.',    '2026-04-12 10:00:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000142', 'B0000001-0001-0001-0001-000000000021', 'meeting',  'outbound', 'Toyota City Visit',  'Reviewed plant floor IoT needs with ops.',            '2026-05-22 14:00:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000143', 'B0000001-0001-0001-0001-000000000022', 'meeting',  'outbound', 'Suwon Demo',         'Demoed dev workstation for Samsung SDS.',             '2026-04-30 10:30:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000144', 'B0000001-0001-0001-0001-000000000022', 'email',    'inbound',  'BOM Requested',      'Client asked for BOM with optional GPU spec.',        '2026-05-30 11:00:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000145', 'B0000001-0001-0001-0001-000000000023', 'multiple', 'inbound',  'Multiple Events',    'Contract + PO + signed quote logged same day.',       '2026-05-29 16:00:00', 3,    '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000146', 'B0000001-0001-0001-0001-000000000023', 'meeting',  'outbound', 'Pune Office Walk',   'Walked Pune campus to plan deployment waves.',        '2026-06-02 10:30:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000147', 'B0000001-0001-0001-0001-000000000024', 'email',    'outbound', 'Intro Email',        'Intro to CommBank IT procurement.',                   '2026-05-04 09:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000148', 'B0000001-0001-0001-0001-000000000024', 'crm',      'outbound', 'Stage Logged',       'Logged opportunity in Qualify stage.',                '2026-05-30 14:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000149', 'B0000001-0001-0001-0001-000000000025', 'meeting',  'outbound', 'Trading Floor Demo', 'Demoed low-latency workstation in Singapore.',        '2026-04-26 13:30:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000150', 'B0000001-0001-0001-0001-000000000025', 'email',    'inbound',  'Capacity Question',  'Client asked about scaling to 800 seats.',            '2026-05-30 10:00:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000151', 'B0000001-0001-0001-0001-000000000026', 'email',    'outbound', 'HPC Proposal',       'Sent expansion proposal for Bangalore HPC.',          '2026-04-29 11:30:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000152', 'B0000001-0001-0001-0001-000000000026', 'meeting',  'outbound', 'Architect Walkthrough','Walked through proposed cluster topology.',         '2026-05-28 15:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000153', 'B0000001-0001-0001-0001-000000000027', 'email',    'outbound', 'Kentucky Pitch',     'Sent mirror-plant pitch deck.',                       '2026-05-06 09:00:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000154', 'B0000001-0001-0001-0001-000000000027', 'meeting',  'inbound',  'Internal Review',    'Client asked for follow-up after internal review.',   '2026-05-30 11:00:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000155', 'B0000001-0001-0001-0001-000000000028', 'multiple', 'outbound', 'Multiple Events',    'Final pricing + legal + PO same day.',                '2026-05-21 14:00:00', 3,    '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000156', 'B0000001-0001-0001-0001-000000000028', 'email',    'inbound',  'PO Confirmed',       'Display plant PO confirmed.',                         '2026-05-30 09:30:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000157', 'B0000001-0001-0001-0001-000000000029', 'email',    'outbound', 'Hyderabad Plan',     'Shared deployment plan for Hyd tower.',               '2026-04-22 10:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000158', 'B0000001-0001-0001-0001-000000000029', 'crm',      'outbound', 'Stage Update',       'Moved to Develop after technical alignment.',         '2026-05-26 13:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000159', 'B0000001-0001-0001-0001-000000000030', 'email',    'outbound', 'Renewal Quote',      'Sent renewal quote with volume discount.',            '2026-05-02 09:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000160', 'B0000001-0001-0001-0001-000000000030', 'meeting',  'inbound',  'Pricing Workshop',   'Pricing walkthrough with CommBank procurement.',      '2026-05-30 14:30:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000161', 'B0000001-0001-0001-0001-000000000031', 'crm',      'outbound', 'Closed Won',         'Hong Kong renewal closed.',                           '2026-05-01 11:00:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000162', 'B0000001-0001-0001-0001-000000000031', 'email',    'inbound',  'Confirmation',       'Procurement sent confirmation note.',                 '2026-05-03 10:00:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000163', 'B0000001-0001-0001-0001-000000000032', 'email',    'outbound', 'Campus Pitch',       'Sent device refresh pitch for Mysore campus.',        '2026-04-30 10:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000164', 'B0000001-0001-0001-0001-000000000032', 'meeting',  'outbound', 'Campus Visit',       'Walked Mysore training campus, mapped fleet.',        '2026-05-28 13:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000165', 'B0000001-0001-0001-0001-000000000033', 'email',    'outbound', 'Mumbai Refresh',     'Sent Mumbai tower refresh proposal.',                 '2026-05-08 09:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000166', 'B0000001-0001-0001-0001-000000000033', 'meeting',  'outbound', 'Procurement Sync',   'Aligned on rollout windows with TCS procurement.',    '2026-05-31 11:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000167', 'B0000001-0001-0001-0001-000000000034', 'email',    'outbound', 'R&D Outreach',       'Intro email to Samsung Tokyo R&D head.',              '2026-05-10 09:30:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000168', 'B0000001-0001-0001-0001-000000000034', 'crm',      'outbound', 'Stage Logged',       'Logged opportunity in Qualify.',                      '2026-05-29 14:00:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000169', 'B0000001-0001-0001-0001-000000000035', 'meeting',  'outbound', 'Bangkok Plant Visit','Walked Bangkok assembly line, mapped fleet.',         '2026-05-15 10:00:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000170', 'B0000001-0001-0001-0001-000000000035', 'email',    'inbound',  'PO Submitted',       'Thailand plant PO submitted.',                        '2026-05-31 11:30:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000171', 'B0000001-0001-0001-0001-000000000036', 'crm',      'outbound', 'Lost to Competitor', 'Lost on pricing — Dell undercut by 18%.',             '2026-04-08 11:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000172', 'B0000001-0001-0001-0001-000000000036', 'email',    'inbound',  'Decision Notice',    'Procurement notified of vendor decision.',            '2026-04-10 09:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000173', 'B0000001-0001-0001-0001-000000000037', 'email',    'outbound', 'APAC Renewal Plan',  'Sent APAC branch renewal plan.',                      '2026-04-24 10:00:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000174', 'B0000001-0001-0001-0001-000000000037', 'meeting',  'outbound', 'Renewal Walkthrough','Walked HSBC APAC team through renewal terms.',        '2026-05-30 13:00:00', NULL, '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('D0000001-0001-0001-0001-000000000175', 'B0000001-0001-0001-0001-000000000038', 'crm',      'outbound', 'Closed Won',         'Infosys Pune expansion closed.',                      '2026-04-18 14:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000176', 'B0000001-0001-0001-0001-000000000038', 'email',    'inbound',  'Sign-off',           'PO + invoice acknowledged.',                          '2026-04-22 10:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000177', 'B0000001-0001-0001-0001-000000000039', 'meeting',  'outbound', 'NY HQ Demo',         'Demoed for trading and IT teams in NY.',              '2026-04-29 14:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000178', 'B0000001-0001-0001-0001-000000000039', 'email',    'inbound',  'BOM Returned',       'Client returned annotated BOM.',                      '2026-05-29 11:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000179', 'B0000001-0001-0001-0001-000000000040', 'email',    'outbound', 'Fremont Pitch',      'Sent Fremont plant tablet pitch.',                    '2026-04-10 09:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000180', 'B0000001-0001-0001-0001-000000000040', 'crm',      'outbound', 'Risk Updated',       'Stale activity flagged — last touch 18 days ago.',    '2026-05-15 10:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000181', 'B0000001-0001-0001-0001-000000000041', 'multiple', 'inbound',  'Multiple Events',    'PO + contract + signed quote logged same day.',       '2026-05-22 16:00:00', 3,    '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000182', 'B0000001-0001-0001-0001-000000000041', 'meeting',  'outbound', 'Rollout Planning',   'Aligned with Walmart on store rollout waves.',        '2026-05-30 13:30:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000183', 'B0000001-0001-0001-0001-000000000042', 'email',    'outbound', 'Lab Refresh Proposal','Sent Pfizer lab refresh proposal.',                  '2026-04-30 10:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000184', 'B0000001-0001-0001-0001-000000000042', 'meeting',  'outbound', 'Lab Tour',           'Walked Boston R&D lab for spec validation.',          '2026-05-28 14:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000185', 'B0000001-0001-0001-0001-000000000043', 'email',    'outbound', 'DaaS Pilot Offer',   'Offered AT&T 30-day DaaS pilot.',                     '2026-05-12 10:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000186', 'B0000001-0001-0001-0001-000000000043', 'crm',      'outbound', 'Stage Logged',       'Logged opportunity in Qualify.',                      '2026-05-30 09:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000187', 'B0000001-0001-0001-0001-000000000044', 'meeting',  'outbound', 'Cuautitlán Visit',   'Walked Ford Mexico plant for fleet planning.',        '2026-05-04 11:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000188', 'B0000001-0001-0001-0001-000000000044', 'email',    'inbound',  'Scope Sign-off',     'Client signed off on scope of work.',                 '2026-05-29 10:30:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000189', 'B0000001-0001-0001-0001-000000000045', 'email',    'outbound', 'Gigafactory Pitch',  'Sent gigafactory engineer workstation pitch.',        '2026-04-26 09:30:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000190', 'B0000001-0001-0001-0001-000000000045', 'meeting',  'outbound', 'Reno Plant Walk',    'Walked Reno facility, mapped engineering seats.',     '2026-05-26 14:30:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000191', 'B0000001-0001-0001-0001-000000000046', 'email',    'outbound', 'DC Devices Pitch',   'Sent distribution-centre device proposal.',           '2026-04-22 09:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000192', 'B0000001-0001-0001-0001-000000000046', 'crm',      'outbound', 'Stage Update',       'Moved to Develop after BANT.',                        '2026-05-30 14:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000193', 'B0000001-0001-0001-0001-000000000047', 'crm',      'outbound', 'Closed Won',         'Toronto branch refresh closed.',                      '2026-05-12 13:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000194', 'B0000001-0001-0001-0001-000000000047', 'email',    'inbound',  'Procurement Sign-off','PO confirmed by Toronto procurement.',                '2026-05-15 09:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000195', 'B0000001-0001-0001-0001-000000000048', 'email',    'outbound', 'Brazil Lab Outreach','Intro email to São Paulo lab manager.',               '2026-05-10 10:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000196', 'B0000001-0001-0001-0001-000000000048', 'crm',      'outbound', 'Stage Logged',       'Logged opportunity in Qualify.',                      '2026-05-30 11:00:00', NULL, '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('D0000001-0001-0001-0001-000000000197', 'B0000001-0001-0001-0001-000000000049', 'multiple', 'outbound', 'Multiple Events',    'Field demo + PO + contract review same day.',         '2026-05-20 15:00:00', 3,    '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000198', 'B0000001-0001-0001-0001-000000000049', 'email',    'inbound',  'PO Submitted',       'AT&T field PO submitted by procurement.',             '2026-05-30 10:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000199', 'B0000001-0001-0001-0001-000000000050', 'crm',      'outbound', 'Closed Won',         'Ford Detroit HPC Phase 2 closed.',                    '2026-05-25 16:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('D0000001-0001-0001-0001-000000000200', 'B0000001-0001-0001-0001-000000000050', 'email',    'inbound',  'Order Confirmation', 'Ford signed off on Phase 2.',                         '2026-05-29 09:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active')
ON CONFLICT (lvo_activityid) DO NOTHING;


-- ============================================================================
-- 5. Quotes  (50 rows, one per new opportunity)
--
--   Links each opportunity to a quote so that the /api/filters/products
--   filter (ThinkPad / ThinkStation / ThinkSystem) can actually filter the
--   new rows. The filter does an EXISTS over (quote -> lvo_quoteitem) keyed
--   by opportunity.opportunityid, with UPPER() on both sides for the
--   uuid::text-lowercases-but-stored-uppercase gotcha.
-- ============================================================================
INSERT INTO quote (
    quoteid, name, opportunityid, accountid,
    lvo_businessgroupid, lvo_country, owninguser,
    totalamount, statecode, effectivefrom, effectiveto
) VALUES
    -- EMEA BG ────────────────────────────────────────────────────────────
    ('AABB0001-0001-0001-0001-000000000001', 'QT-2026-1001', 'B0000001-0001-0001-0001-000000000001', '6DC95C38-9237-4CE9-84D3-F5D1F7431965',    'EMEA BG', 'DE', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 1850000, 'Draft',  '2026-05-01', '2026-11-15'),
    ('AABB0001-0001-0001-0001-000000000002', 'QT-2026-1002', 'B0000001-0001-0001-0001-000000000002', 'A0000001-AAAA-0001-0001-000000000004',    'EMEA BG', 'DE', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 3450000, 'Draft',  '2026-05-01', '2026-09-28'),
    ('AABB0001-0001-0001-0001-000000000003', 'QT-2026-1003', 'B0000001-0001-0001-0001-000000000003', 'A0000001-AAAA-0001-0001-000000000005',    'EMEA BG', 'DE', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 5200000, 'Active', '2026-05-01', '2026-08-18'),
    ('AABB0001-0001-0001-0001-000000000004', 'QT-2026-1004', 'B0000001-0001-0001-0001-000000000004', 'A0000001-AAAA-0001-0001-000000000009',    'EMEA BG', 'GB', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 2400000, 'Draft',  '2026-05-01', '2026-10-05'),
    ('AABB0001-0001-0001-0001-000000000005', 'QT-2026-1005', 'B0000001-0001-0001-0001-000000000005', 'A0000001-AAAA-0001-0001-000000000011',    'EMEA BG', 'DK', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 1750000, 'Active', '2026-05-01', '2026-07-10'),
    ('AABB0001-0001-0001-0001-000000000006', 'QT-2026-1006', 'B0000001-0001-0001-0001-000000000006', 'A0000001-AAAA-0001-0001-000000000013',    'EMEA BG', 'FR', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 4100000, 'Active', '2026-05-01', '2026-09-04'),
    ('AABB0001-0001-0001-0001-000000000007', 'QT-2026-1007', 'B0000001-0001-0001-0001-000000000007', 'A0000001-AAAA-0001-0001-000000000015',    'EMEA BG', 'FR', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 6800000, 'Draft',  '2026-05-01', '2026-12-02'),
    ('AABB0001-0001-0001-0001-000000000008', 'QT-2026-1008', 'B0000001-0001-0001-0001-000000000008', 'D4D580CA-EDF7-46A9-9BA7-C4114B83A3A6',    'EMEA BG', 'DE', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 1200000, 'Draft',  '2026-05-01', '2026-11-22'),
    ('AABB0001-0001-0001-0001-000000000009', 'QT-2026-1009', 'B0000001-0001-0001-0001-000000000009', '6DC95C38-9237-4CE9-84D3-F5D1F7431965',    'EMEA BG', 'DE', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 3200000, 'Active', '2026-05-01', '2026-09-15'),
    ('AABB0001-0001-0001-0001-000000000010', 'QT-2026-1010', 'B0000001-0001-0001-0001-000000000010', 'A0000001-AAAA-0001-0001-000000000004',    'EMEA BG', 'DE', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 1950000, 'Active', '2026-05-01', '2026-08-28'),
    ('AABB0001-0001-0001-0001-000000000011', 'QT-2026-1011', 'B0000001-0001-0001-0001-000000000011', 'A0000001-AAAA-0001-0001-000000000005',    'EMEA BG', 'DE', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 4250000, 'Active', '2026-05-01', '2026-06-30'),
    ('AABB0001-0001-0001-0001-000000000012', 'QT-2026-1012', 'B0000001-0001-0001-0001-000000000012', 'A0000001-AAAA-0001-0001-000000000009',    'EMEA BG', 'GB', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 2850000, 'Active', '2026-05-01', '2026-09-12'),
    ('AABB0001-0001-0001-0001-000000000013', 'QT-2026-1013', 'B0000001-0001-0001-0001-000000000013', 'A0000001-AAAA-0001-0001-000000000011',    'EMEA BG', 'NL', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 1620000, 'Draft',  '2026-05-01', '2026-10-22'),
    ('AABB0001-0001-0001-0001-000000000014', 'QT-2026-1014', 'B0000001-0001-0001-0001-000000000014', 'A0000001-AAAA-0001-0001-000000000013',    'EMEA BG', 'FR', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 5650000, 'Active', '2026-05-01', '2026-07-22'),
    ('AABB0001-0001-0001-0001-000000000015', 'QT-2026-1015', 'B0000001-0001-0001-0001-000000000015', 'A0000001-AAAA-0001-0001-000000000015',    'EMEA BG', 'FR', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 8200000, 'Won',    '2026-04-01', '2026-05-10'),
    ('AABB0001-0001-0001-0001-000000000016', 'QT-2026-1016', 'B0000001-0001-0001-0001-000000000016', 'D4D580CA-EDF7-46A9-9BA7-C4114B83A3A6',    'EMEA BG', 'DE', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 2950000, 'Active', '2026-05-01', '2026-09-29'),
    ('AABB0001-0001-0001-0001-000000000017', 'QT-2026-1017', 'B0000001-0001-0001-0001-000000000017', 'A0000001-AAAA-0001-0001-000000000004',    'EMEA BG', 'CH', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2',  980000, 'Draft',  '2026-05-01', '2026-12-15'),
    ('AABB0001-0001-0001-0001-000000000018', 'QT-2026-1018', 'B0000001-0001-0001-0001-000000000018', 'A0000001-AAAA-0001-0001-000000000009',    'EMEA BG', 'GB', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 3400000, 'Lost',   '2026-03-01', '2026-04-20'),
    ('AABB0001-0001-0001-0001-000000000019', 'QT-2026-1019', 'B0000001-0001-0001-0001-000000000019', 'A0000001-AAAA-0001-0001-000000000015',    'EMEA BG', 'ES', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 3850000, 'Active', '2026-05-01', '2026-10-15'),
    ('AABB0001-0001-0001-0001-000000000020', 'QT-2026-1020', 'B0000001-0001-0001-0001-000000000020', 'A0000001-AAAA-0001-0001-000000000005',    'EMEA BG', 'IT', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 2150000, 'Won',    '2026-03-01', '2026-04-30'),
    -- APAC BG ────────────────────────────────────────────────────────────
    ('AABB0001-0001-0001-0001-000000000021', 'QT-2026-1021', 'B0000001-0001-0001-0001-000000000021', 'A0000001-AAAA-0001-0001-000000000006',    'APAC BG', 'JP', '7D26391E-D020-474E-B1CA-53E6B6C71487', 4400000, 'Active', '2026-05-01', '2026-10-08'),
    ('AABB0001-0001-0001-0001-000000000022', 'QT-2026-1022', 'B0000001-0001-0001-0001-000000000022', 'A0000001-AAAA-0001-0001-000000000007',    'APAC BG', 'KR', '7D26391E-D020-474E-B1CA-53E6B6C71487', 3700000, 'Active', '2026-05-01', '2026-08-25'),
    ('AABB0001-0001-0001-0001-000000000023', 'QT-2026-1023', 'B0000001-0001-0001-0001-000000000023', 'A0000001-AAAA-0001-0001-000000000010',    'APAC BG', 'IN', '7D26391E-D020-474E-B1CA-53E6B6C71487', 2200000, 'Active', '2026-05-01', '2026-07-05'),
    ('AABB0001-0001-0001-0001-000000000024', 'QT-2026-1024', 'B0000001-0001-0001-0001-000000000024', 'A0000001-AAAA-0001-0001-000000000012',    'APAC BG', 'AU', '055DAFE7-9840-451D-8328-5F70A6326C03', 1450000, 'Draft',  '2026-05-01', '2026-12-08'),
    ('AABB0001-0001-0001-0001-000000000025', 'QT-2026-1025', 'B0000001-0001-0001-0001-000000000025', 'E04F3BC9-2FCD-42E7-87C2-636B6328EFA8',    'APAC BG', 'SG', '7D26391E-D020-474E-B1CA-53E6B6C71487', 3950000, 'Active', '2026-05-01', '2026-10-18'),
    ('AABB0001-0001-0001-0001-000000000026', 'QT-2026-1026', 'B0000001-0001-0001-0001-000000000026', '1C603E6A-C4EA-4E97-B73E-345A5A9C2460',    'APAC BG', 'IN', '055DAFE7-9840-451D-8328-5F70A6326C03', 2950000, 'Active', '2026-05-01', '2026-09-08'),
    ('AABB0001-0001-0001-0001-000000000027', 'QT-2026-1027', 'B0000001-0001-0001-0001-000000000027', 'A0000001-AAAA-0001-0001-000000000006',    'APAC BG', 'JP', '7D26391E-D020-474E-B1CA-53E6B6C71487', 1850000, 'Draft',  '2026-05-01', '2026-11-30'),
    ('AABB0001-0001-0001-0001-000000000028', 'QT-2026-1028', 'B0000001-0001-0001-0001-000000000028', 'A0000001-AAAA-0001-0001-000000000007',    'APAC BG', 'KR', '7D26391E-D020-474E-B1CA-53E6B6C71487', 3150000, 'Active', '2026-05-01', '2026-06-25'),
    ('AABB0001-0001-0001-0001-000000000029', 'QT-2026-1029', 'B0000001-0001-0001-0001-000000000029', 'A0000001-AAAA-0001-0001-000000000010',    'APAC BG', 'IN', '055DAFE7-9840-451D-8328-5F70A6326C03', 2750000, 'Active', '2026-05-01', '2026-10-12'),
    ('AABB0001-0001-0001-0001-000000000030', 'QT-2026-1030', 'B0000001-0001-0001-0001-000000000030', 'A0000001-AAAA-0001-0001-000000000012',    'APAC BG', 'AU', '055DAFE7-9840-451D-8328-5F70A6326C03', 2400000, 'Active', '2026-05-01', '2026-09-18'),
    ('AABB0001-0001-0001-0001-000000000031', 'QT-2026-1031', 'B0000001-0001-0001-0001-000000000031', 'E04F3BC9-2FCD-42E7-87C2-636B6328EFA8',    'APAC BG', 'HK', '7D26391E-D020-474E-B1CA-53E6B6C71487', 4250000, 'Won',    '2026-03-01', '2026-05-05'),
    ('AABB0001-0001-0001-0001-000000000032', 'QT-2026-1032', 'B0000001-0001-0001-0001-000000000032', '1C603E6A-C4EA-4E97-B73E-345A5A9C2460',    'APAC BG', 'IN', '055DAFE7-9840-451D-8328-5F70A6326C03', 1850000, 'Active', '2026-05-01', '2026-10-28'),
    ('AABB0001-0001-0001-0001-000000000033', 'QT-2026-1033', 'B0000001-0001-0001-0001-000000000033', 'A0000001-AAAA-0001-0001-000000000010',    'APAC BG', 'IN', '055DAFE7-9840-451D-8328-5F70A6326C03', 3050000, 'Active', '2026-05-01', '2026-09-22'),
    ('AABB0001-0001-0001-0001-000000000034', 'QT-2026-1034', 'B0000001-0001-0001-0001-000000000034', 'A0000001-AAAA-0001-0001-000000000007',    'APAC BG', 'JP', '7D26391E-D020-474E-B1CA-53E6B6C71487', 2400000, 'Draft',  '2026-05-01', '2026-12-22'),
    ('AABB0001-0001-0001-0001-000000000035', 'QT-2026-1035', 'B0000001-0001-0001-0001-000000000035', 'A0000001-AAAA-0001-0001-000000000006',    'APAC BG', 'JP', '7D26391E-D020-474E-B1CA-53E6B6C71487', 1980000, 'Active', '2026-05-01', '2026-07-12'),
    ('AABB0001-0001-0001-0001-000000000036', 'QT-2026-1036', 'B0000001-0001-0001-0001-000000000036', 'A0000001-AAAA-0001-0001-000000000012',    'APAC BG', 'AU', '055DAFE7-9840-451D-8328-5F70A6326C03', 1750000, 'Lost',   '2026-03-01', '2026-04-10'),
    ('AABB0001-0001-0001-0001-000000000037', 'QT-2026-1037', 'B0000001-0001-0001-0001-000000000037', 'E04F3BC9-2FCD-42E7-87C2-636B6328EFA8',    'APAC BG', 'SG', '7D26391E-D020-474E-B1CA-53E6B6C71487', 3650000, 'Active', '2026-05-01', '2026-10-26'),
    ('AABB0001-0001-0001-0001-000000000038', 'QT-2026-1038', 'B0000001-0001-0001-0001-000000000038', '1C603E6A-C4EA-4E97-B73E-345A5A9C2460',    'APAC BG', 'IN', '055DAFE7-9840-451D-8328-5F70A6326C03', 2950000, 'Won',    '2026-03-01', '2026-04-25'),
    -- Americas BG ────────────────────────────────────────────────────────
    ('AABB0001-0001-0001-0001-000000000039', 'QT-2026-1039', 'B0000001-0001-0001-0001-000000000039', 'A0000001-AAAA-0001-0001-000000000001', 'Americas BG', 'US', '81AADFDB-1817-425C-A5B1-45F383F230CE', 6750000, 'Active', '2026-05-01', '2026-08-12'),
    ('AABB0001-0001-0001-0001-000000000040', 'QT-2026-1040', 'B0000001-0001-0001-0001-000000000040', 'A0000001-AAAA-0001-0001-000000000002', 'Americas BG', 'US', '81AADFDB-1817-425C-A5B1-45F383F230CE', 3650000, 'Draft',  '2026-05-01', '2026-10-22'),
    ('AABB0001-0001-0001-0001-000000000041', 'QT-2026-1041', 'B0000001-0001-0001-0001-000000000041', 'A0000001-AAAA-0001-0001-000000000003', 'Americas BG', 'US', '81AADFDB-1817-425C-A5B1-45F383F230CE', 5450000, 'Active', '2026-05-01', '2026-06-18'),
    ('AABB0001-0001-0001-0001-000000000042', 'QT-2026-1042', 'B0000001-0001-0001-0001-000000000042', 'A0000001-AAAA-0001-0001-000000000008', 'Americas BG', 'US', '055DAFE7-9840-451D-8328-5F70A6326C03', 2850000, 'Active', '2026-05-01', '2026-09-02'),
    ('AABB0001-0001-0001-0001-000000000043', 'QT-2026-1043', 'B0000001-0001-0001-0001-000000000043', 'A0000001-AAAA-0001-0001-000000000014', 'Americas BG', 'US', '81AADFDB-1817-425C-A5B1-45F383F230CE', 1750000, 'Draft',  '2026-05-01', '2026-12-05'),
    ('AABB0001-0001-0001-0001-000000000044', 'QT-2026-1044', 'B0000001-0001-0001-0001-000000000044', '18C38124-5400-478E-9DC1-9E8C96ADB8CC', 'Americas BG', 'MX', '81AADFDB-1817-425C-A5B1-45F383F230CE', 2950000, 'Active', '2026-05-01', '2026-10-30'),
    ('AABB0001-0001-0001-0001-000000000045', 'QT-2026-1045', 'B0000001-0001-0001-0001-000000000045', 'A0000001-AAAA-0001-0001-000000000002', 'Americas BG', 'US', '81AADFDB-1817-425C-A5B1-45F383F230CE', 4150000, 'Active', '2026-05-01', '2026-09-08'),
    ('AABB0001-0001-0001-0001-000000000046', 'QT-2026-1046', 'B0000001-0001-0001-0001-000000000046', 'A0000001-AAAA-0001-0001-000000000003', 'Americas BG', 'US', '81AADFDB-1817-425C-A5B1-45F383F230CE', 2350000, 'Active', '2026-05-01', '2026-10-15'),
    ('AABB0001-0001-0001-0001-000000000047', 'QT-2026-1047', 'B0000001-0001-0001-0001-000000000047', 'A0000001-AAAA-0001-0001-000000000001', 'Americas BG', 'CA', '81AADFDB-1817-425C-A5B1-45F383F230CE', 3850000, 'Won',    '2026-03-01', '2026-05-15'),
    ('AABB0001-0001-0001-0001-000000000048', 'QT-2026-1048', 'B0000001-0001-0001-0001-000000000048', 'A0000001-AAAA-0001-0001-000000000008', 'Americas BG', 'BR', '055DAFE7-9840-451D-8328-5F70A6326C03', 1450000, 'Draft',  '2026-05-01', '2026-11-08'),
    ('AABB0001-0001-0001-0001-000000000049', 'QT-2026-1049', 'B0000001-0001-0001-0001-000000000049', 'A0000001-AAAA-0001-0001-000000000014', 'Americas BG', 'US', '81AADFDB-1817-425C-A5B1-45F383F230CE', 2250000, 'Active', '2026-05-01', '2026-07-18'),
    ('AABB0001-0001-0001-0001-000000000050', 'QT-2026-1050', 'B0000001-0001-0001-0001-000000000050', '18C38124-5400-478E-9DC1-9E8C96ADB8CC', 'Americas BG', 'US', '81AADFDB-1817-425C-A5B1-45F383F230CE', 4500000, 'Won',    '2026-03-01', '2026-05-28')
ON CONFLICT (quoteid) DO NOTHING;


-- ============================================================================
-- 6. Quote items  (51 rows = 50 quotes + 1 mixed Volkswagen quote)
--
--   Product series distribution (after running):
--     ThinkPad     : 30 line items   (laptop / tablet / branch / field deals)
--     ThinkStation : 15 line items   (engineering / workstation / trading floor)
--     ThinkSystem  :  6 line items   (HPC / compute cluster deals)
-- ============================================================================
INSERT INTO lvo_quoteitem (
    lvo_quoteitemid, lvo_name, lvo_quoteid, lvo_item_country,
    lvo_quantity, lvo_unitprice, lvo_totallinevalue,
    lvo_productseries, owninguser, statecode
) VALUES
    -- EMEA BG ────────────────────────────────────────────────────────────
    ('CCDD0001-0001-0001-0001-000000000001', 'ThinkStation P620 x529',          'AABB0001-0001-0001-0001-000000000001', 'DE',  529, 3500, 1851500, 'ThinkStation', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000002', 'ThinkSystem SR650 x192',          'AABB0001-0001-0001-0001-000000000002', 'DE',  192, 18000, 3456000, 'ThinkSystem',  'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000003', 'ThinkPad T14 x2600',              'AABB0001-0001-0001-0001-000000000003', 'DE', 2600, 2000, 5200000, 'ThinkPad',     'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000004', 'ThinkPad X1 Yoga rugged x1200',   'AABB0001-0001-0001-0001-000000000004', 'GB', 1200, 2000, 2400000, 'ThinkPad',     'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000005', 'ThinkPad L14 x1000',              'AABB0001-0001-0001-0001-000000000005', 'DK', 1000, 1750, 1750000, 'ThinkPad',     'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000006', 'ThinkPad E16 x2050',              'AABB0001-0001-0001-0001-000000000006', 'FR', 2050, 2000, 4100000, 'ThinkPad',     'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000007', 'ThinkStation P5 x2000',           'AABB0001-0001-0001-0001-000000000007', 'FR', 2000, 3400, 6800000, 'ThinkStation', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000008', 'ThinkPad T14s x600',              'AABB0001-0001-0001-0001-000000000008', 'DE',  600, 2000, 1200000, 'ThinkPad',     'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000009', 'ThinkStation P3 Tower x800',      'AABB0001-0001-0001-0001-000000000009', 'DE',  800, 4000, 3200000, 'ThinkStation', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000010', 'ThinkPad Lab Tablet x975',        'AABB0001-0001-0001-0001-000000000010', 'DE',  975, 2000, 1950000, 'ThinkPad',     'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    -- Volkswagen Plant 2 — MIXED quote (ThinkPad + ThinkSystem)
    ('CCDD0001-0001-0001-0001-000000000011', 'ThinkPad T14 x1000 (plant)',      'AABB0001-0001-0001-0001-000000000011', 'DE', 1000, 2000, 2000000, 'ThinkPad',     'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000012', 'ThinkSystem SR650 x125 (plant)',  'AABB0001-0001-0001-0001-000000000011', 'DE',  125, 18000, 2250000, 'ThinkSystem',  'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000013', 'ThinkPad X1 Carbon x1425',        'AABB0001-0001-0001-0001-000000000012', 'GB', 1425, 2000, 2850000, 'ThinkPad',     'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000014', 'ThinkStation P3 Tiny x540',       'AABB0001-0001-0001-0001-000000000013', 'NL',  540, 3000, 1620000, 'ThinkStation', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000015', 'ThinkStation P5 trading x1614',   'AABB0001-0001-0001-0001-000000000014', 'FR', 1614, 3500, 5649000, 'ThinkStation', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000016', 'ThinkStation PX CAD x2343',       'AABB0001-0001-0001-0001-000000000015', 'FR', 2343, 3500, 8200500, 'ThinkStation', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000017', 'ThinkSystem SR675 V3 x164',       'AABB0001-0001-0001-0001-000000000016', 'DE',  164, 18000, 2952000, 'ThinkSystem',  'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000018', 'ThinkPad sales-bundle x490',      'AABB0001-0001-0001-0001-000000000017', 'CH',  490, 2000,  980000, 'ThinkPad',     'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000019', 'ThinkPad rugged field x1700',     'AABB0001-0001-0001-0001-000000000018', 'GB', 1700, 2000, 3400000, 'ThinkPad',     'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000020', 'ThinkStation P5 Toulouse x1100',  'AABB0001-0001-0001-0001-000000000019', 'ES', 1100, 3500, 3850000, 'ThinkStation', 'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000021', 'ThinkStation P3 design x614',     'AABB0001-0001-0001-0001-000000000020', 'IT',  614, 3500, 2149000, 'ThinkStation', 'D8C589A4-707E-42C7-ACA1-9B1FC683E6B2', 'Active'),
    -- APAC BG ────────────────────────────────────────────────────────────
    ('CCDD0001-0001-0001-0001-000000000022', 'ThinkPad rugged Toyota x2200',    'AABB0001-0001-0001-0001-000000000021', 'JP', 2200, 2000, 4400000, 'ThinkPad',     '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000023', 'ThinkStation P5 Suwon x1057',     'AABB0001-0001-0001-0001-000000000022', 'KR', 1057, 3500, 3699500, 'ThinkStation', '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000024', 'ThinkPad T14 Pune x1100',         'AABB0001-0001-0001-0001-000000000023', 'IN', 1100, 2000, 2200000, 'ThinkPad',     '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000025', 'ThinkPad CommBank branch x725',   'AABB0001-0001-0001-0001-000000000024', 'AU',  725, 2000, 1450000, 'ThinkPad',     '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000026', 'ThinkStation P5 SG trading x1129','AABB0001-0001-0001-0001-000000000025', 'SG', 1129, 3500, 3951500, 'ThinkStation', '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000027', 'ThinkSystem SR655 x164',          'AABB0001-0001-0001-0001-000000000026', 'IN',  164, 18000, 2952000, 'ThinkSystem',  '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000028', 'ThinkPad Toyota Kentucky x925',   'AABB0001-0001-0001-0001-000000000027', 'JP',  925, 2000, 1850000, 'ThinkPad',     '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000029', 'ThinkPad display tablet x1575',   'AABB0001-0001-0001-0001-000000000028', 'KR', 1575, 2000, 3150000, 'ThinkPad',     '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000030', 'ThinkPad TCS Hyd x1375',          'AABB0001-0001-0001-0001-000000000029', 'IN', 1375, 2000, 2750000, 'ThinkPad',     '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000031', 'ThinkStation CommBank WS x686',   'AABB0001-0001-0001-0001-000000000030', 'AU',  686, 3500, 2401000, 'ThinkStation', '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000032', 'ThinkSystem SR650 HK x236',       'AABB0001-0001-0001-0001-000000000031', 'HK',  236, 18000, 4248000, 'ThinkSystem',  '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000033', 'ThinkPad Mysore x925',            'AABB0001-0001-0001-0001-000000000032', 'IN',  925, 2000, 1850000, 'ThinkPad',     '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000034', 'ThinkPad TCS Mumbai x1525',       'AABB0001-0001-0001-0001-000000000033', 'IN', 1525, 2000, 3050000, 'ThinkPad',     '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000035', 'ThinkSystem SR675 Tokyo x133',    'AABB0001-0001-0001-0001-000000000034', 'JP',  133, 18000, 2394000, 'ThinkSystem',  '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000036', 'ThinkPad Toyota Thailand x990',   'AABB0001-0001-0001-0001-000000000035', 'JP',  990, 2000, 1980000, 'ThinkPad',     '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000037', 'ThinkStation CommBank trader x500','AABB0001-0001-0001-0001-000000000036', 'AU',  500, 3500, 1750000, 'ThinkStation', '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000038', 'ThinkPad HSBC APAC x1825',        'AABB0001-0001-0001-0001-000000000037', 'SG', 1825, 2000, 3650000, 'ThinkPad',     '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000039', 'ThinkPad Infosys Pune x1475',     'AABB0001-0001-0001-0001-000000000038', 'IN', 1475, 2000, 2950000, 'ThinkPad',     '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    -- Americas BG ────────────────────────────────────────────────────────
    ('CCDD0001-0001-0001-0001-000000000040', 'ThinkStation JPM NY HQ x1929',    'AABB0001-0001-0001-0001-000000000039', 'US', 1929, 3500, 6751500, 'ThinkStation', '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000041', 'ThinkPad Tesla Fremont x1825',    'AABB0001-0001-0001-0001-000000000040', 'US', 1825, 2000, 3650000, 'ThinkPad',     '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000042', 'ThinkPad Walmart stores x2725',   'AABB0001-0001-0001-0001-000000000041', 'US', 2725, 2000, 5450000, 'ThinkPad',     '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000043', 'ThinkStation Pfizer Lab x814',    'AABB0001-0001-0001-0001-000000000042', 'US',  814, 3500, 2849000, 'ThinkStation', '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000044', 'ThinkPad AT&T DaaS pilot x875',   'AABB0001-0001-0001-0001-000000000043', 'US',  875, 2000, 1750000, 'ThinkPad',     '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000045', 'ThinkPad Ford Mexico x1475',      'AABB0001-0001-0001-0001-000000000044', 'MX', 1475, 2000, 2950000, 'ThinkPad',     '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000046', 'ThinkStation Tesla Reno x1186',   'AABB0001-0001-0001-0001-000000000045', 'US', 1186, 3500, 4151000, 'ThinkStation', '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000047', 'ThinkPad Walmart DC x1175',       'AABB0001-0001-0001-0001-000000000046', 'US', 1175, 2000, 2350000, 'ThinkPad',     '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000048', 'ThinkPad JPM Toronto x1925',      'AABB0001-0001-0001-0001-000000000047', 'CA', 1925, 2000, 3850000, 'ThinkPad',     '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000049', 'ThinkPad Pfizer Brazil x725',     'AABB0001-0001-0001-0001-000000000048', 'BR',  725, 2000, 1450000, 'ThinkPad',     '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000050', 'ThinkPad AT&T field x1125',       'AABB0001-0001-0001-0001-000000000049', 'US', 1125, 2000, 2250000, 'ThinkPad',     '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('CCDD0001-0001-0001-0001-000000000051', 'ThinkSystem Ford HPC x250',       'AABB0001-0001-0001-0001-000000000050', 'US',  250, 18000, 4500000, 'ThinkSystem',  '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active')
ON CONFLICT (lvo_quoteitemid) DO NOTHING;


-- ============================================================================
-- Sanity checks  (uncomment to verify after running)
-- ============================================================================
-- SELECT COUNT(*) AS opportunities FROM opportunity;          -- expect 55
-- SELECT COUNT(*) AS accounts      FROM account;              -- expect 20
-- SELECT COUNT(*) AS competitors   FROM lvo_opportunitycompetitor; -- expect 29
-- SELECT COUNT(*) AS activities    FROM lvo_activity;         -- expect 117
-- SELECT COUNT(*) AS quotes        FROM quote;                -- expect 55
-- SELECT COUNT(*) AS quote_items   FROM lvo_quoteitem;        -- expect 56
-- SELECT lvo_productseries, COUNT(*) FROM lvo_quoteitem GROUP BY 1 ORDER BY 2 DESC;
-- SELECT lvo_businessgroup, COUNT(*) FROM opportunity GROUP BY 1 ORDER BY 1;
-- SELECT stagename, COUNT(*) FROM opportunity GROUP BY 1 ORDER BY 2 DESC;
-- SELECT lvo_forecastcategory, COUNT(*) FROM opportunity GROUP BY 1 ORDER BY 2 DESC;

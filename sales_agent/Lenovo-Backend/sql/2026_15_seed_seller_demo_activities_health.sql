-- ============================================================================
-- Demo polish: seller 81AADFDB-1817-425C-A5B1-45F383F230CE (Amit / Americas)
--
-- Goals:
--   1. Rich activity timeline — 5 touchpoints per OPEN deal, spread across 90d
--   2. Deal health grid — mostly GREEN, some AMBER, one RED for story
--
-- Grid reads stored lvo_dealhealthscore (not live recalc).
-- Timeline shows up to 5 most-recent activities per deal (see opportunities.py).
--
-- Run in pgAdmin against shared Postgres. Re-running is safe.
-- ============================================================================

-- Seller UUID (case-insensitive match everywhere below)
-- 81AADFDB-1817-425C-A5B1-45F383F230CE

-- ----------------------------------------------------------------------------
-- Step 1 — Remove old sparse activities for this seller's opportunities
-- ----------------------------------------------------------------------------
DELETE FROM lvo_activity
 WHERE UPPER(lvo_opportunityid) IN (
     SELECT UPPER(opportunityid::TEXT)
       FROM opportunity
      WHERE UPPER(owninguser::TEXT) = '81AADFDB-1817-425C-A5B1-45F383F230CE'
 );

-- ----------------------------------------------------------------------------
-- Step 2 — Insert 5 distributed activities per OPEN deal (40 rows)
-- Dates anchored to demo "today" ≈ 2026-06-19, spanning ~82d → 3d ago
-- ----------------------------------------------------------------------------

-- B0000001-0001-0001-0001-000000000039  JPMorgan – NY HQ Workstation Refresh
INSERT INTO lvo_activity (
    lvo_activityid, lvo_opportunityid, lvo_activitytype, lvo_direction,
    lvo_subject, lvo_body, lvo_activitydate, lvo_groupedcount,
    owninguser, statecode
) VALUES
    ('E1000001-0001-0001-0001-000000000001', 'B0000001-0001-0001-0001-000000000039', 'email',    'outbound', 'Initial Outreach',    'Shared trader workstation refresh overview deck.',           '2026-03-29 09:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000002', 'B0000001-0001-0001-0001-000000000039', 'meeting',  'outbound', 'NY HQ Discovery',     'Mapped 1,900-seat trading floor replacement scope.',         '2026-04-20 14:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000003', 'B0000001-0001-0001-0001-000000000039', 'email',    'inbound',  'BOM Feedback',        'IT returned annotated BOM with GPU preferences.',            '2026-05-08 11:30:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000004', 'B0000001-0001-0001-0001-000000000039', 'meeting',  'outbound', 'Executive Demo',      'Live ThinkStation demo for CIO and procurement.',            '2026-05-29 10:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000005', 'B0000001-0001-0001-0001-000000000039', 'crm',      'outbound', 'Stage Advanced',      'Moved to Propose after positive exec readout.',              '2026-06-16 15:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active')
ON CONFLICT (lvo_activityid) DO NOTHING;

-- B0000001-0001-0001-0001-000000000040  Tesla – Fremont Plant Tablets  (RED story — stale tail)
INSERT INTO lvo_activity (
    lvo_activityid, lvo_opportunityid, lvo_activitytype, lvo_direction,
    lvo_subject, lvo_body, lvo_activitydate, lvo_groupedcount,
    owninguser, statecode
) VALUES
    ('E1000001-0001-0001-0001-000000000011', 'B0000001-0001-0001-0001-000000000040', 'email',    'outbound', 'Fremont Intro',       'Sent plant-floor tablet pitch to manufacturing IT.',         '2026-03-25 09:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000012', 'B0000001-0001-0001-0001-000000000040', 'meeting',  'outbound', 'Plant Walkthrough',   'Walked Fremont line — identified 1,200 rugged tablet seats.', '2026-04-18 13:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000013', 'B0000001-0001-0001-0001-000000000040', 'email',    'inbound',  'Pilot Request',       'Client asked for 30-day pilot on Line 3.',                   '2026-05-02 10:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000014', 'B0000001-0001-0001-0001-000000000040', 'crm',      'outbound', 'Risk Flagged',        'No customer response in 18 days — monitor closely.',         '2026-05-20 09:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000015', 'B0000001-0001-0001-0001-000000000040', 'email',    'outbound', 'Follow-up Sent',      'Chased pilot scheduling — awaiting plant ops calendar.',     '2026-06-14 08:30:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active')
ON CONFLICT (lvo_activityid) DO NOTHING;

-- B0000001-0001-0001-0001-000000000041  Walmart – Store Endpoint Refresh
INSERT INTO lvo_activity (
    lvo_activityid, lvo_opportunityid, lvo_activitytype, lvo_direction,
    lvo_subject, lvo_body, lvo_activitydate, lvo_groupedcount,
    owninguser, statecode
) VALUES
    ('E1000001-0001-0001-0001-000000000021', 'B0000001-0001-0001-0001-000000000041', 'email',    'outbound', 'Renewal Kickoff',     'Opened annual store-endpoint renewal conversation.',         '2026-03-28 09:30:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000022', 'B0000001-0001-0001-0001-000000000041', 'meeting',  'outbound', 'Rollout Workshop',    'Aligned on 3-wave store deployment schedule.',               '2026-04-22 14:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000023', 'B0000001-0001-0001-0001-000000000041', 'multiple', 'inbound',  'Contract Day',        'PO + contract + signed quote logged same day.',              '2026-05-10 16:00:00', 3,    '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000024', 'B0000001-0001-0001-0001-000000000041', 'meeting',  'outbound', 'Wave-1 Planning',     'Confirmed first 800-store wave kickoff date.',               '2026-05-30 11:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000025', 'B0000001-0001-0001-0001-000000000041', 'email',    'inbound',  'PO Acknowledged',     'Walmart procurement acknowledged Wave-1 PO.',                '2026-06-17 10:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active')
ON CONFLICT (lvo_activityid) DO NOTHING;

-- B0000001-0001-0001-0001-000000000043  AT&T – Call Centre DaaS Pilot
INSERT INTO lvo_activity (
    lvo_activityid, lvo_opportunityid, lvo_activitytype, lvo_direction,
    lvo_subject, lvo_body, lvo_activitydate, lvo_groupedcount,
    owninguser, statecode
) VALUES
    ('E1000001-0001-0001-0001-000000000031', 'B0000001-0001-0001-0001-000000000043', 'email',    'outbound', 'DaaS Intro',          'Introduced 30-day DaaS pilot for call-centre agents.',       '2026-03-30 10:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000032', 'B0000001-0001-0001-0001-000000000043', 'meeting',  'outbound', 'Pilot Scoping',       'Defined success criteria with AT&T ops lead.',               '2026-04-24 13:30:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000033', 'B0000001-0001-0001-0001-000000000043', 'email',    'inbound',  'Security Review',     'Security team returned DaaS architecture questionnaire.',    '2026-05-12 09:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000034', 'B0000001-0001-0001-0001-000000000043', 'crm',      'outbound', 'Qualify Logged',      'Opportunity advanced to Qualify after BANT.',                 '2026-05-28 14:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000035', 'B0000001-0001-0001-0001-000000000043', 'meeting',  'inbound',  'Pilot Readout',       'Client requested pricing for 2,500 seats post-pilot.',       '2026-06-15 11:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active')
ON CONFLICT (lvo_activityid) DO NOTHING;

-- B0000001-0001-0001-0001-000000000044  Ford – Mexico Plant Expansion
INSERT INTO lvo_activity (
    lvo_activityid, lvo_opportunityid, lvo_activitytype, lvo_direction,
    lvo_subject, lvo_body, lvo_activitydate, lvo_groupedcount,
    owninguser, statecode
) VALUES
    ('E1000001-0001-0001-0001-000000000041', 'B0000001-0001-0001-0001-000000000044', 'email',    'outbound', 'Cuautitlán Outreach', 'Shared Mexico plant expansion device plan.',                 '2026-03-27 08:30:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000042', 'B0000001-0001-0001-0001-000000000044', 'meeting',  'outbound', 'Plant Visit',         'Walked Cuautitlán floor — mapped 1,475 seats.',              '2026-04-19 10:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000043', 'B0000001-0001-0001-0001-000000000044', 'email',    'inbound',  'Scope Sign-off',      'Ford IT signed scope-of-work document.',                    '2026-05-06 15:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000044', 'B0000001-0001-0001-0001-000000000044', 'meeting',  'outbound', 'Technical Review',    'Validated rugged spec with plant engineering.',              '2026-05-27 13:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000045', 'B0000001-0001-0001-0001-000000000044', 'crm',      'outbound', 'Develop Stage',       'Moved to Develop after technical alignment.',                '2026-06-16 09:30:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active')
ON CONFLICT (lvo_activityid) DO NOTHING;

-- B0000001-0001-0001-0001-000000000045  Tesla – Gigafactory Engineer Workstations
INSERT INTO lvo_activity (
    lvo_activityid, lvo_opportunityid, lvo_activitytype, lvo_direction,
    lvo_subject, lvo_body, lvo_activitydate, lvo_groupedcount,
    owninguser, statecode
) VALUES
    ('E1000001-0001-0001-0001-000000000051', 'B0000001-0001-0001-0001-000000000045', 'email',    'outbound', 'Gigafactory Pitch',   'Sent engineer workstation proposal for Reno site.',          '2026-03-26 09:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000052', 'B0000001-0001-0001-0001-000000000045', 'meeting',  'outbound', 'Reno Walkthrough',    'Mapped CAD engineering seats with plant manager.',           '2026-04-21 14:30:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000053', 'B0000001-0001-0001-0001-000000000045', 'email',    'inbound',  'GPU Spec Request',    'Client requested ThinkStation P-series with RTX options.',   '2026-05-09 11:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000054', 'B0000001-0001-0001-0001-000000000045', 'meeting',  'outbound', 'Benchmark Demo',      'Ran SolidWorks benchmark on proposed config.',               '2026-05-30 10:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000055', 'B0000001-0001-0001-0001-000000000045', 'multiple', 'outbound', 'Proposal Day',        'Pricing + legal + technical appendix sent.',                 '2026-06-17 16:00:00', 3,    '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active')
ON CONFLICT (lvo_activityid) DO NOTHING;

-- B0000001-0001-0001-0001-000000000046  Walmart – DC Logistics Devices
INSERT INTO lvo_activity (
    lvo_activityid, lvo_opportunityid, lvo_activitytype, lvo_direction,
    lvo_subject, lvo_body, lvo_activitydate, lvo_groupedcount,
    owninguser, statecode
) VALUES
    ('E1000001-0001-0001-0001-000000000061', 'B0000001-0001-0001-0001-000000000046', 'email',    'outbound', 'DC Outreach',         'Introduced distribution-centre rugged device plan.',         '2026-03-31 09:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000062', 'B0000001-0001-0001-0001-000000000046', 'meeting',  'outbound', 'DC Site Survey',      'Surveyed 4 distribution centres for device counts.',       '2026-04-23 13:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000063', 'B0000001-0001-0001-0001-000000000046', 'email',    'inbound',  'Volume Question',     'Ops asked about scaling to 1,175 devices.',                  '2026-05-11 10:30:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000064', 'B0000001-0001-0001-0001-000000000046', 'crm',      'outbound', 'BANT Confirmed',      'Budget and timeline confirmed — moved to Develop.',          '2026-05-29 14:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000065', 'B0000001-0001-0001-0001-000000000046', 'meeting',  'inbound',  'Procurement Sync',    'Walked renewal terms with Walmart DC procurement.',          '2026-06-15 11:30:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active')
ON CONFLICT (lvo_activityid) DO NOTHING;

-- B0000001-0001-0001-0001-000000000049  AT&T – Field Tech Rugged Tablets
INSERT INTO lvo_activity (
    lvo_activityid, lvo_opportunityid, lvo_activitytype, lvo_direction,
    lvo_subject, lvo_body, lvo_activitydate, lvo_groupedcount,
    owninguser, statecode
) VALUES
    ('E1000001-0001-0001-0001-000000000071', 'B0000001-0001-0001-0001-000000000049', 'email',    'outbound', 'Field Fleet Intro',   'Opened rugged tablet refresh for field technicians.',        '2026-03-29 08:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000072', 'B0000001-0001-0001-0001-000000000049', 'meeting',  'outbound', 'Field Demo',          'Hands-on demo with 12 field tech leads in Dallas.',          '2026-04-20 10:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000073', 'B0000001-0001-0001-0001-000000000049', 'multiple', 'outbound', 'Contract Day',        'Field demo + PO + contract review same day.',                '2026-05-08 15:00:00', 3,    '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000074', 'B0000001-0001-0001-0001-000000000049', 'email',    'inbound',  'PO Submitted',        'AT&T field procurement submitted PO.',                     '2026-05-28 09:30:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('E1000001-0001-0001-0001-000000000075', 'B0000001-0001-0001-0001-000000000049', 'crm',      'outbound', 'Execute Stage',       'Deal in Execute — deployment kickoff scheduled.',            '2026-06-17 13:00:00', NULL, '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active')
ON CONFLICT (lvo_activityid) DO NOTHING;

-- ----------------------------------------------------------------------------
-- Step 3 — Deal health scores
--
-- Do NOT hand-set lvo_dealhealthscore here — it causes grid/detail mismatch.
-- After this script, run:
--   sql/2026_16_sync_deal_health_demo.sql
--   python -m app.jobs.recalc_health --seller-id 81AADFDB-1817-425C-A5B1-45F383F230CE
-- ----------------------------------------------------------------------------

-- Keep one red demo story (Tesla Fremont stale activity)
UPDATE opportunity SET
    lvo_riskscore          = 2,
    lvo_riskreason         = 'Stale Activity'
 WHERE opportunityid = 'B0000001-0001-0001-0001-000000000040';

-- ----------------------------------------------------------------------------
-- Verify
-- ----------------------------------------------------------------------------
-- SELECT name, lvo_dealhealthscore, lvo_riskreason
--   FROM opportunity
--  WHERE UPPER(owninguser::TEXT) = '81AADFDB-1817-425C-A5B1-45F383F230CE'
--    AND statecode = 'Open'
--  ORDER BY name;
--
-- SELECT lvo_opportunityid, COUNT(*) AS activity_count
--   FROM lvo_activity
--  WHERE UPPER(lvo_opportunityid) IN (
--      SELECT UPPER(opportunityid::TEXT) FROM opportunity
--       WHERE UPPER(owninguser::TEXT) = '81AADFDB-1817-425C-A5B1-45F383F230CE'
--  )
--  GROUP BY lvo_opportunityid
--  ORDER BY lvo_opportunityid;

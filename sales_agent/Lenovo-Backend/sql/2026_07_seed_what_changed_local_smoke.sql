-- ============================================================================
-- Local smoke data for Sprint 2 US 1.1 — What Changed feed
--
-- Inserts one recent row per notification type for a single seller so
-- GET /api/notifications and GET /api/activity-timeline return items
-- without manual date hacking.
--
-- Prereqs:
--   * Base D365 seed loaded (opportunity, account, systemuser)
--   * sql/2026_06_create_lvo_activity.sql (table + seed)
--   * sql/2026_06_create_next_actions_audit.sql (audit + next actions)
--   * sql/2026_06_create_dealrisk.sql
--   * sql/2026_07_create_lvo_notification_read.sql
--
-- Test seller: AB3499B1-B088-4F86-B9F2-E458F663ECBF (owns Deutsche Bank deal)
-- Test opportunity: CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B
--
-- Smoke IDs (fixed — re-run safe):
--   Activities (UUID): b1010001-…001, b1020002-…002, b1070007-…007
--   Audit / risk / task (TEXT): SMOKE-0003 … SMOKE-0006
-- ============================================================================

-- Clean prior smoke rows (idempotent re-run)
DELETE FROM lvo_notification_read
 WHERE feed_item_key IN (
    'activity:b1010001-0001-4001-8001-000000000001',
    'activity:b1020002-0002-4002-8002-000000000002',
    'activity:b1070007-0007-4007-8007-000000000007',
    'audit:SMOKE-0003-0003-0003-000000000003:stagename',
    'audit:SMOKE-0006-0006-0006-000000000006:lvo_summary',
    'risk:SMOKE-0004-0004-0004-000000000004',
    'task:SMOKE-0005-0005-0005-000000000005'
 );

DELETE FROM lvo_activity
 WHERE lvo_activityid IN (
    'b1010001-0001-4001-8001-000000000001'::uuid,
    'b1020002-0002-4002-8002-000000000002'::uuid,
    'b1070007-0007-4007-8007-000000000007'::uuid
 );

DELETE FROM lvo_audit_log     WHERE lvo_auditlogid IN (
    'SMOKE-0003-0003-0003-000000000003',
    'SMOKE-0006-0006-0006-000000000006'
);
DELETE FROM lvo_dealrisk      WHERE lvo_dealriskid = 'SMOKE-0004-0004-0004-000000000004';
DELETE FROM lvo_nextaction    WHERE lvo_nextactionid = 'SMOKE-0005-0005-0005-000000000005';

-- ---------------------------------------------------------------------------
-- 1. Email Received (inbound) — appears in panel + timeline
-- ---------------------------------------------------------------------------
INSERT INTO lvo_activity (
    lvo_activityid, lvo_opportunityid, lvo_activitytype, lvo_direction,
    lvo_subject, lvo_body, lvo_activitydate, owninguser, statecode
) VALUES (
    'b1010001-0001-4001-8001-000000000001'::uuid,
    'CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B',
    'email', 'inbound',
    'Re: Infosys ThinkPad Proposal',
    'Kiran Rao replied — scope revised to 1,400 units, proposal needed by Jun 17.',
    NOW() - INTERVAL '47 minutes',
    '7D26391E-D020-474E-B1CA-53E6B6C71487',  -- different rep (not the seller)
    'Active'
) ON CONFLICT (lvo_activityid) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 2. Meeting Done — appears in panel + timeline (direction outbound OK)
-- ---------------------------------------------------------------------------
INSERT INTO lvo_activity (
    lvo_activityid, lvo_opportunityid, lvo_activitytype, lvo_direction,
    lvo_subject, lvo_body, lvo_activitydate, lvo_groupedcount,
    owninguser, statecode
) VALUES (
    'b1020002-0002-4002-8002-000000000002'::uuid,
    'CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B',
    'meeting', 'outbound',
    'Bharat Steel - Pricing Review Call',
    'Completed pricing review with VP Finance.',
    NOW() - INTERVAL '2 hours',
    NULL,
    '7D26391E-D020-474E-B1CA-53E6B6C71487',
    'Active'
) ON CONFLICT (lvo_activityid) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 3. CRM stage change (by someone else) — panel + timeline
--    Simulates PATCH /api/opportunities/{id} with a different X-User-Id
-- ---------------------------------------------------------------------------
INSERT INTO lvo_audit_log (
    lvo_auditlogid, lvo_entitytype, lvo_entityid, lvo_opportunityid,
    lvo_action, lvo_changedby, lvo_changedat, lvo_diff
) VALUES (
    'SMOKE-0003-0003-0003-000000000003',
    'opportunity',
    'CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B',
    'CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B',
    'update',
    '7D26391E-D020-474E-B1CA-53E6B6C71487',
    NOW() - INTERVAL '20 minutes',
    '{"before": {"stagename": "Proposal"}, "after": {"stagename": "Negotiation"}}'
) ON CONFLICT (lvo_auditlogid) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 4. Risk Detected — panel + timeline
-- ---------------------------------------------------------------------------
INSERT INTO lvo_dealrisk (
    lvo_dealriskid, lvo_opportunityid, lvo_riskcategory, lvo_riskname,
    lvo_message, lvo_detectedat, statecode
) VALUES (
    'SMOKE-0004-0004-0004-000000000004',
    'CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B',
    'Activity & Engagement',
    'Low Activity',
    'No logged touchpoint in 14+ days on this deal.',
    NOW() - INTERVAL '1 day',
    'Active'
) ON CONFLICT (lvo_dealriskid) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 5. Task Overdue — panel + timeline
-- ---------------------------------------------------------------------------
INSERT INTO lvo_nextaction (
    lvo_nextactionid, lvo_opportunityid, lvo_description, lvo_duedate,
    lvo_status, lvo_createdat, lvo_updatedat, lvo_createdby, statecode
) VALUES (
    'SMOKE-0005-0005-0005-000000000005',
    'CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B',
    'Send revised proposal to procurement',
    CURRENT_DATE - 1,
    'Open',
    NOW() - INTERVAL '5 days',
    NOW() - INTERVAL '1 day',
    '7D26391E-D020-474E-B1CA-53E6B6C71487',
    'Active'
) ON CONFLICT (lvo_nextactionid) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 6. Seller's own CRM change — timeline ONLY (excluded from panel)
-- ---------------------------------------------------------------------------
INSERT INTO lvo_audit_log (
    lvo_auditlogid, lvo_entitytype, lvo_entityid, lvo_opportunityid,
    lvo_action, lvo_changedby, lvo_changedat, lvo_diff
) VALUES (
    'SMOKE-0006-0006-0006-000000000006',
    'opportunity',
    'CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B',
    'CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B',
    'update',
    'AB3499B1-B088-4F86-B9F2-E458F663ECBF',  -- the seller — should NOT notify
    NOW() - INTERVAL '5 minutes',
    '{"before": {"lvo_summary": "Old summary"}, "after": {"lvo_summary": "New summary"}}'
) ON CONFLICT (lvo_auditlogid) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 7. Outbound email — excluded everywhere (seller outbound not a notification)
-- ---------------------------------------------------------------------------
INSERT INTO lvo_activity (
    lvo_activityid, lvo_opportunityid, lvo_activitytype, lvo_direction,
    lvo_subject, lvo_body, lvo_activitydate, lvo_groupedcount,
    owninguser, statecode
) VALUES (
    'b1070007-0007-4007-8007-000000000007'::uuid,
    'CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B',
    'email', 'outbound',
    'Follow-up sent',
    'Seller sent follow-up — should not appear in feed.',
    NOW() - INTERVAL '10 minutes',
    NULL,
    'AB3499B1-B088-4F86-B9F2-E458F663ECBF',
    'Active'
) ON CONFLICT (lvo_activityid) DO NOTHING;

-- ============================================================================
-- Alternate seller: 055DAFE7 (Ford Motor — base 5-deal seed)
-- Ensures the opp is owned by this seller and has fresh in-window events.
-- ============================================================================

UPDATE opportunity
   SET owninguser = '055DAFE7-9840-451D-8328-5F70A6326C03',
       statecode = 'Open'
 WHERE UPPER(opportunityid::TEXT) = '5977B053-8389-4497-BA97-076CBA41FB86';

DELETE FROM lvo_activity
 WHERE lvo_activityid IN (
    'b2010001-0001-4001-8001-000000000001'::uuid,
    'b2020002-0002-4002-8002-000000000002'::uuid
 );

INSERT INTO lvo_activity (
    lvo_activityid, lvo_opportunityid, lvo_activitytype, lvo_direction,
    lvo_subject, lvo_body, lvo_activitydate, owninguser, statecode
) VALUES (
    'b2010001-0001-4001-8001-000000000001'::uuid,
    '5977B053-8389-4497-BA97-076CBA41FB86',
    'email', 'inbound',
    'Re: Ford HPC cluster sizing',
    'Procurement confirmed revised node count for Q3 rollout.',
    NOW() - INTERVAL '35 minutes',
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    'Active'
) ON CONFLICT (lvo_activityid) DO NOTHING;

INSERT INTO lvo_activity (
    lvo_activityid, lvo_opportunityid, lvo_activitytype, lvo_direction,
    lvo_subject, lvo_body, lvo_activitydate, lvo_groupedcount,
    owninguser, statecode
) VALUES (
    'b2020002-0002-4002-8002-000000000002'::uuid,
    '5977B053-8389-4497-BA97-076CBA41FB86',
    'meeting', 'outbound',
    'Ford — Pricing Review Call',
    'Completed pricing review with VP Finance.',
    NOW() - INTERVAL '3 hours',
    NULL,
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    'Active'
) ON CONFLICT (lvo_activityid) DO NOTHING;

-- Sanity:
-- SELECT lvo_activityid, lvo_subject FROM lvo_activity
--  WHERE lvo_activityid IN (
--    'b1010001-0001-4001-8001-000000000001'::uuid,
--    'b1020002-0002-4002-8002-000000000002'::uuid,
--    'b1070007-0007-4007-8007-000000000007'::uuid
--  );

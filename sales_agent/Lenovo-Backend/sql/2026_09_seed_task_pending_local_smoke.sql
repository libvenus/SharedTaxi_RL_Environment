-- Optional local smoke for Sprint 2 US 1.3 Task Pending badge (D365 next actions).
-- Requires an open opportunity owned by SMOKE_SELLER_ID below.

-- Replace with your local seller UUID (matches opportunity.owninguser).
\set SMOKE_SELLER_ID '055DAFE7-9840-451D-8328-5F70A6326C03'
\set SMOKE_OPP_ID 'SMOKE-OPP-TASK-PENDING-0001'

DELETE FROM lvo_nextaction
 WHERE lvo_nextactionid IN (
    'SMOKE-TASK-0001-0001-0001-000000000001',
    'SMOKE-TASK-0001-0001-0001-000000000002',
    'SMOKE-TASK-0001-0001-0001-000000000003'
 );

INSERT INTO lvo_nextaction (
    lvo_nextactionid, lvo_opportunityid, lvo_description, lvo_duedate,
    verbal_written_acceptance, lvo_status, lvo_createdat, lvo_updatedat, statecode
) VALUES
(
    'SMOKE-TASK-0001-0001-0001-000000000001',
    :'SMOKE_OPP_ID',
    'Overdue follow-up call',
    CURRENT_DATE - INTERVAL '2 days',
    'Verbal', 'Open', NOW(), NOW(), 'Active'
),
(
    'SMOKE-TASK-0001-0001-0001-000000000002',
    :'SMOKE_OPP_ID',
    'Send proposal today',
    CURRENT_DATE,
    'Written', 'Open', NOW(), NOW(), 'Active'
),
(
    'SMOKE-TASK-0001-0001-0001-000000000003',
    :'SMOKE_OPP_ID',
    'Future demo prep',
    CURRENT_DATE + INTERVAL '5 days',
    'Verbal', 'Open', NOW(), NOW(), 'Active'
)
ON CONFLICT (lvo_nextactionid) DO NOTHING;

-- curl "http://localhost:8000/api/tasks/pending-summary?sellerId=<SMOKE_SELLER_ID>"
-- Expect count=3, overdueCount=1, dueTodayCount=1, hasOverdue=true, badgeColor=red

-- Sprint 2 · Execute Workspace To-Do — realistic demo seed (AIBackend Postgres)
--
-- Wipes tbl_to_do_list and loads seller-scoped tasks aligned with D365 bulk seed
-- (Infosys, Commonwealth Bank, Pfizer, TCS). Due dates are relative to CURRENT_DATE
-- so overdue / today / upcoming buckets stay fresh for demos.
--
-- Run on the AIBackend database (dev: 10.245.240.33):
--   psql -h <host> -U <user> -d <dbname> -f sql/2026_10_seed_execute_todo_demo.sql
--
-- Verify:
--   curl "http://10.245.240.33/ai-api/todos/?filter_type=all&sellerId=055DAFE7-9840-451D-8328-5F70A6326C03"
-- Expect: summary overdue=3, today=3, upcoming=4, no_due_date=2; filters all=12

BEGIN;

DELETE FROM tbl_to_do_list;

INSERT INTO tbl_to_do_list (
    task_title,
    type_tag,
    priority,
    source_label,
    linked_account_id,
    linked_opportunity_id,
    attendees_email,
    notes,
    status,
    due_date,
    seller_id,
    created_at,
    updated_at
) VALUES
-- Overdue (3)
(
    'Follow up with Infosys procurement on HPC cluster pricing',
    'outreach',
    'high',
    'Meeting · Infosys HPC Architect Walkthrough',
    '1C603E6A-C4EA-4E97-B73E-345A5A9C2460',
    'B0000001-0001-0001-0001-000000000026',
    'rajesh.kumar@infosys.com,procurement-hpc@infosys.com',
    'Rajesh asked for revised TCO vs SGI/HPE before internal steering committee. Include 3-year support bundle.',
    'Open',
    CURRENT_DATE - INTERVAL '3 days',
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    NOW() - INTERVAL '5 days',
    NOW() - INTERVAL '1 day'
),
(
    'Schedule workstation validation session with CommBank IT',
    'action',
    'high',
    'Meeting · CommBank Pricing Workshop',
    'A0000001-AAAA-0001-0001-000000000012',
    'B0000001-0001-0001-0001-000000000030',
    'sarah.chen@cba.com.au,it-procurement@cba.com.au',
    'Procurement wants a 45-min session with endpoint engineering before sign-off on volume discount tier.',
    'Open',
    CURRENT_DATE - INTERVAL '1 day',
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    NOW() - INTERVAL '4 days',
    NOW() - INTERVAL '12 hours'
),
(
    'Upload revised BOM for TCS Mumbai tower refresh',
    'document',
    'medium',
    'Meeting · TCS Procurement Sync',
    'A0000001-AAAA-0001-0001-000000000010',
    'B0000001-0001-0001-0001-000000000033',
    'amit.shah@tcs.com',
    'Version 2.1 should reflect ThinkPad T14s qty change discussed on 31 May call. Attach to opportunity files.',
    'Open',
    CURRENT_DATE - INTERVAL '2 days',
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    NOW() - INTERVAL '6 days',
    NOW() - INTERVAL '2 days'
),

-- Due today (3)
(
    'Send thank-you email after Pfizer Boston lab tour',
    'outreach',
    'medium',
    'Meeting · Pfizer Lab Tour',
    'A0000001-AAAA-0001-0001-000000000008',
    'B0000001-0001-0001-0001-000000000042',
    'dr.martinez@pfizer.com,r-and-d-ops@pfizer.com',
    'Reference validated GPU workstation config and offer onsite POC for two bench chemists.',
    'Open',
    CURRENT_DATE,
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    NOW() - INTERVAL '2 days',
    NOW() - INTERVAL '3 hours'
),
(
    'Confirm attendee list for Infosys Mysore campus walkthrough',
    'action',
    'high',
    'AI Suggested',
    '1C603E6A-C4EA-4E97-B73E-345A5A9C2460',
    'B0000001-0001-0001-0001-000000000032',
    'facilities@infosys.com,training-ops@infosys.com',
    'Need facilities + training ops leads on invite. Campus visit drives device count for Mysore expansion deal.',
    'Open',
    CURRENT_DATE,
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    NOW() - INTERVAL '1 day',
    NOW() - INTERVAL '1 hour'
),
(
    'Attach executed NDA to Commonwealth Bank opportunity',
    'document',
    'low',
    'Manual',
    'A0000001-AAAA-0001-0001-000000000012',
    'B0000001-0001-0001-0001-000000000030',
    NULL,
    'Legal returned signed NDA yesterday — upload PDF to D365 and notify procurement contact.',
    'Open',
    CURRENT_DATE,
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    NOW() - INTERVAL '3 days',
    NOW() - INTERVAL '6 hours'
),

-- Upcoming (4)
(
    'Introduce TCS Hyderabad tower stakeholders to Lenovo APAC architect',
    'outreach',
    'medium',
    'Meeting · TCS Hyderabad Plan Review',
    'A0000001-AAAA-0001-0001-000000000010',
    'B0000001-0001-0001-0001-000000000029',
    'hyd-tower-leads@tcs.com',
    'Warm intro email ahead of deployment planning workshop — mention Hyderabad standardisation timeline.',
    'Open',
    CURRENT_DATE + INTERVAL '2 days',
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    NOW() - INTERVAL '1 day',
    NOW()
),
(
    'Book ThinkPad fleet demo for Infosys Mysore training campus',
    'action',
    'high',
    'AI Suggested',
    '1C603E6A-C4EA-4E97-B73E-345A5A9C2460',
    'B0000001-0001-0001-0001-000000000032',
    'training-ops@infosys.com',
    'Demo should cover ruggedized vs standard T-series for trainee labs. Block 90 minutes on site.',
    'Open',
    CURRENT_DATE + INTERVAL '5 days',
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    NOW(),
    NOW()
),
(
    'Prepare executive summary deck for Pfizer lab workstation refresh',
    'document',
    'medium',
    'Manual',
    'A0000001-AAAA-0001-0001-000000000008',
    'B0000001-0001-0001-0001-000000000042',
    NULL,
    'One-pager for R&D VP: business case, phased rollout, and compliance posture for regulated lab environment.',
    'Open',
    CURRENT_DATE + INTERVAL '7 days',
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    NOW(),
    NOW()
),
(
    'Send HPE displacement case study to Infosys HPC committee',
    'outreach',
    'high',
    'Meeting · Infosys HPC Architect Walkthrough',
    '1C603E6A-C4EA-4E97-B73E-345A5A9C2460',
    'B0000001-0001-0001-0001-000000000026',
    'committee-hpc@infosys.com',
    'Use APAC manufacturing customer reference. Highlight 18% TCO reduction vs incumbent SGI/HPE bid.',
    'Open',
    CURRENT_DATE + INTERVAL '3 days',
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    NOW() - INTERVAL '2 days',
    NOW()
),

-- No due date (2)
(
    'Review competitor intel on SGI/HPE for Infosys Bangalore HPC deal',
    'action',
    'medium',
    'AI Suggested',
    '1C603E6A-C4EA-4E97-B73E-345A5A9C2460',
    'B0000001-0001-0001-0001-000000000026',
    NULL,
    'Cross-check last quarter win/loss notes and update battlecard before next steering committee.',
    'Open',
    NULL,
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    NOW() - INTERVAL '4 days',
    NOW() - INTERVAL '1 day'
),
(
    'Collect customer reference letter for CommBank workstation renewal',
    'document',
    'low',
    'Manual',
    'A0000001-AAAA-0001-0001-000000000012',
    'B0000001-0001-0001-0001-000000000024',
    NULL,
    'Ask branch IT lead for short quote on deployment speed and support SLA — needed for proposal appendix.',
    'Open',
    NULL,
    '055DAFE7-9840-451D-8328-5F70A6326C03',
    NOW() - INTERVAL '7 days',
    NOW() - INTERVAL '3 days'
);

COMMIT;

-- Badge smoke (open count = 12, overdue = 3 → red badge):
-- curl "http://10.245.240.33/ai-api/todos/summary?sellerId=055DAFE7-9840-451D-8328-5F70A6326C03"

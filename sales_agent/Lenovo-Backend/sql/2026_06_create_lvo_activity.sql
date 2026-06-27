-- ============================================================================
-- lvo_activity — Sales activity log per opportunity
--
-- Replaces the hard-coded `acts` array that used to live in the React UI
-- (LenovoD365/src/pages/Opportunities.jsx).  Each row is one logged touchpoint
-- (email / meeting / CRM update / multi-event), tied to a real opportunity
-- already loaded from lenovo_nitro_d365_postgres.sql.
--
-- Run this in pgAdmin against the same database that holds `opportunity`.
-- Re-running is safe: CREATE uses IF NOT EXISTS, INSERTs use ON CONFLICT.
-- ============================================================================

CREATE TABLE IF NOT EXISTS lvo_activity (
    lvo_activityid       UUID PRIMARY KEY,
    lvo_opportunityid    TEXT NOT NULL,                       -- FK -> opportunity.opportunityid (TEXT/UUID stored as text)
    lvo_activitytype     TEXT NOT NULL,                       -- 'email' | 'meeting' | 'crm' | 'multiple'
    lvo_direction        TEXT,                                -- 'inbound' | 'outbound'
    lvo_subject          TEXT,                                -- short title shown in the dot popup
    lvo_body             TEXT,                                -- longer description shown in the offcanvas
    lvo_activitydate     TIMESTAMP NOT NULL,                  -- precise event time
    lvo_groupedcount     INTEGER,                             -- only set when type='multiple' (how many events that day)
    owninguser           TEXT,                                -- systemuser.systemuserid of the rep who logged it
    statecode            TEXT NOT NULL DEFAULT 'Active'       -- 'Active' | 'Inactive'
);

CREATE INDEX IF NOT EXISTS idx_lvo_activity_opportunity
    ON lvo_activity (lvo_opportunityid);

CREATE INDEX IF NOT EXISTS idx_lvo_activity_date
    ON lvo_activity (lvo_activitydate DESC);


-- ============================================================================
-- Seed data
--
-- Five opportunities exist in the sample DB:
--   CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B  Deutsche Bank — Workstation Refresh
--   F86EC95A-F627-4C12-A3A5-4ADBC43C6DBA  Siemens AG — Edge AI Infrastructure
--   116191C4-CE2F-46CB-8666-4861A6A1AE26  HSBC — DaaS Rollout APAC
--   5977B053-8389-4497-BA97-076CBA41FB86  Ford Motor — HPC Cluster Upgrade
--   84D4BB4D-E2DC-4B46-9D32-F7D1D182B414  Infosys — Developer Laptop Refresh
--
-- Activity content is lifted verbatim from the original UI mock so the screen
-- looks identical to before, just driven from the DB.
-- ============================================================================

-- --- Deutsche Bank — Workstation Refresh -----------------------------------
INSERT INTO lvo_activity VALUES
    ('A1A1A1A1-0001-0001-0001-000000000001',
     'CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B',
     'email', 'inbound',
     'Proposal Review Request',
     'Client requested revised pricing for 1,400 units instead of 1,200.',
     '2026-04-18 10:00:00', NULL,
     'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('A1A1A1A1-0001-0001-0001-000000000002',
     'CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B',
     'meeting', 'outbound',
     'Demo Walkthrough',
     'Completed live product demo with IT decision team.',
     '2026-04-25 14:30:00', NULL,
     'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active'),
    ('A1A1A1A1-0001-0001-0001-000000000003',
     'CCA8DD60-9CF7-401F-B17C-F4A72B9FFD7B',
     'multiple', 'inbound',
     'Multiple Events',
     'Contract review + pricing call + CRM update logged same day.',
     '2026-05-23 16:00:00', 3,
     'AB3499B1-B088-4F86-B9F2-E458F663ECBF', 'Active')
ON CONFLICT (lvo_activityid) DO NOTHING;


-- --- Siemens AG — Edge AI Infrastructure -----------------------------------
INSERT INTO lvo_activity VALUES
    ('A2A2A2A2-0002-0002-0002-000000000001',
     'F86EC95A-F627-4C12-A3A5-4ADBC43C6DBA',
     'email', 'inbound',
     'Budget Freeze Alert',
     'Client flagged potential budget freeze for Q2.',
     '2026-03-24 08:00:00', NULL,
     '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('A2A2A2A2-0002-0002-0002-000000000002',
     'F86EC95A-F627-4C12-A3A5-4ADBC43C6DBA',
     'email', 'outbound',
     'Re-engagement Email',
     'Sent ROI deck to CIO to unblock budget.',
     '2026-04-06 15:30:00', NULL,
     '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('A2A2A2A2-0002-0002-0002-000000000003',
     'F86EC95A-F627-4C12-A3A5-4ADBC43C6DBA',
     'email', 'inbound',
     'Response Received',
     'Client confirmed budget review in progress.',
     '2026-04-19 10:00:00', NULL,
     '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active'),
    ('A2A2A2A2-0002-0002-0002-000000000004',
     'F86EC95A-F627-4C12-A3A5-4ADBC43C6DBA',
     'crm', 'outbound',
     'Risk Score Updated',
     'Risk elevated to 3 pending competitor evaluation.',
     '2026-05-23 11:00:00', NULL,
     '7D26391E-D020-474E-B1CA-53E6B6C71487', 'Active')
ON CONFLICT (lvo_activityid) DO NOTHING;


-- --- HSBC — DaaS Rollout APAC ----------------------------------------------
INSERT INTO lvo_activity VALUES
    ('A3A3A3A3-0003-0003-0003-000000000001',
     '116191C4-CE2F-46CB-8666-4861A6A1AE26',
     'crm', 'inbound',
     'Renewal Window Open',
     '60-day renewal window flagged in CRM.',
     '2026-04-24 09:00:00', NULL,
     '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active'),
    ('A3A3A3A3-0003-0003-0003-000000000002',
     '116191C4-CE2F-46CB-8666-4861A6A1AE26',
     'crm', 'outbound',
     'Proposal Sent',
     '3-year renewal proposal sent to procurement.',
     '2026-05-25 16:00:00', NULL,
     '81AADFDB-1817-425C-A5B1-45F383F230CE', 'Active')
ON CONFLICT (lvo_activityid) DO NOTHING;


-- --- Ford Motor — HPC Cluster Upgrade --------------------------------------
INSERT INTO lvo_activity VALUES
    ('A4A4A4A4-0004-0004-0004-000000000001',
     '5977B053-8389-4497-BA97-076CBA41FB86',
     'meeting', 'outbound',
     'Kickoff Meeting',
     'Initial engagement with procurement lead.',
     '2026-03-19 11:00:00', NULL,
     '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('A4A4A4A4-0004-0004-0004-000000000002',
     '5977B053-8389-4497-BA97-076CBA41FB86',
     'multiple', 'inbound',
     'Multiple Events',
     'Contract + pricing + legal review logged.',
     '2026-03-22 14:00:00', 3,
     '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('A4A4A4A4-0004-0004-0004-000000000003',
     '5977B053-8389-4497-BA97-076CBA41FB86',
     'meeting', 'outbound',
     'Pricing Review Call',
     'Walk through final pricing with VP Finance.',
     '2026-04-08 10:00:00', NULL,
     '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active'),
    ('A4A4A4A4-0004-0004-0004-000000000004',
     '5977B053-8389-4497-BA97-076CBA41FB86',
     'crm', 'inbound',
     'PO Submitted',
     'Purchase order submitted by procurement.',
     '2026-05-25 09:30:00', NULL,
     '055DAFE7-9840-451D-8328-5F70A6326C03', 'Active')
ON CONFLICT (lvo_activityid) DO NOTHING;


-- --- Infosys — Developer Laptop Refresh ------------------------------------
INSERT INTO lvo_activity VALUES
    ('A5A5A5A5-0005-0005-0005-000000000001',
     '84D4BB4D-E2DC-4B46-9D32-F7D1D182B414',
     'multiple', 'inbound',
     'Multiple Events',
     'Workshop + BOM request + CRM update same day.',
     '2026-03-22 09:00:00', 3,
     '1B77CECE-1B98-411D-A19B-DB076DC0C871', 'Active'),
    ('A5A5A5A5-0005-0005-0005-000000000002',
     '84D4BB4D-E2DC-4B46-9D32-F7D1D182B414',
     'crm', 'outbound',
     'Stage Update',
     'Moved to Qualify. Next: technical validation.',
     '2026-04-04 09:15:00', NULL,
     '1B77CECE-1B98-411D-A19B-DB076DC0C871', 'Active'),
    ('A5A5A5A5-0005-0005-0005-000000000003',
     '84D4BB4D-E2DC-4B46-9D32-F7D1D182B414',
     'crm', 'outbound',
     'Solution Area Updated',
     'Changed to Enterprise IT. DQR approved.',
     '2026-05-05 14:00:00', NULL,
     '1B77CECE-1B98-411D-A19B-DB076DC0C871', 'Active'),
    ('A5A5A5A5-0005-0005-0005-000000000004',
     '84D4BB4D-E2DC-4B46-9D32-F7D1D182B414',
     'crm', 'inbound',
     'Note Added',
     'PO expected by end of June.',
     '2026-05-29 11:00:00', NULL,
     '1B77CECE-1B98-411D-A19B-DB076DC0C871', 'Active')
ON CONFLICT (lvo_activityid) DO NOTHING;


-- Sanity check (optional — uncomment to run):
-- SELECT lvo_opportunityid, COUNT(*) FROM lvo_activity GROUP BY lvo_opportunityid ORDER BY 1;

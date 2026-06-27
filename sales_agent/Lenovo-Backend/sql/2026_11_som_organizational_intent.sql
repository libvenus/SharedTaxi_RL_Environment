-- ============================================================================
-- Sprint 2 · US 3.2.2 — Sales Operating Model · Organizational Intent Setup
-- ============================================================================

CREATE TABLE IF NOT EXISTS lvo_som_organizational_intent (
    lvo_intenttype      VARCHAR(32) PRIMARY KEY
        CHECK (lvo_intenttype IN (
            'outcome', 'motion', 'focus', 'behavioral', 'constraint', 'tradeoff'
        )),
    lvo_displayname     VARCHAR(64) NOT NULL,
    lvo_status          VARCHAR(32) NOT NULL DEFAULT 'NOT_CONFIGURED'
        CHECK (lvo_status IN ('NOT_CONFIGURED', 'CONFIGURED')),
    lvo_fields          JSONB NOT NULL DEFAULT '{}'::JSONB,
    lvo_is_timeboxed    BOOLEAN NOT NULL DEFAULT FALSE,
    lvo_is_guardrail    BOOLEAN NOT NULL DEFAULT FALSE,
    lvo_expiry_date     DATE,
    lvo_last_synced_at  TIMESTAMPTZ,
    lvo_configured_by   VARCHAR(128),
    lvo_cycleid         VARCHAR(36)
        REFERENCES lvo_som_configuration_cycle (lvo_cycleid)
);

INSERT INTO lvo_som_organizational_intent (
    lvo_intenttype, lvo_displayname, lvo_status, lvo_is_timeboxed, lvo_is_guardrail, lvo_cycleid
)
VALUES
    ('outcome',     'Outcome',     'NOT_CONFIGURED', FALSE, FALSE, '00000000-0000-4000-8000-000000000001'),
    ('motion',      'Motion',      'NOT_CONFIGURED', FALSE, FALSE, '00000000-0000-4000-8000-000000000001'),
    ('focus',       'Focus',       'NOT_CONFIGURED', TRUE,  FALSE, '00000000-0000-4000-8000-000000000001'),
    ('behavioral',  'Behavioral',  'NOT_CONFIGURED', FALSE, FALSE, '00000000-0000-4000-8000-000000000001'),
    ('constraint',  'Constraint',  'NOT_CONFIGURED', FALSE, TRUE,  '00000000-0000-4000-8000-000000000001'),
    ('tradeoff',    'Trade-off',   'NOT_CONFIGURED', FALSE, FALSE, '00000000-0000-4000-8000-000000000001')
ON CONFLICT (lvo_intenttype) DO NOTHING;

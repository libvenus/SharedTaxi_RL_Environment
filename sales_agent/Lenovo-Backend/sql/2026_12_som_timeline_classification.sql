-- ============================================================================
-- Sprint 2 · US 3.4.1 — Sales Operating Model · Timeline Classification
-- Depends on: sql/2026_10_som_interview_setup.sql, sql/2026_11_som_organizational_intent.sql
-- ============================================================================

CREATE TABLE IF NOT EXISTS lvo_som_timeline_classification (
    lvo_cardtype        VARCHAR(64) PRIMARY KEY
        CHECK (lvo_cardtype IN (
            'tempo_classes',
            'anchor_definitions',
            'signal_expectations_time_band',
            'seasonal_delayed_activation',
            'acceleration_decay',
            'multiyear_programmatic',
            'exit_recycle_kill',
            'canonical_timeline'
        )),
    lvo_displayname     VARCHAR(128) NOT NULL,
    lvo_status          VARCHAR(32) NOT NULL DEFAULT 'NOT_CONFIGURED'
        CHECK (lvo_status IN ('NOT_CONFIGURED', 'CONFIGURED')),
    lvo_fields          JSONB NOT NULL DEFAULT '{}'::JSONB,
    lvo_last_synced_at  TIMESTAMPTZ,
    lvo_configured_by   VARCHAR(128),
    lvo_cycleid         VARCHAR(36)
        REFERENCES lvo_som_configuration_cycle (lvo_cycleid)
);

INSERT INTO lvo_som_timeline_classification (
    lvo_cardtype, lvo_displayname, lvo_status, lvo_cycleid
)
VALUES
    ('tempo_classes',                  'Tempo classes',                              'NOT_CONFIGURED', '00000000-0000-4000-8000-000000000001'),
    ('anchor_definitions',             'Anchor definitions',                         'NOT_CONFIGURED', '00000000-0000-4000-8000-000000000001'),
    ('signal_expectations_time_band',  'Signal expectations by time band',         'NOT_CONFIGURED', '00000000-0000-4000-8000-000000000001'),
    ('seasonal_delayed_activation',    'Seasonal and delayed activation',            'NOT_CONFIGURED', '00000000-0000-4000-8000-000000000001'),
    ('acceleration_decay',             'Acceleration and decay',                     'NOT_CONFIGURED', '00000000-0000-4000-8000-000000000001'),
    ('multiyear_programmatic',         'Multi-year and programmatic deals',          'NOT_CONFIGURED', '00000000-0000-4000-8000-000000000001'),
    ('exit_recycle_kill',              'Exit, recycle, and kill',                    'NOT_CONFIGURED', '00000000-0000-4000-8000-000000000001'),
    ('canonical_timeline',             'Canonical (quarter/yearly) timeline',        'NOT_CONFIGURED', '00000000-0000-4000-8000-000000000001')
ON CONFLICT (lvo_cardtype) DO NOTHING;

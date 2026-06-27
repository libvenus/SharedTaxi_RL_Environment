-- ============================================================================
-- Sprint 2 · US 3.2.1 — Sales Operating Model · Interview-First Setup
--
-- Stores interview questions, draft/saved responses, intent card status,
-- and the Context Lake snapshot consumed by AI agents.
-- ============================================================================

CREATE TABLE IF NOT EXISTS lvo_som_interview_question (
    lvo_questionid    VARCHAR(36) PRIMARY KEY,
    lvo_role          VARCHAR(32) NOT NULL
        CHECK (lvo_role IN ('national_manager', 'regional_manager', 'seller_manager')),
    lvo_sortorder     SMALLINT NOT NULL CHECK (lvo_sortorder >= 1),
    lvo_questiontext  TEXT NOT NULL,
    lvo_isactive      BOOLEAN NOT NULL DEFAULT TRUE,
    lvo_createdat     TIMESTAMPTZ NOT NULL DEFAULT now(),
    lvo_updatedat     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_lvo_som_question_role_sort
    ON lvo_som_interview_question (lvo_role, lvo_sortorder)
    WHERE lvo_isactive = TRUE;

CREATE TABLE IF NOT EXISTS lvo_som_configuration_cycle (
    lvo_cycleid       VARCHAR(36) PRIMARY KEY,
    lvo_status        VARCHAR(32) NOT NULL DEFAULT 'in_progress'
        CHECK (lvo_status IN ('in_progress', 'configured')),
    lvo_createdat     TIMESTAMPTZ NOT NULL DEFAULT now(),
    lvo_configuredat  TIMESTAMPTZ,
    lvo_configuredby  VARCHAR(128)
);

CREATE TABLE IF NOT EXISTS lvo_som_interview_response (
    lvo_responseid    VARCHAR(36) PRIMARY KEY,
    lvo_cycleid       VARCHAR(36) NOT NULL
        REFERENCES lvo_som_configuration_cycle (lvo_cycleid),
    lvo_questionid    VARCHAR(36) NOT NULL
        REFERENCES lvo_som_interview_question (lvo_questionid),
    lvo_role          VARCHAR(32) NOT NULL
        CHECK (lvo_role IN ('national_manager', 'regional_manager', 'seller_manager')),
    lvo_responsetext  TEXT NOT NULL DEFAULT '',
    lvo_status        VARCHAR(16) NOT NULL DEFAULT 'draft'
        CHECK (lvo_status IN ('draft', 'saved')),
    lvo_capturedby    VARCHAR(128),
    lvo_capturedat    TIMESTAMPTZ NOT NULL DEFAULT now(),
    lvo_savedat       TIMESTAMPTZ,
    UNIQUE (lvo_cycleid, lvo_questionid, lvo_status)
);

CREATE INDEX IF NOT EXISTS ix_lvo_som_response_role_status
    ON lvo_som_interview_response (lvo_role, lvo_status);

CREATE TABLE IF NOT EXISTS lvo_som_intent_card (
    lvo_role          VARCHAR(32) PRIMARY KEY
        CHECK (lvo_role IN ('national_manager', 'regional_manager', 'seller_manager')),
    lvo_status        VARCHAR(32) NOT NULL DEFAULT 'NOT_CONFIGURED'
        CHECK (lvo_status IN ('NOT_CONFIGURED', 'CONFIGURED')),
    lvo_configuredat  TIMESTAMPTZ,
    lvo_configuredby  VARCHAR(128),
    lvo_cycleid       VARCHAR(36)
        REFERENCES lvo_som_configuration_cycle (lvo_cycleid)
);

CREATE TABLE IF NOT EXISTS lvo_som_context_lake (
    id                SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    lvo_payload       JSONB NOT NULL DEFAULT '{}'::JSONB,
    lvo_updatedat     TIMESTAMPTZ NOT NULL DEFAULT now(),
    lvo_updatedby     VARCHAR(128)
);

-- ---------------------------------------------------------------------------
-- Seed default interview questions (5 per role)
-- ---------------------------------------------------------------------------

INSERT INTO lvo_som_configuration_cycle (lvo_cycleid, lvo_status)
VALUES ('00000000-0000-4000-8000-000000000001', 'in_progress')
ON CONFLICT (lvo_cycleid) DO NOTHING;

INSERT INTO lvo_som_intent_card (lvo_role, lvo_status, lvo_cycleid)
VALUES
    ('national_manager', 'NOT_CONFIGURED', '00000000-0000-4000-8000-000000000001'),
    ('regional_manager', 'NOT_CONFIGURED', '00000000-0000-4000-8000-000000000001'),
    ('seller_manager',   'NOT_CONFIGURED', '00000000-0000-4000-8000-000000000001')
ON CONFLICT (lvo_role) DO NOTHING;

INSERT INTO lvo_som_context_lake (id, lvo_payload)
VALUES (1, '{"version":1,"roles":{}}'::JSONB)
ON CONFLICT (id) DO NOTHING;

-- National Manager
INSERT INTO lvo_som_interview_question (lvo_questionid, lvo_role, lvo_sortorder, lvo_questiontext)
VALUES
    ('11111111-1111-4111-8111-111111111101', 'national_manager', 1, 'What does a successful quarter mean beyond quota?'),
    ('11111111-1111-4111-8111-111111111102', 'national_manager', 2, 'Should we optimize for growth, stability, or defense this quarter?'),
    ('11111111-1111-4111-8111-111111111103', 'national_manager', 3, 'Which motion is primary this period: net-new, expansion, renewal, or partner-led?'),
    ('11111111-1111-4111-8111-111111111104', 'national_manager', 4, 'What constraints are non-negotiable (margin floors, approvals, pricing authority)?'),
    ('11111111-1111-4111-8111-111111111105', 'national_manager', 5, 'When goals conflict, what wins first: revenue, margin, commit, or upside?')
ON CONFLICT (lvo_questionid) DO NOTHING;

-- Regional Manager
INSERT INTO lvo_som_interview_question (lvo_questionid, lvo_role, lvo_sortorder, lvo_questiontext)
VALUES
    ('22222222-2222-4222-8222-222222222201', 'regional_manager', 1, 'Which segments or geos are priority this quarter?'),
    ('22222222-2222-4222-8222-222222222202', 'regional_manager', 2, 'Which segments are temporarily deprioritized?'),
    ('22222222-2222-4222-8222-222222222203', 'regional_manager', 3, 'Are long-cycle enterprise deals acceptable in your region now?'),
    ('22222222-2222-4222-8222-222222222204', 'regional_manager', 4, 'What follow-up intensity is expected in this region?'),
    ('22222222-2222-4222-8222-222222222205', 'regional_manager', 5, 'What events should pause, reset, or restart timeline expectations here?')
ON CONFLICT (lvo_questionid) DO NOTHING;

-- Seller Manager
INSERT INTO lvo_som_interview_question (lvo_questionid, lvo_role, lvo_sortorder, lvo_questiontext)
VALUES
    ('33333333-3333-4333-8333-333333333301', 'seller_manager', 1, 'What behaviors define a strong seller in your team?'),
    ('33333333-3333-4333-8333-333333333302', 'seller_manager', 2, 'How early should multi-threading happen?'),
    ('33333333-3333-4333-8333-333333333303', 'seller_manager', 3, 'When should a deal be paused, recycled, or closed-no-deal?'),
    ('33333333-3333-4333-8333-333333333304', 'seller_manager', 4, 'What evidence is needed to keep a plateaued deal open?'),
    ('33333333-3333-4333-8333-333333333305', 'seller_manager', 5, 'How should coaching differ for fast vs strategic timeline classes?')
ON CONFLICT (lvo_questionid) DO NOTHING;

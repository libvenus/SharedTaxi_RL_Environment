-- ============================================================================
-- lvo_dealhealthconfig — Configurable thresholds for the deal-health calc
--
-- Single-row config table (id = 1). Stores every tunable mentioned in the
-- user story so a future Sales Operating Model admin UI can edit them
-- without code changes.
--
-- Values seeded here MUST match the defaults referenced by
-- app/services/deal_health.py and app/services/deal_risks.py.
-- ============================================================================

CREATE TABLE IF NOT EXISTS lvo_dealhealthconfig (
    id              SMALLINT PRIMARY KEY,
    lvo_settings    JSONB NOT NULL,
    lvo_updatedat   TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO lvo_dealhealthconfig (id, lvo_settings)
VALUES (
    1,
    '{
        "weights": {
            "stage_progress":     25,
            "activity_freshness": 25,
            "stakeholder":        20,
            "close_confidence":   20,
            "risk_adjustment":    10
        },
        "stage_position_score": {
            "Qualify":     20,
            "Develop":     40,
            "Propose":     60,
            "Execute":     80,
            "Closed Won": 100,
            "Closed Lost":  0
        },
        "stage_time_distribution": {
            "Qualify":  0.20,
            "Develop":  0.30,
            "Propose":  0.20,
            "Execute":  0.30
        },
        "close_confidence_distribution": {
            "Qualify":  0.25,
            "Develop":  0.25,
            "Propose":  0.30,
            "Execute":  0.20
        },
        "tempo_class_target_days": {
            "Fast":         30,
            "Quarterly":    90,
            "Programmatic": 365,
            "Strategic":    730
        },
        "tempo_class_cadence_days": {
            "Fast":          4,
            "Quarterly":    10,
            "Programmatic": 20,
            "Strategic":    40
        },
        "stakeholder_required_count": 5,
        "stakeholder_threading_factor": {
            "1":  0.60,
            "2":  0.80,
            "3+": 1.00
        },
        "low_stakeholder_threshold":   40,
        "low_activity_days_threshold": 14,
        "risk_penalty_per_risk":       20,
        "health_band_thresholds": {
            "GREEN":  75,
            "YELLOW": 50
        }
    }'::JSONB
)
ON CONFLICT (id) DO NOTHING;


-- Sanity check (optional — uncomment to run):
-- SELECT id, lvo_settings, lvo_updatedat FROM lvo_dealhealthconfig;

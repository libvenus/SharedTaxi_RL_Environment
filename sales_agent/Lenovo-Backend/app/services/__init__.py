"""Domain-service layer for the Deal Detailed View user story.

Pure-function modules that implement the business logic. Routers stay thin
and just orchestrate HTTP concerns; everything that has rules, formulas, or
branching lives here so it is unit-testable without hitting Postgres.

Modules
-------
- ``deal_health``  — five-component scoring functions (pure, no DB).
- ``deal_risks``   — risk-derivation rule evaluators (pure, no DB).
- ``deal_recalc``  — orchestrator that loads from the DB, runs the
  calculators, and (optionally) persists the result.

Importers should reference the modules directly, e.g.
``from app.services.deal_recalc import recalculate_deal_health`` —
this keeps import cost lazy and avoids circular imports.
"""

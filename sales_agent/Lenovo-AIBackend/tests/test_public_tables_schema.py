"""Schema smoke test: ensure all expected public tables exist.

This pins the minimum table surface the API depends on, so accidental model
import regressions or table-name drift are caught early in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import inspect

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.db.database import Base

# Import modules not loaded in tests/conftest.py so their tables are registered
# on Base.metadata for this schema test.
import app.models.email_template  # noqa: F401
import app.models.meeting_briefing  # noqa: F401
import app.models.outreach  # noqa: F401
import app.models.post_meet  # noqa: F401
import app.models.summaryDetails  # noqa: F401

# Expected tables in the default/public schema used by this backend.
EXPECTED_PUBLIC_TABLES = {
    "account",
    "opportunity",
    "tbl_crm_updates",
    "tbl_data_task",
    "tbl_email_templates",
    "tbl_meeting_briefing",
    "tbl_meeting_consent_email",
    "tbl_meeting_prep_note",
    "tbl_meeting_prep_task",
    "tbl_meeting_transcript",
    "tbl_meeting_transcript_segment",
    "tbl_outreach",
    "tbl_schedule_meetings",
    "tbl_summary_details",
    "tbl_to_do_list",
}


def test_all_expected_public_tables_exist(db_session):
    """Verify all expected public/default-schema tables are present."""
    Base.metadata.create_all(bind=db_session.bind)

    inspector = inspect(db_session.bind)
    actual_tables = set(inspector.get_table_names())

    missing_tables = EXPECTED_PUBLIC_TABLES - actual_tables
    assert not missing_tables, f"Missing tables: {sorted(missing_tables)}"

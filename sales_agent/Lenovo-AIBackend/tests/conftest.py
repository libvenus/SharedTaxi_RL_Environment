"""Shared pytest fixtures for the AIBackend test suite.

Sets up an in-memory SQLite DB so tests don't need a live Postgres,
plus a fresh FastAPI app with the meeting-details router so we can
exercise the HTTP contract end-to-end.

Notes
-----
* The model uses ``sqlalchemy.dialects.postgresql.UUID`` which SQLite
  doesn't understand natively. We register a per-dialect compiler so it
  emits ``VARCHAR(36)`` on SQLite (and is unchanged on Postgres).
* ``JSONB`` columns (e.g. ``tbl_schedule_meetings.attendees``) compile to
  SQLite ``JSON`` for the same reason.
* The CHECK constraint we added on ``bot_status`` lives only in the SQL
  migration (``sql/2026_06_us01_meeting_lifecycle.sql``), not in the ORM
  model — so SQLite-backed tests can't observe it. The Pydantic
  whitelist still rejects bad inputs at the request layer, which is what
  the smoke tests pin.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the repo root importable as ``app...`` regardless of where pytest
# is invoked from. Done before any ``app.*`` import.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Stub the env vars before app.db.database is imported. The DATABASE_URL
# string built there gets ignored once we override get_db, but the
# create_engine call needs *something* parseable — these dummies are
# never connected to.
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "test")
os.environ.setdefault("SMTP_HOST", "localhost")
os.environ.setdefault("SMTP_PORT", "25")
os.environ.setdefault("SMTP_USERNAME", "test")
os.environ.setdefault("SMTP_PASSWORD", "test")
os.environ.setdefault("SMTP_USE_TLS", "false")
os.environ.setdefault("SMTP_USE_SSL", "false")
os.environ.setdefault("EMAIL_FROM_ADDRESS", "sales-assistant@lenovo.com")
os.environ.setdefault("EMAIL_FROM_NAME", "Lenovo Sales Assistant")
os.environ.setdefault("EMAIL_API_KEY", "change-me-in-prod")

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB as PG_JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# ---------------------------------------------------------------------------
# Make Postgres UUID compile to VARCHAR on SQLite so create_all() works
# on the in-memory test DB. This is a no-op when the dialect is Postgres.
# ---------------------------------------------------------------------------
@compiles(PG_UUID, "sqlite")
def _compile_pg_uuid_for_sqlite(_element, _compiler, **_kw) -> str:  # noqa: D401
    return "VARCHAR(36)"


@compiles(PG_JSONB, "sqlite")
def _compile_pg_jsonb_for_sqlite(_element, _compiler, **_kw) -> str:  # noqa: D401
    return "JSON"


# Now we can safely import the app — the engine it builds will never be
# touched because we override ``get_db`` for every test.
from app.core.config import AIBACKEND_API_PREFIX  # noqa: E402
from app.db.database import Base, get_db  # noqa: E402
from app.api.meeting_details import router as meeting_router  # noqa: E402
from app.api.transcripts import router as transcripts_router  # noqa: E402
from app.api.consent_emails import (  # noqa: E402
    router as consent_emails_router,
    meeting_consent_status_router,
)
from app.api.data_tasks import router as data_tasks_router  # noqa: E402
from app.api.to_do_list import router as to_do_list_router  # noqa: E402

# Importing the model modules ensures SQLAlchemy registers the tables on
# Base.metadata before create_all() runs. New models go here.
import app.models.schedulemeeting  # noqa: E402, F401
import app.models.transcript  # noqa: E402, F401
import app.models.consent_email  # noqa: E402, F401
import app.models.data_task  # noqa: E402, F401
import app.models.to_do_list  # noqa: E402, F401
import app.models.crm  # noqa: E402, F401


@pytest.fixture(scope="session")
def test_engine():
    """Session-scoped in-memory SQLite engine + schema.

    NOTE: ``poolclass=StaticPool`` is critical. SQLite ``:memory:`` databases
    are per-connection — without StaticPool, the engine pool would hand out
    a fresh (empty!) connection to each session and our ``create_all`` work
    would be invisible. StaticPool forces every checkout to reuse the one
    connection on which we built the schema.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture()
def db_session(test_engine):
    """Function-scoped session — rolled back at the end so tests stay isolated."""
    Session = sessionmaker(
        bind=test_engine,
        autocommit=False,
        autoflush=False,
    )
    session = Session()
    try:
        yield session
    finally:
        # Wipe all rows so the next test starts clean. Cheaper than
        # rebuilding the schema each time, and keeps the session-scoped
        # engine fixture viable.
        session.rollback()
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
        session.close()


@pytest.fixture()
def client(db_session):
    """Fresh FastAPI app per test, wired to the SQLite session.

    We don't use ``app.main:app`` directly — that triggers
    ``Base.metadata.create_all(bind=engine)`` against the prod engine on
    import, which would fail without a live Postgres. A throwaway app
    with just the routers we need is cleaner and faster.
    """
    app = FastAPI()
    api_router = APIRouter(prefix=AIBACKEND_API_PREFIX)
    api_router.include_router(meeting_router)
    api_router.include_router(transcripts_router)
    api_router.include_router(consent_emails_router)
    api_router.include_router(meeting_consent_status_router)
    api_router.include_router(data_tasks_router)
    api_router.include_router(to_do_list_router)
    app.include_router(api_router)

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    return TestClient(app)

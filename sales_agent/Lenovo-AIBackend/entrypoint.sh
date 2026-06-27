#!/bin/sh
set -e

echo "Waiting for PostgreSQL at ${DB_HOST}:${DB_PORT}..."
until pg_isready -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}"; do
  sleep 2
done

echo "Creating base schema via SQLAlchemy..."
python -c "
from app.db.database import Base, engine
import app.models.schedulemeeting
import app.models.transcript
import app.models.consent_email
import app.models.data_task
import app.models.meeting_briefing
import app.models.email
import app.models.to_do_list
import app.models.crm
import app.models.summaryDetails
import app.models.outreach
import app.models.email_template
import app.models.post_meet
Base.metadata.create_all(bind=engine)
print('Schema ready.')
"

echo "Creating auxiliary tables..."
PGPASSWORD="${DB_PASSWORD}" psql \
  -h "${DB_HOST}" -p "${DB_PORT}" \
  -U "${DB_USER}" -d "${DB_NAME}" << 'SQL'
-- calendar_events: enriches meeting-prep with Zoom/Teams passcode info.
-- LEFT JOINed in meeting_prep.py so rows are optional; table must exist.
CREATE TABLE IF NOT EXISTS calendar_events (
    id            SERIAL PRIMARY KEY,
    join_url      TEXT,
    passcode      TEXT,
    join_meeting_id TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_calendar_events_join_url ON calendar_events (join_url);
SQL

echo "Running SQL migrations..."
for f in $(ls /app/sql/*.sql | sort); do
  echo "  Applying $f"
  PGPASSWORD="${DB_PASSWORD}" psql \
    -h "${DB_HOST}" \
    -p "${DB_PORT}" \
    -U "${DB_USER}" \
    -d "${DB_NAME}" \
    -f "$f"
done

echo "Starting Lenovo-AIBackend..."
exec uvicorn app.main:app --host 0.0.0.0 --port 9101

#!/bin/sh
set -e

echo "Waiting for PostgreSQL at ${DATABASE_HOST}:${DATABASE_PORT}..."
until pg_isready -h "${DATABASE_HOST}" -p "${DATABASE_PORT}" -U "${DATABASE_USERNAME}"; do
  sleep 2
done

echo "Creating base schema via SQLAlchemy..."
python -c "
from app.config import get_settings
from app.database import Base, engine
import app.models
Base.metadata.create_all(bind=engine)
print('Schema ready.')
"

echo "Patching schema for migration compatibility..."
PGPASSWORD="${DATABASE_PASSWORD}" psql \
  -h "${DATABASE_HOST}" -p "${DATABASE_PORT}" \
  -U "${DATABASE_USERNAME}" -d "${DATABASE_NAME}" << 'SQL'
-- 1. Boolean columns: add DB-level defaults so seeds that omit them still work.
ALTER TABLE opportunity ALTER COLUMN lvo_partnerinvolved SET DEFAULT false;

-- 2. lvo_settings must be jsonb (not json) so the ? operator works in migrations.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='lvo_dealhealthconfig' AND column_name='lvo_settings'
    AND data_type='json'
  ) THEN
    ALTER TABLE lvo_dealhealthconfig
      ALTER COLUMN lvo_settings TYPE jsonb USING lvo_settings::jsonb;
  END IF;
END $$;
SQL

echo "Running SQL migrations..."
for f in $(ls /app/sql/*.sql | sort); do
  echo "  Applying $f"
  PGPASSWORD="${DATABASE_PASSWORD}" psql \
    -h "${DATABASE_HOST}" \
    -p "${DATABASE_PORT}" \
    -U "${DATABASE_USERNAME}" \
    -d "${DATABASE_NAME}" \
    -f "$f"
done

echo "Post-seed backfills..."
PGPASSWORD="${DATABASE_PASSWORD}" psql \
  -h "${DATABASE_HOST}" -p "${DATABASE_PORT}" \
  -U "${DATABASE_USERNAME}" -d "${DATABASE_NAME}" << 'SQL'
-- account_view_schema.sql runs before seed data exists, so these backfills
-- must be re-applied after all seeds have inserted rows.
UPDATE account
   SET lvo_accounttype = CASE
       WHEN EXISTS (
            SELECT 1 FROM opportunity o
             WHERE UPPER(o.accountid) = UPPER(a.accountid::TEXT)
               AND (o.statecode IN ('Won','Closed Won') OR o.stagename IN ('Closed Won'))
       ) THEN 'Customer' ELSE 'Prospect'
   END
  FROM account a
 WHERE account.accountid = a.accountid AND account.lvo_accounttype IS NULL;

UPDATE account
   SET lvo_accountstatus = CASE
       WHEN COALESCE(statecode, 'Active') = 'Inactive' THEN 'Inactive'
       ELSE 'Active'
   END
 WHERE lvo_accountstatus IS NULL;
SQL

echo "Starting Lenovo-Backend..."
exec uvicorn app.main:app --host 0.0.0.0 --port 9100

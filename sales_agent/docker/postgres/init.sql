-- Creates the second database needed by Lenovo-AIBackend.
-- The first database (lenovosales) is already created by POSTGRES_DB env var.
SELECT 'CREATE DATABASE lenovo_aibackend'
  WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'lenovo_aibackend')
\gexec

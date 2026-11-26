-- Provision a dedicated least-privilege read-only PostgreSQL role for QueryPilot SQL execution.
-- Pair this role with READ_DATABASE_URL so executed queries never use the write/auth DB user.
-- Usage (run as PostgreSQL superuser / owner against database 'querypilot'):
--   psql -U querypilot -d querypilot -v reader_password="'ChangeMeStrongReaderPass!'" -f infra/init-readonly-role.sql

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'querypilot_reader') THEN
        CREATE ROLE querypilot_reader LOGIN PASSWORD 'change-me-readonly-pass' NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
    END IF;
END
$$;

REVOKE ALL ON SCHEMA public FROM querypilot_reader;
GRANT USAGE ON SCHEMA public TO querypilot_reader;

-- Grant SELECT on shared demo tables (customers, products, orders)
GRANT SELECT ON TABLE customers, products, orders TO querypilot_reader;

-- Ensure zero privileges on auth and Schema RAG internal tables
REVOKE ALL ON TABLE users, schema_embeddings FROM querypilot_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM querypilot_reader;

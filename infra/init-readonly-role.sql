-- Provision the least-privilege shared PostgreSQL login role for QueryPilot SQL execution.
-- Pair this role with READ_DATABASE_URL (required when DATABASE_URL is PostgreSQL) so
-- generated SQL never runs as the write/auth database owner.
--
-- Usage (run as PostgreSQL superuser / owner against database 'querypilot'):
--   psql -U querypilot -d querypilot -f infra/init-readonly-role.sql
--
-- IMPORTANT: Replace 'change-me-readonly-pass' below (or run ALTER ROLE querypilot_reader
-- WITH PASSWORD '<strong-password>' immediately after running this script) before production use.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'querypilot_reader') THEN
        CREATE ROLE querypilot_reader LOGIN PASSWORD 'change-me-readonly-pass' NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
    END IF;
END
$$;

REVOKE ALL ON SCHEMA public FROM querypilot_reader;
GRANT USAGE ON SCHEMA public TO querypilot_reader;

-- Grant SELECT on shared demo tables only. Uploaded tenant tables (u_{id}_*) are granted
-- exclusively to per-user qp_tenant_{id} NOLOGIN roles whose membership is granted to querypilot_reader.
DO $$
BEGIN
    IF to_regclass('public.customers') IS NOT NULL
       AND to_regclass('public.products') IS NOT NULL
       AND to_regclass('public.orders') IS NOT NULL THEN
        GRANT SELECT ON TABLE customers, products, orders TO querypilot_reader;
    END IF;

    IF to_regclass('public.users') IS NOT NULL THEN
        REVOKE ALL ON TABLE users FROM querypilot_reader;
    END IF;

    IF to_regclass('public.schema_embeddings') IS NOT NULL THEN
        REVOKE ALL ON TABLE schema_embeddings FROM querypilot_reader;
    END IF;
END
$$;

ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM querypilot_reader;

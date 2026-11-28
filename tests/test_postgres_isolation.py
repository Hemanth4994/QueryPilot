"""Real PostgreSQL database-layer tenant role isolation and XML-leak blocking tests."""

import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.app.config import settings
from backend.app.core import auth, database, sql_guard


def _postgres_env_urls() -> tuple[str, str] | None:
    """Return (write_url, read_url) when PostgreSQL + READ_DATABASE_URL are configured in env."""
    write_url = (os.environ.get("DATABASE_URL") or os.environ.get("DB_URL") or "").strip()
    read_url = (os.environ.get("READ_DATABASE_URL") or "").strip()
    if write_url.startswith("postgres") and read_url.startswith("postgres"):
        return write_url, read_url
    return None


@pytest.fixture()
def live_pg_engines():
    """Configure real PostgreSQL write/read engines when PG env vars are present."""
    urls = _postgres_env_urls()
    if urls is None:
        if os.environ.get("REQUIRE_POSTGRES_TESTS") == "1":
            pytest.fail(
                "REQUIRE_POSTGRES_TESTS=1 is set, but PostgreSQL DATABASE_URL and "
                "READ_DATABASE_URL are missing."
            )
        pytest.skip("PostgreSQL DATABASE_URL and READ_DATABASE_URL are not set in environment.")

    write_url, read_url = urls
    prev_db_url = settings.database_url
    prev_read_url = settings.read_database_url
    try:
        settings.database_url = write_url
        settings.read_database_url = read_url
        database.configure_database(write_url, read_database_url=read_url)
        database.check_and_seed_db()
        auth.ensure_users_table()
        database.sync_readonly_role_privileges()
        yield database.get_engine(), database.get_read_engine()
    finally:
        settings.database_url = prev_db_url
        settings.read_database_url = prev_read_url
        database.configure_database(prev_db_url, read_database_url=prev_read_url)


def test_postgres_refuses_startup_when_read_database_url_empty():
    """Postgres engine initialization must fail closed if READ_DATABASE_URL is empty."""
    with pytest.raises(RuntimeError, match="READ_DATABASE_URL is required"):
        database._create_pair(
            "postgresql+psycopg://querypilot:querypilot@localhost:5432/querypilot",
            read_database_url="",
        )


def test_postgres_per_tenant_roles_block_leaks_at_db_layer_and_allow_own_table(live_pg_engines):
    """Verify PostgreSQL enforces qp_tenant_{id} permissions even when SQL text guard is bypassed."""
    write_eng, read_eng = live_pg_engines
    suffix = uuid.uuid4().hex[:8]
    user1 = None
    user2 = None
    u1_table: str | None = None
    u2_table: str | None = None

    try:
        user1 = auth.register_user(f"pg_iso_u1_{suffix}@example.com", "StrongPass123!", target_engine=write_eng)
        user2 = auth.register_user(f"pg_iso_u2_{suffix}@example.com", "StrongPass123!", target_engine=write_eng)
        u1_table = f"u_{user1.id}_metrics_{suffix}"
        u2_table = f"u_{user2.id}_secrets_{suffix}"

        with write_eng.begin() as conn:
            conn.execute(
                text(f'CREATE TABLE "{u1_table}" (id INTEGER PRIMARY KEY, metric TEXT, revenue DOUBLE PRECISION);')
            )
            conn.execute(
                text(f"""INSERT INTO "{u1_table}" (id, metric, revenue) VALUES (1, 'q1_arr', 4200.5);""")
            )
            conn.execute(
                text(f'CREATE TABLE "{u2_table}" (id INTEGER PRIMARY KEY, secret_note TEXT);')
            )
            conn.execute(
                text(f"""INSERT INTO "{u2_table}" (id, secret_note) VALUES (1, 'tenant_2_confidential');""")
            )

        assert database.grant_readonly_select(u1_table, target_engine=write_eng, user_id=user1.id) is True
        assert database.grant_readonly_select(u2_table, target_engine=write_eng, user_id=user2.id) is True

        # User 1 CAN query their own uploaded table through bounded execution
        own_df, own_err, _ = sql_guard.execute_sql_bounded(
            f'SELECT metric, revenue FROM "{u1_table}" ORDER BY id;',
            user_id=user1.id,
            allowed_tables={u1_table, "customers", "products", "orders"},
        )
        assert own_err is None
        assert own_df is not None
        assert own_df.to_dict(orient="records") == [{"metric": "q1_arr", "revenue": 4200.5}]

        # Text guard must ALSO reject all three leaking XML queries and broad XML dumps upfront
        leaking_queries = [
            "SELECT table_to_xml('users', false, true, '');",
            "SELECT query_to_xml_and_xmlschema('SELECT id, email, hashed_password FROM users', false, true, '');",
            f"""SELECT query_to_xml('SELECT * FROM "{u2_table}"', false, true, '');""",
        ]
        schema_dump_queries = [
            "SELECT schema_to_xml('public', false, true, '');",
            "SELECT database_to_xml(false, true, '');",
        ]
        for query in [*leaking_queries, *schema_dump_queries]:
            guard_err = sql_guard.validate_read_only_sql(
                query,
                user_id=user1.id,
                allowed_tables={u1_table, "customers", "products", "orders"},
            )
            assert guard_err is not None and "Security Error" in guard_err

        # Even bypassing validate_read_only_sql completely, PostgreSQL itself must deny all 3 leaks
        # under SET LOCAL ROLE qp_tenant_{user1.id}, and schema/database dumps omit inaccessible tables.
        role1 = database.tenant_role_name(user1.id)
        for query in leaking_queries:
            with read_eng.connect() as conn:
                with pytest.raises(SQLAlchemyError) as exc_info:
                    with conn.begin():
                        conn.execute(text("SET LOCAL TRANSACTION READ ONLY"))
                        conn.execute(text(f"SET LOCAL ROLE {role1}"))
                        conn.execute(text(query))
                assert "permission denied" in str(exc_info.value).lower(), (
                    f"Expected PostgreSQL permission denied for {query!r}, got: {exc_info.value}"
                )

        for dump_query in schema_dump_queries:
            with read_eng.connect() as conn:
                with conn.begin():
                    conn.execute(text("SET LOCAL TRANSACTION READ ONLY"))
                    conn.execute(text(f"SET LOCAL ROLE {role1}"))
                    xml_output = str(conn.execute(text(dump_query)).scalar() or "")
            assert u1_table in xml_output and "q1_arr" in xml_output
            assert u2_table not in xml_output
            assert "tenant_2_confidential" not in xml_output
            assert f"pg_iso_u2_{suffix}" not in xml_output
            assert "<users>" not in xml_output and "<schema_embeddings>" not in xml_output
    finally:
        with write_eng.begin() as conn:
            if u1_table:
                conn.execute(text(f'DROP TABLE IF EXISTS "{u1_table}" CASCADE;'))
            if u2_table:
                conn.execute(text(f'DROP TABLE IF EXISTS "{u2_table}" CASCADE;'))
            user_ids = [u.id for u in (user1, user2) if u is not None]
            if user_ids:
                conn.execute(
                    text("DELETE FROM users WHERE id = ANY(:uids);"),
                    {"uids": user_ids},
                )
                for uid in user_ids:
                    rname = database.tenant_role_name(uid)
                    conn.execute(
                        text(
                            f"""
                            DO $$
                            BEGIN
                                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{rname}') THEN
                                    DROP OWNED BY {rname};
                                    DROP ROLE {rname};
                                END IF;
                            END
                            $$;
                            """
                        )
                    )

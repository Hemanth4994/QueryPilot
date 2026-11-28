"""Unit tests for PostgreSQL guardrails, cross-tenant SQL checks, pooling, and Schema RAG."""

from unittest.mock import MagicMock

from backend.app.config import settings
from backend.app.core import database, schema_rag, sql_guard


def test_postgres_sql_guard_rejects_unsafe_statements():
    unsafe_queries = [
        "COPY orders TO '/tmp/orders.csv' CSV HEADER;",
        "TRUNCATE TABLE orders;",
        "ALTER TABLE orders ADD COLUMN note TEXT;",
        "SET statement_timeout = 0;",
        "GRANT ALL PRIVILEGES ON TABLE orders TO public;",
        "SELECT * INTO new_orders FROM orders;",
        "DO $$ BEGIN PERFORM 1; END $$;",
        "SELECT * FROM customers; DROP TABLE orders;",
        "/* sneaky */ DROP TABLE customers;",
    ]
    for bad_sql in unsafe_queries:
        err = sql_guard.validate_read_only_sql(bad_sql)
        assert err is not None and "Security Error" in err, f"Expected block for: {bad_sql}"


def test_postgres_sql_guard_allows_semicolon_and_keywords_inside_literals():
    safe_queries = [
        "SELECT company_name FROM customers WHERE region = 'North; DROP TABLE orders';",
        'SELECT "order_id", revenue FROM orders ORDER BY revenue DESC LIMIT 10;',
        "WITH top_orders AS (SELECT customer_id, SUM(revenue) AS total FROM orders GROUP BY customer_id) SELECT * FROM top_orders;",
    ]
    for safe_sql in safe_queries:
        assert sql_guard.validate_read_only_sql(safe_sql) is None


def test_cross_tenant_sql_guard_blocks_other_users_and_internal_tables():
    allowed = {"customers", "products", "orders", "u_2_sales"}

    blocked_queries = [
        "SELECT * FROM u_1_sales;",
        'SELECT * FROM "u_1_sales";',
        "SELECT * FROM public.u_1_customers;",
        "SELECT * FROM customers JOIN u_1_orders ON customers.customer_id = u_1_orders.customer_id;",
        "SELECT * FROM customers, u_99_targets;",
        "SELECT * FROM users;",
        'SELECT * FROM "users";',
        "SELECT * FROM schema_embeddings;",
        "SELECT * FROM information_schema.tables;",
        "SELECT * FROM pg_catalog.pg_tables;",
        "SELECT pg_read_file('/etc/passwd');",
        "SELECT * FROM unauthorized_table;",
        # XML / cursor / database export functions capable of evaluating arbitrary SQL or table names
        "SELECT table_to_xml('users', false, true, '');",
        "SELECT query_to_xml_and_xmlschema('SELECT email, hashed_password FROM users', false, true, '');",
        "SELECT query_to_xml('SELECT * FROM u_1_sales', false, true, '');",
        "SELECT schema_to_xml('public', false, true, '');",
        "SELECT cursor_to_xml('cur', 10, false, true, '');",
        "SELECT database_to_xml(false, true, '');",
        "SELECT query_to_xmlschema('SELECT * FROM users', false, true, '');",
        "SELECT table_to_xmlschema('users', false, true, '');",
        "SELECT table_to_xml_and_xmlschema('users', false, true, '');",
        "SELECT schema_to_xmlschema('public', false, true, '');",
        "SELECT schema_to_xml_and_xmlschema('public', false, true, '');",
        "SELECT cursor_to_xmlschema('cur', 10, false, true, '');",
        "SELECT cursor_to_xml_and_xmlschema('cur', 10, false, true, '');",
        "SELECT database_to_xmlschema(false, true, '');",
        "SELECT database_to_xml_and_xmlschema(false, true, '');",
        "SELECT pg_catalog.table_to_xml('users', false, true, '');",
    ]
    for sql in blocked_queries:
        err = sql_guard.validate_read_only_sql(sql, user_id=2, allowed_tables=allowed)
        assert err is not None and "Security Error" in err, f"Expected tenant block for: {sql}"

    allowed_queries = [
        "SELECT * FROM u_2_sales;",
        "SELECT EXTRACT(YEAR FROM order_date) AS yr, SUM(revenue) FROM orders GROUP BY yr;",
        "WITH my_cte AS (SELECT * FROM u_2_sales) SELECT * FROM my_cte JOIN customers ON my_cte.id = customers.customer_id;",
        "SELECT * FROM customers WHERE company_name = 'u_1_sales';",
    ]
    for sql in allowed_queries:
        err = sql_guard.validate_read_only_sql(sql, user_id=2, allowed_tables=allowed)
        assert err is None, f"Unexpected block for valid query: {sql} -> {err}"


def test_schema_rag_top_k_selection_and_hash_deduplication(tmp_path, monkeypatch):
    prev_url = settings.database_url
    db_file = tmp_path / "rag_unit.sqlite3"
    settings.database_url = f"sqlite:///{db_file.as_posix()}"
    database.configure_database(settings.database_url)
    database.check_and_seed_db()

    vectors_by_key = {
        "customers": [0.95, 0.1, 0.0],
        "orders": [0.85, 0.2, 0.0],
        "products": [0.2, 0.95, 0.0],
    }
    embed_calls: list[str] = []

    def fake_embed(text_input: str, task_type: str = "RETRIEVAL_DOCUMENT", **_kwargs):
        embed_calls.append(task_type)
        lowered = text_input.lower()
        for key, vec in vectors_by_key.items():
            if f"create table {key}" in lowered or key in lowered:
                return vec
        return [0.0, 0.0, 1.0]

    monkeypatch.setattr("backend.app.core.schema_rag.generate_embedding", fake_embed)

    try:
        first_run = schema_rag.index_all_tables(force=False)
        assert set(first_run["indexed_tables"]) == {"customers", "orders", "products"}
        assert first_run["skipped_tables"] == []

        second_run = schema_rag.index_all_tables(force=False)
        assert second_run["indexed_tables"] == []
        assert set(second_run["skipped_tables"]) == {"customers", "orders", "products"}

        top_two = schema_rag.retrieve_relevant_tables(
            "customers",
            top_k=2,
            min_similarity=0.35,
        )
        assert [t.table_name for t in top_two] == ["customers", "orders"]

        no_match = schema_rag.retrieve_relevant_tables(
            "unrelated payroll telemetry",
            top_k=5,
            min_similarity=0.35,
        )
        assert no_match == []
    finally:
        settings.database_url = prev_url
        database.configure_database(prev_url)


def test_postgres_execution_applies_read_only_timeout_and_row_cap(monkeypatch):
    executed_sql: list[str] = []
    mock_result = MagicMock()
    mock_result.keys.return_value = ["id"]
    mock_result.fetchmany.return_value = [(1,), (2,), (3,)]

    mock_conn = MagicMock()

    def fake_execute(stmt, *_args, **_kwargs):
        sql_str = str(stmt)
        executed_sql.append(sql_str)
        if sql_str.startswith("SELECT"):
            return mock_result
        return MagicMock()

    mock_conn.execute.side_effect = fake_execute
    mock_conn.begin.return_value.__enter__ = MagicMock(return_value=None)
    mock_conn.begin.return_value.__exit__ = MagicMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.dialect.name = "postgresql"
    mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr(sql_guard, "get_read_engine", lambda: mock_engine)
    monkeypatch.setattr(settings, "max_query_rows", 2)
    monkeypatch.setattr(settings, "query_timeout_ms", 4321)

    df, error, truncated = sql_guard.execute_sql_bounded("SELECT id FROM orders;")

    assert error is None
    assert truncated is True
    assert len(df) == 2
    assert executed_sql[0] == "SET LOCAL TRANSACTION READ ONLY"
    assert executed_sql[1] == "SET LOCAL statement_timeout = 4321"
    assert executed_sql[2] == "SELECT id FROM orders"
    mock_result.fetchmany.assert_called_once_with(3)

    executed_sql.clear()
    df_tenant, err_tenant, _ = sql_guard.execute_sql_bounded("SELECT id FROM orders;", user_id=7)
    assert err_tenant is None
    assert len(df_tenant) == 2
    assert executed_sql[0] == "SET LOCAL TRANSACTION READ ONLY"
    assert executed_sql[1] == "SET LOCAL ROLE qp_tenant_7"
    assert executed_sql[2] == "SET LOCAL statement_timeout = 4321"
    assert executed_sql[3] == "SELECT id FROM orders"


def test_postgres_pooling_vector_extension_and_cosine_query(monkeypatch):
    import pytest

    captured_create_engine: list[dict] = []

    def fake_create_engine(url, **kwargs):
        captured_create_engine.append({"url": url, "kwargs": kwargs})
        fake_eng = MagicMock()
        fake_eng.dialect.name = "postgresql"
        return fake_eng

    monkeypatch.setattr(database, "create_engine", fake_create_engine)
    write_url = "postgresql+psycopg://querypilot:querypilot@db:5432/querypilot"
    read_url = "postgresql+psycopg://querypilot_reader:rpass@db:5432/querypilot"

    with pytest.raises(RuntimeError, match="READ_DATABASE_URL is required"):
        database._create_pair(write_url)

    write_eng, read_eng = database._create_pair(write_url, read_database_url=read_url)
    assert write_eng is not read_eng
    assert captured_create_engine[0]["kwargs"]["pool_pre_ping"] is True
    assert captured_create_engine[0]["kwargs"]["pool_size"] == settings.db_pool_size
    assert captured_create_engine[0]["kwargs"]["max_overflow"] == settings.db_max_overflow
    assert captured_create_engine[1]["url"] == read_url

    mock_conn = MagicMock()
    write_eng.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    write_eng.begin.return_value.__exit__ = MagicMock(return_value=False)
    database.ensure_vector_extension(write_eng)
    ext_sql = str(mock_conn.execute.call_args[0][0])
    assert "CREATE EXTENSION IF NOT EXISTS vector" in ext_sql

    fake_session = MagicMock()
    fake_session.scalar.return_value = 1
    fake_session.__enter__ = MagicMock(return_value=fake_session)
    fake_session.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(schema_rag, "Session", lambda _eng: fake_session)
    monkeypatch.setattr(schema_rag, "ensure_schema_embeddings_table", lambda _eng=None: None)
    monkeypatch.setattr(schema_rag, "generate_embedding", lambda *_a, **_kw: [1.0, 0.0])

    query_conn = MagicMock()
    query_conn.execute.return_value.mappings.return_value.all.return_value = [
        {"table_name": "orders", "ddl_text": "CREATE TABLE orders (...);", "similarity": 0.91},
        {"table_name": "products", "ddl_text": "CREATE TABLE products (...);", "similarity": 0.2},
    ]
    write_eng.connect.return_value.__enter__ = MagicMock(return_value=query_conn)
    write_eng.connect.return_value.__exit__ = MagicMock(return_value=False)

    matched = schema_rag.retrieve_relevant_tables(
        "Show orders",
        user_id=7,
        top_k=5,
        min_similarity=0.35,
        target_engine=write_eng,
    )
    assert len(matched) == 1
    assert matched[0].table_name == "orders"
    rag_sql = str(query_conn.execute.call_args[0][0])
    assert "<=>" in rag_sql
    assert "owner_id IS NULL OR owner_id = :user_id" in rag_sql


def test_dedicated_readonly_database_url_creates_separate_read_engine_and_grants(monkeypatch):
    created_urls: list[str] = []

    def fake_create_engine(url, **kwargs):
        _ = kwargs
        created_urls.append(url)
        fake_eng = MagicMock()
        fake_eng.dialect.name = "postgresql"
        return fake_eng

    monkeypatch.setattr(database, "create_engine", fake_create_engine)
    write_url = "postgresql+psycopg://querypilot:pass@db:5432/querypilot"
    read_url = "postgresql+psycopg://querypilot_reader:rpass@db:5432/querypilot"

    write_eng, read_eng = database._create_pair(write_url, read_database_url=read_url)
    assert write_eng is not read_eng
    assert created_urls == [write_url, read_url]

    role = database.get_readonly_role(read_database_url=read_url, write_database_url=write_url)
    assert role == "querypilot_reader"

    prev_read_url = settings.read_database_url
    prev_db_url = settings.database_url
    try:
        settings.database_url = write_url
        settings.read_database_url = read_url
        monkeypatch.setattr(database, "_configured_url", write_url)
        monkeypatch.setattr(database, "_configured_read_url", read_url)

        mock_conn = MagicMock()
        mock_conn.execute.return_value.scalar.return_value = 1
        write_eng.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        write_eng.begin.return_value.__exit__ = MagicMock(return_value=False)

        assert database.grant_readonly_select("u_1_targets", target_engine=write_eng) is True
        grant_sqls = [str(call.args[0]) for call in mock_conn.execute.call_args_list]
        assert any("CREATE ROLE qp_tenant_1 NOLOGIN" in s for s in grant_sqls)
        assert any("GRANT qp_tenant_1 TO \"querypilot_reader\"" in s for s in grant_sqls)
        assert any('GRANT SELECT ON TABLE "u_1_targets" TO qp_tenant_1' in s for s in grant_sqls)
        assert any('REVOKE ALL ON TABLE "u_1_targets" FROM "querypilot_reader"' in s for s in grant_sqls)
        assert not any('GRANT SELECT ON TABLE "u_1_targets" TO "querypilot_reader"' in s for s in grant_sqls)
        assert database.grant_readonly_select("users", target_engine=write_eng) is False
    finally:
        settings.read_database_url = prev_read_url
        settings.database_url = prev_db_url

import pytest
from database import check_and_seed_db, get_schema_for_llm, execute_sql

def setup_module(module):
    check_and_seed_db()

def test_schema_introspection_contains_core_tables():
    schema = get_schema_for_llm()
    assert "Table: customers" in schema
    assert "Table: products" in schema
    assert "Table: orders" in schema

def test_seeded_table_counts():
    df_cust, err1 = execute_sql("SELECT COUNT(*) AS cnt FROM customers;")
    df_prod, err2 = execute_sql("SELECT COUNT(*) AS cnt FROM products;")
    df_ord, err3 = execute_sql("SELECT COUNT(*) AS cnt FROM orders;")
    assert err1 is None and df_cust.iloc[0]["cnt"] == 100
    assert err2 is None and df_prod.iloc[0]["cnt"] == 20
    assert err3 is None and df_ord.iloc[0]["cnt"] == 500

def test_read_only_guard_blocks_destructive_sql():
    df, error = execute_sql("DROP TABLE orders;")
    assert df is None
    assert "Security Error" in error

def test_multi_statement_injection_blocked():
    df, error = execute_sql("SELECT * FROM customers; DROP TABLE customers;")
    assert df is None
    assert "Security Error" in error

def test_invalid_column_returns_clean_error():
    df, error = execute_sql("SELECT non_existent_col FROM orders;")
    assert df is None
    assert error is not None


def test_cte_wrapped_write_cannot_change_data():
    before, _ = execute_sql("SELECT COUNT(*) AS cnt FROM orders")
    for sql in ("WITH x AS (SELECT 1) DELETE FROM orders",
                "WITH x AS (SELECT 1) UPDATE orders SET quantity = 0"):
        df, error = execute_sql(sql)
        assert df is None and error is not None
    after, _ = execute_sql("SELECT COUNT(*) AS cnt FROM orders")
    assert after.iloc[0]["cnt"] == before.iloc[0]["cnt"] == 500

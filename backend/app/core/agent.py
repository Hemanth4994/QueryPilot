"""Text-to-SQL orchestration with Schema RAG, 2-stage self-correction, Plotly, and telemetry."""

from datetime import date, datetime
import json
import math
import time
from typing import Any
import uuid

import pandas as pd
import plotly.express as px

from ..config import settings
from ..schemas import ChartSpec, QueryResponse
from .database import get_allowed_tables_for_user, get_schema_for_llm
from .llm_client import generate_with_retry_telemetry
from .schema_rag import retrieve_relevant_tables
from .sql_guard import execute_sql_bounded
from .telemetry import QueryTelemetryEvent, emit_query_telemetry, trace_query_span

SQL_PROMPT_TEMPLATE = """
You are a Senior Data Engineer, SQL Architect, and Text-to-SQL specialist.

Your task is to convert the user's natural-language question into ONE
correct, executable PostgreSQL SELECT query using ONLY the database schema
provided below.

==================== DATABASE SCHEMA ====================
{schema}
==========================================================

==================== USER QUESTION ======================
{user_question}
==========================================================

STRICT INSTRUCTIONS:

1. Generate ONLY ONE SQL query.
2. Return ONLY the raw SQL query.
3. DO NOT return markdown.
4. DO NOT use ```sql or ``` formatting.
5. DO NOT provide explanations, comments, assumptions, or extra text.
6. The query MUST be valid PostgreSQL syntax.
7. The database schema is the ONLY source of truth.
8. NEVER invent:
   - tables
   - columns
   - relationships
   - values
   - business rules
9. Use ONLY tables and columns that actually exist in the schema.
10. Carefully determine the correct table(s) before generating the query.
11. When multiple tables are required, use the correct JOIN conditions
    based on the schema.
12. NEVER create a Cartesian/CROSS JOIN unless explicitly required.
13. Use explicit JOIN syntax.
14. Use meaningful table aliases when multiple tables are involved.
15. Qualify column names when necessary to avoid ambiguity.

==================== QUERY UNDERSTANDING ====================

Before generating the SQL, internally identify:
- What data the user is asking for.
- Which table contains that data.
- Which columns represent the requested information.
- Whether aggregation is required.
- Whether filtering is required.
- Whether grouping is required.
- Whether sorting is required.
- Whether the user is asking for a ranking or Top-N result.
- Whether DISTINCT is required.
- Whether date/time filtering is required.
- The appropriate level of granularity.

Do this reasoning internally.
DO NOT output the reasoning.

==================== AGGREGATION RULES ====================

Choose aggregation carefully.
Examples:
"How many customers?" → COUNT(...)
"How many unique customers?" → COUNT(DISTINCT customer_id)
"Total sales" → SUM(sales_column)
"Average salary" → AVG(salary)

==================== FILTERING RULES ====================

Use WHERE for filtering individual rows.
Use HAVING only when filtering aggregated results.
Use GROUP BY when the question requires results grouped by a dimension.

==================== SORTING AND RANKING ====================

If the user asks for highest, lowest, top, bottom, etc., use the appropriate ORDER BY and LIMIT logic.
Examples:
"Top 5 products by sales" → ORDER BY sales DESC LIMIT 5

==================== SAFETY ====================

This is an analytics system. Generate READ-ONLY queries only.
Allowed: SELECT, WITH ... SELECT
Do NOT generate: INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, REPLACE, COPY, SET.

==================== AMBIGUITY ====================

If the user's question cannot be answered using the provided schema,
DO NOT invent an answer. Return:
SELECT 'INSUFFICIENT_SCHEMA' AS error;

==================== FINAL OUTPUT ====================

Return ONLY the final SQL query.
No explanation. No markdown. No reasoning. No comments. No additional text.
"""


def strip_sql_fences(raw_sql: str) -> str:
    """Remove optional markdown code fences returned around SQL."""
    cleaned = raw_sql.strip()
    if cleaned.startswith("```sql"):
        cleaned = cleaned[6:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def get_sql_query_from_text_with_telemetry(
    user_question: str,
    schema_context: str | None = None,
) -> tuple[str, str, int, int]:
    """Generate PostgreSQL query and return (sql, model_used, input_tokens, output_tokens)."""
    schema = schema_context if schema_context is not None else get_schema_for_llm()
    prompt = SQL_PROMPT_TEMPLATE.format(schema=schema, user_question=user_question)
    raw_sql, model_used, in_tok, out_tok = generate_with_retry_telemetry(prompt)
    return strip_sql_fences(raw_sql), model_used, in_tok, out_tok


def get_sql_query_from_text_with_model(
    user_question: str,
    schema_context: str | None = None,
) -> tuple[str, str]:
    """Generate an initial PostgreSQL query from natural language and return (sql, model)."""
    sql_query, model_used, _, _ = get_sql_query_from_text_with_telemetry(
        user_question,
        schema_context=schema_context,
    )
    return sql_query, model_used


def get_sql_query_from_text(user_question: str, schema_context: str | None = None) -> str:
    """Generate an initial PostgreSQL query from natural language."""
    sql_query, _ = get_sql_query_from_text_with_model(user_question, schema_context=schema_context)
    return sql_query


def _to_json_value(value: Any) -> Any:
    """Convert pandas/numpy/datetime cell values into JSON-compliant primitives."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def serialize_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize DataFrame rows into JSON-safe dictionaries."""
    records = df.to_dict(orient="records")
    return [
        {str(key): _to_json_value(val) for key, val in row.items()}
        for row in records
    ]


def build_chart_payload(df: pd.DataFrame) -> tuple[ChartSpec | None, dict[str, Any] | None]:
    """Build Plotly bar chart metadata and figure JSON when result shape fits."""
    if len(df.columns) < 2 or df.empty:
        return None, None

    x_col = str(df.columns[0])
    y_col = str(df.columns[1])
    if not pd.api.types.is_numeric_dtype(df[df.columns[1]]):
        return None, None

    title = f"{y_col} by {x_col}"
    fig = px.bar(df, x=x_col, y=y_col, title=title, template="plotly_dark")
    spec = ChartSpec(
        chart_type="bar",
        x_col=x_col,
        y_col=y_col,
        title=title,
        template="plotly_dark",
    )
    return spec, json.loads(fig.to_json())


def _is_insufficient_schema(df: pd.DataFrame | None, sql_query: str) -> bool:
    """Detect the ambiguity sentinel returned by the LLM or executed query."""
    if sql_query.strip().upper() == "INSUFFICIENT_SCHEMA":
        return True
    if df is not None and not df.empty:
        if "error" in df.columns and str(df.iloc[0]["error"]).strip() == "INSUFFICIENT_SCHEMA":
            return True
        if df.shape == (1, 1) and str(df.iloc[0, 0]).strip() == "INSUFFICIENT_SCHEMA":
            return True
    return False


def process_user_question(question: str) -> tuple[pd.DataFrame | None, str, str | None]:
    """Run the 2-stage Text-to-SQL + self-correction pipeline returning (df, sql, error)."""
    sql_query, _ = get_sql_query_from_text_with_model(question)
    df, error, _ = execute_sql_bounded(sql_query)

    if error:
        schema = get_schema_for_llm()
        correction_prompt = f"""
        You wrote this SQL query: {sql_query}
        It failed in the database with this exact error: {error}

        Here is the Schema again:
        {schema}

        Fix the SQL query based on the error. RETURN ONLY THE RAW CORRECTED SQL.
        """
        raw_corrected, _, _, _ = generate_with_retry_telemetry(correction_prompt)
        sql_query = strip_sql_fences(raw_corrected)
        df, error, _ = execute_sql_bounded(sql_query)

    return df, sql_query, error


def _finalize_and_emit(
    response: QueryResponse,
    user_id: int | None,
) -> QueryResponse:
    """Emit structured telemetry for a freshly executed QueryResponse and return it."""
    emit_query_telemetry(
        QueryTelemetryEvent(
            request_id=response.request_id or uuid.uuid4().hex,
            user_id=user_id,
            model_used=response.model,
            fallback_triggered=response.fallback_triggered,
            self_correction_attempts=response.self_correction_attempts,
            cache_hit=response.cache_hit,
            retrieved_tables=response.retrieved_tables,
            sql_guard_status=response.sql_guard_status,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
    )
    return response


def run_query_pipeline(
    question: str,
    user_id: int | None = None,
    request_id: str | None = None,
    emit_telemetry: bool = True,
) -> QueryResponse:
    """Execute tenant-scoped Schema RAG, 2-stage SQL repair, Plotly output, and telemetry."""
    req_id = request_id or uuid.uuid4().hex
    start = time.perf_counter()

    with trace_query_span("querypilot.run_query_pipeline", {"request_id": req_id, "user_id": user_id}):
        retrieved = retrieve_relevant_tables(question, user_id=user_id)
        retrieved_tables = [item.table_name for item in retrieved]

        if not retrieved:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            resp = QueryResponse(
                request_id=req_id,
                question=question,
                sql_query="",
                status="INSUFFICIENT_SCHEMA",
                rows=[],
                columns=[],
                row_count=0,
                retrieved_tables=[],
                message="The question cannot be answered using the available database schema.",
                model=None,
                latency_ms=latency_ms,
                corrected=False,
                self_correction_attempts=0,
                fallback_triggered=False,
                sql_guard_status="short_circuited",
                input_tokens=0,
                output_tokens=0,
            )
            return _finalize_and_emit(resp, user_id) if emit_telemetry else resp

        schema_context = "\n\n".join(item.ddl_text for item in retrieved)
        allowed_tables = get_allowed_tables_for_user(user_id=user_id)
        corrected = False
        self_correction_attempts = 0
        fallback_triggered = False

        sql_query, model_used, in_tok, out_tok = get_sql_query_from_text_with_telemetry(
            question,
            schema_context=schema_context,
        )
        if model_used == settings.fallback_model:
            fallback_triggered = True

        df, error, truncated = execute_sql_bounded(
            sql_query,
            user_id=user_id,
            allowed_tables=allowed_tables,
        )

        if error:
            corrected = True
            self_correction_attempts = 1
            correction_prompt = f"""
            You wrote this PostgreSQL query: {sql_query}
            It failed in the database with this exact error: {error}

            Here is the Schema again:
            {schema_context}

            Fix the SQL query based on the error. RETURN ONLY THE RAW CORRECTED SQL.
            """
            raw_corrected, model_used, corr_in, corr_out = generate_with_retry_telemetry(
                correction_prompt
            )
            in_tok += corr_in
            out_tok += corr_out
            if model_used == settings.fallback_model:
                fallback_triggered = True

            sql_query = strip_sql_fences(raw_corrected)
            df, error, truncated = execute_sql_bounded(
                sql_query,
                user_id=user_id,
                allowed_tables=allowed_tables,
            )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        if error:
            guard_status = "blocked" if error.startswith("Security Error") else "error"
            resp = QueryResponse(
                request_id=req_id,
                question=question,
                sql_query=sql_query,
                status="ERROR",
                retrieved_tables=retrieved_tables,
                error=error,
                model=model_used,
                latency_ms=latency_ms,
                corrected=corrected,
                self_correction_attempts=self_correction_attempts,
                fallback_triggered=fallback_triggered,
                sql_guard_status=guard_status,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
            return _finalize_and_emit(resp, user_id) if emit_telemetry else resp

        if _is_insufficient_schema(df, sql_query):
            resp = QueryResponse(
                request_id=req_id,
                question=question,
                sql_query=sql_query,
                status="INSUFFICIENT_SCHEMA",
                rows=[],
                columns=[str(col) for col in df.columns] if df is not None else [],
                row_count=0,
                retrieved_tables=retrieved_tables,
                message="The question cannot be answered using the available database schema.",
                model=model_used,
                latency_ms=latency_ms,
                corrected=corrected,
                self_correction_attempts=self_correction_attempts,
                fallback_triggered=fallback_triggered,
                sql_guard_status="passed",
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
            return _finalize_and_emit(resp, user_id) if emit_telemetry else resp

        if df is None or df.empty:
            resp = QueryResponse(
                request_id=req_id,
                question=question,
                sql_query=sql_query,
                status="NO_DATA",
                rows=[],
                columns=[str(col) for col in df.columns] if df is not None else [],
                row_count=0,
                retrieved_tables=retrieved_tables,
                message="The query executed successfully but returned no data.",
                model=model_used,
                latency_ms=latency_ms,
                corrected=corrected,
                self_correction_attempts=self_correction_attempts,
                fallback_triggered=fallback_triggered,
                sql_guard_status="passed",
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
            return _finalize_and_emit(resp, user_id) if emit_telemetry else resp

        rows = serialize_dataframe(df)
        columns = [str(col) for col in df.columns]
        chart_spec, plotly_json = build_chart_payload(df)

        resp = QueryResponse(
            request_id=req_id,
            question=question,
            sql_query=sql_query,
            status="SUCCESS",
            rows=rows,
            columns=columns,
            row_count=len(rows),
            retrieved_tables=retrieved_tables,
            truncated=truncated,
            chart_spec=chart_spec,
            plotly_json=plotly_json,
            model=model_used,
            latency_ms=latency_ms,
            corrected=corrected,
            self_correction_attempts=self_correction_attempts,
            fallback_triggered=fallback_triggered,
            sql_guard_status="passed",
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
        return _finalize_and_emit(resp, user_id) if emit_telemetry else resp

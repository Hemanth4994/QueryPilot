# QueryPilot — AI-Powered Text-to-SQL Analytics Engine

QueryPilot is an interactive Business Intelligence agent built with Python, Streamlit, SQLAlchemy, SQLite, and the Google GenAI SDK (`gemini-3.6-flash` / `gemini-3.5-flash`). It translates plain-English questions into executable SQLite queries, enforces read-only execution, automatically repairs failed queries using database error tracebacks, and renders interactive Plotly visualizations.

## Architecture & Key Capabilities

- **Dynamic Schema Introspection:** Uses SQLAlchemy `inspect()` to read live table schemas and column data types at runtime—supporting both the seeded 3-table relational store (`customers`, `products`, `orders`) and user-uploaded CSV tables.
- **Read-Only Execution Guard:** Enforces `SELECT` / `WITH` read-only execution and single-statement validation in Python prior to database execution, while protecting core tables from CSV upload collisions.
- **Self-Correcting SQL Loop:** Captures SQLite execution exceptions and re-prompts the LLM once with the live schema and exact database error message to automatically repair invalid SQL.
- **High-Availability API Routing:** Handles `503 UNAVAILABLE` model spikes using exponential backoff (`1s`, `2s`, `4s`) before automatically failing over from `gemini-3.6-flash` to `gemini-3.5-flash`.
- **Deterministic Ambiguity Handling:** Returns a structured `INSUFFICIENT_SCHEMA` response when a question cannot be answered from the available schema instead of hallucinating columns.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Running Tests

```bash
pip install pytest
pytest -v
```
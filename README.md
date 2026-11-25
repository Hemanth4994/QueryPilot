# QueryPilot — AI-Powered Text-to-SQL Analytics Engine

QueryPilot is an interactive Business Intelligence agent decoupled into a **FastAPI** backend, a thin **Streamlit** frontend client, **SQLite** via SQLAlchemy, **Plotly** visualizations, and the **Google GenAI SDK** (`gemini-3.6-flash` primary with `gemini-3.5-flash` fallback).

## Architecture (Phase 1)

- **Backend (`backend/`)**
  - `app/main.py`: FastAPI entrypoint with CORS, lifespan DB seed validation, and structured JSON error handling.
  - `app/config.py`: Pydantic Settings (`GEMINI_API_KEY`, `DATABASE_URL`, `API_HOST`, `PORT`, `CORS_ORIGINS`).
  - `app/core/`:
    - `database.py`: SQLAlchemy write engine and read-only SQLite engine (`PRAGMA query_only = ON`), seeding `customers` (100), `products` (20), and `orders` (500) on startup, plus dynamic schema introspection.
    - `sql_guard.py`: Deterministic single-statement `SELECT`/`WITH` guard before execution on the read-only engine.
    - `llm_client.py`: Google GenAI wrapper with 3-attempt exponential backoff (`1s`, `2s`) for HTTP `503` / `UNAVAILABLE` and failover from `gemini-3.6-flash` to `gemini-3.5-flash`.
    - `agent.py`: Text-to-SQL prompt, 2-stage LLM self-correction loop, `INSUFFICIENT_SCHEMA` detection, and Plotly Express bar chart JSON generation.
    - `ingestion.py`: CSV upload ingestion with protected table-name prefixing (`customers`, `products`, `orders` → `user_<name>`).
  - `app/api/routes.py`: Versioned `/api/v1` endpoints:
    - `GET /api/v1/health`
    - `GET /api/v1/schema`
    - `POST /api/v1/query`
    - `POST /api/v1/upload`
- **Frontend (`frontend/`)**
  - `app.py`: Thin Streamlit client communicating with FastAPI over `httpx` (`BACKEND_URL`). Zero direct SQLAlchemy or Gemini SDK imports.
- **Tests (`tests/`)**
  - `test_querypilot.py`: Core schema, seeded row counts, and read-only SQL guard tests.
  - `test_api.py`: Offline FastAPI `TestClient` integration tests with mocked Gemini responses.

## Running Locally

1. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
2. Activate your virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI backend:
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. In a second terminal, start the Streamlit frontend:
   ```bash
   streamlit run frontend/app.py
   ```

## Running with Docker Compose

```bash
docker compose up --build
```

- Backend API: `http://localhost:8000/api/v1/health`
- Streamlit UI: `http://localhost:8501`

## Running Tests

```bash
pytest -v
```

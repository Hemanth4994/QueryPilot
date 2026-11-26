# QueryPilot — Multi-Tenant Text-to-SQL Analytics Engine

QueryPilot is a multi-tenant Text-to-SQL analytics application built with **FastAPI**, **PostgreSQL 16 + `pgvector`** (Schema RAG), **Redis** (caching, rate limiting, and job state), an **ARQ** background worker queue, a thin **Streamlit** frontend, **Plotly** visualizations, and the **Google GenAI SDK**.

## System Architecture

```mermaid
flowchart LR
    User(["User Browser"]) --> UI["Streamlit Frontend\nfrontend/app.py"]
    UI -->|"httpx + Bearer JWT"| API["FastAPI Backend\nbackend/app/main.py"]

    API --> Auth["JWT Auth & bcrypt\ncore/auth.py"]
    API --> Cache["Redis Cache & Rate Limiter\ncore/cache.py"]
    API -->|"202 Accepted"| Queue["ARQ Worker\napp/worker.py"]
    Queue --> Cache

    API --> RAG["Schema RAG\ncore/schema_rag.py"]
    Queue --> RAG
    RAG -->|"Cosine Search\nowner_id IS NULL OR owner_id = user"| PG[("PostgreSQL 16 + pgvector\ncustomers, products, orders,\nu_{id}_*, schema_embeddings, users")]

    RAG -->|"0 tables >= threshold\nINSUFFICIENT_SCHEMA"| API
    RAG -->|"Top-K DDLs"| LLM["GenAI Client\ngemini-3.6-flash to 3.5-flash\n2-Stage SQL Self-Correction"]
    LLM --> Guard["Read-Only & Tenant SQL Guard\nSELECT/WITH + u_{id}_* Check\nSET LOCAL READ ONLY + Timeout + Row Cap"]
    Guard --> PG

    API --> Telemetry["Structured JSON Telemetry\n/api/v1/metrics & Optional OTLP"]
    Queue --> Telemetry
```

## Components

| Layer | Component | Purpose |
|---|---|---|
| Frontend | `frontend/app.py` | Thin Streamlit client with JWT login/register, `httpx` API calls, `Cache Hit` badge, `Schema RAG & Query Details`, and async job polling. |
| API | `backend/app/api/routes.py` | Versioned `/api/v1` REST API (`/health`, `/metrics`, `/auth/*`, `/schema`, `/schema/reindex`, `/query`, `/query/async`, `/upload`, `/jobs/{job_id}`). |
| Auth & Isolation | `backend/app/core/auth.py`, `sql_guard.py` | Bcrypt password hashing (`pwdlib`), HS256 Bearer JWTs, `u_{user_id}_*` tenant table prefixing, and cross-tenant SQL guard + `SET LOCAL TRANSACTION READ ONLY` + `statement_timeout` + `MAX_QUERY_ROWS`. |
| Schema RAG | `backend/app/core/schema_rag.py` | Indexes table DDLs into `schema_embeddings` (`pgvector`) with SHA-256 deduplication and retrieves top-k tenant-visible tables above `SCHEMA_RAG_MIN_SIMILARITY` (short-circuiting `INSUFFICIENT_SCHEMA` before SQL LLM calls). |
| LLM & Repair | `backend/app/core/llm_client.py`, `agent.py` | `.env`-driven `PRIMARY_MODEL` / `FALLBACK_MODEL` routing with 3-attempt exponential backoff on HTTP `503` and 2-stage SQL self-correction. |
| Cache & Queue | `backend/app/core/cache.py`, `worker.py` | Redis schema/query caching with invalidation on upload/reindex, 60s sliding-window rate limiter (`429`), graceful Redis fallback, and ARQ background workers. |
| Observability | `backend/app/core/telemetry.py` | Single-line JSON query telemetry logs, authenticated `GET /api/v1/metrics` counters, and optional OpenTelemetry OTLP export (`ENABLE_TRACING=true`). |
| Evaluation & CI | `evals/`, `.github/workflows/ci.yml` | 30-query golden dataset (`--mode offline` and `--mode live` normalized result-set comparison) and automated lint/test/eval/Docker CI. |

## Environment Variables (`.env`)

Copy `.env.example` to `.env` before running:

| Variable | Default / Example | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required for live LLM/RAG)* | Google GenAI API key |
| `JWT_SECRET_KEY` | *(required for auth; no code default)* | Secret used to sign/verify HS256 Bearer JWTs |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `JWT_EXPIRE_MINUTES` | `60` | Access token lifetime in minutes |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `querypilot` | PostgreSQL container credentials |
| `DATABASE_URL` | `postgresql+psycopg://querypilot:querypilot@localhost:5432/querypilot` | SQLAlchemy PostgreSQL connection URL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for caching, rate limiting, and ARQ |
| `CACHE_TTL_SECONDS` | `300` | Schema and query cache TTL |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max query/upload requests per user per minute |
| `ASYNC_UPLOAD_THRESHOLD_BYTES` | `524288` | File size threshold triggering async worker upload |
| `PRIMARY_MODEL` | `gemini-3.6-flash` | Primary SQL generation model |
| `FALLBACK_MODEL` | `gemini-3.5-flash` | Fallback model after 503 retries |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model for Schema RAG |
| `SCHEMA_RAG_TOP_K` | `5` | Max tables retrieved per query |
| `SCHEMA_RAG_MIN_SIMILARITY` | `0.35` | Minimum cosine similarity threshold |
| `QUERY_TIMEOUT_MS` | `10000` | PostgreSQL `SET LOCAL statement_timeout` |
| `MAX_QUERY_ROWS` | `1000` | Row cap enforced on executed SQL queries |
| `ENABLE_TRACING` | `false` | Enable OpenTelemetry OTLP tracing |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `""` | Optional OTLP HTTP trace collector endpoint |
| `OTEL_SERVICE_NAME` | `querypilot-backend` | OpenTelemetry service name |
| `BACKEND_URL` | `http://localhost:8000` | FastAPI URL used by Streamlit frontend |

## Docker Compose Quickstart

```bash
cp .env.example .env
# Populate GEMINI_API_KEY and a strong random JWT_SECRET_KEY in .env
docker compose up --build
```

- **Streamlit UI:** `http://localhost:8501`
- **Backend Health:** `http://localhost:8000/api/v1/health`

## Local Development, Tests & Evaluations

Activate your virtual environment first:

- **macOS / Linux:**
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv; .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

Then run lint, tests, and evaluations from the activated environment:

```bash
# Lint with Ruff
ruff check .

# Unit & integration test suite (offline, mocked GenAI + FakeRedis)
pytest -v

# 30-query golden dataset validation (zero API cost; run in CI)
python evals/run_eval.py --mode offline
```

### Live Model Evaluation (`--mode live`)

CI runs `--mode offline` only (validating the 30 golden SQL queries, SQL guard rejections, and `INSUFFICIENT_SCHEMA` threshold behavior without calling Gemini). To measure live end-to-end **Execution Accuracy**, self-correction rate, and latency against PostgreSQL + `pgvector` and your `GEMINI_API_KEY`:

1. Start PostgreSQL (`pgvector/pgvector:pg16`):
   ```bash
   docker compose up -d db
   ```
2. Run the live harness (writes timestamped JSON and Markdown summaries to `evals/reports/`, which is gitignored):
   - **macOS / Linux:**
     ```bash
     DATABASE_URL="postgresql+psycopg://querypilot:querypilot@localhost:5432/querypilot" \
       python evals/run_eval.py --mode live
     ```
   - **Windows (PowerShell):**
     ```powershell
     $env:DATABASE_URL="postgresql+psycopg://querypilot:querypilot@localhost:5432/querypilot"
     python evals/run_eval.py --mode live
     ```

### Evaluation Results

| Mode | Scope & Stack | Verified Result | Latency & Notes |
|---|---|---|---|
| `--mode offline` | 30 golden cases (`8` single-table, `10` multi-table `JOIN`/`HAVING`, `5` CTE/window/time-series, `4` `INSUFFICIENT_SCHEMA`, `3` adversarial SQL blocks) | **30 / 30 passed (100%)** | Runs in CI with zero LLM calls; verifies SQL validity, guard rejections, and similarity threshold short-circuiting. |
| `--mode live` *(2026-11-26)* | End-to-end Text-to-SQL on **PostgreSQL 16 + `pgvector`** (`gemini-embedding-001`, `gemini-3-flash-preview` with automatic fallback to `gemini-3.5-flash-lite` on free-tier quota) | **29 / 30 passed (96.67%)** | `single_table`: 8/8, `multi_table_join`: 9/10 (`mt_12` projected 1 col vs 2), `cte_window_timeseries`: 5/5, `insufficient_schema`: 4/4, `adversarial`: 3/3. **p50:** `2,404 ms`, **p95:** `5,480 ms` (with `--delay-seconds 4.5` free-tier pacing; `29` fallback routings, `28,048` input / `1,345` output tokens). |

# QueryPilot — AI-Powered Text-to-SQL Analytics Engine

QueryPilot is an enterprise-ready, multi-tenant Business Intelligence agent built with **FastAPI**, **PostgreSQL 16 + `pgvector`** (Schema RAG), **Redis** (caching & sliding-window rate limiting), an **ARQ** background worker queue, a thin **Streamlit** frontend, **Plotly** visualizations, **Google GenAI**, structured JSON telemetry, a 30-query golden evaluation harness, and GitHub Actions CI.

## Architecture Overview

| Layer | Component | Purpose |
|---|---|---|
| Frontend | `frontend/app.py` | Thin Streamlit client with JWT login/register, `httpx` API integration, `Cache Hit` badge, `Schema RAG & Query Details`, and async job polling. |
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

## Running Lint, Tests & Evaluations Locally

```bash
# Lint with Ruff
.\venv\Scripts\ruff.exe check .

# Full unit & integration test suite (offline, mocked GenAI + FakeRedis)
.\venv\Scripts\pytest.exe -v

# 30-query golden dataset evaluation (zero API cost)
.\venv\Scripts\python.exe evals\run_eval.py --mode offline

# Live end-to-end execution accuracy evaluation (writes report to evals/reports/)
.\venv\Scripts\python.exe evals\run_eval.py --mode live
```

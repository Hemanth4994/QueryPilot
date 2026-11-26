"""Thin Streamlit frontend client for QueryPilot with JWT auth and async job polling."""

import hashlib
import json
import os
import time

import httpx
import pandas as pd
import plotly.io as pio
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="AI-BI Analytics Agent", page_icon="📊", layout="wide")
st.title("🤖 AI-Powered Business Intelligence")
st.markdown("Ask questions in plain English. The AI will write the SQL, run it, and visualize the results.")


def _auth_headers() -> dict[str, str]:
    """Build Authorization header from the active session JWT."""
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _clear_auth_session(notice: str | None = None) -> None:
    """Clear auth credentials and tenant-scoped session caches on logout or 401."""
    for key in (
        "access_token",
        "user_email",
        "messages",
        "last_upload_signature",
        "last_upload_result",
    ):
        st.session_state.pop(key, None)
    if notice:
        st.session_state["auth_notice"] = notice


def _extract_error_message(response: httpx.Response, fallback: str) -> str:
    """Extract a readable string message from FastAPI JSON or 422 validation lists."""
    try:
        payload = response.json()
    except ValueError:
        text_body = response.text.strip()
        return text_body or fallback

    if isinstance(payload, dict):
        detail = payload.get("detail", payload.get("message"))
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        if isinstance(detail, list):
            parts: list[str] = []
            for item in detail:
                if isinstance(item, dict):
                    loc = item.get("loc") or []
                    field = str(loc[-1]) if loc else "input"
                    msg = str(item.get("msg") or "Invalid value").removeprefix("Value error, ")
                    parts.append(f"{field}: {msg}" if field != "body" else msg)
                else:
                    parts.append(str(item))
            if parts:
                return "; ".join(parts)
    return fallback


def _handle_auth_status(response: httpx.Response) -> bool:
    """Return True if request failed with 401, clear state, and redirect to sign-in."""
    if response.status_code == 401:
        _clear_auth_session("Your session expired or is invalid. Please sign in again.")
        st.rerun()
        return True
    return False


def _fetch_backend_health() -> dict | None:
    """Probe backend health endpoint without requiring authentication."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/api/v1/health", timeout=3.0)
        if resp.status_code == 200:
            return resp.json()
    except (httpx.HTTPError, ValueError):
        return None
    return None


def _validate_auth_inputs(
    email: str,
    password: str,
    confirm_password: str | None = None,
) -> tuple[str, str | None]:
    """Validate email/password inputs locally before calling auth endpoints."""
    cleaned_email = email.strip().lower()
    if "@" not in cleaned_email or "." not in cleaned_email.split("@", 1)[-1]:
        return cleaned_email, "Please enter a valid email address (for example, name@company.com)."
    if not password:
        return cleaned_email, "Please enter your password."
    if confirm_password is not None:
        if len(password) < 8:
            return cleaned_email, "Password must be at least 8 characters long."
        if password != confirm_password:
            return cleaned_email, "Passwords do not match."
    return cleaned_email, None


def _submit_auth(
    mode: str,
    email: str,
    password: str,
    confirm_password: str | None = None,
) -> str | None:
    """Validate inputs, authenticate against /api/v1/auth/{mode}, and populate session."""
    cleaned_email, validation_error = _validate_auth_inputs(email, password, confirm_password)
    if validation_error:
        return validation_error

    try:
        resp = httpx.post(
            f"{BACKEND_URL}/api/v1/auth/{mode}",
            json={"email": cleaned_email, "password": password},
            timeout=15.0,
        )
    except httpx.HTTPError as exc:
        return f"Unable to reach authentication service at {BACKEND_URL}: {exc}"

    if resp.status_code >= 400:
        fallback = "Sign in failed." if mode == "login" else "Account creation failed."
        detail = _extract_error_message(resp, fallback)
        if mode == "login" and resp.status_code == 401:
            return f"{detail} If this is your first time here, open the Create Account tab first."
        if mode == "register" and resp.status_code == 409:
            return f"{detail} Switch to the Sign In tab to log in."
        return detail

    auth_data = resp.json()
    _clear_auth_session()
    st.session_state["access_token"] = auth_data["access_token"]
    st.session_state["user_email"] = auth_data["user"]["email"]
    return None


def fetch_schema() -> dict | None:
    """Fetch current user-scoped database schema metadata from FastAPI."""
    try:
        response = httpx.get(
            f"{BACKEND_URL}/api/v1/schema",
            headers=_auth_headers(),
            timeout=10.0,
        )
        if _handle_auth_status(response):
            return None
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def _parse_json_or_raise(response: httpx.Response, context: str) -> dict:
    """Safely decode a JSON object response or raise RuntimeError with helpful detail."""
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{context}: backend returned non-JSON response ({response.status_code}).") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{context}: unexpected payload format from backend.")
    return data


def poll_job_until_complete(job_id: str, timeout_seconds: float = 120.0) -> dict | None:
    """Poll GET /api/v1/jobs/{job_id} until the ARQ worker finishes or times out."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        resp = httpx.get(
            f"{BACKEND_URL}/api/v1/jobs/{job_id}",
            headers=_auth_headers(),
            timeout=10.0,
        )
        if _handle_auth_status(resp):
            return None
        if resp.status_code >= 400:
            raise RuntimeError(_extract_error_message(resp, f"Job polling failed ({resp.status_code})."))
        job = _parse_json_or_raise(resp, "Job polling")
        if job.get("status") == "completed":
            return job.get("result")
        if job.get("status") == "failed":
            raise RuntimeError(job.get("error") or "Background job failed.")
        time.sleep(0.5)
    raise TimeoutError("Background job timed out while waiting for completion.")


def _render_query_details(
    sql_query: str,
    retrieved_tables: list[str] | None,
    caption: str,
    truncated: bool = False,
    cache_hit: bool = False,
) -> None:
    """Render SQL, Schema RAG retrieved tables, cache badge, and telemetry."""
    if cache_hit:
        st.caption("⚡ **Cache Hit** · served from Redis")
    with st.expander("🔍 Schema RAG & Query Details"):
        tables_label = ", ".join(retrieved_tables) if retrieved_tables else "None matched threshold"
        st.markdown(f"**Schema RAG Retrieved Tables:** `{tables_label}`")
        if sql_query:
            st.code(sql_query, language="sql")
        if truncated:
            st.info("Result set reached the configured row limit and was truncated.")
        st.caption(caption)


# Main Authentication Page
if not st.session_state.get("access_token"):
    health = _fetch_backend_health()
    with st.sidebar:
        st.header("QueryPilot Workspace")
        st.caption(
            "Multi-tenant Text-to-SQL BI agent with pgvector Schema RAG, Redis cache, and ARQ workers."
        )
        if health and health.get("database_ready"):
            redis_label = "Redis ready" if health.get("redis_ready") else "Redis bypassed"
            st.success(f"API connected · {health.get('tables_count', 3)} shared tables · {redis_label}")
        else:
            st.warning(f"Backend status check unreachable at `{BACKEND_URL}`.")

    _, auth_col, _ = st.columns([1, 1.8, 1])
    with auth_col:
        with st.container(border=True):
            st.subheader("🔐 Sign in to QueryPilot")
            st.caption(
                "Sign in to access the shared B2B demo dataset (`customers`, `products`, `orders`) "
                "and your private CSV workspaces."
            )

            auth_notice = st.session_state.pop("auth_notice", None)
            if auth_notice:
                st.warning(auth_notice)

            signin_tab, register_tab = st.tabs(["Sign In", "Create Account"])

            with signin_tab:
                st.caption("Existing user? Enter your email and password below.")
                with st.form("login_form"):
                    login_email = st.text_input(
                        "Email",
                        placeholder="you@company.com",
                        key="signin_email",
                    )
                    login_password = st.text_input(
                        "Password",
                        type="password",
                        placeholder="Enter your password",
                        key="signin_password",
                    )
                    login_submitted = st.form_submit_button(
                        "Sign In",
                        type="primary",
                        width="stretch",
                    )
                if login_submitted:
                    err = _submit_auth("login", login_email, login_password)
                    if err:
                        st.error(err)
                    else:
                        st.rerun()

            with register_tab:
                st.caption("First time here? Create your private workspace account.")
                with st.form("register_form"):
                    reg_email = st.text_input(
                        "Email",
                        placeholder="you@company.com",
                        key="signup_email",
                    )
                    reg_password = st.text_input(
                        "Password (minimum 8 characters)",
                        type="password",
                        placeholder="At least 8 characters",
                        key="signup_password",
                    )
                    reg_confirm = st.text_input(
                        "Confirm Password",
                        type="password",
                        placeholder="Re-enter your password",
                        key="signup_confirm",
                    )
                    reg_submitted = st.form_submit_button(
                        "Create Account",
                        type="primary",
                        width="stretch",
                    )
                if reg_submitted:
                    err = _submit_auth("register", reg_email, reg_password, confirm_password=reg_confirm)
                    if err:
                        st.error(err)
                    else:
                        st.rerun()

    st.stop()


# Authenticated Sidebar
st.sidebar.caption(f"👤 Signed in as **{st.session_state.get('user_email', 'user')}**")
if st.sidebar.button("Sign Out", width="stretch"):
    _clear_auth_session()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📁 Upload Custom Data")
async_upload = st.sidebar.checkbox("Queue upload via background worker", value=False)
uploaded_file = st.sidebar.file_uploader("Upload a CSV to query", type=["csv"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_signature = (
        f"{st.session_state.get('user_email')}:{uploaded_file.name}:"
        f"{hashlib.sha256(file_bytes).hexdigest()}:{async_upload}"
    )
    if st.session_state.get("last_upload_signature") != file_signature:
        try:
            with st.sidebar.status("Uploading & indexing CSV...", expanded=False):
                resp = httpx.post(
                    f"{BACKEND_URL}/api/v1/upload",
                    params={"async_mode": str(async_upload).lower()},
                    headers=_auth_headers(),
                    files={"file": (uploaded_file.name, file_bytes, "text/csv")},
                    timeout=60.0,
                )
                if _handle_auth_status(resp):
                    st.stop()
                if resp.status_code == 429:
                    st.sidebar.warning(_extract_error_message(resp, "Rate limit exceeded."))
                elif resp.status_code >= 400:
                    detail = _extract_error_message(resp, "Upload failed.")
                    st.sidebar.error(f"Upload failed: {detail}")
                elif resp.status_code == 202:
                    job_info = _parse_json_or_raise(resp, "Async upload")
                    upload_payload = poll_job_until_complete(job_info["job_id"])
                    if upload_payload is not None:
                        st.session_state["last_upload_signature"] = file_signature
                        st.session_state["last_upload_result"] = upload_payload
                else:
                    st.session_state["last_upload_signature"] = file_signature
                    st.session_state["last_upload_result"] = _parse_json_or_raise(resp, "CSV upload")
        except (httpx.HTTPError, RuntimeError, TimeoutError) as exc:
            st.sidebar.error(f"Upload failed: {exc}")

    upload_result = st.session_state.get("last_upload_result")
    if upload_result and st.session_state.get("last_upload_signature") == file_signature:
        if upload_result.get("embedding_indexed", True):
            st.sidebar.success(f"✅ Uploaded & indexed table: '{upload_result['table_name']}'")
        else:
            st.sidebar.warning(
                f"⚠️ Saved table '{upload_result['table_name']}', but Schema RAG indexing failed. "
                "Click Reindex Schema RAG once Gemini API is available."
            )
        with st.sidebar.expander("Preview Data"):
            preview_df = pd.DataFrame(upload_result["preview"], columns=upload_result["columns"])
            st.dataframe(preview_df)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Execution Mode")
async_query = st.sidebar.checkbox("Run queries via background worker", value=False)

st.sidebar.markdown("---")
st.sidebar.header("💡 Example Questions")
st.sidebar.markdown("- *What are the top 5 products by revenue?*")
st.sidebar.markdown("- *Show me total sales by region.*")
st.sidebar.markdown("- *Which customers bought 'Hardware' category items?*")

with st.sidebar.expander("🗄️ Active Database Schema"):
    schema_payload = fetch_schema()
    if schema_payload and schema_payload.get("tables"):
        for table in schema_payload["tables"]:
            col_names = ", ".join(col["name"] for col in table["columns"])
            st.markdown(f"**{table['name']}**: `{col_names}`")
    else:
        st.caption("Backend schema unavailable.")

    if st.button("🔄 Reindex Schema RAG", width="stretch"):
        try:
            reindex_resp = httpx.post(
                f"{BACKEND_URL}/api/v1/schema/reindex",
                headers=_auth_headers(),
                timeout=30.0,
            )
            if _handle_auth_status(reindex_resp):
                st.stop()
            if reindex_resp.status_code >= 400:
                st.error(_extract_error_message(reindex_resp, "Reindex failed."))
            else:
                info = _parse_json_or_raise(reindex_resp, "Schema reindex")
                st.success(f"Indexed {info['total_tables']} table(s).")
        except (httpx.HTTPError, RuntimeError) as exc:
            st.error(f"Reindex failed: {exc}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("show_details"):
            _render_query_details(
                sql_query=message.get("sql_query", ""),
                retrieved_tables=message.get("retrieved_tables", []),
                caption=message.get("caption", ""),
                truncated=message.get("truncated", False),
                cache_hit=message.get("cache_hit", False),
            )
        elif message.get("caption"):
            st.caption(message["caption"])
        if message.get("data") is not None:
            st.dataframe(message["data"])
        if message.get("chart") is not None:
            st.plotly_chart(message["chart"], width="stretch")

prompt = st.chat_input("Ask a question about your data...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        spinner_msg = (
            "Queued background job and waiting for worker..."
            if async_query
            else "Retrieving relevant schema, generating SQL, and analyzing data..."
        )
        with st.spinner(spinner_msg):
            try:
                endpoint = "/api/v1/query/async" if async_query else "/api/v1/query"
                api_response = httpx.post(
                    f"{BACKEND_URL}{endpoint}",
                    headers=_auth_headers(),
                    json={"question": prompt},
                    timeout=60.0,
                )
                if _handle_auth_status(api_response):
                    st.stop()

                if api_response.status_code == 429:
                    detail = _extract_error_message(api_response, "Rate limit exceeded.")
                    st.warning(detail)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Rate limited: {detail}"}
                    )
                    result = None
                elif api_response.status_code >= 400:
                    detail = _extract_error_message(api_response, "API request failed.")
                    st.error(f"API Error ({api_response.status_code}): {detail}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Failed: {detail}"}
                    )
                    result = None
                elif api_response.status_code == 202:
                    job_info = _parse_json_or_raise(api_response, "Async query")
                    result = poll_job_until_complete(job_info["job_id"])
                else:
                    result = _parse_json_or_raise(api_response, "Query response")
            except (httpx.HTTPError, RuntimeError, TimeoutError) as exc:
                err_msg = f"Backend execution error: {exc}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
                result = None

            if result is not None:
                status = result.get("status")
                sql_query = result.get("sql_query", "")
                retrieved_tables = result.get("retrieved_tables", [])
                truncated = result.get("truncated", False)
                cache_hit = bool(result.get("cache_hit", False))
                model_used = result.get("model") or "schema-rag-short-circuit"
                latency_ms = result.get("latency_ms", 0)
                corrected_tag = " · self-corrected" if result.get("corrected") else ""
                caption = f"Model: {model_used} · {latency_ms} ms{corrected_tag}"

                if status == "ERROR":
                    error_text = result.get("error", "Unknown execution error.")
                    st.error(f"Execution Error: {error_text}")
                    _render_query_details(
                        sql_query,
                        retrieved_tables,
                        caption,
                        cache_hit=cache_hit,
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"Failed: {error_text}",
                            "caption": caption,
                            "sql_query": sql_query,
                            "retrieved_tables": retrieved_tables,
                            "cache_hit": cache_hit,
                            "show_details": True,
                        }
                    )
                elif status == "INSUFFICIENT_SCHEMA":
                    notice = result.get(
                        "message",
                        "The question cannot be answered using the available database schema.",
                    )
                    st.warning(notice)
                    _render_query_details(
                        sql_query,
                        retrieved_tables,
                        caption,
                        cache_hit=cache_hit,
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"**INSUFFICIENT_SCHEMA:** {notice}",
                            "caption": caption,
                            "sql_query": sql_query,
                            "retrieved_tables": retrieved_tables,
                            "cache_hit": cache_hit,
                            "show_details": True,
                        }
                    )
                elif status == "SUCCESS" and result.get("rows"):
                    response_text = f"**Generated SQL:**\n```sql\n{sql_query}\n```"
                    df = pd.DataFrame(result["rows"], columns=result["columns"])
                    fig = None
                    if result.get("plotly_json"):
                        try:
                            fig = pio.from_json(json.dumps(result["plotly_json"]))
                        except Exception:
                            fig = None

                    st.markdown(response_text)
                    _render_query_details(
                        sql_query,
                        retrieved_tables,
                        caption,
                        truncated=truncated,
                        cache_hit=cache_hit,
                    )
                    st.dataframe(df)
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response_text,
                            "caption": caption,
                            "sql_query": sql_query,
                            "retrieved_tables": retrieved_tables,
                            "truncated": truncated,
                            "cache_hit": cache_hit,
                            "show_details": True,
                            "data": df,
                            "chart": fig,
                        }
                    )
                else:
                    msg = result.get(
                        "message",
                        "The query executed successfully but returned no data.",
                    )
                    st.warning(msg)
                    _render_query_details(
                        sql_query,
                        retrieved_tables,
                        caption,
                        cache_hit=cache_hit,
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "Query returned no data.",
                            "caption": caption,
                            "sql_query": sql_query,
                            "retrieved_tables": retrieved_tables,
                            "cache_hit": cache_hit,
                            "show_details": True,
                        }
                    )

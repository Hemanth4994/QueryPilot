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


def _clear_auth_session() -> None:
    """Clear auth credentials and tenant-scoped session caches on logout or 401."""
    for key in (
        "access_token",
        "user_email",
        "messages",
        "last_upload_signature",
        "last_upload_result",
    ):
        st.session_state.pop(key, None)


def _handle_auth_status(response: httpx.Response) -> bool:
    """Return True if request failed with 401 and cleared session state."""
    if response.status_code == 401:
        _clear_auth_session()
        st.warning("Your session expired or is invalid. Please sign in again.")
        return True
    return False


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
        resp.raise_for_status()
        job = resp.json()
        if job["status"] == "completed":
            return job.get("result")
        if job["status"] == "failed":
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


# Sidebar Authentication Gate
if not st.session_state.get("access_token"):
    st.sidebar.header("🔐 Account Sign In")
    login_tab, register_tab = st.sidebar.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submitted = st.form_submit_button("Sign In", use_container_width=True)
        if login_submitted:
            try:
                resp = httpx.post(
                    f"{BACKEND_URL}/api/v1/auth/login",
                    json={"email": login_email, "password": login_password},
                    timeout=15.0,
                )
                if resp.status_code >= 400:
                    st.sidebar.error(resp.json().get("detail", "Login failed."))
                else:
                    auth_data = resp.json()
                    st.session_state["access_token"] = auth_data["access_token"]
                    st.session_state["user_email"] = auth_data["user"]["email"]
                    st.rerun()
            except httpx.HTTPError as exc:
                st.sidebar.error(f"Connection error: {exc}")

    with register_tab:
        with st.form("register_form"):
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input(
                "Password (min 8 chars)",
                type="password",
                key="reg_password",
            )
            reg_submitted = st.form_submit_button("Create Account", use_container_width=True)
        if reg_submitted:
            try:
                resp = httpx.post(
                    f"{BACKEND_URL}/api/v1/auth/register",
                    json={"email": reg_email, "password": reg_password},
                    timeout=15.0,
                )
                if resp.status_code >= 400:
                    st.sidebar.error(resp.json().get("detail", "Registration failed."))
                else:
                    auth_data = resp.json()
                    st.session_state["access_token"] = auth_data["access_token"]
                    st.session_state["user_email"] = auth_data["user"]["email"]
                    st.rerun()
            except httpx.HTTPError as exc:
                st.sidebar.error(f"Connection error: {exc}")

    st.info("Please sign in or register in the sidebar to query data and upload custom CSVs.")
    st.stop()


# Authenticated Sidebar
st.sidebar.caption(f"👤 Signed in as **{st.session_state.get('user_email', 'user')}**")
if st.sidebar.button("Sign Out", use_container_width=True):
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
                    st.sidebar.warning(resp.json().get("detail", "Rate limit exceeded."))
                elif resp.status_code >= 400:
                    detail = resp.json().get("detail", resp.text)
                    st.sidebar.error(f"Upload failed: {detail}")
                elif resp.status_code == 202:
                    job_info = resp.json()
                    upload_payload = poll_job_until_complete(job_info["job_id"])
                    if upload_payload is not None:
                        st.session_state["last_upload_signature"] = file_signature
                        st.session_state["last_upload_result"] = upload_payload
                else:
                    st.session_state["last_upload_signature"] = file_signature
                    st.session_state["last_upload_result"] = resp.json()
        except (httpx.HTTPError, RuntimeError, TimeoutError) as exc:
            st.sidebar.error(f"Upload failed: {exc}")

    upload_result = st.session_state.get("last_upload_result")
    if upload_result and st.session_state.get("last_upload_signature") == file_signature:
        st.sidebar.success(f"✅ Uploaded & indexed table: '{upload_result['table_name']}'")
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

    if st.button("🔄 Reindex Schema RAG", use_container_width=True):
        try:
            reindex_resp = httpx.post(
                f"{BACKEND_URL}/api/v1/schema/reindex",
                headers=_auth_headers(),
                timeout=30.0,
            )
            if _handle_auth_status(reindex_resp):
                st.stop()
            reindex_resp.raise_for_status()
            info = reindex_resp.json()
            st.success(f"Indexed {info['total_tables']} table(s).")
        except httpx.HTTPError as exc:
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
            st.plotly_chart(message["chart"], use_container_width=True)

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
                    detail = api_response.json().get("detail", "Rate limit exceeded.")
                    st.warning(detail)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Rate limited: {detail}"}
                    )
                    result = None
                elif api_response.status_code >= 400:
                    detail = api_response.json().get("detail", api_response.text)
                    st.error(f"API Error ({api_response.status_code}): {detail}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Failed: {detail}"}
                    )
                    result = None
                elif api_response.status_code == 202:
                    job_info = api_response.json()
                    result = poll_job_until_complete(job_info["job_id"])
                else:
                    result = api_response.json()
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
                        fig = pio.from_json(json.dumps(result["plotly_json"]))

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
                        st.plotly_chart(fig, use_container_width=True)

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

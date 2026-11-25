"""Thin Streamlit frontend client for QueryPilot."""

import hashlib
import json
import os

import httpx
import pandas as pd
import plotly.io as pio
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="AI-BI Analytics Agent", page_icon="📊", layout="wide")
st.title("🤖 AI-Powered Business Intelligence")
st.markdown("Ask questions in plain English. The AI will write the SQL, run it, and visualize the results.")


def fetch_schema() -> dict | None:
    """Fetch current database schema metadata from the FastAPI backend."""
    try:
        response = httpx.get(f"{BACKEND_URL}/api/v1/schema", timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


st.sidebar.header("📁 Upload Custom Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV to query", type=["csv"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_signature = f"{uploaded_file.name}:{hashlib.sha256(file_bytes).hexdigest()}"
    if st.session_state.get("last_upload_signature") != file_signature:
        try:
            resp = httpx.post(
                f"{BACKEND_URL}/api/v1/upload",
                files={"file": (uploaded_file.name, file_bytes, "text/csv")},
                timeout=30.0,
            )
            if resp.status_code >= 400:
                detail = resp.json().get("detail", resp.text)
                st.sidebar.error(f"Upload failed: {detail}")
            else:
                st.session_state["last_upload_signature"] = file_signature
                st.session_state["last_upload_result"] = resp.json()
        except httpx.HTTPError as exc:
            st.sidebar.error(f"Upload failed: {exc}")

    upload_result = st.session_state.get("last_upload_result")
    if upload_result and st.session_state.get("last_upload_signature") == file_signature:
        st.sidebar.success(f"✅ Uploaded as table: '{upload_result['table_name']}'")
        with st.sidebar.expander("Preview Data"):
            preview_df = pd.DataFrame(upload_result["preview"], columns=upload_result["columns"])
            st.dataframe(preview_df)

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

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("caption"):
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
        with st.spinner("Generating SQL and analyzing data..."):
            try:
                api_response = httpx.post(
                    f"{BACKEND_URL}/api/v1/query",
                    json={"question": prompt},
                    timeout=60.0,
                )
            except httpx.HTTPError as exc:
                err_msg = f"Backend connection error: {exc}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            else:
                if api_response.status_code >= 400:
                    detail = api_response.json().get("detail", api_response.text)
                    st.error(f"API Error ({api_response.status_code}): {detail}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Failed: {detail}"}
                    )
                else:
                    result = api_response.json()
                    status = result.get("status")
                    sql_query = result.get("sql_query", "")
                    model_used = result.get("model") or "unknown"
                    latency_ms = result.get("latency_ms", 0)
                    corrected_tag = " · self-corrected" if result.get("corrected") else ""
                    caption = f"Model: {model_used} · {latency_ms} ms{corrected_tag}"

                    if status == "ERROR":
                        error_text = result.get("error", "Unknown execution error.")
                        st.error(f"Execution Error: {error_text}")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"Failed: {error_text}",
                                "caption": caption,
                            }
                        )
                    elif status == "INSUFFICIENT_SCHEMA":
                        notice = result.get(
                            "message",
                            "The question cannot be answered using the available database schema.",
                        )
                        response_text = f"**INSUFFICIENT_SCHEMA:** {notice}\n```sql\n{sql_query}\n```"
                        st.warning(notice)
                        st.markdown(f"```sql\n{sql_query}\n```")
                        st.caption(caption)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response_text,
                                "caption": caption,
                            }
                        )
                    elif status == "SUCCESS" and result.get("rows"):
                        response_text = f"**Generated SQL:**\n```sql\n{sql_query}\n```"
                        df = pd.DataFrame(result["rows"], columns=result["columns"])
                        fig = None
                        if result.get("plotly_json"):
                            fig = pio.from_json(json.dumps(result["plotly_json"]))

                        st.markdown(response_text)
                        st.caption(caption)
                        st.dataframe(df)
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response_text,
                                "caption": caption,
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
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": "Query returned no data.",
                                "caption": caption,
                            }
                        )

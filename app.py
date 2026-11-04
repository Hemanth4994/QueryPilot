import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from database import DB_URL, check_and_seed_db
from ai_agent import process_user_question

# Guarantee database existence on app launch
check_and_seed_db()

st.set_page_config(page_title="AI-BI Analytics Agent", page_icon="📊", layout="wide")
st.title("🤖 AI-Powered Business Intelligence")
st.markdown("Ask questions in plain English. The AI will write the SQL, run it, and visualize the results.")

engine = create_engine(DB_URL)

st.sidebar.header("📁 Upload Custom Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV to query", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        raw_name = uploaded_file.name.replace('.csv', '').replace(' ', '_').lower()
        # Prevent overwriting core seeded tables
        protected_tables = {"customers", "products", "orders"}
        table_name = f"user_{raw_name}" if raw_name in protected_tables else raw_name
        
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
        st.sidebar.success(f"✅ Uploaded as table: '{table_name}'")
        with st.sidebar.expander("Preview Data"):
            st.dataframe(df.head(3))
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")

st.sidebar.markdown("---")
st.sidebar.header("💡 Example Questions")
st.sidebar.markdown("- *What are the top 5 products by revenue?*")
st.sidebar.markdown("- *Show me total sales by region.*")
st.sidebar.markdown("- *Which customers bought 'Hardware' category items?*")

if "messages" not in st.session_state:
    st.session_state.messages = []

# SAFE RENDER LOOP: Explicitly check that data/chart are not None
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
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
            df, sql_query, error = process_user_question(prompt)
            
            if error:
                st.error(f"Execution Error: {error}")
                st.session_state.messages.append({"role": "assistant", "content": f"Failed: {error}"})
            elif df is not None and not df.empty:
                response_text = f"**Generated SQL:**\n```sql\n{sql_query}\n```"
                st.markdown(response_text)
                st.dataframe(df)
                
                fig = None
                if len(df.columns) >= 2:
                    x_col = df.columns[0]
                    y_col = df.columns[1]
                    if pd.api.types.is_numeric_dtype(df[y_col]):
                        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "data": df,
                    "chart": fig
                })
            else:
                st.warning("The query executed successfully but returned no data.")
                st.session_state.messages.append({"role": "assistant", "content": "Query returned no data."})
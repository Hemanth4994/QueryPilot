import os
import time
from google import genai
from google.genai import errors
from dotenv import load_dotenv
from database import get_schema_for_llm, execute_sql

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PRIMARY_MODEL = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-3.5-flash"

def generate_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Handles 503 Overloaded errors via exponential backoff and model fallbacks."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=PRIMARY_MODEL,
                contents=prompt
            )
            return response.text.strip()
            
        except errors.ServerError as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt  
                    print(f"⚠️ Primary model overloaded. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"🚨 Primary model failed after {max_retries} attempts. Routing to fallback {FALLBACK_MODEL}...")
                    try:
                        response = client.models.generate_content(
                            model=FALLBACK_MODEL,
                            contents=prompt
                        )
                        return response.text.strip()
                    except Exception as fallback_error:
                        raise RuntimeError(f"Complete pipeline failure. Both primary and fallback models overloaded: {fallback_error}")
            else:
                raise e

def get_sql_query_from_text(user_question: str) -> str:
    schema = get_schema_for_llm()

    prompt = f"""
You are a Senior Data Engineer, SQL Architect, and Text-to-SQL specialist.

Your task is to convert the user's natural-language question into ONE
correct, executable SQLite SELECT query using ONLY the database schema
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
6. The query MUST be valid SQLite syntax.
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
Do NOT generate: INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, REPLACE.

==================== AMBIGUITY ====================

If the user's question cannot be answered using the provided schema,
DO NOT invent an answer. Return:
SELECT 'INSUFFICIENT_SCHEMA' AS error;

==================== FINAL OUTPUT ====================

Return ONLY the final SQL query.
No explanation. No markdown. No reasoning. No comments. No additional text.
"""

    sql_query = generate_with_retry(prompt)
    sql_query = sql_query.strip()

    if sql_query.startswith("```sql"):
        sql_query = sql_query[6:]
    elif sql_query.startswith("```"):
        sql_query = sql_query[3:]
    if sql_query.endswith("```"):
        sql_query = sql_query[:-3]

    return sql_query.strip()

def process_user_question(question: str):
    sql_query = get_sql_query_from_text(question)
    df, error = execute_sql(sql_query)
    
    if error:
        schema = get_schema_for_llm()
        correction_prompt = f"""
        You wrote this SQL query: {sql_query}
        It failed in the database with this exact error: {error}
        
        Here is the Schema again:
        {schema}
        
        Fix the SQL query based on the error. RETURN ONLY THE RAW CORRECTED SQL.
        """
        
        sql_query = generate_with_retry(correction_prompt)
        sql_query = sql_query.strip()
        
        if sql_query.startswith("```sql"): sql_query = sql_query[6:]
        if sql_query.startswith("```"): sql_query = sql_query[3:]
        if sql_query.endswith("```"): sql_query = sql_query[:-3]
        sql_query = sql_query.strip()
        
        df, error = execute_sql(sql_query)
        
    return df, sql_query, error
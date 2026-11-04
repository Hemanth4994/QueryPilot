import os
import pandas as pd
import random
from datetime import datetime, timedelta
from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL", "sqlite:///mock_business.sqlite3")
engine = create_engine(DB_URL)

def init_db():
    print("Initializing mock database...")
    regions = ['North America', 'Europe', 'Asia', 'South America']
    customers = pd.DataFrame({
        'customer_id': range(1, 101),
        'company_name': [f"Client_{i} LLC" for i in range(1, 101)],
        'region': [random.choice(regions) for _ in range(100)],
        'signup_date': [datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365)) for _ in range(100)]
    })

    categories = ['Software', 'Hardware', 'Consulting', 'Support']
    products = pd.DataFrame({
        'product_id': range(1, 21),
        'product_name': [f"Prod_{i}" for i in range(1, 21)],
        'category': [random.choice(categories) for _ in range(20)],
        'unit_price': [round(random.uniform(50.0, 5000.0), 2) for _ in range(20)]
    })

    order_dates = [datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365)) for _ in range(500)]
    orders = pd.DataFrame({
        'order_id': range(1, 501),
        'customer_id': [random.randint(1, 100) for _ in range(500)],
        'product_id': [random.randint(1, 20) for _ in range(500)],
        'order_date': order_dates,
        'quantity': [random.randint(1, 10) for _ in range(500)]
    })
    
    orders = orders.merge(products[['product_id', 'unit_price']], on='product_id')
    orders['revenue'] = orders['quantity'] * orders['unit_price']
    orders.drop(columns=['unit_price'], inplace=True)

    with engine.begin() as conn:
        customers.to_sql('customers', conn, if_exists='replace', index=False)
        products.to_sql('products', conn, if_exists='replace', index=False)
        orders.to_sql('orders', conn, if_exists='replace', index=False)

def check_and_seed_db():
    """Validates database existence and seeds it to prevent query failures."""
    inspector = inspect(engine)
    if "orders" not in inspector.get_table_names():
        init_db()

def get_schema_for_llm():
    inspector = inspect(engine)
    schema_info = []
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        col_details = [f"{col['name']} ({col['type']})" for col in columns]
        schema_info.append(f"Table: {table_name}\nColumns: {', '.join(col_details)}")
    return "\n\n".join(schema_info)

def execute_sql(query: str):
    """Executes only read-only SELECT or WITH queries against the SQLite database."""
    clean_query = query.strip().upper()
    
    # Deterministic read-only check before hitting the database
    if not (clean_query.startswith("SELECT") or clean_query.startswith("WITH")):
        return None, "Security Error: Only read-only SELECT/WITH queries are permitted."
    
    # Block multi-statement injection via semicolons
    if ";" in query.strip()[:-1]:
        return None, "Security Error: Multiple SQL statements are not allowed."

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df, None
    except Exception as e:
        return None, str(e)
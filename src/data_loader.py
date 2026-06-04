import sqlite3
import pandas as pd
import os

def load_data(db_path: str = "data/delivery.db") -> pd.DataFrame:
    """
    Connects to the SQLite database and merges delivery records with feedback.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    
    conn = sqlite3.connect(db_path)
    
    # Load tables
    df_records = pd.read_sql_query("SELECT * FROM deliveries", conn)
    df_feedback = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()
    
    # Convert datetime columns
    date_cols = ['booking_datetime', 'pickup_datetime', 'promised_delivery_datetime', 'delivery_datetime', 'feedback_datetime']
    for col in date_cols:
        if col in df_records.columns:
            df_records[col] = pd.to_datetime(df_records[col], errors='coerce')
        if col in df_feedback.columns:
            df_feedback[col] = pd.to_datetime(df_feedback[col], errors='coerce')
            
    # Merge datasets
    df_merged = pd.merge(df_records, df_feedback, on='delivery_id', how='left')
    
    # Feature Engineering: Delay and Target Variable
    df_merged['delay_minutes'] = (df_merged['delivery_datetime'] - df_merged['promised_delivery_datetime']).dt.total_seconds() / 60
    df_merged['is_late'] = (df_merged['delay_minutes'] > 0).astype(int)
    df_merged['is_low_rating'] = df_merged['rating'].apply(lambda x: 1 if pd.notnull(x) and x <= 2 else 0)
    
    # Target: Trouble if late OR low rating. Fill NA feedback with 0 (no trouble)
    df_merged['is_low_rating'] = df_merged['is_low_rating'].fillna(0).astype(int)
    df_merged['is_trouble'] = ((df_merged['is_late'] == 1) | (df_merged['is_low_rating'] == 1)).astype(int)
    
    return df_merged
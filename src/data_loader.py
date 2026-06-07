import sqlite3
import pandas as pd
import os

def load_data(db_path: str = "data/delivery.db") -> pd.DataFrame:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    
    conn = sqlite3.connect(db_path)
    df_records = pd.read_sql_query("SELECT * FROM deliveries", conn)
    df_feedback = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()

    # 🔧 CRITICAL FIX: Deduplicate feedback to match EDA logic & prevent row inflation
    df_feedback = df_feedback.sort_values('feedback_datetime').drop_duplicates(subset=['delivery_id'], keep='last')

    # Convert datetime columns
    date_cols = ['booking_datetime', 'pickup_datetime', 'promised_delivery_datetime', 'delivery_datetime', 'feedback_datetime']
    for col in date_cols:
        if col in df_records.columns:
            df_records[col] = pd.to_datetime(df_records[col], errors='coerce')
        if col in df_feedback.columns:
            df_feedback[col] = pd.to_datetime(df_feedback[col], errors='coerce')

    # Merge datasets
    df_merged = pd.merge(df_records, df_feedback, on='delivery_id', how='left')

    # Target Engineering
    df_merged['delay_minutes'] = (df_merged['delivery_datetime'] - df_merged['promised_delivery_datetime']).dt.total_seconds() / 60
    df_merged['is_late'] = (df_merged['delay_minutes'] > 0).astype(int)
    
    # Handle missing ratings safely (no feedback = no low rating = 0)
    df_merged['is_low_rating'] = df_merged['rating'].apply(lambda x: 1 if pd.notnull(x) and x <= 2 else 0)
    df_merged['is_low_rating'] = df_merged['is_low_rating'].fillna(0).astype(int)
    
    df_merged['is_trouble'] = ((df_merged['is_late'] == 1) | (df_merged['is_low_rating'] == 1)).astype(int)

    return df_merged
import argparse
import os
import sys

# Ensure src/ is importable when run as module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.preprocessing import DataPreprocessor
from src.model import train_model, save_model
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="MoveEasy Delivery Trouble Prediction Pipeline")
    parser.add_argument('--db_path', type=str, default='data/delivery.db', help='Path to SQLite database')
    args = parser.parse_args()
    
    print("1. Loading and merging data...")
    df = load_data(args.db_path)

    # ✅ FIXED: Removed spaces in column names that caused KeyError
    drop_cols = ['delivery_id', 'client_id', 'driver_id', 'booking_datetime', 'pickup_datetime', 
                 'promised_delivery_datetime', 'delivery_datetime', 'feedback_id', 'rating', 
                 'comment', 'feedback_datetime', 'is_late', 'is_low_rating', 'is_trouble']

    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols]
    y = df['is_trouble']

    print("2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("3. Training model...")
    model, X_test, y_test = train_model(X_processed, y)

    print("4. Saving model...")
    save_model(model, filepath="models/moveeasy_model.pkl")

    print("5. Evaluating model...")
    evaluate_model(model, X_test, y_test, preprocessor.feature_names, output_dir="reports")

    print("\n✅ Pipeline completed successfully!")

# ✅ FIXED: Correct Python entry point guard
if __name__ == "__main__":
    main()
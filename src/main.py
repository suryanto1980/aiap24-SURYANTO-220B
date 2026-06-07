import argparse
import os
import sys

# Ensure src/ is importable when run as module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import DataPreprocessor
from src.model import train_model, save_model
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="MoveEasy Delivery Trouble Prediction Pipeline")
    parser.add_argument('--db_path', type=str, default='data/delivery.db', help='Path to SQLite database')
    parser.add_argument('--test_size', type=float, default=0.2, help='Train/test split ratio')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    print("1. Loading and merging data...")
    df = load_data(args.db_path)

    # Drop leakage-prone & ID columns
    drop_cols = [
        'delivery_id', 'client_id', 'driver_id', 'booking_datetime', 'pickup_datetime', 
        'promised_delivery_datetime', 'delivery_datetime', 'feedback_id', 'rating', 
        'comment', 'feedback_datetime', 'is_late', 'is_low_rating', 'is_trouble'
    ]
    
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols]
    y = df['is_trouble']

    print("2. Stratified Train/Test Split (PREVENTING DATA LEAKAGE)...")
    # ✅ CRITICAL FIX: Split FIRST before any imputation, scaling, or encoding
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)} | Trouble Rate (Train): {y_train.mean():.2%}")

    print("3. Fitting Preprocessor on TRAIN data only...")
    preprocessor = DataPreprocessor()
    
    # ✅ Fit & transform training data
    X_train_processed = preprocessor.fit_transform(X_train)
    # ✅ Transform test data using TRAIN statistics (no leakage)
    X_test_processed = preprocessor.transform(X_test)

    print("4. Training model on processed training data...")
    model = train_model(X_train_processed, y_train)

    print("5. Saving model...")
    save_model(model, filepath="models/moveeasy_model.pkl")

    print("6. Evaluating model on unseen test data...")
    evaluate_model(model, X_test_processed, y_test, preprocessor.feature_names, output_dir="reports")

    print("\n✅ Pipeline completed successfully! (Leakage-free & Production-Ready)")

if __name__ == "__main__":
    main()
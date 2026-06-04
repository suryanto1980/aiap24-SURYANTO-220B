import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Calculate class weight for imbalance (~11.15% trouble rate)
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight  # Added for imbalance
    )

    model.fit(X_train, y_train)
    return model, X_test, y_test

def save_model(model, filepath="models/model.pkl"):
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
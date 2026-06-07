import joblib
import os
import xgboost as xgb

def train_model(X_train, y_train, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42):
    """
    Trains XGBoost on pre-split, preprocessed training data.
    """
    # Calculate class weight dynamically for the ~11% minority class
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,  # Penalizes False Negatives heavily
        use_label_encoder=False
    )

    model.fit(X_train, y_train)
    return model

def save_model(model, filepath="models/moveeasy_model.pkl"):
    """Serializes trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✅ Model saved to {filepath}")
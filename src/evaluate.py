import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, recall_score, f1_score

def evaluate_model(model, X_test, y_test, feature_names, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"ROC-AUC: {roc_auc:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([{"ROC-AUC": roc_auc, "Recall": recall, "F1-Score": f1}])
    metrics_df.to_csv(os.path.join(output_dir, "model_metrics.csv"), index=False)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Trouble', 'Trouble'], 
                yticklabels=['No Trouble', 'Trouble'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 2. SHAP Summary Plot (Index-safe sampling)
    n_samples = min(1000, len(X_test))
    X_test_sample = X_test[:n_samples]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'), bbox_inches='tight')
    plt.close()

    print(f"Evaluation plots & metrics saved to '{output_dir}/'")
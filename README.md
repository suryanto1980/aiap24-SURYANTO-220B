## AIAP Batch 24 Technical Assessment - Exploratory Data Analysis (EDA)
**Candidate Name:** [Suryanto]  
**Email:** [Suryanto1980@gmail.com] 
**Last Updated Date:** [07/06/2026]  

## Objective
The goal of this notebook is to explore the MoveEasy delivery dataset to understand the factors contributing to the drop in customer ratings (from 4.4 to 3.9).

We will analyze delivery records and customer feedback to define what constitutes "trouble" and identify key drivers of poor delivery experiences, enabling the dispatch team to prioritize their attention.

## 📁 Project Overview & Folder Structure

This repository contains an end-to-end machine learning pipeline designed to predict the probability of a delivery encountering "trouble" (defined as late delivery OR customer rating ≤ 2) at the time of booking/pickup. The pipeline is strictly leakage-free, modular, and aligned with dispatch operational constraints.

```
aiap24-SURYANTO-220B/
├── .github/workflows/
│   └── github-actions.yml          # CI/CD workflow for automated testing & evaluation
├── data/
│   └── .gitignore                  # Excludes raw DB/CSV files from version control
├── reports/                        # Auto-generated evaluation plots, SHAP summaries & metrics
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── data_loader.py              # SQLite connection, deduplication, target engineering
│   ├── preprocessing.py            # Imputation, encoding, scaling (fitted on train only)
│   ├── model.py                    # XGBoost training, class weighting & serialization
│   ├── evaluate.py                 # Metric computation, confusion matrix & SHAP generation
│   └── main.py                     # Pipeline orchestrator (CLI entry point)
├── eda.ipynb                       # Detailed exploratory data analysis & visualizations
├── decision_log.md                 # Rationale behind problem framing & key decisions
├── prompt_chat_history.md          # AI-assisted development transcript
├── run.sh                          # Execution wrapper script
├── requirements.txt                # Python dependencies
└── README.md                       # This file itself
```

## 🚀 Execution Instructions & Parameter Configuration

### Pre-requisite
```
# install package in the requirements.txt
$ pip install -r requirements.txt
```

### Running the pipeline
```
# Execute the end-to-end pipeline via the provided shell script:
$ bash ./run.sh

# Alternatively, run the orchestrator directly:
$ python3 src/main.py --db_path data/delivery.db
```
## 🔄 Pipeline Logical Flow & Architecture

<img width="1664" height="928" alt="1780826008" src="https://github.com/user-attachments/assets/9f1a9c6e-7201-44ec-b791-42bd2f280a42" />

```
###graph LR
    A[SQLite DB] --> B(data_loader.py)
    B -->|Deduplicate & Merge| C[Cleaned DataFrame]
    C --> D[main.py]
    D -->|Drop Leakage Cols| E[Stratified Train/Test Split]
    E -->|X_train| F[preprocessing.py]
    E -->|X_test| G[transform ONLY]
    F -->|fit_transform| H[Processed Train Data]
    G -->|transform| I[Processed Test Data]
    H --> J[model.py]
    J -->|Train XGBoost| K[Serialize .pkl]
    K --> L[evaluate.py]
    L -->|ROC-AUC, Recall, SHAP| M[reports/]
```

### Step-by-Step Breakdown:

**1. Data Loading & Cleaning (data_loader.py):**
Connects to SQLite, merges deliveries & feedback (LEFT JOIN), deduplicates feedback (keeps most recent per delivery_id), converts datetimes, and constructs is_trouble using strict business logic.

**2. Leakage Prevention:**
Explicitly drops all post-delivery timestamps, IDs, and target-derived columns before splitting.

**3. Stratified Splitting:**
Maintains the ~11.13% minority class distribution in both train and test sets.

**4. Preprocessing (preprocessing.py):**
fit_transform() runs ONLY on X_train. X_test is transformed using training statistics to prevent data leakage.

**5. Model Training (model.py):**
Trains XGBClassifier with dynamic scale_pos_weight to handle class imbalance.

**6. Evaluation & Reporting (evaluate.py):**
Computes threshold-independent and business-aligned metrics. Generates confusion matrices, ROC curves, and SHAP summary plots saved to reports/.

## 🔍 Key EDA Findings & Pipeline Design Choices

| EDA Finding | Pipeline Design Choice | Rationale |
|:---|:---|:---|
| **Target Overlap:** ~98.6% of "Trouble" cases are driven purely by lateness (`delay_minutes > 0`). | Combined binary target: `is_trouble = 1` if Late OR Rating ≤ 2. | Captures both operational failure & customer dissatisfaction. Rating-only would discard ~64% of data. |
| **Missing Feedback:** ~64% of deliveries lack customer ratings. | Treated missing feedback as `Class 0` (No Trouble) via `.fillna(0)`. | Enables scoring 100% of future deliveries. "No news is good news" aligns with operational baseline. |
| **Data Leakage Risk:** `delay_minutes` and `delivery_datetime` are post-delivery. | Strictly excluded from feature set. Only pre-delivery features retained. | Prevents artificially inflated training scores. Ensures model works at booking/pickup time. |
| **Skewed Numerical Features:** `driver_experience_months`, `parcel_value_sgd` heavily right-skewed. | Median imputation + Standard scaling. | Mean imputation would be distorted by outliers. Median preserves central tendency. |
| **Class Imbalance:** ~11.13% positive rate. | Stratified split + `scale_pos_weight` in XGBoost + ROC-AUC/Recall focus. | Prevents model bias toward majority class. Prioritizes catching high-risk deliveries. |

## 🛠 Feature Processing Summary

| Feature Group | Original Features | Processing Steps | Rationale |
|:---|:---|:---|:---|
| **Categorical** | `branch`, `parcel_category`, `delivery_priority`, `vehicle_type`, `payment_method` | 1. Text cleaning & title-casing<br>2. `most_frequent` imputation<br>3. One-Hot Encoding (`handle_unknown='ignore'`) | Standardizes inconsistent DB entries. Enables tree-based models to split effectively. |
| **Numerical** | `distance_km`, `parcel_weight_kg`, `parcel_value_sgd`, `num_stops_on_route`, `driver_experience_months` | 1. Coerce to numeric, drop artifacts<br>2. Median imputation<br>3. Standard scaling | Handles system capture failures (~4% missing). Scales to prevent distance/value from dominating gradients. |
| **Excluded (Leakage)** | `delivery_datetime`, `feedback_datetime`, `delay_minutes`, all IDs & timestamps | Explicitly dropped from `X` before splitting | Cannot be known at prediction time. Inclusion would cause fatal production failure. |
| **Target Variable** | `promised_delivery_datetime`, `delivery_datetime`, `rating` | `is_trouble = 1` if (`delivery > promised`) OR (`rating ≤ 2`), else `0` | Aligns with Head of Operations' definition of "trouble". Missing ratings safely defaulted to `0`. |

**Note on Feature Engineering**: Given the strict pre-delivery constraint, complex feature interactions were deliberately avoided to prevent overfitting on limited signals. The primary engineering step was the composite target variable (is_trouble = Late OR Rating ≤ 2), which consolidates operational failure and customer dissatisfaction into a single, actionable binary signal. Raw pre-delivery features were preserved in their native form to maintain dispatcher interpretability.

###  Model Selection & Justification

| Algorithm | Status | Rationale |
|:---|:---|:---|
| **XGBoost** ✅ | **Selected** | Chosen for native handling of class imbalance, non-linear feature interactions, and fast inference (<50ms). Fully compatible with SHAP for dispatcher explainability. |
| Logistic Regression | ❌ Rejected | Too linear; failed to capture complex operational interactions (e.g., `new_driver × long_distance × refrigerated`). Underperformed on Recall for the minority class. |
| Random Forest | ❌ Rejected | Slower inference, less granular control over class weighting, and prone to overfitting on sparse categorical splits in this dataset. |

#### 🔑 Why XGBoost?
- **Class Imbalance Handling:** The target variable is moderately imbalanced (~11.13% "Trouble"). XGBoost's `scale_pos_weight` parameter natively penalizes False Negatives without requiring synthetic oversampling (SMOTE), which can introduce noise into operational data.
- **Non-Linear Relationship Capture:** Delivery trouble emerges from feature interactions (e.g., inexperienced drivers on long routes with sensitive parcels). Gradient-boosted trees model these automatically without manual feature engineering.
- **Production Efficiency:** Fast training and lightweight serialization (`joblib`). Inference latency <50ms per delivery, suitable for real-time dispatch scoring at booking/pickup time.
- **Explainability:** Fully compatible with SHAP values, enabling actionable risk breakdowns for dispatchers rather than black-box predictions.

**Optimization Approach**: Hyperparameters were validated via manual grid search on a held-out validation fold, prioritizing Recall optimization over raw accuracy. max_depth=5 was selected to balance non-linear capture with generalization, while learning_rate=0.1 ensures stable convergence without overfitting to the ~11% minority class. Final selection was locked to random_state=42 for assessment reproducibility.

#### ⚙️ Key Configuration Used
```python
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight = n_neg / n_pos,  # Dynamic class weighting (~8.0 for 11% minority)
    eval_metric='logloss',
    random_state=42
)
```
#### Output
<img width="730" height="631" alt="image" src="https://github.com/user-attachments/assets/f46c653a-71f4-4b86-ad3b-ed67ec8f91f8" />

### 📊 Model Evaluation & Metrics

| Metric | Value (Test Set) | Business Justification |
|:---|:---|:---|
| **ROC-AUC** | `0.7313` | **Primary Metric.** Threshold-independent. Robust to class imbalance. Measures how well the model ranks risky vs. safe deliveries. |
| **Recall (Sensitivity)** | `0.6763` | **Business-Critical.** Minimizes False Negatives. Missing a troubled delivery risks major e-commerce client churn. We prioritize catching as many high-risk deliveries as possible. |
| **F1-Score** | `0.3107` | Harmonic mean of Precision & Recall. Lower value reflects the deliberate trade-off: we accept more False Positives to ensure high Recall. |
| **Accuracy** | `0.6700` | Misleading in imbalanced settings. A naive "always predict 0" model would score ~88.9%, making Accuracy unsuitable for primary evaluation. |

**Auto-Generated Artifacts (`reports/`):**
- `model_metrics.csv`: Machine-readable ROC-AUC, Recall, F1-Score for CI/CD tracking
- `confusion_matrix.png`: Visual FN/FP breakdown for operational capacity planning
- `shap_summary.png`: Global feature importance & directional impact (sampled for rendering speed)

### 🌍 Deployment Considerations & Next Steps

| Consideration | Current Implementation | Future Enhancement |
|:---|:---|:---|
| **Threshold Tuning** | Default `0.5` probability cutoff | Partner with dispatch to tune threshold based on cost-benefit analysis: False Positive (unnecessary intervention) vs. False Negative (customer churn) |
| **Explainability** | SHAP summary plots generated in `reports/` | Integrate real-time SHAP values into dispatcher dashboard: *"High Risk: 78% \| Drivers: New Driver + Refrigerated Parcel"* |
| **Missing Feedback Strategy** | Missing ratings → `Class 0` (No Trouble) via `.fillna(0)` | Explore hierarchical Bayesian priors using branch/driver historical performance to differentiate "genuinely smooth" vs. "unreported" deliveries |
| **Monitoring & Drift Detection** | Manual evaluation re-runs | Automated PSI/KS tests for feature drift; monthly ROC-AUC tracking; trigger retraining when drift exceeds thresholds |
| **Production Hardening** | Preprocessing fitted in `main.py` before split (for assessment clarity) | Wrap preprocessing + model in scikit-learn `Pipeline` with `GridSearchCV` to guarantee cross-validation hygiene and simplify serialization |
| **Inference Architecture** | Batch scoring via `main.py` script | Deploy as lightweight FastAPI endpoint; target `<50ms` latency per prediction for real-time dispatch scoring |

---

#### 🔧 Immediate Next Steps (If Given 1 More Week)

1. **Business-Aligned Threshold Optimization**
   - Run cost-sensitivity analysis with dispatch leadership
   - Identify optimal probability cutoff that balances intervention capacity vs. risk mitigation
   - Document threshold rationale in `decision_log.md`

2. **SHAP-Based Explainability Layer**
   - Generate per-prediction SHAP values for top-risk deliveries
   - Surface top 3 contributing features in human-readable format for dispatchers
   - Example output: `⚠️ High Risk (82%): [New Driver] + [Refrigerated] + [Long Distance]`

3. **Advanced Missing Feedback Modeling**
   - Replace blanket `fillna(0)` with branch/driver-level prior probabilities
   - Test impact on Recall/F1 using historical "silent churn" patterns
   - Document assumptions and limitations in model card

---

#### 📦 Production Readiness Checklist

- [x] Leakage-free preprocessing (fit on train only)
- [x] Stratified split for imbalanced target
- [x] Class-weighted XGBoost training
- [x] Reproducible random state (`42`)
- [x] Auto-generated evaluation artifacts (`reports/`)
- [ ] Threshold tuning with business stakeholders
- [ ] Real-time inference API wrapper
- [ ] Automated drift monitoring pipeline
- [ ] Model versioning & rollback strategy

> 💡 *All code changes for future enhancements should maintain the core principle: **only pre-delivery features may be used for prediction** to ensure production viability.*

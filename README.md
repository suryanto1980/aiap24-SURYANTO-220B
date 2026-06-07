## AIAP Batch 24 Technical Assessment - Exploratory Data Analysis (EDA)
**Candidate Name:** [Suryanto]  
**Date:** [05/06/2026]  

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

## 🚀 Execution Instructions & Parameter Configuration

### Pre-requisite
install package in the requirements.txt
$ pip install -r requirements.txt

### Running the pipeline
Execute the end-to-end pipeline via the provided shell script:
$ bash ./run.sh

Alternatively, run the orchestrator directly:
$ python3 src/main.py --db_path data/delivery.db

## 🔄 Pipeline Logical Flow & Architecture


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


## Fraud Detection — E-commerce + Bank Transactions

This repository implements an end-to-end workflow for **fraud detection**:

- E-commerce transactions (`Fraud_Data.csv`) with **IP→country geolocation** enrichment
- Bank transactions (`creditcard.csv`) with PCA features

The project is structured to match the tasks: **EDA → preprocessing/feature engineering → modeling → explainability**.

## Repository structure

Key folders:

- `data/raw/`: put raw CSVs here (gitignored)
- `data/processed/`: generated artifacts (gitignored)
- `src/`: reusable code
- `scripts/`: runnable CLI scripts
- `notebooks/`: EDA / feature engineering notebooks
- `tests/`: unit tests

## Setup

Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
```

## Where to put the data (important)

Place the files in **`data/raw/`** with these exact names:

- `data/raw/Fraud_Data.csv`
- `data/raw/IpAddress_to_Country.csv`
- `data/raw/creditcard.csv`

These are intentionally **not committed** (see `.gitignore`).

## Task 1 — Preprocessing & Feature Engineering

Run Task 1 preprocessing (creates engineered datasets + encoders/scalers + train/test splits + SMOTE on train only):

```bash
python -m scripts.task1_preprocess --dataset all
```

Outputs are written to `data/processed/`, including:

- `fraud_engineered.parquet` and `creditcard_clean.parquet`
- `*_preprocessor.joblib`
- `*_X_train.npz`, `*_y_train.npy`, `*_X_train_smote.npz`, ...
- `*_task1_metadata.json` (includes class distributions before/after SMOTE)

## Task 2 — Modeling

Train baseline + ensemble models with Stratified K-Fold CV and evaluate using AUC-PR / F1 / confusion matrix:

```bash
python -m scripts.task2_train --dataset all
```

Outputs:

- `models/task2_<dataset>_<model>.joblib`
- `reports/task2_<dataset>_results.json`

## Task 3 — Explainability (SHAP)

Install SHAP (optional dependency):

```bash
pip install -r requirements-task3.txt
```

Then run the notebook:

- `notebooks/shap-explainability.ipynb`

Note: Task 3 expects Task 2 models to exist. Run Task 2 first if needed.
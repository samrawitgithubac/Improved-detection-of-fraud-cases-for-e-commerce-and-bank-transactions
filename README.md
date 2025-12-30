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

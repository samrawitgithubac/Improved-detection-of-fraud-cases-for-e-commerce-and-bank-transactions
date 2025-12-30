from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import dump
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .ip_utils import attach_country_by_ip_range


@dataclass(frozen=True)
class FraudTask1Paths:
    raw_fraud_csv: Path
    raw_ip_country_csv: Path
    out_dir: Path


def clean_fraud_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()

    # Correct dtypes
    for col in ["signup_time", "purchase_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Basic missing handling (documented in notebook/README)
    # - drop if target missing or timestamps missing (can't engineer time features)
    df = df.dropna(subset=["class", "signup_time", "purchase_time"])

    # numeric imputations
    for col in ["age", "purchase_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # categorical imputations
    for col in ["source", "browser", "sex", "device_id", "user_id"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("Unknown")

    # keep ip_address as-is; will be converted for geo join
    return df


def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for Fraud_Data.csv (after geo enrichment).

    Includes:
    - hour_of_day, day_of_week
    - time_since_signup (seconds)
    - user transaction velocity counts (1h, 24h), computed on purchase_time
    - user/device aggregate signals
    """
    df = df.copy()

    df["hour_of_day"] = df["purchase_time"].dt.hour.astype("int16")
    df["day_of_week"] = df["purchase_time"].dt.dayofweek.astype("int16")
    df["time_since_signup_sec"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds().clip(lower=0)

    # Sort for time-window features
    df = df.sort_values(["user_id", "purchase_time"])

    # Rolling window counts per user (include current tx then subtract 1 to make "prior activity")
    def _user_rolling_counts(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g = g.set_index("purchase_time", drop=False)
        c1h = g["purchase_value"].rolling("1h").count().astype("int32") - 1
        c24h = g["purchase_value"].rolling("24h").count().astype("int32") - 1
        g["user_txn_count_1h"] = c1h.clip(lower=0).to_numpy()
        g["user_txn_count_24h"] = c24h.clip(lower=0).to_numpy()
        return g.reset_index(drop=True)

    df = df.groupby("user_id", group_keys=False).apply(_user_rolling_counts)

    # Simple aggregates (computed on full dataset; acceptable for Task-1 EDA/features,
    # but for strict leakage control you can recompute on train only in modeling stage)
    df["user_total_txns"] = df.groupby("user_id")["purchase_time"].transform("count").astype("int32")
    df["user_unique_devices"] = df.groupby("user_id")["device_id"].transform("nunique").astype("int16")
    df["device_unique_users"] = df.groupby("device_id")["user_id"].transform("nunique").astype("int16")

    return df


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, list[str], list[str]]:
    feature_cols = [
        # numeric
        "purchase_value",
        "age",
        "hour_of_day",
        "day_of_week",
        "time_since_signup_sec",
        "user_txn_count_1h",
        "user_txn_count_24h",
        "user_total_txns",
        "user_unique_devices",
        "device_unique_users",
        # categorical
        "source",
        "browser",
        "sex",
        "country",
    ]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    num_cols = [
        "purchase_value",
        "age",
        "hour_of_day",
        "day_of_week",
        "time_since_signup_sec",
        "user_txn_count_1h",
        "user_txn_count_24h",
        "user_total_txns",
        "user_unique_devices",
        "device_unique_users",
    ]
    cat_cols = ["source", "browser", "sex", "country"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre, feature_cols, num_cols + cat_cols


def class_distribution(y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y)
    n = float(len(y))
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return {"n": n, "n_pos": pos, "n_neg": neg, "pos_rate": (pos / n) if n else 0.0}


def run_task1_fraud(paths: FraudTask1Paths, *, test_size: float = 0.2, random_state: int = 42) -> Dict:
    out_dir = Path(paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fraud = pd.read_csv(paths.raw_fraud_csv)
    ip_map = pd.read_csv(paths.raw_ip_country_csv)

    fraud = clean_fraud_df(fraud)
    fraud = attach_country_by_ip_range(fraud, ip_map, out_col="country")
    fraud = engineer_fraud_features(fraud)

    y = fraud["class"].astype(int).to_numpy()
    pre, _, _ = build_preprocessor(fraud)

    X = fraud.drop(columns=["class"])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train_enc = pre.fit_transform(X_train)
    X_test_enc = pre.transform(X_test)

    # SMOTE only on train
    smote = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_enc, y_train)

    # Persist artifacts
    dump(pre, out_dir / "fraud_preprocessor.joblib")

    sparse.save_npz(out_dir / "fraud_X_train.npz", sparse.csr_matrix(X_train_enc))
    sparse.save_npz(out_dir / "fraud_X_test.npz", sparse.csr_matrix(X_test_enc))
    np.save(out_dir / "fraud_y_train.npy", y_train)
    np.save(out_dir / "fraud_y_test.npy", y_test)

    sparse.save_npz(out_dir / "fraud_X_train_smote.npz", sparse.csr_matrix(X_train_sm))
    np.save(out_dir / "fraud_y_train_smote.npy", y_train_sm)

    # Also save engineered (pre-encoded) dataset for EDA/debugging
    fraud.to_parquet(out_dir / "fraud_engineered.parquet", index=False)

    meta = {
        "dataset": "Fraud_Data",
        "raw_rows_after_cleaning": int(len(fraud)),
        "split": {"test_size": test_size, "random_state": random_state},
        "class_dist_train": class_distribution(y_train),
        "class_dist_test": class_distribution(y_test),
        "class_dist_train_smote": class_distribution(y_train_sm),
    }
    (out_dir / "fraud_task1_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta



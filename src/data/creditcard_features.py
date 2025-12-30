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
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CreditcardTask1Paths:
    raw_creditcard_csv: Path
    out_dir: Path


def clean_creditcard_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()

    # Ensure required columns exist
    required = {"Class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # numeric coercion for safety
    for col in df.columns:
        if col == "Class":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Class"] = pd.to_numeric(df["Class"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Class"])
    df["Class"] = df["Class"].astype(int)

    # Drop rows with too many NaNs (rare in this dataset, but safe)
    df = df.dropna(axis=0, how="any")
    return df


def build_creditcard_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, list[str]]:
    # Typical dataset columns: Time, V1..V28, Amount, Class
    feature_cols = [c for c in df.columns if c != "Class"]
    if "Amount" not in feature_cols:
        raise ValueError("Expected column 'Amount' in creditcard dataset.")

    # Scale Time and Amount; keep PCA components as-is
    scale_cols = [c for c in ["Time", "Amount"] if c in feature_cols]
    pass_cols = [c for c in feature_cols if c not in scale_cols]

    pre = ColumnTransformer(
        transformers=[
            ("scale", Pipeline([("scaler", StandardScaler())]), scale_cols),
            ("pass", "passthrough", pass_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre, feature_cols


def class_distribution(y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y)
    n = float(len(y))
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return {"n": n, "n_pos": pos, "n_neg": neg, "pos_rate": (pos / n) if n else 0.0}


def run_task1_creditcard(
    paths: CreditcardTask1Paths, *, test_size: float = 0.2, random_state: int = 42
) -> Dict:
    out_dir = Path(paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(paths.raw_creditcard_csv)
    df = clean_creditcard_df(df)

    y = df["Class"].astype(int).to_numpy()
    X = df.drop(columns=["Class"])

    pre, _ = build_creditcard_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train_enc = pre.fit_transform(X_train)
    X_test_enc = pre.transform(X_test)

    smote = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_enc, y_train)

    dump(pre, out_dir / "creditcard_preprocessor.joblib")

    sparse.save_npz(out_dir / "creditcard_X_train.npz", sparse.csr_matrix(X_train_enc))
    sparse.save_npz(out_dir / "creditcard_X_test.npz", sparse.csr_matrix(X_test_enc))
    np.save(out_dir / "creditcard_y_train.npy", y_train)
    np.save(out_dir / "creditcard_y_test.npy", y_test)

    sparse.save_npz(out_dir / "creditcard_X_train_smote.npz", sparse.csr_matrix(X_train_sm))
    np.save(out_dir / "creditcard_y_train_smote.npy", y_train_sm)

    df.to_parquet(out_dir / "creditcard_clean.parquet", index=False)

    meta = {
        "dataset": "creditcard",
        "raw_rows_after_cleaning": int(len(df)),
        "split": {"test_size": test_size, "random_state": random_state},
        "class_dist_train": class_distribution(y_train),
        "class_dist_test": class_distribution(y_test),
        "class_dist_train_smote": class_distribution(y_train_sm),
    }
    (out_dir / "creditcard_task1_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta



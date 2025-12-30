from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.creditcard_features import clean_creditcard_df
from src.data.fraud_features import clean_fraud_df, engineer_fraud_features
from src.data.ip_utils import attach_country_by_ip_range
from src.modeling.metrics import EvalResult, evaluate_binary_classifier


DatasetName = Literal["fraud", "creditcard"]


@dataclass(frozen=True)
class Task2Paths:
    raw_dir: Path
    reports_dir: Path
    models_dir: Path


def _build_fraud_preprocessor_dense(df: pd.DataFrame) -> Tuple[ColumnTransformer, list[str]]:
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

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for fraud modeling: {missing}")

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre, feature_cols


def _build_creditcard_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, list[str]]:
    feature_cols = [c for c in df.columns if c != "Class"]
    scale_cols = [c for c in ["Time", "Amount"] if c in feature_cols]
    pass_cols = [c for c in feature_cols if c not in scale_cols]

    pre = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scale_cols),
            ("pass", "passthrough", pass_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre, feature_cols


def load_fraud_dataframe(raw_dir: Path) -> pd.DataFrame:
    fraud_path = raw_dir / "Fraud_Data.csv"
    ip_path = raw_dir / "IpAddress_to_Country.csv"

    fraud = pd.read_csv(fraud_path)
    ip_map = pd.read_csv(ip_path)

    fraud = clean_fraud_df(fraud)
    fraud = attach_country_by_ip_range(fraud, ip_map, out_col="country")
    fraud = engineer_fraud_features(fraud)
    return fraud


def load_creditcard_dataframe(raw_dir: Path) -> pd.DataFrame:
    cc_path = raw_dir / "creditcard.csv"
    cc = pd.read_csv(cc_path)
    cc = clean_creditcard_df(cc)
    return cc


def _make_models(random_state: int = 42) -> Dict[str, Any]:
    baseline = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        n_jobs=None,
        class_weight="balanced",
    )
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    return {"logreg": baseline, "random_forest": rf}


def _predict_proba_pos(estimator, X) -> np.ndarray:
    # Both LogisticRegression and RandomForest have predict_proba
    proba = estimator.predict_proba(X)
    return proba[:, 1]


def _cv_scores(
    pipeline: ImbPipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs: list[float] = []
    f1s: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_va_proba = _predict_proba_pos(pipeline, X_va)
        res = evaluate_binary_classifier(y_va, y_va_proba)
        aucs.append(res.auc_pr)
        f1s.append(res.f1)

    return {
        "n_splits": n_splits,
        "auc_pr_mean": float(np.mean(aucs)),
        "auc_pr_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
    }


def train_and_evaluate_task2(
    dataset: DatasetName,
    paths: Task2Paths,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_splits: int = 5,
    sample_frac: float | None = None,
) -> Dict[str, Any]:
    """
    Task 2:
    - stratified train/test split
    - baseline: Logistic Regression
    - ensemble: Random Forest
    - train with SMOTE on training folds only (via imblearn Pipeline)
    - evaluate via AUC-PR, F1, confusion matrix (test set) and CV mean±std
    """
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "fraud":
        df = load_fraud_dataframe(paths.raw_dir)
        y = df["class"].astype(int).to_numpy()
        X = df.drop(columns=["class"])
        pre, _ = _build_fraud_preprocessor_dense(df)
    elif dataset == "creditcard":
        df = load_creditcard_dataframe(paths.raw_dir)
        y = df["Class"].astype(int).to_numpy()
        X = df.drop(columns=["Class"])
        pre, _ = _build_creditcard_preprocessor(df)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if sample_frac is not None and 0 < sample_frac < 1:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=int(len(X) * sample_frac), replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    models = _make_models(random_state=random_state)
    results: Dict[str, Any] = {
        "dataset": dataset,
        "n_rows": int(len(X)),
        "split": {"test_size": test_size, "random_state": random_state},
        "cv_splits": cv_splits,
        "models": {},
    }

    for name, model in models.items():
        pipe = ImbPipeline(
            steps=[
                ("preprocess", pre),
                ("smote", SMOTE(random_state=random_state)),
                ("model", model),
            ]
        )

        cv = _cv_scores(pipe, X_train, y_train, n_splits=cv_splits, random_state=random_state)

        pipe.fit(X_train, y_train)
        y_proba = _predict_proba_pos(pipe, X_test)
        test_res: EvalResult = evaluate_binary_classifier(y_test, y_proba)

        # Save model pipeline (reproducible end-to-end)
        model_path = paths.models_dir / f"task2_{dataset}_{name}.joblib"
        dump(pipe, model_path)

        results["models"][name] = {
            "cv": cv,
            "test": test_res.as_dict(),
            "model_path": str(model_path).replace("\\", "/"),
        }

    report_path = paths.reports_dir / f"task2_{dataset}_results.json"
    report_path.write_text(json.dumps(results, indent=2))
    return results



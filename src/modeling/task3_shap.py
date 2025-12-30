from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split

from src.data.creditcard_features import clean_creditcard_df
from src.data.fraud_features import clean_fraud_df, engineer_fraud_features
from src.data.ip_utils import attach_country_by_ip_range


@dataclass(frozen=True)
class Task3Paths:
    raw_dir: Path
    reports_dir: Path
    models_dir: Path


def _load_fraud(raw_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    fraud = pd.read_csv(raw_dir / "Fraud_Data.csv")
    ip_map = pd.read_csv(raw_dir / "IpAddress_to_Country.csv")
    fraud = clean_fraud_df(fraud)
    fraud = attach_country_by_ip_range(fraud, ip_map, out_col="country")
    fraud = engineer_fraud_features(fraud)
    y = fraud["class"].astype(int).to_numpy()
    X = fraud.drop(columns=["class"])
    return X, y


def _load_creditcard(raw_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    cc = pd.read_csv(raw_dir / "creditcard.csv")
    cc = clean_creditcard_df(cc)
    y = cc["Class"].astype(int).to_numpy()
    X = cc.drop(columns=["Class"])
    return X, y


def pick_best_model_name_from_task2_report(report_path: Path) -> Optional[str]:
    """
    Return best model key by test AUC-PR from task2 report JSON, or None.
    """
    if not report_path.exists():
        return None
    data = json.loads(report_path.read_text())
    models: Dict[str, Any] = data.get("models", {})
    if not models:
        return None
    best = None
    best_auc = -1.0
    for name, m in models.items():
        auc = float(m.get("test", {}).get("auc_pr", -1.0))
        if auc > best_auc:
            best_auc = auc
            best = name
    return best


def _get_feature_names(preprocessor) -> list[str]:
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return []


def build_shap_explainer(model, X_background: np.ndarray, feature_names: list[str]):
    try:
        import shap  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SHAP is not installed. Install Task 3 deps via: pip install -r requirements-task3.txt"
        ) from e

    # Use generic Explainer (picks Tree/Linear/etc.) when possible
    try:
        return shap.Explainer(model, X_background, feature_names=feature_names)
    except Exception:
        return shap.Explainer(model, X_background)


def explain_task3(
    *,
    dataset: str,
    paths: Task3Paths,
    model_name: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    background_size: int = 200,
    explain_size: int = 2000,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Load the best Task 2 model pipeline (or a requested model), then compute SHAP values
    on a sample of the test set and identify TP/FP/FN examples for local explanations.

    Returns lightweight metadata (not SHAP arrays).
    """
    if dataset not in {"fraud", "creditcard"}:
        raise ValueError("dataset must be 'fraud' or 'creditcard'")

    report_path = paths.reports_dir / f"task2_{dataset}_results.json"
    if model_name is None:
        model_name = pick_best_model_name_from_task2_report(report_path) or "random_forest"

    model_path = paths.models_dir / f"task2_{dataset}_{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing Task 2 model artifact: {model_path}. Run Task 2 first:\n"
            f"  python -m scripts.task2_train --dataset {dataset}"
        )

    pipe = load(model_path)
    pre = pipe.named_steps["preprocess"]
    clf = pipe.named_steps["model"]

    if dataset == "fraud":
        X, y = _load_fraud(paths.raw_dir)
    else:
        X, y = _load_creditcard(paths.raw_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Transform to model feature space (dense)
    X_train_t = pre.transform(X_train)
    X_test_t = pre.transform(X_test)

    # Background and explain samples (keep memory bounded)
    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(len(X_train_t), size=min(background_size, len(X_train_t)), replace=False)
    ex_idx = rng.choice(len(X_test_t), size=min(explain_size, len(X_test_t)), replace=False)

    X_bg = np.asarray(X_train_t)[bg_idx]
    X_ex = np.asarray(X_test_t)[ex_idx]
    y_ex = y_test[ex_idx]

    proba = clf.predict_proba(X_ex)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    # Identify one TP / FP / FN (if present)
    tp = np.where((y_ex == 1) & (y_pred == 1))[0]
    fp = np.where((y_ex == 0) & (y_pred == 1))[0]
    fn = np.where((y_ex == 1) & (y_pred == 0))[0]

    feature_names = _get_feature_names(pre)
    explainer = build_shap_explainer(clf, X_bg, feature_names)

    # Compute SHAP on the explain sample (for summary plot)
    shap_values = explainer(X_ex)

    examples = {
        "tp_index": int(tp[0]) if len(tp) else None,
        "fp_index": int(fp[0]) if len(fp) else None,
        "fn_index": int(fn[0]) if len(fn) else None,
    }

    return {
        "dataset": dataset,
        "model_name": model_name,
        "model_path": str(model_path).replace("\\", "/"),
        "n_test_sample_explained": int(len(X_ex)),
        "examples": examples,
        # return shap objects for notebook usage
        "shap_values": shap_values,
        "X_ex": X_ex,
        "feature_names": feature_names,
        "explainer": explainer,
        "y_ex": y_ex,
        "y_pred": y_pred,
        "proba": proba,
    }



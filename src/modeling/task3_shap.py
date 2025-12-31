from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd
from joblib import load
import shap
from sklearn.ensemble import RandomForestClassifier

from src.modeling.task2_train import (
    DatasetName,
    load_fraud_dataframe,
    load_creditcard_dataframe,
    _build_fraud_preprocessor_dense,
    _build_creditcard_preprocessor,
    _predict_proba_pos,
)


@dataclass(frozen=True)
class Task3Paths:
    raw_dir: Path
    reports_dir: Path
    models_dir: Path


def _get_best_model_name(dataset: DatasetName, reports_dir: Path) -> str:
    """Determine best model based on Task 2 results (highest test AUC-PR)."""
    results_path = reports_dir / f"task2_{dataset}_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Task 2 results not found: {results_path}. Run Task 2 first."
        )
    
    with open(results_path) as f:
        results = json.load(f)
    
    best_name = None
    best_auc_pr = -1.0
    
    for name, model_results in results["models"].items():
        auc_pr = model_results["test"]["auc_pr"]
        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_name = name
    
    if best_name is None:
        raise ValueError("No models found in Task 2 results")
    
    return best_name


def _get_feature_names(pipeline, X_sample: pd.DataFrame) -> list[str]:
    """Extract feature names after preprocessing."""
    preprocessor = pipeline.named_steps["preprocess"]
    X_transformed = preprocessor.transform(X_sample.head(1))
    
    # Try to get feature names using sklearn's get_feature_names_out
    try:
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out().tolist()
        elif hasattr(preprocessor, "transformers_"):
            # Manual extraction for ColumnTransformer
            feature_names = []
            for name, transformer, cols in preprocessor.transformers_:
                if name == "drop":
                    continue
                if hasattr(transformer, "get_feature_names_out"):
                    input_cols = cols if isinstance(cols, list) else [cols] if isinstance(cols, str) else list(cols)
                    feature_names.extend(transformer.get_feature_names_out(input_cols))
                else:
                    # Passthrough or StandardScaler - use column names
                    if isinstance(cols, list):
                        feature_names.extend(cols)
                    elif isinstance(cols, str):
                        feature_names.append(cols)
                    else:
                        feature_names.extend(list(cols))
        else:
            raise AttributeError("Cannot extract feature names")
    except (AttributeError, ValueError):
        # Fallback: use generic names
        if hasattr(X_transformed, "shape"):
            n_features = X_transformed.shape[1]
        elif hasattr(X_transformed, "toarray"):
            n_features = X_transformed.toarray()[0].shape[0]
        else:
            n_features = len(X_transformed) if hasattr(X_transformed, "__len__") else 0
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return feature_names


def _extract_builtin_feature_importance(pipeline) -> Tuple[np.ndarray, list[str]]:
    """Extract built-in feature importance from ensemble model."""
    model = pipeline.named_steps["model"]
    
    if hasattr(model, "feature_importances_"):
        # Random Forest
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Logistic Regression - use absolute coefficients
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError(f"Model {type(model)} does not have feature_importances_ or coef_")
    
    return importances


def _find_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float = 0.5,
    max_samples: int = 1000,
) -> Dict[str, int | None]:
    """Find indices for TP, FP, FN examples."""
    y_pred_binary = (y_proba >= threshold).astype(int)
    
    tp_mask = (y_true == 1) & (y_pred_binary == 1)
    fp_mask = (y_true == 0) & (y_pred_binary == 1)
    fn_mask = (y_true == 1) & (y_pred_binary == 0)
    
    # Limit search to first max_samples for efficiency
    search_range = min(len(y_true), max_samples)
    
    tp_indices = np.where(tp_mask[:search_range])[0]
    fp_indices = np.where(fp_mask[:search_range])[0]
    fn_indices = np.where(fn_mask[:search_range])[0]
    
    return {
        "tp_index": int(tp_indices[0]) if len(tp_indices) > 0 else None,
        "fp_index": int(fp_indices[0]) if len(fp_indices) > 0 else None,
        "fn_index": int(fn_indices[0]) if len(fn_indices) > 0 else None,
    }


def explain_task3(
    dataset: DatasetName,
    paths: Task3Paths,
    *,
    explain_size: int = 100,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Task 3: SHAP explainability for best Task 2 model.
    
    Returns:
        Dictionary with:
        - model_name: str
        - builtin_importance: np.ndarray
        - feature_names: list[str]
        - shap_values: shap.Explanation
        - examples: dict with tp_index, fp_index, fn_index
        - n_test_sample_explained: int
    """
    # Determine best model
    best_model_name = _get_best_model_name(dataset, paths.reports_dir)
    model_path = paths.models_dir / f"task2_{dataset}_{best_model_name}.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load pipeline
    pipeline = load(model_path)
    
    # Load data (same split as Task 2)
    if dataset == "fraud":
        df = load_fraud_dataframe(paths.raw_dir)
        y = df["class"].astype(int).to_numpy()
        X = df.drop(columns=["class"])
    elif dataset == "creditcard":
        df = load_creditcard_dataframe(paths.raw_dir)
        y = df["Class"].astype(int).to_numpy()
        X = df.drop(columns=["Class"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Recreate same train/test split as Task 2
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get feature names
    feature_names = _get_feature_names(pipeline, X_train)
    
    # Extract built-in feature importance
    builtin_importance = _extract_builtin_feature_importance(pipeline)
    
    # Prepare data for SHAP (use transformed data)
    # Sample from test set for SHAP (for efficiency)
    rng = np.random.default_rng(random_state)
    n_sample = min(explain_size, len(X_test))
    sample_idx = rng.choice(len(X_test), size=n_sample, replace=False)
    X_test_sample = X_test.iloc[sample_idx]
    y_test_sample = y_test[sample_idx]
    
    # Transform sample
    X_test_transformed = pipeline.named_steps["preprocess"].transform(X_test_sample)
    
    # Convert sparse to dense if needed
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    
    # Get model (without SMOTE step for SHAP)
    model = pipeline.named_steps["model"]
    
    # Create SHAP explainer
    # For tree models, use TreeExplainer; for linear, use LinearExplainer
    if isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
    else:
        # For linear models, use a sample of background data
        X_train_transformed = pipeline.named_steps["preprocess"].transform(X_train.head(100))
        if hasattr(X_train_transformed, "toarray"):
            X_train_transformed = X_train_transformed.toarray()
        explainer = shap.LinearExplainer(model, X_train_transformed)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_test_transformed)
    
    # For binary classification, shap_values might be a list [class_0, class_1]
    # Use class_1 (fraud) values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    # Get predictions for finding examples
    y_proba_sample = _predict_proba_pos(pipeline, X_test_sample)
    y_pred_sample = pipeline.predict(X_test_sample)
    
    # Find TP, FP, FN examples (using original sample indices)
    examples = _find_examples(y_test_sample, y_pred_sample, y_proba_sample)
    
    # Create SHAP Explanation object for better visualization
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        data=X_test_transformed,
        feature_names=feature_names[:shap_values.shape[1]] if len(feature_names) == shap_values.shape[1] else [f"feature_{i}" for i in range(shap_values.shape[1])],
    )
    
    return {
        "model_name": best_model_name,
        "builtin_importance": builtin_importance,
        "feature_names": feature_names,
        "shap_values": shap_explanation,
        "examples": examples,
        "n_test_sample_explained": n_sample,
        "y_test_sample": y_test_sample,
        "y_proba_sample": y_proba_sample,
    }


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)


@dataclass(frozen=True)
class EvalResult:
    auc_pr: float
    f1: float
    confusion_matrix: list[list[int]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "auc_pr": float(self.auc_pr),
            "f1": float(self.f1),
            "confusion_matrix": self.confusion_matrix,
        }


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float = 0.5,
) -> EvalResult:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    auc_pr = average_precision_score(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).astype(int).tolist()
    return EvalResult(auc_pr=auc_pr, f1=f1, confusion_matrix=cm)


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Optional helper: find threshold that maximizes F1 on a validation set.
    (Not used by default in Task 2 metrics, but useful for analysis.)
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # thresholds has length n-1; precision/recall have length n
    f1s = (2 * precision[:-1] * recall[:-1]) / np.clip((precision[:-1] + recall[:-1]), 1e-12, None)
    best_idx = int(np.nanargmax(f1s)) if len(f1s) else 0
    return float(thresholds[best_idx]) if len(thresholds) else 0.5



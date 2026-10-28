"""
Model evaluation module.
Calculates comprehensive metrics emphasizing failure recall, false negatives,
ROC-AUC, Precision-Recall AUC, confusion matrix, and multi-model comparisons.
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import (
    ARTIFACTS_DIR,
    CONFUSION_MATRIX_PATH,
    FAILURE_MODE_METRICS_PATH,
    METRICS_PATH,
)


def evaluate_binary_classifier(
    model: Any,
    X_test: Any,
    y_test: np.ndarray,
    model_name: str = "CandidateModel",
) -> Dict[str, Any]:
    """
    Evaluates a binary classifier on test data, capturing failure-specific recall,
    precision, false negative rate, ROC-AUC, and PR-AUC.
    """
    y_pred = model.predict(X_test)

    # Probabilities for class 1 if supported
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            if proba.shape[1] > 1:
                y_prob = proba[:, 1]
            else:
                y_prob = proba[:, 0]
        except Exception:
            y_prob = None
    elif hasattr(model, "decision_function"):
        try:
            dfunc = model.decision_function(X_test)
            # Min-max scale decision function as pseudo-prob for AUC
            y_prob = (dfunc - dfunc.min()) / (dfunc.max() - dfunc.min() + 1e-9)
        except Exception:
            y_prob = None

    # Base Metrics
    acc = float(accuracy_score(y_test, y_pred))
    # zero_division=0 handles models predicting only majority class
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    total_actual_failures = tp + fn
    total_actual_normals = tn + fp

    fnr = float(fn / total_actual_failures) if total_actual_failures > 0 else 0.0
    fpr = float(fp / total_actual_normals) if total_actual_normals > 0 else 0.0

    # AUC metrics
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    if y_prob is not None and len(np.unique(y_test)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_test, y_prob))
            pr_auc = float(average_precision_score(y_test, y_prob))
        except Exception:
            roc_auc = None
            pr_auc = None

    # Classification report
    clf_report = classification_report(
        y_test, y_pred, target_names=["Normal", "Failure"], output_dict=True, zero_division=0
    )

    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else "N/A",
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "false_negative_rate": round(fnr, 4),
            "false_positive_rate": round(fpr, 4),
        },
        "classification_report": clf_report,
        "total_test_samples": int(len(y_test)),
        "total_test_failures": total_actual_failures,
    }

    return metrics


def evaluate_failure_modes(
    mode_models: Dict[str, Any],
    X_test: Any,
    Y_modes_test: Any,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates individual failure mode models (TWF, HDF, PWF, OSF, RNF).
    Safely handles rare classes where positive instances in test set are minimal.
    """
    mode_metrics: Dict[str, Dict[str, Any]] = {}

    for mode, model in mode_models.items():
        if mode not in Y_modes_test.columns:
            continue

        y_true = Y_modes_test[mode].values
        y_pred = model.predict(X_test)

        pos_count = int(np.sum(y_true))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))

        mode_metrics[mode] = {
            "positive_count_in_test": pos_count,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }

    return mode_metrics


def save_metrics(metrics: Dict[str, Any], path: Path = METRICS_PATH) -> None:
    """Save metrics dictionary as JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def load_metrics(path: Path = METRICS_PATH) -> Optional[Dict[str, Any]]:
    """Load metrics dictionary from JSON artifact if exists."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

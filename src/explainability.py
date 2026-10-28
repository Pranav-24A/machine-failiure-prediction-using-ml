"""
Model explainability and feature importance module.
Implements Random Forest tree-based importance, permutation importance,
and local contribution breakdowns for individual predictions.
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config import ARTIFACTS_DIR, FEATURE_IMPORTANCE_PATH


def compute_global_feature_importance(
    model: Any,
    feature_names: List[str],
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes global feature importances using native tree importance or permutation importance.
    Returns sorted list of features and their relative weights.
    """
    importance_dict: Dict[str, float] = {}

    # Method 1: Native Gini / MDI importance
    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
        for name, score in zip(feature_names, raw_importances):
            importance_dict[name] = float(score)
    # Method 2: Permutation importance if validation data provided
    elif X_val is not None and y_val is not None:
        perm_res = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42)
        for name, score in zip(feature_names, perm_res.importances_mean):
            importance_dict[name] = float(max(score, 0.0))
    # Method 3: Linear model coefficients
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_[0])
        total = np.sum(coefs) + 1e-9
        norm_coefs = coefs / total
        for name, score in zip(feature_names, norm_coefs):
            importance_dict[name] = float(score)

    # Sort descending
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    result = {
        "features": [item[0] for item in sorted_features],
        "scores": [round(item[1], 4) for item in sorted_features],
        "normalized_percentage": [
            round((item[1] / (sum(importance_dict.values()) + 1e-9)) * 100, 2)
            for item in sorted_features
        ],
    }

    return result


def compute_local_feature_contribution(
    model: Any,
    preprocessor: Any,
    feature_names: List[str],
    input_row: pd.DataFrame,
    baseline_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Calculates local contribution/deviation for an individual prediction instance.
    Compares the instance's engineered features against training means and weights
    by global feature importance to explain which sensor readings drove the risk score.
    """
    # Transform row
    X_trans = preprocessor.transform(input_row)

    contributions = []

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    # For each transformed feature, measure deviation from zero (StandardScaler mean = 0)
    # in scaled space, absolute z-score * feature_importance provides impact measure
    row_values = X_trans[0]

    for idx, (feat_name, val, imp) in enumerate(zip(feature_names, row_values, importances)):
        impact = float(abs(val) * imp)
        direction = "Elevated" if val > 0.5 else ("Depressed" if val < -0.5 else "Nominal")
        contributions.append({
            "feature": feat_name,
            "transformed_value": round(float(val), 3),
            "importance_weight": round(float(imp), 4),
            "impact_score": round(impact, 4),
            "operating_state": direction,
        })

    # Sort by impact score descending
    contributions.sort(key=lambda x: x["impact_score"], reverse=True)
    return contributions


def save_feature_importance(importance: Dict[str, Any], path: Path = FEATURE_IMPORTANCE_PATH) -> None:
    """Save feature importance JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(importance, f, indent=4)


def load_feature_importance(path: Path = FEATURE_IMPORTANCE_PATH) -> Optional[Dict[str, Any]]:
    """Load feature importance JSON artifact."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

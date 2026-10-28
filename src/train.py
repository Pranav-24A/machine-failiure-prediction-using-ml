"""
Model training and selection pipeline.
Implements leak-free stratified splits, baseline comparisons (Dummy, Logistic Regression),
tree ensembles (Random Forest, HistGradientBoosting), multi-label failure mode models,
and artifact persistence.
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from config import (
    ARTIFACTS_DIR,
    COMPARISON_PATH,
    CONFUSION_MATRIX_PATH,
    CV_FOLDS,
    FAILURE_MODE_METRICS_PATH,
    FAILURE_MODE_MODEL_PATH,
    FAILURE_MODES,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
)
from src.data_loader import get_feature_target_split, load_dataset
from src.evaluate import evaluate_binary_classifier, evaluate_failure_modes, save_metrics
from src.explainability import compute_global_feature_importance, save_feature_importance
from src.preprocessing import build_preprocessing_pipeline, get_feature_names_out, save_preprocessor


def train_and_evaluate_all(
    dataset_path: Path = None,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Executes the complete training workflow:
    1. Loads dataset
    2. Stratified train/val/test split
    3. Fits preprocessing pipeline on training data only
    4. Evaluates Baseline (Dummy, LogisticRegression) and Candidates (RandomForest, HistGradientBoosting)
    5. Selects best model based on Validation F1-score & Recall
    6. Trains failure-mode models for TWF, HDF, PWF, OSF, RNF
    7. Evaluates on isolated test set and saves all models & artifacts
    """
    print("=" * 60)
    print("STARTING PREDICTIVE MAINTENANCE TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load data
    df = load_dataset(dataset_path)
    X, y, Y_modes = get_feature_target_split(df)
    print(f"Loaded {len(df)} records. Failures: {int(y.sum())} ({y.mean():.2%})")

    # 2. Stratified Train / Val / Test split
    # First split out test set (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )

    # Split remaining into Train (81.25% of 80% = 65% total) and Val (18.75% of 80% = 15% total)
    val_ratio_adjusted = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=random_state, stratify=y_train_val
    )

    print(f"Dataset Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 3. Fit preprocessing pipeline ONLY on training split
    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = get_feature_names_out(preprocessor)
    print(f"Engineered & Preprocessed {len(feature_names)} features: {feature_names}")

    # 4. Define candidate models
    candidate_models = {
        "Dummy (Stratified Baseline)": DummyClassifier(strategy="stratified", random_state=random_state),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=150, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced",
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    # 5. Evaluate models on validation split
    model_comparisons = []
    trained_models = {}
    best_model_name = None
    best_val_f1 = -1.0

    print("\nEvaluating Candidate Models on Validation Set...")
    for name, model in candidate_models.items():
        model.fit(X_train_proc, y_train)
        trained_models[name] = model

        val_metrics = evaluate_binary_classifier(model, X_val_proc, y_val.values, model_name=name)

        # Cross-validation score on training data for stability check
        cv_scores = cross_val_score(
            model, X_train_proc, y_train, cv=StratifiedKFold(n_splits=CV_FOLDS), scoring="f1"
        )

        comparison_entry = {
            "model": name,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1_score"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "train_cv_f1_mean": round(float(cv_scores.mean()), 4),
            "train_cv_f1_std": round(float(cv_scores.std()), 4),
            "fn_count": val_metrics["confusion_matrix"]["fn"],
            "fp_count": val_metrics["confusion_matrix"]["fp"],
        }
        model_comparisons.append(comparison_entry)

        print(
            f"  {name:30s} | Val Recall: {val_metrics['recall']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | Val F1: {val_metrics['f1_score']:.4f}"
        )

        # Selection criteria: prioritize F1 score on validation set (balances recall and precision)
        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            best_model_name = name

    print(f"\n--> Selected Champion Model: {best_model_name} (Validation F1 = {best_val_f1:.4f})")
    champion_model = trained_models[best_model_name]

    # 6. Final Evaluation on isolated Test Set
    test_metrics = evaluate_binary_classifier(
        champion_model, X_test_proc, y_test.values, model_name=best_model_name
    )
    print("\nFinal Test Set Performance for Champion Model:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']}")
    print(f"  PR-AUC:    {test_metrics['pr_auc']}")
    print(f"  Confusion Matrix: {test_metrics['confusion_matrix']}")

    # 7. Train Failure-Mode Classifiers
    failure_mode_models = {}
    failure_mode_evals = {}
    if Y_modes is not None:
        print("\nTraining Failure Mode Classifiers (TWF, HDF, PWF, OSF, RNF)...")
        # Split Y_modes with exact same train/val/test indices
        Y_modes_train = Y_modes.loc[X_train.index]
        Y_modes_test = Y_modes.loc[X_test.index]

        for mode in FAILURE_MODES:
            if mode in Y_modes_train.columns:
                y_mode_tr = Y_modes_train[mode].values
                # Check if positive samples exist in training set
                if np.sum(y_mode_tr) >= 3:
                    mode_clf = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=8,
                        class_weight="balanced",
                        random_state=random_state,
                    )
                    mode_clf.fit(X_train_proc, y_mode_tr)
                    failure_mode_models[mode] = mode_clf
                    print(f"  Fitted failure-mode classifier for: {mode}")
                else:
                    print(f"  Insufficient positive samples for {mode} ({np.sum(y_mode_tr)}). Skipping.")

        failure_mode_evals = evaluate_failure_modes(failure_mode_models, X_test_proc, Y_modes_test)

    # 8. Compute and Save Global Feature Importance
    feature_importance_data = compute_global_feature_importance(
        champion_model, feature_names, X_val=X_val_proc, y_val=y_val.values
    )

    # 9. Save all artifacts and models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    save_preprocessor(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(champion_model, MODEL_PATH)
    joblib.dump(failure_mode_models, FAILURE_MODE_MODEL_PATH)

    save_metrics(test_metrics, METRICS_PATH)
    save_feature_importance(feature_importance_data, FEATURE_IMPORTANCE_PATH)

    with open(COMPARISON_PATH, "w") as f:
        json.dump(model_comparisons, f, indent=4)

    with open(CONFUSION_MATRIX_PATH, "w") as f:
        json.dump(test_metrics["confusion_matrix"], f, indent=4)

    with open(FAILURE_MODE_METRICS_PATH, "w") as f:
        json.dump(failure_mode_evals, f, indent=4)

    print("\nAll models, preprocessing pipelines, and evaluation artifacts saved successfully.")
    return {
        "best_model_name": best_model_name,
        "test_metrics": test_metrics,
        "comparisons": model_comparisons,
        "feature_importance": feature_importance_data,
        "failure_mode_evals": failure_mode_evals,
    }


if __name__ == "__main__":
    train_and_evaluate_all()

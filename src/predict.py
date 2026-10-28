"""
Reusable inference module for machine failure prediction.
Validates sensor inputs, computes overall failure probabilities and risk levels,
identifies active failure modes, calculates local feature contributions,
and generates actionable maintenance recommendations.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from config import (
    DEFAULT_RISK_THRESHOLDS,
    FAILURE_MODE_MODEL_PATH,
    FAILURE_MODE_NAMES,
    FAILURE_MODES,
    FEATURE_COLUMNS,
    MODEL_PATH,
    NUMERICAL_COLUMNS,
    PREPROCESSOR_PATH,
    SENSOR_LIMITS,
    VALID_TYPES,
)
from src.explainability import compute_local_feature_contribution
from src.preprocessing import get_feature_names_out, load_preprocessor
from src.recommendations import generate_recommendations


class InferenceError(Exception):
    """Exception raised for errors during inference or prediction pipeline execution."""
    pass


class PredictiveMaintenancePredictor:
    """
    Production-grade inference engine for the Predictive Maintenance System.
    Loads and caches model artifacts, validates inputs, and produces structured outputs.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        failure_mode_model_path: Path = FAILURE_MODE_MODEL_PATH,
        preprocessor_path: Path = PREPROCESSOR_PATH,
        risk_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.model_path = model_path
        self.failure_mode_model_path = failure_mode_model_path
        self.preprocessor_path = preprocessor_path
        self.risk_thresholds = risk_thresholds or DEFAULT_RISK_THRESHOLDS

        self.model: Optional[Any] = None
        self.failure_mode_models: Optional[Dict[str, Any]] = None
        self.preprocessor: Optional[Any] = None
        self.feature_names: List[str] = []

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained models and fitted preprocessors from disk."""
        if not self.model_path.exists() or not self.preprocessor_path.exists():
            # Return early if models have not yet been trained
            return

        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            self.feature_names = get_feature_names_out(self.preprocessor)

            if self.failure_mode_model_path.exists():
                self.failure_mode_models = joblib.load(self.failure_mode_model_path)
            else:
                self.failure_mode_models = {}
        except Exception as e:
            raise InferenceError(f"Failed to load model artifacts: {e}")

    @property
    def is_ready(self) -> bool:
        """Returns True if model and preprocessor are loaded and ready for inference."""
        return self.model is not None and self.preprocessor is not None

    def validate_input_dict(self, input_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates individual input dictionary against required columns and physical sensor ranges.
        Returns (is_valid, error_messages).
        """
        errors = []

        # Check required fields
        for col in FEATURE_COLUMNS:
            if col not in input_dict:
                errors.append(f"Missing required parameter: '{col}'")

        if errors:
            return False, errors

        # Check categorical Type
        m_type = str(input_dict.get("Type", "")).strip().upper()
        if m_type not in VALID_TYPES:
            errors.append(f"Invalid Type '{m_type}'. Expected one of {VALID_TYPES}")

        # Check numeric limits
        for col, limits in SENSOR_LIMITS.items():
            val = input_dict.get(col)
            try:
                num_val = float(val)
                if num_val < limits["min"] or num_val > limits["max"]:
                    errors.append(
                        f"Value for '{col}' ({num_val}) is outside physical range [{limits['min']}, {limits['max']} {limits['unit']}]"
                    )
            except (ValueError, TypeError):
                errors.append(f"Value for '{col}' must be a valid number, got: '{val}'")

        return (len(errors) == 0), errors

    def _determine_risk_level(self, probability: float) -> str:
        """Classify probability into Low, Medium, or High Risk."""
        if probability < self.risk_thresholds["low"]:
            return "Low"
        elif probability < self.risk_thresholds["medium"]:
            return "Medium"
        else:
            return "High"

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute prediction for a single machine record.
        Returns structured results with status, probability, risk level, modes, and recommendations.
        """
        if not self.is_ready:
            self._load_artifacts()
            if not self.is_ready:
                raise InferenceError("Models are not trained or loaded. Please train models first.")

        # Validate input
        is_valid, validation_errors = self.validate_input_dict(input_data)
        if not is_valid:
            return {
                "is_valid": False,
                "validation_errors": validation_errors,
                "status": "Invalid Input",
                "failure_probability": 0.0,
                "risk_level": "Unknown",
                "predicted_modes": [],
                "contributing_features": [],
                "recommendations": ["Resolve input validation errors before re-evaluating machine."],
            }

        # Build clean single-row DataFrame with consistent column ordering
        row_df = pd.DataFrame([{col: input_data[col] for col in FEATURE_COLUMNS}])

        # Transform features
        X_trans = self.preprocessor.transform(row_df)

        # Predict primary failure
        prediction_bin = int(self.model.predict(X_trans)[0])

        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(X_trans)[0][1])
        else:
            proba = 1.0 if prediction_bin == 1 else 0.0

        risk_level = self._determine_risk_level(proba)
        status_label = "Failure Risk" if (prediction_bin == 1 or proba >= 0.50) else "Normal"

        # Predict individual failure modes
        predicted_modes = []
        mode_probabilities = {}
        if self.failure_mode_models:
            for mode, m_model in self.failure_mode_models.items():
                m_pred = int(m_model.predict(X_trans)[0])
                m_prob = float(m_model.predict_proba(X_trans)[0][1]) if hasattr(m_model, "predict_proba") else (1.0 if m_pred == 1 else 0.0)
                full_name = FAILURE_MODE_NAMES.get(mode, mode)
                mode_probabilities[full_name] = round(m_prob, 4)

                # Consider mode active if classified positive or probability high (>40%)
                if m_pred == 1 or m_prob >= 0.40:
                    predicted_modes.append(full_name)

        if not predicted_modes and status_label == "Failure Risk":
            predicted_modes.append("Unspecified / Combined Stress Failure")

        # Local feature contributions
        contributions = compute_local_feature_contribution(
            self.model, self.preprocessor, self.feature_names, row_df
        )

        # Maintenance recommendations
        recommendations = generate_recommendations(
            sensor_data=input_data,
            risk_level=risk_level,
            failure_probability=proba,
            predicted_modes=predicted_modes,
        )

        return {
            "is_valid": True,
            "validation_errors": [],
            "status": status_label,
            "prediction_binary": prediction_bin,
            "failure_probability": round(proba, 4),
            "risk_level": risk_level,
            "predicted_modes": predicted_modes,
            "mode_probabilities": mode_probabilities,
            "contributing_features": contributions[:5],  # Top 5 most influential
            "all_contributions": contributions,
            "recommendations": recommendations,
        }

    def predict_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Execute prediction on a batch DataFrame.
        Validates columns, generates predictions, and returns enriched DataFrame.
        """
        if not self.is_ready:
            self._load_artifacts()
            if not self.is_ready:
                raise InferenceError("Models are not trained or loaded. Please train models first.")

        # Check required columns
        missing = [col for col in FEATURE_COLUMNS if col not in df_input.columns]
        if missing:
            raise InferenceError(f"Batch dataset missing required columns: {missing}")

        X_batch = df_input[FEATURE_COLUMNS].copy()

        # Transform batch
        X_trans = self.preprocessor.transform(X_batch)

        # Predict primary failure
        preds = self.model.predict(X_trans)
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X_trans)[:, 1]
        else:
            probas = preds.astype(float)

        risk_levels = [self._determine_risk_level(p) for p in probas]
        status_labels = ["Failure Risk" if (p == 1 or prob >= 0.50) else "Normal" for p, prob in zip(preds, probas)]

        results_df = df_input.copy()
        results_df["Predicted_Status"] = status_labels
        results_df["Failure_Probability"] = np.round(probas, 4)
        results_df["Risk_Level"] = risk_levels

        # Predict failure modes for batch
        if self.failure_mode_models:
            for mode, m_model in self.failure_mode_models.items():
                full_name = FAILURE_MODE_NAMES.get(mode, mode)
                m_preds = m_model.predict(X_trans)
                results_df[f"Risk_{mode}"] = m_preds

        return results_df

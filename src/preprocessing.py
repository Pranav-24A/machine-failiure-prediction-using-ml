"""
Preprocessing and domain-specific feature engineering module.
Implements scikit-learn compatible transformers, pipeline construction,
and leak-free train/validation/test preprocessing.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, PREPROCESSOR_PATH


class PhysicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific physical feature engineer for machine sensor data.
    Derived strictly from physics:
    1. Temperature Difference: Process Temp - Air Temp (Key for Heat Dissipation Failure)
    2. Mechanical Power (Watts): Torque * (Rotational speed * 2 * pi / 60) (Key for Power Failure)
    3. Overstrain Metric: Tool wear * Torque (Key for Overstrain Failure)
    4. Temperature Ratio: Process Temp / Air Temp
    """

    def __init__(self, include_power: bool = True, include_overstrain: bool = True):
        self.include_power = include_power
        self.include_overstrain = include_overstrain

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        air_temp = X_out["Air temperature [K]"]
        proc_temp = X_out["Process temperature [K]"]
        rot_speed = X_out["Rotational speed [rpm]"]
        torque = X_out["Torque [Nm]"]
        tool_wear = X_out["Tool wear [min]"]

        # 1. Thermal delta
        X_out["Temp_Difference [K]"] = proc_temp - air_temp
        X_out["Temp_Ratio"] = proc_temp / (air_temp + 1e-6)

        # 2. Power in Watts: Torque (Nm) * Angular Velocity (rad/s)
        # Angular Velocity = rpm * 2 * pi / 60
        if self.include_power:
            angular_velocity = rot_speed * (2.0 * np.pi / 60.0)
            X_out["Power [W]"] = torque * angular_velocity

        # 3. Tool wear x Torque: Overstrain indicator
        if self.include_overstrain:
            X_out["Overstrain [min*Nm]"] = tool_wear * torque

        return X_out


def build_preprocessing_pipeline() -> Pipeline:
    """
    Constructs a complete leak-free Scikit-Learn preprocessing pipeline.
    Steps:
    1. Feature Engineering (Domain physics)
    2. Categorical One-Hot Encoding + Numerical Imputation & Scaling
    """
    numerical_base = list(NUMERICAL_COLUMNS)
    engineered_cols = ["Temp_Difference [K]", "Temp_Ratio", "Power [W]", "Overstrain [min*Nm]"]
    all_numeric = numerical_base + engineered_cols

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, all_numeric),
            ("cat", cat_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline([
        ("engineer", PhysicalFeatureEngineer()),
        ("preprocessor", column_preprocessor),
    ])

    return full_pipeline


def get_feature_names_out(pipeline: Pipeline) -> List[str]:
    """Retrieve output feature names after transformation for explainability."""
    column_preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = []

    # Numeric features
    num_cols = column_preprocessor.transformers_[0][2]
    feature_names.extend(num_cols)

    # Categorical features
    cat_encoder = column_preprocessor.transformers_[1][1].named_steps["onehot"]
    cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_COLUMNS)
    feature_names.extend(cat_features)

    return feature_names


def save_preprocessor(pipeline: Pipeline, path: Path = PREPROCESSOR_PATH) -> None:
    """Save fitted preprocessing pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_preprocessor(path: Path = PREPROCESSOR_PATH) -> Pipeline:
    """Load fitted preprocessing pipeline from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {path}. Run training first.")
    return joblib.load(path)

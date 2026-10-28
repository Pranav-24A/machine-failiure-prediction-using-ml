"""
Unit tests for preprocessing, domain feature engineering, and pipeline serialization.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    PhysicalFeatureEngineer,
    build_preprocessing_pipeline,
    get_feature_names_out,
    load_preprocessor,
    save_preprocessor,
)


@pytest.fixture
def sample_feature_data():
    return pd.DataFrame({
        "Type": ["M", "L", "H"],
        "Air temperature [K]": [300.0, 302.0, 298.0],
        "Process temperature [K]": [310.0, 310.5, 308.0],
        "Rotational speed [rpm]": [1500.0, 1300.0, 2000.0],
        "Torque [Nm]": [40.0, 50.0, 35.0],
        "Tool wear [min]": [50.0, 120.0, 200.0],
    })


def test_physical_feature_engineer(sample_feature_data):
    """Verify engineered physical variables match exact formulas."""
    engineer = PhysicalFeatureEngineer()
    df_eng = engineer.transform(sample_feature_data)

    # 1. Check Temp Difference
    expected_delta = sample_feature_data["Process temperature [K]"] - sample_feature_data["Air temperature [K]"]
    assert np.allclose(df_eng["Temp_Difference [K]"], expected_delta)

    # 2. Check Overstrain = Tool wear * Torque
    expected_overstrain = sample_feature_data["Tool wear [min]"] * sample_feature_data["Torque [Nm]"]
    assert np.allclose(df_eng["Overstrain [min*Nm]"], expected_overstrain)

    # 3. Check Mechanical Power = Torque * (rpm * 2 * pi / 60)
    angular_vel = sample_feature_data["Rotational speed [rpm]"] * (2 * np.pi / 60)
    expected_power = sample_feature_data["Torque [Nm]"] * angular_vel
    assert np.allclose(df_eng["Power [W]"], expected_power)


def test_pipeline_fit_transform(sample_feature_data):
    """Verify Scikit-Learn pipeline fits and transforms cleanly."""
    pipeline = build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(sample_feature_data)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 3

    feature_names = get_feature_names_out(pipeline)
    assert len(feature_names) == transformed.shape[1]
    assert "Temp_Difference [K]" in feature_names
    assert "Power [W]" in feature_names
    assert "Overstrain [min*Nm]" in feature_names


def test_pipeline_serialization(sample_feature_data, tmp_path):
    """Verify that preprocessing pipeline can be saved and reloaded with identical transforms."""
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(sample_feature_data)

    save_path = tmp_path / "test_preprocessor.joblib"
    save_preprocessor(pipeline, save_path)

    loaded_pipeline = load_preprocessor(save_path)
    res_orig = pipeline.transform(sample_feature_data)
    res_loaded = loaded_pipeline.transform(sample_feature_data)

    assert np.allclose(res_orig, res_loaded)


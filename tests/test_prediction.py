"""
Unit tests for single and batch prediction pipelines, input validation,
risk categorization, and error handling.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from src.predict import InferenceError, PredictiveMaintenancePredictor


@pytest.fixture
def predictor():
    return PredictiveMaintenancePredictor()


def test_predictor_loaded(predictor):
    """Verify that predictor is ready and artifacts are loaded."""
    assert predictor.is_ready is True
    assert predictor.model is not None
    assert predictor.preprocessor is not None


def test_predict_single_valid_input(predictor):
    """Verify prediction output structure and data types for a normal machine."""
    valid_input = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0,
    }

    result = predictor.predict_single(valid_input)

    assert result["is_valid"] is True
    assert result["status"] in ["Normal", "Failure Risk"]
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["risk_level"] in ["Low", "Medium", "High"]
    assert isinstance(result["predicted_modes"], list)
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) > 0
    assert isinstance(result["contributing_features"], list)


def test_predict_single_missing_column(predictor):
    """Verify missing required parameter is caught and reported."""
    incomplete_input = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        # Missing other sensors
    }

    result = predictor.predict_single(incomplete_input)
    assert result["is_valid"] is False
    assert len(result["validation_errors"]) > 0
    assert result["status"] == "Invalid Input"


def test_predict_single_out_of_bounds(predictor):
    """Verify out-of-physical-range sensor values trigger validation warnings/errors."""
    invalid_input = {
        "Type": "M",
        "Air temperature [K]": 999.0,  # Unrealistic temperature
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0,
    }

    result = predictor.predict_single(invalid_input)
    assert result["is_valid"] is False
    assert any("Air temperature [K]" in err for err in result["validation_errors"])


def test_predict_batch_workflow(predictor):
    """Verify batch prediction processes DataFrame and appends required result columns."""
    batch_df = pd.DataFrame([
        {
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551.0,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 0.0,
        },
        {
            "Type": "L",
            "Air temperature [K]": 302.5,
            "Process temperature [K]": 310.4,
            "Rotational speed [rpm]": 1340.0,
            "Torque [Nm]": 46.0,
            "Tool wear [min]": 85.0,
        },
    ])

    results = predictor.predict_batch(batch_df)
    assert len(results) == 2
    assert "Predicted_Status" in results.columns
    assert "Failure_Probability" in results.columns
    assert "Risk_Level" in results.columns


def test_predict_batch_missing_columns_raises_error(predictor):
    """Verify malformed batch DataFrame raises InferenceError."""
    bad_df = pd.DataFrame([{"Type": "M", "Air temperature [K]": 300.0}])
    with pytest.raises(InferenceError, match="missing required columns"):
        predictor.predict_batch(bad_df)


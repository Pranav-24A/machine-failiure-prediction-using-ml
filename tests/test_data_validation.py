"""
Unit tests for dataset schema validation, missing data handling, and target leakage prevention.
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

from config import CATEGORICAL_COLUMNS, FAILURE_MODES, IDENTIFIER_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN
from src.data_loader import DataValidationError, get_feature_target_split, validate_raw_dataframe


def test_valid_dataframe():
    """Verify that a compliant dataframe passes validation successfully."""
    df = pd.DataFrame({
        "UDI": [1, 2],
        "Product ID": ["L1001", "M1002"],
        "Type": ["L", "M"],
        "Air temperature [K]": [298.1, 298.5],
        "Process temperature [K]": [308.1, 308.5],
        "Rotational speed [rpm]": [1500, 1600],
        "Torque [Nm]": [40.0, 42.0],
        "Tool wear [min]": [10.0, 20.0],
        "Machine failure": [0, 1],
        "TWF": [0, 0],
        "HDF": [0, 1],
        "PWF": [0, 0],
        "OSF": [0, 0],
        "RNF": [0, 0],
    })

    report = validate_raw_dataframe(df)
    assert report["is_valid"] is True
    assert report["total_records"] == 2
    assert report["total_failures"] == 1


def test_empty_dataframe_raises_error():
    """Verify that an empty dataframe raises DataValidationError."""
    df_empty = pd.DataFrame()
    with pytest.raises(DataValidationError):
        validate_raw_dataframe(df_empty)


def test_missing_required_feature_raises_error():
    """Verify missing required sensor features raises validation error."""
    df_bad = pd.DataFrame({
        "Type": ["L"],
        "Air temperature [K]": [298.1],
        # Missing Process temperature, Torque, etc.
        "Machine failure": [0],
    })
    with pytest.raises(DataValidationError, match="Missing required feature columns"):
        validate_raw_dataframe(df_bad)


def test_missing_target_raises_error():
    """Verify missing target column raises validation error."""
    df_no_target = pd.DataFrame({
        "Type": ["L"],
        "Air temperature [K]": [298.1],
        "Process temperature [K]": [308.1],
        "Rotational speed [rpm]": [1500],
        "Torque [Nm]": [40.0],
        "Tool wear [min]": [10.0],
    })
    with pytest.raises(DataValidationError, match="Missing target column"):
        validate_raw_dataframe(df_no_target)


def test_target_leakage_prevention():
    """
    CRITICAL TEST: Ensure target columns (Machine failure, TWF, HDF, PWF, OSF, RNF)
    and identifier columns (UDI, Product ID) are STRICTLY excluded from feature matrix X.
    """
    df = pd.DataFrame({
        "UDI": [1, 2],
        "Product ID": ["L1001", "M1002"],
        "Type": ["L", "M"],
        "Air temperature [K]": [298.1, 298.5],
        "Process temperature [K]": [308.1, 308.5],
        "Rotational speed [rpm]": [1500, 1600],
        "Torque [Nm]": [40.0, 42.0],
        "Tool wear [min]": [10.0, 20.0],
        "Machine failure": [0, 1],
        "TWF": [0, 0],
        "HDF": [0, 1],
        "PWF": [0, 0],
        "OSF": [0, 0],
        "RNF": [0, 0],
    })

    X, y, Y_modes = get_feature_target_split(df)

    # Assert no target in X
    assert TARGET_COLUMN not in X.columns
    for mode in FAILURE_MODES:
        assert mode not in X.columns

    # Assert no identifiers in X
    for ident in IDENTIFIER_COLUMNS:
        assert ident not in X.columns

    # Assert X contains ONLY legitimate features
    expected_features = set(CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS)
    assert set(X.columns) == expected_features


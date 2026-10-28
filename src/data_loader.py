"""
Data loading and raw data validation module.
Enforces strict target leakage prevention, verifies schema integrity, and extracts metadata.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from config import (
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    FAILURE_MODES,
    IDENTIFIER_COLUMNS,
    NUMERICAL_COLUMNS,
    SYNTHETIC_DATASET_PATH,
    TARGET_COLUMN,
)
from scripts.download_dataset import acquire_dataset


class DataValidationError(Exception):
    """Exception raised for errors in raw dataset validation."""
    pass


def load_dataset(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load dataset from specified path or default DATASET_PATH.
    If the file does not exist, automatically initiates acquisition/fallback.
    """
    path_to_load = filepath or DATASET_PATH

    if not path_to_load.exists():
        path_to_load = acquire_dataset()

    try:
        df = pd.read_csv(path_to_load)
    except Exception as e:
        raise DataValidationError(f"Could not read dataset CSV at {path_to_load}: {e}")

    validate_raw_dataframe(df)
    return df


def validate_raw_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates dataset schema, types, missing values, duplicates, and target existence.
    Returns validation report dictionary. Raises DataValidationError if critical schema missing.
    """
    if df.empty:
        raise DataValidationError("Dataset is empty.")

    # 1. Check required feature columns
    missing_features = [col for col in (CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS) if col not in df.columns]
    if missing_features:
        raise DataValidationError(f"Missing required feature columns: {missing_features}")

    # 2. Check main target column
    if TARGET_COLUMN not in df.columns:
        raise DataValidationError(f"Missing target column: '{TARGET_COLUMN}'")

    # 3. Check failure mode columns
    available_failure_modes = [col for col in FAILURE_MODES if col in df.columns]

    # 4. Check data types and ranges
    invalid_numeric = []
    for col in NUMERICAL_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            invalid_numeric.append(col)
    if invalid_numeric:
        raise DataValidationError(f"Columns must be numeric: {invalid_numeric}")

    # 5. Check categorical validity
    invalid_types = df[~df["Type"].isin(["L", "M", "H"])]["Type"].dropna().unique()
    if len(invalid_types) > 0:
        raise DataValidationError(f"Column 'Type' contains invalid categories: {list(invalid_types)}")

    # 6. Duplicates and nulls report
    duplicates_count = int(df.duplicated(subset=CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS).sum())
    missing_count = int(df.isnull().sum().sum())

    total_records = len(df)
    total_failures = int(df[TARGET_COLUMN].sum())
    failure_rate = float(total_failures / total_records) if total_records > 0 else 0.0

    return {
        "total_records": total_records,
        "features_present": CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS,
        "target_present": TARGET_COLUMN,
        "available_failure_modes": available_failure_modes,
        "duplicates_count": duplicates_count,
        "missing_values_count": missing_count,
        "total_failures": total_failures,
        "failure_rate": round(failure_rate, 4),
        "is_valid": True,
    }


def get_feature_target_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
    """
    Extracts features (X), primary target (y), and failure mode labels (Y_modes).
    Enforces strict target leakage prevention:
    - Excludes UDI and Product ID
    - Excludes 'Machine failure' from X
    - Excludes all failure mode columns (TWF, HDF, PWF, OSF, RNF) from X
    """
    validate_raw_dataframe(df)

    # Features: Only Type + Numerical sensors
    feature_cols = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
    X = df[feature_cols].copy()

    # Primary target
    y = df[TARGET_COLUMN].copy()

    # Failure modes target (if present)
    available_modes = [m for m in FAILURE_MODES if m in df.columns]
    Y_modes = df[available_modes].copy() if available_modes else None

    return X, y, Y_modes


def get_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Retrieve descriptive metadata and statistical summaries."""
    val = validate_raw_dataframe(df)
    stats = df[NUMERICAL_COLUMNS].describe().to_dict()

    mode_counts = {}
    for m in FAILURE_MODES:
        if m in df.columns:
            mode_counts[m] = int(df[m].sum())

    type_counts = df["Type"].value_counts().to_dict()

    return {
        "validation": val,
        "numerical_stats": stats,
        "failure_mode_counts": mode_counts,
        "type_distribution": type_counts,
    }

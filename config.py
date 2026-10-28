"""
Configuration module for AI-Based Predictive Maintenance and Machine Failure Detection System.
Centralizes all paths, schema definitions, risk thresholds, and sensor operating limits.
"""

from pathlib import Path

# Base Paths (Relative to project root, resilient to working directory variations)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DB_PATH = BASE_DIR / "maintenance_history.db"

# Dataset Files
DATASET_FILENAME = "ai4i2020.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME
SYNTHETIC_DATASET_PATH = DATA_DIR / "synthetic_maintenance_data.csv"
DATASET_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
BACKUP_DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/ai4i2020.csv"

# Schema Definitions
TARGET_COLUMN = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]
CATEGORICAL_COLUMNS = ["Type"]
NUMERICAL_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS

# Failure Mode Full Names and Descriptions
FAILURE_MODE_NAMES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

# Sensor Valid Operating Ranges (Physical limits for validation & UI controls)
SENSOR_LIMITS = {
    "Air temperature [K]": {"min": 290.0, "max": 315.0, "default": 300.0, "unit": "K"},
    "Process temperature [K]": {"min": 300.0, "max": 325.0, "default": 310.0, "unit": "K"},
    "Rotational speed [rpm]": {"min": 1000.0, "max": 3000.0, "default": 1500.0, "unit": "rpm"},
    "Torque [Nm]": {"min": 1.0, "max": 90.0, "default": 40.0, "unit": "Nm"},
    "Tool wear [min]": {"min": 0.0, "max": 270.0, "default": 80.0, "unit": "min"},
}

VALID_TYPES = ["L", "M", "H"]

# Risk Category Thresholds (Configurable operational guidelines)
DEFAULT_RISK_THRESHOLDS = {
    "low": 0.30,      # Probability < 0.30 -> Low Risk
    "medium": 0.60,   # 0.30 <= Probability < 0.60 -> Medium Risk
    # Probability >= 0.60 -> High Risk
}

# Model and Preprocessing Artifact Paths
MODEL_PATH = MODELS_DIR / "failure_classifier.joblib"
FAILURE_MODE_MODEL_PATH = MODELS_DIR / "failure_mode_classifier.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"

# Evaluation and Reporting Artifacts
METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"
COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.json"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.json"
FAILURE_MODE_METRICS_PATH = ARTIFACTS_DIR / "failure_mode_metrics.json"

# Training Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.15
CV_FOLDS = 5


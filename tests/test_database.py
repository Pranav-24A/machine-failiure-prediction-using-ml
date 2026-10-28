"""
Unit tests for SQLite database operations: insertion, querying, filtering,
summary statistics, and transaction safety.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.database import (
    clear_history,
    get_history,
    get_summary_stats,
    init_db,
    log_batch_predictions,
    log_prediction,
)


@pytest.fixture
def temp_db(tmp_path):
    """Provides a temporary SQLite database path for isolated test runs."""
    db_file = tmp_path / "test_maintenance.db"
    init_db(db_file)
    return db_file


def test_log_and_retrieve_prediction(temp_db):
    """Verify logging a prediction and retrieving it from SQLite."""
    record = {
        "machine_id": "TEST-MACH-01",
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50.0,
        "prediction": 0,
        "failure_probability": 0.05,
        "risk_level": "Low",
        "predicted_modes": [],
        "recommendations": ["Normal operation."],
        "model_version": "v1.0",
    }

    inserted_id = log_prediction(record, db_path=temp_db)
    assert inserted_id > 0

    logs = get_history(db_path=temp_db, limit=10)
    assert len(logs) == 1
    assert logs[0]["machine_id"] == "TEST-MACH-01"
    assert logs[0]["risk_level"] == "Low"
    assert logs[0]["prediction"] == 0


def test_filter_history(temp_db):
    """Verify filtering by machine_id and risk_level."""
    rec_low = {
        "machine_id": "ASSET-A",
        "Type": "M",
        "Air temperature [K]": 298.0,
        "Process temperature [K]": 308.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 20.0,
        "prediction": 0,
        "failure_probability": 0.05,
        "risk_level": "Low",
    }
    rec_high = {
        "machine_id": "ASSET-B",
        "Type": "L",
        "Air temperature [K]": 302.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1300.0,
        "Torque [Nm]": 65.0,
        "Tool wear [min]": 210.0,
        "prediction": 1,
        "failure_probability": 0.85,
        "risk_level": "High",
    }

    log_prediction(rec_low, db_path=temp_db)
    log_prediction(rec_high, db_path=temp_db)

    # Filter by risk
    high_logs = get_history(db_path=temp_db, risk_level="High")
    assert len(high_logs) == 1
    assert high_logs[0]["machine_id"] == "ASSET-B"

    # Filter by machine_id
    a_logs = get_history(db_path=temp_db, machine_id="ASSET-A")
    assert len(a_logs) == 1
    assert a_logs[0]["machine_id"] == "ASSET-A"


def test_summary_stats(temp_db):
    """Verify aggregated KPI metrics calculation."""
    records = [
        {
            "machine_id": f"M-{i}",
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 10.0,
            "prediction": 1 if i == 0 else 0,
            "failure_probability": 0.8 if i == 0 else 0.1,
            "risk_level": "High" if i == 0 else "Low",
        }
        for i in range(3)
    ]
    log_batch_predictions(records, db_path=temp_db)

    stats = get_summary_stats(db_path=temp_db)
    assert stats["total_logged"] == 3
    assert stats["high_risk_count"] == 1
    assert stats["low_risk_count"] == 2
    assert stats["avg_risk_prob"] > 0.0


def test_clear_history(temp_db):
    """Verify clearing database records."""
    rec = {
        "machine_id": "TEMP",
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 10.0,
        "prediction": 0,
        "failure_probability": 0.1,
        "risk_level": "Low",
    }
    log_prediction(rec, db_path=temp_db)
    assert len(get_history(db_path=temp_db)) == 1

    clear_history(db_path=temp_db)
    assert len(get_history(db_path=temp_db)) == 0


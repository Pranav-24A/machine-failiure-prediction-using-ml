"""
SQLite database module for persistent prediction logs and maintenance history.
Enforces parameterized queries, schema creation, querying, and filtering.
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional

from config import DB_PATH


def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create a thread-safe connection to the SQLite database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize the maintenance_logs table with appropriate schema and index."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS maintenance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                machine_id TEXT,
                type TEXT NOT NULL,
                air_temperature REAL NOT NULL,
                process_temperature REAL NOT NULL,
                rotational_speed REAL NOT NULL,
                torque REAL NOT NULL,
                tool_wear REAL NOT NULL,
                prediction INTEGER NOT NULL,
                failure_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                predicted_modes TEXT,
                recommendations TEXT,
                model_version TEXT NOT NULL,
                notes TEXT
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON maintenance_logs (timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_machine_id ON maintenance_logs (machine_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_level ON maintenance_logs (risk_level)")
        conn.commit()


def log_prediction(data: Dict[str, Any], db_path: Path = DB_PATH) -> int:
    """
    Log an individual or batch prediction record to SQLite.
    Returns the newly inserted record id.
    """
    init_db(db_path)
    timestamp = data.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    predicted_modes = data.get("predicted_modes", [])
    if isinstance(predicted_modes, (list, tuple)):
        predicted_modes_str = json.dumps(predicted_modes)
    else:
        predicted_modes_str = str(predicted_modes)

    recs = data.get("recommendations", [])
    if isinstance(recs, (list, tuple)):
        recs_str = "\n".join(recs)
    else:
        recs_str = str(recs)

    query = """
    INSERT INTO maintenance_logs (
        timestamp, machine_id, type, air_temperature, process_temperature,
        rotational_speed, torque, tool_wear, prediction, failure_probability,
        risk_level, predicted_modes, recommendations, model_version, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        timestamp,
        data.get("machine_id", "MACH-UNASSIGNED"),
        data.get("Type", "M"),
        float(data.get("Air temperature [K]", 0.0)),
        float(data.get("Process temperature [K]", 0.0)),
        float(data.get("Rotational speed [rpm]", 0.0)),
        float(data.get("Torque [Nm]", 0.0)),
        float(data.get("Tool wear [min]", 0.0)),
        int(data.get("prediction", 0)),
        float(data.get("failure_probability", 0.0)),
        str(data.get("risk_level", "Low")),
        predicted_modes_str,
        recs_str,
        str(data.get("model_version", "v1.0")),
        data.get("notes", ""),
    )

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.lastrowid


def log_batch_predictions(records: List[Dict[str, Any]], db_path: Path = DB_PATH) -> int:
    """Bulk log predictions from batch runs."""
    init_db(db_path)
    inserted = 0
    for record in records:
        log_prediction(record, db_path=db_path)
        inserted += 1
    return inserted


def get_history(
    db_path: Path = DB_PATH,
    limit: int = 100,
    machine_id: Optional[str] = None,
    risk_level: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve filtered prediction logs from database."""
    init_db(db_path)
    query = "SELECT * FROM maintenance_logs WHERE 1=1"
    params: List[Any] = []

    if machine_id:
        query += " AND machine_id LIKE ?"
        params.append(f"%{machine_id}%")

    if risk_level and risk_level != "All":
        query += " AND risk_level = ?"
        params.append(risk_level)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_summary_stats(db_path: Path = DB_PATH) -> Dict[str, Any]:
    """Calculate summary statistics from stored maintenance logs."""
    init_db(db_path)
    query = """
    SELECT
        COUNT(*) as total_logged,
        SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END) as high_risk_count,
        SUM(CASE WHEN risk_level = 'Medium' THEN 1 ELSE 0 END) as medium_risk_count,
        SUM(CASE WHEN risk_level = 'Low' THEN 1 ELSE 0 END) as low_risk_count,
        AVG(failure_probability) as avg_risk_prob
    FROM maintenance_logs
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        row = cursor.fetchone()
        if row and row["total_logged"]:
            return {
                "total_logged": row["total_logged"] or 0,
                "high_risk_count": row["high_risk_count"] or 0,
                "medium_risk_count": row["medium_risk_count"] or 0,
                "low_risk_count": row["low_risk_count"] or 0,
                "avg_risk_prob": round(row["avg_risk_prob"] or 0.0, 4),
            }
        return {
            "total_logged": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "avg_risk_prob": 0.0,
        }


def clear_history(db_path: Path = DB_PATH) -> None:
    """Clear all records from database."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM maintenance_logs")
        conn.commit()

"""
Script to verify predictions across standard and all failure modes.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import PredictiveMaintenancePredictor

def verify_all_scenarios():
    predictor = PredictiveMaintenancePredictor()
    assert predictor.is_ready, "Predictor is not ready!"

    scenarios = {
        "Nominal / Normal": {
            "Type": "M",
            "Air temperature [K]": 298.2,
            "Process temperature [K]": 308.7,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 30.0,
        },
        "Heat Dissipation Failure (HDF)": {
            "Type": "L",
            "Air temperature [K]": 302.5,
            "Process temperature [K]": 310.4,
            "Rotational speed [rpm]": 1340.0,
            "Torque [Nm]": 46.0,
            "Tool wear [min]": 85.0,
        },
        "Power Delivery Failure (PWF)": {
            "Type": "L",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 2850.0,
            "Torque [Nm]": 65.0,
            "Tool wear [min]": 60.0,
        },
        "Overstrain Failure (OSF)": {
            "Type": "L",
            "Air temperature [K]": 299.0,
            "Process temperature [K]": 309.0,
            "Rotational speed [rpm]": 1350.0,
            "Torque [Nm]": 62.0,
            "Tool wear [min]": 215.0,
        },
        "Tool Wear Failure (TWF)": {
            "Type": "H",
            "Air temperature [K]": 298.0,
            "Process temperature [K]": 308.0,
            "Rotational speed [rpm]": 1520.0,
            "Torque [Nm]": 39.0,
            "Tool wear [min]": 235.0,
        },
    }

    print("-" * 75)
    print(f"{'Scenario':32s} | {'Status':12s} | {'Risk':8s} | {'Prob':6s} | {'Modes'}")
    print("-" * 75)

    for name, data in scenarios.items():
        res = predictor.predict_single(data)
        modes = ", ".join(res["predicted_modes"]) if res["predicted_modes"] else "None"
        print(f"{name:32s} | {res['status']:12s} | {res['risk_level']:8s} | {res['failure_probability']:5.1%} | {modes}")

    print("-" * 75)
    print("Verification completed successfully!")

if __name__ == "__main__":
    verify_all_scenarios()

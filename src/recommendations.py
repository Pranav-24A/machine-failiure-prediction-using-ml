"""
Actionable Maintenance Recommendations Engine.
Generates rule-based, physics-informed inspection suggestions based on sensor values,
risk levels, and predicted failure modes.
"""

from typing import Any, Dict, List


def generate_recommendations(
    sensor_data: Dict[str, Any],
    risk_level: str,
    failure_probability: float,
    predicted_modes: List[str],
) -> List[str]:
    """
    Generate tailored maintenance recommendations.
    Provides suggested inspections, clearly stating they are recommendations, not absolute diagnoses.
    """
    recommendations: List[str] = []

    # 1. High-level status recommendation
    if risk_level == "High":
        recommendations.append(
            "[CRITICAL] Machine is operating at HIGH risk of failure (Estimated Risk: {:.1%}). "
            "Recommend immediate load reduction or temporary halt for physical inspection.".format(
                failure_probability
            )
        )
    elif risk_level == "Medium":
        recommendations.append(
            "[CAUTION] Elevated risk detected (Estimated Risk: {:.1%}). "
            "Schedule non-disruptive inspection during next planned maintenance window.".format(
                failure_probability
            )
        )
    else:
        recommendations.append(
            "[NORMAL] Operating parameters within nominal limits (Estimated Risk: {:.1%}). "
            "Maintain standard preventive maintenance schedule.".format(failure_probability)
        )

    # 2. Extract sensor readings
    air_temp = float(sensor_data.get("Air temperature [K]", 300.0))
    proc_temp = float(sensor_data.get("Process temperature [K]", 310.0))
    rot_speed = float(sensor_data.get("Rotational speed [rpm]", 1500.0))
    torque = float(sensor_data.get("Torque [Nm]", 40.0))
    tool_wear = float(sensor_data.get("Tool wear [min]", 80.0))

    temp_diff = proc_temp - air_temp
    power_watts = torque * (rot_speed * (2 * 3.14159265 / 60))
    overstrain = tool_wear * torque

    # 3. Mode-specific & Sensor-specific diagnostics

    # Tool Wear Failure (TWF)
    if "Tool Wear Failure" in predicted_modes or "TWF" in predicted_modes or tool_wear >= 200:
        recommendations.append(
            "Tool Wear Alert (Wear: {:.0f} min): Inspect cutting insert/tool edges for micro-fractures, "
            "cratering, and flank wear. Schedule replacement before resuming high-speed operations.".format(
                tool_wear
            )
        )

    # Heat Dissipation Failure (HDF)
    if "Heat Dissipation Failure" in predicted_modes or "HDF" in predicted_modes or (
        temp_diff < 8.6 and rot_speed < 1380
    ):
        recommendations.append(
            "Heat Dissipation Alert (Delta T: {:.2f} K, Speed: {:.0f} rpm): Heat dissipation capacity compromised. "
            "Inspect coolant supply lines, heat exchanger cleanliness, and cooling fan functionality.".format(
                temp_diff, rot_speed
            )
        )

    # Power Failure (PWF)
    if "Power Failure" in predicted_modes or "PWF" in predicted_modes or (
        power_watts < 3500 or power_watts > 9000
    ):
        recommendations.append(
            "Power Delivery Alert (Calculated Power: {:.0f} W): Drive power is outside nominal limits [3500W - 9000W]. "
            "Verify motor drive inverter, phase currents, power supply stability, and mechanical transmission resistance.".format(
                power_watts
            )
        )

    # Overstrain Failure (OSF)
    if "Overstrain Failure" in predicted_modes or "OSF" in predicted_modes or overstrain > 11000:
        recommendations.append(
            "Overstrain Alert (Wear-Torque Product: {:.0f} min*Nm): Mechanical stress threshold exceeded. "
            "Check feed rate, workpiece clamping tension, and inspect spindle bearings for structural strain.".format(
                overstrain
            )
        )

    # Random Failure (RNF)
    if "Random Failure" in predicted_modes or "RNF" in predicted_modes:
        recommendations.append(
            "Random Failure Anomaly: Uncorrelated anomaly detected. Perform comprehensive sensor self-test, "
            "check wiring continuity, verify grounding, and inspect emergency stop circuits."
        )

    # Secondary Sensor Alerts
    if torque > 65.0 and "Overstrain Alert" not in str(recommendations):
        recommendations.append(
            "High Torque Reading ({:.1f} Nm): Check for material hardness variations or tool binding.".format(
                torque
            )
        )

    if rot_speed > 2500.0:
        recommendations.append(
            "High Rotational Velocity ({:.0f} rpm): Check bearing lubrication and dynamic balance.".format(
                rot_speed
            )
        )

    return recommendations


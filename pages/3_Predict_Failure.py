"""
Page 3: Individual Machine Failure Prediction.
Provides interactive input form with validation, failure scenario presets,
risk gauges, failure mode probabilities, local feature contribution charts,
and actionable maintenance recommendations.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from config import SENSOR_LIMITS, VALID_TYPES
from src.database import log_prediction
from src.predict import PredictiveMaintenancePredictor
from src.utils import create_risk_gauge, get_status_badge_html, plot_failure_mode_breakdown

st.set_page_config(page_title="Predict Failure - Predictive Maintenance", page_icon="⚡", layout="wide")


@st.cache_resource
def get_predictor():
    return PredictiveMaintenancePredictor()


# Predefined realistic failure scenario presets for demonstration
PRESETS = {
    "Select a preset scenario...": None,
    "🟢 Nominal / Normal Operation": {
        "Type": "M",
        "Air temperature [K]": 298.2,
        "Process temperature [K]": 308.7,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 30.0,
    },
    "🔴 Heat Dissipation Failure (HDF) Scenario": {
        "Type": "L",
        "Air temperature [K]": 302.5,
        "Process temperature [K]": 310.4,  # Delta T = 7.9 K (< 8.6 K)
        "Rotational speed [rpm]": 1340.0,  # Speed < 1380 rpm
        "Torque [Nm]": 46.0,
        "Tool wear [min]": 85.0,
    },
    "🔴 Power Delivery Failure (PWF) Scenario": {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 2850.0,  # Very high speed and torque -> Power > 9000W
        "Torque [Nm]": 65.0,
        "Tool wear [min]": 60.0,
    },
    "🔴 Overstrain Failure (OSF) Scenario": {
        "Type": "L",
        "Air temperature [K]": 299.0,
        "Process temperature [K]": 309.0,
        "Rotational speed [rpm]": 1350.0,
        "Torque [Nm]": 62.0,
        "Tool wear [min]": 215.0,  # 215 * 62 = 13,330 min*Nm (> 11,000 threshold for L)
    },
    "🔴 Tool Wear Failure (TWF) Scenario": {
        "Type": "H",
        "Air temperature [K]": 298.0,
        "Process temperature [K]": 308.0,
        "Rotational speed [rpm]": 1520.0,
        "Torque [Nm]": 39.0,
        "Tool wear [min]": 235.0,  # Severe wear > 220 min
    },
}


def main():
    st.title("⚡ Predict Machine Failure Risk")
    st.markdown(
        "Enter machine sensor telemetry or choose an industrial failure scenario preset to evaluate "
        "operational failure probability, identify active failure modes, and receive maintenance guidance."
    )

    predictor = get_predictor()
    if not predictor.is_ready:
        st.error("Predictor model artifacts not found. Please train models first.")
        return

    # Scenario Preset Selector
    selected_preset_name = st.selectbox(
        "Quick Scenario Presets (Loads realistic industrial test cases):",
        list(PRESETS.keys()),
        index=0,
    )
    preset_values = PRESETS.get(selected_preset_name)

    # Input Form
    with st.form("manual_prediction_form"):
        st.subheader("Machine Operating Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            default_type = preset_values["Type"] if preset_values else "M"
            machine_type = st.selectbox(
                "Product Quality Variant (Type):",
                VALID_TYPES,
                index=VALID_TYPES.index(default_type),
                help="L = Low variant (50%), M = Medium (30%), H = High (20%)",
            )

            default_air = preset_values["Air temperature [K]"] if preset_values else SENSOR_LIMITS["Air temperature [K]"]["default"]
            air_temp = st.number_input(
                "Air Temperature [K]:",
                min_value=float(SENSOR_LIMITS["Air temperature [K]"]["min"]),
                max_value=float(SENSOR_LIMITS["Air temperature [K]"]["max"]),
                value=float(default_air),
                step=0.1,
                format="%.1f",
                help="Ambient air temperature in Kelvin (Nominal ~300 K).",
            )

        with col2:
            default_proc = preset_values["Process temperature [K]"] if preset_values else SENSOR_LIMITS["Process temperature [K]"]["default"]
            proc_temp = st.number_input(
                "Process Temperature [K]:",
                min_value=float(SENSOR_LIMITS["Process temperature [K]"]["min"]),
                max_value=float(SENSOR_LIMITS["Process temperature [K]"]["max"]),
                value=float(default_proc),
                step=0.1,
                format="%.1f",
                help="Process temperature in Kelvin (Nominal ~310 K).",
            )

            default_rot = preset_values["Rotational speed [rpm]"] if preset_values else SENSOR_LIMITS["Rotational speed [rpm]"]["default"]
            rot_speed = st.number_input(
                "Rotational Speed [rpm]:",
                min_value=float(SENSOR_LIMITS["Rotational speed [rpm]"]["min"]),
                max_value=float(SENSOR_LIMITS["Rotational speed [rpm]"]["max"]),
                value=float(default_rot),
                step=10.0,
                format="%.0f",
                help="Spindle rotational velocity (Nominal 1200 - 2800 rpm).",
            )

        with col3:
            default_torque = preset_values["Torque [Nm]"] if preset_values else SENSOR_LIMITS["Torque [Nm]"]["default"]
            torque = st.number_input(
                "Torque [Nm]:",
                min_value=float(SENSOR_LIMITS["Torque [Nm]"]["min"]),
                max_value=float(SENSOR_LIMITS["Torque [Nm]"]["max"]),
                value=float(default_torque),
                step=0.5,
                format="%.1f",
                help="Applied torque in Newton-meters (Nominal ~40 Nm).",
            )

            default_wear = preset_values["Tool wear [min]"] if preset_values else SENSOR_LIMITS["Tool wear [min]"]["default"]
            tool_wear = st.number_input(
                "Tool Wear [min]:",
                min_value=float(SENSOR_LIMITS["Tool wear [min]"]["min"]),
                max_value=float(SENSOR_LIMITS["Tool wear [min]"]["max"]),
                value=float(default_wear),
                step=1.0,
                format="%.0f",
                help="Cumulative active cutting tool operation time in minutes.",
            )

        machine_id = st.text_input("Optional Machine Identifier:", value="MACH-CUSTOM-01")

        submitted = st.form_submit_button("⚡ Evaluate Machine Health Risk", type="primary")

    if submitted:
        input_dict = {
            "Type": machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": proc_temp,
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }

        # Run prediction
        result = predictor.predict_single(input_dict)

        if not result["is_valid"]:
            st.error("Input Validation Errors Detected:")
            for err in result["validation_errors"]:
                st.write(f"- {err}")
            return

        st.markdown("---")
        st.subheader("Diagnostic Results & Health Assessment")

        # Top summary bar
        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            st.markdown(f"**Predicted Status:** **{result['status']}**")
            st.markdown(get_status_badge_html(result["risk_level"]), unsafe_allow_html=True)
        with res_c2:
            st.metric("Estimated Failure Probability", f"{result['failure_probability']:.1%}")
        with res_c3:
            modes_str = ", ".join(result["predicted_modes"]) if result["predicted_modes"] else "None (Nominal)"
            st.markdown(f"**Identified Failure Categories:**\n`{modes_str}`")

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge & Mode Probabilities
        gauge_col, mode_col = st.columns([1, 1])

        with gauge_col:
            st.plotly_chart(
                create_risk_gauge(result["failure_probability"], result["risk_level"]),
                use_container_width=True,
            )

        with mode_col:
            if result.get("mode_probabilities"):
                st.plotly_chart(
                    plot_failure_mode_breakdown(result["mode_probabilities"]),
                    use_container_width=True,
                )

        # Local Feature Contribution Chart
        st.subheader("Key Influencing Sensor Contributions")
        st.caption("Shows how deviations in individual sensor readings drive the machine's risk score relative to normal baseline operating ranges.")

        contribs = result.get("contributing_features", [])
        if contribs:
            df_contrib = pd.DataFrame(contribs)
            fig_contrib = px.bar(
                df_contrib,
                x="impact_score",
                y="feature",
                orientation="h",
                color="operating_state",
                color_discrete_map={"Elevated": "#ef4444", "Depressed": "#f59e0b", "Nominal": "#10b981"},
                title="Top Influencing Features for This Prediction",
                labels={"impact_score": "Relative Impact Score", "feature": "Feature"},
            )
            fig_contrib.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_contrib, use_container_width=True)

        # Maintenance Recommendations Box
        st.subheader("🛠️ Recommended Maintenance Interventions")
        for rec in result["recommendations"]:
            if "[CRITICAL]" in rec:
                st.error(rec)
            elif "[CAUTION]" in rec:
                st.warning(rec)
            else:
                st.info(rec)

        # Log prediction to SQLite
        record_to_log = {
            **input_dict,
            "machine_id": machine_id,
            "prediction": result["prediction_binary"],
            "failure_probability": result["failure_probability"],
            "risk_level": result["risk_level"],
            "predicted_modes": result["predicted_modes"],
            "recommendations": result["recommendations"],
            "model_version": "RandomForest-v1.0",
        }
        log_id = log_prediction(record_to_log)
        st.success(f"Prediction successfully recorded in persistent SQLite maintenance history (Record #{log_id}).")


if __name__ == "__main__":
    main()


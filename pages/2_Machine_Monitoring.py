"""
Page 2: Machine Health Monitoring.
Provides telemetry inspection for machines in the historical dataset,
with an optional clearly-labeled simulated streaming mode.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from config import DATASET_PATH, FEATURE_COLUMNS, SENSOR_LIMITS
from src.data_loader import load_dataset
from src.database import log_prediction
from src.predict import PredictiveMaintenancePredictor
from src.utils import create_risk_gauge, get_status_badge_html

st.set_page_config(page_title="Machine Monitoring - Predictive Maintenance", page_icon="🔍", layout="wide")


@st.cache_resource
def get_predictor():
    return PredictiveMaintenancePredictor()


@st.cache_data
def get_dataset():
    return load_dataset(DATASET_PATH)


def main():
    st.title("🔍 Machine Health Monitoring")
    st.markdown(
        "Inspect telemetry and evaluate the health state of specific machine assets. "
        "Select a record from the historical fleet or run a simulated telemetry stream."
    )

    st.caption("ℹ️ **Data Transparency**: Primary data source is the historical tabular AI4I dataset. Live telemetry below is simulated based on dataset distributions.")

    predictor = get_predictor()
    if not predictor.is_ready:
        st.error("Predictive models are not loaded. Please train models from the terminal or main page.")
        return

    df = get_dataset()

    # Tab navigation: Historical Machine Inspector vs. Simulated Live Telemetry
    tab1, tab2 = st.tabs(["📋 Historical Asset Inspector", "📡 Simulated Telemetry Stream"])

    # TAB 1: Historical Machine Inspector
    with tab1:
        st.subheader("Select Monitored Machine from Dataset")

        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            # Let user select machine by UDI or index
            max_udi = len(df)
            selected_udi = st.number_input(
                "Enter Machine UDI (1 to 10,000):",
                min_value=1,
                max_value=max_udi,
                value=1,
                step=1,
            )

        with col_sel2:
            quick_filter = st.selectbox(
                "Or Jump to Example Incident:",
                [
                    "Choose example...",
                    "Normal Machine (UDI 1)",
                    "Heat Dissipation Incident (UDI 51)",
                    "Power Failure Incident (UDI 161)",
                    "Tool Wear Incident (UDI 1088)",
                    "Overstrain Incident (UDI 4762)",
                ],
            )
            if quick_filter == "Normal Machine (UDI 1)":
                selected_udi = 1
            elif quick_filter == "Heat Dissipation Incident (UDI 51)":
                selected_udi = 51
            elif quick_filter == "Power Failure Incident (UDI 161)":
                selected_udi = 161
            elif quick_filter == "Tool Wear Incident (UDI 1088)":
                selected_udi = 1088
            elif quick_filter == "Overstrain Incident (UDI 4762)":
                selected_udi = 4762

        # Retrieve row
        row = df[df["UDI"] == selected_udi].iloc[0]

        input_data = {
            "Type": row["Type"],
            "Air temperature [K]": float(row["Air temperature [K]"]),
            "Process temperature [K]": float(row["Process temperature [K]"]),
            "Rotational speed [rpm]": float(row["Rotational speed [rpm]"]),
            "Torque [Nm]": float(row["Torque [Nm]"]),
            "Tool wear [min]": float(row["Tool wear [min]"]),
        }

        # Run prediction
        result = predictor.predict_single(input_data)

        # Display Machine Overview Header
        st.markdown("---")
        h_col1, h_col2, h_col3 = st.columns([1, 1, 1])

        with h_col1:
            st.markdown(f"**Asset Identifier:** `{row.get('Product ID', f'MACH-{selected_udi}')}`")
            st.markdown(f"**Variant Quality:** Type **{row['Type']}**")

        with h_col2:
            st.markdown(f"**Predicted Status:** **{result['status']}**")
            st.markdown(get_status_badge_html(result["risk_level"]), unsafe_allow_html=True)

        with h_col3:
            actual_label = "Failed" if row.get("Machine failure", 0) == 1 else "Normal"
            st.markdown(f"**Ground Truth Label:** `{actual_label}`")
            if result["predicted_modes"]:
                st.markdown(f"**Identified Risk Modes:** `{', '.join(result['predicted_modes'])}`")

        # Telemetry Gauges and Values
        st.markdown("#### Live Sensor Readings & Operating Envelope")
        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:
            st.metric("Air Temp", f"{input_data['Air temperature [K]']} K", delta=f"{input_data['Air temperature [K]'] - 300:.1f} from 300K")
        with m2:
            delta_t = input_data["Process temperature [K]"] - input_data["Air temperature [K]"]
            st.metric("Process Temp", f"{input_data['Process temperature [K]']} K", delta=f"ΔT: {delta_t:.1f} K")
        with m3:
            st.metric("Rotational Speed", f"{input_data['Rotational speed [rpm]']:.0f} rpm")
        with m4:
            st.metric("Torque", f"{input_data['Torque [Nm]']} Nm")
        with m5:
            st.metric("Tool Wear", f"{input_data['Tool wear [min]']:.0f} min", delta=f"{240 - input_data['Tool wear [min]']:.0f} min to max", delta_color="inverse")

        # Risk Gauge & Recommendations
        g_col1, g_col2 = st.columns([1, 1])

        with g_col1:
            gauge_fig = create_risk_gauge(result["failure_probability"], result["risk_level"])
            st.plotly_chart(gauge_fig, use_container_width=True)

        with g_col2:
            st.markdown("#### Actionable Maintenance Recommendations")
            for rec in result["recommendations"]:
                if "[CRITICAL]" in rec:
                    st.error(rec)
                elif "[CAUTION]" in rec:
                    st.warning(rec)
                else:
                    st.info(rec)

            if st.button("💾 Log Inspection Record to SQLite", key="log_hist_inspect"):
                record = {**input_data, "machine_id": str(row.get("Product ID", f"UDI-{selected_udi}")), **result}
                log_prediction(record)
                st.success("Successfully logged inspection to SQLite history!")

    # TAB 2: Simulated Live Telemetry Stream
    with tab2:
        st.subheader("Simulated Real-Time Machine Telemetry")
        st.markdown(
            "This simulator synthesizes operational machine readings by perturbing baseline sensor distributions. "
            "Click **Generate Telemetry Sample** to simulate the next operational cycle."
        )

        sim_type = st.selectbox("Simulate Machine Asset Quality:", ["Medium (M)", "Low (L)", "High (H)"], key="sim_type_select")
        machine_type = sim_type[0]

        col_sim_btn1, col_sim_btn2 = st.columns([1, 3])
        with col_sim_btn1:
            simulate_now = st.button("🔄 Generate Telemetry Sample", type="primary")

        if simulate_now or "sim_data" not in st.session_state:
            # Generate random realistic sensor readings
            air_t = round(float(np.random.normal(300.0, 1.8)), 2)
            proc_t = round(float(air_t + 10.0 + np.random.normal(0, 0.9)), 2)
            torque_val = round(float(np.clip(np.random.normal(40.0, 9.5), 10.0, 75.0)), 1)
            rot_val = round(float(np.clip((2860.0 / (torque_val * (2 * np.pi / 60))) + np.random.normal(0, 40), 1150, 2800)), 0)
            wear_val = round(float(np.random.uniform(10, 225)), 0)

            st.session_state["sim_data"] = {
                "Type": machine_type,
                "Air temperature [K]": air_t,
                "Process temperature [K]": proc_t,
                "Rotational speed [rpm]": rot_val,
                "Torque [Nm]": torque_val,
                "Tool wear [min]": wear_val,
                "machine_id": f"SIM-{machine_type}{int(time.time()) % 10000:04d}",
            }

        sim_data = st.session_state["sim_data"]
        sim_res = predictor.predict_single(sim_data)

        # Show simulated readings
        st.markdown(f"**Asset:** `{sim_data['machine_id']}` | **Status:** {get_status_badge_html(sim_res['risk_level'])}", unsafe_allow_html=True)

        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Air Temp", f"{sim_data['Air temperature [K]']} K")
        sc2.metric("Process Temp", f"{sim_data['Process temperature [K]']} K")
        sc3.metric("Rotational Speed", f"{sim_data['Rotational speed [rpm]']:.0f} rpm")
        sc4.metric("Torque", f"{sim_data['Torque [Nm]']} Nm")
        sc5.metric("Tool Wear", f"{sim_data['Tool wear [min]']:.0f} min")

        sg1, sg2 = st.columns([1, 1])
        with sg1:
            st.plotly_chart(create_risk_gauge(sim_res["failure_probability"], sim_res["risk_level"]), use_container_width=True)
        with sg2:
            st.markdown("#### Diagnostic Prescriptions")
            for r in sim_res["recommendations"]:
                st.write(r)

            if st.button("💾 Log Simulated Reading", key="log_sim_btn"):
                rec_sim = {**sim_data, **sim_res}
                log_prediction(rec_sim)
                st.success("Logged simulated reading into SQLite maintenance database.")


if __name__ == "__main__":
    main()


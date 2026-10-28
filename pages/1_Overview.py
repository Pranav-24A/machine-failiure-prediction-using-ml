"""
Page 1: Overview Dashboard.
Provides high-level fleet statistics, risk distribution, historical failure rates,
and recent prediction activity.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DATASET_PATH, FAILURE_MODES, TARGET_COLUMN
from src.data_loader import load_dataset
from src.database import get_history, get_summary_stats
from src.utils import get_status_badge_html

st.set_page_config(page_title="Overview Dashboard - Predictive Maintenance", page_icon="📊", layout="wide")


def load_cached_data():
    try:
        return load_dataset(DATASET_PATH)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None


def main():
    st.title("📊 Fleet Overview & Operational Health")
    st.markdown(
        "Executive operational overview providing fleet-wide health metrics, historical breakdown rates, "
        "and logged inference activity."
    )

    df = load_cached_data()
    db_stats = get_summary_stats()
    history_logs = get_history(limit=5)

    if df is not None:
        total_machines = len(df)
        total_failures = int(df[TARGET_COLUMN].sum())
        failure_rate = total_failures / total_machines
    else:
        total_machines = 0
        total_failures = 0
        failure_rate = 0.0

    st.markdown("### Historical Dataset Baseline vs. Runtime Inference Activity")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Monitored Machines", f"{total_machines:,}", help="Total historical machine records in the AI4I 2020 dataset.")
    with col2:
        st.metric("Historical Failures", f"{total_failures:,}", delta=f"{failure_rate:.2%} failure rate", delta_color="inverse")
    with col3:
        st.metric("Logged Inferences", f"{db_stats['total_logged']:,}", help="Total predictions executed and logged to SQLite history.")
    with col4:
        st.metric("Recent High-Risk Machines", f"{db_stats['high_risk_count']:,}", delta=f"{db_stats['medium_risk_count']} medium risk", delta_color="inverse")

    st.markdown("---")

    # Visualizations
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Historical Failure Distribution by Machine Type")
        if df is not None:
            type_agg = df.groupby("Type")[TARGET_COLUMN].agg(
                Total="count",
                Failures="sum",
            ).reset_index()
            type_agg["Failure_Rate"] = (type_agg["Failures"] / type_agg["Total"]) * 100

            fig_type = px.bar(
                type_agg,
                x="Type",
                y="Failure_Rate",
                color="Type",
                text=type_agg["Failure_Rate"].apply(lambda x: f"{x:.2f}%"),
                title="Failure Rate by Product Quality Variant (L, M, H)",
                color_discrete_map={"L": "#f97316", "M": "#3b82f6", "H": "#10b981"},
                labels={"Failure_Rate": "Failure Rate (%)", "Type": "Machine Type"},
            )
            fig_type.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_type, use_container_width=True)

    with col_chart2:
        st.subheader("Historical Failure Mode Breakdown")
        if df is not None:
            mode_counts = {m: int(df[m].sum()) for m in FAILURE_MODES if m in df.columns}
            df_modes = pd.DataFrame(list(mode_counts.items()), columns=["Mode", "Incidents"])
            df_modes = df_modes.sort_values(by="Incidents", ascending=False)

            fig_modes = px.pie(
                df_modes,
                names="Mode",
                values="Incidents",
                title="Incidents per Failure Mode (AI4I 2020)",
                color_discrete_sequence=px.colors.qualitative.Bold,
                hole=0.4,
            )
            fig_modes.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_modes, use_container_width=True)

    st.markdown("---")

    # Recent Inferences Table
    st.subheader("Recent System Inferences (SQLite Persistence)")
    if history_logs:
        df_hist = pd.DataFrame(history_logs)
        display_cols = [
            "timestamp",
            "machine_id",
            "type",
            "air_temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
            "risk_level",
            "failure_probability",
        ]
        available_cols = [c for c in display_cols if c in df_hist.columns]
        df_display = df_hist[available_cols].copy()
        df_display["failure_probability"] = df_display["failure_probability"].apply(lambda p: f"{p:.1%}")

        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No runtime predictions have been logged yet. Use the **Predict Failure** or **Batch Prediction** pages to generate and log inferences.")


if __name__ == "__main__":
    main()


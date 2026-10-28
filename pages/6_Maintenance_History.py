"""
Page 6: Maintenance History.
Provides persistent audit logging and searchable history using SQLite,
tracking machine predictions, operational risk levels, and recommended actions.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.database import clear_history, get_history, get_summary_stats
from src.utils import get_status_badge_html

st.set_page_config(page_title="Maintenance History - Predictive Maintenance", page_icon="🗄️", layout="wide")


def main():
    st.title("🗄️ Maintenance Audit Log & Prediction History")
    st.markdown(
        "Audit trail of machine telemetry inspections and risk predictions stored persistently in SQLite. "
        "Filter by machine ID, inspect risk levels, and review generated recommendations."
    )

    st.caption(
        "⚠️ **Audit Disclaimer**: The entries below record **model-predicted risk estimates** at the time of inference. "
        "They do **not** represent verified historical breakdowns unless confirmed by physical maintenance work orders."
    )

    stats = get_summary_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Inferences Logged", f"{stats['total_logged']:,}")
    col2.metric("High-Risk Alerts", f"{stats['high_risk_count']:,}")
    col3.metric("Medium-Risk Warnings", f"{stats['medium_risk_count']:,}")
    col4.metric("Average Risk Probability", f"{stats['avg_risk_prob']:.1%}")

    st.markdown("---")

    # Filters
    st.subheader("Filter Audit Records")
    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])

    with f_col1:
        machine_search = st.text_input("Search Machine Identifier:", placeholder="e.g. MACH-01 or UDI-100")
    with f_col2:
        risk_filter = st.selectbox("Filter Risk Level:", ["All", "High", "Medium", "Low"])
    with f_col3:
        record_limit = st.selectbox("Max Records:", [25, 50, 100, 250, 500], index=1)

    records = get_history(
        limit=record_limit,
        machine_id=machine_search if machine_search.strip() else None,
        risk_level=risk_filter,
    )

    if records:
        df_hist = pd.DataFrame(records)

        # Reorder and format columns for presentation
        display_cols = [
            "id",
            "timestamp",
            "machine_id",
            "type",
            "air_temperature",
            "process_temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
            "prediction",
            "failure_probability",
            "risk_level",
            "predicted_modes",
            "recommendations",
            "model_version",
        ]
        available_cols = [c for c in display_cols if c in df_hist.columns]
        df_display = df_hist[available_cols].copy()

        df_display["failure_probability"] = df_display["failure_probability"].apply(lambda x: f"{x:.1%}")
        df_display["status"] = df_display["prediction"].apply(lambda p: "Failure Risk" if p == 1 else "Normal")

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Download button
        csv_data = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export History Log to CSV",
            data=csv_data,
            file_name="maintenance_history_audit.csv",
            mime="text/csv",
        )
    else:
        st.info("No matching maintenance records found in SQLite database.")

    st.markdown("---")

    # Danger Zone / Administration
    with st.expander("⚙️ Database Administration"):
        st.warning("Clearing history will permanently remove all logged predictions from SQLite.")
        if st.button("🗑️ Clear All Maintenance History", type="secondary"):
            clear_history()
            st.success("Maintenance history table has been cleared.")
            st.rerun()


if __name__ == "__main__":
    main()


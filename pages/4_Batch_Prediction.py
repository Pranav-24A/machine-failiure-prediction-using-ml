"""
Page 4: Batch Prediction.
Allows CSV upload of industrial sensor telemetry, performs batch validation,
executes high-throughput predictions, summarizes risk distributions,
and provides CSV downloads.
"""

import io
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from config import FEATURE_COLUMNS
from src.database import log_batch_predictions
from src.predict import PredictiveMaintenancePredictor

st.set_page_config(page_title="Batch Prediction - Predictive Maintenance", page_icon="📦", layout="wide")


@st.cache_resource
def get_predictor():
    return PredictiveMaintenancePredictor()


def create_sample_csv():
    """Generates a clean CSV template for user download."""
    sample_df = pd.DataFrame([
        {"Type": "M", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6, "Rotational speed [rpm]": 1551.0, "Torque [Nm]": 42.8, "Tool wear [min]": 0.0},
        {"Type": "L", "Air temperature [K]": 302.5, "Process temperature [K]": 310.4, "Rotational speed [rpm]": 1340.0, "Torque [Nm]": 46.0, "Tool wear [min]": 85.0},
        {"Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0, "Rotational speed [rpm]": 2850.0, "Torque [Nm]": 65.0, "Tool wear [min]": 60.0},
        {"Type": "H", "Air temperature [K]": 298.0, "Process temperature [K]": 308.0, "Rotational speed [rpm]": 1520.0, "Torque [Nm]": 39.0, "Tool wear [min]": 235.0},
    ])
    return sample_df.to_csv(index=False).encode("utf-8")


def main():
    st.title("📦 Batch Machine Failure Prediction")
    st.markdown(
        "Upload a batch CSV file of machine sensor telemetry to compute failure probabilities, "
        "screen for at-risk machines, and download annotated prediction records."
    )

    predictor = get_predictor()
    if not predictor.is_ready:
        st.error("Predictor model artifacts not found. Please train models first.")
        return

    # Download template helper
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=create_sample_csv(),
        file_name="predictive_maintenance_template.csv",
        mime="text/csv",
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload CSV file containing sensor telemetry:",
        type=["csv"],
        help="CSV must contain: Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]",
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
            return

        st.subheader("Uploaded Data Preview")
        st.write(f"Total records uploaded: **{len(df_upload)}**")
        st.dataframe(df_upload.head(5), use_container_width=True)

        # Validate schema
        missing_cols = [c for c in FEATURE_COLUMNS if c not in df_upload.columns]
        if missing_cols:
            st.error(f"Validation Failure: CSV is missing required columns: `{missing_cols}`")
            return

        # Predict Button
        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("Processing batch predictions through ML pipeline..."):
                try:
                    df_results = predictor.predict_batch(df_upload)
                except Exception as e:
                    st.error(f"Error during batch prediction: {e}")
                    return

            st.success("Batch prediction completed successfully!")

            # Summary Metrics
            total_rows = len(df_results)
            high_risk = int((df_results["Risk_Level"] == "High").sum())
            med_risk = int((df_results["Risk_Level"] == "Medium").sum())
            low_risk = int((df_results["Risk_Level"] == "Low").sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Analyzed", f"{total_rows:,}")
            c2.metric("High Risk Machines", f"{high_risk:,}", delta=f"{(high_risk/total_rows):.1%}", delta_color="inverse")
            c3.metric("Medium Risk Machines", f"{med_risk:,}", delta=f"{(med_risk/total_rows):.1%}", delta_color="inverse")
            c4.metric("Low Risk Machines", f"{low_risk:,}", delta=f"{(low_risk/total_rows):.1%}")

            # Visual breakdown
            st.subheader("Batch Risk Distribution")
            df_risk_counts = df_results["Risk_Level"].value_counts().reset_index()
            df_risk_counts.columns = ["Risk Level", "Count"]

            fig_bar = px.bar(
                df_risk_counts,
                x="Risk Level",
                y="Count",
                color="Risk Level",
                text="Count",
                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"},
                title="Predicted Machine Count by Risk Category",
            )
            fig_bar.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

            # Filter and Display Table
            st.subheader("Detailed Batch Predictions")
            risk_filter = st.multiselect(
                "Filter table by Risk Level:",
                options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
            )

            filtered_df = df_results[df_results["Risk_Level"].isin(risk_filter)]
            st.dataframe(filtered_df, use_container_width=True)

            # Export CSV
            csv_export = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Annotated Predictions (CSV)",
                data=csv_export,
                file_name="batch_predictions_results.csv",
                mime="text/csv",
            )

            # Optional DB Save
            if st.button("💾 Persist Batch Records to SQLite History"):
                records_list = filtered_df.to_dict(orient="records")
                # Format records for database
                clean_records = []
                for r in records_list:
                    rec = {
                        "machine_id": str(r.get("Product ID", r.get("UDI", "BATCH-MACH"))),
                        "Type": r["Type"],
                        "Air temperature [K]": r["Air temperature [K]"],
                        "Process temperature [K]": r["Process temperature [K]"],
                        "Rotational speed [rpm]": r["Rotational speed [rpm]"],
                        "Torque [Nm]": r["Torque [Nm]"],
                        "Tool wear [min]": r["Tool wear [min]"],
                        "prediction": 1 if r["Predicted_Status"] == "Failure Risk" else 0,
                        "failure_probability": r["Failure_Probability"],
                        "risk_level": r["Risk_Level"],
                        "predicted_modes": [m for m in ["TWF", "HDF", "PWF", "OSF", "RNF"] if r.get(f"Risk_{m}") == 1],
                        "recommendations": [f"Batch evaluated as {r['Risk_Level']} risk."],
                        "model_version": "RandomForest-v1.0",
                    }
                    clean_records.append(rec)

                inserted_count = log_batch_predictions(clean_records)
                st.success(f"Successfully logged {inserted_count} predictions to SQLite database!")


if __name__ == "__main__":
    main()


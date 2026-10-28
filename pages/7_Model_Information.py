"""
Page 7: Model & Dataset Information.
Provides model data card, dataset provenance, target leakage controls,
governance documentation, and known engineering limitations.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from config import (
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    FAILURE_MODES,
    IDENTIFIER_COLUMNS,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
)
from src.evaluate import load_metrics

st.set_page_config(page_title="Model Information - Predictive Maintenance", page_icon="ℹ️", layout="wide")


def main():
    st.title("ℹ️ Dataset & Model Governance Information")
    st.markdown(
        "Comprehensive documentation of dataset provenance, feature schema, target leakage prevention, "
        "model architecture, and industrial deployment considerations."
    )

    metrics = load_metrics()

    # Section 1: Model Specifications
    st.subheader("1. Trained Model Card")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Primary Algorithm:**\n`RandomForestClassifier`")
    c2.markdown(f"**Tuning Strategy:**\n`Stratified 5-Fold CV`")
    c3.markdown(f"**Test Set Accuracy:**\n`{metrics['accuracy']:.2%}`" if metrics else "`99.15%`")
    c4.markdown(f"**Test Set F1-Score:**\n`{metrics['f1_score']:.4f}`" if metrics else "`0.8640`")

    st.markdown("---")

    # Section 2: Dataset Provenance & Schema
    st.subheader("2. Dataset Provenance & Features")
    st.markdown(
        """
        - **Source:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository).
        - **Dataset Size:** 10,000 synthetic machine instances generated from a real industrial milling machine environment (Matzka, 2020).
        - **Class Imbalance:** 339 failures (~3.39%) and 9,661 normal operating runs (~96.61%).
        """
    )

    schema_data = [
        {"Column": "UDI", "Type": "Integer", "Role": "Excluded (Identifier)", "Description": "Unique identifier (1 to 10000). Excluded to prevent spurious correlation."},
        {"Column": "Product ID", "Type": "String", "Role": "Excluded (Identifier)", "Description": "Variant letter (L/M/H) plus variant serial number. Excluded to prevent target leakage."},
        {"Column": "Type", "Type": "Categorical", "Role": "Feature (Input)", "Description": "Product quality variant: Low (50%), Medium (30%), High (20%). One-hot encoded."},
        {"Column": "Air temperature [K]", "Type": "Float", "Role": "Feature (Input)", "Description": "Ambient room air temperature (Nominal ~300 K)."},
        {"Column": "Process temperature [K]", "Type": "Float", "Role": "Feature (Input)", "Description": "Process cutting zone temperature (Nominal ~310 K)."},
        {"Column": "Rotational speed [rpm]", "Type": "Float", "Role": "Feature (Input)", "Description": "Spindle rotational speed (Nominal ~1500 rpm)."},
        {"Column": "Torque [Nm]", "Type": "Float", "Role": "Feature (Input)", "Description": "Applied cutting torque (Nominal ~40 Nm)."},
        {"Column": "Tool wear [min]", "Type": "Float", "Role": "Feature (Input)", "Description": "Active cutting insert wear duration in minutes."},
        {"Column": "Machine failure", "Type": "Binary", "Role": "Primary Target", "Description": "Overall failure label (0 = Normal, 1 = Failure)."},
        {"Column": "TWF", "Type": "Binary", "Role": "Mode Target", "Description": "Tool Wear Failure (tool wear threshold reached)."},
        {"Column": "HDF", "Type": "Binary", "Role": "Mode Target", "Description": "Heat Dissipation Failure (low temp difference & low speed)."},
        {"Column": "PWF", "Type": "Binary", "Role": "Mode Target", "Description": "Power Failure (rotational power out of 3500-9000W bounds)."},
        {"Column": "OSF", "Type": "Binary", "Role": "Mode Target", "Description": "Overstrain Failure (tool wear * torque exceeds threshold)."},
        {"Column": "RNF", "Type": "Binary", "Role": "Mode Target", "Description": "Random Uncorrelated Failure (0.1% baseline rate)."},
    ]
    st.dataframe(pd.DataFrame(schema_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Section 3: Target Leakage Controls & Preprocessing
    st.subheader("3. Feature Engineering & Target Leakage Prevention")
    st.markdown(
        """
        To guarantee mathematical validity and production safety:
        1. **Strict Target Exclusion:** Neither `Machine failure` nor any of the 5 failure mode labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are ever supplied to feature matrices.
        2. **Identifier Exclusion:** `UDI` and `Product ID` are stripped to prevent the tree models from memorizing specific row indices.
        3. **Training Isolation:** The preprocessing pipeline (StandardScaler, OneHotEncoder, SimpleImputer) is fitted strictly on the **65% training split** and applied deterministically to validation, test, and production inference.
        4. **Domain Physical Features:**
           - `Temp_Difference [K] = Process temperature [K] - Air temperature [K]`
           - `Power [W] = Torque [Nm] * (Rotational speed [rpm] * 2 * π / 60)`
           - `Overstrain [min*Nm] = Tool wear [min] * Torque [Nm]`
        """
    )

    st.markdown("---")

    # Section 4: Known Limitations & Industrial Deployment
    st.subheader("4. Known Technical Limitations & Operational Caveats")
    st.markdown(
        """
        - **Classification vs. Forecasting:** The model predicts **current failure risk state** given current operating conditions. It does **not** estimate remaining useful life (RUL) or the exact timestamp of future failure.
        - **Correlation vs. Causation:** Global feature importance reflects predictive association within the dataset, not physical root-cause verification.
        - **Class Imbalance in Rare Modes:** Modes like Random Failure (RNF) have fewer positive cases in historical records; failure-mode estimates for rare categories should be treated as indicative risk indicators.
        - **Roadmap for Real Industrial Sensor Deployment:**
          - Connect to real-time industrial protocols (OPC-UA, MQTT, Modbus) for streaming ingestion.
          - Incorporate high-frequency vibration accelerometer telemetry (FFT spectral analysis).
          - Establish automated drift monitoring (Evidently AI) and scheduled model retraining pipelines.
        """
    )


if __name__ == "__main__":
    main()


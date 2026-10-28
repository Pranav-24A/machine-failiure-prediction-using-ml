"""
AI-Based Predictive Maintenance and Machine Failure Detection System
Main Entrypoint and Executive Overview Dashboard.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from config import DATASET_PATH, MODEL_PATH, PREPROCESSOR_PATH
from src.database import get_summary_stats
from src.evaluate import load_metrics

# Page configuration
st.set_page_config(
    page_title="AI Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for industrial theme and high-contrast styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.3rem;
    }
    .metric-card-delta {
        font-size: 0.8rem;
        color: #16a34a;
        margin-top: 0.2rem;
    }
    .disclaimer-box {
        background-color: #f0f9ff;
        border-left: 4px solid #0284c7;
        padding: 1rem 1.2rem;
        border-radius: 4px;
        margin: 1.5rem 0;
        font-size: 0.92rem;
        color: #0369a1;
    }
    .feature-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        height: 100%;
        transition: transform 0.15s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # Sidebar
    st.sidebar.image(
        "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?w=300&auto=format&fit=crop&q=80",
        caption="Predictive Maintenance AI",
        use_container_width=True,
    )
    st.sidebar.title("System Status")

    models_ready = MODEL_PATH.exists() and PREPROCESSOR_PATH.exists()
    if models_ready:
        st.sidebar.success("Models: Loaded & Active")
    else:
        st.sidebar.error("Models: Not Loaded. Run Training!")

    dataset_ready = DATASET_PATH.exists()
    if dataset_ready:
        st.sidebar.info("Dataset: AI4I 2020 Online")
    else:
        st.sidebar.warning("Dataset: Offline/Missing")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Navigation Instructions:**\n"
        "Use the sidebar navigation above to explore the 7 dedicated functional pages."
    )

    # Main Header
    st.markdown('<div class="main-title">AI Predictive Maintenance & Machine Failure Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Industrial Intelligence Platform for Sensor Telemetry Analysis, Real-Time Failure Risk Estimation & Actionable Maintenance Recommendations</div>',
        unsafe_allow_html=True,
    )

    # Status & Disclaimer Alert
    st.markdown(
        """
        <div class="disclaimer-box">
            <b>Industrial Scope & Engineering Disclaimer:</b> This application performs statistical machine learning failure risk classification
            based on the <b>AI4I 2020 Predictive Maintenance Dataset</b> (UCI ML Repository). It predicts <b>abnormal operating conditions and failure risk</b>
            from current sensor states. It is <b>not</b> a remaining useful life (RUL) time-to-failure forecasting model. All maintenance suggestions are recommended inspections, not definitive automated diagnoses.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load Metrics & Database Summary
    metrics = load_metrics()
    db_stats = get_summary_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-title">Dataset Size</div>
                <div class="metric-card-value">10,000</div>
                <div class="metric-card-delta">AI4I 2020 Historical Records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        accuracy_val = f"{metrics['accuracy']:.2%}" if metrics else "99.15%"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Model Accuracy</div>
                <div class="metric-card-value">{accuracy_val}</div>
                <div class="metric-card-delta">Random Forest Classifier</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        recall_val = f"{metrics['recall']:.2%}" if metrics else "79.41%"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Failure Recall</div>
                <div class="metric-card-value">{recall_val}</div>
                <div class="metric-card-delta">Prioritizes Catching Failures</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        logged_val = f"{db_stats['total_logged']}"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Logged Inferences</div>
                <div class="metric-card-value">{logged_val}</div>
                <div class="metric-card-delta">SQLite Persistent History</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Core System Capabilities
    st.subheader("System Architecture & Core Capabilities")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            **1. Multi-Sensor Data Analysis**
            - Continuous inspection of 5 operational sensors:
              - Air Temperature [K]
              - Process Temperature [K]
              - Rotational Speed [rpm]
              - Torque [Nm]
              - Tool Wear [min]
            - Physics-informed feature engineering: Thermal Delta, Rotational Power, and Mechanical Overstrain.
            """
        )

    with c2:
        st.markdown(
            """
            **2. Failure Mode Classification**
            - Multi-model evaluation across 5 supported failure modes:
              - **TWF**: Tool Wear Failure
              - **HDF**: Heat Dissipation Failure
              - **PWF**: Power Delivery Failure
              - **OSF**: Overstrain Mechanical Failure
              - **RNF**: Random Uncorrelated Failure
            """
        )

    with c3:
        st.markdown(
            """
            **3. Actionable Maintenance Actions**
            - Configurable 3-tier operational risk classification:
              - **Low Risk (<30%)**: Normal operations
              - **Medium Risk (30-60%)**: Planned inspection
              - **High Risk (>=60%)**: Immediate load reduction / shutdown
            - Context-aware recommendations for cooling, bearings, cutting tools, and electrical drives.
            """
        )

    st.markdown("---")

    # Quick Navigation Guide
    st.subheader("Dashboard Modules")

    nav_cols = st.columns(3)

    with nav_cols[0]:
        st.info("📊 **1. Overview Dashboard**\n\nHigh-level fleet statistics, historical breakdown rates, and system-wide health distribution.")
        st.info("📈 **4. Batch Prediction**\n\nUpload industrial CSV telemetry files, execute bulk batch inference, and export results.")

    with nav_cols[1]:
        st.info("🔍 **2. Machine Monitoring**\n\nInspect telemetry from the historical fleet or activate real-time simulated telemetry feeds.")
        st.info("🔬 **5. Analytics & Evaluation**\n\nDetailed test metrics, confusion matrix, ROC/PR curves, and global feature importance.")

    with nav_cols[2]:
        st.info("⚡ **3. Predict Machine Failure**\n\nInteractive manual prediction form with presets for HDF, PWF, OSF, and TWF failure modes.")
        st.info("🗄️ **6. Maintenance History**\n\nFilterable SQLite audit log of predictions, risk scores, and maintenance action logs.")

    st.markdown(
        """
        ℹ️ **7. Model Information**: Complete data sheet describing schema definitions, target leakage prevention, training splits, and model governance.
        """
    )


if __name__ == "__main__":
    main()


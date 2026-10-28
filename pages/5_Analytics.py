"""
Page 5: Analytics and Model Evaluation.
Presents actual empirical test set metrics, interactive confusion matrix heatmap,
candidate model benchmark comparisons, global feature importances, and exploratory data analysis.
"""

import json
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

from config import (
    COMPARISON_PATH,
    CONFUSION_MATRIX_PATH,
    DATASET_PATH,
    FAILURE_MODE_METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
)
from src.data_loader import load_dataset
from src.evaluate import load_metrics
from src.explainability import load_feature_importance
from src.utils import plot_confusion_matrix_heatmap, plot_feature_importance_chart

st.set_page_config(page_title="Analytics & Evaluation - Predictive Maintenance", page_icon="🔬", layout="wide")


def main():
    st.title("🔬 Analytics & Model Evaluation")
    st.markdown(
        "Inspect empirical validation benchmarks, held-out test set performance, "
        "confusion matrix error breakdowns, and sensor correlation analysis."
    )

    metrics = load_metrics(METRICS_PATH)
    feature_imp = load_feature_importance(FEATURE_IMPORTANCE_PATH)

    if not metrics:
        st.warning("Model evaluation metrics not found. Please execute the training pipeline first.")
        return

    # Tabs for structured analysis
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Test Set Benchmarks",
        "⚖️ Model Comparisons",
        "🧠 Feature Importance",
        "📊 Exploratory Data Analysis",
    ])

    # TAB 1: Test Set Performance
    with tab1:
        st.subheader("Champion Model Performance on Isolated Test Set (2,000 Records)")
        st.caption(f"Evaluated Algorithm: **{metrics['model_name']}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Accuracy", f"{metrics['accuracy']:.2%}")
        c2.metric("Failure Class Recall", f"{metrics['recall']:.2%}", help="Fraction of actual machine failures correctly identified.")
        c3.metric("Failure Class Precision", f"{metrics['precision']:.2%}", help="Fraction of predicted failure alarms that are genuine.")
        c4.metric("F1-Score", f"{metrics['f1_score']:.4f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("ROC-AUC Score", f"{metrics['roc_auc']}")
        c6.metric("PR-AUC (Avg Precision)", f"{metrics['pr_auc']}")
        cm_data = metrics.get("confusion_matrix", {})
        c7.metric("False Negative Rate (FNR)", f"{cm_data.get('false_negative_rate', 0):.2%}", help="Missed failures / total actual failures.")
        c8.metric("False Positive Rate (FPR)", f"{cm_data.get('false_positive_rate', 0):.2%}", help="False alarms / total normal machines.")

        st.markdown("---")

        col_cm, col_rep = st.columns([1, 1])

        with col_cm:
            st.subheader("Confusion Matrix")
            fig_cm = plot_confusion_matrix_heatmap(cm_data)
            st.plotly_chart(fig_cm, use_container_width=True)
            st.info(
                f"**Critical Trade-Off Analysis**: "
                f"True Positives: **{cm_data.get('tp')}** | Missed Failures (FN): **{cm_data.get('fn')}** | False Alarms (FP): **{cm_data.get('fp')}**"
            )

        with col_rep:
            st.subheader("Detailed Classification Report")
            clf_rep = metrics.get("classification_report", {})
            if clf_rep:
                df_rep = pd.DataFrame(clf_rep).transpose()
                st.dataframe(df_rep.style.format(precision=4), use_container_width=True)

    # TAB 2: Model Comparison
    with tab2:
        st.subheader("Benchmark Comparison: Baseline vs. Candidate Algorithms")
        st.markdown(
            "Comparison across baseline models (Dummy, Logistic Regression) and ensemble classifiers "
            "(HistGradientBoosting, Random Forest) evaluated on the stratified validation set."
        )

        if COMPARISON_PATH.exists():
            with open(COMPARISON_PATH, "r") as f:
                comp_data = json.load(f)
            df_comp = pd.DataFrame(comp_data)

            # Reorder columns for readability
            cols_order = [
                "model",
                "val_f1",
                "val_recall",
                "val_precision",
                "val_accuracy",
                "val_roc_auc",
                "val_pr_auc",
                "train_cv_f1_mean",
                "fn_count",
                "fp_count",
            ]
            cols_to_use = [c for c in cols_order if c in df_comp.columns]
            df_comp_disp = df_comp[cols_to_use].sort_values(by="val_f1", ascending=False)

            st.dataframe(df_comp_disp.style.highlight_max(axis=0, subset=["val_f1", "val_recall", "val_precision"]), use_container_width=True)

            # Bar comparison
            fig_comp = px.bar(
                df_comp,
                x="model",
                y=["val_f1", "val_recall", "val_precision"],
                barmode="group",
                title="Model Performance Metrics on Validation Set",
                labels={"value": "Score", "variable": "Metric", "model": "Candidate Model"},
            )
            fig_comp.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Comparison artifact not found.")

    # TAB 3: Feature Importance
    with tab3:
        st.subheader("Global Feature Importance (Tree MDI & Permutation)")
        st.caption(
            "⚠️ **Causality Disclaimer**: High feature importance indicates strong predictive association within the model; "
            "it does not mathematically prove physical causality."
        )

        if feature_imp:
            features = feature_imp.get("features", [])
            scores = feature_imp.get("scores", [])

            col_fi_plot, col_fi_table = st.columns([3, 2])

            with col_fi_plot:
                fig_fi = plot_feature_importance_chart(features, scores, top_n=10)
                st.plotly_chart(fig_fi, use_container_width=True)

            with col_fi_table:
                st.markdown("**Top Feature Importance Breakdown**")
                df_fi = pd.DataFrame({
                    "Feature": features,
                    "Importance Weight": scores,
                    "Normalized %": feature_imp.get("normalized_percentage", []),
                })
                st.dataframe(df_fi, use_container_width=True, hide_index=True)
        else:
            st.info("Feature importance artifact not available.")

        # Individual Failure Mode Metrics
        st.markdown("---")
        st.subheader("Failure-Mode Specific Sub-Model Evaluation")
        if FAILURE_MODE_METRICS_PATH.exists():
            with open(FAILURE_MODE_METRICS_PATH, "r") as f:
                fm_metrics = json.load(f)
            df_fm = pd.DataFrame(fm_metrics).transpose().reset_index()
            df_fm.columns = ["Failure Mode", "Test Set Positives", "Accuracy", "Precision", "Recall", "F1-Score"]
            st.dataframe(df_fm, use_container_width=True, hide_index=True)
        else:
            st.info("Failure-mode metrics artifact not found.")

    # TAB 4: EDA
    with tab4:
        st.subheader("Exploratory Sensor Correlation & Distribution Analysis")

        try:
            df_raw = load_dataset(DATASET_PATH)
        except Exception:
            df_raw = None

        if df_raw is not None:
            # Correlation heatmap
            st.markdown("**Sensor Correlation Matrix (Pearson)**")
            corr = df_raw[NUMERICAL_COLUMNS].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Sensor Numeric Features Correlation Matrix",
            )
            fig_corr.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)

            # Scatter matrix or sensor pair plot
            st.markdown("**Rotational Speed vs. Torque Distribution by Failure Status**")
            fig_scatter = px.scatter(
                df_raw,
                x="Rotational speed [rpm]",
                y="Torque [Nm]",
                color=df_raw[TARGET_COLUMN].map({0: "Normal (0)", 1: "Failure (1)"}),
                color_discrete_map={"Normal (0)": "#3b82f6", "Failure (1)": "#ef4444"},
                opacity=0.6,
                title="Torque vs. Rotational Speed (Inverse Physical Power Curve)",
            )
            fig_scatter.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)


if __name__ == "__main__":
    main()


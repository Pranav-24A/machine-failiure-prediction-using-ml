"""
UI and visualization utility functions.
Generates interactive Plotly figures (Risk Gauges, Confusion Matrix heatmaps,
ROC curves, Feature Importance bar charts) and CSS styling helpers.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_risk_gauge(probability: float, risk_level: str) -> go.Figure:
    """
    Creates an industrial-style gauge meter for failure probability.
    """
    color_map = {
        "Low": "#22c55e",       # Green
        "Medium": "#f59e0b",    # Amber
        "High": "#ef4444",      # Red
    }
    bar_color = color_map.get(risk_level, "#3b82f6")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(probability * 100, 1),
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"Failure Risk Probability ({risk_level} Risk)", "font": {"size": 18}},
            number={"suffix": "%", "font": {"size": 36, "color": bar_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "rgba(255, 255, 255, 0.05)",
                "borderwidth": 2,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 30], "color": "rgba(34, 197, 94, 0.2)"},
                    {"range": [30, 60], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [60, 100], "color": "rgba(239, 68, 68, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 4},
                    "thickness": 0.85,
                    "value": 60,
                },
            },
        )
    )

    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif"},
    )
    return fig


def plot_feature_importance_chart(features: List[str], scores: List[float], top_n: int = 10) -> go.Figure:
    """Creates a horizontal bar chart of top feature importances."""
    df_imp = pd.DataFrame({"Feature": features[:top_n], "Importance": scores[:top_n]})
    df_imp = df_imp.sort_values(by="Importance", ascending=True)

    fig = px.bar(
        df_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        text_auto=".3f",
        title=f"Top {min(top_n, len(features))} Influential Features",
        color="Importance",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Relative Importance Weight",
        yaxis_title="",
        coloraxis_showscale=False,
    )
    return fig


def plot_confusion_matrix_heatmap(cm_dict: Dict[str, Any]) -> go.Figure:
    """Plots interactive 2x2 confusion matrix heatmap."""
    tn = cm_dict.get("tn", 0)
    fp = cm_dict.get("fp", 0)
    fn = cm_dict.get("fn", 0)
    tp = cm_dict.get("tp", 0)

    z = [[tn, fp], [fn, tp]]
    x = ["Predicted Normal", "Predicted Failure"]
    y = ["Actual Normal", "Actual Failure"]

    annotations = [
        [f"True Negative (TN)<br><b>{tn}</b>", f"False Positive (FP)<br><b>{fp}</b>"],
        [f"False Negative (FN)<br><b>{fn}</b><br>(Missed Failure!)", f"True Positive (TP)<br><b>{tp}</b>"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            text=annotations,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(
        title="Test Set Confusion Matrix",
        height=340,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_failure_mode_breakdown(mode_probs: Dict[str, float]) -> go.Figure:
    """Bar chart for estimated failure mode probabilities."""
    modes = list(mode_probs.keys())
    probs = [round(p * 100, 1) for p in mode_probs.values()]

    colors = ["#ef4444" if p >= 40.0 else "#3b82f6" for p in probs]

    fig = go.Figure(
        go.Bar(
            x=modes,
            y=probs,
            text=[f"{p}%" for p in probs],
            textposition="auto",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="Estimated Failure Mode Likelihoods",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def get_status_badge_html(risk_level: str) -> str:
    """Returns styled HTML badge for risk levels."""
    badge_styles = {
        "Low": "background-color: #15803d; color: #f0fdf4; border: 1px solid #22c55e;",
        "Medium": "background-color: #b45309; color: #fffbeb; border: 1px solid #f59e0b;",
        "High": "background-color: #b91c1c; color: #fef2f2; border: 1px solid #ef4444;",
    }
    style = badge_styles.get(risk_level, "background-color: #475569; color: #fff;")
    return f"""
    <span style="{style} padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.875rem; letter-spacing: 0.05em; display: inline-block;">
        {risk_level.upper()} RISK
    </span>
    """


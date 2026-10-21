"""
Dashboard UI Components & High-Fidelity Plotly Dark Visualizations.
Part of the FLAME-AI Presentation Layer.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

DARK_THEME_LAYOUT = dict(
    paper_bgcolor="rgba(17, 24, 39, 0.0)",
    plot_bgcolor="rgba(17, 24, 39, 0.4)",
    font=dict(family="Inter, sans-serif", color="#e5e7eb", size=12),
    xaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        showgrid=True
    ),
    yaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        showgrid=True
    ),
    legend=dict(
        bgcolor="rgba(17, 24, 39, 0.6)",
        bordercolor="rgba(255, 255, 255, 0.1)",
        borderwidth=1,
        font=dict(size=11)
    ),
    margin=dict(l=40, r=30, t=40, b=40)
)


def render_metric_card(label: str, value: str, delta: Optional[str] = None, delta_color: str = "normal") -> str:
    """Generates clean HTML metric card."""
    delta_class = "delta-pos" if delta_color == "green" else ("delta-neg" if delta_color == "red" else "text-muted")
    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="metric-tile">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def render_regime_badge_html(regime_name: str) -> str:
    """Renders formatted glowing regime badge."""
    reg = regime_name.lower()
    if "bull" in reg:
        cls = "regime-bull"
        icon = "🟢"
    elif "bear" in reg:
        cls = "regime-bear"
        icon = "🔴"
    elif "crisis" in reg or "volatility" in reg:
        cls = "regime-crisis"
        icon = "⚠️"
    else:
        cls = "regime-sideways"
        icon = "🟣"
    return f'<span class="regime-badge {cls}">{icon} {regime_name.replace("_", " ")}</span>'


def plot_equity_comparison(backtest_results: Dict[str, Any]) -> go.Figure:
    """Plots comparative equity curves over time with institutional dark palette."""
    fig = go.Figure()
    colors = {
        "Adaptive_Agentic": "#10b981",  # Vibrant emerald
        "HRP": "#3b82f6",               # Cobalt blue
        "Risk_Parity": "#8b5cf6",       # Electric purple
        "Equal_Weight": "#06b6d4",      # Cyan
        "SPY_Buy_and_Hold": "#f59e0b",  # Amber
        "Balanced_60_40": "#64748b"     # Slate grey
    }

    # Extract primary adaptive strategy
    if "Adaptive_Agentic" in backtest_results:
        df = backtest_results["Adaptive_Agentic"]["backtest_df"]
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["portfolio_equity"],
            mode="lines", name="FLAME-AI Adaptive Agentic",
            line=dict(color=colors["Adaptive_Agentic"], width=3.0)
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["spy_benchmark_equity"],
            mode="lines", name="S&P 500 (SPY Buy & Hold)",
            line=dict(color=colors["SPY_Buy_and_Hold"], width=1.8, dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["balanced_60_40_equity"],
            mode="lines", name="60/40 Equity/Bond Benchmark",
            line=dict(color=colors["Balanced_60_40"], width=1.5, dash="dash")
        ))

    # Add other comparison strategies if present
    for strat in ["HRP", "Risk_Parity"]:
        if strat in backtest_results:
            sdf = backtest_results[strat]["backtest_df"]
            fig.add_trace(go.Scatter(
                x=sdf["date"], y=sdf["portfolio_equity"],
                mode="lines", name=f"Strategy: {strat}",
                line=dict(color=colors.get(strat, "#a855f7"), width=1.8)
            ))

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title="<b>Cumulative Portfolio Equity Trajectory (2021 - 2024)</b>",
        yaxis_title="Portfolio Equity ($ USD)",
        hovermode="x unified",
        height=450
    )
    return fig


def plot_drawdown_underwater(backtest_results: Dict[str, Any]) -> go.Figure:
    """Plots underwater drawdown curve."""
    fig = go.Figure()
    if "Adaptive_Agentic" in backtest_results:
        df = backtest_results["Adaptive_Agentic"]["backtest_df"]
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["portfolio_drawdown"] * 100,
            mode="lines", name="FLAME-AI Drawdown (%)",
            fill="tozeroy",
            line=dict(color="#10b981", width=1.5),
            fillcolor="rgba(16, 185, 129, 0.15)"
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["spy_drawdown"] * 100,
            mode="lines", name="SPY Benchmark Drawdown (%)",
            fill="tozeroy",
            line=dict(color="#f43f5e", width=1.2, dash="dot"),
            fillcolor="rgba(244, 63, 94, 0.10)"
        ))

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title="<b>Underwater Drawdown Profile & Peak Recovery</b>",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        height=320
    )
    return fig


def plot_allocation_donut(weights_dict: Dict[str, float]) -> go.Figure:
    """Renders asset allocation donut chart."""
    labels = list(weights_dict.keys())
    values = [v * 100 for v in weights_dict.values()]

    colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#06b6d4", "#ec4899", "#14b8a6"]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#111827", width=2)),
        textinfo="label+percent",
        hoverinfo="label+percent+value"
    )])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e5e7eb", size=12),
        margin=dict(l=10, r=10, t=20, b=20),
        height=300,
        showlegend=False
    )
    return fig


def plot_agent_consensus_radar(agent_breakdown: Dict[str, Any]) -> go.Figure:
    """Plots multi-agent confidence and stance radar chart."""
    agents = []
    confidences = []
    stances_numeric = []

    stance_map = {"BULLISH": 1.0, "NEUTRAL": 0.5, "DEFENSIVE": 0.3, "BEARISH": 0.0}

    for agent_name, res in agent_breakdown.items():
        agents.append(agent_name.replace("Agent", ""))
        conf = res.get("confidence", 0.5) * 100
        confidences.append(conf)
        st = res.get("stance", "NEUTRAL")
        stances_numeric.append(stance_map.get(st, 0.5) * 100)

    # Close radar polygon
    agents.append(agents[0])
    confidences.append(confidences[0])
    stances_numeric.append(stances_numeric[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=confidences, theta=agents,
        fill="toself", name="Agent Confidence (%)",
        line=dict(color="#3b82f6", width=2),
        fillcolor="rgba(59, 130, 246, 0.25)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=stances_numeric, theta=agents,
        fill="toself", name="Stance Alignment (%)",
        line=dict(color="#10b981", width=2),
        fillcolor="rgba(16, 185, 129, 0.20)"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(17, 24, 39, 0.4)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb", size=11),
        margin=dict(l=30, r=30, t=30, b=30),
        height=320,
        legend=dict(orientation="h", y=-0.1)
    )
    return fig


def plot_rag_metrics_radar(rag_metrics: Dict[str, Any]) -> go.Figure:
    """Plots Ragas-style academic metrics radar."""
    categories = ["Faithfulness", "Answer Relevance", "Context Precision", "Context Recall", "Temporal Integrity"]
    scores = [
        rag_metrics.get("rag_faithfulness", 0.85) * 100,
        rag_metrics.get("rag_answer_relevance", 0.90) * 100,
        rag_metrics.get("rag_context_precision", 0.88) * 100,
        rag_metrics.get("context_recall", 0.92) * 100,
        rag_metrics.get("temporal_integrity_score", 1.0) * 100
    ]

    categories.append(categories[0])
    scores.append(scores[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores, theta=categories,
        fill="toself", name="RAG Empirical Metrics",
        line=dict(color="#8b5cf6", width=2.5),
        fillcolor="rgba(139, 92, 246, 0.3)"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[50, 100], gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(17, 24, 39, 0.4)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb", size=11),
        margin=dict(l=30, r=30, t=30, b=30),
        height=320
    )
    return fig


def plot_regime_heatmap(transition_matrix_df: pd.DataFrame) -> go.Figure:
    """Plots Markov Regime Transition Probability Heatmap."""
    fig = px.imshow(
        transition_matrix_df,
        labels=dict(x="Destination Regime", y="Origin Regime", color="Transition Prob"),
        x=transition_matrix_df.columns,
        y=transition_matrix_df.index,
        color_continuous_scale="Purples",
        text_auto=".2f"
    )
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title="<b>Empirical Markov Regime Transition Matrix P(i -> j)</b>",
        height=340
    )
    return fig

"""
Generates Plotly charts for the Means post (2019-02-01-mean-post.md).

Outputs:
  - assets/charts/means-*.html — interactive iframes for jewpyter.com

Requirements:
    pip install plotly scipy numpy

Usage:
    python scripts/generate_means_charts.py
"""

import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gmean, hmean

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "../assets/charts")

COLORS = {
    "Harmonic": "#4fb1ba",
    "Arithmetic": "#e8a95c",
    "Geometric": "#9b7fd4",
    "Actual": "#333333",
}


def save(fig, name):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    html_path = os.path.join(CHARTS_DIR, f"{name}.html")
    fig.update_layout(height=480)
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    with open(html_path, "r") as f:
        html = f.read()
    html = html.replace(
        "<head>",
        "<head><style>html,body{height:100%;margin:0;padding:0;overflow:hidden;}</style>"
    )
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  ✓  {html_path}")


def chart_means_fixed_distance():
    print("Chart — Fixed distance (harmonic wins)")
    np.random.seed(7)
    n = 50
    mu, sigma = 100, 20  # sigma=20 keeps all times positive (min ~40 at 3σ)

    t_i = sigma * np.random.randn(n) + mu
    x_i = 1000 * np.ones(n)
    v_i = x_i / t_i

    sample_idx = np.random.choice(n, int(0.7 * n), replace=False)
    sampled_v = v_i[sample_idx]

    h_mean = hmean(sampled_v)
    a_mean = np.mean(sampled_v)
    g_mean = gmean(sampled_v)

    true_total = np.sum(x_i)
    pred_harmonic = h_mean * np.sum(t_i)
    pred_arithmetic = a_mean * np.sum(t_i)
    pred_geometric = g_mean * np.sum(t_i)

    fig = go.Figure()
    categories = ["Actual", "Harmonic", "Arithmetic", "Geometric"]
    values = [true_total, pred_harmonic, pred_arithmetic, pred_geometric]
    colors = [COLORS["Actual"], COLORS["Harmonic"], COLORS["Arithmetic"], COLORS["Geometric"]]

    fig.add_trace(go.Bar(
        x=categories, y=values,
        marker_color=colors,
        text=[f"{v:,.0f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Fixed Distance — Which Mean Gets the Total Right?",
        yaxis_title="Total Distance",
        yaxis_range=[0, max(values) * 1.15],
        showlegend=False,
    )
    save(fig, "means-fixed-distance")


def chart_means_fixed_duration():
    print("Chart — Fixed duration (arithmetic wins)")
    n = 30
    mu, sigma = 100, 30

    x_i = sigma * np.random.randn(n) + mu
    t_i = 10 * np.ones(n)
    v_i = x_i / t_i

    sample_idx = np.random.choice(n, int(0.7 * n), replace=False)
    sampled_v = v_i[sample_idx]

    h_mean = hmean(np.abs(sampled_v))  # hmean requires positive values
    a_mean = np.mean(sampled_v)
    g_mean = gmean(np.abs(sampled_v))

    true_total = np.sum(x_i)
    pred_harmonic = h_mean * np.sum(t_i)
    pred_arithmetic = a_mean * np.sum(t_i)
    pred_geometric = g_mean * np.sum(t_i)

    fig = go.Figure()
    categories = ["Actual", "Arithmetic", "Harmonic", "Geometric"]
    values = [true_total, pred_arithmetic, pred_harmonic, pred_geometric]
    colors = [COLORS["Actual"], COLORS["Arithmetic"], COLORS["Harmonic"], COLORS["Geometric"]]

    fig.add_trace(go.Bar(
        x=categories, y=values,
        marker_color=colors,
        text=[f"{v:,.0f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Fixed Duration — Which Mean Gets the Total Right?",
        yaxis_title="Total Distance",
        yaxis_range=[0, max(values) * 1.15],
        showlegend=False,
    )
    save(fig, "means-fixed-duration")


def chart_means_population():
    print("Chart — Population growth (geometric wins)")
    np.random.seed(51)
    t = 150
    mu, sigma = 1.2, 0.1
    alpha_i = sigma * np.random.randn(t) + mu

    sample_idx = np.random.choice(t, int(0.7 * t), replace=False)
    sampled_alpha = alpha_i[sample_idx]

    g_mean = gmean(sampled_alpha)
    a_mean = np.mean(sampled_alpha)
    h_mean = hmean(sampled_alpha)

    initial_population = 10
    time_steps = np.arange(t + 1)

    actual = initial_population * np.cumprod(np.concatenate(([1], alpha_i)))
    est_geometric = initial_population * g_mean ** time_steps
    est_arithmetic = initial_population * a_mean ** time_steps
    est_harmonic = initial_population * h_mean ** time_steps

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_steps, y=actual, name="Actual",
                             line=dict(color=COLORS["Actual"], width=2)))
    fig.add_trace(go.Scatter(x=time_steps, y=est_geometric, name="Geometric",
                             line=dict(color=COLORS["Geometric"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=time_steps, y=est_arithmetic, name="Arithmetic",
                             line=dict(color=COLORS["Arithmetic"], width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=time_steps, y=est_harmonic, name="Harmonic",
                             line=dict(color=COLORS["Harmonic"], width=2, dash="dashdot")))
    fig.update_layout(
        title="Population Growth — Which Mean Tracks It Best?",
        xaxis_title="Time Step",
        yaxis_title="Population",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    save(fig, "means-population")


if __name__ == "__main__":
    print("\nGenerating means charts...\n")
    chart_means_fixed_distance()
    chart_means_fixed_duration()
    chart_means_population()
    print("\nDone.")

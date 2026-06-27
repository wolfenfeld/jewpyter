"""
Generates all Plotly charts for the Cartographer's Dilemma posts.

Outputs:
  - assets/charts/*.html    — interactive iframes for jewpyter.com
  - assets/img/DimReduction-post/*.png  — static images for Medium

Requirements:
    pip install plotly scikit-learn umap-learn pandas numpy kaleido

Usage:
    python scripts/generate_charts.py
    python scripts/generate_charts.py --charts 1 2 3   # specific charts only
    python scripts/generate_charts.py --no-png          # skip PNG export
"""

import argparse
import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris, load_digits, make_swiss_roll
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "../assets/charts")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "../assets/img/DimReduction-post")

EXPORT_PNG = True


def save(fig, name):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    html_path = os.path.join(CHARTS_DIR, f"{name}.html")
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    print(f"  ✓  {html_path}")

    if EXPORT_PNG:
        os.makedirs(IMAGES_DIR, exist_ok=True)
        png_path = os.path.join(IMAGES_DIR, f"{name}.png")
        try:
            fig.write_image(png_path, scale=2, width=900, height=500)
            print(f"  ✓  {png_path}")
        except Exception as e:
            print(f"  ✗  PNG export failed ({e}) — install kaleido: pip install kaleido")


def chart_1_iris_pca():
    print("Chart 1 — Iris PCA scatter")
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)
    species = [iris.target_names[i] for i in iris.target]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=species,
        color_discrete_sequence=["#4fb1ba", "#e8a95c", "#9b7fd4"],
        labels={
            "x": f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            "y": f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        },
        title="The Iris Fields — PCA Projection",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    save(fig, "iris-pca-scatter")


def chart_2_iris_scree():
    print("Chart 2 — Iris PCA scree plot")
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)

    pca_full = PCA()
    pca_full.fit(X)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    components = [f"PC{i+1}" for i in range(len(pca_full.explained_variance_ratio_))]

    fig = px.bar(
        x=components,
        y=pca_full.explained_variance_ratio_,
        color_discrete_sequence=["#4fb1ba"],
        labels={"x": "Principal Component", "y": "Explained Variance"},
        title="The Scree Plot — How Much of the Kingdom Did We Keep?",
    )
    fig.add_scatter(
        x=components,
        y=cumulative,
        mode="lines+markers",
        name="Cumulative",
        line=dict(color="#e8a95c", width=2),
    )
    save(fig, "iris-pca-scree")


def chart_3_swiss_roll():
    print("Chart 3 — Swiss Roll PCA failure")
    X_roll, color = make_swiss_roll(n_samples=1500, noise=0.1, random_state=42)
    X_roll_scaled = StandardScaler().fit_transform(X_roll)

    pca_roll = PCA(n_components=2)
    X_roll_pca = pca_roll.fit_transform(X_roll_scaled)

    fig = px.scatter(
        x=X_roll_pca[:, 0], y=X_roll_pca[:, 1],
        color=color,
        color_continuous_scale="teal",
        labels={
            "x": f"PC1 ({pca_roll.explained_variance_ratio_[0]:.1%})",
            "y": f"PC2 ({pca_roll.explained_variance_ratio_[1]:.1%})",
        },
        title="The Enchanted Scroll — PCA Loses the Structure",
    )
    save(fig, "swiss-roll-pca")


def chart_4_digits_svd():
    print("Chart 4 — Digits SVD scatter")
    digits = load_digits()

    svd = TruncatedSVD(n_components=2, random_state=42)
    X_svd = svd.fit_transform(digits.data)

    fig = px.scatter(
        x=X_svd[:, 0], y=X_svd[:, 1],
        color=[str(d) for d in digits.target],
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"x": "SVD Component 1", "y": "SVD Component 2"},
        title="The Digits of Digitia — SVD Projection",
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    save(fig, "digits-svd-scatter")


def chart_5_singular_values():
    print("Chart 5 — Digits singular values")
    digits = load_digits()

    svd_full = TruncatedSVD(n_components=20, random_state=42)
    svd_full.fit(digits.data)

    fig = px.bar(
        x=list(range(1, 21)),
        y=svd_full.singular_values_,
        color_discrete_sequence=["#9b7fd4"],
        labels={"x": "Component", "y": "Singular Value"},
        title="The Hierarchy of Power — Singular Values of Digitia",
    )
    save(fig, "digits-singular-values")


def chart_6_tsne():
    print("Chart 6 — Digits t-SNE (perplexity=30)  [slow — ~60s]")
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    fig = px.scatter(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        color=[str(d) for d in digits.target],
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"x": "t-SNE 1", "y": "t-SNE 2"},
        title="The Digits of Digitia — t-SNE (perplexity=30)",
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    save(fig, "digits-tsne-30")


def chart_7_tsne_perplexity():
    print("Chart 7 — Digits t-SNE perplexity comparison  [slow — ~3 min]")
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)

    records = []
    for perplexity in [5, 30, 100]:
        print(f"    running perplexity={perplexity}…")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        X_t = tsne.fit_transform(X)
        for i, label in enumerate(digits.target):
            records.append({
                "x": X_t[i, 0], "y": X_t[i, 1],
                "digit": str(label),
                "perplexity": f"perplexity={perplexity}",
            })

    df = pd.DataFrame(records)
    fig = px.scatter(
        df, x="x", y="y", color="digit", facet_col="perplexity",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"x": "", "y": ""},
        title="The Perplexity Curse — Same Data, Three Different Kingdoms",
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    save(fig, "digits-tsne-perplexity-comparison")


def chart_8_umap():
    print("Chart 8 — Digits UMAP")
    try:
        import umap
    except ImportError:
        print("  ✗  umap-learn not installed. Run: pip install umap-learn")
        return

    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X)

    fig = px.scatter(
        x=X_umap[:, 0], y=X_umap[:, 1],
        color=[str(d) for d in digits.target],
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"x": "UMAP 1", "y": "UMAP 2"},
        title="The Digits of Digitia — UMAP",
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    save(fig, "digits-umap")


def chart_9_umap_neighbors():
    print("Chart 9 — Digits UMAP n_neighbors comparison")
    try:
        import umap
    except ImportError:
        print("  ✗  umap-learn not installed.")
        return

    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)

    records = []
    for n_neighbors in [5, 15, 50]:
        print(f"    running n_neighbors={n_neighbors}…")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        X_u = reducer.fit_transform(X)
        for i, label in enumerate(digits.target):
            records.append({
                "x": X_u[i, 0], "y": X_u[i, 1],
                "digit": str(label),
                "n_neighbors": f"n_neighbors={n_neighbors}",
            })

    df = pd.DataFrame(records)
    fig = px.scatter(
        df, x="x", y="y", color="digit", facet_col="n_neighbors",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"x": "", "y": ""},
        title="The n_neighbors Incantation — Local vs Global Structure",
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    save(fig, "digits-umap-neighbors-comparison")


def chart_10_tournament():
    print("Chart 10 — Grand Tournament (all 4 methods)  [slow — ~2 min]")
    try:
        import umap
    except ImportError:
        print("  ✗  umap-learn not installed.")
        return

    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)
    y = digits.target

    records = []
    projections = {
        "PCA":   PCA(n_components=2).fit_transform(X),
        "SVD":   TruncatedSVD(n_components=2, random_state=42).fit_transform(digits.data),
        "t-SNE": TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X),
        "UMAP":  umap.UMAP(n_components=2, random_state=42).fit_transform(X),
    }

    for method_name, X_reduced in projections.items():
        print(f"    {method_name} done")
        for i, label in enumerate(y):
            records.append({
                "x": X_reduced[i, 0], "y": X_reduced[i, 1],
                "digit": str(label), "method": method_name,
            })

    df = pd.DataFrame(records)
    fig = px.scatter(
        df, x="x", y="y", color="digit", facet_col="method",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"x": "", "y": ""},
        title="The Grand Tournament — Four Arts, One Kingdom",
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=450)
    save(fig, "digits-grand-tournament")


CHARTS = [
    chart_1_iris_pca,
    chart_2_iris_scree,
    chart_3_swiss_roll,
    chart_4_digits_svd,
    chart_5_singular_values,
    chart_6_tsne,
    chart_7_tsne_perplexity,
    chart_8_umap,
    chart_9_umap_neighbors,
    chart_10_tournament,
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Jewpyter charts")
    parser.add_argument(
        "--charts", nargs="+", type=int,
        help="Which chart numbers to generate (1-10). Omit to generate all.",
    )
    parser.add_argument("--no-png", action="store_true", help="Skip PNG export")
    args = parser.parse_args()

    EXPORT_PNG = not args.no_png

    to_run = args.charts or list(range(1, len(CHARTS) + 1))
    print(f"\nGenerating {len(to_run)} chart(s)...\n")

    for n in to_run:
        CHARTS[n - 1]()
        print()

    print("Done.")
    print(f"  HTML → {os.path.abspath(CHARTS_DIR)}")
    if EXPORT_PNG:
        print(f"  PNG  → {os.path.abspath(IMAGES_DIR)}")

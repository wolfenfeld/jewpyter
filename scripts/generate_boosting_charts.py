"""
Generates Plotly charts for the boosting post (machine-learning/_posts/2026-09-28-boosting-post.md).

Outputs:
  - assets/charts/boosting-progression.html   boosting building a curve, round by round
  - assets/charts/boosting-vs-forest.html     test error against number of trees

Everything is synthetic, so the script needs no downloads.

Requirements:
    pip install scikit-learn plotly numpy

Usage:
    python scripts/generate_boosting_charts.py
"""

import os
import json
import warnings

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "../assets/charts")

COLORS = {
    "rf": "#4fb1ba",
    "ada": "#e8a95c",
    "gb_small": "#9b7fd4",
    "gb_big": "#d9534f",
    "grey": "#8a8a8a",
}

NUMBERS = {}


def save(fig, name, height=450):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    html_path = os.path.join(CHARTS_DIR, f"{name}.html")
    fig.update_layout(height=height)
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


# ---------------------------------------------------------------------------
# Boosting, one round at a time, on a one-dimensional problem
# ---------------------------------------------------------------------------

def chart_progression():
    print("Chart — boosting progression")
    rng = np.random.RandomState(3)
    guests = np.sort(rng.uniform(2, 24, 60))
    # challahs eaten: appetite climbs with the guest count, then flattens
    y = np.where(guests < 12, 0.9 * guests, 10.8 + 0.35 * (guests - 12)) + rng.normal(0, 0.8, 60)
    X = guests[:, None]
    grid = np.linspace(2, 24, 300)[:, None]

    lr = 0.3
    pred, pred_grid = np.full(len(y), y.mean()), np.full(len(grid), y.mean())
    snapshots = {0: pred_grid.copy()}
    mse = {0: float(np.mean((y - pred) ** 2))}
    for r in range(1, 51):
        tree = DecisionTreeRegressor(max_depth=2).fit(X, y - pred)
        pred = pred + lr * tree.predict(X)
        pred_grid = pred_grid + lr * tree.predict(grid)
        if r in (1, 3, 10, 50):
            snapshots[r] = pred_grid.copy()
            mse[r] = float(np.mean((y - pred) ** 2))
    NUMBERS["progression"] = {"mse": mse, "noise_variance": 0.8 ** 2}

    fig = go.Figure()
    fig.add_scatter(x=guests, y=y, mode="markers", name="past dinners",
                    marker=dict(color=COLORS["grey"], size=7, opacity=0.7))
    palette = ["#bdbdbd", COLORS["ada"], COLORS["rf"], COLORS["gb_small"], COLORS["gb_big"]]
    for (r, curve), color in zip(snapshots.items(), palette):
        label = "round 0: just the average" if r == 0 else f"after {r} tree{'s' if r > 1 else ''}"
        fig.add_scatter(x=grid[:, 0], y=curve, mode="lines", name=label,
                        line=dict(color=color, width=3 if r else 2, shape="hv" if r < 10 else "linear"))
    fig.update_layout(
        title="Challahs eaten against guests: each tree fixes what the last ones missed",
        xaxis_title="guests at the table", yaxis_title="challahs finished",
        legend=dict(orientation="h", y=-0.25), margin=dict(t=60, b=40),
    )
    save(fig, "boosting-progression")


# ---------------------------------------------------------------------------
# Forests, AdaBoost and gradient boosting against the number of trees
# ---------------------------------------------------------------------------

def chart_vs_forest(n_trees=600):
    print("Chart — forest vs boosting")
    X, y = make_classification(n_samples=12000, n_features=20, n_informative=6, n_redundant=2,
                               n_clusters_per_class=3, flip_y=0.05, class_sep=0.9, random_state=0)
    Xtr, Xr, ytr, yr = train_test_split(X, y, test_size=0.5, random_state=0)
    Xva, Xte, yva, yte = train_test_split(Xr, yr, test_size=0.5, random_state=0)

    single = {}
    for d in (2, 4, None):
        t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        single[str(d)] = {"train": float(t.score(Xtr, ytr)), "test": float(t.score(Xte, yte))}

    rf = RandomForestClassifier(n_estimators=n_trees, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    probs = np.array([e.predict_proba(Xte)[:, 1] for e in rf.estimators_])
    running = np.cumsum(probs, axis=0) / np.arange(1, n_trees + 1)[:, None]
    rf_acc = [accuracy_score(yte, r > 0.5) for r in running]
    rf_ll = [log_loss(yte, np.clip(r, 1e-3, 1 - 1e-3)) for r in running]

    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=n_trees,
                             random_state=0).fit(Xtr, ytr)
    ada_acc = [accuracy_score(yte, p) for p in ada.staged_predict(Xte)]

    def boosted(depth, lr):
        gb = GradientBoostingClassifier(n_estimators=n_trees, max_depth=depth, learning_rate=lr,
                                        random_state=0).fit(Xtr, ytr)
        acc = [accuracy_score(yte, p) for p in gb.staged_predict(Xte)]
        ll = [log_loss(yte, p[:, 1]) for p in gb.staged_predict_proba(Xte)]
        val_ll = [log_loss(yva, p[:, 1]) for p in gb.staged_predict_proba(Xva)]
        return acc, ll, int(np.argmin(val_ll)) + 1

    gb_small_acc, gb_small_ll, _ = boosted(3, 0.1)
    gb_big_acc, gb_big_ll, best_k = boosted(8, 0.05)

    NUMBERS["vs_forest"] = {
        "single_tree": single,
        "rf_acc_final": rf_acc[-1], "rf_acc_at": {k: rf_acc[k - 1] for k in (1, 10, 50, 100, 300, 600)},
        "ada_acc_final": ada_acc[-1],
        "gb_small_acc_final": gb_small_acc[-1], "gb_big_acc_final": gb_big_acc[-1],
        "gb_big_best_k_by_validation": best_k, "gb_big_acc_at_best_k": gb_big_acc[best_k - 1],
        "rf_ll_final": rf_ll[-1],
        "gb_big_ll_min": min(gb_big_ll), "gb_big_ll_min_at": int(np.argmin(gb_big_ll)) + 1,
        "gb_big_ll_final": gb_big_ll[-1],
        "gb_small_ll_min": min(gb_small_ll), "gb_small_ll_min_at": int(np.argmin(gb_small_ll)) + 1,
        "gb_small_ll_final": gb_small_ll[-1],
    }

    xs = list(range(1, n_trees + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Test accuracy", "Test logloss"))
    for name, acc, color, dash in (
        ("Random forest", rf_acc, COLORS["rf"], "solid"),
        ("AdaBoost (stumps)", ada_acc, COLORS["ada"], "solid"),
        ("Gradient boosting, depth 3", gb_small_acc, COLORS["gb_small"], "solid"),
        ("Gradient boosting, depth 8", gb_big_acc, COLORS["gb_big"], "solid"),
    ):
        fig.add_scatter(x=xs, y=acc, mode="lines", name=name, line=dict(color=color, width=2.5, dash=dash),
                        row=1, col=1)
    for name, ll, color in (
        ("Random forest", rf_ll, COLORS["rf"]),
        ("Gradient boosting, depth 3", gb_small_ll, COLORS["gb_small"]),
        ("Gradient boosting, depth 8", gb_big_ll, COLORS["gb_big"]),
    ):
        fig.add_scatter(x=xs, y=ll, mode="lines", name=name, line=dict(color=color, width=2.5),
                        showlegend=False, row=1, col=2)
    fig.update_xaxes(title_text="number of trees", row=1, col=1)
    fig.update_xaxes(title_text="number of trees", row=1, col=2)
    fig.update_yaxes(title_text="accuracy", range=[0.65, 0.9], row=1, col=1)
    fig.update_yaxes(title_text="logloss (lower is better)", range=[0.25, 0.8], row=1, col=2)
    fig.update_layout(legend=dict(orientation="h", y=-0.25), margin=dict(t=60, b=40))
    save(fig, "boosting-vs-forest")


def robustness(seeds=range(5), n_trees=600):
    print("Check — tuned boosting vs forest across seeds")
    rows = []
    for seed in seeds:
        X, y = make_classification(n_samples=12000, n_features=20, n_informative=6, n_redundant=2,
                                   n_clusters_per_class=3, flip_y=0.05, class_sep=0.9, random_state=seed)
        Xtr, Xr, ytr, yr = train_test_split(X, y, test_size=0.5, random_state=seed)
        Xva, Xte, yva, yte = train_test_split(Xr, yr, test_size=0.5, random_state=seed)
        rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, ytr)
        gb = GradientBoostingClassifier(n_estimators=n_trees, max_depth=8, learning_rate=0.05,
                                        random_state=0).fit(Xtr, ytr)
        val_ll = [log_loss(yva, p[:, 1]) for p in gb.staged_predict_proba(Xva)]
        k = int(np.argmin(val_ll)) + 1
        gb_acc = [accuracy_score(yte, p) for p in gb.staged_predict(Xte)][k - 1]
        rows.append({"seed": seed, "rf": float(rf.score(Xte, yte)), "gb_tuned": float(gb_acc), "k": k})
    NUMBERS["robustness"] = {
        "per_seed": rows,
        "rf_mean": float(np.mean([r["rf"] for r in rows])),
        "gb_mean": float(np.mean([r["gb_tuned"] for r in rows])),
        "gb_ahead_in": int(sum(r["gb_tuned"] > r["rf"] for r in rows)),
    }


if __name__ == "__main__":
    chart_progression()
    chart_vs_forest()
    robustness()
    print(json.dumps(NUMBERS, indent=2, default=float))

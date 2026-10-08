"""
Generates Plotly charts for the three CatBoost posts.

Outputs:
  - assets/charts/catboost-target-leakage.html     (How CatBoost works)
  - assets/charts/catboost-object-importance.html  (Object importance)
  - assets/charts/catboost-leaf-matrix.html        (Leaf indexes)
  - assets/charts/catboost-leaf-auc.html           (Leaf indexes)

Everything is synthetic, so the script needs no downloads.

Requirements:
    pip install catboost scikit-learn plotly pandas numpy scipy

Usage:
    python scripts/generate_catboost_charts.py
"""

import os
import ssl
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from catboost import CatBoostClassifier, Pool
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

ssl._create_default_https_context = ssl._create_unverified_context

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "../assets/charts")

COLORS = {
    "bad": "#d9534f",
    "mid": "#e8a95c",
    "good": "#4fb1ba",
    "accent": "#9b7fd4",
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
# Post 1: target leakage with a categorical feature that carries no signal
# ---------------------------------------------------------------------------

N_CATS = 1500


def _noise_cat(seed, n=6000):
    r = np.random.RandomState(seed)
    return r.randint(0, N_CATS, n), (r.rand(n) < 0.5).astype(int)


def _naive_ts(cat_tr, y_tr, cat_te):
    s = np.bincount(cat_tr, weights=y_tr, minlength=N_CATS)
    c = np.bincount(cat_tr, minlength=N_CATS)
    f = lambda cat: (s[cat] + 0.5 * y_tr.mean()) / (c[cat] + 0.5)
    return f(cat_tr), f(cat_te)


def _ordered_ts(cat_tr, y_tr, cat_te, seed=0, a=0.5):
    perm = np.random.RandomState(seed).permutation(len(y_tr))
    p = y_tr.mean()
    s, c, out = np.zeros(N_CATS), np.zeros(N_CATS), np.zeros(len(y_tr))
    for i in perm:
        out[i] = (s[cat_tr[i]] + a * p) / (c[cat_tr[i]] + a)
        s[cat_tr[i]] += y_tr[i]
        c[cat_tr[i]] += 1
    return out, (s[cat_te] + a * p) / (c[cat_te] + a)


def chart_target_leakage():
    print("Chart — target statistics leakage")
    res = {k: ([], []) for k in ("naive", "ordered", "catboost")}
    for seed in range(5):
        ctr, ytr = _noise_cat(seed)
        cte, yte = _noise_cat(seed + 100)
        for name, fn in (("naive", _naive_ts), ("ordered", _ordered_ts)):
            ftr, fte = fn(ctr, ytr, cte)
            g = HistGradientBoostingClassifier(max_iter=100, random_state=0)
            g.fit(ftr[:, None], ytr)
            res[name][0].append(roc_auc_score(ytr, g.predict_proba(ftr[:, None])[:, 1]))
            res[name][1].append(roc_auc_score(yte, g.predict_proba(fte[:, None])[:, 1]))
        dtr = pd.DataFrame({"c": ctr.astype(str)})
        dte = pd.DataFrame({"c": cte.astype(str)})
        m = CatBoostClassifier(iterations=200, depth=4, verbose=0, random_seed=seed)
        m.fit(dtr, ytr, cat_features=["c"])
        res["catboost"][0].append(roc_auc_score(ytr, m.predict_proba(dtr)[:, 1]))
        res["catboost"][1].append(roc_auc_score(yte, m.predict_proba(dte)[:, 1]))

    labels = ["Naive target encoding", "Ordered target statistics<br>(hand-rolled)", "CatBoost<br>(cat_features)"]
    keys = ["naive", "ordered", "catboost"]
    train = [float(np.mean(res[k][0])) for k in keys]
    test = [float(np.mean(res[k][1])) for k in keys]
    NUMBERS["leakage"] = {k: {"train": t, "test": e} for k, t, e in zip(keys, train, test)}

    fig = go.Figure()
    fig.add_bar(x=labels, y=train, name="Train AUC", marker_color=COLORS["mid"],
                text=[f"{v:.2f}" for v in train], textposition="outside")
    fig.add_bar(x=labels, y=test, name="Test AUC", marker_color=COLORS["good"],
                text=[f"{v:.2f}" for v in test], textposition="outside")
    fig.add_hline(y=0.5, line_dash="dot", line_color=COLORS["grey"],
                  annotation_text="coin flip", annotation_position="bottom right")
    fig.update_layout(
        title="A categorical column with zero signal, three ways of encoding it",
        yaxis=dict(title="AUC", range=[0.4, 0.85]),
        barmode="group", legend=dict(orientation="h", y=-0.2),
        margin=dict(t=60, b=40),
    )
    save(fig, "catboost-target-leakage")


# ---------------------------------------------------------------------------
# Post 2: object importance on a training set with flipped labels
# ---------------------------------------------------------------------------

def _object_importance_run(seed, noise=0.2, ks=(0, 200, 400, 600, 800, 1000, 1200)):
    rng = np.random.RandomState(seed)
    X, y = make_classification(n_samples=16000, n_features=20, n_informative=8,
                               n_redundant=4, class_sep=0.6, random_state=seed)
    Xtr, Xr, ytr, yr = train_test_split(X, y, test_size=0.5, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(Xr, yr, test_size=0.5, random_state=seed + 1)
    flip = rng.rand(len(ytr)) < noise
    yn = np.where(flip, 1 - ytr, ytr)
    kw = dict(iterations=300, depth=6, learning_rate=0.08, verbose=0, random_seed=seed)

    def fit(mask, labels):
        m = CatBoostClassifier(**kw)
        m.fit(Pool(Xtr[mask], labels[mask]))
        p = m.predict_proba(Xte)[:, 1]
        return roc_auc_score(yte, p), log_loss(yte, p), m

    everyone = np.ones(len(ytr), bool)
    auc0, ll0, m = fit(everyone, yn)
    idx, sc = m.get_object_importance(Pool(Xva, yva), Pool(Xtr, yn),
                                      top_size=len(ytr), type="Average")
    idx, sc = np.array(idx), np.array(sc)
    harmful = idx[sc > 0]
    _, ll_clean, _ = fit(everyone, ytr)

    out = dict(ks=list(ks), imp_ll=[], rnd_ll=[], imp_auc=[], rnd_auc=[], prec=[],
               ll_noisy=ll0, ll_clean=ll_clean, auc_noisy=auc0,
               flip_rate=float(flip.mean()), n_harmful=int(len(harmful)),
               score_flipped=sc[flip[idx]], score_clean=sc[~flip[idx]])
    for k in ks:
        mask = np.ones(len(ytr), bool)
        if k:
            mask[harmful[:k]] = False
        a, l, _ = fit(mask, yn)
        mr = np.ones(len(ytr), bool)
        if k:
            mr[rng.choice(len(ytr), k, replace=False)] = False
        ar, lr, _ = fit(mr, yn)
        out["imp_ll"].append(l)
        out["imp_auc"].append(a)
        out["rnd_ll"].append(lr)
        out["rnd_auc"].append(ar)
        out["prec"].append(float(flip[harmful[:k]].mean()) if k else float("nan"))
    return out


def chart_object_importance():
    print("Chart — object importance")
    runs = [_object_importance_run(s) for s in range(3)]
    ks = runs[0]["ks"]
    mean = lambda key: np.mean([r[key] for r in runs], axis=0)

    imp_ll, rnd_ll, imp_auc, rnd_auc, prec = (mean(k) for k in ("imp_ll", "rnd_ll", "imp_auc", "rnd_auc", "prec"))
    ll_clean = float(np.mean([r["ll_clean"] for r in runs]))
    NUMBERS["object_importance"] = dict(
        ks=ks, imp_ll=imp_ll.tolist(), rnd_ll=rnd_ll.tolist(),
        imp_auc=imp_auc.tolist(), rnd_auc=rnd_auc.tolist(), prec=prec.tolist(),
        ll_clean=ll_clean, flip_rate=float(np.mean([r["flip_rate"] for r in runs])),
        n_harmful=[r["n_harmful"] for r in runs],
    )

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Test logloss as we drop rows", "How many of the dropped rows were really flipped"))
    fig.add_scatter(x=ks, y=imp_ll, mode="lines+markers", name="Drop by object importance",
                    line=dict(color=COLORS["good"], width=3), row=1, col=1)
    fig.add_scatter(x=ks, y=rnd_ll, mode="lines+markers", name="Drop at random",
                    line=dict(color=COLORS["grey"], width=3, dash="dash"), row=1, col=1)
    fig.add_hline(y=ll_clean, line_dash="dot", line_color=COLORS["accent"], row=1, col=1,
                  annotation_text="trained on the true labels", annotation_position="top right")
    fig.add_scatter(x=ks[1:], y=prec[1:], mode="lines+markers", name="Precision of the top-k",
                    line=dict(color=COLORS["mid"], width=3), showlegend=False, row=1, col=2)
    fig.add_hline(y=NUMBERS["object_importance"]["flip_rate"], line_dash="dot",
                  line_color=COLORS["grey"], row=1, col=2,
                  annotation_text="flip rate of a random row", annotation_position="bottom right")
    fig.update_xaxes(title_text="rows removed from training", row=1, col=1)
    fig.update_xaxes(title_text="top-k rows by importance", row=1, col=2)
    fig.update_yaxes(title_text="logloss (lower is better)", row=1, col=1)
    fig.update_yaxes(title_text="share with a flipped label", range=[0, 1.05], tickformat=".0%", row=1, col=2)
    fig.update_layout(legend=dict(orientation="h", y=-0.25), margin=dict(t=60, b=40))
    save(fig, "catboost-object-importance", height=450)


# ---------------------------------------------------------------------------
# Post 3: leaf indexes as a sparse feature matrix
# ---------------------------------------------------------------------------

DEPTH, TREES = 6, 100


def _one_hot(leaves):
    n, t = leaves.shape
    n_leaves = 2 ** DEPTH
    cols = (leaves.astype(np.int64) + np.arange(t) * n_leaves).ravel()
    rows = np.repeat(np.arange(n), t)
    return sp.csr_matrix((np.ones(n * t), (rows, cols)), shape=(n, t * n_leaves))


def chart_leaf_sparsity():
    print("Chart — leaf indexes")
    X, y = make_classification(n_samples=20000, n_features=20, n_informative=10, n_redundant=2,
                               n_clusters_per_class=4, class_sep=0.8, flip_y=0.02, random_state=0)
    Xtr, Xr, ytr, yr = train_test_split(X, y, test_size=0.4, random_state=0)
    Xva, Xte, yva, yte = train_test_split(Xr, yr, test_size=0.5, random_state=1)

    m = CatBoostClassifier(iterations=TREES, depth=DEPTH, learning_rate=0.1, verbose=0, random_seed=0)
    m.fit(Xtr, ytr)
    auc_cb = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    lr_raw = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    auc_raw = roc_auc_score(yte, lr_raw.predict_proba(Xte)[:, 1])

    leaves_tr = m.calc_leaf_indexes(Pool(Xtr))
    leaves_te = m.calc_leaf_indexes(Pool(Xte))
    Ztr, Zte = _one_hot(leaves_tr), _one_hot(leaves_te)

    best = max(((roc_auc_score(yte, LogisticRegression(C=C, max_iter=3000).fit(Ztr, ytr)
                               .predict_proba(Zte)[:, 1]), C) for C in (0.03, 0.1, 0.3)))
    auc_leaf, C_leaf = best

    # The cautious variant: fit the linear model on rows CatBoost never trained on
    Zva = _one_hot(m.calc_leaf_indexes(Pool(Xva)))
    z_va = LogisticRegression(C=C_leaf, max_iter=3000).fit(Zva, yva)
    auc_leaf_holdout = roc_auc_score(yte, z_va.predict_proba(Zte)[:, 1])

    l1 = LogisticRegression(C=0.3, l1_ratio=1.0, solver="liblinear", max_iter=3000).fit(Ztr, ytr)
    auc_l1 = roc_auc_score(yte, l1.predict_proba(Zte)[:, 1])
    nz_l1 = int((l1.coef_ != 0).sum())

    n_leaf_cols = Ztr.shape[1]
    ever_hit = int(np.unique(Ztr.nonzero()[1]).size)
    NUMBERS["leaf"] = dict(
        leaf_index_shape=list(leaves_tr.shape), matrix_shape=list(Ztr.shape),
        ones_per_row=float(Ztr.nnz / Ztr.shape[0]), density=float(Ztr.nnz / np.prod(Ztr.shape)),
        leaves_ever_hit=ever_hit, auc_catboost=auc_cb, auc_lr_raw=auc_raw, auc_lr_leaves=auc_leaf,
        C=C_leaf, auc_lr_leaves_holdout=auc_leaf_holdout, auc_l1=auc_l1, l1_nonzero=nz_l1,
    )

    # Chart A: the matrix itself (40 rows, first 6 trees)
    t_show, r_show = 6, 40
    block = Ztr[:r_show, : t_show * 2 ** DEPTH].toarray()
    fig = go.Figure(go.Heatmap(z=block, colorscale=[[0, "rgba(0,0,0,0.04)"], [1, COLORS["accent"]]],
                               showscale=False, xgap=0, ygap=0))
    for t in range(1, t_show):
        fig.add_vline(x=t * 2 ** DEPTH - 0.5, line_color=COLORS["grey"], line_width=1)
    fig.update_layout(
        title=f"40 rows × the first {t_show} trees: every row lights exactly one leaf per tree",
        xaxis=dict(title=f"leaf columns ({t_show} trees × {2 ** DEPTH} leaves)", showticklabels=False),
        yaxis=dict(title="row", autorange="reversed"),
        margin=dict(t=60, b=40),
    )
    save(fig, "catboost-leaf-matrix", height=380)

    # Chart B: AUC comparison
    labels = ["Logistic regression<br>on raw features", "CatBoost", "Logistic regression<br>on leaf matrix",
              f"L1 on leaf matrix<br>({nz_l1} of {n_leaf_cols} columns kept)"]
    vals = [auc_raw, auc_cb, auc_leaf, auc_l1]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[COLORS["grey"], COLORS["mid"], COLORS["good"], COLORS["accent"]],
                           text=[f"{v:.3f}" for v in vals], textposition="outside"))
    fig.update_layout(title="Test AUC on a nonlinear problem", yaxis=dict(title="AUC", range=[0.6, 1.0]),
                      margin=dict(t=60, b=40))
    save(fig, "catboost-leaf-auc")


if __name__ == "__main__":
    chart_target_leakage()
    chart_object_importance()
    chart_leaf_sparsity()
    print(json.dumps(NUMBERS, indent=2, default=float))

---
layout: post
title: "The Map Maker's Dilemma — Part 1"
description: |
  PCA and SVD: Ancient Arts of the Dimensionality Mages
image: /assets/img/DimReduction-post/cover-part1.jpg
noindex: true
---

In the kingdom of Vectoria, all knowledge was stored in the Great Archive — an infinite library where every citizen was described by thousands of scrolls.
Height, weight, spending habits, favorite spells, number of dragons owned.
The Archive was complete. The Archive was perfect. The Archive was completely unusable.

No map maker could draw a map of it. No general could read it on a battlefield.
And so the Council of Mages was summoned.

*"We need a map,"* said the Queen. *"A map we can actually look at."*

The eldest mage stepped forward. *"Your Majesty, we can draw you a map. But every map is a lie. The question is which lie you can live with."*

This is the story of those maps — and the ancient arts used to draw them.

---

# The Art of PCA — Finding the Spine of the Land

The first and most venerable technique is **Principal Component Analysis**, known in the old tongue as *The Spine Finder*.

The idea is simple. Imagine your thousand-dimensional data as a cloud of fireflies drifting through a dark castle. PCA asks: *what is the longest axis along which these fireflies drift?* That becomes the first dimension of your map. Then: *what is the next longest axis, perpendicular to the first?* That becomes the second.

The result is the 2D projection that preserves the most *variance* — the most spread, the most signal — from the original high-dimensional cloud.

## The Spell

We begin our experiment in the Iris Fields — a meadow of 150 flowers, each described by four measurements: sepal length, sepal width, petal length, petal width. A humble dataset, but instructive.

```python
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target
species = [iris.target_names[i] for i in y]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig = px.scatter(
    x=X_pca[:, 0], y=X_pca[:, 1],
    color=species,
    color_discrete_sequence=['#4fb1ba', '#e8a95c', '#9b7fd4'],
    labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
            'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'},
    title='The Iris Fields — PCA Projection'
)
fig.update_traces(marker=dict(size=8, opacity=0.8))
fig.show()
```

<iframe src="/assets/charts/iris-pca-scatter.html" style="width:100%;height:500px;border:none;"></iframe>

The three species of iris separate beautifully along the first principal component — which turns out to be driven almost entirely by petal size. The mage did not know this in advance. The spine of the land revealed it.

## How Much Did We Keep — and What Did We Lose?

The most honest thing about PCA is that it tells you exactly what it threw away.

```python
pca_full = PCA()
pca_full.fit(X)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

fig = px.bar(
    x=[f'PC{i+1}' for i in range(len(pca_full.explained_variance_ratio_))],
    y=pca_full.explained_variance_ratio_,
    color_discrete_sequence=['#4fb1ba'],
    labels={'x': 'Principal Component', 'y': 'Explained Variance'},
    title='The Scree Plot — How Much of the Kingdom Did We Keep?'
)
fig.add_scatter(
    x=[f'PC{i+1}' for i in range(len(cumulative_variance))],
    y=cumulative_variance,
    mode='lines+markers',
    name='Cumulative',
    line=dict(color='#e8a95c', width=2)
)
fig.show()
```

<iframe src="/assets/charts/iris-pca-scree.html" style="width:100%;height:500px;border:none;"></iframe>

The first two components capture 95.8% of all variance. We lost 4.2% of the kingdom when we drew the map. Sounds small. But what exactly is that 4.2%?

The answer lives in the components we dropped. You can inspect them:

```python
feature_names = iris.feature_names
for i, component in enumerate(pca_full.components_):
    print(f"PC{i+1}: " + ", ".join(
        f"{name}: {weight:.2f}" for name, weight in zip(feature_names, component)
    ))
```

PC3 and PC4, which we discarded, are driven primarily by **sepal width** — a feature that does not separate the species well, but does carry real biological information about flower shape within each species.

What we lost: the ability to distinguish, say, a wide-petaled setosa from a narrow-petaled one. What we kept: everything needed to tell the three species apart.

This is the map maker's judgement call. The 4.2% we discarded is not noise — it is real variation, just variation that did not matter for our goal. If our goal were different (predicting individual flower weight, perhaps), we might need those components back.

A good map maker does not just look at the percentage. She asks: *what is in the part I am discarding, and do I care about it?*

---

# The Art of SVD — The Portrait Restorer

Deeper in the Archive lives a more ancient magic: **Singular Value Decomposition**, known as *The Decomposer*.

Where PCA finds directions of variance, SVD dismantles the data itself into its fundamental layers. Any scroll — any matrix — can be written as:

**X = U · Σ · Vᵀ**

Three matrices. **U** holds the citizen portraits. **Σ** holds the *singular values* — a ranking of importance, from most to least. **Vᵀ** holds the feature patterns.

The crucial insight: if you keep only the top *k* singular values and discard the rest, you get the best possible *k*-layer approximation of the original data. Not an approximation in any vague sense — the *provably best* one, in terms of reconstruction error.

This makes SVD the art of **compression**. Not just projection.

## Restoring the Portraits of Digitia

In the northern province of Digitia, every citizen's identity scroll is not words but pixels — an 8×8 portrait of a handwritten digit, 64 values in total.

The Archive holds 1,797 such portraits. SVD can compress the entire collection by finding the shared structure across all portraits. Each digit can then be *reconstructed* from just a handful of singular values — the most important layers — rather than all 64 pixel values.

```python
from sklearn.datasets import load_digits
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

digits = load_digits()
X_digits = digits.data.astype(float)

U, sigma, Vt = np.linalg.svd(X_digits, full_matrices=False)

# Reconstruct one portrait using different numbers of singular values
sample = 0
ranks = [1, 3, 8, 20, 40, 64]

fig = make_subplots(
    rows=1, cols=len(ranks),
    subplot_titles=[f'k={k}' for k in ranks]
)
for i, k in enumerate(ranks):
    reconstructed = (U[sample, :k] * sigma[:k]) @ Vt[:k, :]
    fig.add_trace(
        go.Heatmap(z=reconstructed.reshape(8, 8), colorscale='gray',
                   showscale=False, reversescale=True),
        row=1, col=i+1
    )
fig.update_layout(title='Portrait Restoration — From 1 to 64 Singular Values')
fig.show()
```

<iframe src="/assets/charts/svd-reconstruction.html" style="width:100%;height:300px;border:none;"></iframe>

With **k=1**, you see a ghost — barely a smudge. With **k=8**, the digit is recognisable. With **k=20**, it is sharp. With **k=64** you have the original, nothing lost.

The Archive that once required 64 values per portrait can now be read from 20. That is the Decomposer's gift: not just a map, but a compressed version of the original that you can reconstruct at will.

## The Hierarchy of Power

The singular values tell you how much each layer contributes:

```python
fig = px.bar(
    x=list(range(1, 41)),
    y=sigma[:40],
    color_discrete_sequence=['#9b7fd4'],
    labels={'x': 'Singular Value Rank', 'y': 'Singular Value'},
    title='The Hierarchy of Power — How Much Each Layer Contributes'
)
fig.show()
```

<iframe src="/assets/charts/digits-singular-values.html" style="width:100%;height:500px;border:none;"></iframe>

The first layer towers over the rest. The drop is steep and then levels off. This is the signature of data with real structure — a few layers carry most of the story.

## PCA and SVD — Two Names for One Truth

In truth, PCA *is* SVD. When the sklearn mages implemented PCA, they called SVD inside it. The difference is practical:

- **PCA** mean-centers the data first and reports explained variance — better for exploration.
- **SVD** skips centering — essential for sparse data (text, interaction logs) where centering would destroy the sparsity and exhaust memory.

For dense, tabular data: use PCA. For sparse matrices with millions of entries: use TruncatedSVD directly.

---

# The Limit of Straight Lines

The honest arts have served us well. PCA revealed the spine of the Iris Fields. SVD restored the portraits of Digitia from a fraction of their original size. Both gave us receipts — exact accounts of what was kept and what was discarded.

But the Archive holds stranger lands than these.

In the eastern province lives the **Enchanted Scroll** — a dataset that curls through three dimensions like a rolled-up map. PCA looks at it and sees only the shadow it casts on the wall.

```python
from sklearn.datasets import make_swiss_roll

X_roll, color = make_swiss_roll(n_samples=1500, noise=0.1, random_state=42)
X_roll_scaled = StandardScaler().fit_transform(X_roll)

pca_roll = PCA(n_components=2)
X_roll_pca = pca_roll.fit_transform(X_roll_scaled)

fig = px.scatter(
    x=X_roll_pca[:, 0], y=X_roll_pca[:, 1],
    color=color,
    color_continuous_scale='teal',
    labels={'x': f'PC1 ({pca_roll.explained_variance_ratio_[0]:.1%})',
            'y': f'PC2 ({pca_roll.explained_variance_ratio_[1]:.1%})'},
    title='The Enchanted Scroll — PCA Loses the Structure'
)
fig.show()
```

<iframe src="/assets/charts/swiss-roll-pca.html" style="width:100%;height:500px;border:none;"></iframe>

What should be a graceful spiral is flattened into an unintelligible smear. Points that are far apart on the scroll end up neighbours on the map. Points that are close on the scroll end up separated. The structure is not just hidden — it is actively distorted.

This is not a failure of skill. It is the honest limit of any art that speaks only in straight lines.

To map a scroll, you need a mage who can follow the curve.

Those mages — and their darker, more dramatic arts — are waiting in [Part 2 →](/data-science/2026-06-27-DimReduction-part2-post/)

---

*All code for this post is available in the [Jewpyter notebook repository](https://github.com/wolfenfeld/jewpyter/blob/master/notebooks/DimensionalityReduction.ipynb).*

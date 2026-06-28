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

# The Art of SVD — The Face in the Mosaic

Deeper in the Archive lives a more ancient magic: **Singular Value Decomposition**, known as *The Decomposer*.

Where PCA finds directions of variance, SVD dismantles the data itself into its fundamental layers. Any scroll — any matrix — can be written as:

**X = U · Σ · Vᵀ**

Three matrices. **U** holds the citizen portraits. **Σ** holds the *singular values* — a ranking of importance, from most to least. **Vᵀ** holds the shared patterns across all portraits.

The crucial insight: if you keep only the top *k* singular values and discard the rest, you get the best possible *k*-layer approximation of the original data. Not an approximation in any vague sense — the *provably best* one, in terms of reconstruction error.

This makes SVD the art of **recognition without memorisation**.

## The Face Vault of Vectoria

The kingdom's Face Vault holds 400 portraits — 64×64 pixel paintings of 40 noble families, ten portraits each. The Royal Guard must recognise any citizen at the gate, but carrying 4,096 pixel values per face is impractical. They need a way to compress the knowledge.

The SVD mage studies 350 of the portraits and learns the **common patterns of faces** — the way light falls on cheekbones, the typical shape of brows, the common structure of noses. These patterns, ranked by importance, are the singular vectors.

When a stranger arrives at the gate, the Guard does not compare all 4,096 pixel values. They project the face onto the learned patterns — just the top *k* — and reconstruct it. If the reconstruction matches a known face, the citizen is recognised.

```python
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
faces = dataset.data  # 400 faces, each 4096 pixels (64x64)

# Train on 350 faces, test recognition on face 351
X_train = faces[:350]
test_face = faces[351]

U, sigma, Vt = np.linalg.svd(X_train, full_matrices=False)

def reconstruct(face, Vt, k):
    compressed = face @ Vt[:k].T   # project onto top k patterns
    return compressed @ Vt[:k]     # reconstruct from those patterns

ranks = [5, 20, 50, 100, 200, 350]

fig = make_subplots(
    rows=1, cols=len(ranks) + 1,
    subplot_titles=["Original"] + [f'k={k}' for k in ranks],
    horizontal_spacing=0.02,
)
fig.add_trace(
    go.Heatmap(z=test_face.reshape(64, 64), colorscale='gray',
               showscale=False, reversescale=True),
    row=1, col=1,
)
for i, k in enumerate(ranks):
    reconstructed = reconstruct(test_face, Vt, k)
    fig.add_trace(
        go.Heatmap(z=reconstructed.reshape(64, 64), colorscale='gray',
                   showscale=False, reversescale=True),
        row=1, col=i + 2,
    )
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False, autorange="reversed")
fig.update_layout(title='Recognising the Stranger — Face Reconstruction at Different Ranks')
fig.show()
```

<iframe src="/assets/charts/svd-reconstruction.html" style="width:100%;height:320px;border:none;"></iframe>

With **k=5**, a ghostly suggestion of a face emerges — enough to confirm it is a face, not a dragon. With **k=50**, the features are clear. With **k=200**, it is nearly indistinguishable from the original. And crucially: face 351 was **never seen during training**. The mage learned the patterns of faces, not the faces themselves.

This is the power of SVD over brute-force memorisation. The Guard does not need to store 4,096 numbers per citizen. They store the shared patterns once, and a short code — just *k* coefficients — per citizen. Recognition becomes comparison of codes, not pixels.

## The Hierarchy of Power

Which patterns matter most? The singular values tell you:

```python
fig = px.bar(
    x=list(range(1, len(sigma) + 1)),
    y=sigma,
    color_discrete_sequence=['#9b7fd4'],
    labels={'x': 'Singular Value Rank', 'y': 'Singular Value'},
    title='The Hierarchy of Power — How Much Each Pattern Contributes'
)
fig.show()
```

<iframe src="/assets/charts/digits-singular-values.html" style="width:100%;height:500px;border:none;"></iframe>

The first pattern towers over the rest — it captures the single most common structure across all 350 faces. The drop is steep and then gradual. This is the signature of data with real shared structure: a few patterns carry most of the story, and the rest is individual detail.

## PCA and SVD — Two Names for One Truth

In truth, PCA *is* SVD. When the sklearn mages implemented PCA, they called SVD inside it. The difference is practical:

- **PCA** mean-centers the data first and reports explained variance — better for exploration.
- **SVD** skips centering — essential for sparse data (text, interaction logs) where centering would destroy the sparsity and exhaust memory. It also enables the reconstruction trick above: project a new, unseen point onto learned patterns and reconstruct it.

For dense tabular data and exploration: use PCA. For compression, recognition, and sparse matrices: use SVD directly.

---

# The Limit of Straight Lines

The honest arts have served the kingdom well.

PCA mapped the Iris Fields in two dimensions, preserving 95.8% of everything that mattered — and told us exactly what the remaining 4.2% contained. SVD learned the shared structure of a thousand faces, compressed each portrait to a fraction of its original size, and recognised a stranger it had never met.

Both arts gave receipts. Both kept their promises.

But a messenger arrived from the eastern province with troubling news. A peculiar land had been discovered — the **Enchanted Scroll**, a territory that had rolled itself into a spiral through three dimensions. The Royal Map Makers were summoned. They applied PCA. They applied SVD.

The map they produced looked like this:

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
    title='The Enchanted Scroll — What the Honest Arts Produce'
)
fig.show()
```

<iframe src="/assets/charts/swiss-roll-pca.html" style="width:100%;height:500px;border:none;"></iframe>

A smear. An unintelligible, useless smear.

The scroll's colour tells you where each point sits along the spiral — inner curl, middle band, outer edge. On the true map those colours should form clean, separated bands. Instead they are scrambled together. Citizens who live on opposite ends of the scroll appear as neighbours. Citizens who live side by side appear as strangers.

PCA reported that it captured 64% of the variance. It was not lying. But the 36% it discarded was the part that encoded the curl — and without the curl, the map of the Enchanted Scroll is worse than useless. It is actively misleading.

This is not a failure of the mages. It is the honest limit of any art that speaks only in straight lines. The scroll's structure is curved, and no straight line can follow a curve.

The Queen looked at the smear and made her decision. *"Summon the others,"* she said. *"The ones who work in shadow."*

Those mages — and their darker, more dramatic arts — are waiting in [Part 2 →](/data-science/2026-06-27-DimReduction-part2-post/)

---

*All code for this post is available in the [Jewpyter notebook repository](https://github.com/wolfenfeld/jewpyter/blob/master/notebooks/DimensionalityReduction.ipynb).*

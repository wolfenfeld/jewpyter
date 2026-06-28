---
layout: post
title: "The Cartographer's Dilemma — Part 1"
description: |
  PCA and SVD: Ancient Arts of the Dimensionality Mages
image: /assets/img/DimReduction-post/cover-part1.jpg
noindex: true
---

In the kingdom of Vectoria, all knowledge was stored in the Great Archive — an infinite library where every citizen was described by thousands of scrolls.
Height, weight, spending habits, favorite spells, number of dragons owned.
The Archive was complete. The Archive was perfect. The Archive was completely unusable.

No cartographer could draw a map of it. No general could read it on a battlefield.
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

## How Much Did We Keep?

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

In the Iris Fields, the first two components capture 95.8% of all variance. We lost barely 4% of the kingdom when we drew the map. That is a good map.

## When the Spine Finder Fails

But what if the kingdom is not shaped like a spine? What if it is shaped like a scroll?

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

The Enchanted Scroll — a manifold that curls through 3D space — is flattened by PCA into an unintelligible smear. The structure is gone. The map is useless.

This is not a failure of the mage. This is the honest limit of a linear art.

---

# The Art of SVD — The Decomposition of All Things

Deeper in the Archive lives a more ancient magic: **Singular Value Decomposition**, known as *The Decomposer*.

Where PCA finds directions of variance, SVD dismantles the data itself into its fundamental components. Any scroll — any matrix — can be written as:

**X = U · Σ · Vᵀ**

Three matrices. U holds the citizen portraits. Σ holds the *singular values* — a ranking of importance, from most to least. Vᵀ holds the feature patterns. Discard the small singular values and you keep the essential story.

In truth, PCA *is* SVD. When the sklearn mages implemented PCA, they called SVD inside it. The difference is in what you can do directly with SVD — especially with scrolls too large and too sparse to mean-center.

## SVD on the Digits of the Realm

In the northern province of Digitia, citizens are described not by measurements but by 64 pixel values — an 8×8 portrait of a handwritten digit.

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import TruncatedSVD

digits = load_digits()
X_digits = digits.data  # 1797 samples, 64 features
y_digits = digits.target

svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X_digits)

fig = px.scatter(
    x=X_svd[:, 0], y=X_svd[:, 1],
    color=[str(d) for d in y_digits],
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={'x': 'SVD Component 1', 'y': 'SVD Component 2'},
    title='The Digits of Digitia — SVD Projection',
    hover_data={'digit': y_digits}
)
fig.update_traces(marker=dict(size=5, opacity=0.7))
fig.show()
```

<iframe src="/assets/charts/digits-svd-scatter.html" style="width:100%;height:500px;border:none;"></iframe>

The digits partially separate — you can see the 0s pulling away from the 1s — but most of the kingdom remains tangled. The linear decomposition has shown us the bones. The flesh is still hidden.

## The Singular Values — A Ranking of Power

```python
svd_full = TruncatedSVD(n_components=20, random_state=42)
svd_full.fit(X_digits)

fig = px.bar(
    x=list(range(1, 21)),
    y=svd_full.singular_values_,
    color_discrete_sequence=['#9b7fd4'],
    labels={'x': 'Component', 'y': 'Singular Value'},
    title='The Hierarchy of Power — Singular Values of Digitia'
)
fig.show()
```

<iframe src="/assets/charts/digits-singular-values.html" style="width:100%;height:500px;border:none;"></iframe>

The first component towers over the rest. The hierarchy of power drops sharply. This is the Decomposer's gift: it tells you how many dimensions truly matter.

---

# The Mage's Conclusion

PCA and SVD are the honest arts. They make a promise — *preserve the most variance, or reveal the most important components* — and they keep it. They give you a receipt. They tell you exactly what they kept and what they discarded.

But their honesty has a price. They speak only in straight lines. The Enchanted Scroll defeated them. The tangled neighborhoods of Digitia remain tangled.

For those problems, the kingdom turned to darker, more dramatic arts.

Those stories are told in [Part 2 →](/data-science/2026-06-27-DimReduction-part2-post/)

---

*All code for this post is available in the [Jewpyter notebook repository](https://github.com/wolfenfeld/jewpyter/blob/master/notebooks/DimensionalityReduction.ipynb).*

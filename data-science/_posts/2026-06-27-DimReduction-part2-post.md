---
layout: post
title: "Too Much Information — Part 2"
description: |
  t-SNE and UMAP: The Methods Your Statistics Professor Warned You About
image: /assets/img/DimReduction-post/cover-part2.jpg
noindex: true
---

In [Part 1](/data-science/2026-06-27-DimReduction-part1-post/) we tried to bring order to chaos using PCA and SVD.

PCA found the main argument in the iris dataset and separated three species beautifully. SVD learned the shared patterns of human faces and could reconstruct a complete stranger from those patterns alone. Impressive results. Very respectable. The kind of thing you'd mention at Shabbat dinner without embarrassment.

But at the end of Part 1, we asked a harder question: can PCA look at 400 family photos and tell us which faces belong together? The answer was a resounding no — a tangle so hopeless it looked like the seating chart at a divorced family's wedding.

PCA found the biggest sources of variation in the data — lighting, head angles, expressions — and completely missed identity. The thing we actually cared about was invisible to it.

Mathityahu knew two people. They were not the kind you'd invite to a conference, or introduce to your mother. But when the data doesn't cooperate, you don't call the respectable ones.

They promised results. They delivered. They just came with more caveats than a rental car agreement.

---

# Why Straight Lines Aren't Enough

PCA failed on the family photos because it thinks in straight lines. It finds the single global direction of maximum spread and calls that the first dimension. But identity isn't spread along a global direction — forty families are forty local neighborhoods scattered through face-space. No single line passes through all of them.

This turns out to be true of most interesting data:

- Faces don't lie on a flat plane. They curve through pixel-space along axes of age, expression, lighting, and identity — all tangled together.
- The meaning of words isn't linear. "King" minus "man" plus "woman" lands near "queen" only because the embedding has learned a curved surface of relationships.
- Handwritten digits aren't arranged along a straight axis. A `4` doesn't differ from a `9` in a single direction. The structure is curved, folded, complicated.

To find structure in folded data, you need a method that follows the folds instead of trying to flatten everything with one stroke.

---

# t-SNE — The Cousin Who Always Knows Who's Related to Whom

The first of the newer methods is **t-SNE** — *t-distributed Stochastic Neighbor Embedding*. A name that takes longer to say than most Talmudic tractates, but the idea is simpler than it sounds.

t-SNE doesn't ask *what's the biggest source of variation?* It asks something much more local: *who are your closest neighbors?*

For every point in the data, it figures out who that point is most similar to — its neighborhood. Then it arranges everything in 2D so that neighbors stay close to each other and non-neighbors get pushed far away. The heavy tails of a t-distribution are used to create dramatic separation between clusters.

It ignores global structure entirely. It only cares about *who sits next to whom*.

## The Family Photos, Take Two

Here's what PCA produced on our 400 family photos — the same mess from Part 1:

<iframe src="/assets/charts/faces-pca.html" style="width:100%;height:500px;border:none;"></iframe>

Forty families, completely indistinguishable. Lighting, angle, and expression drowned out identity.

Now t-SNE tries the same job.

```python
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px

dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
X_faces = dataset.data
y_faces = dataset.target.astype(str)

X_scaled = StandardScaler().fit_transform(X_faces)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_faces_tsne = tsne.fit_transform(X_scaled)

fig = px.scatter(
    x=X_faces_tsne[:, 0], y=X_faces_tsne[:, 1],
    color=y_faces,
    labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
    title='The Family Photos — t-SNE Finds the Families',
    color_discrete_sequence=px.colors.qualitative.Alphabet
)
fig.update_traces(marker=dict(size=8, opacity=0.8))
fig.update_layout(showlegend=False)
fig.show()
```

<iframe src="/assets/charts/faces-tsne.html" style="width:100%;height:500px;border:none;"></iframe>

Forty islands. Each one a family — ten photos of the same person, clustered together, without t-SNE ever being told who belongs to whom.

It didn't know the labels. It only asked: *who looks most like whom?* And the families sorted themselves out.

This is the difference. PCA looks for global variation — and the biggest sources of variation across 400 photos are lighting and angle. t-SNE looks for local similarity — and within any neighborhood, the dominant signal is *this face looks like that face*, which is exactly identity.

It's the difference between trying to find your relatives by height (global) versus by nose shape (local). The nose works better. It usually does.

## Now Let's Try It on Numbers

Word spread. A delegation arrived from a province whose citizens were handwritten digits — each described by 64 pixel measurements, 1,797 of them total. They had the same problem: ten kinds of digit, all piled together with no visible organisation.

They asked t-SNE to draw a map.

```python
import numpy as np
import plotly.express as px
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X)

fig = px.scatter(
    x=X_tsne[:, 0], y=X_tsne[:, 1],
    color=[str(d) for d in y],
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
    title='The Digits — t-SNE (perplexity=30)'
)
fig.update_traces(marker=dict(size=5, opacity=0.8))
fig.show()
```

<iframe src="/assets/charts/digits-tsne-30.html" style="width:100%;height:500px;border:none;"></iframe>

Ten islands. One per digit. The chaos became a constellation.

But — and there is always a but — the statistician in the corner raised his hand.

*"Ask it what the distances between the islands mean."*

## The Perplexity Problem, or: It Depends

t-SNE has a parameter called **perplexity** — roughly, how many neighbors it considers for each point. Change this number and you get an entirely different map. Not a slightly different map. An *entirely* different one.

```python
fig_list = []
for perplexity in [5, 30, 100]:
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    X_t = tsne.fit_transform(X)
    for i, label in enumerate(y):
        fig_list.append({'x': X_t[i, 0], 'y': X_t[i, 1],
                         'digit': str(label), 'perplexity': f'perplexity={perplexity}'})

import pandas as pd
df = pd.DataFrame(fig_list)

fig = px.scatter(
    df, x='x', y='y', color='digit', facet_col='perplexity',
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={'x': '', 'y': ''},
    title='The Perplexity Problem — Same Data, Three Different Stories'
)
fig.update_traces(marker=dict(size=4, opacity=0.7))
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.show()
```

<iframe src="/assets/charts/digits-tsne-perplexity-comparison.html" style="width:100%;height:500px;border:none;"></iframe>

Three maps. Three entirely different stories. All technically correct.

This is t-SNE's dirty secret, which it will not volunteer unless you ask. Read the fine print:

1. **The distances between clusters are meaningless.** t-SNE pushed non-neighbors apart regardless of how far apart they really are. Two clusters that look far apart on the map might be close in the original data.
2. **The size of clusters is meaningless.** Dense regions get expanded, sparse regions get compressed. A large island might be a small village.
3. **Run it twice, get two different maps.** Always set `random_state`. Always.

**When to use t-SNE:** When you want to check whether local structure exists — do similar things cluster together? It's excellent for this. Just don't try to read the distances between clusters. That's asking for trouble.

---

# UMAP — The More Reasonable One

The second method is **UMAP** — *Uniform Manifold Approximation and Projection*. If t-SNE is the brilliant but unreliable cousin who produces spectacular results you can't quite trust, UMAP is the one who went to therapy and came back with better boundaries.

UMAP preserves local neighborhoods like t-SNE does, but it also tries to preserve the global shape of the data. It's faster, more reproducible, and the relative positions of clusters carry at least *some* meaning.

*"I care about neighbors too,"* UMAP says, *"but I also remember where things came from."*

```python
import umap  # pip install umap-learn

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)

fig = px.scatter(
    x=X_umap[:, 0], y=X_umap[:, 1],
    color=[str(d) for d in y],
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
    title='The Digits — UMAP'
)
fig.update_traces(marker=dict(size=5, opacity=0.8))
fig.show()
```

<iframe src="/assets/charts/digits-umap.html" style="width:100%;height:500px;border:none;"></iframe>

Beautiful clusters again — but now the layout is more stable between runs, and the global arrangement reflects actual similarity. The `4`s and `9`s live nearby (they look similar). The `0`s and `1`s are far apart (they don't).

## The n_neighbors Parameter, or: How Many Relatives Count?

UMAP has its own tuning knob: `n_neighbors`. Small values focus on very local structure — tight clusters, many of them. Large values zoom out and show the broader continental layout.

```python
records = []
for n_neighbors in [5, 15, 50]:
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    X_u = reducer.fit_transform(X)
    for i, label in enumerate(y):
        records.append({'x': X_u[i, 0], 'y': X_u[i, 1],
                        'digit': str(label), 'n_neighbors': f'n_neighbors={n_neighbors}'})

df = pd.DataFrame(records)

fig = px.scatter(
    df, x='x', y='y', color='digit', facet_col='n_neighbors',
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={'x': '', 'y': ''},
    title='Local vs Global — The n_neighbors Tradeoff'
)
fig.update_traces(marker=dict(size=4, opacity=0.7))
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.show()
```

<iframe src="/assets/charts/digits-umap-neighbors-comparison.html" style="width:100%;height:500px;border:none;"></iframe>

Small `n_neighbors`: many small clusters, fine village structure. Large `n_neighbors`: fewer, broader regions, continental layout. Neither is more correct. They answer different questions, like asking how big is your family — immediate household or everyone who shows up at Pesach.

---

# The Full Comparison — All Four Methods, One Dataset

Let's put all four methods side by side on the same data and see what each one tells us.

```python
import time
from sklearn.decomposition import PCA, TruncatedSVD

records = []
methods = {}

pca = PCA(n_components=2)
methods['PCA'] = pca.fit_transform(X)

svd = TruncatedSVD(n_components=2, random_state=42)
methods['SVD'] = svd.fit_transform(digits.data)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
methods['t-SNE'] = tsne.fit_transform(X)

reducer = umap.UMAP(n_components=2, random_state=42)
methods['UMAP'] = reducer.fit_transform(X)

for method_name, X_reduced in methods.items():
    for i, label in enumerate(y):
        records.append({'x': X_reduced[i, 0], 'y': X_reduced[i, 1],
                        'digit': str(label), 'method': method_name})

df = pd.DataFrame(records)

fig = px.scatter(
    df, x='x', y='y', color='digit', facet_col='method',
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={'x': '', 'y': ''},
    title='Four Methods, One Dataset'
)
fig.update_traces(marker=dict(size=4, opacity=0.7))
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_layout(height=450)
fig.show()
```

<iframe src="/assets/charts/digits-grand-tournament.html" style="width:100%;height:500px;border:none;"></iframe>

PCA and SVD: honest, interpretable, partial. t-SNE and UMAP: dramatic, beautiful, handle with care.

Both sets are correct. They answer different questions.

---

# So When Do You Use What?

After all of this, the practical guide is actually pretty short:

| What you want to know | Use this |
|---|---|
| Which features drive the most variance? | PCA |
| Sparse data — text, clicks, ratings | SVD |
| Do similar things cluster together in my embedding? | t-SNE |
| Fast exploration of a large dataset | UMAP |
| Preprocessing before model training | PCA |
| A reproducible visualization to show someone | UMAP (with fixed `random_state`) |

And the one rule that covers all cases:

> *Always run PCA first. It's fast, interpretable, and free. If your first two components explain 95% of variance, you don't need t-SNE or UMAP at all. The simple answer is the right one. This applies to data science and also to most arguments.*

---

# Epilogue

We started with bubbe's filing system — complete, impeccable, unusable.

We ended with four methods for making it usable, each with its own strengths and its own caveats, each telling a slightly different version of the truth.

The data is still high-dimensional. It always will be. But now at least we can look at it.

And somewhere in the data, a `4` and a `9` are sitting next to each other in the original 64-dimensional space, wondering why every map puts them so far apart.

*They're not that different, those two. If you look closely enough.*

---

[← Part 1: PCA and SVD](/data-science/2026-06-27-DimReduction-part1-post/)

*All code for this post is available in the [Jewpyter notebook repository](https://github.com/wolfenfeld/jewpyter/blob/master/notebooks/DimensionalityReduction.ipynb).*

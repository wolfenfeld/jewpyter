---
layout: post
title: "The Map Maker's Dilemma — Part 2"
description: |
  t-SNE and UMAP: The Dark Arts of the Dimensionality Mages
image: /assets/img/DimReduction-post/cover-part2.jpg
noindex: true
---

In [Part 1](/data-science/2026-06-27-DimReduction-part1-post/) we met the honest mages — those who practised PCA and SVD.
PCA revealed the spine of the Iris Fields. SVD learned the patterns of a thousand faces and could recognise a stranger it had never seen.
Their maps were truthful. Their receipts were impeccable.

But at the end of Part 1, the honest arts met their limit.

The **Face Vault** held portraits of 40 noble families. SVD could compress and reconstruct any face — but when PCA was asked to arrange them on a map, hoping that family members would cluster together, it produced a tangle. Forty families, indistinguishable. The directions of greatest variance turned out to be lighting and expression — not identity. The structure the Queen cared about was invisible to a straight-line art.

The Queen was not satisfied.

And so the Council summoned two younger mages — practitioners of the **Dark Arts of Nonlinear Projection**.
They promised maps of impossible beauty. They delivered.

They just didn't always tell the whole truth.

---

# The Problem With Straight Lines

The honest arts failed on the Face Vault because they speak only in straight lines — linear combinations of features. PCA finds the global direction of greatest variance across all 400 portraits. But the 40 families are not arranged along a global axis — they occupy 40 small, scattered neighborhoods in face-space. No straight line can pass through all 40 of them at once.

This is not a quirk of this particular dataset. Most real high-dimensional data has this property:

- The space of human faces does not lie on a flat plane — it curves through pixel space along axes of age, expression, lighting, and identity.
- The space of word meanings is not linear — "king" minus "man" plus "woman" lands near "queen" only because the embedding has learned a curved manifold of relationships.
- The digits in Digitia are not arranged along a straight axis. A `4` does not differ from a `9` in a single linear direction. The structure is curved, tangled, folded.

To map a folded land, you need a mage who can follow the folds.

---

# t-SNE — The Map Maker of Neighborhoods

The first dark mage arrived from the eastern province of Stochastia, carrying a technique called **t-SNE** — *t-distributed Stochastic Neighbor Embedding*, or as she called it: *The Neighborhood Preserving Enchantment*.

Her philosophy was different from the honest mages.

*"I do not care about distances,"* she said. *"I care about neighbors. Tell me who lives next to whom in the high-dimensional kingdom, and I will place them next to each other on the map."*

The algorithm worked like this: for each citizen, she asked — *who are your closest neighbors in the high-dimensional archive?* She then arranged everyone in 2D so that those neighbors stayed close. Non-neighbors were pushed far away, using the heavy tails of a t-distribution to create dramatic separation.

## Reuniting the Families in the Face Vault

The Face Vault held a tangle. Recall the map PCA produced in Part 1 — forty families, completely indistinguishable:

<iframe src="/assets/charts/faces-pca.html" style="width:100%;height:500px;border:none;"></iframe>

PCA found the directions of greatest variance — lighting, head angle, expression — but identity was drowned out. The Queen sent for t-SNE.

The dark mage did not look at the whole vault at once. She asked every portrait a single question: *who are your closest neighbors?* Then she arranged all 400 portraits in 2D so that neighbors stayed near each other. The global distances between families were irrelevant to her — only the local question mattered.

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
    title='The Face Vault — t-SNE Finds the Families',
    color_discrete_sequence=px.colors.qualitative.Alphabet
)
fig.update_traces(marker=dict(size=8, opacity=0.8))
fig.update_layout(showlegend=False)
fig.show()
```

<iframe src="/assets/charts/faces-tsne.html" style="width:100%;height:500px;border:none;"></iframe>

Forty islands. Each one a family — ten portraits of the same person, clustered together without ever being told who belongs to whom. t-SNE did not know the labels. It only asked: *who looks most like whom?* And the families revealed themselves.

The Queen looked at the forty islands and smiled. *"Now we can find anyone."*

The eldest mage cleared his throat. *"Before we celebrate, Your Majesty, there is the matter of the perplexity parameter."*

## Casting the Spell on Digitia

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
    title='The Digits of Digitia — t-SNE (perplexity=30)'
)
fig.update_traces(marker=dict(size=5, opacity=0.8))
fig.show()
```

<iframe src="/assets/charts/digits-tsne-30.html" style="width:100%;height:500px;border:none;"></iframe>

The Queen gasped. Ten beautiful islands floated in the void — one for each digit. The tangled mass had become a constellation.

But the eldest mage leaned over and whispered: *"Ask her what the distances mean."*

## The Perplexity Curse

The dark mage had a secret. Her map depended on a single incantation parameter — **perplexity** — that controlled how many neighbors she considered. Change the perplexity, and the map changed entirely.

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
    title='The Perplexity Curse — Same Data, Three Different Kingdoms'
)
fig.update_traces(marker=dict(size=4, opacity=0.7))
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.show()
```

<iframe src="/assets/charts/digits-tsne-perplexity-comparison.html" style="width:100%;height:500px;border:none;"></iframe>

Three maps. Three entirely different stories. All of them are technically correct t-SNE outputs.

The Council declared three laws governing the use of t-SNE:

1. **The distances between islands are meaningless.** The mage pushed non-neighbors apart regardless of how far they truly were. Two islands far apart on the map may be neighbors in reality.
2. **The size of islands is meaningless.** Dense regions are expanded, sparse ones are compressed. A large island may represent a tiny village.
3. **The map is not reproducible without the random seed.** Run the spell twice, get two different kingdoms. Always record your `random_state`.

**When to summon t-SNE:** When you want to verify that local structure exists. If you trained a word embedding and want to confirm that similar words cluster together — t-SNE will show you. Just do not read the inter-cluster distances.

---

# UMAP — The Pragmatist of the Dark Arts

The second dark mage arrived from the western province of Topology. She practised **UMAP** — *Uniform Manifold Approximation and Projection*, or as she modestly called it: *The Reasonable Art*.

She had studied the failures of the t-SNE mage and built something more balanced.

*"I preserve neighborhoods too,"* she said, *"but I also try to preserve the global shape of the land. My maps are faster to draw, more reproducible, and the relative positions of the islands carry at least some meaning."*

Her magic was built on Riemannian geometry and topological data analysis — arts so ancient even the eldest mage had only read about them. But the results spoke for themselves.

## Summoning UMAP

```python
import umap  # pip install umap-learn

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)

fig = px.scatter(
    x=X_umap[:, 0], y=X_umap[:, 1],
    color=[str(d) for d in y],
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
    title='The Digits of Digitia — UMAP'
)
fig.update_traces(marker=dict(size=5, opacity=0.8))
fig.show()
```

<iframe src="/assets/charts/digits-umap.html" style="width:100%;height:500px;border:none;"></iframe>

Beautiful islands again — but now with more consistent positioning between runs, and a global layout that roughly reflects how similar the digits are to each other. The `4`s and `9`s live nearby. The `0`s and `1`s are far apart.

## The n_neighbors Incantation

Like t-SNE, UMAP has its own parameter that changes the story. `n_neighbors` controls the balance between local and global structure.

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
    title='The n_neighbors Incantation — Local vs Global Structure'
)
fig.update_traces(marker=dict(size=4, opacity=0.7))
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.show()
```

<iframe src="/assets/charts/digits-umap-neighbors-comparison.html" style="width:100%;height:500px;border:none;"></iframe>

Small `n_neighbors` reveals fine village structure — many small clusters. Large `n_neighbors` shows the continental layout — fewer, broader regions. Neither is more true. They answer different questions.

---

# The Great Tournament — All Four Arts on One Field

Finally, the Council held a tournament. All four mages cast their spells on the same kingdom — the Digits of Digitia — and the results were displayed side by side.

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
    title='The Grand Tournament — Four Arts, One Kingdom'
)
fig.update_traces(marker=dict(size=4, opacity=0.7))
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_layout(height=450)
fig.show()
```

<iframe src="/assets/charts/digits-grand-tournament.html" style="width:100%;height:500px;border:none;"></iframe>

PCA and SVD show the honest, partial picture. t-SNE and UMAP show the dramatic, beautiful one.

Both are true. They answer different questions.

---

# The Map Maker's Code of Honor

After the tournament, the Council inscribed the following laws on the Archive walls:

| Question to answer | Summon |
|---|---|
| Which features drive the most variance? | PCA |
| Sparse text or interaction data | SVD |
| Does my embedding group similar items together? | t-SNE |
| Fast exploration of a large dataset | UMAP |
| Preprocessing before training a model | PCA |
| A reproducible visualization for a report | UMAP (with fixed `random_state`) |
| Making something that impresses the Queen | any of them |

And above all laws, the eldest mage added one final rule:

> *Always run PCA first. It costs nothing. If your first two components explain 95% of variance, you do not need the dark arts. The honest map is enough.*

---

# Epilogue

The Queen received her maps. They were beautiful. Some were honest. Some were dramatic. All of them were useful, provided you knew which lie each one was telling.

The mages returned to their towers. The Archive remained infinite.

And somewhere in the northern province of Digitia, a `4` and a `9` sat next to each other in the high-dimensional space, wondering why every map placed them so far apart.

---

[← Part 1: PCA and SVD](/data-science/2026-06-27-DimReduction-part1-post/)

*All code for this post is available in the [Jewpyter notebook repository](https://github.com/wolfenfeld/jewpyter/blob/master/notebooks/DimensionalityReduction.ipynb).*

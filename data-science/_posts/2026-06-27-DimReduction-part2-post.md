---
layout: post
title: "Don't Trust the Map — Part 2"
description: |
  t-SNE and UMAP: The Dramatic Storytellers
image: /assets/img/DimReduction-post/cover.jpg
noindex: true
---

In [Part 1](/data-science/2026-06-27-DimReduction-part1-post/) we looked at PCA and SVD — the linear methods.
They're fast, honest, and interpretable. They also leave a lot of structure on the table.

In this part we meet the non-linear methods: t-SNE and UMAP.
They produce the beautiful cluster plots you see in every ML paper and conference talk.
They are also, in specific and important ways, liars.

Let's understand exactly what they're doing and when to trust them.

---

# The Problem With Linear

On the digits dataset from Part 1, PCA gave us a plot with overlapping blobs.
That's not because the digits are hard to separate — a simple classifier gets 97%+ accuracy on this data.
It's because the separating structure is non-linear, and PCA can only see straight lines.

t-SNE and UMAP can see curves.

---

# t-SNE — The Method That Made Everyone Excited About Embeddings

t-SNE (t-distributed Stochastic Neighbor Embedding) was introduced in 2008 and became the default visualization tool for high-dimensional data for years. It produces the kind of plots where tight colored clusters snap into view and everything looks learned and structured.

## What It's Actually Doing

t-SNE doesn't try to preserve distances. It tries to preserve **neighborhoods**.

For each point in high-dimensional space, it builds a probability distribution over all other points: nearby points get high probability, far points get near-zero. It then arranges points in 2D so that the neighborhood distributions match as closely as possible.

The trick that makes it work is the **t-distribution** in 2D (hence the name). The t-distribution has heavier tails than a Gaussian, which means it's happy to put non-neighbors very far apart in 2D. This is what creates the dramatic cluster separation you see in the plots.

## Running t-SNE

```python
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target

# perplexity is the key hyperparameter — roughly "how many neighbors to consider"
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=10, alpha=0.8)
plt.colorbar(scatter)
plt.title('t-SNE on digits dataset (perplexity=30)')
plt.show()
```

Beautiful. The digit clusters are clean and well-separated. This is the plot that gets screenshotted.

## Now Watch It Lie

Change one number — the perplexity — and the map changes completely:

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, perplexity in zip(axes, [5, 30, 100]):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
    ax.set_title(f'perplexity={perplexity}')

plt.suptitle('Same data, same method, different perplexity — different map', y=1.02)
plt.tight_layout()
plt.show()
```

Three very different maps. All of them are "correct" t-SNE outputs. None of them is more true than the others.

This is the core problem with t-SNE: the map is a function of your hyperparameters as much as it is of your data.

## The Three Rules of t-SNE

**1. Distances between clusters are meaningless.**
The algorithm actively pushes non-neighbors apart regardless of how far they actually were. Two clusters being far apart in the plot tells you nothing about their actual separation in feature space.

**2. Cluster sizes are meaningless.**
Dense regions get expanded, sparse regions get compressed. A large cluster in t-SNE space might correspond to a tiny, tight cluster in the original space.

**3. It is non-deterministic.**
Without `random_state`, two runs give two different maps. Always set `random_state` when reporting results.

**What t-SNE is good for:** Verifying that local structure exists. If you trained a word embedding and want to check that similar words cluster together — t-SNE is perfect. You're asking "do neighbors stay neighbors?" and t-SNE answers that question well. Just don't read anything into the global layout.

---

# UMAP — The Pragmatist

UMAP (Uniform Manifold Approximation and Projection) is the current state of the art for most use cases. It's faster than t-SNE, more reproducible, and does a better job of preserving global structure alongside local structure.

Under the hood it's mathematically more sophisticated than t-SNE — it's built on ideas from topology and Riemannian geometry — but intuitively it's asking a similar question: how do I arrange points in 2D such that the neighborhood graph looks similar to the one in high-dimensional space?

## Running UMAP

```python
# pip install umap-learn
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=10, alpha=0.8)
plt.colorbar(scatter)
plt.title('UMAP on digits dataset')
plt.show()
```

## The Key Hyperparameters

`n_neighbors` controls the balance between local and global structure:
- Small values (5–10): focus on fine-grained local neighborhoods, many small clusters
- Large values (50–200): zoom out, global structure dominates

`min_dist` controls how tightly points are packed in 2D:
- Small values (0.0–0.1): tight clusters, good for cluster identification
- Large values (0.5–1.0): more spread out, better for seeing continuous structure

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
params = [(5, 0.0), (5, 0.5), (50, 0.0), (50, 0.5)]

for ax, (n_neighbors, min_dist) in zip(axes.flat, params):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X_umap = reducer.fit_transform(X)
    ax.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
    ax.set_title(f'n_neighbors={n_neighbors}, min_dist={min_dist}')

plt.tight_layout()
plt.show()
```

Again: same data, different maps. The story UMAP tells depends on the parameters you choose.

## UMAP vs t-SNE — A Direct Comparison

```python
import time

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# t-SNE
start = time.time()
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
tsne_time = time.time() - start
axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
axes[0].set_title(f't-SNE ({tsne_time:.1f}s)')

# UMAP
start = time.time()
X_umap = umap.UMAP(random_state=42).fit_transform(X)
umap_time = time.time() - start
axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
axes[1].set_title(f'UMAP ({umap_time:.1f}s)')

plt.tight_layout()
plt.show()
```

On most datasets UMAP runs 5–10x faster than t-SNE. For large datasets (100k+ points) this difference is the deciding factor.

---

# The Complete Picture

Here's all four methods on the same data side by side:

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

fig, axes = plt.subplots(1, 4, figsize=(20, 4))

methods = {
    'PCA': PCA(n_components=2).fit_transform(X),
    'SVD': TruncatedSVD(n_components=2).fit_transform(X),
    't-SNE': TSNE(n_components=2, random_state=42).fit_transform(X),
    'UMAP': umap.UMAP(random_state=42).fit_transform(X),
}

for ax, (name, X_reduced) in zip(axes, methods.items()):
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Digits dataset — four projections of the same truth', y=1.02)
plt.tight_layout()
plt.show()
```

PCA and SVD show you the honest picture: partial separation, lots of overlap, because the linear projection can only do so much. t-SNE and UMAP show you the dramatic picture: tight, clean clusters that feel definitive.

Neither is more "correct." They're answering different questions.

---

# The Decision Guide

| Question | Method |
|---|---|
| What features drive the variance in my data? | PCA |
| I have sparse text/interaction data | SVD (TruncatedSVD) |
| I want to check if my embedding learned local structure | t-SNE |
| I need fast exploration of a large dataset | UMAP |
| I'm preprocessing features for a downstream model | PCA |
| I want the global layout to be somewhat meaningful | UMAP over t-SNE |
| I need a reproducible plot for a paper | UMAP (with fixed `random_state`) |

One meta-rule: **always run PCA first.** It takes seconds. If your first two principal components explain 95% of variance, you don't need UMAP — you're already done. Use the non-linear methods when PCA leaves too much on the floor.

---

# Conclusion

PCA and SVD (Part 1) make honest promises: they preserve variance, they tell you how much they preserved, and the result is the same every time.

t-SNE and UMAP make your data look beautiful, but the beauty is partly construction. The distances they show you are not distances in your original space. The clusters they reveal are real — but their sizes, separations, and shapes are artifacts of the algorithm and its hyperparameters.

This doesn't make them bad tools. It makes them tools you need to understand before you trust.

The next time someone shows you a beautiful cluster plot, ask them two questions:
1. Which method did you use?
2. What happens when you change the perplexity?

The map is not the territory. But a map you understand is still useful.

[← Part 1: PCA and SVD](/data-science/2026-06-27-DimReduction-part1-post/)

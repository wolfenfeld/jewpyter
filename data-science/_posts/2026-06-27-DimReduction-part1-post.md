---
layout: post
title: "Don't Trust the Map — Part 1"
description: |
  PCA and SVD: The Honest Cartographers
image: /assets/img/DimReduction-post/cover.jpg
noindex: true
---

I once spent three hours staring at a beautiful 2D plot of our production data.
Tight blobs, clear separation, the kind of thing you screenshot and send to your manager.
Then my colleague pointed out that I had forgotten to scale the features.
The beautiful structure was entirely an artifact of one column having values in the millions while everything else was between 0 and 1.

The map looked great. The map was lying.

Dimensionality reduction is one of the most visually satisfying tools in a data scientist's toolkit, and one of the most frequently misunderstood.
In this two-part series I want to pull back the curtain on what these methods are actually doing — and, more importantly, when they deceive you.

**Part 1** covers PCA and SVD — the linear methods, the honest ones.
**Part 2** will cover t-SNE and UMAP — the non-linear ones, the dramatic ones.

Let's start.

---

# The Setup

We have data in high-dimensional space. A customer with 500 behavioral features. A document with 10,000 word counts. An image with 50,000 pixels.
We want to look at it. We can only look in 2D.

So we project it down and look. The question is always: **what did we lose, and does it matter?**

---

# PCA — Find the Directions That Matter

Principal Component Analysis answers a simple question: *what are the directions in my data along which things vary the most?*

Think of your data as a cloud of points in high-dimensional space. PCA finds the axis along which that cloud is most stretched out (the first principal component), then the axis perpendicular to that which has the next most stretch (the second), and so on.

Projecting your data onto the first two principal components gives you the 2D view that preserves the most variance.

## Why This Is Honest

PCA makes a simple, auditable promise: **maximize preserved variance**. You can measure exactly how much information you kept:

```python
from sklearn.decomposition import PCA
import numpy as np

pca = PCA()
pca.fit(X)

# How much variance does each component explain?
print(pca.explained_variance_ratio_)

# How many components to keep 95% of variance?
cumulative = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumulative >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components}")
```

The `explained_variance_ratio_` is PCA's receipt. It tells you exactly what you kept and what you threw away. No other method gives you this.

## Where PCA Lies

PCA is a linear method. It can only draw straight lines through your data.
If the meaningful structure is non-linear — a Swiss roll, concentric rings, a curved manifold — PCA will flatten it and you'll see nothing.

```python
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

X, color = make_swiss_roll(n_samples=1500, noise=0.05)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap='viridis', s=5)
plt.title(f'PCA — explained variance: {pca.explained_variance_ratio_.sum():.1%}')
plt.show()
```

Run this and you'll see a smear. The Swiss roll has been flattened and the structure is invisible. PCA told you it explained 90%+ of variance and technically it wasn't lying — but the 10% it dropped was the part you cared about.

## When to Use PCA

- As your **first exploratory step**, always. It's fast and the result is interpretable.
- When you need a **preprocessing step** before feeding data to a classifier or clustering algorithm. PCA-whitened features often train faster and more stably.
- When you want to know **which original features matter**. The loadings (`pca.components_`) tell you which features each principal component is made of.
- When your data structure is approximately linear.

---

# SVD — The Engine Under the Hood

Singular Value Decomposition is not usually listed alongside PCA and t-SNE in "dimensionality reduction" tutorials. But it should be, because PCA *is* SVD.

When `sklearn` computes PCA, it calls SVD internally. Understanding SVD gives you a clearer picture of what's actually happening, and opens up a family of related techniques (LSA for text, matrix factorization for recommender systems, image compression) that all share the same core idea.

## What SVD Does

SVD decomposes any matrix **X** into three matrices:

**X = U · Σ · Vᵀ**

- **U** — the left singular vectors (one per data point)
- **Σ** — a diagonal matrix of singular values, sorted largest to smallest
- **Vᵀ** — the right singular vectors (one per feature, these are your principal components)

The singular values in **Σ** tell you how important each dimension is. Keeping only the top *k* singular values and their corresponding vectors gives you the best possible rank-*k* approximation of your data — in a precise mathematical sense.

```python
import numpy as np

# Manual SVD
U, sigma, Vt = np.linalg.svd(X, full_matrices=False)

# Project to 2D — equivalent to PCA
k = 2
X_reduced = U[:, :k] * sigma[:k]

# Singular values tell you the same story as explained_variance_ratio_
variance_explained = (sigma**2) / (sigma**2).sum()
print(variance_explained[:10])
```

## SVD for Text — Latent Semantic Analysis

One place where SVD shines on its own (not just as PCA's engine) is text.

If you build a term-document matrix — rows are documents, columns are words, values are counts — it's enormous and sparse. SVD finds the *latent topics* hiding in the word co-occurrence patterns.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

corpus = [
    "machine learning is great",
    "deep learning neural networks",
    "pandas numpy data science",
    "scikit learn classification regression",
]

vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(corpus)

# TruncatedSVD is efficient SVD for sparse matrices (what sklearn uses for LSA)
svd = TruncatedSVD(n_components=2)
X_lsa = svd.fit_transform(X_text)

print("Document positions in latent topic space:")
print(X_lsa)
```

The two dimensions in the output aren't "word1" and "word2" — they're latent topics that SVD discovered from the patterns of word co-occurrence. Documents about ML cluster together. Documents about data tools cluster together. Nobody told it that.

## The Connection Between SVD and PCA

If your data matrix **X** is mean-centered (subtract the column means), then:
- The right singular vectors **Vᵀ** are the principal components
- The singular values **σᵢ** are related to the explained variance by **σᵢ² / (n-1)**
- The projection **U · Σ** gives the same result as `pca.transform(X)`

They are the same thing. The difference is implementation: `PCA` mean-centers for you and reports variance; `TruncatedSVD` skips centering (useful for sparse matrices where centering destroys sparsity).

---

# Putting It Together

Here's a practical comparison on real data:

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
axes[0].set_title(f'PCA — {pca.explained_variance_ratio_.sum():.1%} variance')

# SVD
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)
axes[1].scatter(X_svd[:, 0], X_svd[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
axes[1].set_title(f'SVD — {svd.explained_variance_ratio_.sum():.1%} variance')

plt.tight_layout()
plt.show()
```

On the digits dataset the structure is partially visible — you can see some digit clusters emerging — but plenty of overlap remains. That's not a failure. That's PCA being honest: the linear projection can only do so much with this data.

In Part 2 we'll throw t-SNE and UMAP at the same dataset and watch the clusters snap into place. Then we'll talk about why that feels good and why you should be suspicious of it.

---

# Summary

| | PCA | SVD |
|---|---|---|
| What it finds | Directions of max variance | Low-rank matrix approximation |
| Linear? | Yes | Yes |
| Interpretable? | Yes (loadings, variance explained) | Yes (singular values) |
| Good for preprocessing? | Yes | Yes (especially sparse data) |
| Handles non-linear structure? | No | No |
| Use when | Tabular data, first exploration | Text, sparse matrices, recommender systems |

Both PCA and SVD make an honest promise and keep it. They tell you what they preserved and what they dropped.

The methods in Part 2 are less forthcoming. They'll show you something beautiful and not always tell you what they changed to make it look that way.

[Part 2 →](/data-science/2026-06-27-DimReduction-part2-post/)

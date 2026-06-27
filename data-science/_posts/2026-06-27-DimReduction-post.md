---
layout: post
title: Don't Trust the Map
description: |
  A Practical Guide to Dimensionality Reduction — PCA vs. t-SNE vs. UMAP
image: /assets/img/DimReduction-post/cover.jpg
noindex: true
---

A few months ago a colleague showed me a beautiful UMAP plot of our customer embeddings.
Tight clusters, clear separation, very satisfying.
"Look how well the model learned!" he said.

I asked him what the axes meant.
He wasn't sure.
I asked him if we could trust the distances between clusters.
He paused.

That's the thing about dimensionality reduction — the map always looks great. The question is whether it's lying to you.

In this post I want to do three things:
1. Explain intuitively what PCA, t-SNE, and UMAP are actually doing
2. Show where each one misleads you
3. Give you a practical rule for picking one

# Why We Reduce Dimensions At All

Most interesting data is high-dimensional. A customer has hundreds of behavioral features. A word embedding has 768 dimensions. An image has millions of pixels.

Humans can only see in 2D or 3D. So we project the data down and look at it.
The problem: any projection loses information. The map is not the territory.
The art is in understanding *which* information you're losing.

# PCA — The Honest Cartographer

Principal Component Analysis is the oldest and most transparent of the three.
It finds the directions in your high-dimensional space that capture the most variance, and projects onto those.

The key word is **linear**. PCA draws straight lines through your data.
This means:
- Distances in the 2D plot are meaningful — if two points are far apart in PCA space, they were far apart in the original space (roughly).
- Global structure is preserved. You can trust the big picture.
- Local structure (tight clusters of similar points) is often squashed. PCA doesn't care about small-scale geometry.

**When PCA lies:** When the meaningful structure in your data is non-linear — a Swiss roll, concentric circles, a manifold. PCA will flatten it into a smear and you'll see nothing useful.

**When to use it:** As your first step, always. It's fast, interpretable, and the loadings tell you which original features matter. If PCA already separates your classes, you're done.

# t-SNE — The Dramatic Storyteller

t-SNE (t-distributed Stochastic Neighbor Embedding) does something fundamentally different.
It doesn't try to preserve distances. It tries to preserve *neighborhoods*.

The algorithm asks: for each point, who are its nearest neighbors in high-dimensional space?
It then arranges points in 2D so that neighbors stay neighbors — and non-neighbors are pushed far away.

This produces the beautiful, tight-cluster plots that make everyone excited in presentations.

**When t-SNE lies:**
- **Distances between clusters are meaningless.** The fact that the "fraud" cluster is far from the "loyal customer" cluster in your t-SNE plot tells you nothing about how different they actually are in feature space.
- **Cluster sizes are meaningless.** t-SNE expands dense regions and compresses sparse ones to make things look clean.
- **It's non-deterministic.** Run it twice, get two different maps. The hyperparameter `perplexity` changes the story dramatically — low perplexity gives many small clusters, high perplexity gives fewer large ones. Both can look convincing.
- **It doesn't scale.** On large datasets it gets very slow.

**When to use it:** When you want to visually verify that local structure exists — e.g., checking that a trained embedding actually groups similar items together. Never for quantitative analysis. Always report your perplexity setting.

# UMAP — The Pragmatist

UMAP (Uniform Manifold Approximation and Projection) is the new default for most practitioners, and for good reason.
Like t-SNE it preserves local structure, but it also does a better job of preserving global structure — the relative positions of clusters carry more meaning than they do in t-SNE.

It's also dramatically faster.

**When UMAP lies:**
- Global distances are *better* than t-SNE but still not fully trustworthy. Don't measure inter-cluster distances and draw conclusions.
- Like t-SNE, it has hyperparameters (`n_neighbors`, `min_dist`) that change the shape of the output. A small `n_neighbors` reveals fine-grained local structure; a large one shows the big picture. Neither is "true."
- It can create the illusion of clusters where none exist in the original space, particularly with certain `min_dist` settings.

**When to use it:** When t-SNE is too slow, when you need something more reproducible, or when you want the global layout to be somewhat meaningful. It's the best general-purpose choice for visualization today.

# The Practical Rule

| Goal | Use |
|---|---|
| Understand which features drive variance | PCA |
| Verify local cluster structure in embeddings | t-SNE or UMAP |
| Fast exploration of large datasets | UMAP |
| Preprocessing before a classifier | PCA |
| Making a slide that impresses your PM | any of them, they all look great |

One more rule: **always run PCA first.** Even if you plan to use UMAP in the end. PCA will tell you in seconds whether your data has any structure at all, and how many components you actually need. If the first two principal components explain 95% of variance, you don't need UMAP.

# Show Me The Code

The full notebook with a live comparison on the MNIST digits dataset — showing exactly how each method distorts the same data differently — is available [here](https://github.com/wolfenfeld/jewpyter/blob/master/notebooks/DimensionalityReduction.ipynb).

Run it yourself. Change the perplexity. Change `n_neighbors`. Watch the map shift.
That's the point — once you've seen the map change without the data changing, you'll never trust it unconditionally again.

# Conclusion

PCA, t-SNE, and UMAP are all telling you a story about your data.
PCA tells a simple, honest story that leaves out the local details.
t-SNE tells a dramatic story full of beautiful clusters that may or may not correspond to reality.
UMAP finds a middle ground — more honest than t-SNE, more expressive than PCA.

The map is always a simplification. Know which simplification you're making.

And the next time someone shows you a beautiful cluster plot — ask them what the axes mean.

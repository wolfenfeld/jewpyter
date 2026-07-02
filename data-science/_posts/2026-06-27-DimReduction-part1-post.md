---
layout: post
title: "Too Much Information — Part 1"
description: |
  PCA and SVD: Or, How to Stop Drowning in Data and Start Complaining About Less of It
image: /assets/img/DimReduction-post/cover-part1.jpg
noindex: true
---

My bubbe kept every piece of information she had ever received. Every receipt, every letter, every grudge — all of it, organised in a system that made perfect sense to her and to nobody else on earth. You'd ask her one simple question and she'd hand you fourteen folders and a story about something your uncle did in 1987.

This is high-dimensional data. You'd ask it what's for dinner and it would tell you the full history of the chicken.

So we called in the experts. Not rabbis — mathematicians. Which in some circles is worse.

They said: *"We can simplify this for you. But every simplification throws something away. The question is whether you care about what you're throwing away."*

Nu, so let's find out.

---

# PCA — Finding the One Thing Everyone Argues About

The first and most widely used technique is **Principal Component Analysis**, or PCA.

Here is the idea. Imagine you are at a family Passover seder. There are forty people in the room, all talking at once. If you had to summarise the conversation in one sentence, what would you say? You would find the single biggest source of disagreement — probably something about the haggadah, or whether Elijah actually showed up — and follow that thread. That's your first dimension. Then you'd find the second biggest argument, completely unrelated to the first. And so on.

PCA does exactly this with data. It finds the direction along which your data varies the most, calls it the first principal component, then finds the next most-varying direction perpendicular to that, and so on. The result is a lower-dimensional map that preserves as much of the original spread — the *variance* — as possible.

It won't capture everything. But it'll capture the main arguments.

## Let's Try It on Something Real

We start with the iris dataset — 150 flowers, each described by four measurements: sepal length, sepal width, petal length, petal width. Very modest. Very well-behaved. The kind of dataset you'd want to take home to meet your parents.

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

Look at that. Three species, separated beautifully. PCA found that the main argument among these flowers is petal size — and once you know that, the species practically sort themselves out.

We went from four dimensions to two, and the important structure is still there. That's the whole trick.

## Okay But What Did We Throw Away?

PCA is refreshingly honest about its own shortcomings — unlike most people I know.

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

The first two components capture 95.8% of all variance. We discarded 4.2%. Sounds small — and it is — but *what* is that 4.2%?

```python
feature_names = iris.feature_names
for i, component in enumerate(pca_full.components_):
    print(f"PC{i+1}: " + ", ".join(
        f"{name}: {weight:.2f}" for name, weight in zip(feature_names, component)
    ))
```

PC3 and PC4, which we threw out, are mostly **sepal width** — a measurement that doesn't help you tell the species apart, but does carry real information about flower shape within each species.

So what did we lose? The ability to tell a wide-petaled setosa from a narrow-petaled one. What did we keep? Everything needed to tell the three species apart.

This is always the question you have to ask. Not *how much* did we discard, but *do we care about what we discarded?* 4.2% of a Passover argument might be your cousin's opinion on the brisket. You can probably live without it.

---

# SVD — Recognising Your Relatives Without Remembering All of Them

Now we go deeper. **Singular Value Decomposition** — SVD — is PCA's older, more mathematically sophisticated cousin. The one who went to graduate school and never stopped mentioning it.

Where PCA finds directions of variance, SVD dismantles the entire data matrix into its fundamental building blocks:

**X = U · Σ · Vᵀ**

Three matrices. **U** holds a compact code for each data point. **Σ** holds the *singular values* — a ranked list from most important to least. **Vᵀ** holds the shared patterns underlying all the data.

The crucial insight: keep only the top *k* singular values and you get the *provably best* k-dimensional approximation of your data. Not approximately best. Provably, mathematically best.

This is what makes SVD the tool of choice for **recognition without memorisation**.

## The Shul's Photo Archive

Imagine your synagogue has been photographing every member for forty years. Four hundred photos. Sixty-four by sixty-four pixels each. That's 4,096 numbers per face. The shammes needs to recognise anyone who walks in — but carrying 4,096 numbers per person in his head is not realistic. The shammes is a busy man.

SVD offers a better way. Study 350 of the photos and learn the **shared patterns of human faces** — the way light falls on a forehead, the typical shape of eyebrows, the common structure of a nose. These patterns, ranked from most to least common, are the singular vectors. Once you have them, you don't need to store entire photos. You just store a short code — a handful of numbers — saying how much of each pattern each person has.

When someone new walks in, you project their face onto the learned patterns and reconstruct it. If the reconstruction matches a known member, you've got them.

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

With **k=5**, you get a ghostly smear — enough to confirm it's a human face, not a piece of kugel. With **k=50**, the features are clear. With **k=200**, it is nearly indistinguishable from the original.

And the really remarkable part: face 351 was **never seen during training**. SVD didn't memorise faces. It learned the *concept* of a face — and then applied that concept to a stranger.

This is how you recognise a distant relative at a bar mitzvah. You've never met them. But you know the family patterns — the nose, the eyebrows, the particular way they argue — and you say: *"You must be a Goldstein."*

## Which Patterns Matter Most?

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

The first singular value towers over the rest — it captures the single most universal pattern across all 350 faces. The drop is steep, then gradual. A handful of patterns do most of the work; the rest is individual noise.

Like a family. Most of what makes you a Goldstein is shared. The rest is just you being difficult.

## PCA and SVD — Two Names for the Same Meshugas

In truth, PCA *is* SVD under the hood. The sklearn implementation calls SVD internally. The practical difference:

- **PCA** mean-centers first and reports explained variance — ideal for exploration and understanding.
- **SVD** skips centering — essential for sparse data (text, ratings, clicks) where centering would destroy sparsity and exhaust your RAM. It also unlocks the reconstruction trick: project any new, unseen point onto the learned patterns.

Dense tabular data? Use PCA. Text, interaction logs, face recognition, compression? SVD directly.

---

# When PCA Runs Into a Wall

PCA and SVD have been impressive. PCA separated three flower species in two dimensions. SVD reconstructed an unseen face from patterns it learned without ever seeing that face.

But then someone asked the harder question.

*"SVD can reconstruct any face. But can PCA tell us which faces in the archive belong to the same family?"*

We have 400 photos of 40 families — ten photos each — all shuffled together with no labels. Can PCA arrange them so that family members cluster together?

```python
from sklearn.datasets import fetch_olivetti_faces

dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
X_faces = dataset.data
y_faces = dataset.target.astype(str)

X_scaled = StandardScaler().fit_transform(X_faces)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig = px.scatter(
    x=X_pca[:, 0], y=X_pca[:, 1],
    color=y_faces,
    labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
            'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
    title='The Face Vault — Can PCA Find the Families?',
    color_discrete_sequence=px.colors.qualitative.Alphabet
)
fig.update_traces(marker=dict(size=6, opacity=0.7))
fig.update_layout(showlegend=False)
fig.show()
```

<iframe src="/assets/charts/faces-pca.html" style="width:100%;height:500px;border:none;"></iframe>

A complete mess. Forty families, totally indistinguishable.

PCA found the biggest sources of variation across all 400 photos — and those turned out to be lighting conditions, head angles, and expressions. Those things vary far more across the dataset than identity does. So identity got buried. PCA wasn't wrong. It found what varies most. It just turns out that *what varies most* is not *what we care about*.

This is the fundamental limitation. PCA speaks only in global directions — straight lines through high-dimensional space. But the signal we want here is *local*: this face looks like that face, these ten photos belong together. No straight line can capture that.

It's like trying to find your relatives at a crowded wedding by height. You'd pick up a lot of tall strangers before you found your cousins.

*"Oy,"* said the statistician, staring at the tangle. *"We need the other guys."*

Those other guys — and what they can do — are waiting in [Part 2 →](/data-science/2026-06-27-DimReduction-part2-post/)

---

*All code for this post is available in the [Jewpyter notebook repository](https://github.com/wolfenfeld/jewpyter/blob/master/notebooks/DimensionalityReduction.ipynb).*

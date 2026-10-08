---
layout: post
title: "Every Row Gets an Address"
description: |
  CatBoost leaf indexes: turn a tree ensemble into a sparse matrix a linear model can use, and why to fit it on rows the trees never saw
noindex: true
---

Aunt Rivka does not remember people by their names. She remembers them by where they sat. The cousin who always argued about the brisket was *table four, left side.* The one who owed her money was *table nine, near the door.* Ask about anyone and she gives you an address, and from the address she knows the rest.

A trained CatBoost model does something very similar with every row you hand it, and it will tell you the address if you ask. It is not obvious why you would want to, and the method that does it has a dull name: `calc_leaf_indexes`.

---

# What a Row's Address Looks Like

Last time, in [the how-it-works post](/machine-learning/2026-10-01-catboost-post/), we saw that CatBoost's trees are symmetric: depth 6 means 64 leaves, and every row ends up in exactly one of them. A model with 100 trees therefore gives each row 100 addresses, one per tree.

```python
from catboost import CatBoostClassifier, Pool

model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0)
model.fit(X_train, y_train)

leaves = model.calc_leaf_indexes(Pool(X_train))
leaves.shape    # (12000, 100)
leaves.dtype    # uint32
leaves.max()    # 63
```

Twelve thousand rows, a hundred trees, and each entry a number from 0 to 63. Nothing here is a prediction. It is just where each row landed. The method also takes `ntree_start` and `ntree_end` if you want the addresses from only part of the ensemble.

---

# Making It Sparse

That array is a poor input for most models, because "leaf 37" is not bigger than "leaf 12". It is a label. So give every leaf of every tree its own column and put a 1 where the row landed:

```python
import numpy as np
import scipy.sparse as sp

def one_hot(leaves, depth=6):
    n, t = leaves.shape
    n_leaves = 2 ** depth
    cols = (leaves.astype(np.int64) + np.arange(t) * n_leaves).ravel()
    rows = np.repeat(np.arange(n), t)
    return sp.csr_matrix((np.ones(n * t), (rows, cols)), shape=(n, t * n_leaves))

Z_train = one_hot(leaves)
Z_train.shape     # (12000, 6400)
```

The docs do not call this anything. I'd call it the sparse leaf matrix, and the sparsity is not an accident of this dataset, it is guaranteed. Every row has exactly one 1 per tree, so 100 ones in a row of 6,400 columns, a density of 1.56%, which is 1 divided by 64 whatever the data. Here is what it looks like, forty rows against the first six trees:

<iframe src="/assets/charts/catboost-leaf-matrix.html" width="100%" height="380" frameborder="0" scrolling="no"></iframe>

Six bands, one bright cell per row per band. Not every leaf gets used, either: in this run only 5,450 of the 6,400 leaf columns were ever hit by a training row. The others are leaves in the tree that no data fell into.

---

# What a Linear Model Does With It

Here is the puzzle I used. A synthetic classification problem built to be nonlinear, with four clusters per class. A logistic regression on the raw features gets an AUC of 0.68, which is bad, as it should be.

Logistic regression on the leaf matrix, with a regularization strength of 0.1:

```python
from sklearn.linear_model import LogisticRegression

Z_test = one_hot(model.calc_leaf_indexes(Pool(X_test)))
clf = LogisticRegression(C=0.1, max_iter=3000).fit(Z_train, y_train)
roc_auc_score(y_test, clf.predict_proba(Z_test)[:, 1])    # 0.950
```

<iframe src="/assets/charts/catboost-leaf-auc.html" width="100%" height="450" frameborder="0" scrolling="no"></iframe>

0.68 to 0.95. The linear model is still linear. The trees did the bending, and each leaf column carries one small, learned rule: "rows that look like *this*." The weights just say how much each rule is worth. CatBoost itself scored 0.941 on the same test set.

That 0.950 is slightly flattering, and I want to be straight about it. The trees were trained on those 12,000 rows, and then I fit the linear model on the same rows. The leaves already know things about them. The cleaner way is to give the linear model rows the trees never saw. I split off a separate set and tried that, and the AUC came out at 0.935, a little *below* CatBoost's own 0.941. So the honest summary is that a linear model on leaf indexes recovers essentially everything the ensemble knows, but it does not beat it.

---

# So What Is It For

If you only care about accuracy, you can stop here, because the model you already have is as good. The use is when you need something other than the ensemble.

The matrix is sparse, so you can sparsify it further. With an L1 penalty (`C=0.3` in my run) the linear model kept 1,123 of the 6,400 leaf columns and reached an AUC of 0.946. That is a model with a short list of nonzero rules instead of a forest, which you can read, ship into a system that cannot run CatBoost, or hand to someone who asks which rules matter.

You can also feed the matrix into anything built for sparse input: a linear model that updates online as new rows arrive, a factorization machine, a hashing pipeline. The trees become a feature extractor, and the downstream part is up to you.

One caution about the whole idea. Which rows to fit the second stage on is not a detail. Fit it on the training rows and you will overestimate how well it works, as I almost did.

---

*"So the trees don't answer the question,"* said Devorah. *"They tell you where to sit."*

Mathityahu said that Rivka would have put it better.

---
layout: post
title: "Don't Peek at the Answer"
description: |
  How CatBoost works: ordered target statistics, ordered boosting, and why a row should never see its own label
noindex: true
---

Uncle Shimon does the books for half the family, and for every wedding he predicts who will RSVP yes. His method is simple. He looks at each family's track record. The Goldbergs said yes 80% of the time, so a Goldberg gets an 80% chance.

Then cousin Devorah looked over his spreadsheet and asked one question.

*"Shimon, when you count the Goldbergs' track record, does it include the RSVP you're trying to predict?"*

It did. It always had. For the Levys, who had sent exactly one guest in ten years, the "track record" was the answer itself. Shimon's model was brilliant on every wedding that had already happened and no better than a coin on the next one.

Mathityahu calls this the oldest mistake in the book. CatBoost is, more or less, a gradient boosting library built by people who took that mistake personally.

---

# The Problem With Categories

A tree needs numbers, and a column like `customer_id` or `zip_code` or `which_shul` is not a number. The standard options are not great. One-hot encoding a column with 1,500 distinct values gives you 1,500 new columns, most of them nearly empty. Label encoding invents an order that isn't there.

The tempting alternative is Shimon's: replace each category with the average of the target inside it. It is compact, it is informative, and it leaks.

To see how bad, here is the cleanest experiment I could think of. One categorical column, 1,500 levels, 6,000 rows, and the labels are coin flips. The column carries no information. At all. Whatever a model learns from it is, by construction, a hallucination.

```python
def naive_ts(cat_tr, y_tr, cat_te, a=0.5):
    s = np.bincount(cat_tr, weights=y_tr, minlength=N_CATS)   # sum of y per category
    c = np.bincount(cat_tr, minlength=N_CATS)                 # count per category
    f = lambda cat: (s[cat] + a * y_tr.mean()) / (c[cat] + a)
    return f(cat_tr), f(cat_te)
```

Feed that encoding to a boosted model and score it on the rows it trained on and on fresh rows:

<iframe src="/assets/charts/catboost-target-leakage.html" width="100%" height="450" frameborder="0" scrolling="no"></iframe>

The naive encoding reaches a train AUC of 0.78 on a column that is pure noise. On new data it drops to 0.50, which is what noise deserves. Each row's own label is sitting inside the number that describes it, and a flexible model finds it immediately.

---

# Ordered Target Statistics

CatBoost's fix is almost insultingly simple. Shuffle the rows into a random order. To encode row *k*, use only the rows that come before it in that order.

<script type="math/tex; mode=display">\hat{x}_k = \frac{\sum_{j<k} \mathbb{1}[x_j = x_k]\, y_j \;+\; a\,p}{\sum_{j<k} \mathbb{1}[x_j = x_k] \;+\; a}</script>

Here *p* is the overall average of the target and *a* is a small weight on that prior, so a category with no history starts at the global average instead of dividing by zero. A row never contributes to its own encoding. The first Levy gets the prior. The second Levy gets whatever the first one said.

```python
def ordered_ts(cat_tr, y_tr, cat_te, seed=0, a=0.5):
    perm = np.random.RandomState(seed).permutation(len(y_tr))
    p = y_tr.mean()
    s, c, out = np.zeros(N_CATS), np.zeros(N_CATS), np.zeros(len(y_tr))
    for i in perm:
        out[i] = (s[cat_tr[i]] + a * p) / (c[cat_tr[i]] + a)
        s[cat_tr[i]] += y_tr[i]
        c[cat_tr[i]] += 1
    return out, (s[cat_te] + a * p) / (c[cat_te] + a)
```

Same noise column, same model. The hand-rolled version above falls to a train AUC of 0.53, and the test AUC stays at 0.50. CatBoost itself, given the raw column through `cat_features`, lands at 0.50 on train and 0.50 on test. It looked at a column of nothing and reported nothing.

That is the whole feature. Most of the time you will not write the loop yourself.

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(iterations=200, depth=4, verbose=0)
model.fit(df[["customer_id"]], y, cat_features=["customer_id"])
```

Low-cardinality columns are treated differently, by the way. Anything with at most `one_hot_max_size` distinct values gets plain one-hot encoding, and the target statistics are skipped.

---

# The Same Sin, One Level Up

Shimon's mistake has a second form, and it is harder to see.

Gradient boosting builds trees one at a time, and each tree is fit to the *gradients* of the loss, the errors the model is currently making. But those errors are measured on the same rows the model was trained on. The residual of a row the model has already seen is smaller, and more flattering, than the residual of a row it hasn't. The authors call this prediction shift: the gradients you train on are drawn from a different distribution than the ones you will meet at test time.

The fix has the same shape as the last one. In **ordered boosting**, a row's gradient is computed by a model that was trained only on the rows before it in the permutation. No row is ever judged by a model that has seen it.

Done literally, that would mean training a separate model for every prefix of the data, which nobody can afford. The paper describes how CatBoost approximates it with a small set of supporting models, and you opt in with one parameter:

```python
model = CatBoostClassifier(boosting_type="Ordered")
```

The docs say it usually gives better quality on small datasets and can be slower than the classic `Plain` scheme, which is the one the CPU version uses by default. With a million rows you will probably not notice the difference. With three thousand you might.

---

# Why the Trees Look Strange

One more thing, because the next two posts depend on it.

An ordinary decision tree can ask a different question in every branch. CatBoost's default trees don't. In a symmetric, or *oblivious*, tree, every node on the same level asks the same question. Depth 6 means six questions in total, and every row answers all six.

So there are exactly 2<sup>6</sup> = 64 leaves, and a leaf is nothing more than a six-bit number: yes-no-no-yes-yes-no. Predicting is six comparisons and a lookup in a table of 64 values. No branching, no pointer chasing. It is fast, and it is quietly strict: a tree this rigid cannot contort itself around a handful of rows, which is a kind of regularisation you did not have to ask for.

It also means that every row, in every tree, lands in exactly one leaf. Hold that thought.

---

*"So the whole library,"* said Devorah, *"is Shimon not counting the wedding he's predicting."*

Mathityahu thought about it for a moment.

*"Twice,"* he said. *"Once for the categories, once for the trees."*

Next: [the training rows that hurt your model, and how to find them](/data-science/2026-10-05-catboost-object-importance-post/).

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

# One Tree

Before anyone can peek at anything, a quick pass through the machinery. (There is a longer version, with random forests and AdaBoost for comparison, in [the boosting post](/machine-learning/2026-09-28-boosting-post/).)

Say Devorah wants to predict how many challahs a Shabbat table will finish, given the number of guests. A **decision tree** answers with a short list of yes-or-no questions. More than 9 guests? Then more than 5? Each answer sends you down a branch, and at the bottom, in a **leaf**, sits a number: the average challah count of all the past dinners that ended up there.

The tree picks its questions greedily. It tries every possible threshold, keeps the one that makes the leaves most uniform, and repeats on each side until it runs out of depth. Nothing needs scaling, nothing needs a linear relationship, and a column of weird units is no problem. That is why trees are so popular on messy tabular data.

One tree is also fragile. Too shallow and it misses the pattern, too deep and it memorises the dinners it has seen.

---

# Many Small Trees

**Gradient boosting** takes a different route. Instead of one good tree, build many weak ones, each correcting the last.

Start with the dullest prediction possible: the average, for every table. It will be wrong everywhere, and *how* wrong, row by row, is information. Those errors are the residuals. Now fit a small tree to predict the residuals, not the challahs, and add a fraction of its answer to the prediction. Recompute the residuals, which are smaller now. Fit another tree to those. Repeat.

I wrote it out by hand on 60 invented dinners, where appetite climbs with the guest count and then flattens out:

```python
from sklearn.tree import DecisionTreeRegressor

pred = np.full(len(y), y.mean())           # round 0: everyone gets the average
learning_rate = 0.3

for step in range(50):
    residual = y - pred                    # what we still get wrong
    tree = DecisionTreeRegressor(max_depth=2).fit(X, residual)
    pred += learning_rate * tree.predict(X)
```

The mean squared error after each round:

| round | 0 | 1 | 3 | 5 | 10 | 50 |
|---|---|---|---|---|---|---|
| MSE | 11.76 | 6.25 | 2.04 | 0.86 | 0.39 | 0.09 |

The first tree, with a depth of just 2, already cuts the error roughly in half. Notice the last column too. I added noise with a variance of 0.64, so an error of 0.09 means the model has stopped learning the pattern and started memorising the noise. Boosting will do that if you let it run. Keep that in mind.

Why "gradient"? Because the residual was a special case.

With squared error, the direction that reduces the loss fastest, the negative gradient, is exactly the residual: what the truth says, minus what the model says. So "fit a tree to the residuals" was gradient descent all along. It only looked like common sense.

Change the question and the residual stops being the right thing. Suppose Devorah asks a yes-or-no question instead: will this client come back? The model now keeps a raw score, and the probability is that score squeezed through a sigmoid. It has to be: the score is a running sum of tree outputs and can reach 7 or minus 12, while a probability must stay between 0 and 1, and the sigmoid maps any number into that range. The raw score is also the log-odds, so each tree is saying "shift the odds by this much". The negative gradient of the logloss turns out to be the label minus the predicted *probability*. For a client who did come back, that looks like this:

| what the model believes | predicted probability | target for the next tree |
|---|---|---|
| no idea | 0.50 | 0.50 |
| sure they won't come back | 0.12 | 0.88 |
| sure they will | 0.95 | 0.05 |

A row the model has badly wrong gets a large push, and a row it already has right is left nearly alone. Same instinct as before, with the loss doing the arithmetic.

The picture to hold on to is ordinary gradient descent, except that what you adjust is the prediction function itself, one small tree at a time. The **learning rate** is the step size. Each tree only points roughly downhill, so you take a fraction of its answer, because a full step overshoots and starts fitting noise. And because only the line that computes the target changes, you can swap the loss without touching anything else.

(CatBoost goes a little further than the gradient when it sets the values in the leaves. For classification, the docs say the default is Newton steps, which use the curvature of the loss as well. For plain regression it is a gradient step.)

This recipe is what XGBoost and LightGBM do, and CatBoost does it too. A random forest does not: each tree there is a complete model trained on its own, and the forest simply averages their answers, so there is no running sum, no loss to descend and nothing to squash. So what is CatBoost for?

Look at the recipe again. There are two places where a row's own label can slip into something used to predict it. One is the step where you turn a column of categories into numbers, before any tree is built. The other is the residuals, which are computed by a model that has already trained on those very rows. Uncle Shimon's mistake lives in both. CatBoost is built around closing both.

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

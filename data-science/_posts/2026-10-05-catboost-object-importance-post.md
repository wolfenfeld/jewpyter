---
layout: post
title: "Fire the Guilty Rows"
description: |
  CatBoost object importance: find the training rows that hurt your model, and drop them
noindex: true
---

Everyone knows which features matter. You plot the importances, you argue about the top three, you move on.

Nobody asks which *rows* matter, which is odd, because a row is a person. Somebody typed it in or logged it or labeled it, and some of them were having a bad Tuesday.

Mathityahu's cousin Devorah runs a small kosher-catering operation and keeps a sheet of past orders: did the client come back, yes or no. Over the years the sheet had been filled in by a nephew, two bookkeepers, and Devorah herself after a long wedding. She wanted to predict repeat business. Her model was fine, not great.

*"The model is only as good as the sheet,"* she said. *"Which rows do I not trust?"*

CatBoost can answer that. It is one method call, and it is more useful than it sounds.

---

# What the Method Does

`get_object_importance` implements an idea from a 2018 paper on finding influential training samples in gradient boosted trees. Given a trained model, a set of *validation* rows you trust, and the training rows, it estimates how much each training row moved the loss on that validation set. It does this without retraining the model once per row, which would take forever.

```python
idx, scores = model.get_object_importance(
    Pool(X_val, y_val),          # rows you trust
    Pool(X_train, y_train),      # rows you're suspicious of
    top_size=len(y_train),       # return all of them
    type="Average",              # one score per training row
)
```

You get back training-row indices and a score for each. The sign is where people get confused, because the docs say a positive score means the optimized metric *increases*, and whether that is good depends on the metric. For Logloss, going up is bad. In my runs the rows with large positive scores were the damaging ones, and the list came back sorted by absolute score, biggest effect first.

Two other things the validation set is doing quietly. First, the labels in it have to be right. The method asks "which training rows hurt on *these* rows", so if the reference is wrong the whole answer is. Second, the supported losses are a short list (Logloss, CrossEntropy, RMSE, MAE, Quantile, Expectile, LogLinQuantile, MAPE and Poisson), so you cannot use it on everything.

---

# A Test With a Known Answer

To find out whether it works, I needed to know which rows were bad, so I made them bad on purpose. I generated a binary classification problem, flipped the label on a random 20% of the training rows, and kept the validation and test labels clean. Then I trained CatBoost on the dirty set and asked for object importances.

```python
model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08, verbose=0)
model.fit(Pool(X_train, y_noisy))

idx, scores = model.get_object_importance(
    Pool(X_val, y_val), Pool(X_train, y_noisy),
    top_size=len(y_noisy), type="Average")
idx, scores = np.array(idx), np.array(scores)

harmful = idx[scores > 0]     # sorted by |score|, so these are in order of damage
```

About 2,700 of the 8,000 training rows had a positive score, which is far more than the 20% I had actually corrupted. Positive does not mean flipped. It means *this row nudged the validation loss up*, and plenty of perfectly honest rows do that a little. The ones at the top of the list are another story.

I took the top *k* of the `harmful` list and checked how many of them were really flipped:

<iframe src="/assets/charts/catboost-object-importance.html" width="100%" height="450" frameborder="0" scrolling="no"></iframe>

On the right: a random row has a 20% chance of being flipped. The top 200 rows by importance were flipped 97% of the time, the top 600 about 94%, and by 1,200 it was still 82%. That is a flagging tool, not a coin toss.

On the left is what you get for acting on it. I removed the top *k* harmful rows, retrained, and measured test logloss, then did the same with *k* rows removed at random. Random removal does nothing, as it should: 0.384 stays between 0.385 and 0.389. Removing the flagged rows takes the logloss from 0.384 down to 0.277 at 800 rows and 0.267 at 1,200. Training on the true labels would have given 0.186, so you do not get all of the way back. You get most of the way.

---

# What It Doesn't Fix

AUC barely moved. It went from about 0.958 to about 0.960 with the flagged rows dropped, against 0.952 to 0.957 when I dropped random ones, a small gain. The ranking was already mostly right. What the noisy labels had ruined was the *confidence*, the model saying 0.7 where it should have said 0.95. If you care about calibrated probabilities, the effect is large. If you only rank, it is small.

And the cleanup has a ceiling. The higher *k* goes, the more honest rows you throw out along with the flipped ones, which is why the precision curve falls. Past a certain point you are paying for a cleaner set with rows that were fine.

---

# How I'd Use It

I would not pipe the output straight into a `drop()`. I'd take the top few hundred, look at them, and ask what they have in common. Rows from the same source, rows entered in the same week, rows with a label that contradicts a near-identical neighbor. A flipped label is the easy case. Often the finding is "the nephew did this one," and the right fix is upstream.

Speed was fine. With 8,000 training rows and 4,000 validation rows, the default `SinglePoint` update took around 13 seconds on a laptop. The docs describe it as the fastest and least accurate of the three methods, with `TopKLeaves` and `AllPoints` trading time for accuracy, and I stayed with the default. I did not test the others.

---

Devorah ran it on her catering sheet. The top thirty rows were almost all from one month.

*"That's when the nephew was helping,"* she said.

She did not drop them. She went to check.

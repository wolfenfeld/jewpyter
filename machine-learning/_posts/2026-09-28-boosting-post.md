---
layout: post
title: "The Committee and the Relay"
description: |
  Decision trees, random forests, AdaBoost and gradient boosting: forests average independent trees, boosting chains them so each fixes the last one's mistakes
noindex: true
---

When the family needs an answer, there are two ways to get it.

You can ask Aunt Rivka, who has opinions on everything and is right about two thirds of the time. Or you can ask the whole family, separately, without letting them talk to each other, and go with the majority.

Or there is a third way, which my cousin Devorah calls the relay. You ask one relative for a first guess. You write down what they got wrong. You hand that list to the next relative and ask only about the mistakes. Then the next, with what is still wrong. Nobody has to be good. Everyone has to be a little less wrong than the last person.

All three are decision trees, and the second and third are what people mean when they say "tree-based models." They look similar on a diagram and behave very differently.

---

# One Aunt

A **decision tree** asks yes-or-no questions about a row, such as more than 9 guests, more than 5, and lands in a leaf holding an answer. For a classification problem the leaf holds the share of past examples there that were positive. The tree picks each question greedily: it tries every threshold on every column and keeps the one that makes the two sides most uniform.

The one thing to decide is how deep to let it go. I made a noisy classification problem to find out: 12,000 rows, 20 columns, and 5% of the labels flipped at random. Half the rows are for training, a quarter for validation later on, and a quarter for testing. Three trees, trained on the 6,000 training rows and scored on the 3,000 test rows:

| depth | accuracy on rows it trained on | accuracy on new rows |
|---|---|---|
| 2 | 74.8% | 73.8% |
| 4 | 79.0% | 78.9% |
| until it runs out of questions | 100% | 80.6% |

The shallow tree is wrong on the training rows and wrong on new rows by about the same amount. It cannot express the pattern. Statisticians call that **bias**. The deep tree gets every training row right, including the mislabelled ones, and then does only a little better than the depth-4 tree on rows it has never seen. It has memorised the noise, and a different 6,000 training rows would have given a very different tree. That sensitivity is **variance**.

Everything that follows is a way of getting out of this trade.

---

# The Whole Family, Separately

A **random forest** keeps the deep trees, which have low bias and high variance, and fights the variance with numbers. Grow hundreds of them. Give each tree a random sample of the rows (drawn with replacement) and let each split consider only a random subset of the columns. Then average what they say.

Each tree makes its own mistakes, because each saw different rows and different columns. Averaging washes out mistakes that do not agree with each other. It cannot fix a mistake that every tree shares, but it does remarkably well on the rest.

Two things follow from the trees being independent. They can be trained in parallel. And adding more trees never makes the forest worse, it only makes the average steadier. There is no dial to over-turn.

---

# The Relay

**Boosting** goes the other way. It starts from weak models, here very shallow trees, which have high bias but low variance, and chains them so each one attacks what is still wrong.

The oldest version is **AdaBoost**. Train a stump (a tree with a single question). Find the rows it got wrong and raise their weights. Train the next stump on the reweighted rows, so it works hardest on the previous one's failures. After a few hundred rounds, let all the stumps vote, with the more accurate ones getting a larger say.

**Gradient boosting** keeps the relay and changes what gets handed over. Instead of reweighted rows, it hands over the *gradient of the loss* at the current predictions, which for squared error is just the residual. Each new tree is fit to that and a small fraction of its answer is added to the running sum:

```python
pred = np.full(len(y), y.mean())            # round 0: the average, for everyone
for step in range(50):
    residual = y - pred                      # what is still wrong
    tree = DecisionTreeRegressor(max_depth=2).fit(X, residual)
    pred += 0.3 * tree.predict(X)            # a fraction of the correction
```

Here it is on 60 invented Shabbat dinners, guests against challahs eaten, with appetite that climbs and then flattens:

<iframe src="/assets/charts/boosting-progression.html" width="100%" height="450" frameborder="0" scrolling="no"></iframe>

The first tree draws a crude staircase. By ten trees the curve has the shape of the data. By fifty it is tracing individual dinners, and its error on those 60 points is 0.09, well below the 0.64 of noise I put in. That is the memorising again. Each tree is deliberately fitted to whatever error remains, noise included, so a boosted model keeps driving its training error down for as long as you let it run. Adding trees to a forest does not work that way.

For classification, the model keeps a running score and squeezes it through a sigmoid to get a probability. [The next post](/machine-learning/2026-10-01-catboost-post/) goes through that, and the loss behind it, in detail.

---

# Side by Side

| | Random forest | AdaBoost | Gradient boosting |
|---|---|---|---|
| Trees are built | in parallel, independently | in sequence | in sequence |
| Each tree is | deep, low bias | a stump | shallow, a few levels |
| What the next tree sees | nothing from the others | reweighted rows | the gradient of the loss |
| Combined by | averaging | weighted vote | adding, with a small learning rate |
| Mainly reduces | variance | bias | bias |
| Can more trees hurt? | no | can | can |

---

# What Happens in Practice

I put four ensembles on that same problem and tracked the test error as trees were added.

<iframe src="/assets/charts/boosting-vs-forest.html" width="100%" height="450" frameborder="0" scrolling="no"></iframe>

**The forest** is at 84% accuracy by ten trees and sits at about 86% from fifty on. After that it barely moves. Its logloss ends at 0.36 and does not rise.

**AdaBoost with stumps** reached only 78.7% after 600 rounds. A single question per tree is too little for this data, however many of them you stack.

**Gradient boosting with depth-3 trees** and a learning rate of 0.1 got to 85.3%, a little behind the forest.

**Gradient boosting with depth-8 trees** and a learning rate of 0.05 finished on top: 87.3% at the end, and 87.6% if you stop at the 203 trees a validation set picks. The right-hand panel shows the price of letting it run, though. Its logloss reaches 0.305 after 141 trees and climbs back to 0.364 by tree 600, a little worse than the forest's. Accuracy holds roughly steady while the model gets worse at knowing how sure it should be, because it has started spending trees on noise.

Before you read too much into that, a warning about my own chart. I ran the tuned boosted model against the forest on five different random versions of this problem, with the number of trees chosen on the validation rows each time. The one in the chart, the first, is the one where boosting did best. Boosting was ahead in only one of the five. Averaged over all five, the forest scored 88.0% and tuned boosting 88.1%, which is a tie.

So the honest summary is not "boosting wins." The forest needed no tuning and landed in the same place as a boosted model that did. The untuned boosted model came out behind it. If you are not going to tune anything, a forest is a very respectable first model, and the argument for boosting is mostly that it gives you more dials to turn.

---

# XGBoost, LightGBM, CatBoost

These three libraries are all gradient boosting, and most of what separates them is engineering. XGBoost grows its trees level by level by default. LightGBM grows them leaf by leaf, always splitting the leaf that helps the most, which gets to a good fit with fewer splits and can overfit small data if you let it. CatBoost builds symmetric trees, where every node on a level asks the same question, and it treats categorical columns and the gradient computation differently from the other two, for a reason that is the subject of the next post.

---

*"So the forest is the committee,"* said Devorah, *"and the boosting is the relay."*

*"And the committee can't hurt itself,"* said Mathityahu. *"The relay can run too far."*

Next: [how CatBoost keeps a row from peeking at its own answer](/machine-learning/2026-10-01-catboost-post/).

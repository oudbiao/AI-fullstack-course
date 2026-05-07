---
title: "E.C.2 K-Nearest Neighbors"
sidebar_position: 13
description: "Start from the idea of 'letting neighbors vote' to understand why KNN is valuable for small-data, low-training-cost scenarios, and why it is strongly affected by feature scale."
keywords: [KNN, k-nearest neighbors, distance metric, lazy learning, classification]
---

# E.C.2 K-Nearest Neighbors

![KNN neighbor voting diagram](/img/course/elective-knn-neighbor-voting-en.png)

:::tip Section overview
KNN is a type of algorithm that is especially good for building intuition.
It has almost no complicated training process, and its core idea is very direct:

> **Look at the few neighbors most similar to a new sample, then make a judgment based on those neighbors.**

Because it is so direct, KNN is very useful for helping you understand:

- distance metrics
- feature scaling
- local decision-making

These are all very important concepts in classic machine learning.
:::

## Learning objectives

- Understand the basic working mechanism of KNN
- Understand why the K value and distance metric can significantly affect results
- Understand why feature scaling is especially important for KNN
- Master the minimal way to use KNN through a runnable example

---

## What exactly does KNN do?

### It almost does not “train”

KNN is often called a:

- lazy learner

because unlike many models, it does not learn a set of explicit parameters during training.
It is more like:

- remembering the samples first
- finding the nearest neighbors when making predictions

### Why is it so easy to understand?

Because the logic is really very similar to human intuition:

- Which people is this new customer most like?
- Which old pictures is this new image most like?

If most of the nearest samples belong to one class,
the new sample is also likely to belong to that class.

### An analogy

KNN is like asking:

- “What occupations do most of your nearby neighbors have?”

If 4 out of the 5 most similar people around you are engineers,
you would tend to judge that this person is also more like an engineer.

---

## Why is the K value so important?

### If K is too small

If K=1, the model becomes very sensitive:

- it can be easily affected by noisy points

### If K is too large

If K is very large,
the model becomes too averaged:

- local differences get smoothed out

### So what is K really controlling?

You can roughly understand it as:

- small K: focuses more on local details
- large K: focuses more on the overall average

That is why KNN, although simple, still depends a lot on tuning.

---

## Let’s first run a real example that shows “neighbor voting”

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X = np.array([
    [1, 1],
    [2, 2],
    [2, 1],
    [8, 8],
    [9, 9],
    [8, 9],
])
y = np.array([0, 0, 0, 1, 1, 1])

clf = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3),
)

clf.fit(X, y)
pred = clf.predict([[3, 3], [8.5, 8.2]])
print(pred.tolist())
```

### What should you focus on in this code?

It highlights the two most basic points of KNN:

1. `n_neighbors=3`
   means “look at the 3 nearest neighbors”
2. `StandardScaler()`
   means feature scaling is applied before distance calculation

### Why does KNN also need scaling?

Because its core is distance.
If one feature has a much larger numeric range than another, that feature will dominate the distance.

For example:

- age is in the tens
- income is in the hundreds of thousands

Without scaling, the income feature will almost “swallow” the age feature.

---

## What are the advantages and costs of KNN?

### Advantages

- intuitive idea
- very light training phase
- a good baseline for small datasets

### Costs

- prediction can be slower
- query cost is high when the dataset is large
- very sensitive to feature scaling and the distance definition

### An engineering judgment

KNN is a good choice when:

- the sample size is not large
- the features are not too complex
- you want to quickly build an interpretable baseline

It is not ideal when:

- the dataset is huge
- the inference needs to be highly real-time

---

## Why is the distance metric worth paying attention to?

### Euclidean distance is only the most common one

In many cases, the default is:

- Euclidean distance

But different tasks may also consider:

- Manhattan distance
- Cosine distance

### Why does the distance definition change model behavior?

Because all KNN decisions are based on:

- “who is closer”

If the definition of “close” changes, the neighbor set also changes.

---

## The most common pitfalls with KNN

### Mistake 1: Using KNN by default on large data

Neighbor search becomes increasingly expensive during prediction.

### Mistake 2: Forgetting to scale

This is just like SVM — also a very common pitfall.

### Mistake 3: Only tuning K and ignoring feature quality

If the features themselves are not discriminative,
no matter how well you tune K, it is still hard to save the model.

---

## Summary

The most important thing in this section is to view KNN as a “local voting” model:

> **It does not rely on learning complex parameters, but instead makes decisions through distance metrics and neighbor voting, which makes it especially useful for building intuition about distance, feature scaling, and local decision-making.**

Once you understand this layer clearly, KNN is no longer just an introductory algorithm — it becomes a very useful small tool when you need a baseline.

---

## Exercises

1. Change `n_neighbors` to `1` and `5`, and see whether the predictions change.
2. Remove `StandardScaler()` and try again, then observe the difference in results.
3. Think about this: why do we say KNN has “light training” but “prediction may not be light”?
4. In what kinds of projects would you try KNN first as a baseline?

---
title: "E.C.4 Linear Discriminant Analysis"
sidebar_position: 15
description: "Starting from the goal of “tighter within-class, farther between-class,” understand why LDA can be used for classification and is also often treated as a supervised dimensionality reduction method."
keywords: [LDA, linear discriminant analysis, dimensionality reduction, classification, classic ML]
---

# E.C.4 Linear Discriminant Analysis

:::tip Where this section fits
LDA is easy to confuse with other acronyms,
and it is also easy to misunderstand as “just another linear classifier.”

A more accurate understanding is:

> **LDA is concerned with how to find a projection direction that makes samples from the same class cluster together and samples from different classes separate further apart.**

So it can be used for classification, and it can also be viewed as a supervised dimensionality reduction method.
:::

![LDA supervised projection intuition diagram](/img/course/elective-lda-projection-map-en.png)

## Learning Objectives

- Understand LDA’s core goal: compact within-class structure and separated between-class structure
- Understand how it differs from a standard linear classifier
- Use a runnable example to understand LDA’s dimensionality reduction and classification effects
- Build a basic judgment for when to try LDA

---

## What problem is LDA solving?

### More than just “separating classes”

LDA’s goal is more specific:

- Samples within the same class should be as close together as possible
- Different classes should be as far apart as possible

### Why is this more interesting than ordinary linear splitting?

Because it is not only looking for a boundary,
it is also looking for a “more discriminative representation space.”

That means in addition to classification, it can also be used for:

- Supervised dimensionality reduction

### An analogy

If PCA is more like:

- Finding the direction that explains the overall variation best

Then LDA is more like:

- Finding the direction that is most helpful for distinguishing classes

---

## Why is LDA often treated as “label-aware dimensionality reduction”?

### Because it uses class labels

PCA does not care about classes; it only looks at overall variance.
LDA explicitly uses labels to ask:

- Which direction is most useful for classification?

### So what kinds of scenarios is it suitable for?

It is suitable when:

- You already have supervised labels
- You want a lower-dimensional representation with stronger discriminative power
- Or you want a lightweight linear classifier

---

## First run a minimal executable example

This example does two things at the same time:

1. Use LDA for classification
2. Project the data into a lower-dimensional space

```python
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [2.0, 1.5],
    [4.0, 5.0],
    [4.5, 4.8],
    [5.0, 4.5],
])
y = np.array([0, 0, 0, 1, 1, 1])

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X, y)

pred = lda.predict([[1.4, 1.9], [4.8, 4.6]])
projection = lda.transform(X)

print("predictions:", pred.tolist())
print("projection shape:", projection.shape)
print("projected values:", projection.ravel().round(3).tolist())
```

### Why is this code more valuable than just `predict`?

Because it lets you see both:

- The classification output
- The low-dimensional representation after projection

This nicely shows LDA’s dual value:

- It can classify
- It can also perform supervised dimensionality reduction

### Why `n_components=1`?

Because there are only two classes here.
In this case, LDA can project to at most:

- 1 discriminative direction

This is one feature of LDA that depends on the number of classes.

---

## What is the difference between LDA and SVM / Logistic Regression?

### Difference from SVM

SVM focuses more on:

- Maximizing the margin

LDA focuses more on:

- Small within-class variance
- Large differences between class means

### Difference from Logistic Regression

Logistic Regression is more like learning:

- Conditional probability boundaries

LDA is more like first assuming a data distribution, then finding a direction that separates classes better.

### Why is this worth learning?

Because it shows you:

- Classical models are not all based on the same idea of “linear classification”

---

## When is LDA worth trying?

### When the dataset is not very large and the class structure is fairly clear

LDA can be very useful in this kind of scenario.

### When you need a more interpretable low-dimensional representation

For example:

- Project first, then visualize
- Project first, then feed into a simple classifier

### When it is less suitable

If the class boundaries are very complex and clearly nonlinear,
LDA will usually struggle.

---

## Common misconceptions

### Misconception 1: LDA is just another classifier

That is incomplete.
Its value as a “discriminative representation” is also important.

### Misconception 2: If labels exist, LDA is always better than PCA

Not necessarily.
It depends on the task goal and the data distribution.

### Misconception 3: LDA here is the same as LDA in topic models

It is not.
The LDA here stands for:

- Linear Discriminant Analysis

Not the topic-model LDA:

- Latent Dirichlet Allocation

---

## Summary

The most important thing to establish in this section is:

> **The core value of LDA is to use labels to find a more discriminative projection direction, so it can be used for lightweight classification and also for supervised dimensionality reduction.**

Once you understand this, it is no longer just an easily confused acronym.

---

## Exercises

1. Add a new class to the example data and see how `n_components` changes.
2. Think about why LDA is more like “label-aware dimensionality reduction.”
3. If the class boundaries are very curved and clearly nonlinear, would you still try LDA first? Why?
4. Explain in your own words: what is the biggest difference between LDA and PCA?

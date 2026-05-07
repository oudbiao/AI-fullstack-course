---
title: "6.8.2 Project: Image Classification System"
sidebar_position: 1
description: "Work through a complete delivery loop for a real, demo-friendly image classification project, from topic selection, data, and baseline to training, evaluation, and presentation."
keywords: [image classification project, CNN, confusion matrix, error analysis, computer vision]
---

# 6.8.2 Project: Image Classification System

:::tip Section overview
Image classification is a great first vision project, not because it is the easiest, but because it is the easiest way to explain the full engineering pipeline clearly:

- How to define the classes
- How to organize the data
- How to build a baseline
- How to read the metrics
- How to analyze mistakes

The goal of this section is not to build a model that “runs,” but to build a project that you can **explain clearly**.
:::

## Learning Objectives

- Learn how to define an image classification problem that works well for a portfolio
- Learn how to connect data, baseline, evaluation, and error analysis into a closed loop
- Learn how to express the project structure with a minimal runnable example
- Understand what is most worth showing in an image classification project

---

## Don’t rush to pick a model; first pick the right project topic

### A good practice topic usually has three characteristics

1. Clear class boundaries
   For example: `cat / dog / bird`, `apple leaf disease classification`, `garbage sorting`
2. Accessible data
   Don’t start with a topic for which you have no realistic way to collect samples
3. Explainable mistakes
   After a wrong prediction, you can describe possible reasons instead of ending with “the model is bad”

### A very stable project topic

For example:

> **Build a “pet photo classifier” that splits images into three classes: `cat / dog / rabbit`.**

Its advantages are:

- The classes are intuitive
- The data is relatively easy to collect
- It is very suitable for a confusion matrix and error sample analysis

### Topics not recommended for the beginning

For example:

- Hundreds of fine-grained classes
- Extremely ambiguous class boundaries
- Severe class imbalance before you are ready to handle it

---

## What does the minimum project loop look like?

A minimal but complete image classification project should usually include at least:

1. Problem and label definition
2. Dataset organization and splitting
3. Baseline
4. Training and validation
5. Evaluation and error analysis
6. Demo method

If you can explain all 6 parts clearly, the project will be convincing even if the model is not complicated.

![Image classification project closed loop](/img/course/ch06-project-image-classification-loop-en.png)

:::tip How to read this diagram
Read it from top to bottom: labels define the task boundary, data splitting protects evaluation, the CNN baseline gives you a runnable starting point, metrics show the overall result, and error cases tell you what to improve next.
:::

---

## First, look at a minimal project planning object

```python
from dataclasses import dataclass, field


@dataclass
class CVProjectPlan:
    name: str
    classes: list
    dataset_split: dict
    baseline: str
    metrics: list
    risks: list = field(default_factory=list)


plan = CVProjectPlan(
    name="pet_image_classifier",
    classes=["cat", "dog", "rabbit"],
    dataset_split={"train": 900, "val": 180, "test": 180},
    baseline="small_cnn",
    metrics=["accuracy", "confusion_matrix", "error_cases"],
    risks=["class imbalance", "background leakage", "label noise"],
)

print(plan)
```

### Why is this object important?

Because at the start of a project, what you lack most often is not code, but boundaries.
This minimal object forces you to explain first:

- What you are building
- What classes you have
- What baseline you will use
- What metrics you will use to judge success

---

## Start by understanding project evaluation with a “pseudo-feature” baseline

To avoid adding extra dependencies, we’ll use a tiny toy baseline to simulate the validation flow of an image classification project.

Here, we assume each image already has three very rough statistical features:

- `fur`
- `ear_shape`
- `size`

Of course, a real project would not do this, but it is very helpful for understanding:

- the training set
- class prototypes
- prediction
- the confusion matrix

This chain.

```python
train_data = [
    ("cat", [0.9, 0.8, 0.4]),
    ("cat", [0.8, 0.7, 0.5]),
    ("dog", [0.7, 0.5, 0.8]),
    ("dog", [0.6, 0.4, 0.9]),
    ("rabbit", [0.5, 0.9, 0.3]),
    ("rabbit", [0.4, 0.8, 0.2]),
]

test_data = [
    ("cat", [0.85, 0.75, 0.45]),
    ("dog", [0.65, 0.45, 0.85]),
    ("rabbit", [0.45, 0.85, 0.25]),
    ("dog", [0.82, 0.72, 0.42]),  # Intentionally add an error sample that looks more like cat
]


def class_prototypes(data):
    grouped = {}
    for label, features in data:
        grouped.setdefault(label, []).append(features)

    prototypes = {}
    for label, rows in grouped.items():
        dim = len(rows[0])
        prototypes[label] = [
            sum(row[i] for row in rows) / len(rows)
            for i in range(dim)
        ]
    return prototypes


def l1_distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def predict(features, prototypes):
    distances = {label: l1_distance(features, proto) for label, proto in prototypes.items()}
    return min(distances, key=distances.get), distances


prototypes = class_prototypes(train_data)
print("prototypes:", prototypes)

results = []
for gold, features in test_data:
    pred, distances = predict(features, prototypes)
    results.append({"gold": gold, "pred": pred, "distances": distances})
    print(results[-1])
```

### Why does this example still have project value?

Because the most important thing in a project is not the library name, but the evaluation logic.
This toy baseline already lets you see:

- train -> prototype
- test -> predict
- gold vs pred

This is exactly the same line that a real CNN project must follow later.

---

## A minimal confusion matrix and error analysis

```python
labels = ["cat", "dog", "rabbit"]


def confusion_matrix(rows, labels):
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for row in rows:
        matrix[row["gold"]][row["pred"]] += 1
    return matrix


cm = confusion_matrix(results, labels)
print("confusion matrix:")
for gold in labels:
    print(gold, cm[gold])

error_cases = [row for row in results if row["gold"] != row["pred"]]
print("\nerror cases:", error_cases)
```

### Why is the confusion matrix especially important for image classification?

Because overall accuracy only tells you:

- how many were correct

But the confusion matrix tells you:

- which two classes are most often confused

That is exactly the information you need next when improving the data and the model.

### Why are error samples more valuable than the overall score?

Because you can actually inspect whether:

- the background misled the model
- one class has inconsistent photo angles
- some labels were incorrect

This is the most insightful part of an image project.

---

## Three layers you should add in a real project

### Data layer

You should at least explain:

- Roughly how many images each class has
- How train / val / test are split
- Whether there is class imbalance

### Model layer

It is highly recommended to start with two baselines:

1. A small CNN
2. A transfer learning model

Then you can clearly explain:

- What the more complex model actually buys you

### Presentation layer

When turning an image classification project into a portfolio piece, the most valuable things to show are usually:

- Label definition
- Confusion matrix
- Typical correct samples
- Typical error samples

Not just a screenshot saying “training completed.”

---

## The most common pitfalls in this project

### Looking only at overall accuracy

You will very easily miss the real distribution of problems.

### Defining classes too casually

If the class boundaries are fuzzy to begin with, both the model and the evaluation will be unstable.

### Data leakage

If similar images appear in both training and testing,
the results will be overestimated.

---

## Summary

The most important thing in this section is to build a project mindset:

> **The real value of an image classification project is not the model name, but whether you can turn the class boundaries, data organization, baseline, confusion matrix, and error analysis into a complete closed loop.**

Once that loop is in place, even a small project will feel like a portfolio-level course project.

---



## Suggested version roadmap

| Version | Goal | Delivery focus |
|---|---|---|
| Basic version | Run through the minimum loop | Can input, process, and output, while keeping one set of examples |
| Standard version | Become a presentable project | Add configuration, logging, error handling, README, and screenshots |
| Challenge version | Approach portfolio quality | Add evaluation, comparison experiments, failed sample analysis, and next-step directions |

It is recommended to finish the basic version first. Do not chase something large and complete from the beginning. Every time you upgrade the version, write into the README what new capability was added, how you verified it, and what problems still remain.

## Exercises

1. Add two more `dog` samples to the toy data and see how the confusion matrix changes.
2. If `cat` and `rabbit` are often confused, would you inspect the data, labels, or model first? Why?
3. Think about it: why is image classification especially suitable for showing a confusion matrix?
4. If you were turning this project into a portfolio page, which 4 sections would you place first?

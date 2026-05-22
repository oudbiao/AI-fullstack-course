---
title: "E.C Classic ML Roadmap"
description: "A concise hands-on roadmap for supplementary classic machine learning: SVM, KNN, Naive Bayes, and LDA as strong baselines for small and medium data."
sidebar:
  order: 0
---
Use this elective when your dataset is small, your features are clear, or you need a strong baseline before trying a heavier model.

## See the Baseline Map First

![Module map for supplementary classic ML algorithms](/img/course/elective-classic-ml-module-map-en.webp)

![KNN neighbor voting diagram](/img/course/elective-knn-neighbor-voting-en.webp)

Classic ML helps you answer: is the problem already solvable with simple features?

## Run the Smallest KNN Baseline

```python
def distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

train = [
    ([0.1, 0.2], "low"),
    ([0.2, 0.1], "low"),
    ([0.8, 0.9], "high"),
    ([0.9, 0.8], "high"),
]

point = [0.75, 0.85]
nearest = min(train, key=lambda row: distance(row[0], point))
print("prediction:", nearest[1])
print("neighbor:", nearest[0])
```

Expected output:

```text
prediction: high
neighbor: [0.8, 0.9]
```

This is the smallest baseline habit: define features, compare distance, predict, and keep the result for later comparison.

## Learn in This Order

| Step | Lesson | Practice Output |
|---|---|---|
| 1 | [E.C.1 SVM](./01-svm.md) | Explain margin, support vectors, `C`, and kernel choice |
| 2 | [E.C.2 KNN](./02-knn.md) | Build a distance-voting baseline |
| 3 | [E.C.3 Naive Bayes](./03-naive-bayes.md) | Convert evidence counts into class probabilities |
| 4 | [E.C.4 LDA](./04-lda.md) | Project features to separate classes |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
model_family: SVM, KNN, Naive Bayes, LDA, or another classical baseline
dataset_view: feature scale, class balance, decision boundary, and train/test split
metric: accuracy/F1, confusion matrix, margin, neighbor behavior, or projection quality
failure_check: scaling, high dimensionality, weak assumptions, leakage, or poor baseline fit
Expected_output: classical-ML baseline result with one limitation note
```

## Pass Check

You pass this module when you can build one classic baseline, explain why it is appropriate, and compare it with a heavier model or later project result.

<details>
<summary>Check reasoning and explanation</summary>

A solid baseline answer names the dataset shape, model family, and comparison point. For example: “KNN is acceptable here because the dataset is small and distances are meaningful; I will compare it against a later neural model using the same split and F1 score.”

The answer is incomplete if it only reports accuracy. Classic ML is most useful when it gives a fast, interpretable reference and a clear limitation, such as feature scaling, nonlinear boundaries, or high-dimensional sparsity.

</details>

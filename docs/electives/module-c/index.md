---
title: "E.C Classic ML Roadmap"
sidebar_position: 0
description: "A concise hands-on roadmap for supplementary classic machine learning: SVM, KNN, Naive Bayes, and LDA as strong baselines for small and medium data."
---

# E.C Classic ML Roadmap

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

## Pass Check

You pass this module when you can build one classic baseline, explain why it is appropriate, and compare it with a heavier model or later project result.

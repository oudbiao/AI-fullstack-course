---
title: "5.4.1 Evaluation Roadmap: Trust the Score Before Tuning"
sidebar_position: 9
description: "A compact model evaluation roadmap: metrics, cross-validation, bias-variance, hyperparameter tuning, and evidence."
keywords: [model evaluation guide, cross-validation, bias-variance, hyperparameter tuning]
---

# 5.4.1 Evaluation Roadmap: Trust the Score Before Tuning

Model evaluation answers: is the model actually good, or did the score only look good by accident?

## 5.4.1.1 Look at the Evaluation Map First

![Model Evaluation Learning Map](/img/course/ml-evaluation-roadmap-en.png)

![Chapter Flow for Model Evaluation](/img/course/ch05-evaluation-chapter-flow-en.png)

| Topic | First question |
|---|---|
| metrics | what score matches the task? |
| cross-validation | is the score stable across splits? |
| bias-variance | is the model too simple or too flexible? |
| tuning | which parameter change is actually better? |

## 5.4.1.2 Run One Cross-Validation Check

Create `evaluation_first_loop.py` and run it after installing `scikit-learn`.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier(max_depth=2, random_state=42)
scores = cross_val_score(model, X, y, cv=5)

print("fold_scores:", [float(round(score, 3)) for score in scores])
print("mean_accuracy:", round(scores.mean(), 3))
```

Expected output:

```text
fold_scores: [0.933, 0.967, 0.9, 0.867, 1.0]
mean_accuracy: 0.933
```

One score is a snapshot. Several folds tell you whether the result is stable enough to trust.

## 5.4.1.3 Learn in This Order

| Order | Read | What to practice |
|---|---|---|
| 1 | [5.4.2 Evaluation Metrics](./01-metrics.md) | accuracy, precision, recall, F1, R2, RMSE |
| 2 | [5.4.3 Cross-Validation](./02-cross-validation.md) | stable estimates, data split risk |
| 3 | [5.4.4 Bias and Variance](./03-bias-variance.md) | underfitting, overfitting, learning curves |
| 4 | [5.4.5 Hyperparameter Tuning](./04-hyperparameter-tuning.md) | grid search, comparison records |

## 5.4.1.4 Pass Check

You pass this roadmap when you can choose a metric for the task, explain one score stability check, and avoid tuning before the evaluation method is trustworthy.

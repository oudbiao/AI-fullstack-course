---
title: "5.2.1 Supervised Learning Roadmap: Learn From Labeled Examples"
sidebar_position: 2
description: "A compact supervised learning roadmap: regression, classification, trees, ensembles, SVM, and model choice."
keywords: [supervised learning guide, linear regression, logistic regression, decision tree, ensemble learning]
---

# 5.2.1 Supervised Learning Roadmap: Learn From Labeled Examples

Supervised learning answers one question: when examples already have labels, how do we learn a model that predicts labels for new examples?

## Look at the Model Choice Map First

![Supervised Learning Roadmap](/img/course/supervised-learning-roadmap-en.png)

![Supervised Learning Chapter Flow](/img/course/ch05-supervised-chapter-flow-en.png)

| Model family | First use |
|---|---|
| linear regression | predict a continuous number |
| logistic regression | classify with a simple probability model |
| decision tree | split data with readable rules |
| ensemble models | combine many models for stronger tabular baselines |
| SVM | learn a stable boundary with margin intuition |

## Run One Regression Baseline

Create `supervised_first_loop.py` and run it after installing `scikit-learn`.

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_test)

print("task: regression")
print("r2:", round(r2_score(y_test, predictions), 3))
print("first_prediction:", round(predictions[0], 1))
```

Expected output:

```text
task: regression
r2: 0.485
first_prediction: 137.9
```

The score is not perfect, and that is useful. A baseline tells you where later models or feature work must improve.

## Learn in This Order

| Order | Read | What to compare |
|---|---|---|
| 1 | [5.2.2 Linear Regression](./01-linear-regression.md) | simple numeric prediction |
| 2 | [5.2.3 Logistic Regression](./02-logistic-regression.md) | classification probability |
| 3 | [5.2.4 Decision Trees](./03-decision-trees.md) | rules, nonlinearity, overfitting |
| 4 | [5.2.5 Ensemble Learning](./04-ensemble-learning.md) | bagging, boosting, stronger tabular models |
| 5 | [5.2.6 Support Vector Machines](./05-svm.md) | margin, boundary, classic classifier intuition |

## Pass Check

You pass this roadmap when you can decide whether a labeled task is regression or classification, run one baseline, and explain one reason the model may fail.

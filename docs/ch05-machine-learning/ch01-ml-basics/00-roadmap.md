---
title: "5.1.1 Machine Learning Basics Roadmap: Task, Data, Model, Score"
sidebar_position: 0
description: "A compact machine learning basics roadmap: task types, data split, fit/predict/score, baseline, and sklearn workflow."
keywords: [machine learning guide, ML introduction, sklearn guide, supervised learning, unsupervised learning]
---

# 5.1.1 Machine Learning Basics Roadmap: Task, Data, Model, Score

Machine learning starts when you stop hand-writing every rule and let a model learn patterns from data. The first habit is not algorithm memorization. It is a small project loop.

## 5.1.1.1 Look at the Map First

![Machine Learning Basics Learning Map](/img/course/ml-basics-roadmap-en.png)

![Machine Learning Basics Chapter Flow](/img/course/ch05-basics-chapter-flow-en.png)

Keep this loop:

```text
define task -> split data -> fit model -> predict -> score -> decide next step
```

| Word | First meaning |
|---|---|
| feature | input column used by the model |
| label / target | answer the model learns to predict |
| train set | data used to learn |
| test set | data kept aside to check generalization |
| baseline | a simple first model used for comparison |

## 5.1.1.2 Run the Smallest sklearn Loop

Create `ml_first_loop.py` and run it after installing `scikit-learn`.

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("task: classification")
print("test_accuracy:", round(model.score(X_test, y_test), 3))
print("prediction_count:", len(predictions))
```

Expected output:

```text
task: classification
test_accuracy: 0.967
prediction_count: 30
```

This is the smallest useful machine learning loop: split first, train only on the training set, evaluate on the test set.

## 5.1.1.3 Learn in This Order

| Order | Read | What to practice |
|---|---|---|
| 1 | [5.1.2 What Is Machine Learning?](./01-what-is-ml.md) | task types, features, labels |
| 2 | [5.1.3 Scikit-learn Introduction](./02-sklearn-intro.md) | `fit`, `predict`, `score` |
| 3 | [5.1.4 How Math Flows Into ML](./03-math-to-ml-bridge.md) | vectors, probability, loss, optimization |
| 4 | [5.1.5 Machine Learning History](./04-history-breakthroughs.md) | why major algorithms appeared |
| 5 | [5.1.6 sklearn and Matplotlib Workshop](./05-sklearn-matplotlib-workshop.md) | run, plot, explain a baseline |

## 5.1.1.4 Pass Check

You pass this roadmap when you can name the task type, identify `X` and `y`, explain why train/test split matters, and keep one baseline score as evidence.

---
title: "5 Introduction to Machine Learning: From Basics to Practice"
sidebar_position: 0
description: "Learn the practical modeling loop: define a task, split data, train a baseline, evaluate, inspect errors, improve features, and write a report."
keywords: [machine learning, Scikit-learn, supervised learning, unsupervised learning, regression, classification, clustering]
---

# 5 Introduction to Machine Learning: From Basics to Practice

![Main visual for machine learning](/img/course/ch05-machine-learning-en.webp)

Chapter 5 has one job: help you turn a data problem into **a trainable, evaluable, improvable machine learning project**.

## See The Modeling Loop

![Main loop of machine learning modeling](/img/course/ch05-modeling-loop-backbone-en.webp)

Read the picture first. Most reliable ML work follows this loop:

```text
define task -> split data -> train baseline -> evaluate -> inspect errors -> improve
```

Start with a baseline before chasing model names. A baseline tells you whether later changes actually improve anything.

## Learning Order And Task List

Use this table as both the chapter guide and the task sheet.

| Page | Follow-along action | Evidence to keep |
|---|---|---|
| [5.1 ML Basics](ch01-ml-basics/00-roadmap.md) | Identify classification, regression, clustering, anomaly detection, features, labels, train/test split, and sklearn flow | A problem-definition note |
| [5.1.5 ML History](ch01-ml-basics/04-history-breakthroughs.md) | Optional background: skim how classic algorithms appeared | A short “why this algorithm exists” note |
| [5.2 Supervised Learning](ch02-supervised/00-roadmap.md) | Run regression and classification examples before comparing many models | One baseline score and one improved score |
| [5.3 Unsupervised Learning](ch03-unsupervised/00-roadmap.md) | Try clustering, dimensionality reduction, and anomaly detection when labels are missing | One chart or cluster interpretation |
| [5.4 Evaluation](ch04-evaluation/00-roadmap.md) | Choose metrics, use cross-validation, diagnose bias/variance, tune carefully | Metric choice and error samples |
| [5.5 Feature Engineering](ch05-feature-engineering/00-roadmap.md) | Handle missing values, categories, scaling, feature construction, feature selection, and Pipeline | Feature processing log and leakage check |
| [5.6 Projects](ch06-projects/00-roadmap.md) and [5.6.6 Workshop](ch06-projects/05-hands-on-ml-workshop.md) | Build a reproducible evidence pack before larger house-price, churn, segmentation, or Kaggle work | README, model comparison, errors, and next-step plan |

Key terms for this chapter:

| Term | Meaning |
|---|---|
| `feature` | Input column the model can use |
| `label` / `target` | Answer the model should learn to predict |
| `baseline` | Simplest model or rule you must beat |
| `metric` | Ruler for judging the model, such as F1, AUC, MAE, or RMSE |
| `leakage` | Test or target information accidentally entering training |
| `Pipeline` | Preprocessing and model packaged together to reduce leakage |

## First Runnable Loop

Install sklearn if needed:

```bash
python -m pip install scikit-learn
```

Then run this self-contained baseline. It uses a built-in dataset, splits data, trains a dummy baseline, trains a real model, and compares both.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
print("Baseline")
print(classification_report(y_test, baseline.predict(X_test), zero_division=0))

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print("Logistic regression")
print(classification_report(y_test, model.predict(X_test), zero_division=0))
```

Expected shape:

```text
Baseline
...
Logistic regression
...
```

Do not only compare the final scores. Ask: which classes are easy, which are hard, and what error would matter most in the real use case?

## Common Failures

| Symptom | First thing to check | Usual fix |
|---|---|---|
| Score is strangely high | Leakage or wrong train/test split | Inspect features and split before training |
| Train score high, test score low | Overfitting | Simplify the model, regularize, or add data |
| All models are weak | Poor labels, weak features, or wrong metric | Inspect error samples and label definition |
| Accuracy looks fine but product risk is high | Class imbalance or costly false negatives | Use recall, precision, F1, AUC, or threshold review |
| Results cannot be reproduced | Random seed, data version, or dependency changed | Fix seeds and record versions |

## Pass Check

Move to Chapter 6 when you can answer these five questions:

- Is this task classification, regression, clustering, or anomaly detection?
- What is the baseline, and what score must a real model beat?
- Which metric matches the goal, and when is accuracy misleading?
- How did you check for leakage?
- What does the model do well, what does it do poorly, and what would you improve next?

For a printable checklist, use [5.0 Study Guide and Task Sheet](./study-guide.md). The next chapter moves from sklearn models into neural networks and deep learning training.

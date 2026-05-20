---
title: "5 Introduction to Machine Learning: From Basics to Practice"
sidebar_position: 0
description: "Learn the practical modeling loop: define a task, split data, train a baseline, evaluate, inspect errors, improve features, and write a report."
keywords: [machine learning, Scikit-learn, supervised learning, unsupervised learning, regression, classification, clustering]
---

# 5 Introduction to Machine Learning: From Basics to Practice

![Main visual for machine learning](/img/course/ch05-machine-learning-en.webp)

Chapter 5 has one job: help you turn a data problem into **a trainable, evaluable, improvable machine learning project**.

## Where You Are In The Main Route

You have already learned how data becomes numbers and how loss and gradients explain model improvement. This chapter makes those ideas practical: define a prediction problem, build a baseline, choose a metric, inspect errors, and improve only when evidence says the change helped.

This is the bridge from math intuition to model engineering. Chapter 6 will keep the same evidence habit, but the model will become a neural network trained with tensors and backpropagation.

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

## Core Path, Extensions, And Depth

| Layer | What to study now | How to use it |
|---|---|---|
| Required core | Task type, train/test split, baseline, metric, error samples, leakage check, Pipeline | These become the evaluation habits for LLM prompts, RAG retrieval, and Agent behavior later |
| Optional extension | Extra classic algorithms, ML history, Kaggle-style iteration | Return here when a project needs broader algorithm comparison or competition workflow |
| Depth challenge | Keep the data and metric fixed, change one feature or model choice, then explain the before/after errors | This prevents model shopping without evidence |

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

### How to read this output

- The baseline tells you what a naive model can do before learning useful patterns.
- Logistic regression should beat the baseline, but the class-level precision and recall matter more than one headline score.
- If one class has poor recall, inspect those missed examples before changing the model.
- Keep the split, metric, and failure samples fixed when comparing the next experiment.

## Depth Ladder

| Level | What you can prove |
|---|---|
| Minimum pass | You can name the task type, split the data, train a baseline, and read the score. |
| Project-ready | You can explain why the chosen metric matches the goal, and show one error sample instead of trusting one score. |
| Deeper check | You can test for leakage, compare two feature choices, and say what would change in a real product or dataset update. |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
modeling_loop: data, features, model, metric, error review, and next experiment
artifact: code, score, chart, pipeline, or project README
failure_check: leakage, metric mismatch, unstable split, overfitting, or unclear business target
next_action: one controlled experiment rather than many parameter changes
Expected_output: reproducible ML evidence that prepares for deep learning
```

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

<details>
<summary>Reference answers and explanation</summary>

1. Decide the task from the target: categories mean classification, numbers mean regression, no labels usually means clustering or anomaly detection.
2. The baseline is the simplest reproducible model or rule. A real model only matters if it beats that baseline under the same split and metric.
3. Choose the metric from the cost of mistakes. Accuracy is misleading when classes are imbalanced or when one error type is much more expensive.
4. Check leakage by asking whether any feature contains target, future, test-set, or human-review information that would not exist at prediction time.
5. A good next step names one weakness, one evidence sample, and one controlled change rather than changing many knobs at once.

</details>

For a printable checklist, use [5.0 Study Guide and Task Sheet](./study-guide.md). The next chapter moves from sklearn models into neural networks and deep learning training.

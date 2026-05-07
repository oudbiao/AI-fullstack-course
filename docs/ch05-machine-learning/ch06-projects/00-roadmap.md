---
title: "5.6.1 ML Projects Roadmap: Baseline, Evidence, Improvement"
sidebar_position: 18
description: "A compact machine learning project roadmap: define the problem, build a baseline, evaluate, improve, analyze failures, and package evidence."
keywords: [machine learning project guide, house price prediction, customer churn, user segmentation, Kaggle, machine learning portfolio]
---

# 5.6.1 ML Projects Roadmap: Baseline, Evidence, Improvement

This chapter is the exit point of Chapter 5. It proves you can turn a data problem into a modeling workflow that can be evaluated, explained, and shown in a portfolio.

## 5.6.1.1 Look at the Project Loop First

![Machine Learning Project Practice Roadmap](/img/course/ml-projects-roadmap-en.png)

![Machine Learning Project Portfolio Loop](/img/course/ch05-projects-portfolio-loop-en.png)

Keep this project loop:

```text
problem -> data -> baseline -> metric -> improvement -> failure cases -> report
```

Do not jump straight to a complex model. A project without a baseline, metric, and failure analysis is only a demo run.

## 5.6.1.2 Keep One Experiment Log

Create `ml_project_log_first_loop.py`. This is not a model; it is the habit every model project needs.

```python
experiments = [
    {"version": "v1_baseline", "metric": 0.72, "change": "default model"},
    {"version": "v2_features", "metric": 0.78, "change": "add ratio features"},
    {"version": "v3_tuned", "metric": 0.80, "change": "tune max_depth"},
]

best = max(experiments, key=lambda row: row["metric"])

print("best_version:", best["version"])
print("best_metric:", best["metric"])
print("next_step: inspect failure cases before adding more models")
```

Expected output:

```text
best_version: v3_tuned
best_metric: 0.8
next_step: inspect failure cases before adding more models
```

This is the mindset shift: from "I ran a model" to "I can compare versions and explain the next step."

## 5.6.1.3 Learn in This Order

| Order | Read | What to deliver |
|---|---|---|
| 1 | [5.6.2 House Price Prediction](./01-house-price.md) | regression baseline and improvement |
| 2 | [5.6.3 Customer Churn Prediction](./02-customer-churn.md) | classification metric and threshold thinking |
| 3 | [5.6.4 User Segmentation](./03-user-segmentation.md) | cluster interpretation and business labels |
| 4 | [5.6.5 Kaggle Practice](./04-kaggle.md) | real submission workflow |
| 5 | [5.6.6 Hands-on ML Workshop](./05-hands-on-ml-workshop.md) | one complete evidence pack rehearsal |

The workshop comes last because it packages the project habits into one reproducible evidence pack.

## 5.6.1.4 Project Deliverable Standards

![Machine Learning Project Report Storyboard](/img/course/ch05-project-report-storyboard-en.png)

Keep these files for at least one project: `README.md`, run command, metric table, experiment log, one failure case, one chart, and a next-step plan.

## 5.6.1.5 Pass Check

You pass this roadmap when you can clearly say: how I defined the task, what baseline I used, which metric I trusted, what improved, where the model failed, and what I would do next.

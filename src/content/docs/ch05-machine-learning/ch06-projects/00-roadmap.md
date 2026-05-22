---
title: "5.6.1 ML Projects Roadmap: Baseline, Evidence, Improvement"
description: "A compact machine learning project roadmap: define the problem, build a baseline, evaluate, improve, analyze failures, and package evidence."
sidebar:
  order: 18
head:
  - tag: meta
    attrs:
      name: keywords
      content: "machine learning project guide, house price prediction, customer churn, user segmentation, Kaggle, machine learning portfolio"
---
This chapter is the exit point of Chapter 5. It proves you can turn a data problem into a modeling workflow that can be evaluated, explained, and shown in a portfolio.

## Look at the Project Loop First

![Machine Learning Project Practice Roadmap](/img/course/ml-projects-roadmap-en.webp)

![Machine Learning Project Portfolio Loop](/img/course/ch05-projects-portfolio-loop-en.webp)

Keep this project loop:

```text
problem -> data -> baseline -> metric -> improvement -> failure cases -> report
```

Do not jump straight to a complex model. A project without a baseline, metric, and failure analysis is only a demo run.

## Keep One Experiment Log

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

## Learn in This Order

| Order | Read | What to deliver |
|---|---|---|
| 1 | [5.6.2 House Price Prediction](./01-house-price.md) | regression baseline and improvement |
| 2 | [5.6.3 Customer Churn Prediction](./02-customer-churn.md) | classification metric and threshold thinking |
| 3 | [5.6.4 User Segmentation](./03-user-segmentation.md) | cluster interpretation and business labels |
| 4 | [5.6.5 Kaggle Practice](./04-kaggle.md) | real submission workflow |
| 5 | [5.6.6 Hands-on ML Workshop](./05-hands-on-ml-workshop.md) | one complete evidence pack rehearsal |

The workshop comes last because it packages the project habits into one reproducible evidence pack.

## Project Deliverable Standards

![Machine Learning Project Report Storyboard](/img/course/ch05-project-report-storyboard-en.webp)

Keep these files for at least one project: `README.md`, run command, metric table, experiment log, one failure case, one chart, and a next-step plan.

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
project_goal: prediction, segmentation, Kaggle, or end-to-end ML portfolio target
pipeline: data split, preprocessing, model, evaluation, and report artifacts
result: metric table, chart, predictions, failure samples, and README note
failure_check: non-reproducible run, leakage, overfitting, weak baseline, or missing deployment boundary
Expected_output: ML project folder with pipeline, metrics, and failure review
```

## Pass Check

You pass this roadmap when you can clearly say: how I defined the task, what baseline I used, which metric I trusted, what improved, where the model failed, and what I would do next.

<details>
<summary>Check reasoning and explanation</summary>

1. A complete answer defines the task type, the target, and the success metric before discussing model names.
2. The baseline should be the simplest repeatable version: fixed split, minimal preprocessing, one model, and one metric table.
3. An improvement only counts if it is compared against the same split or validation protocol. Changing the split and the model at the same time makes the result hard to trust.
4. Failure analysis should name at least one segment or sample type where the model is weak, then turn that observation into the next controlled experiment.
5. A passing project folder should include a run command, README, experiment log, metric table, chart, failure case, and next-step note.

</details>

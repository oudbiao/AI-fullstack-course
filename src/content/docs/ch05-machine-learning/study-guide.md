---
title: "5.0 Study Guide and Task Sheet: Machine Learning"
description: "A short printable checklist for Chapter 5 after the main guide has been merged into the chapter entry page."
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "machine learning study guide, sklearn, machine learning project, baseline, feature engineering"
---

# 5.0 Study Guide and Task Sheet: Machine Learning

![Machine learning study guide project loop](/img/course/ch05-study-guide-project-loop-en.webp)

The main study route is now in [Chapter 5 entry](./). Use this page only as a quick checklist while you practice.

## One-Line Mental Model

```text
define task -> split data -> train baseline -> evaluate -> inspect errors -> improve
```

If you do not know which model to use, start with a baseline.

## Practice Checklist

| Check | Evidence |
|---|---|
| I can define the task type | problem note |
| I can split data without leakage | train/test split note |
| I can train a dummy baseline and one real model | baseline comparison |
| I can choose a metric for the task | metric note |
| I can inspect errors | error samples |
| I can finish the evidence-pack workshop | `ml_workshop_run/` |

<details>
<summary>Check reasoning and explanation</summary>

1. A task note should say whether the problem is regression, classification, clustering, evaluation, or feature engineering, and what success means.
2. A safe split note explains when the data is split and which preprocessing steps are fitted only on training data.
3. A baseline comparison should include a dummy or simple model and one stronger model under the same evaluation protocol.
4. A metric note should justify the metric using the task goal. Accuracy alone is not enough for imbalanced classification.
5. Error samples should become a next action, not just a screenshot. Good next actions are controlled feature, data, threshold, or model changes.
6. You are ready for Chapter 6 when another person can rerun your evidence pack and understand the modeling decisions.

</details>

## Evidence Rubric

| Artifact | It should answer |
|---|---|
| Problem note | What is the task type, and what counts as success? |
| Split note | How did you keep test data away from training? |
| Baseline comparison | What is the minimum score to beat? |
| Metric note | Why does this metric match the goal better than plain accuracy? |
| Error note | Which mistakes matter most, and what feature or label issue might explain them? |

## Ready To Continue

Continue to Chapter 6 when one tabular project includes a baseline, a real model, metrics, error analysis, and a README that another person can rerun.

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
modeling_loop: data, features, model, metric, error review, and next experiment
artifact: code, score, chart, pipeline, or project README
failure_check: leakage, metric mismatch, unstable split, overfitting, or unclear business target
next_action: one controlled experiment rather than many parameter changes
Expected_output: reproducible ML evidence that prepares for deep learning
```

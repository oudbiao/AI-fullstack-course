---
title: "6.8.1 Deep Learning Projects Roadmap: Train, Inspect, Package"
sidebar_position: 0
description: "A compact deep learning project roadmap: image classification, sentiment analysis, generation, training evidence, and portfolio packaging."
keywords: [deep learning project guide, image classification, sentiment analysis, generative practice, PyTorch portfolio]
---

# 6.8.1 Deep Learning Projects Roadmap: Train, Inspect, Package

This chapter is the exit point of Chapter 6. A deep learning project is not just a training script. It needs data evidence, shape checks, loss logs, prediction samples, failure cases, and a README.

## 6.8.1.1 Look at the Project Loop First

![Deep Learning Project Portfolio Roadmap](/img/course/ch06-projects-portfolio-loop-en.png)

![Deep Learning Project Training Review Loop](/img/course/ch06-deep-learning-project-cycle-en.png)

```text
dataset -> model -> training log -> evaluation -> failure cases -> package
```

## 6.8.1.2 Keep One Evidence Record

Create `dl_project_evidence_first_loop.py`.

```python
evidence = {
    "task": "image classification",
    "baseline_accuracy": 0.71,
    "current_accuracy": 0.82,
    "failure_case_count": 5,
    "next_step": "inspect confused classes and add augmentation",
}

print("task:", evidence["task"])
print("improvement:", round(evidence["current_accuracy"] - evidence["baseline_accuracy"], 3))
print("failure_case_count:", evidence["failure_case_count"])
print("next_step:", evidence["next_step"])
```

Expected output:

```text
task: image classification
improvement: 0.11
failure_case_count: 5
next_step: inspect confused classes and add augmentation
```

This is the project habit: every improvement needs a baseline, metric, failure evidence, and next step.

## 6.8.1.3 Learn in This Order

| Order | Read | What to deliver |
|---|---|---|
| 1 | [6.8.2 Image Classification](./01-image-classification.md) | dataset, CNN/transfer baseline, prediction samples |
| 2 | [6.8.3 Sentiment Analysis](./02-sentiment-analysis.md) | text pipeline, training log, error examples |
| 3 | [6.8.4 Generative Practice](./03-generative-practice.md) | generated samples and review notes |
| 4 | [6.8.5 Hands-on DL Workshop](./04-hands-on-dl-workshop.md) | one reproducible PyTorch evidence pack |

## 6.8.1.4 Project Deliverable Standards

Keep at least these files for one project: `README.md`, run command, dataset note, model summary, loss curve or log, metric table, prediction samples, failure cases, and next-step plan.

## 6.8.1.5 Pass Check

You pass this roadmap when another learner can run your project, inspect the training evidence, see both success and failure samples, and understand what you would improve next.

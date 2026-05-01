---
title: "Stage Learning Task Sheet"
description: "Break the machine learning journey from beginner to hands-on practice into executable learning tasks, practice deliverables, and completion criteria."
keywords: [machine learning, sklearn, feature engineering, model evaluation, learning task sheet]
---

# Stage Learning Task Sheet: Machine Learning from Basics to Practice

![Machine Learning Completion Checklist Diagram](/img/course/ml-task-checklist-en.png)

The goal of this stage is to help you understand the full machine learning project loop: define the problem, prepare the data, build features, train the model, evaluate the results, and analyze errors. Do not focus on memorizing algorithm formulas; instead, place each algorithm back into the context of “what problem it is good at solving and how to judge whether it works well.”

## Tasks you must complete in this stage

| Task | Deliverable | Passing Criteria |
| --- | --- | --- |
| Build ML problem awareness | A problem definition note | Can distinguish classification, regression, clustering, and anomaly detection |
| Run an sklearn baseline end to end | A minimal training script | Can complete train/test split, fit, predict, and score |
| Complete feature engineering | A feature processing log | Can explain missing values, categorical variables, standardization, and leakage risks |
| Complete model evaluation | A metric comparison table | Can explain the use cases of accuracy, recall, F1, AUC, or RMSE |
| Complete the stage project | A reproducible experiment project | Includes README, data description, metrics, failure cases, and improvement plan |

## Recommended learning order

First learn the basic machine learning concepts and the sklearn workflow, then learn supervised learning, unsupervised learning, evaluation methods, and feature engineering. Do not leave feature engineering until the very end, because in real projects model performance often depends first on the data and features.

For every algorithm you learn, always ask three questions at the same time: what is its input, what does it output, and in what situations might it fail. This is more useful than remembering the algorithm name by itself.

## Relationship to the AI Learning Assistant project

This stage corresponds to the v0.4 learning recommendation baseline of the AI Learning Assistant. You can train a simple classifier using historical learning questions and stage labels to predict which stage or topic a new question belongs to. This model does not need to be very powerful, but it can help you understand the shift “from rules to models.”

A simple implementation is recommended: manually prepare dozens of learning question samples, use TF-IDF or simple text features, train a classification model, and output recommended chapters. Then compare it with a rule-matching approach and record the strengths and weaknesses of each.

## Common sticking points

Common issues include mixing up the training set and test set, tuning parameters on the test set, looking only at accuracy, ignoring class imbalance, completing feature processing before splitting the data and causing data leakage, and being unable to reproduce model results. In machine learning projects, whether the evaluation process is correct is usually more important than how advanced the model is.

## Easy / Standard / Challenge Tasks

| Difficulty | What You Need to Do | Suitable For |
|---|---|---|
| Easy | Train a Dummy baseline and output metrics | First-time learners, learners with little time, or beginners |
| Standard | Train a real model and compare it with the baseline | Learners who want to include this stage in their portfolio |
| Challenge | Check for data leakage, class imbalance, or error samples | Learners with some experience who want stronger project evidence |

## Stage badges and boss fight

| Type | Content |
|---|---|
| Boss Fight | Baseline Guardian |
| Unlockable Badges | Baseline Guardian, Leakage Detective |
| Minimum Clear Slogan | Get it running first, then explain it, then record the failures |
| Evidence-saving suggestion | Save screenshots, logs, failure samples, or evaluation tables to `reports/`, `evals/`, or `logs/` |

You can move on after completing the easy version; you are recommended to add it to your portfolio only after completing the standard version; do the challenge version only when you have extra time.

## Stage portfolio deliverables

If you want to turn the results of this stage into a portfolio item, it is recommended to keep at least the following files or equivalent materials.

| Deliverable | Description |
| --- | --- |
| `train.py` or Notebook | A reproducible baseline training flow, including data splitting, training, and evaluation |
| `feature_report.md` | Feature meanings, missing-value handling, encoding, standardization, and leakage risk checks |
| `metrics.md` | Records metrics such as accuracy, F1, AUC, and RMSE, along with the reasons for choosing them |
| `error_analysis.md` | Saves incorrectly predicted samples and analyzes issues in the data, features, model, or metrics |
| `README.md` | Problem definition, run commands, model results, limitations, and next-step plan |

These materials will make the machine learning project more than “just getting a score”; they create a modeling loop that is reproducible, explainable, and improvable.

## Stage completion questions

After learning this stage, you should be able to answer these questions: why do we split the training set and test set, what is data leakage, when is accuracy unreliable, what problem does Pipeline solve, and why is a simple baseline the starting point of all ML projects.

## Completion Status Checklist

- [ ] I can determine whether a problem is classification, regression, clustering, or anomaly detection.
- [ ] I can run a baseline with sklearn and save the evaluation results.
- [ ] I can distinguish the responsibilities of the training set, validation set, and test set.
- [ ] I can identify at least one data leakage risk and reduce the risk with a Pipeline.
- [ ] I have completed a machine learning project and recorded metrics, failure samples, and an improvement plan.

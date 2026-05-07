---
title: "5.0 Study Guide and Task Sheet: How to Learn Machine Learning Without Getting Lost"
sidebar_position: 1
description: "A machine learning study guide for AI full-stack beginners: task definition, baseline, metrics, feature engineering, project roadmap, and acceptance criteria."
keywords: [machine learning study guide, how to learn sklearn, machine learning project, baseline, feature engineering]
---

# 5.0 Study Guide and Task Sheet: How to Learn Machine Learning Without Getting Lost

![Machine learning study loop diagram](/img/course/ml-study-loop-en.png)

If you reach `Chapter 5: Introduction to Machine Learning and Practical Applications` and feel like there are too many model names, don’t rush to memorize algorithms one by one. On your first pass through machine learning, the most important thing is understanding the complete modeling workflow.

## Overall Principles for This Stage

In machine learning, you need to follow one main project line: translate the problem into a task, prepare the data, build a baseline, evaluate with metrics, and then improve the results through features and models.

![Machine learning study guide project loop](/img/course/ch05-study-guide-project-loop-en.png)

## Tasks You Must Complete in This Stage

![Machine Learning Completion Checklist Diagram](/img/course/ml-task-checklist-en.png)

Use this checklist to keep the stage practical. A machine learning project is complete only when you can rerun it and explain the model's failures, not just when you get a score.

| Task | Deliverable | Passing Criteria |
|---|---|---|
| Build ML problem awareness | A problem definition note | Can distinguish classification, regression, clustering, and anomaly detection |
| Run an sklearn baseline end to end | A minimal training script | Can complete train/test split, fit, predict, and score |
| Complete the guided evidence workshop | A generated `ml_workshop_run/` evidence pack | Can rerun the script and explain `model_comparison.csv`, `threshold_review.csv`, `error_samples.csv`, and `leakage_check.md` |
| Complete feature engineering | A feature processing log | Can explain missing values, categorical variables, standardization, and leakage risks |
| Complete model evaluation | A metric comparison table | Can explain the use cases of accuracy, recall, F1, AUC, or RMSE |
| Complete one stage project | A reproducible experiment project | Includes README, data description, metrics, failure cases, and improvement plan |

## Recommended Learning Order

In the first round, study the major historical breakthroughs in machine learning, foundational concepts, and basic Scikit-learn usage. First, you should understand why Bayes, linear models, decision trees, SVM, random forests, Boosting, and XGBoost appeared in that order, and then understand what training sets, test sets, features, labels, models, metrics, and pipelines are.

In the second round, study supervised learning. Linear regression, logistic regression, decision trees, and ensemble learning are enough to support many beginner projects.

In the third round, study unsupervised learning. Clustering, dimensionality reduction, and anomaly detection help you understand how to discover structure when there are no labels.

In the fourth round, study model evaluation and selection. Metrics, cross-validation, bias-variance, and hyperparameter tuning determine whether you can judge if a model is truly good.

In the fifth round, study feature engineering and projects. In many tabular data projects, feature processing matters more than switching models.

If you want a concrete rehearsal before the larger projects, complete [5.6.6 Hands-on Workshop: Build a Reproducible ML Evidence Pack](./ch06-projects/05-hands-on-ml-workshop.md). It turns the abstract loop into one runnable file and a folder of evidence.

## Modeling-term translator for beginners

When machine learning terms start to feel like a vocabulary test, translate them into project actions first:

| Term | First intuition | What you should do in code or reports |
|---|---|---|
| Feature | The input column the model can look at | Decide whether it is numeric, categorical, time-based, or text |
| Label / target | The answer the model should learn to predict | Keep it separate from the features and avoid leakage |
| Baseline | The simplest model or rule to beat | Train it first before chasing complex models |
| Metric | The ruler used to judge the model | Choose accuracy, F1, AUC, MAE, RMSE, or silhouette based on the task |
| Train / validation / test | Learn, choose, and final-check data splits | Never let test-set information leak into preprocessing |
| Overfitting | The model memorizes training data too much | Compare train and test scores, simplify or regularize |
| Underfitting | The model is too weak to learn the pattern | Add useful features or try a more expressive model |
| Cross-validation | Repeatedly test on different splits | Use it when one split may be unstable |
| Pipeline | Preprocessing and model packaged together | Prevent leakage and make experiments reproducible |
| Hyperparameter | A human-chosen setting before training | Tune it with validation or cross-validation, not the test set |

If you can explain these terms as project actions, you are no longer only learning APIs; you are learning how to run a modeling investigation.

## Suggested Learning Pace

| Content Type | Suggested Time | Learning Goal |
|---|---|---|
| Basic concept pages | 2–3 hours | Be able to explain tasks, data, and metrics |
| Algorithm pages | 2–4 hours | Know the use cases and intuition behind algorithms |
| Evaluation pages | 2–4 hours | Be able to judge whether model performance is trustworthy |
| Project pages | 8–16 hours | Complete the full modeling loop |

## Project Roadmap for This Stage

Before choosing a larger topic, run the [5.6.6 hands-on evidence-pack workshop](./ch06-projects/05-hands-on-ml-workshop.md) once. Treat it as a warm-up: you will create data, split it, train a baseline, compare models, inspect errors, and write project evidence in the same sitting.

For your first project, it is recommended to predict house prices and practice regression, feature processing, and error analysis.

For your second project, it is recommended to predict customer churn and practice classification, confusion matrix, AUC, recall, and business interpretation.

For your third project, it is recommended to segment users and practice unsupervised learning, interpreting clustering results, and visualization.

If time is limited, complete at least one project end to end instead of only studying the algorithm pages.

## Common Sticking Points

The most common sticking point is “I don’t know which model to use.” The solution is to start with a baseline. In many cases, confirming that the data and metrics are working with a simple model is more important than trying complex models from the start.

The second sticking point is metric confusion. Classification, regression, and clustering use different metrics, so first ask what the task is, then choose the metric.

The third sticking point is data leakage. Standardization, encoding, and feature selection all require attention to the boundary between the training set and the test set.

## Stage Portfolio Deliverables

If you want this stage to become a portfolio item, keep at least these files or equivalent evidence.

| Deliverable | Description |
|---|---|
| `train.py` or Notebook | A reproducible baseline training flow, including data splitting, training, and evaluation |
| `feature_report.md` | Feature meanings, missing-value handling, encoding, standardization, and leakage risk checks |
| `metrics.md` | Records accuracy, F1, AUC, RMSE, or other task-appropriate metrics with the reasons for choosing them |
| `error_analysis.md` | Saves incorrectly predicted samples and analyzes issues in the data, features, model, or metrics |
| `README.md` | Problem definition, run commands, model results, limitations, and next-step plan |

## Stage Completion Questions

After finishing this stage, you should be able to complete a machine learning project independently: define the task, process the data, train a baseline, choose metrics, improve the model, and explain the results.

Before moving to Chapter 6, check that you can answer these questions:

- Why do we split the training set and test set?
- What is data leakage, and how does `Pipeline` reduce the risk?
- When is accuracy unreliable?
- Why is a simple baseline the starting point of an ML project?
- What does the model do well, what does it do poorly, and what would you improve next?

If you can organize a tabular data project into a report and answer these questions, you are ready to move on to the deep learning stage.

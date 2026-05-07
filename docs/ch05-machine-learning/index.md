---
title: "5 Introduction to Machine Learning: From Basics to Practice"
sidebar_position: 0
description: "Systematically learn machine learning task decomposition, supervised learning, unsupervised learning, model evaluation, feature engineering, and Scikit-learn practice to complete a full modeling loop."
keywords: [machine learning, Scikit-learn, supervised learning, unsupervised learning, regression, classification, clustering]
---

# 5 Introduction to Machine Learning: From Basics to Practice

![Main visual for machine learning](/img/course/ch05-machine-learning-en.png)

This stage is about solving the question: “Can a problem be turned into a trainable, evaluable, and improvable model project?” You will start from data and complete task definition, baseline creation, training, evaluation, feature engineering, and review.

## Story-based introduction: Training an “intern who knows how to review”

You can think of machine learning as training an intern: first, you give it some historical cases and let it summarize patterns from them; then you test it with cases it has never seen before to see whether it really learned; if the result is poor, you go back and inspect the data, features, metrics, and model. This process is not a one-shot success, but an ongoing cycle of trial, evaluation, and improvement.

## Interactive exercise: Guess first, then train, then review

For every machine learning task, before training, write down your own guesses: which features may be useful, which metric matters most, and where the model may fail. After training, compare the results and review them. In this way, you are not just “calling fit and predict”; you are practicing the modeling judgment that AI engineers really need.

## Project bonus content

The bonus project for this stage can be a “model detective report”: not only showing model scores, but also explaining where the data came from, why you chose this metric, how strong the baseline is, what common patterns appear in the error samples, and how to improve next. This kind of report feels much closer to real-world work than simply pasting an accuracy number.

## Stage overview

| Information | Description |
|---|---|
| Suitable for | Learners who have mastered Python, data analysis, and the minimum foundation in AI math |
| Estimated study time | 120–160 hours |
| Prerequisites | Complete the first three stages |
| Stage deliverables | House price prediction, customer churn prediction, user segmentation, or a Kaggle starter project |

## Beginner minimum path to completion

Beginners should first master the core concepts of classification, regression, clustering, training set and test set, baseline, and evaluation metrics. You do not need to memorize every algorithm detail at the beginning. As long as you can use Scikit-learn to complete a full modeling workflow and explain what the model score means, you have achieved the minimum completion goal.

## Advanced path

Experienced learners can focus on feature engineering, cross-validation, bias and variance, hyperparameter tuning strategies, data leakage, and error analysis. You can also try wrapping the model training workflow into a Pipeline and compare the performance of multiple models under the same metric.

## Machine learning’s place in the history of AI

Machine learning moved AI from “human-written rules” to “machines learning patterns from data.” The era of expert systems relied on large numbers of manual rules, while machine learning started to let models fit patterns automatically from data. Today’s large models are very powerful, but machine learning ideas such as data splits, evaluation metrics, overfitting, generalization, features, and error analysis are still the foundation of AI engineering.

![Main loop of machine learning modeling](/img/course/ch05-modeling-loop-backbone-en.png)

If you want to understand each technical breakthrough in historical order, you can first read [5.1.2 Main line of machine learning historical breakthroughs](./ch01-ml-basics/04-history-breakthroughs.md). It maps Bayes, MLE, EM, linear models, decision trees, SVM, random forests, boosting, XGBoost, and sklearn to the corresponding sections in this chapter, so you can understand why each algorithm appeared.

## What beginners should do first, and what advanced learners should do later

When beginners study this stage for the first time, they should first complete a minimum classification or regression project: split the data, build a baseline, train, predict, evaluate, and write conclusions. Do not rush to compare many models; first learn how to judge whether a score is trustworthy.

Experienced learners can focus on error analysis and experiment design: whether the baseline is strong enough, whether the metric matches the goal, whether there is data leakage, and what pattern the errors show. Your goal is to write a modeling report that others can reproduce, question, and continue improving.

:::info Hands-on checkpoint
If you want one guided run before choosing a larger project, start with [5.6.6 Hands-on Workshop: Build a Reproducible ML Evidence Pack](./ch06-projects/05-hands-on-ml-workshop.md). It creates a local dataset, baseline, Pipeline, model comparison, threshold review, error samples, leakage notes, and README evidence.
:::

## Learning path for this stage

Section 5.1 covers the basic concepts of machine learning, including the main line of historical breakthroughs, task types, training and test sets, basic usage of Scikit-learn, and how mathematics enters machine learning.

Section 5.2 covers supervised learning, including linear regression, logistic regression, decision trees, and ensemble learning. You will understand how classification and regression tasks are modeled.

Section 5.3 covers unsupervised learning, including clustering, dimensionality reduction, and anomaly detection. You will understand how to discover structure when there are no labels.

Section 5.4 covers model evaluation and selection, including metrics, cross-validation, bias and variance, and hyperparameter tuning. This is where you decide whether you can really tell if a model is good.

Section 5.5 covers feature engineering. For many real tabular-data projects, understanding and processing features is more important than choosing a model.

Section 5.6 moves into hands-on projects and runs through the full modeling workflow.

## What you should be able to do after learning

- Determine whether a problem is classification, regression, clustering, or anomaly detection
- Build a baseline model with Scikit-learn
- Split training and test sets correctly and avoid data leakage
- Choose suitable metrics to evaluate models
- Perform basic feature processing and model improvement
- Organize model results into an interpretable project report

## Common mistakes

Do not turn machine learning into a catalog of model names. What really matters is the full workflow: how the problem is defined, how the data is prepared, how the first version of the model is built, how the metrics are read, and which step to check first when things fail.

Also, do not pursue complex models at the very beginning. Many projects should start with a simple baseline to make sure the data and evaluation are correct, and then gradually try more complex methods.

## Modeling failure theater: A high score may still be untrustworthy

If the model score is unusually high, first suspect data leakage, incorrect train-test splitting, or the target column being indirectly included in the features; if the training score is high but the test score is low, check overfitting first; if all models perform poorly, go back and inspect label quality, feature meaning, and metric choice instead of immediately switching to a more complex model.

## Stage review card: From data table to modeling report

After finishing this stage, you can use the table below to check whether you have turned machine learning into a complete chain rather than just memorizing a few model names.

| Review question | What you should be able to answer |
|---|---|
| Problem definition | Is this classification, regression, clustering, or anomaly detection? Why? |
| Data preparation | Which columns are features, which column is the label, and do they need cleaning or encoding? |
| Baseline | What is the simplest first model, and roughly what score does it get? |
| Metric choice | Why use accuracy, F1, MAE, RMSE, or another metric? |
| Error analysis | Which samples or scenarios does the model mainly get wrong? |
| Next-step optimization | Should you improve the data, features, model, or evaluation method first? |

The real output of this stage is being able to write a modeling report that others can understand: it should include not only the score, but also the problem definition, data description, baseline, metrics, error analysis, and next-step plan.

## Minimum runnable experiment: Start with a trustworthy baseline

The minimum experiment for this stage is to choose a tabular-data task and, instead of chasing a high score, complete only data splitting, baseline training, metric calculation, and error sample inspection. You need to prove that the model result comes from the correct process, not from data leakage or luck.

```python
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DummyClassifier(strategy="most_frequent")
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
```

Start with a baseline, then talk about optimization. Without a baseline, it is hard to tell whether the model score has actually improved.

## Machine learning failure case library: Check the data and evaluation first

| Phenomenon | Common cause | How to locate it | How to fix it |
|---|---|---|---|
| Abnormally high score | Data leakage, splitting errors, target column included in features | Check the feature list and splitting method | Split again and remove leaked features |
| Good training, poor test | Overfitting, too few samples, noisy features | Compare training and test metrics | Simplify the model, add regularization, increase data |
| All models perform poorly | Poor label quality, weak features, unsuitable metric | Inspect error samples and label definition | Improve the data, add features, change the metric |
| Results are not reproducible | Random seed, dependency, or data version not fixed | Rerun and compare data versions | Fix the seed, save configuration and data notes |


## Stage projects

The basic version is to complete a baseline project on tabular data, including data splitting, model training, and basic metrics. The standard version should add feature processing, cross-validation, model comparison, and error analysis, forming an interpretable modeling report. The challenge version can use a Kaggle starter task or real business data, adding experiment logs, feature iteration, and pre-deployment risk notes.

For a step-by-step project rehearsal, use [5.6.6 Hands-on Workshop: Build a Reproducible ML Evidence Pack](./ch06-projects/05-hands-on-ml-workshop.md) before the house price, churn, segmentation, or Kaggle projects.

If you want to see a more detailed learning sequence, you can read [5.0 Study guide: How to learn machine learning without getting confused](./study-guide.md).


## Stage deliverables

| Deliverable | Minimum version | Portfolio version |
|---|---|---|
| Baseline model | Complete one training and test evaluation | Explain the problem type, data split, metric, and baseline meaning |
| Feature processing record | Perform basic missing-value handling and category encoding | Record feature selection, transformation rationale, and data leakage checks |
| Model comparison table | Compare 2 models or parameter settings | Include cross-validation, metric interpretation, and error analysis |
| Error sample analysis | List several mispredicted samples | Attribute them to data, features, model, or metric choice |
| Modeling report | Clearly write the runtime command and score | Show problem definition, experiment process, results, limitations, and next steps |

## Relationship to the full AI learning assistant project

This stage can correspond to AI Learning Assistant v0.5: predicting the risk of learning-task delays, or classifying learning problems into environment, syntax, data, model, RAG, Agent, and other types. If you are learning according to the end-to-end project path, it is recommended that by the end of this stage you submit at least one version note: what new capability was added in this stage, how to run it, what the sample input and output are, what problems were encountered, and how you plan to improve it next.


## Stage completion criteria

| Completion level | What you need to do |
|---|---|
| Minimum pass | Can complete regression, classification, and clustering projects, and explain features, metrics, overfitting, and baseline. |
| Recommended pass | Complete at least one runnable mini-project in this stage and record the run method, sample input/output, and problems encountered in the README. |
| Portfolio pass | Connect the output of this stage to the “AI Learning Assistant” end-to-end project, leaving screenshots, logs, evaluation samples, and a next-step plan. |

After finishing this stage, you do not need to memorize every detail. What matters more is being able to clearly explain: what problem this stage solves, how it relates to the previous stage, and how it supports future learning. The next stage will move into neural networks and deep learning training.

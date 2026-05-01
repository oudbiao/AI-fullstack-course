---
title: "Elective Module: Supplementary Classic ML Algorithms"
sidebar_position: 0
description: "Overview of the supplementary classic machine learning module, helping you understand the learning order, applicable scenarios, and the relationships between lessons."
---

# Elective Module: Supplementary Classic ML Algorithms

:::tip Module Positioning
These algorithms are still very valuable in many small- to medium-sized data tasks. Studying them helps complete your toolbox for making judgments.
:::

![Module map for supplementary classic ML algorithms](/img/course/elective-classic-ml-module-map.png)

## Learning Objectives

- Understand the role of the supplementary classic machine learning module in the overall learning path
- Know what problem each lesson in this module solves
- Be clear about which topics to learn first and which to learn later
- Build intuition quickly with a minimal example

---

## 1. What problem does this module solve?

### 1.1 Module Positioning

The purpose of the supplementary classic machine learning module is not to “learn a little more,” but to fill in capabilities that often determine the upper limit of engineering performance.

You can first think of it as a set of topic-based toolboxes:

- Come back to them when you encounter relevant projects
- No need to finish everything at once
- But once you enter the corresponding scenario, they become very valuable

### 1.2 Recommended learning order

A relatively safe learning approach is usually:

1. First look at the overview to understand what each lesson is roughly about
2. Start with the most basic topics that can be applied immediately
3. Then move on to content that is more engineering-focused or project-oriented

---

## 2. What topics are included in this module?

### 2.1 Chapter list

| Chapter | Topic |
|---|---|
| Lesson 1 | Support Vector Machine |
| Lesson 2 | K-Nearest Neighbors |
| Lesson 3 | Naive Bayes |
| Lesson 4 | Linear Discriminant Analysis |

### 2.2 How should you use this module?

A very practical strategy is:

- First use the main course to get the overall workflow running
- When you have a specific need, come back to the elective module for focused improvement

This way, you won’t lose the rhythm of the main learning path because there are too many specialized topics.

---

## 3. A minimal runnable example

:::info Run Tip
```bash
pip install numpy scikit-learn
```
:::

```python
import numpy as np
from sklearn.svm import SVC

X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y = np.array([0, 1, 1, 0])

clf = SVC(kernel="linear")
clf.fit(X, y)
print(clf.predict([[0.8, 0.9]]))
```

### 3.2 What should you take away from this example?

This small piece of code is not meant to cover the whole module. Instead, it is meant to help you quickly build a sense of “what exactly does this module do?”

When reading it, focus on these three things first:

- What the input is
- What happens in the middle
- How the output corresponds to a real project

---

## 4. Learning recommendations

### 4.1 If time is limited, what should you learn first?

Prioritize topics that will appear frequently in later projects and can immediately help you reduce cost or improve efficiency.

### 4.2 Common mistakes

- Seeing it as elective and skipping it completely
- Trying to finish all elective topics at once
- Only reading concepts without running the minimal example

---

## 5. When is the best time to come back and study this module?

When the following signs appear, it means you are a good fit to return and fill in this set of topics:

- Your current task uses small- to medium-sized data, and deep learning is not necessarily the best choice
- You need a strong baseline but don’t know where to start
- You want to improve your judgment about “why choose this model”
- You want to complete your classic ML toolbox instead of only knowing tree models and linear models

## 6. What can you do after finishing this module?

- Select models more flexibly in small- to medium-sized data tasks
- Understand the applicable boundaries of SVM, KNN, Naive Bayes, and LDA
- Add a more solid classic baseline to mainline projects

---

## Summary

This overview page is meant to give you a map. When you actually study the module, you do not need to aim for “understanding everything.” Instead, you should know when to come back and which part to fill in first.

---
title: "Phase Learning Task Sheet"
description: "Break down the minimum necessary foundation stage of AI mathematics into executable learning tasks, practice deliverables, and pass criteria."
keywords: [AI mathematics, linear algebra, probability and statistics, gradient descent, learning task sheet]
---

# Phase Learning Task Sheet: Minimum Necessary Foundation for AI Mathematics

![Minimal math checkpoint task sheet diagram](/img/course/math-task-checklist-en.png)

The goal of this stage is not to turn math into an exam subject, but to help you understand the core intuitions behind AI models. You need to understand how vectors represent samples, how matrices represent transformations, how probability expresses uncertainty, and how gradients guide model updates.

## Required tasks for this stage

| Task | Deliverable | Pass Criteria |
| --- | --- | --- |
| Understand vectors and matrices | A vector similarity experiment | Can explain dot product, norm, cosine similarity, and matrix multiplication |
| Understand probability and statistics | A distribution visualization Notebook | Can explain mean, variance, conditional probability, and common distributions |
| Understand information content and entropy | An information theory note | Can explain the relationship between entropy, cross-entropy, and model loss |
| Understand gradient descent | A parameter update visualization experiment | Can explain learning rate, gradient direction, and the convergence process |
| Complete the stage project | A small math-intuition visualization project | Can connect math concepts to ML, RAG, or LLM scenarios |

## Recommended learning order

First learn vectors and matrices, then probability and statistics, and finally calculus and optimization. Do not start by pursuing rigorous proofs. In the first pass, focus on building intuition for “what these concepts do inside a model.”

When learning math, it is recommended to pair each concept with a small experiment. For example, use vector similarity to explain retrieval, use probability distributions to explain classification confidence, and use gradient descent to explain why a model improves gradually.

## Relationship to the AI Learning Assistant project

This stage corresponds to the v0.4 math explanation capability of the AI learning assistant. You can add a “concept explanation card” feature to the learning assistant: input a math concept, and output an intuitive explanation, AI scenario, simple example, and common misconceptions.

It is recommended to start with a static template and not call the model yet. The key is to translate abstract mathematical concepts into engineering language that can be used in later projects.

## Common sticking points

Common issues include treating formulas as isolated memorization, understanding derivations but not knowing when to use them, confusing probability with frequency, being unable to understand gradient direction, and only knowing how to use libraries without understanding what the metrics mean. When you get stuck, first return to graphs, numerical experiments, and concrete AI scenarios.

## Easy version / Standard version / Challenge version tasks

| Difficulty | What you need to complete | Who it is for |
|---|---|---|
| Easy version | Use code to calculate similarity or probability metrics | Learners studying for the first time, with limited time, or just getting started |
| Standard version | Compare two metrics and explain the differences | Learners who want to include this stage in their portfolio |
| Challenge version | Explain one metric clearly with a hand-calculated example, code, and diagrams | Learners with a foundation who want stronger project evidence |

## Stage badges and Boss battle

| Type | Content |
|---|---|
| Boss battle | Metric maze |
| Unlockable badges | Vector Translator, Metric Explainer |
| Minimal pass slogan | Get it running first, then explain it, then record failures |
| Evidence-saving suggestion | Save screenshots, logs, failure samples, or evaluation tables into `reports/`, `evals/`, or `logs/` |

You can move on after completing the easy version; you should only include it in your portfolio after completing the standard version; do the challenge version only when you have extra bandwidth.

## Stage portfolio deliverables

If you want to capture the results of this stage in your portfolio, it is recommended to keep at least the following files or equivalent materials.

| Deliverable | Description |
| --- | --- |
| `vector_similarity.ipynb` | Use vectors, dot products, and cosine similarity to explain sample similarity |
| `probability_demo.ipynb` | Visualize random variables, distributions, mean, variance, and uncertainty |
| `gradient_descent.ipynb` | Show the learning rate, gradient direction, and convergence path |
| `math_cards.md` | Translate vectors, probability, entropy, and gradients into intuition for AI applications |
| `reflection.md` | Explain how these mathematical concepts support machine learning, RAG, and large models |

These materials will prove that you are not learning math just for exams, but can turn math into a tool for understanding models, debugging models, and explaining systems.

## Stage checkpoint questions

After finishing, you should be able to answer these questions: why text and images can be represented as vectors, why similarity can be used for retrieval, why probability can represent model uncertainty, why cross-entropy is commonly used for classification, and why gradient descent can gradually improve model parameters.

## Completion status Checklist

- [ ] I can explain the role of vectors, matrices, and similarity in AI.
- [ ] I can use charts to explain at least one probability distribution and uncertainty.
- [ ] I can explain the relationship between entropy, cross-entropy, and classification loss.
- [ ] I can demonstrate the gradient descent process with a small experiment.
- [ ] I have written at least 3 math concepts into my own AI application explanation cards.

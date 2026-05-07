---
title: "4.0 Study Guide and Task Sheet: AI Math Foundations"
sidebar_position: 1
description: "A compact study guide and task sheet for Chapter 4: learn linear algebra, probability and statistics, calculus, and optimization in the right order before running the final workshop."
keywords: [AI math study guide, AI math task sheet, linear algebra, probability and statistics, gradient descent]
---

# 4.0 Study Guide and Task Sheet: AI Math Foundations

![AI math learning loop diagram](/img/course/math-study-loop-en.png)

Use this page as the control panel for Chapter 4. The goal is not to learn every proof. The goal is to build enough math intuition to understand model inputs, uncertainty, loss, and parameter updates.

## 4.0.1 Study This Stage in Order

![Minimum closed loop for AI math study guide](/img/course/ch04-study-guide-math-minimum-loop-en.png)

Study the theory first, then run the full workshop. This keeps the workshop from becoming a pile of unexplained commands.

| Order | Section | What to focus on | Evidence to produce |
|---|---|---|---|
| 1 | `4.1 Linear Algebra in Practice` | Vectors, matrices, dot product, norm, cosine similarity | A tiny vector similarity comparison |
| 2 | `4.2 Probability and Statistics in Practice` | Probability, distributions, mean, variance, entropy | A small probability or uncertainty note |
| 3 | `4.3 Calculus and Optimization in Practice` | Derivatives, gradients, learning rate, gradient descent | A short parameter-update trace |
| 4 | `4.4 Hands-on Math Workshop` | Connect the three ideas in one runnable script | The generated evidence folder |

If you only have one day, read the roadmap page in each theory section, run one small code example, and then complete `4.4`.

## 4.0.2 Terms You Need Before You Start

| Term | Full name | First meaning in this chapter |
|---|---|---|
| `ML` | Machine Learning | A model learns patterns from data and uses them to predict or decide. |
| `RAG` | Retrieval-Augmented Generation | Retrieve useful documents first, then answer with that evidence. |
| `LLM` | Large Language Model | A large text model that predicts and generates tokens. |
| `Embedding` | Vector representation | Turn text, images, users, or items into vectors for comparison. |
| `Notebook` | Jupyter Notebook | A file that keeps code, charts, notes, and outputs together. |

You do not need to master these engineering topics now. They are here so each math idea has a future place to land.

## 4.0.3 Minimum Task Sheet

| Task | Deliverable | Pass criteria |
|---|---|---|
| Understand vectors and matrices | `vector_similarity` example or notes | Can explain dot product, norm, cosine similarity, and matrix multiplication in plain language |
| Understand probability and statistics | Distribution or sampling note | Can explain mean, variance, conditional probability, and uncertainty |
| Understand entropy and loss | One short information-theory card | Can explain why confident and uncertain predictions have different loss values |
| Understand gradient descent | Parameter update table or chart | Can explain gradient direction, learning rate, and why loss can fall step by step |
| Complete the chapter artifact | `ch04_math_workshop_evidence/` | Can rerun the workshop and explain the generated CSV, SVG, cards, and README |

## 4.0.4 Formula-to-Intuition Translator

When a formula looks intimidating, translate it into model language before trying to derive it.

| Term | First intuition | Where it appears later |
|---|---|---|
| Vector | A row of features, or the coordinates of meaning | Embeddings, similarity search, model inputs |
| Matrix | Many rows of data, or a machine that transforms vectors | Neural network layers, attention weights |
| Dot product | How much two directions agree | Cosine similarity, attention scores |
| Norm | Length or strength | Distance, regularization, gradient clipping |
| PCA | Rotate the coordinate system and keep the most informative axes | Dimensionality reduction, visualization |
| MLE | Choose parameters that best explain observed data | Loss functions, logistic regression |
| Entropy | Uncertainty or surprise | Classification loss, language models |
| GD / SGD | Walk downhill on the loss surface | Model training, optimizers |

If you can explain the right column aloud, you have already learned the useful first-pass version.

## 4.0.5 Stage Checkpoint

Before moving to Chapter 5, check that you can do these five things:

- Explain why one sample can be represented as a vector and a dataset as a matrix.
- Explain why a model output can be read as probability or confidence.
- Explain why entropy and cross-entropy are related to uncertainty and loss.
- Demonstrate a minimal gradient descent process with code or a table.
- Run [4.4 Hands-on: Full Chapter 4 Math Workshop](./hands-on-math-workshop.md) and keep the generated evidence folder.

You are ready to continue when you can connect each formula to a model action: represent data, measure uncertainty, measure loss, or update parameters.

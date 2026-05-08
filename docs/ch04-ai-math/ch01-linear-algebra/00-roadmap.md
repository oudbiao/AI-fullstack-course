---
title: "4.1.1 Linear Algebra Roadmap: Data as Vectors, Batches as Matrices"
sidebar_position: 0
description: "A compact linear algebra roadmap for AI: vectors, matrices, dot products, eigenvalues, and transformations."
keywords: [Linear Algebra Guide, AI Math Guide, Vectors, Matrices, Eigenvalues, PCA]
---

# 4.1.1 Linear Algebra Roadmap: Data as Vectors, Batches as Matrices

Linear algebra is the language AI uses to represent data and transform it. Do not start by memorizing proofs. First see what each object does in code.

## Look at the Map First

![Linear Algebra Learning Map](/img/course/ch04-linear-algebra-roadmap-vertical-en.webp)

The chapter flow is:

![Linear Algebra Chapter Flow](/img/course/ch04-linear-algebra-chapter-flow-en.webp)

| Idea | First meaning in AI |
|---|---|
| vector | one object written as numbers |
| matrix | many vectors stacked together, or a transformation |
| dot product | compare matching positions and add them up |
| matrix multiplication | many dot products at once |
| eigenvalue / eigenvector | important directions, useful for PCA intuition |

## Run the Smallest Loop

Create `linear_algebra_first_loop.py` and run it after installing `numpy`.

```python
import numpy as np

student = np.array([90, 85, 92])
students = np.array(
    [
        [90, 85, 92],
        [70, 88, 75],
        [95, 91, 89],
    ]
)
weights = np.array([0.4, 0.2, 0.4])

single_score = student @ weights
all_scores = students @ weights

print("student_vector:", student)
print("matrix_shape:", students.shape)
print("single_score:", round(single_score, 2))
print("all_scores:", all_scores.round(2))
```

Expected output:

```text
student_vector: [90 85 92]
matrix_shape: (3, 3)
single_score: 89.8
all_scores: [89.8 75.6 91.8]
```

If you accidentally use `*` instead of `@`, you get element-by-element multiplication, not a weighted score. This is the most useful early distinction.

## Learn in This Order

| Order | Read | What to focus on first |
|---|---|---|
| 1 | [4.1.2 Vectors](./01-vectors.md) | object -> vector, length, dot product, cosine similarity |
| 2 | [4.1.3 Matrices](./02-matrices.md) | batch data, matrix multiplication, `X @ W + b` |
| 3 | [4.1.4 Eigenvalues and Eigenvectors](./03-eigenvalues.md) | special directions, PCA intuition |
| 4 | [4.1.5 Vector Spaces](./04-vector-spaces.md) | basis, dimension, linear transformation |

## Pass Check

You pass this roadmap when you can explain why one sample is a vector, why a batch is a matrix, what `@` does, and why these ideas reappear in RAG similarity, PCA, and neural network layers.

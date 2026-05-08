---
title: "E.C.4 Linear Discriminant Analysis"
sidebar_position: 15
description: "Use LDA for lightweight classification and label-aware dimensionality reduction."
keywords: [LDA, linear discriminant analysis, dimensionality reduction, classification, classic ML]
---

# E.C.4 Linear Discriminant Analysis

![LDA supervised projection intuition diagram](/img/course/elective-lda-projection-map-en.webp)

LDA finds a projection that keeps samples from the same class close and different classes far apart. It can act as a classifier and as supervised dimensionality reduction.

## What You Need

- Python 3.10+
- Current stable `scikit-learn` and `numpy`

```bash
python -m pip install -U scikit-learn numpy
```

## Key Terms

- **Within-class variance**: how spread out each class is.
- **Between-class separation**: how far class centers are from each other.
- **Projection**: mapping features into fewer dimensions.
- **Supervised dimensionality reduction**: reducing dimensions while using labels.
- **LDA here**: Linear Discriminant Analysis, not Latent Dirichlet Allocation.

## Run LDA Classification And Projection

Create `lda_projection.py`:

```python
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [2.0, 1.5],
    [4.0, 5.0],
    [4.5, 4.8],
    [5.0, 4.5],
])
y = np.array([0, 0, 0, 1, 1, 1])

model = LinearDiscriminantAnalysis(n_components=1)
model.fit(X, y)

pred = model.predict([[1.4, 1.9], [4.8, 4.6]])
projection = model.transform(X)

print("predictions:", pred.tolist())
print("projection_shape:", projection.shape)
```

Run it:

```bash
python lda_projection.py
```

Expected output:

```text
predictions: [0, 1]
projection_shape: (6, 1)
```

The same model classified new points and projected the training data into one discriminative direction.

## Compare With PCA

PCA finds directions with high overall variance and ignores labels. LDA uses labels and asks which direction separates classes best. That makes LDA useful when class separation matters more than general compression.

## Practical Rule

Try LDA when:

1. Labels are available.
2. Classes are reasonably compact.
3. You want a lightweight linear baseline.
4. You want a low-dimensional representation for visualization or downstream models.

Avoid it as the first choice when class boundaries are highly nonlinear.

## Common Mistakes

- Confusing this LDA with topic-model LDA.
- Assuming LDA is always better than PCA because it uses labels.
- Forgetting that with two classes, LDA can project to at most one component.

## Practice

Add a third class and set `n_components=2`. Then print the new projection shape and explain why the maximum number of components changed.

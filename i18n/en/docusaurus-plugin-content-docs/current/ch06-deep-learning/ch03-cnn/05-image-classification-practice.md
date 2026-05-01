---
title: "3.6 CNN Practice: Image Classification"
sidebar_position: 5
description: "Walk through a complete small CNN image classification project from data generation, network building, training, validation, to prediction."
keywords: [image classification, CNN, PyTorch, train loop, validation, synthetic dataset]
---

# CNN Practice: Image Classification

:::tip Section Overview
After we have finished convolution, CNN architectures, classic models, and transfer learning, the most important thing is:

> **To truly connect these concepts into a complete training loop.**

This section does not aim for “large model performance”; it aims for something more important:

> To help you complete an image classification project end to end.
:::

## Learning Objectives

- Build a minimal trainable image classification task
- Run training, validation, and prediction with a CNN end to end
- Understand how data, model, loss function, and metrics work together in an image classification project
- Learn how to tell from the results whether the model has actually learned anything

---

## 1. What is the minimal closed loop for an image classification project?

An image classification project needs at least these parts:

1. Data
2. Class labels
3. Model
4. Loss function
5. Training loop
6. Validation / testing

Many beginners feel confused when learning because they only see the “model architecture” and never connect the whole loop.

The key point of this lesson is to walk through this chain completely.

---

## 2. First, prepare data that can run directly

### 2.1 Why keep using synthetic images?

Because:

- It does not depend on external downloads
- The class patterns are very clear
- It is ideal for teaching

### 2.2 Create three small image classes

We create 3 classes:

- Vertical line
- Horizontal line
- Diagonal line

```python
import numpy as np
import matplotlib.pyplot as plt

def make_image(label, size=12):
    img = np.zeros((size, size), dtype=np.float32)

    if label == 0:
        img[:, size // 2] = 1.0
    elif label == 1:
        img[size // 2, :] = 1.0
    else:
        for i in range(size):
            img[i, i] = 1.0

    img += np.random.randn(size, size).astype(np.float32) * 0.05
    return np.clip(img, 0.0, 1.0)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for label in range(3):
    axes[label].imshow(make_image(label), cmap="gray")
    axes[label].set_title(f"class {label}")
    axes[label].axis("off")
plt.tight_layout()
plt.show()
```

### 2.3 This dataset is simple, but what can it teach you?

It can teach you enough to understand:

- How image tensors should be organized
- How class labels align with the data
- How CNNs learn local patterns

That makes it much better for beginners than throwing a large dataset at you right away.

---

## 3. Turn the data into a training set and a validation set

```python
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

X, y = [], []
for label in range(3):
    for _ in range(100):
        X.append(make_image(label))
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

# Shuffle
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# Convert to tensors
X = torch.tensor(X).unsqueeze(1)  # [N, 1, H, W]
y = torch.tensor(y)

split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print("train:", X_train.shape, y_train.shape)
print("val  :", X_val.shape, y_val.shape)
```

### 3.2 Why do we use `unsqueeze(1)`?

Because PyTorch convolution inputs need the shape:

- `[batch, channel, height, width]`

Here we are using grayscale images, so the channel count is 1.

---

## 4. Define a minimal CNN classifier

```python
import torch
from torch import nn

class TinyCNNClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = TinyCNNClassifier(num_classes=3)
sample_out = model(X_train[:4])
print("sample output shape:", sample_out.shape)
```

### 4.2 Why is it `16 * 3 * 3` here?

Because the original image size is `12x12`:

- After the first `MaxPool2d(2)`, it becomes `6x6`
- After the second `MaxPool2d(2)`, it becomes `3x3`

The final output has 16 channels, so after flattening it is:

> `16 * 3 * 3`

This is the most common shape calculation problem in CNN practice.

---

## 5. Complete training loop

```python
import torch
from torch import nn

torch.manual_seed(42)

model = TinyCNNClassifier(num_classes=3)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    train_logits = model(X_train)
    train_loss = loss_fn(train_logits, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val)
            train_acc = (train_logits.argmax(dim=1) == y_train).float().mean().item()
            val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()

        print(
            f"epoch={epoch:3d}, "
            f"train_loss={train_loss.item():.4f}, "
            f"val_loss={val_loss.item():.4f}, "
            f"train_acc={train_acc:.3f}, "
            f"val_acc={val_acc:.3f}"
        )
```

### 5.2 What should you pay the most attention to in this code?

When learning image classification, the four most important things to watch are:

- `train_loss`
- `val_loss`
- `train_acc`
- `val_acc`

Because they tell you:

- Whether the model is learning
- Whether it is overfitting
- Whether it is still converging stably

---

## 6. Make a real prediction

### 6.1 Look at a single sample

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    sample = X_val[0:1]
    pred = model(sample).argmax(dim=1).item()
    true = y_val[0].item()

plt.imshow(sample[0, 0].numpy(), cmap="gray")
plt.title(f"pred={pred}, true={true}")
plt.axis("off")
plt.show()
```

### 6.2 Why is this step important?

Because often:

- The metrics look good
- But you do not know what the model is actually “looking at”

Looking at a few prediction examples yourself helps you build intuition much faster.

---

## 7. How do you tell whether the model has really learned?

### 7.1 A few typical signs

If the model has really learned:

- train loss will go down
- val loss will usually go down too
- train / val acc will improve
- single-sample predictions will become more stable

### 7.2 A few typical issues

#### Both the training set and validation set are poor

Possible reasons:

- The model is too weak
- The learning rate is not appropriate
- There is a problem with how the data is created

#### The training set is good, but the validation set is poor

Possible reasons:

- Overfitting
- Too little data
- Too much noise

#### The loss does not change

Possible reasons:

- Shape mismatch
- Wrong labels
- Learning rate too small

---

## 8. What else is needed in a real image classification project?

We intentionally keep this teaching example very small.  
In real projects, you usually still need to add:

- DataLoader
- Data augmentation
- A more realistic dataset
- A stronger backbone
- More systematic validation metrics
- Model saving and loading

In other words:

> This section teaches the “complete closed loop,” not the final industrial-grade solution.

---

## 9. Common mistakes beginners make

### 9.1 Copying the model without checking data shape

In image tasks, shape is almost always the first thing to check.

### 9.2 Only watching train loss

Validation metrics are equally important.

### 9.3 Thinking the task is done once the model runs

In a real project, success is not just making it run; you also need to explain the results.

---

## Summary

The most important thing in this section is not simply getting CNN to run, but walking through the complete image classification loop:

> **Create data / organize data / define model / train / validate / predict on a single sample.**

Only when this chain is truly complete will you avoid feeling lost later when you switch to more complex datasets and stronger models.

---

## Exercises

1. Add a 4th image pattern to the current data, such as a “reverse diagonal.”
2. Increase the channel counts in `TinyCNNClassifier` and see whether the convergence speed changes.
3. Try adding `Dropout` and observe whether the validation performance becomes more stable.
4. Think about this: why are data creation and validation methods often more important than adding a few more network layers in image classification projects?

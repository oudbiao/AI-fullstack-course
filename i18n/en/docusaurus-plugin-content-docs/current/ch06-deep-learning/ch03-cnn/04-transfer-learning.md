---
title: "3.5 Transfer Learning 🔧"
sidebar_position: 4
description: "From why we don’t start training from scratch, to freezing the backbone, replacing the classification head, and progressively fine-tuning—truly understanding transfer learning in vision."
keywords: [transfer learning, fine-tuning, feature extractor, freeze backbone, transfer learning, CNN]
---

# Transfer Learning

:::tip Section Focus
If you already know that CNNs extract features and how classic architectures evolve, then the next very natural engineering question is:

> **When I build my own image task, do I really need to train an entire CNN from scratch?**

Most of the time, the answer is no.  
Transfer learning answers this question: how do we borrow visual knowledge learned on other tasks?
:::

## Learning Objectives

- Understand why transfer learning is often more practical than training from scratch for image tasks
- Distinguish between two common approaches: “fixed feature extractor” and “fine-tuning”
- Learn how to replace the classification head and freeze backbone parameters
- Read a small but fully runnable transfer learning example
- Understand when to train only the head, and when to unfreeze more layers

---

## 1. Why has transfer learning become the default option for vision tasks?

### 1.1 How expensive is training from scratch?

If you want to train a decent vision model from scratch, you will usually run into these problems:

- Not enough data
- High labeling cost
- Long training time
- Easy to overfit

For example, suppose you only have 2,000 images and want to classify 5 classes.  
That is not especially small in a real project, but for training a deep CNN from scratch, it still may not be stable enough.

### 1.2 What exactly has a pretrained model “pretrained”?

A model trained on large-scale image data has usually already learned many general visual features:

- Edges
- Textures
- Color combinations
- Part shapes
- Common object patterns

These capabilities are not exclusive to “cat and dog tasks”; they are basic visual knowledge useful for many image tasks.

So the core intuition of transfer learning is:

> **First reuse the low-level visual capabilities already learned, then adapt the last few layers to fit your own task.**

### 1.3 A helpful analogy

Transfer learning is like asking someone who already knows general drawing skills to help create professional illustrations:

- They do not need to relearn how to hold a pen
- You only need them to adapt to your specific style and subject

That is why transfer learning is usually such a good deal in vision tasks.

---

## 2. The two most common transfer learning approaches

### 2.1 Approach 1: Fixed feature extractor

Method:

- Keep the pretrained backbone parameters unchanged
- Train only the final classification head

Advantages:

- Fast
- Less likely to damage pretrained capabilities
- Suitable for very small datasets

Disadvantages:

- Limited ability to adapt to a new task

### 2.2 Approach 2: Fine-tuning

Method:

- Replace the final classification head
- In addition to the head, gradually unfreeze part or even all of the backbone

Advantages:

- Better adaptation to the target task

Disadvantages:

- Easier to overfit
- Slower training
- Requires more careful learning rate choices

### 2.3 One-line memory trick

- Small dataset: prefer a fixed feature extractor first
- Large dataset / big task difference: consider progressive fine-tuning

![Decision diagram for freezing the backbone and progressive fine-tuning in transfer learning](/img/course/ch06-transfer-learning-freeze-finetune-map.png)

:::tip Reading hint
When reading this diagram, first ask two questions: how much data do you have, and how similar is the new task to the pretrained task. If the data is small and the tasks are similar, freeze the backbone and train only the head first; if you have more data or the task is more different, gradually unfreeze later layers and fine-tune with a smaller learning rate.
:::

---

## 3. A “directly runnable” toy transfer learning example

To make sure the code runs without downloading any external model, we will simulate a “pretrained backbone” ourselves.

### 3.1 First define a small backbone

```python
import torch
from torch import nn

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        return x.flatten(1)
```

The output of this backbone is a fixed-length feature vector.  
This is very similar to the “feature output from the backbone” in many real pretrained models.

---

## 4. First build the “fixed feature extractor” version

### 4.1 Replace the classification head and freeze the backbone

```python
import torch
from torch import nn

class TransferClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = nn.Linear(16, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

model = TransferClassifier(num_classes=3)

# Freeze the backbone
for param in model.backbone.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    print(name, "trainable =", param.requires_grad)
```

### 4.2 What should you see in the output?

You will find that:

- All parameters in `backbone` are not trainable
- Only the parameters in `head` are trainable

This is the standard “train only the head” form of transfer learning.

---

## 5. Build a small image classification task that can actually be trained

### 5.1 Use synthetic data to simulate a small task

We create 3 simple image classes:

- Vertical line
- Horizontal line
- Diagonal line

This way, we do not need an external dataset, and we can still complete the training loop.

```python
import numpy as np
import torch

def make_image(label, size=12):
    img = np.zeros((size, size), dtype=np.float32)

    if label == 0:  # Vertical line
        img[:, size // 2] = 1.0
    elif label == 1:  # Horizontal line
        img[size // 2, :] = 1.0
    else:  # Diagonal line
        for i in range(size):
            img[i, i] = 1.0

    img += np.random.randn(size, size).astype(np.float32) * 0.05
    return np.clip(img, 0.0, 1.0)

X, y = [], []
for label in range(3):
    for _ in range(80):
        X.append(make_image(label))
        y.append(label)

X = torch.tensor(np.array(X)).unsqueeze(1)
y = torch.tensor(np.array(y))

print(X.shape, y.shape)
```

---

## 6. Full training: train only the head

```python
import torch
from torch import nn

torch.manual_seed(42)

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.features(x).flatten(1)

class TransferClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = nn.Linear(16, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

model = TransferClassifier(num_classes=3)

# Freeze the backbone
for param in model.backbone.parameters():
    param.requires_grad = False

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.head.parameters(), lr=0.05)

for epoch in range(80):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        acc = (pred.argmax(dim=1) == y).float().mean().item()
        print(f"epoch={epoch:3d}, loss={loss.item():.4f}, acc={acc:.3f}")
```

### 6.2 What is this code really teaching you?

Not the syntax of “freezing parameters” itself, but rather:

> In transfer learning, the first step is often not retraining the entire model. Instead, you first check whether the existing features are already enough to support your task.

---

## 7. When should you fine-tune further?

### 7.1 A very common next step

If training only the head is not good enough, you can consider:

- Unfreezing the last convolution block
- Continuing training with a smaller learning rate

### 7.2 A minimal fine-tuning example

```python
# Unfreeze the last convolution layer
for param in model.backbone.features[3].parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.005
)

for epoch in range(40):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        acc = (pred.argmax(dim=1) == y).float().mean().item()
        print(f"finetune epoch={epoch:3d}, loss={loss.item():.4f}, acc={acc:.3f}")
```

### 7.3 Why do we usually use a smaller learning rate for fine-tuning?

Because the backbone already has a set of features learned from before.  
If the learning rate is too large, it is easy to destroy those already good representations.

So a common rule of thumb is:

- Use a larger learning rate for the head
- Use a smaller learning rate for the backbone

---

## 8. How is transfer learning usually done in real projects?

### 8.1 The most common workflow

1. Choose a pretrained backbone
2. Replace the final classification head
3. Train only the head first
4. If results are not good enough, gradually unfreeze layers
5. Keep watching validation performance

### 8.2 Why is this workflow so popular?

Because it balances:

- Training speed
- Stability
- Final performance

It is usually much more reliable than “train everything from the beginning.”

---

## 9. Common mistakes beginners make

### 9.1 Thinking transfer learning just means “copy a big model”

What really matters is:

- Which layers are frozen
- Which layers are unfrozen
- How the learning rates are set

### 9.2 Fine-tuning everything right away

This is often both slow and unstable, especially for small-data tasks.

### 9.3 Forgetting to check which parameters are being trained

This is a very common mistake.  
Before training, it is best to print the `requires_grad` status once.

---

## Summary

The most important thing in this section is not memorizing the words “transfer learning,” but building a stable engineering intuition:

> **First reuse the general features already learned by a pretrained model, then decide whether to train the head, part of the network, or the whole model based on your task.**

That is also why in many real-world vision projects, transfer learning is not just a trick—it is almost the default starting point.

---

## Exercises

1. Expand the number of classes in the example from 3 to 4, and design a new image pattern.
2. Compare the training curves for “train only the head” and “unfreeze one more layer.”
3. Print `requires_grad` for all model parameters to make sure you really know which layers are training.
4. Think about this: if your target task is very different from the original pretrained task, why might you need to unfreeze more layers?

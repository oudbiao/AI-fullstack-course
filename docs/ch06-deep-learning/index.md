---
title: "6 Deep Learning and Transformer Basics"
sidebar_position: 0
description: "Learn the practical deep learning loop: tensors, model, loss, backpropagation, optimizer, curves, CNN/RNN/Transformer, and small projects."
keywords: [deep learning, PyTorch, neural network, CNN, RNN, Transformer, Attention]
---

# 6 Deep Learning and Transformer Basics

![Main visual for Deep Learning and Transformer](/img/course/ch06-deep-learning-en.png)

Chapter 6 has one job: help you understand **how a model learns from loss, gradients, and repeated training steps**.

## See The Training Loop

![Main diagram of the deep learning training loop](/img/course/ch06-training-loop-backbone-en.png)

Read the picture first. Most deep learning training code is this loop:

```text
batch data -> model forward -> loss -> backward gradients -> optimizer step -> curves
```

Do not start by chasing large models. First make a small model train, log what happened, and explain why it improved or failed.

## Learning Order And Task List

Use this table as both the chapter guide and the task sheet.

| Page | Follow-along action | Evidence to keep |
|---|---|---|
| [6.1 Neural Network Basics](ch01-nn-basics/00-roadmap.md) | Understand neurons, activations, forward/backward pass, optimizers, regularization, and initialization | One hand-written training-loop explanation |
| [6.1.2 DL History](ch01-nn-basics/06-history-breakthroughs.md) | Optional background: skim why backprop, CNN, RNN, Attention, and Transformer appeared | A short “why this architecture exists” note |
| [6.2 PyTorch](ch02-pytorch/00-roadmap.md) | Practice tensors, autograd, `nn.Module`, Dataset, DataLoader, and a minimal training loop | One runnable PyTorch script |
| [6.3 CNN](ch03-cnn/00-roadmap.md) | Use image classification to connect data shape, convolution, pooling, and transfer learning | Shape notes and one image-classification run |
| [6.4 RNN](ch04-rnn/00-roadmap.md) | Learn why sequence data needs memory and how LSTM/GRU helped before Transformer | One sequence-model note |
| [6.5 Transformer](ch05-transformer/00-roadmap.md) | Learn Query, Key, Value, self-attention, positional encoding, and Transformer blocks | One attention input/output diagram |
| [6.6 Generative Models](ch06-generative/00-roadmap.md) and [6.7 Training Tips](ch07-training-tips/00-roadmap.md) | Treat as extensions after the training loop is stable | One tuning or diagnosis note |
| [6.8 Projects](ch08-projects/00-roadmap.md) and [6.8.5 Workshop](ch08-projects/04-hands-on-dl-workshop.md) | Build a PyTorch evidence pack before larger image, sentiment, or generative projects | Logs, curves, checkpoint, shape trace, README |

Key terms for this chapter:

| Term | Meaning |
|---|---|
| `tensor` | Multi-dimensional array used by PyTorch |
| `forward` | Data passes through the model to produce predictions |
| `loss` | Number that measures prediction error |
| `backward` | Computes gradients from the loss |
| `optimizer` | Updates parameters using gradients |
| `epoch` | One pass through the training data |
| `batch` | A small group of samples processed together |

## First Runnable Loop

Install PyTorch from the official selector if needed, then run this tiny loop after PyTorch is available:

```python
import torch
from torch import nn

torch.manual_seed(42)
x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
y = torch.tensor([[0.0], [2.0], [4.0], [6.0]])

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(20):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch in {0, 1, 5, 19}:
        print(epoch, round(loss.item(), 4))
```

Expected shape:

```text
0 ...
1 ...
5 ...
19 ...
```

The exact numbers can differ, but the loss should generally move down. If it does, you have seen the training loop work.

## Common Failures

| Symptom | First thing to check | Usual fix |
|---|---|---|
| Shape mismatch | Input shape, batch dimension, output classes | Print tensor shapes at each layer |
| Loss does not decrease | Learning rate, labels, normalization, loss function | Try overfitting one small batch first |
| Train good, validation poor | Overfitting or bad split | Add validation curve, augmentation, regularization, early stopping |
| Out of memory | Batch size, image size, model size | Reduce batch/resolution or use a smaller model |
| Transformer feels abstract | Q/K/V and sequence length | Draw one attention table before code |

## Pass Check

Move to Chapter 7 when you can answer these five questions:

- What happens in `forward`, `loss.backward()`, and `optimizer.step()`?
- What problem do Dataset and DataLoader solve?
- How do training and validation curves reveal overfitting?
- Why can Attention model context?
- How does Transformer connect to later large models?

For a printable checklist, use [6.0 Study Guide and Task Sheet](./study-guide.md). Later LLMs, RAG, and multimodal models all build on these representation-learning ideas.

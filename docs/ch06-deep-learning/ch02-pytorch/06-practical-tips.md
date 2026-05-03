---
title: "2.8 Practical Tips"
sidebar_position: 6
description: "From device switching, random seeds, and AMP to gradient clipping and checkpoints, master the most common and practical engineering techniques in PyTorch training."
keywords: [PyTorch, AMP, mixed precision, gradient clipping, checkpoint, device, reproducibility]
---

# Practical Tips

## Learning Objectives

By the end of this section, you will be able to:

- Handle CPU / GPU device switching correctly
- Use random seeds to improve experiment reproducibility
- Understand the role of mixed precision training and gradient clipping
- Save and restore model checkpoints
- Build a PyTorch debugging checklist

---

## 1. Start by Solving the Most Common Engineering Problems

### 1.1 Device Switching: Don’t Assume You Always Have a GPU

Many beginners hard-code their code with `cuda()`, which immediately causes errors on machines without a GPU.

A safer way is:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Current device:", device)

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(device)
print(x)
print("Tensor device:", x.device)
```

You can think of `device` as “which workbench the training happens on”:

- CPU: a regular desk
- GPU: a large workbench for parallel computation

### 1.2 Fix the Random Seed: Make Experiments as Reproducible as Possible

When training is unstable, the first thing to do is often not to change the model, but to fix randomness first.

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

print(torch.randn(3))
set_seed(42)
print(torch.randn(3))
```

If the two outputs are the same, it means this part of the randomness has been fixed.

:::info Why “as much as possible” and not “absolutely”?
Some GPU operators and parallel execution details may still introduce tiny differences, so reproducibility is usually “closer to identical,” not “exactly identical.”
:::

---

## 2. Make the Training Process More Stable

### 2.1 `train()`, `eval()`, and `no_grad()` Should Become Muscle Memory

The easiest place to get confused during training and validation is not the model structure, but mode switching.

Standard practice:

```python
model.train()   # before training
...
model.eval()    # before validation / inference
with torch.no_grad():
    ...
```

You can think of it this way:

- `train()`: the model enters “practice mode”
- `eval()`: the model enters “exam mode”
- `no_grad()`: no need to draft backpropagation during the exam, which saves memory

### 2.2 Gradient Clipping: Prevent Gradients from Suddenly Exploding

In RNNs, Transformers, or deeper networks, gradients can sometimes become very large, making training unstable.
Gradient clipping is like “setting an upper limit for gradients.”

```python
import torch
from torch import nn

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

x = torch.randn(32, 10)
y = torch.randn(32, 1) * 50

loss_fn = nn.MSELoss()
pred = model(x)
loss = loss_fn(pred, y)
loss.backward()

def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return total ** 0.5

before = grad_norm(model)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
after = grad_norm(model)

print("Gradient norm before clipping:", round(before, 4))
print("Gradient norm after clipping:", round(after, 4))
```

It is like adding a speed limiter to a bicycle going downhill to prevent it from going too fast.

---

## 3. Make Training Faster

### 3.1 Mixed Precision Training (AMP): Less Memory, More Speed

The core idea of AMP is:

> Use lower precision in the right places to gain faster speed and lower memory usage.

It is especially suitable for GPU training.
To make sure the code below can run directly even on machines without a GPU, we write it so that it enables AMP when a GPU is available and trains normally otherwise.

```python
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

x = torch.randn(64, 16).to(device)
y = torch.randn(64, 1).to(device)

if device == "cuda":
    scaler = torch.amp.GradScaler("cuda")
    for _ in range(3):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            pred = model(x)
            loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print("Completed 3 training steps on GPU with AMP")
else:
    for _ in range(3):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    print("No GPU available; completed 3 training steps with standard precision")
```

### 3.2 What If the Batch Is Too Large?

If you often run out of memory:

1. First reduce `batch_size`
2. Then consider AMP
3. Then consider gradient accumulation

The intuition behind gradient accumulation is:

> Even if you cannot fit a large batch at once, you can eat it in several bites and then update the model once.

---

## 4. Save and Restore Training Progress

### 4.1 Why Are Checkpoints So Important?

Training can be interrupted for many reasons:

- Power outage
- Notebook timeout
- GPU being reclaimed
- Program error

A checkpoint is like a “game save file.”

### 4.2 A Minimal Runnable Example

```python
import torch
from torch import nn

model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

checkpoint_path = "demo_checkpoint.pt"

# Save
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": 5
}, checkpoint_path)

print("Checkpoint saved:", checkpoint_path)

# Restore
new_model = nn.Linear(2, 1)
new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)

ckpt = torch.load(checkpoint_path, map_location="cpu")
new_model.load_state_dict(ckpt["model_state_dict"])
new_optimizer.load_state_dict(ckpt["optimizer_state_dict"])

print("Restored epoch:", ckpt["epoch"])
```

In real projects, you usually also save:

- Best validation metric
- Training configuration
- tokenizer / label mapping

---

## 5. Where Should You Look When Debugging?

### 5.1 Shape Always Comes First

In PyTorch, many bugs are not really because “the model is too hard,” but because:

- the shape is wrong
- the dtype is wrong
- the device is inconsistent

Before training, it is a good idea to print a few more lines:

```python
print("x shape:", x.shape)
print("y shape:", y.shape)
print("x dtype:", x.dtype)
print("x device:", x.device)
```

### 5.2 What Is the Check Order When Training Does Not Decrease?

You can check in this order:

1. Whether the data was loaded correctly
2. Whether the labels are aligned correctly
3. Whether the loss is computed correctly
4. Whether `optimizer.zero_grad()` was written
5. Whether the order of `backward()` and `step()` is correct
6. Whether the learning rate is too large or too small

### 5.3 What Should You Do When You See `nan`?

Common causes include:

- Learning rate too large
- Input values too large
- Gradient explosion
- Numerical issues such as division by zero or `log(0)`

The most practical first response is:

1. Lower the learning rate
2. Print the loss and parameter ranges
3. Enable gradient clipping

---

## 6. A Training Template Worth Saving

```python
model.train()
for batch_x, batch_y in train_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    pred = model(batch_x)
    loss = loss_fn(pred, batch_y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

model.eval()
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        val_loss = loss_fn(pred, batch_y)
```

This template is not flashy, but it is very practical.

---

## Summary

The most important thing in this lesson is not a new API, but training engineering intuition:

- Do not hard-code the device
- Fix the random seed first
- Distinguish clearly between `train / eval / no_grad`
- Know how to clip large gradients
- Know how to save training progress

When many model training jobs get stuck, it is not because the algorithm is unknown, but because these “small engineering details” were not handled well.

---

## Exercises

1. Add `device` handling to your own PyTorch training code to make sure it can run on both CPU and GPU.
2. Add gradient clipping to your existing training loop and print the gradient norm before and after clipping.
3. Add a checkpoint saving mechanism and try restoring after an interruption.

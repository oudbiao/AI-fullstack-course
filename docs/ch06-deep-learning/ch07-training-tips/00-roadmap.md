---
title: "6.7.1 Training Tips Roadmap: Diagnose Before Changing Everything"
sidebar_position: 0
description: "A compact deep learning training tips roadmap: tuning, diagnosis, compression, and evidence-based decisions."
keywords: [deep learning training tips, hyperparameter tuning, training diagnosis, model compression]
---

# 6.7.1 Training Tips Roadmap: Diagnose Before Changing Everything

Training tips are useful only when they answer a diagnosis. Do not change optimizer, learning rate, model size, and data at the same time.

## Look at the Diagnosis Flow First

![Deep learning training tips chapter relationship diagram](/img/course/ch06-training-tips-chapter-flow-en.webp)

![Training diagnosis dashboard map](/img/course/ch06-training-diagnosis-dashboard-map-en.webp)

| Symptom | First check |
|---|---|
| training loss high | model too small, learning rate too low, bad data |
| training good, validation bad | overfitting, leakage, weak augmentation |
| unstable loss | learning rate too high, bad batch, exploding gradients |
| too slow | batch size, device, model size |
| too heavy to deploy | compression, quantization, pruning |

## Read a Tiny Loss Log

Create `training_tips_first_loop.py`.

```python
val_loss = [0.62, 0.51, 0.48, 0.49, 0.53]
best_epoch = min(range(len(val_loss)), key=val_loss.__getitem__) + 1

print("best_epoch:", best_epoch)
print("best_val_loss:", val_loss[best_epoch - 1])
print("action: stop or reduce learning rate if validation keeps worsening")
```

Expected output:

```text
best_epoch: 3
best_val_loss: 0.48
action: stop or reduce learning rate if validation keeps worsening
```

![Training tips first loss output result map](/img/course/ch06-training-tips-first-loop-result-map-en.webp)

Before adding tricks, read the curve. A simple log often tells you what to try next.

## Learn in This Order

| Order | Read | What to practice |
|---|---|---|
| 1 | [6.7.2 Hyperparameter Tuning](./01-hyperparameter-tuning.md) | learning rate, batch size, optimizer |
| 2 | [6.7.3 Training Diagnosis](./02-training-diagnosis.md) | loss curves, overfitting, instability |
| 3 | [6.7.4 Model Compression](./03-model-compression.md) | smaller, faster, deployable models |

## Pass Check

You pass this roadmap when you can look at a training/validation curve and choose one next action with a reason.

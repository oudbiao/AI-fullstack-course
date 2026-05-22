---
title: "6.7.1 Training Tips Roadmap: Diagnose Before Changing Everything"
description: "A compact deep learning training tips roadmap: tuning, diagnosis, compression, and evidence-based decisions."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "deep learning training tips, hyperparameter tuning, training diagnosis, model compression"
---
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

## Evidence to Keep

After this mini-chapter, keep one diagnosis decision record:

```text
visible_symptom: what did the curve or output show?
first_check: data, shape, gradient, or validation split
one_change: which single setting changed?
before_after: metric or artifact comparison
decision: keep, tune, rollback, or investigate
```

The point is to make training changes reversible. If you change five things and the run improves, you still do not know which change helped.

## Learn in This Order

| Order | Read | What to practice |
|---|---|---|
| 1 | [6.7.2 Hyperparameter Tuning](/ch06-deep-learning/ch07-training-tips/01-hyperparameter-tuning/) | learning rate, batch size, optimizer |
| 2 | [6.7.3 Training Diagnosis](/ch06-deep-learning/ch07-training-tips/02-training-diagnosis/) | loss curves, overfitting, instability |
| 3 | [6.7.4 Model Compression](/ch06-deep-learning/ch07-training-tips/03-model-compression/) | smaller, faster, deployable models |

## Pass Check

You pass this roadmap when you can look at a training/validation curve and choose one next action with a reason.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer connects tensors, model layers, loss, `backward()`, and optimizer updates into one training loop.
2. The evidence should include a runnable mini experiment, tensor-shape checks, and a loss or validation curve you can explain.
3. A good self-check names one failure mode such as shape mismatch, no loss decrease, overfitting, data leakage, or using Attention/Transformer words without explaining the data flow.

</details>

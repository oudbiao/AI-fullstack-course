---
title: "6.0 Study Guide and Task Sheet: Deep Learning and Transformer Basics"
sidebar_position: 1
description: "A short printable checklist for Chapter 6 after the main guide has been merged into the chapter entry page."
keywords: [deep learning study guide, PyTorch, CNN, Transformer, Attention]
---

# 6.0 Study Guide and Task Sheet: Deep Learning and Transformer Basics

![Deep learning study guide training loop](/img/course/ch06-study-guide-training-loop-en.webp)

The main study route is now in [Chapter 6 entry](./). Use this page only as a quick checklist while you practice.

## One-Line Mental Model

```text
batch data -> model forward -> loss -> backward gradients -> optimizer step -> curves
```

If the code feels long, find these six steps first.

## Practice Checklist

| Check | Evidence |
|---|---|
| I can explain forward, loss, backward, optimizer | training-loop note |
| I can run a minimal PyTorch script | `train.py` |
| I can print tensor shapes through a model | shape trace |
| I can compare training and validation curves | curve image or CSV |
| I can explain what Attention changes | attention note |
| I can finish the evidence-pack workshop | `deep_learning_workshop_run/` |

## Ready To Continue

Continue to Chapter 7 when you can train one small model, save the training log, inspect failure cases, and explain why the model improved or failed.

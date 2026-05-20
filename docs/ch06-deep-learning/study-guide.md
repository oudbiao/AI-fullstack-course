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

## Expected Final Output

By the end of Chapter 6, your visible output should be a small evidence folder, not only finished reading notes:

```text
deep_learning_evidence/
  shape_trace.txt
  training_log.csv
  loss_curve.png
  best_checkpoint_note.md
  attention_note.md
  failure_sample_note.md
```

If this folder is missing, the chapter is not finished yet, even if every page has been read.

## Practice Checklist

| Check | Evidence |
|---|---|
| I can explain forward, loss, backward, optimizer | training-loop note |
| I can run a minimal PyTorch script | `train.py` |
| I can print tensor shapes through a model | shape trace |
| I can compare training and validation curves | curve image or CSV |
| I can explain what Attention changes | attention note |
| I can finish the evidence-pack workshop | `deep_learning_workshop_run/` |

<details>
<summary>Reference answers and explanation</summary>

Use this checklist as a self-review rubric:

1. A training-loop note should explain forward pass, loss, backward pass, and optimizer step without copying code line by line.
2. A valid PyTorch script should run from a clean folder and print at least one shape, loss, or metric that proves it executed.
3. A useful shape trace should include batch size, channel/feature dimensions, and the point where tensors enter the classifier or loss.
4. A curve artifact should support a diagnosis: improving, underfitting, overfitting, unstable, or unclear.
5. An attention note should explain what Q/K/V and masking change compared with earlier sequence models.
6. A finished evidence pack should be rerunnable and should contain enough artifacts for someone else to understand the result.

</details>

## Evidence Rubric

| Artifact | It should answer |
|---|---|
| Training-loop note | What happens in forward, loss, backward, and optimizer step? |
| Shape trace | How do tensor shapes change through the model? |
| Curve image or CSV | Is the model underfitting, overfitting, or improving steadily? |
| Attention note | What information does attention add, and what remains hard? |
| Failure sample note | Which sample fails, and what does that tell you about data, model, or labels? |

## Evidence to Keep

Before leaving Chapter 6, keep one compact evidence pack:

```text
shape_trace: one model with printed tensor shapes
training_log: train and validation loss over time
best_checkpoint: how the best model was selected
attention_note: Q/K/V, mask, and next-token bridge
failure_sample: one wrong or weak prediction with next action
project_folder: runnable evidence pack or README
```

## Ready To Continue

Continue to Chapter 7 when you can train one small model, save the training log, inspect failure cases, and explain why the model improved or failed.

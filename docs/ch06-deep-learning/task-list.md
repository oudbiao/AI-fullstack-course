---
title: "Phase Learning Task Sheet"
description: "Break the deep learning and Transformer basics phase into actionable learning tasks, practice deliverables, and completion criteria."
keywords: [deep learning, PyTorch, Transformer, CNN, learning task sheet]
---

# Phase Learning Task Sheet: Deep Learning and Transformer Basics

The goal of this phase is to help you understand how neural networks are trained, how to build models with PyTorch, and why Transformer has become the foundation of modern large models. Don’t rush into large model training yet. First, make sure you understand tensors, models, losses, optimizers, training loops, and evaluation workflows.

## Required tasks for this phase

| Task | Deliverable | Passing criteria |
| --- | --- | --- |
| Understand the neural network training loop | A hand-drawn training flowchart | Can explain forward propagation, loss, backpropagation, and parameter updates |
| Get PyTorch basics working | A minimal training script | Can use Dataset, DataLoader, nn.Module, and optimizer |
| Complete the guided PyTorch evidence workshop | A generated `deep_learning_workshop_run/` evidence pack | Can rerun the script and explain `training_log.csv`, `model_comparison.csv`, `loss_curve.png`, and `shape_trace.md` |
| Complete a small image or text task | A runnable training project | Can record training curves, validation metrics, and failure cases |
| Understand Attention and Transformer | A notes document explaining the structure | Can explain Query, Key, Value, Self-Attention, and positional encoding |
| Complete the phase project | A deep learning practice project | Includes training logs, metrics, reproducible commands, and reflections |

## Recommended learning order

First understand the neural network training process, then learn PyTorch basics, and then learn CNN/RNN/Attention/Transformer. Don’t treat Transformer as an isolated formula. At its core, it handles sequence information, contextual relationships, and parallel computing efficiency.

When you need a concrete runnable checkpoint, use [Hands-on Workshop: Build a PyTorch Training Evidence Pack](./ch08-projects/04-hands-on-dl-workshop.md) before starting a larger project.

When writing PyTorch code, pay close attention to data shapes. Most beginner mistakes are related to tensor shapes, batch dimensions, loss input formats, and device mismatches. Every time you write a module, it is a good idea to print the input and output shapes once.

## Relationship to the AI Learning Assistant project

This phase corresponds to version v0.5 of the AI Learning Assistant, the understanding-and-learning stage. You do not have to train a large model for the learning assistant, but you should understand the basics of embedding, sequence modeling, and Transformer, since this will directly affect your later understanding of RAG, Prompt, fine-tuning, and Agent.

It is recommended to do a small experiment: use a simple text classification or similarity task to observe how different text representation methods perform. The key is not to chase high scores, but to understand “how text becomes vectors” and “why vector similarity can be used for retrieval.”

## Common sticking points

Common problems include mismatched tensor dimensions, training loss not decreasing, poor validation performance, learning rates that are too large or too small, overfitting, CPU/GPU device mismatches, and mistaking training metrics for generalization ability. When you run into training issues, first try an overfitting test on a small dataset to confirm that the code can actually learn, then scale up the data.

## Easy / Standard / Challenge tasks

| Difficulty | What you need to complete | Who it is for |
|---|---|---|
| Easy | Get a minimal training loop running | First-time learners, learners with limited time, or complete beginners |
| Standard | Save training curves and validation metrics | Learners who want to include this phase in their portfolio |
| Challenge | Create and fix a shape mismatch or a loss-not-decreasing issue | Learners with some foundation who want stronger project evidence |

## Phase badge and boss fight

| Type | Content |
|---|---|
| Boss fight | Shape Beast |
| Unlockable badges | Loss Observer, Shape Tracker |
| Minimum clear slogan | Get it running first, then explain it, then record the failures |
| Evidence-saving suggestion | Save screenshots, logs, failed cases, or evaluation tables in `reports/`, `evals/`, or `logs/` |

Once you complete the easy version, you can move forward. Only after completing the standard version is it recommended to include it in your portfolio. Do the challenge version only if you have extra capacity.

## Phase portfolio deliverables

If you want to turn this phase into portfolio material, it is recommended to keep at least the following files or equivalent materials.

| Deliverable | Description |
| --- | --- |
| `train.py` | A PyTorch training script containing Dataset, DataLoader, model, loss, and optimizer |
| `config.yaml` | Experiment configuration such as learning rate, batch size, epoch, and model structure |
| `training_log.csv` | Loss, metrics, time cost, and validation results for each epoch |
| `curves/` | Training curves, validation curves, confusion matrices, or prediction visualizations |
| `failure_cases.md` | Error samples, overfitting/underfitting patterns, and improvement actions |
| `README.md` | Data description, run commands, model results, limitations, and reflections |

These materials will upgrade your deep learning project from “the training runs” to “the training process can be diagnosed, experiments can be reproduced, and model failures can be explained.”

## Phase completion questions

After finishing this phase, you should be able to answer these questions: Why do we need backpropagation? What does the optimizer update? What problems do Dataset and DataLoader solve respectively? Why can Attention model context? What is the relationship between Transformer and later large models?

## Completion checklist

- [ ] I can explain forward propagation, loss, backpropagation, and parameter updates.
- [ ] I can write Dataset, DataLoader, nn.Module, and a training loop in PyTorch.
- [ ] I can record training curves, validation metrics, and failure cases.
- [ ] I can explain what problems Attention and Transformer solve.
- [ ] I have completed a small deep learning project and can explain its inputs, outputs, and failure reasons.

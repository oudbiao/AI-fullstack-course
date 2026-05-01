---
title: "6 Deep Learning and Transformer Basics"
sidebar_position: 0
description: "Learn neural networks, PyTorch, training loops, CNN, RNN, Attention, and Transformer to build a deep learning foundation for moving into large models and multimodal directions."
keywords: [deep learning, PyTorch, neural network, CNN, RNN, Transformer, Attention]
---

# 6 Deep Learning and Transformer Basics

![Main visual for Deep Learning and Transformer](/img/course/ch06-deep-learning.png)

This stage is about answering the question: “How does the model actually learn internally?” In the machine learning stage, you mostly use ready-made model interfaces. In the deep learning stage, you will truly see parameters, gradients, training loops, network structures, and representation learning.

## Story-Style Introduction: Enter the Model’s Engine Room

If the machine learning stage is like driving a car, the deep learning stage is like opening the hood and seeing how the power is produced. Tensors are the fuel, network layers are the parts, the loss function is like the dashboard, and gradients and the optimizer keep tuning the system. For the first time, you will really see how a model learns from mistakes.

## Learning Quest Map

![Deep learning learning quest map](/img/course/ch06-learning-quest-map.png)

## Interactive Exercise: Watch the Four Numbers in the Training Loop

When training a small model, do not only look at the final accuracy. In every experiment, observe training loss, validation loss, training accuracy, and validation accuracy. If the training set keeps improving but the validation set does not, it may be overfitting; if both are poor, there may be a problem with the model, the data, or the learning rate. Understanding these curves is more important than blindly switching models.

## Project Easter Egg

The easter egg project for this stage can be a “small model lab”: on the same dataset, record the effects of different network structures, learning rates, batch sizes, and numbers of training epochs, and draw comparison curves. This lab will become a basic template for understanding fine-tuning, large model training, and multimodal models.

## Stage Positioning

| Information | Description |
|---|---|
| Suitable for | Learners who have completed machine learning and want to move into deep learning, Transformer, large models, or multimodal directions |
| Estimated time | 140–190 hours |
| Prerequisites | Complete the first four stages |
| Stage output | Image classification, text sentiment classification, or a simple generative model project |

## Beginner’s Minimum Completion Path

Beginners should first get through the closed training loop of tensors, automatic differentiation, data loading, model definition, loss computation, backpropagation, and optimizer updates. As long as you can use PyTorch to train a small classification model and understand the training loss and validation metrics, you have completed the minimum path.

## Advanced Learning Path

Experienced learners can go deeper into CNN, RNN, Attention, Transformer, regularization, initialization, and training diagnostics. You can further try recording experimental results for different network structures and hyperparameters to form your own small model lab.

## Why Deep Learning Matters

Deep learning allows models to automatically learn complex representations from data. Edges, textures, and objects in images, as well as word meaning and context in text, can all be represented through multi-layer networks step by step. Transformer has further become the core architecture of large language models and multimodal models.

![Main diagram of the deep learning training loop](/img/course/ch06-training-loop-backbone.png)

If you want to understand each technical breakthrough in historical order, you can first read [1.2 Main Thread of Deep Learning Historical Breakthroughs](./ch01-nn-basics/06-history-breakthroughs.md). It will map the perceptron, XOR setbacks, backpropagation, vanishing gradients, LSTM, RBM/DBN, AlexNet, ResNet, Attention, and Transformer to the corresponding subsections in this chapter, helping you understand why models keep evolving generation after generation.

## What Beginners Should Do First, and What Advanced Learners Should Do Later

When beginners learn this stage for the first time, they should first understand the minimum closed loop of neural network training: prepare data, define the model, compute the loss, backpropagate, update parameters, and observe the curves. Do not start by chasing complex architectures.

Experienced learners can focus on training diagnostics: how to detect overfitting, how learning rate affects the curve, when data augmentation and regularization are useful, and why Transformer changed sequence modeling. Your goal is to be able to explain why a training run succeeded or failed.

## Learning Path for This Stage

Chapter 1 covers neural network basics and the main thread of historical breakthroughs. You will understand perceptrons, XOR limitations, backpropagation, neurons, activation functions, forward propagation, backpropagation, optimizers, regularization, and parameter initialization.

Chapter 2 covers PyTorch. You will start with tensors, automatic differentiation, `nn.Module`, data loading, and the training loop to build a truly trainable model.

Chapter 3 covers CNN. Visual tasks are the most intuitive and are a good entry point for understanding deep network structures for the first time.

Chapter 4 covers RNNs and sequence models. You will see why sequence data is different from ordinary tabular data, and you will also understand the historical significance of LSTM and GRU.

Chapter 5 covers Attention and Transformer. It is the key bridge to the later main line of large models.

Chapters 6 and 7 are extensions that help you understand generative models and training tuning.

## What You Should Be Able to Do After Learning This Stage

- Explain forward propagation, loss computation, and parameter updates in neural networks
- Write a minimal training loop in PyTorch
- Train a simple CNN or text classification model
- Understand the basic differences between RNN, Attention, and Transformer
- Build a foundation for later learning about LLM principles, fine-tuning, and multimodal learning

## Common Misconceptions

Do not just copy training code without knowing what each step does. At the very least, you should be able to explain how data enters the model, how the output loss is computed, how gradients are backpropagated, and how the optimizer updates the parameters.

Also, do not start by trying to train large models. The most important thing in the first stop of deep learning is to get a small model and a small dataset working and to understand the training loop clearly.

## Training Failure Theater: Curves Matter More Than Model Names

If the loss does not decrease, first check whether the learning rate, label format, input normalization, and loss function match; if the training set looks good but the validation set is poor, suspect overfitting and data splitting issues first; if you run out of GPU memory, first reduce the batch size, image size, or model scale.

## Minimum Runnable Experiment: A Complete PyTorch Training Loop

The minimum experiment in this stage is not training a large model, but getting a training loop working with a small dataset, a small model, and a small number of epochs. You should be able to point out each step: data enters the model, the output computes the loss, the loss is backpropagated, and the optimizer updates the parameters.

```python
for x, y in dataloader:
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
```

If you can plot the training loss and validation loss, and explain why they change, then you have already grasped the main thread of deep learning engineering.

## Deep Learning Failure Case Library: First Check Shape, Loss, and Curves

| Phenomenon | Common Cause | Diagnosis Method | Fix Direction |
|---|---|---|---|
| shape mismatch | Input dimensions, batch dimension, or number of classes do not match | Print the input and output shape of each layer | Adjust reshape, the model head, or the data format |
| loss does not decrease | Learning rate, label format, normalization, or loss mismatch | First try an overfitting test on a small batch | Tune the learning rate, check labels and input scale |
| good training, poor validation | Overfitting, unreasonable data split | Compare training and validation curves | Data augmentation, regularization, early stopping |
| out of memory | Batch size, image size, or model too large | Check memory usage | Reduce the batch size, lower the resolution, switch to a lighter model |

## Stage Acceptance Rubric

| Level | Acceptance Criteria | Portfolio Evidence |
|---|---|---|
| Minimum pass | Can run Dataset, DataLoader, model, loss, and optimizer | `train.py`, training output |
| Recommended pass | Can record training curves and explain overfitting/underfitting | Curve plots, validation metrics, config files |
| Portfolio pass | Can compare model approaches and analyze failed samples | Experiment report, error samples, improvement plan |

## Stage Projects

The basic version is to train a simple image classification or text sentiment classification model that can complete data loading, training, and evaluation. The standard version should add a validation set, metric curves, overfitting analysis, and model save/load. The challenge version can compare CNN, RNN, Transformer, or transfer learning approaches, and write an experiment report explaining why the model got better or worse.

If you want a more detailed learning rhythm, you can read [Study Guide: The Easiest Way to Learn Deep Learning Basics Without Getting Confused](./study-guide.md).




## Fun Task Card for This Stage

| Play Style | Task for This Stage |
|---|---|
| Story quest | Help the assistant understand the training process: run the training loop, observe the loss curve, and locate the shape mismatch. |
| Boss battle | **Shape Beast** |
| Unlockable badges | Loss Observer, Shape Tracker |
| Easy beginner mode | Only complete one minimal input-to-output loop, and keep a screenshot or command output |
| Portfolio evidence | Training logs, curves, and one failure review |

If you feel that this stage has a lot of content, first use this task card as your minimum goal. Once you can complete the easy beginner mode, you can keep learning forward; later, when preparing your portfolio, come back and upgrade to the standard and challenge versions.

## Stage Deliverables

| Deliverable | Minimum Version | Portfolio Version |
|---|---|---|
| Training script | Runs data loading, forward propagation, loss, and optimization | Clear structure, supports config parameters, model saving, and experiment reproducibility |
| Metric curves | Records loss and accuracy | Shows training/validation curves, overfitting judgment, and tuning process |
| Model comparison | Compares a baseline model and an improved model | Explains the trade-offs of CNN, RNN, Transformer, or transfer learning |
| Failed samples | Saves several misclassified samples | Analyzes data quality, class confusion, augmentation strategy, and model limitations |
| Experiment report | Clearly writes down the run commands and results | Includes data, model, metrics, curves, error analysis, and next steps |

## Relationship with the AI Learning Assistant Capstone Project

This stage can correspond to AI Learning Assistant v0.6: build a small text or image classification experiment, and record training curves, metrics, and failed samples. If you are learning along the capstone project path, it is recommended that by the end of this stage you submit at least one version note: what new capability was added in this stage, how to run it, what the sample input/output is, what problems were encountered, and what to improve next.

## Stage Completion Criteria

| Completion Level | What You Need to Be Able to Do |
|---|---|
| Minimum pass | Run the training loop with PyTorch, and understand CNN, RNN, Transformer, and training diagnostics. |
| Recommended pass | Complete at least one runnable small project in this stage, and record the run method, sample input/output, and issues encountered in the README. |
| Portfolio pass | Connect the outputs of this stage to the “AI Learning Assistant” capstone project, and leave screenshots, logs, evaluation samples, and a next-step plan. |

After finishing this stage, you do not need to memorize every detail. More importantly, you should be able to clearly explain: what problem this stage solves, how it relates to the previous stage, and how it will support later learning. Large models, RAG, and multimodal models later on will all be built on these representation learning concepts.

---
title: "Study Guide: How to Learn Deep Learning and Transformer Basics Without Getting Confused"
sidebar_position: 1
description: "A deep learning study guide for AI full-stack beginners: neural networks, PyTorch, CNN, RNN, Attention, Transformer, project roadmap, and acceptance criteria."
keywords: [deep learning study guide, how to learn PyTorch, how to learn CNN, how to learn Transformer, Attention]
---

# Study Guide: How to Learn Deep Learning and Transformer Basics Without Getting Confused

If you reach `Chapter 6 Deep Learning and Transformer Basics` and feel that the code is getting longer and the models are getting more numerous, first bring your attention back to the training loop. On your first pass through deep learning, the most important thing is to understand how data flows through the model, how the loss is computed, and how gradients update parameters.

## Core principle for this stage

Deep learning follows one main thread: data enters the network, forward propagation produces outputs, the loss function measures the gap, backpropagation computes gradients, and the optimizer updates parameters.

![Deep learning study guide training loop](/img/course/ch06-study-guide-training-loop-en.png)

## Recommended learning order

In the first round, learn the historical breakthroughs and neural network basics. Focus on understanding why the perceptron appeared, why XOR challenged single-layer models, and why backpropagation matters. Then understand neurons, activation functions, forward propagation, backpropagation, loss functions, optimizers, and regularization.

In the second round, learn PyTorch. Do not just copy code; understand what tensors, automatic differentiation, `nn.Module`, Dataset, DataLoader, and the training loop each do.

In the third round, learn CNNs. Image classification is the most intuitive and is a good way to connect network structures with tasks for the first time.

In the fourth round, learn RNNs and sequence models. They will help you understand sequence tasks and also provide historical context for the emergence of Transformer.

In the fifth round, learn Attention and Transformer. This is the most important bridge before entering the main line of large models.

Generative models and training techniques can be treated as extensions; you do not need to fully master them in the first pass.

Before larger projects, complete [Hands-on Workshop: Build a PyTorch Training Evidence Pack](./ch08-projects/04-hands-on-dl-workshop.md). It turns the abstract training loop into one runnable script plus logs, curves, checkpoint, shape trace, and review samples.

## Suggested learning pace

| Content type | Suggested time | Learning goal |
|---|---|---|
| Neural network basics | 3–6 hours | Be able to explain the training loop |
| PyTorch basics | 6–10 hours | Be able to write a minimal training loop |
| CNN / RNN | 4–8 hours | Be able to understand which network fits which data structure |
| Transformer | 4–8 hours | Be able to explain the basic intuition of Attention |
| Project section | 10–20 hours | Complete a small model that can be trained and evaluated |

## Stage project roadmap

Before choosing a larger topic, run the [PyTorch evidence-pack workshop](./ch08-projects/04-hands-on-dl-workshop.md) once. Treat it as a warm-up: you will create data, trace shapes, train a baseline, train a CNN, validate, save curves, and write project evidence.

For your first project, I recommend handwritten digit recognition or a small image classification task to practice Dataset, DataLoader, CNN, training, and evaluation.

For your second project, I recommend text sentiment classification to practice sequence inputs, Embedding, and basic text models.

For your third project, you can do a Transformer architecture reading exercise or a small experiment, with a focus on understanding the input/output of Attention and context modeling.

## Common sticking points

The most common sticking point is not being able to connect loss, gradients, and the optimizer. You can use a very small model and a few samples to print the input, output, loss, and parameter changes at each step.

The second sticking point is that the PyTorch code template feels too long. It is better to start by writing the minimal training loop and then gradually wrap functions; do not pursue engineering polish from the beginning.

The third sticking point is poor model performance. First check the data, labels, learning rate, and whether the loss is decreasing, then consider changing the model.

## Passing criteria

After completing this stage, you should be able to write a PyTorch training script from scratch, train a simple model, plot the loss curve, and explain why the model updates in that way.

If you can clearly explain what problems CNNs, RNNs, and Transformer each solve, you are ready to move on to the principles of large models.

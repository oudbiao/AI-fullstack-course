---
title: "6.1.1 Neural Network Roadmap: Linear Layer, Activation, Loss, Update"
sidebar_position: 0
description: "A compact neural network basics roadmap: neuron, activation, forward pass, loss, backward pass, optimizer, and regularization."
keywords: [neural network guide, deep learning basics, activation function, backpropagation, optimizer]
---

# 6.1.1 Neural Network Roadmap: Linear Layer, Activation, Loss, Update

Neural networks are not magic. A layer first does a weighted sum, then an activation changes the shape of the signal, then training adjusts weights to reduce loss.

## Look at the Flow First

![Neural network basics chapter relationship diagram](/img/course/ch06-nn-basics-chapter-flow-en.webp)

Keep this loop:

```text
input -> weighted sum -> activation -> loss -> gradient -> update weights
```

| Word | First meaning |
|---|---|
| neuron | weighted sum plus bias |
| activation | nonlinearity such as ReLU |
| forward pass | compute prediction |
| backward pass | compute responsibility for error |
| optimizer | update weights using gradients |

## Run One Neuron

Create `nn_first_loop.py` and run it after installing `torch`.

```python
import torch

x = torch.tensor([[1.0, -2.0, 3.0]])
weights = torch.tensor([[0.5], [-1.0], [0.25]])
bias = torch.tensor([0.1])

linear_output = x @ weights + bias
activated = torch.relu(linear_output)

print("linear_output:", round(linear_output.item(), 3))
print("relu_output:", round(activated.item(), 3))
```

Expected output:

```text
linear_output: 3.35
relu_output: 3.35
```

If the linear output were negative, ReLU would turn it into `0`. That small gate is what lets stacked layers model nonlinear patterns.

## Learn in This Order

| Order | Read | What to focus on |
|---|---|---|
| 1 | [6.1.2 ML to DL Bridge](./00-ml-to-dl-bridge.md) | what changes after sklearn |
| 2 | [6.1.3 Neurons and Activation](./01-neurons-activation.md) | weighted sum, bias, ReLU |
| 3 | [6.1.4 Forward and Backward](./02-forward-backward.md) | prediction, loss, gradient |
| 4 | [6.1.5 Optimizers](./03-optimizers.md) | SGD, Momentum, Adam intuition |
| 5 | [6.1.6 Regularization](./04-regularization.md) | overfitting controls |
| 6 | [6.1.7 Weight Initialization](./05-weight-init.md) | stable starting points |
| 7 | [6.1.8 Optional History](./06-history-breakthroughs.md) | why backprop, CNN, RNN, Attention, and Transformer appeared |

## Evidence to Keep

By the end of 6.1, keep one short note with these four lines:

```text
one_layer: input @ weights + bias
nonlinearity: activation lets stacked layers model curved patterns
training: forward -> loss -> backward -> optimizer step
debug_first: check shape, loss, gradient, update
```

This note becomes the pocket map for PyTorch, CNN, RNN, and Transformer later in Chapter 6.

## Pass Check

You pass this roadmap when you can explain one layer as `input @ weights + bias`, describe what an activation does, and connect loss, gradient, and optimizer into one training loop.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer connects tensors, model layers, loss, `backward()`, and optimizer updates into one training loop.
2. The evidence should include a runnable mini experiment, tensor-shape checks, and a loss or validation curve you can explain.
3. A good self-check names one failure mode such as shape mismatch, no loss decrease, overfitting, data leakage, or using Attention/Transformer words without explaining the data flow.

</details>

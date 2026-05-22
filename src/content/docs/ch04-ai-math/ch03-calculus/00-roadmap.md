---
title: "4.3.1 Calculus Roadmap: How Models Learn by Reducing Loss"
description: "A compact calculus and optimization roadmap for AI: derivatives, gradients, gradient descent, and backpropagation."
sidebar:
  order: 8
head:
  - tag: meta
    attrs:
      name: keywords
      content: "calculus guide, optimization guide, derivatives, gradients, gradient descent, backpropagation"
---
Calculus explains how a model changes its parameters. The first goal is intuition: measure change, move in a better direction, repeat.

## Look at the Map First

![Calculus and Optimization Learning Map](/img/course/ch04-calculus-roadmap-vertical-en.webp)

The training flow is:

![Relationship diagram of calculus and optimization sections](/img/course/ch04-calculus-training-flow-en.webp)

| Idea | First meaning in AI |
|---|---|
| derivative | how fast one value changes |
| gradient | how many parameters should change together |
| gradient descent | update parameters toward lower loss |
| chain rule | connect changes across steps |
| backpropagation | compute many gradients efficiently |

When you later see `loss.backward()` and `optimizer.step()`, this chapter is the background.

## Run the Smallest Loop

Create `gradient_descent_first_loop.py`. It finds a number close to `3` by reducing `(w - 3)^2`.

```python
w = 0.0
learning_rate = 0.2

for step in range(1, 7):
    gradient = 2 * (w - 3)
    w = w - learning_rate * gradient
    loss = (w - 3) ** 2
    print(step, "w=", round(w, 3), "loss=", round(loss, 3))
```

Expected output:

```text
1 w= 1.2 loss= 3.24
2 w= 1.92 loss= 1.166
3 w= 2.352 loss= 0.42
4 w= 2.611 loss= 0.151
5 w= 2.767 loss= 0.054
6 w= 2.86 loss= 0.02
```

The number moves toward `3`, and the loss gets smaller. That is the training idea before the neural network becomes large.

## Learn in This Order

| Order | Read | What to focus on first |
|---|---|---|
| 1 | [4.3.2 Derivatives](./01-derivatives.md) | rate of change |
| 2 | [4.3.3 Partial Derivatives and Gradients](./02-partial-derivatives-gradient.md) | many parameters changing together |
| 3 | [4.3.4 Gradient Descent](./03-gradient-descent.md) | update loop, learning rate, loss curve |
| 4 | [4.3.5 Backpropagation](./04-chain-rule-backprop.md) | chain rule, `loss.backward()` intuition |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
function: objective, loss, derivative, gradient, or chain-rule expression
calculation: numeric derivative, gradient step, or backprop trace
output: slope, gradient vector, updated parameter, or loss change
failure_check: sign error, learning rate too large, local slope misunderstanding, or broken chain
Expected_output: calculation trace showing how a parameter changes
```

## Pass Check

You pass this roadmap when you can explain why gradient descent repeats “compute loss -> compute gradient -> update parameter,” and why a learning rate that is too large can make training unstable.


<details>
<summary>Check reasoning and explanation</summary>

- The calculus route is passed when you can explain derivative as local change, gradient as multi-parameter direction, and gradient descent as repeated loss-reducing updates.
- Keep a derivative plot, one gradient vector, one loss curve, and one manual-versus-autograd comparison as evidence.
- The safest habit is to always ask: if the parameter moves a little in this direction, does the loss go up or down?

</details>

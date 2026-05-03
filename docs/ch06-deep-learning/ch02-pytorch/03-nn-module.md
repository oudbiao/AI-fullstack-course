---
title: "2.5 nn Module"
sidebar_position: 3
description: "Learn to organize models with nn.Module, nn.Linear, and nn.Sequential, and understand forward and parameter management."
keywords: [nn.Module, nn.Linear, nn.Sequential, forward, parameters, PyTorch]
---

# nn Module

## Learning Objectives

- Understand why PyTorch uses `nn.Module` to organize models
- Master `nn.Linear`, `nn.ReLU`, and `nn.Sequential`
- Be able to write the simplest custom network yourself
- Understand the roles of `forward()`, `parameters()`, `train()`, and `eval()`

---

## First, Build a Map

The most important thing in this section is not memorizing class names, but seeing clearly:

![nn.Module parameter organization flowchart](/img/course/ch06-nn-module-parameter-flow-en.png)

So what this section really solves is:

- How a model structure is organized into a "trainable object"

## How This Section Connects to the Previous and Next Ones

- The previous section, `autograd`, already solved "where gradients come from"
- This section starts solving "where these parameters are stored and how they are managed together"
- The next section, `DataLoader`, will solve "how data is fed in batch by batch"

So this section is really preparing the "model half" of the training loop.

## 1. Why Do We Need `nn.Module`?

If a tensor is a "data box," then `nn.Module` is a "model box."

It helps you organize a bunch of things:

- Network layers
- Parameters
- Forward computation logic
- Train / evaluation mode switching

As an analogy:

| Component | Analogy |
|---|---|
| `Tensor` | A brick |
| `nn.Linear` | A standard part |
| `nn.Module` | A composable machine |

Without `nn.Module`, you could still write networks by hand, but it would be very messy.
With it, a model is like LEGO blocks that can be stacked layer by layer.

### 1.1 A More Beginner-Friendly Intuition: `nn.Module` Is a "Model Container"

You can first think of it as a unified model box that holds:

- Network layers
- Parameters
- Forward logic
- Train / evaluation mode

This is why many later places only need to pass in a `model` object to complete:

- Forward computation
- Parameter updates
- Saving and loading

---

## 2. The Most Common Layer: `nn.Linear`

A linear layer does this:

> `y = xW + b`

```python
import torch
from torch import nn

layer = nn.Linear(in_features=3, out_features=2)

x = torch.tensor([[1.0, 2.0, 3.0]])
y = layer(x)

print("Output:", y)
print("weight shape:", layer.weight.shape)
print("bias shape:", layer.bias.shape)
```

You need to understand the shapes here:

- The input is `[1, 3]`, meaning 1 sample, and each sample has 3 features
- The output is `[1, 2]`, meaning it is mapped to 2 output values

### 2.1 When You See `nn.Linear(in, out)`, What Should Immediately Come to Mind?

The most important thing to think is:

- This is not some "mysterious transformation"
- It maps each sample from an `in`-dimensional representation to an `out`-dimensional representation

So the most practical way to understand a linear layer is usually:

- The input space is re-encoded into a new feature space

---

## 3. Build a Network Quickly with `nn.Sequential`

If the model is relatively simple, you can connect layers in order directly:

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

x = torch.tensor([[1.0, 2.0]])
pred = model(x)

print(pred)
```

This code means:

1. Input 2 features
2. First map them to a 4-dimensional hidden layer
3. Pass through `ReLU` activation
4. Then output 1 value

This is already a minimal multilayer perceptron.

---

## 4. Define a Model Class Yourself

When the model gets a little more complex, it is recommended to inherit from `nn.Module`.

```python
import torch
from torch import nn

class ScorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ScorePredictor()

x = torch.tensor([
    [3.0, 4.0],   # Study time, number of assignments completed
    [5.0, 8.0]
])

print(model(x))
```

### What Do `__init__()` and `forward()` Do Respectively?

| Method | Responsibility |
|---|---|
| `__init__()` | Define layers and submodules |
| `forward()` | Define how data flows through the model |

A simple way to remember it:

- `__init__` is responsible for "building the machine"
- `forward` is responsible for "how the machine works"

### 4.1 Why Does `forward()` Only Contain Data Flow and Not Training Logic?

Because training logic belongs to another level.
The responsibility of `forward()` is very pure:

- Given an input
- Return an output

And things like:

- loss
- backward
- optimizer.step

do not belong in `forward()`.
Being clear about this separation is very important when you read large model code later.

---

## 5. How Are Model Parameters Managed?

One big advantage of `nn.Module` is that:
the layers you define are automatically registered by the framework, and the parameters also automatically appear in `model.parameters()`.

```python
import torch
from torch import nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = TinyNet()

for name, param in model.named_parameters():
    print(name, param.shape)
```

This is why an optimizer can be written directly as:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### 5.1 Why Is `model.parameters()` So Important?

Because it unifies the idea that "a model is a collection of many parameters."

In other words, the optimizer does not really care how many layers you wrote or what structure you used. What it cares about most is:

- Which parameters do I need to update?

And `nn.Module` is automatically organizing this for you.

Because the model has already packaged all the parameters that need to be learned.

---

## 6. What Are `train()` and `eval()`?

Many beginners think:

- `model.train()` starts training
- `model.eval()` starts evaluation

That is not completely correct.
Their real role is to **switch the behavior mode of certain internal layers**.

The two most typical layers are:

- `Dropout`
- `BatchNorm`

Although we have not focused on them yet, you should first remember this habit:

```python
model.train()   # Before training
model.eval()    # Before validation / testing
```

### 6.1 At the Beginner Stage, Fix This in Your Memory — It Is Very Worth It

You may not fully understand:

- Dropout
- BatchNorm

right now, but you should still build this reflex:

- `model.train()` before training
- `model.eval()` before validation

The more complex the network becomes later, the more this habit will save you.

---

## 7. A Complete Small Example: Predicting Scores

Below is a small network that you can run directly.
It takes two features as input:

- Study hours per week
- Number of practice problems completed per week

It outputs a predicted score.

```python
import torch
from torch import nn

torch.manual_seed(42)

X = torch.tensor([
    [2.0, 1.0],
    [3.0, 2.0],
    [4.0, 3.0],
    [5.0, 5.0],
    [6.0, 6.0],
    [7.0, 8.0]
])

y = torch.tensor([
    [55.0],
    [60.0],
    [68.0],
    [78.0],
    [85.0],
    [92.0]
])

class ScorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ScorePredictor()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch={epoch:3d}, loss={loss.item():.4f}")

test = torch.tensor([[6.5, 7.0]])
print("Predicted score:", round(model(test).item(), 2))
```

---

## 8. When Should You Use `Sequential`, and When Should You Define a Custom `Module`?

### Use `nn.Sequential`

Suitable when:

- Layers are stacked in a strict order
- There is no branching structure
- There is no special control logic

### Use a Custom `nn.Module`

Suitable when:

- There are multiple inputs / outputs
- There are skip connections, branches, or conditional logic
- You want the structure to be clearer and easier to maintain

In practice, custom `Module`s are more common.

---

## 9. Common Beginner Mistakes

### 1. Creating New Layers Temporarily Inside `forward()`

Not recommended.
Layers should be defined in `__init__()` so that parameters can be registered correctly.

### 2. Only Knowing How to Write `Sequential`, but Not a Class

`Sequential` is convenient, but you will eventually need to know how to write a custom `Module`.
The CNNs and Transformer later on both depend on it.

### 3. Not Knowing What Parameters Exist in the Model

Develop the habit of using `named_parameters()`. It is very useful for debugging.

---

## Summary

The core ideas you should take away from this section are:

1. `nn.Module` is the standard way to organize models
2. `forward()` describes data flow, not the training process
3. Model parameters are automatically collected so the optimizer can update them together

Once you have the model container, the next step is to feed data in batch by batch.

## What Should You Take Away Most from This Section?

If we compress it into one sentence, it is:

> **The core value of `nn.Module` is not making code look more object-oriented, but allowing layers, parameters, forward logic, and training mode to be managed together.**

---

## Exercises

1. Change the hidden layer in `ScorePredictor` from `8` to `16`, and observe how the loss changes.
2. Remove `ReLU()`, and see whether the model can still learn the pattern.
3. Use `named_parameters()` to print each layer’s parameter names and shapes, and make sure you understand every layer.

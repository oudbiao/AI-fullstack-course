---
title: "4.3 LSTM and GRU"
sidebar_position: 2
description: "From why RNNs forget to how gated mechanisms control information flow, understand the role of LSTM and GRU in sequence modeling."
keywords: [LSTM, GRU, gated mechanism, cell state, update gate, forget gate]
---

# LSTM and GRU

![LSTM gated memory flow diagram](/img/course/lstm-gate-memory-flow.png)

:::tip Section Overview
In the previous section, you saw that an RNN can “read while remembering.”
This section solves a more realistic problem:

> **What if a plain RNN cannot remember for very long?**

LSTM and GRU were designed to solve this “it can read, but it forgets easily” problem.
:::

## Learning Objectives

- Understand why a plain RNN easily forgets distant information
- Build an intuition for what a gated mechanism does
- Master LSTM cell state and its three gates
- Master the update gate and reset gate in GRU
- Understand the input and output of `nn.LSTM` and `nn.GRU` in PyTorch
- Know when LSTM is a better choice and when GRU is already enough

## Historical Background: Why Did the Field Eventually Move to LSTM?

The most important historical milestones in this section are:

| Year | Milestone | Key Authors | What It Most Importantly Solved |
|---|---|---|---|
| 1994 | Learning Long-Term Dependencies is Difficult | Bengio, Simard, Frasconi | Systematically revealed the gradient vanishing problem in training plain RNNs on long dependencies |
| 1997 | LSTM | Hochreiter, Schmidhuber | Used gated memory to ease long-term dependency and gradient problems |

For beginners, the most important thing to remember first is:

> **LSTM is not “just a more complicated RNN.” It was designed to solve the core problem that plain RNNs struggle to reliably remember long-range information.**

So the real main thread of this lesson is not:

- memorizing a few gate names

but:

- understanding why these gates were invented

### Why Did Many People at the Time See LSTM as a “Rescue Move”?

Because before that, RNNs were not unpopular.  
Quite the opposite—they looked very appealing:

- Text is a sequence
- Speech is a sequence
- Time series are also sequences

Intuitively, RNNs seemed like the natural choice for these tasks.

But once people actually trained them, they kept running into the same wall:

- Early information could not be preserved
- Gradients became weaker as they were propagated backward
- On long sequences, the model could “read but forget”

So what excited people about LSTM was not just that “the gates are clever,”  
but that it clearly said for the first time:

> **The problem is not that the RNN direction is wrong, but that it needs a structure that manages memory more carefully.**

### Why Did “Gradient Vanishing” Make So Many People Worry About the RNN Path?

Because on paper, RNNs looked like they could handle almost any sequence:

- Text
- Speech
- Time series

But once people trained them on long sequences, they discovered:

- Earlier information was much harder to keep
- Gradients became weaker and weaker as they propagated backward

It is like thinking at first that you can retell a long story accurately,  
but by the time you reach the end, the details from the beginning have already become blurry.

So what really impressed people about LSTM was not simply “it has a few more gates,”  
but that it seemed to say:

> **If a plain RNN cannot remember through natural propagation alone, then we should add a management mechanism to memory itself.**

That is why many people later saw LSTM as:

- not a small tweak
- but a real response to the long-dependency problem

---

## 1. Why Is a Plain RNN Not Enough?

### 1.1 A Classic Problem: Long-Range Dependencies

Look at this sentence:

> “When I was young, I lived in Shanghai for many years, so although I moved away now, the city I know best is still Shanghai.”

If the model needs to know which city is being referred to when it reaches “Shanghai” at the end,  
it must keep track of information from a long time ago.

A plain RNN can theoretically do this, but in practice it often faces these issues:

- Earlier information becomes diluted over time
- Gradients tend to vanish during training
- On long sequences, memory becomes unstable

### 1.2 An Intuitive Analogy

A plain RNN is like repeatedly rewriting a short summary on a piece of paper:

- Every time a new sentence comes in, you revise the old summary

The problem is:

- The summary space is too small
- Old information is easily overwritten

So a smarter idea appeared later:

> **Instead of relying only on a changing “summary,” let the model learn what should be forgotten, what should be kept, and what should be output.**

That is the gated mechanism.

---

## 2. LSTM’s Core Intuition: Adding “Gates” to Memory

### 2.1 What Does LSTM Add?

Compared with a plain RNN, the key enhancements in LSTM are:

- A more stable memory pathway: `cell state`
- Several gates that control information flow

You can first understand it as:

> **A plain RNN is like having only a small notebook, while LSTM is like a more refined memory management system.**

### 2.2 The Three Gates in LSTM

| Gate | Function |
|---|---|
| Forget Gate | Decides how much of the old memory to keep |
| Input Gate | Decides how much new information to write |
| Output Gate | Decides how much to expose as output |

These gates are not hand-written rules; they are learned by the model itself.

---

## 3. Build Intuition First with a “Scalar LSTM”

### 3.1 Why Look at the Scalar Version First?

Because a real LSTM is full of matrices and vectors at the beginning, which can overwhelm beginners.  
Starting with a smaller version makes it easier to grasp the essence.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Assume this is the memory from the previous time step
c_prev = 0.8

# Current input and previous hidden state
x_t = 1.2
h_prev = 0.5

# We manually set some gate values here; in a real model, the network learns them
forget_gate = sigmoid(1.0)   # about 0.73
input_gate = sigmoid(0.2)    # about 0.55
output_gate = sigmoid(0.7)   # about 0.67

# New candidate information
c_tilde = np.tanh(0.9)

# Update cell state
c_t = forget_gate * c_prev + input_gate * c_tilde

# Update hidden state
h_t = output_gate * np.tanh(c_t)

print("forget_gate =", round(float(forget_gate), 4))
print("input_gate  =", round(float(input_gate), 4))
print("output_gate =", round(float(output_gate), 4))
print("c_t         =", round(float(c_t), 4))
print("h_t         =", round(float(h_t), 4))
```

### 3.2 What Is This Code Teaching?

It teaches you that:

- `forget_gate` decides how much old memory to discard
- `input_gate` decides how much new information to write
- `output_gate` decides how much to reveal outward

In other words, what makes LSTM powerful is not that it is “more complicated,” but that:

> **It has learned to control information flow.**

![LSTM gated information flow control diagram](/img/course/ch06-lstm-gates-information-control-map.png)

:::tip Reading Tip
When reading this diagram, focus on just three things first: the Forget Gate decides how much old memory to keep, the Input Gate decides how much new information to write, and the Output Gate decides how much to output externally. The key point of LSTM is not the gate names, but that it finally starts “managing memory.”
:::

---

## 4. The Two States in LSTM: `c_t` and `h_t`

### 4.1 Why Are There Two States?

An LSTM usually has:

- `c_t`: cell state, more like the main long-term memory path
- `h_t`: hidden state, more like the current output at this time step

### 4.2 An Easy-to-Remember Analogy

You can think of it as:

- `c_t`: your long-term draft notebook
- `h_t`: what you currently say out loud

You do not necessarily say everything in the draft notebook, but it determines what you can still remember later.

---

## 5. GRU: A Lighter Gated Version

### 5.1 Why Did GRU Appear?

LSTM is powerful, but it is also more complex.  
Later, people proposed GRU (Gated Recurrent Unit) as a version that is:

- simpler
- has fewer parameters
- performs similarly well in many cases

### 5.2 The Two Core Gates in GRU

| Gate | Function |
|---|---|
| Update Gate | Decides how much old state to keep and how much new state to mix in |
| Reset Gate | Decides how much old information to forget when computing the new state |

### 5.3 A Minimal GRU Intuition Example

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

h_prev = 0.7
x_t = 1.1

update_gate = sigmoid(0.8)
reset_gate = sigmoid(-0.3)

h_candidate = np.tanh(x_t + reset_gate * h_prev)
h_t = (1 - update_gate) * h_prev + update_gate * h_candidate

print("update_gate =", round(float(update_gate), 4))
print("reset_gate  =", round(float(reset_gate), 4))
print("h_candidate =", round(float(h_candidate), 4))
print("h_t         =", round(float(h_t), 4))
```

### 5.4 Intuitive Difference from LSTM

- LSTM: more like a detailed memory management system
- GRU: more like a compressed memory management system

So it is often convenient to remember it this way:

> **GRU = a lighter version of LSTM.**

---

## 6. How Should You Choose Between LSTM and GRU?

### 6.1 General Rule of Thumb

If you just want a baseline sequence model:

- Trying GRU first is often easier

If the task is especially sensitive to long-range dependencies:

- LSTM is often worth trying

### 6.2 But Do Not Overhype Them

In the age of large models, many long-text tasks are handled more often by Transformers.  
Still, LSTM and GRU remain very common in these scenarios:

- Shorter sequence modeling
- Small-data settings
- Time-series baselines
- Teaching and understanding the essence of sequence modeling

---

## 7. How Do You Use LSTM and GRU in PyTorch?

### 7.1 Minimal Runnable Example

```python
import torch

torch.manual_seed(42)

# batch=4, seq_len=6, input_size=8
x = torch.randn(4, 6, 8)

lstm = torch.nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
gru = torch.nn.GRU(input_size=8, hidden_size=16, batch_first=True)

lstm_out, (lstm_h, lstm_c) = lstm(x)
gru_out, gru_h = gru(x)

print("lstm_out shape:", lstm_out.shape)
print("lstm_h shape  :", lstm_h.shape)
print("lstm_c shape  :", lstm_c.shape)
print("gru_out shape :", gru_out.shape)
print("gru_h shape   :", gru_h.shape)
```

### 7.2 What Are the Outputs?

For LSTM:

- `lstm_out`: output at each time step
- `lstm_h`: final hidden state
- `lstm_c`: final cell state

For GRU:

- `gru_out`: output at each time step
- `gru_h`: final hidden state

Here you can also see one difference at a glance:

> **LSTM maintains one extra `c` state compared with GRU.**

---

## 8. A Small Task: Let the Model Remember Information from the Beginning of a Sequence

Next, we construct a very small task:

- The first position of the input sequence may be `+1` or `-1`
- The label depends on this first value
- Noise is added in the middle

In other words, the model must remember information from “far earlier.”

```python
import torch
from torch import nn

torch.manual_seed(42)

def build_dataset(n=100):
    X, y = [], []
    for _ in range(n):
        first = 1.0 if torch.rand(1).item() > 0.5 else -1.0
        seq = torch.randn(8, 1) * 0.2
        seq[0, 0] = first
        X.append(seq)
        y.append(1 if first > 0 else 0)
    return torch.stack(X), torch.tensor(y)

X, y = build_dataset(120)

class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=8, batch_first=True)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        out, h = self.gru(x)
        return self.fc(h[-1])

model = GRUClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

for epoch in range(80):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        acc = (pred.argmax(dim=1) == y).float().mean().item()
        print(f"epoch={epoch:3d}, loss={loss.item():.4f}, acc={acc:.3f}")

with torch.no_grad():
    final_acc = (model(X).argmax(dim=1) == y).float().mean().item()
    print("final acc =", round(final_acc, 3))
```

This task is very small, but it does teach you something important:

> Gated recurrent networks are better than plain RNNs at preserving important early information.

---

## 9. Common Pitfalls for Beginners

### 9.1 Thinking of LSTM / GRU as “deeper than RNN”

It is not “deeper,” but “smarter about memory management.”

### 9.2 Confusing `out`, `h`, and `c`

Remember:

- `out`: output at each step
- `h`: final hidden state
- `c`: LSTM’s long-term memory state

### 9.3 Thinking LSTM Automatically Never Forgets

Not true.  
It is just better than a plain RNN at controlling what to forget and what to keep; that does not mean it can handle infinitely long dependencies with ease.

---

## Summary

The key idea in this section is not to memorize gate formulas, but to understand this:

> **The essence of LSTM and GRU is using gated mechanisms to learn “what to forget, what to keep, and what to output now.”**

They are an important upgrade over plain RNNs, and they are also a great stepping stone toward understanding attention mechanisms and Transformers later on.

---

## Exercises

1. Change the gate values in the scalar LSTM example and observe how `c_t` and `h_t` change.
2. Modify the GRU classification task so that the label depends on the last value, and see whether the model learns more easily.
3. Replace the same task with LSTM and GRU separately, and compare training speed and code complexity.
4. Explain in your own words: why is the key idea of LSTM / GRU not “more complexity,” but “more fine-grained control over information flow”?

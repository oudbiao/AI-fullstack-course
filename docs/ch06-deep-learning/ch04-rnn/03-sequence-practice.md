---
title: "4.4 Sequence Modeling in Practice"
sidebar_position: 3
description: "Use a real, trainable small time-series task to connect window construction, RNN/LSTM training, validation, and prediction."
keywords: [sequence modeling, time series, RNN, LSTM, sliding window, forecast]
---

# Sequence Modeling in Practice

:::tip Section Overview
In the previous two sections, you already learned:

- RNNs “read while remembering”
- LSTM / GRU “control memory more intelligently”

In this section, we will turn those ideas into a small project:

> **Given a sequence, predict the next value.**
:::

## Learning Goals

- Learn how to split a continuous sequence into training samples
- Build a minimal time-series predictor with LSTM
- Understand the training set, validation set, and prediction workflow
- Learn how to tell whether a model is learning patterns or just memorizing blindly
- Know the most common pitfalls in sequence practice

---

## 1. Why choose “time-series forecasting” as a practical exercise?

### 1.1 Because it is ideal for practicing the basics of sequence modeling

Many sequence tasks can be abstracted as:

- A preceding input segment
- A following output

Time-series forecasting is the most typical example.

For example:

- Predict the sales on day 8 based on the previous 7 days of sales
- Predict the next temperature based on the previous 12 temperature values

### 1.2 A very important intuition

When doing this kind of task, the model is not memorizing individual numbers. It is learning:

> **change patterns.**

For example:

- Periodicity
- Trend
- Fluctuation

This is very different from a normal classification task.

---

## 2. First, generate a dataset we can run directly

### 2.1 Use a sine wave + noise to create a minimal sequence

The benefits are:

- No external dataset needed
- Clear patterns
- Very suitable for teaching

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

t = np.arange(0, 200)
series = np.sin(t * 0.1) + np.random.randn(200) * 0.05

plt.figure(figsize=(10, 4))
plt.plot(t, series)
plt.title("Toy Time Series")
plt.grid(True, alpha=0.3)
plt.show()
```

### 2.2 What does this data look like?

It has two characteristics:

- It fluctuates overall
- It includes a bit of random noise

That makes it a little closer to a real task than a perfectly regular sequence.

---

## 3. Sliding window: how do we turn a whole sequence into samples?

### 3.1 Core idea

A model cannot directly consume “an entire infinite sequence.”
We usually cut it into many small segments:

- The first `window_size` points are used as input
- The point at `window_size + 1` is used as the label

This is called a sliding window.

### 3.2 Runnable example

```python
import numpy as np

series = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
window_size = 3

X, y = [], []
for i in range(len(series) - window_size):
    X.append(series[i:i + window_size])
    y.append(series[i + window_size])

X = np.array(X)
y = np.array(y)

print("X =\n", X)
print("y =", y)
```

### 3.3 Why is this step so important?

Because it determines how sequence-task samples are defined.
If the window construction is wrong, training, validation, and prediction will all be wrong too.

---

## 4. Organize the data into a PyTorch-trainable format

### 4.1 Complete data preparation

```python
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

t = np.arange(0, 200)
series = np.sin(t * 0.1) + np.random.randn(200) * 0.05
series = series.astype(np.float32)

window_size = 12
X, y = [], []

for i in range(len(series) - window_size):
    X.append(series[i:i + window_size])
    y.append(series[i + window_size])

X = np.array(X)
y = np.array(y)

# Convert to [batch, seq_len, input_size]
X = torch.tensor(X).unsqueeze(-1)
y = torch.tensor(y).unsqueeze(-1)

print("X shape:", X.shape)
print("y shape:", y.shape)
```

### 4.2 Why use `unsqueeze(-1)`?

Because LSTM usually expects input in the form:

- `[batch, seq_len, input_size]`

Here each time step has only one feature value, so:

- `input_size = 1`

---

## 5. A small LSTM predictor that can really be trained

### 5.1 Define the model

```python
import torch
from torch import nn

class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)
```

### 5.2 Why do we take only the last time step?

Because the current task is:

> Use the previous window to predict the “next value”

So the most natural approach is to use the representation at the last time step as a summary of the whole window.

---

## 6. Complete training workflow

### 6.1 Training + validation

```python
import numpy as np
import torch
from torch import nn

np.random.seed(42)
torch.manual_seed(42)

t = np.arange(0, 200)
series = np.sin(t * 0.1) + np.random.randn(200) * 0.05
series = series.astype(np.float32)

window_size = 12
X, y = [], []
for i in range(len(series) - window_size):
    X.append(series[i:i + window_size])
    y.append(series[i + window_size])

X = torch.tensor(np.array(X)).unsqueeze(-1)
y = torch.tensor(np.array(y)).unsqueeze(-1)

train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMForecaster(hidden_size=32)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 40 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val)
        print(f"epoch={epoch:3d}, train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")
```

### 6.2 What should you pay closest attention to in this training code?

The most important things are:

- Whether the input shape is correct
- Whether only `out[:, -1, :]` is used at the end
- Whether the loss really decreases

Once you understand these three points, you have truly stepped into sequence modeling practice.

---

## 7. Make a real prediction

### 7.1 Single-window prediction

```python
model.eval()
with torch.no_grad():
    sample_x = X_val[0:1]
    pred = model(sample_x)
    print("Predicted value:", float(pred.item()))
    print("True value:", float(y_val[0].item()))
```

### 7.2 Plot the prediction against the true value

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    val_pred = model(X_val).squeeze(-1).numpy()
    val_true = y_val.squeeze(-1).numpy()

plt.figure(figsize=(10, 4))
plt.plot(val_true, label="true")
plt.plot(val_pred, label="pred")
plt.legend()
plt.title("Validation Prediction")
plt.grid(True, alpha=0.3)
plt.show()
```

When working on sequence tasks in practice, plots are often more helpful than a single metric for finding problems:

- Is the model failing to follow peaks and valleys?
- Is there an overall lag?
- Has the model learned a flat line?

---

## 8. The most common pitfalls in sequence modeling practice

### 8.1 Data leakage

If you split the training set / validation set incorrectly, you may leak future information to the model.

For time-series tasks, the safest principle is usually:

> Split in time order; do not shuffle randomly.

### 8.2 Window too short or too long

- Too short: the model cannot see enough history
- Too long: training becomes harder, and there is more noise

### 8.3 Only looking at loss, not the curve

In sequence prediction, plotting is often very important.
Because two models with similar loss can have completely different trends.

### 8.4 Thinking the model learned “causality” when it actually learned only short-term patterns

This is something you must be careful about in all sequence prediction tasks.
A model can predict something without truly understanding the mechanism.

---

## 9. A very important engineering intuition

In real projects, sequence tasks do not always use RNN / LSTM.
Today, many tasks also use:

- Transformer
- Temporal Convolution
- Traditional statistical models

But no matter what model you use in the future, the window construction, time-based splitting, and validation methods taught in this section are still foundational.

---

## Summary

The most important thing in this section is not “getting the LSTM to run,” but understanding:

> **The key to practical sequence work is how to split continuous data into training samples, and how to let the model learn change patterns without leaking future information.**

When you can connect data construction, the training workflow, validation, and prediction plotting, then sequence modeling is truly taking shape.

---

## Exercises

1. Change `window_size` from 12 to 6 and 24, and compare the prediction results.
2. Replace the model with a GRU and see whether the training curves are different.
3. Randomly shuffle the training set and validation set on purpose, then think about why this is dangerous for time-series data.
4. Think about this: if your sequence has a clear weekly cycle, how should you design the window length?

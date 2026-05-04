---
title: "2.3 PyTorch Basics"
sidebar_position: 1
description: "Build a solid first layer of PyTorch fundamentals, from tensors, shapes, and indexing to broadcasting and its relationship with NumPy."
keywords: [PyTorch, tensor, shape, broadcasting, numpy]
---

# PyTorch Basics

## Learning Objectives

- Understand what a `Tensor` is
- Master tensor creation, shapes, data types, and common operations
- Understand the relationship between PyTorch and NumPy
- Be able to read the most basic tensor operation code independently

---

## First, Build a Map

Don’t treat this section as a “PyTorch syntax table.” A better way to think about it is:

![PyTorch Tensor lifecycle map](/img/course/ch06-pytorch-tensor-lifecycle-map-en.png)

In other words, what you really need to solidify here is:

- Can you put data into PyTorch?
- Can you read tensor shapes?
- Can you safely perform the most basic operations?

## How This Section Connects to Station 5 and NumPy

If you’re coming from Station 5, you can first understand this section like this:

- The `X`, `y`, and matrix multiplication from Station 5 are still here
- The difference is that now they enter a container more suitable for deep learning training: `Tensor`

If you’re already familiar with NumPy, you can remember it this way:

- `Tensor` is very similar to `ndarray`
- But it can also run on GPU and participate in automatic differentiation

So this section is not about learning “brand-new math,” but about:

- Expressing the same data objects in a way that is more suitable for training neural networks

## 1. What Exactly Is a Tensor?

The most practical way to understand it is:

> **Tensor = a multi-dimensional array that can be computed on CPU / GPU**

If you’ve studied NumPy, you can think of it as an “upgraded `ndarray`”:

- It can perform numerical operations
- It can be moved to the GPU
- It can participate in automatic differentiation

By analogy:

| Concept | Analogy |
|---|---|
| Scalar (0D) | A single number |
| Vector (1D) | A row of numbers |
| Matrix (2D) | A table |
| Tensor (higher dimensions) | A stack of tables / a batch of images / a video clip |

In deep learning, almost all data eventually becomes a tensor:

- A grayscale image: `[height, width]`
- A color image: `[channels, height, width]`
- A batch of images: `[batch_size, channels, height, width]`
- A batch of sentence embeddings: `[batch_size, seq_len, embedding_dim]`

### 1.1 What Should You Ask First When You See a Tensor?

Don’t rush to ask about the API. First ask these three questions:

1. What data does it contain?
2. What does each dimension represent?
3. Which layer will this data be sent to next?

This helps you connect “shape” and “meaning” from the very beginning.

---

## 2. Creating Tensors

:::info Runtime Environment
The following code can be run directly. If PyTorch is not installed locally:

```bash
pip install torch
```
:::

```python
import torch

# Create from a Python list
scores = torch.tensor([88, 92, 76, 95])
print(scores)

# Specify data type
prices = torch.tensor([12.5, 19.9, 8.8], dtype=torch.float32)
print(prices.dtype)

# Common initialization methods
zeros = torch.zeros((2, 3))
ones = torch.ones((2, 3))
randn = torch.randn((2, 3))
arange = torch.arange(0, 10, 2)

print("zeros:\n", zeros)
print("ones:\n", ones)
print("randn:\n", randn)
print("arange:", arange)
```

---

## 3. Shape, Dimensions, and Data Types

When you’re learning deep learning, what trips you up most often is not formulas, but **shape**.

You can think of `shape` as “how many layers this data box has, and how many elements each layer contains.”

```python
import torch

X = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

print("tensor:\n", X)
print("shape:", X.shape)       # torch.Size([2, 3])
print("ndim:", X.ndim)         # 2 dimensions
print("dtype:", X.dtype)       # float32
print("total elements:", X.numel())   # 6
```

### A Very Important Habit

Before writing a model, ask yourself:

1. What does each dimension of this tensor mean?
2. Is the current shape correct?
3. What shape will the next layer expect?

Many training errors are, at their core, shape mismatches.

### 3.1 A More Reliable “4-Step Tensor Check”

Whenever you get a tensor, it’s a good idea to check it in these four steps:

1. Look at `shape`
2. Look at `dtype`
3. Clarify the meaning of each dimension
4. Clarify what operation you’ll do next

For example:

```python
print(X.shape, X.dtype)
print("meaning: [batch, features]")
```

This habit will save you from a lot of mysterious errors.

### A Recording Habit Beginners Should Build

When you first start using PyTorch, it’s a good idea to jot down a sentence whenever you see a tensor:

```python
print("shape:", X.shape, "| meaning: [batch, features]")
```

Writing the “shape” and the “meaning” together is much clearer than looking at `torch.Size(...)` alone.

![Quick reference map for PyTorch tensor shapes and meanings](/img/course/ch06-tensor-shape-meaning-map-en.png)

:::tip Reading Tip
You can use this diagram as a quick shape reference: tabular data commonly uses `[batch, features]`, images commonly use `[batch, channels, height, width]`, and text sequences commonly use `[batch, seq_len, embedding_dim]`. First clarify the meaning of each dimension, then write the model, and you’ll get far fewer errors.
:::

---

## 4. Indexing, Slicing, and Reshaping

```python
import torch

X = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("row 0:", X[0])
print("row 1, column 2:", X[1, 2])
print("first two rows:\n", X[:2])
print("column 2:", X[:, 1])

# Reshape
flat = X.reshape(9)
grid = flat.reshape(3, 3)

print("flattened:", flat)
print("back to 3x3:\n", grid)
```

### Intuition for `reshape`

It’s like rearranging a box of building blocks:

- The number of elements cannot change
- You’re only changing how they are organized

### 4.1 A Common Pitfall for Beginners with `reshape`

The most common misunderstanding is:

- Thinking `reshape` changes the actual data content

In fact, it usually just changes how you view the same set of elements.
So a safer habit is to ask after every `reshape`:

- What does each dimension mean now?

---

## 5. Tensor Operations

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("addition:", a + b)
print("subtraction:", a - b)
print("element-wise multiplication:", a * b)
print("square:", a ** 2)
print("sum:", a.sum())
print("mean:", a.mean())
```

### Matrix Multiplication

One of the most common operations in deep learning is matrix multiplication:

```python
import torch

X = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])

W = torch.tensor([[2.0, 0.0],
                  [0.0, 2.0]])

Y = X @ W
print(Y)
```

This is the same linear algebra you learned at Station 4.
Many layers in neural networks are essentially “tensor linear transformations followed by a nonlinear function.”

### 5.1 When You See `@`, What Should Immediately Come to Mind?

The first thing worth thinking is:

- This is usually not ordinary arithmetic
- This is “recombining a batch of inputs according to weights”

In other words, when you see:

```python
X @ W
```

you can first understand it as:

- This layer is transforming the input into a new representation

---

## 6. Broadcasting

Broadcasting is a very code-saving mechanism in PyTorch.

The intuition is:

> “If two tensors don’t have exactly the same shape but are close enough, PyTorch will automatically expand them for you.”

```python
import torch

scores = torch.tensor([
    [80.0, 85.0, 90.0],
    [70.0, 75.0, 88.0]
])

bonus = torch.tensor([5.0, 5.0, 5.0])

print(scores + bonus)
```

Here `bonus` has shape `[3]`, and `scores` has shape `[2, 3]`.
PyTorch automatically treats `bonus` as being added to each row.

### Common Uses of Broadcasting

- Adding a shared bias to a batch of samples
- Normalizing images
- Scaling each feature in a batch

### 6.1 Why Is Broadcasting Both Convenient and Risky?

It’s convenient because it saves a lot of code.
It’s risky because:

- Sometimes the code runs
- But the broadcasting direction is not the one you expected

So in broadcasting scenarios, the safest habit is:

- First write down the shapes of both tensors
- Then make it clear which one is being expanded

---

## 7. Converting Between NumPy and PyTorch

NumPy and PyTorch are very closely related, so converting between them is common.

```python
import numpy as np
import torch

arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
tensor = torch.from_numpy(arr)

print("NumPy -> Tensor:\n", tensor)

back_to_numpy = tensor.numpy()
print("Tensor -> NumPy:\n", back_to_numpy)
```

### When Should You Use NumPy, and When Should You Use PyTorch?

- For data analysis and traditional numerical experiments: NumPy is convenient
- For training neural networks, automatic differentiation, and GPU usage: PyTorch is more suitable

---

## 8. A Small Example: Calculating Students’ Total Scores and Averages

This example doesn’t feel very “deep learning”-like, but it’s great for practicing tensor thinking.

```python
import torch

# 3 students, 4 subjects
scores = torch.tensor([
    [85.0, 92.0, 78.0, 90.0],
    [76.0, 88.0, 91.0, 84.0],
    [93.0, 87.0, 89.0, 95.0]
])

student_totals = scores.sum(dim=1)
student_means = scores.mean(dim=1)
subject_means = scores.mean(dim=0)

print("total score for each student:", student_totals)
print("average score for each student:", student_means)
print("average score for each subject:", subject_means)
```

Here you’re already using one of the most important tensor-thinking ideas:
**“Along which dimension should the calculation be performed?”**

- `dim=1` means aggregate by rows
- `dim=0` means aggregate by columns

---

## 9. Common Mistakes Beginners Make

### 1. Ignoring shape

Many people only look at the numbers and not the tensor shape.
As a result, the code seems correct at a glance, but throws a dimension error when run.

### 2. Treating `*` as matrix multiplication

In PyTorch:

- `*` is element-wise multiplication
- `@` is matrix multiplication

### 3. Not Being Clear About `dtype`

Some models require `float32`, while labels sometimes need `long`.
If the type is wrong, the loss function may fail immediately.

### 4. Looking at the values but not their meaning

The most common beginner problem is not inability to write code, but:

- The tensor prints out
- But you don’t know whether a dimension represents batch, features, channels, or classes

Once the semantics are unclear, `Linear`, `Conv`, and `Loss` all start to feel confusing.

---

## Summary

The most important thing in this section is not memorizing how many APIs you know, but building three basic instincts:

1. When you see data, first check `shape`
2. When you see an operation, first distinguish between “element-wise” and “matrix multiplication”
3. Know that inputs, parameters, and outputs in deep learning are all tensors at their core

Next, we’ll make these tensors “know how to change themselves,” and that is automatic differentiation.

## What You Should Take Away from This Section

If you only remember one sentence, I hope it is this:

> **What you really need to practice in the PyTorch basics section is not syntax fluency, but whether you can reliably match up a tensor’s shape, meaning, and operation style.**

Because most deep learning code problems later on eventually come back to these three things:

- shape
- dtype
- what the operation means

---

## Exercises

1. Create a random tensor with shape `(2, 3, 4)` and print its `shape`, `ndim`, and `numel()`.
2. Create a `3x3` tensor and reshape it into `1x9` and `9x1`.
3. Construct two matrices that can be multiplied, and try matrix multiplication with `@`.

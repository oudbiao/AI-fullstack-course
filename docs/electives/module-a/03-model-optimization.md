---
title: "1.3 Model Optimization Techniques"
sidebar_position: 3
description: "Starting from quantization, pruning, distillation, operator fusion, and batching, understand what model optimization is really trading off in deployment."
keywords: [model optimization, quantization, pruning, distillation, fusion, batching, deployment]
---

# Model Optimization Techniques

![Model Optimization Roadmap](/img/course/elective-model-optimization-map-en.png)

![Model Optimization Trade-off Dashboard](/img/course/elective-optimization-tradeoff-dashboard-en.png)

:::tip Reading guide
Before optimizing, first identify where the bottleneck is: memory, latency, throughput, accuracy, hardware compatibility, and maintenance cost often constrain one another. When reading this diagram, do not only ask “Can it be faster?”, but ask “What cost am I paying to improve which metric?”
:::

:::tip Where this section fits
Model optimization is easiest to turn into a pile of jargon:

- Quantization
- Pruning
- Distillation
- Fusion

But in real deployment, the question is much simpler:

> **What exactly are you trying to save: memory, latency, throughput, or hardware adaptation cost?**

Only after you make this clear can optimization stop becoming “optimization for optimization’s sake.”
:::

## Learning Objectives

- Understand what problems several mainstream model optimization methods are trying to solve
- Understand that optimization usually trades accuracy for performance rather than giving it away for free
- See a runnable example to understand quantization and cost trade-offs
- Build a deployment-oriented priority framework for choosing optimization methods

---

## 1. Model Optimization Is Not a Single Goal

### 1.1 What people usually want to optimize are actually four things

- Model size
- Inference latency
- Throughput
- Device compatibility

### 1.2 Why doesn’t “faster” always mean “better”?

Because many optimizations involve trade-offs:

- Smaller, but lower accuracy
- Faster, but harder to debug
- Less memory usage, but more difficult post-processing

### 1.3 An analogy

Model optimization is more like packing a suitcase.
You are not stuffing everything in blindly; instead, you are making trade-offs among:

- Size
- Weight
- Practicality

---

## 2. Five of the Most Common Optimization Paths

### 2.1 Quantization

Compress high-precision weights into lower precision.
The usual goals are:

- Reduce memory usage
- Improve throughput
- Better fit edge devices

### 2.2 Pruning

Remove unimportant weights, channels, or layers.
The usual goal is:

- Reduce computation

### 2.3 Distillation

Let a smaller model learn from a larger model’s outputs.
The usual goals are:

- Retain as much capability as possible
- Lower deployment cost

### 2.4 Operator Fusion

Merge multiple computation steps in the execution graph.
The usual goals are:

- Reduce memory reads and writes
- Improve execution efficiency

### 2.5 Batching and Scheduling Optimization

This does not change the model itself,
but changes how it runs.
The usual goal is:

- Increase throughput

---

## 3. First Run a Quantization Error Example

This example does one very direct thing:

- Quantize floating-point weights to a coarser scale
- Compute the error before and after quantization

```python
import numpy as np

weights = np.array([0.12, -1.87, 3.44, -0.03], dtype=np.float32)


def fake_quantize(values, scale):
    return np.round(values * scale) / scale


q8_like = fake_quantize(weights, scale=16)
q4_like = fake_quantize(weights, scale=4)

print("original :", weights)
print("q8_like  :", q8_like)
print("q4_like  :", q4_like)
print("q8 mae   :", np.mean(np.abs(weights - q8_like)))
print("q4 mae   :", np.mean(np.abs(weights - q4_like)))
```

### 3.1 What is the most important takeaway from this code?

Quantization is not “free compression.”
It introduces error.

So when you see:

- `int8`
- `int4`

your first reaction should not only be “this saves more,”
but also:

- How much accuracy is lost?

### 3.2 Why is lower bit width usually harder?

Because the representation space is coarser.
The more aggressively you compress, the more original detail may be lost.

---

## 4. Why Is Distillation Often Considered a “Deployment-Friendly” Approach?

### 4.1 Because it does not just compress the model, it replaces the model

The essence of distillation is not to directly modify the original model,
but to train a smaller student model.

### 4.2 What scenarios is it best suited for?

It is suitable for:

- Stable request patterns
- Clear task boundaries
- Cases where you are willing to trade training effort for deployment gains

### 4.3 How is it different from quantization?

- Quantization: keep the same model, but at lower precision
- Distillation: switch to a new, smaller model

---

## 5. How Is the Optimization Order Usually Arranged?

### 5.1 Check the pipeline first, not the trick first

First ask:

- Where is it slow?
- Where is it expensive?
- Where does memory blow up?

### 5.2 A very practical order

1. Start with runtime-level optimization
   For example: batching, caching, scheduling
2. Then apply lower-risk model optimization
   For example: int8 quantization
3. Finally consider more aggressive approaches
   For example: heavy pruning, distillation, structural changes

### 5.3 Why is this order more stable?

Because many problems are not in the model itself.
If the runtime is not tuned well,
changing the model first often brings limited benefit.

---

## 6. The Most Common Misunderstandings

### 6.1 Misunderstanding 1: Optimization means making it as small as possible

No.
The key is:

- Save as much as possible while staying within acceptable business accuracy

### 6.2 Misunderstanding 2: If you quantize it, it will definitely be faster

Not necessarily.
It also depends on:

- Whether the hardware supports it
- Whether the inference engine is well optimized

### 6.3 Misunderstanding 3: All models should use the same optimization methods

Different models, different hardware, and different business needs
often require different best paths.

---

## Summary

The most important thing in this section is not memorizing optimization methods as a glossary,
but building a deployment mindset:

> **The essence of model optimization is making trade-offs around size, latency, throughput, and hardware compatibility, rather than simply pursuing “more aggressive compression.”**

Once you are clear on this, it becomes much easier to choose the right method in actual deployment work.

---

## Exercises

1. Change the `scale` in the example to a larger or smaller value and observe how the error changes.
2. Why is quantization more like “compressing the original model,” while distillation is more like “switching to a smaller model”?
3. Think about this: if your main problem is insufficient throughput rather than running out of memory, which type of optimization should you look at first?
4. How would you determine whether a given optimization is “worth shipping”?

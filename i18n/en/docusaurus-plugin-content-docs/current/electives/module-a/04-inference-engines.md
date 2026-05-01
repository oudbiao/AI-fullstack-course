---
title: "1.4 Inference Engines"
sidebar_position: 4
description: "Starting from the roles and differences of inference engines such as ONNX Runtime, TensorRT, and OpenVINO, understand why deployment is not as simple as just “exporting the model.”"
keywords: [inference engine, ONNX Runtime, TensorRT, OpenVINO, execution graph, deployment]
---

# Inference Engines

![Inference engine and hardware adaptation diagram](/img/course/elective-inference-engine-hardware-en.png)

![Inference engine selection matrix diagram](/img/course/elective-inference-engine-selection-matrix-en.png)

:::tip Reading guide
Inference engines are not about being as fast as possible in isolation; they need to match the model format, target hardware, latency/throughput requirements, deployment environment, and the team’s maintenance capability. When reading the diagram, think of ONNX Runtime, TensorRT, and OpenVINO as toolboxes under different constraints.
:::

:::tip Section overview
A trained model does not automatically become a high-performance online service.  
There is usually a very important system component in between:

- Inference engine

It is responsible for running the model graph efficiently on a specific kind of hardware.

So the question this lesson answers is:

> **Why in deployment do we often not “just load the model weights and infer,” but instead go through an inference engine first?**
:::

## Learning objectives

- Understand the role of an inference engine in the deployment pipeline
- Distinguish what different inference engines are generally suitable for in terms of hardware and scenarios
- Use runnable examples to understand the three metrics of latency, throughput, and adaptability
- Build a first-level judgment for choosing an engine

---

## 1. What exactly does an inference engine do?

### 1.1 It is not the model itself

The model answers:

- What are the network structure and parameters?

The inference engine answers:

- How can this structure be executed more efficiently on the target device?

### 1.2 What does it usually do?

Common tasks include:

- Graph optimization
- Operator fusion
- Memory planning
- Backend kernel selection

### 1.3 An analogy

A model is like a recipe.  
An inference engine is like a kitchen scheduling system.

With the same recipe,  
using different workflows in different kitchens will lead to different speed and quality.

---

## 2. Why are there so many inference engines?

### 2.1 Because the hardware is different

Common target environments include:

- General-purpose CPU
- NVIDIA GPU
- Intel CPU / NPU
- Edge devices

### 2.2 Because the optimization goals are different

Some care more about:

- Ease of use

Others care more about:

- Extreme performance

### 2.3 So there is no “absolutely best engine”

A more reasonable question is:

- Under this kind of model, this kind of hardware, and this kind of goal, which engine is more suitable?

---

## 3. Understand “engine selection” with a small example first

This example does not actually run ONNX Runtime or TensorRT,  
but it very directly simulates:

- Latency, throughput, and adaptability scores of different engines in different scenarios

```python
engines = [
    {"name": "onnxruntime", "latency_ms": 32, "throughput_qps": 31, "hardware_fit": 8},
    {"name": "tensorrt", "latency_ms": 14, "throughput_qps": 70, "hardware_fit": 10},
    {"name": "openvino", "latency_ms": 26, "throughput_qps": 38, "hardware_fit": 9},
]


def score(engine, prefer_low_latency=True):
    latency_score = 100 / engine["latency_ms"]
    throughput_score = engine["throughput_qps"] / 10
    hardware_score = engine["hardware_fit"]

    if prefer_low_latency:
        return round(latency_score * 0.5 + throughput_score * 0.2 + hardware_score * 0.3, 2)
    return round(latency_score * 0.2 + throughput_score * 0.5 + hardware_score * 0.3, 2)


latency_first = sorted(
    [{**e, "score": score(e, True)} for e in engines],
    key=lambda x: x["score"],
    reverse=True,
)

throughput_first = sorted(
    [{**e, "score": score(e, False)} for e in engines],
    key=lambda x: x["score"],
    reverse=True,
)

print("latency_first:")
for item in latency_first:
    print(item)

print("\nthroughput_first:")
for item in throughput_first:
    print(item)
```

### 3.1 What does this code teach?

It reminds you that:

- Engine selection is not about sorting by a single metric

If you care more about:

- Low latency

and you care more about:

- High throughput

the final ranking may be different.

### 3.2 Why is this more useful than just remembering “TensorRT is faster”?

Because real decisions are never only about:

- Who is theoretically the fastest

They also include:

- Can it be integrated into the current pipeline?
- Does it support the target model?
- Is it worth adding complexity for this amount of performance gain?

---

## 4. Broad differences among several common engines

### 4.1 ONNX Runtime

It is more like a general-purpose player.  
Its strengths are usually:

- Broad ecosystem
- Strong compatibility
- Relatively balanced ease of use

### 4.2 TensorRT

It is more like a high-performance path in the NVIDIA ecosystem.  
Common characteristics:

- Strong in GPU scenarios
- Large optimization potential
- Relatively higher engineering barrier

### 4.3 OpenVINO

It is more focused on the Intel ecosystem and adaptation to specific hardware.  
Common characteristics:

- Good performance on certain CPUs / Intel devices
- Suitable for specific deployment environments

### 4.4 How should you choose among these three?

Do not ask first, “Which one is the most popular?”  
Instead ask:

- What hardware do I have?
- What is my model format?
- Do I care more about latency or maintainability?

---

## 5. What deployment outcomes can inference engines directly affect?

### 5.1 Latency

What users notice first is:

- Is it fast or not?

### 5.2 Throughput

The service side cares more about:

- How many requests can it handle at the same time?

### 5.3 Resource utilization

For example:

- Is it more memory-efficient?
- Is CPU usage more reasonable?

### 5.4 Maintenance complexity

A route with higher performance  
sometimes also means:

- More complex export
- Harder debugging
- Stronger platform lock-in

---

## 6. The most common misconceptions

### 6.1 Misconception 1: An inference engine is just “running the model with a different library”

Not really.  
It often changes:

- The graph execution method
- Optimization strategy
- Hardware utilization

### 6.2 Misconception 2: The fastest engine is the best engine

If compatibility is poor, debugging is complicated, or deployment is too difficult,  
“fastest” may not be the best choice.

### 6.3 Misconception 3: Choose the engine first, then look at the hardware

A more reasonable order is usually the opposite:

- First look at the hardware and target constraints
- Then choose the engine

---

## Summary

The most important thing in this lesson is not memorizing a few engine names,  
but building a deployment judgment:

> **An inference engine is a system layer that adapts efficient execution between the model and the hardware. Its value is not only “making it run,” but “making it faster, leaner, and more compatible.”**

Once you understand this layer clearly, learning service deployment and edge deployment later will feel much more natural.

---

## Exercises

1. Adjust the scoring weights in the example and see how the ranking changes when you “care more about hardware fit.”
2. Why is choosing an inference engine essentially a joint decision of “hardware + model + goal”?
3. If you deploy on an NVIDIA GPU, why is TensorRT often worth considering first?
4. Think about this: if the team’s maintenance capability is average, but the project needs to go online quickly, would you prefer a general-purpose engine or an extreme-optimization engine?

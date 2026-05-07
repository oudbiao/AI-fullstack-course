---
title: "E.A.7 Deployment Integrated Project"
sidebar_position: 7
description: "Combine C++, optimization, inference engines, service deployment, and edge deployment into one complete project, and build a lightweight model deployment project that can be shown in a portfolio."
keywords: [deployment project, edge inference, model serving, optimization, portfolio project]
---

# E.A.7 Deployment Integrated Project

:::tip Section Positioning
This earlier set of elective lessons covered quite a few building blocks:

- C++ basics
- Resource management
- Model optimization
- Inference engines
- Edge deployment
- Model serving

What we want to do in this section is assemble these pieces into a project that is easy to explain and easy to present.
:::

![Deployment integrated project delivery loop](/img/course/elective-deployment-project-delivery-loop-en.png)

## Learning Objectives

- Learn how to organize “model deployment capabilities” into a presentable project
- Understand the difference in presentation focus between deployment projects and training projects
- Build a project skeleton through a runnable example
- Know how to present performance, resource usage, and stability

---

## What should a deployment project highlight most?

### Not “how strong the model is,” but “how stable the system is”

The most convincing things in a deployment project are usually not:

- how impressive a single example looks

Instead, they are:

- What is the latency?
- What is the throughput?
- How much memory does it use?
- What does the deployment environment look like?
- Are rollback and monitoring considered?

### A project topic suitable for beginners

A good portfolio project topic could be:

> **Lightweight image classification service: supports local inference, batch processing, and edge device adaptation.**

This topic is good because:

- the input is clear
- the output is clear
- different deployment solutions can be compared
- optimization and service deployment can be shown

### Why is this topic better than “building a huge general-purpose system”?

Because deployment projects are most afraid of being too broad.
Clear, testable, and presentable is more important than spreading features too thin.

---

## What modules should a deployment integrated project minimally include?

At minimum, it should include:

1. Model preparation
   For example, exporting, quantization, or format conversion
2. Inference execution
   For example, choosing an inference engine
3. Service interface
   For example, an HTTP interface or a local batch-processing interface
4. Metric collection
   Latency, throughput, memory
5. Deployment documentation
   Environment, hardware, startup method

If time allows, add:

- an edge device version
- canary release or version switching
- more complete monitoring

---

## First run a project skeleton example

This example will not actually start a service,
but it directly expresses the most important planning structure of a deployment project:

```python
from dataclasses import dataclass, field


@dataclass
class DeploymentProject:
    name: str
    model_name: str
    target_device: str
    engine: str
    modules: list
    metrics: dict
    risks: list = field(default_factory=list)


project = DeploymentProject(
    name="lightweight-image-classifier-serving",
    model_name="tiny_classifier.onnx",
    target_device="raspberry_pi_5",
    engine="onnxruntime",
    modules=[
        "preprocess",
        "inference_engine",
        "postprocess",
        "batch_scheduler",
        "http_api",
        "metrics_exporter",
    ],
    metrics={
        "p95_latency_ms": 85,
        "throughput_qps": 18,
        "peak_memory_mb": 620,
    },
    risks=[
        "batch size too large increases latency",
        "edge device thermal throttling",
        "version rollback not automated yet",
    ],
)

print(project)
```

### Why is this example useful?

Because in a deployment project, the biggest risk is ending up with only:

- “I did a lot of optimization”

but not being able to explain clearly:

- What is the deployment target?
- What device is it running on?
- What are the core modules?
- What metrics prove the result?

This structured skeleton forces you to make those points clear.

### Why must a deployment project include metrics?

Because without metrics,
it is hard for others to judge whether your optimization is truly valuable.

At a minimum, it is recommended to clearly state:

- latency
- throughput
- memory or VRAM usage

---

## Recommended presentation style for deployment projects

### Start with the problem and goals

For example:

- deploy a lightweight classification model to an edge device
- offline inference is required
- target latency is under 100ms

### Then explain the system architecture

For example:

- model format
- inference engine
- service architecture
- target hardware

### Finally, present the results and trade-offs

For example:

- why choose ONNX Runtime instead of TensorRT
- why the batch size was not increased further
- why more aggressive quantization was not done yet

This part often shows engineering judgment best.

---

## Three things that are easiest to overlook in a project

### Environment reproducibility

If others cannot quickly reproduce:

- dependency versions
- startup commands
- input examples

the project’s credibility will drop a lot.

### Baseline comparison

It is best to explain:

- what it looked like before optimization
- what improved after optimization

### Failure cases

For example:

- power consumption is too high on some devices
- large batch sizes lead to poor tail latency

Writing these down often better proves that you really made engineering trade-offs.

---

## Common misunderstandings

### Misunderstanding 1: A deployment project only needs to run

Deployment projects should emphasize more:

- performance
- stability
- reproducibility

### Misunderstanding 2: Only show the interface, not the metrics

For this kind of project, metrics matter more than the UI.

### Misunderstanding 3: Trying to cover cloud, edge, and mobile scenarios all at once

A better approach is usually:

- first choose one target device and go deep

---

## Summary

The most important thing in this section is to build a deployment-project mindset:

> **The most valuable part of a deployment integrated project is not how complex the model itself is, but whether you can turn the target device, engine choice, service structure, performance metrics, and engineering trade-offs into a complete closed loop.**

As long as this loop is clear, this kind of project is very suitable for portfolio presentation.

---

## Exercises

1. Change the target device in the example from `raspberry_pi_5` to another device, and rethink the engine and metric targets.
2. Add another set of “before optimization vs. after optimization” comparison fields for your project.
3. Think about this: if you can only show 3 metrics, which 3 would you prioritize? Why?
4. How would you present this project to an interviewer, and what three parts would you focus on?

---
title: "E.A.5 Edge Device Deployment"
sidebar_position: 5
description: "Understand why edge deployment and cloud deployment are two completely different sets of engineering constraints from the perspectives of memory, power, latency, offline capability, and model size."
keywords: [edge deployment, Jetson, Raspberry Pi, memory budget, latency, offline inference]
---

# E.A.5 Edge Device Deployment

:::tip Section Overview
The easiest thing to underestimate in edge device deployment is:

- It is not “moving a cloud service onto a small machine”

Common constraints in edge environments include:

- Less memory
- More sensitive power consumption
- Unstable network
- More difficult upgrades and troubleshooting

So what you really need to learn is:

> **Build a system that is “good enough” on constrained devices, rather than chasing the perfect form on a desktop machine.**
:::

![Edge deployment constraint decision map](/img/course/elective-edge-deployment-constraint-map-en.png)

## Learning Objectives

- Understand the core differences between edge deployment and cloud deployment
- Learn how to evaluate a solution from four dimensions: memory, power, latency, and offline capability
- Understand device selection and model adaptation through runnable examples
- Build a sense of priority when making edge deployment decisions

---

## What exactly makes edge deployment difficult?

### Device resources are usually much smaller than servers

Common limitations include:

- Less available memory
- Limited CPU / GPU compute
- Limited power budget

This means many solutions that are “default okay” in the cloud become unrealistic at the edge.

### The network is not always reliable

Edge devices are often located in:

- Factories
- Retail stores
- Camera nodes
- Vehicle-mounted or mobile scenarios

Once the network becomes unstable, the system still needs basic capabilities.
That is why edge deployment pays close attention to:

- Local inference
- Caching
- Offline fallback

### Operations and maintenance cost is often higher

If a server goes down, you can usually restart it remotely, roll out gradually, or scale up.
If an edge device has a problem, troubleshooting is often much more costly.

So edge systems usually care more about:

- Stability
- Predictability
- Fewer moving parts

---

## What should you look at first when choosing edge devices?

### Memory budget

This is the first threshold.
The model, runtime, input cache, and the service itself all consume memory.

### Power budget

An edge device is not just about “can it run”; you also need to ask:

- Can it run stably for a long time?
- Can the cooling system handle it?

### Target latency

Different tasks have completely different latency requirements:

- Access control recognition: may require more real-time performance
- Batch statistics: can be a bit slower

### Whether offline operation is required

If the scenario requires:

- Working even when disconnected from the internet

then the model and dependency design will be very different.

---

## First, run a device-model compatibility example

The following example simulates:

1. A set of device resource constraints
2. A set of model deployment requirements
3. Automatically filtering out combinations that can run

```python
devices = [
    {"name": "jetson_nano", "memory_gb": 4, "power_w": 10, "offline_required": True},
    {"name": "raspberry_pi_5", "memory_gb": 8, "power_w": 8, "offline_required": True},
    {"name": "industrial_box", "memory_gb": 16, "power_w": 35, "offline_required": True},
]

models = [
    {"name": "tiny_classifier", "memory_need_gb": 1.2, "power_need_w": 4, "latency_ms": 25},
    {"name": "medium_detector", "memory_need_gb": 6.0, "power_need_w": 12, "latency_ms": 90},
    {"name": "large_vlm", "memory_need_gb": 18.0, "power_need_w": 40, "latency_ms": 250},
]


def can_deploy(device, model, latency_target_ms=120):
    return (
        device["memory_gb"] >= model["memory_need_gb"]
        and device["power_w"] >= model["power_need_w"]
        and model["latency_ms"] <= latency_target_ms
    )


for device in devices:
    candidates = [model["name"] for model in models if can_deploy(device, model)]
    print(device["name"], "->", candidates)
```

### What is the most important thing about this code?

It is not the formula itself.
It is the way it helps you establish a screening order:

1. First, check whether the resources can fit
2. Then, check whether the power budget can handle it
3. Finally, check whether the latency meets the target

### Why does “can fit” not mean “good for deployment”?

For example, a model may barely load, but:

- The latency is too high
- The power consumption is too high
- It becomes unstable after running for a long time

In that case, it is still not a good solution.

---

## The most common optimization directions for edge deployment

### Make the model smaller

Common methods:

- Quantization
- Model distillation
- A lighter architecture

### Optimize the runtime

For example:

- Choose the right inference engine
- Reduce unnecessary preprocessing
- Limit the input resolution

### Optimize at the system level

For example:

- Cache results locally
- Keep only core functions when offline
- Reduce background logging and debugging overhead

---

## Common pitfalls in edge deployment

### Mistake 1: Choose the model first, then think about the device

A more stable order is usually:

- First look at device constraints
- Then choose the model and inference engine

### Mistake 2: Only test single inference, not long-term running

Many systems are like this:

- No problem when run once
- Start slowing down or overheating after running continuously for half an hour

### Mistake 3: Assume edge devices are always online

In many real-world scenarios,
offline capability is not optional — it is a basic requirement.

---

## Summary

The most important thing in this section is to build an edge perspective:

> **Edge deployment is first a resource and stability problem, and only then a model performance problem.**

Only by considering memory, power, latency, and offline capability together
can a solution be truly reliable.

---

## Exercises

1. Adjust `latency_target_ms` in the example and see which model combinations are filtered out.
2. If the device only has 4GB of memory, would you prioritize changing the model, changing the input resolution, or changing the engine? Why?
3. Think about this: why is “long-term stable operation” more important in edge deployment than a flashy single benchmark result?
4. If the network is frequently interrupted, which two capabilities would you prioritize adding to the system?

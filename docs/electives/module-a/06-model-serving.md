---
title: "E.A.6 Model Serving"
sidebar_position: 6
description: "From request queues, batching, version routing, and health checks to service metrics, understand why model serving is a complete engineering system."
keywords: [model serving, batching, request queue, version routing, health check, deployment]
---

# E.A.6 Model Serving

![Model serving architecture](/img/course/elective-model-serving-architecture-en.png)

![Model serving metrics and version routing diagram](/img/course/elective-serving-metrics-version-routing-map-en.png)

:::tip Reading guide
After a model goes live, the most important things to watch are not just single inference time, but queue wait time, batch efficiency, P95/P99 latency, error rate, version routing, and rollback capability. When you read the diagram, think of it as a system that runs continuously over time.
:::

:::tip Where this section fits
“Getting a model to run” and “serving a model” are two different things.

The former is more like:

- Calling a model once in a script

The latter is more like:

- Receiving requests
- Queuing them
- Scheduling them
- Returning results
- Monitoring and upgrading

What this lesson solves is the step from “able to call a model” to “able to provide a model service.”
:::

## Learning Objectives

- Understand the core components of model serving
- Understand the role of request queues, batching, and version routing in a service
- Build a minimal serving flow with a runnable example
- Develop an awareness of the metrics you should care about before a model service goes live

---

## Why is model serving not finished just by “wrapping it in an API”?

### Because real requests do not arrive one by one at a steady pace

A service will face:

- Sudden traffic spikes
- Requests of different sizes
- A mix of slow and fast requests

That means you need:

- A queue
- Scheduling
- Timeouts
- Metrics

### Because models will be upgraded

After launch, you will still face:

- Canary releases for new versions
- Model rollbacks
- Multiple versions running at the same time

So serving is not just about “exposing the current model,”
it also includes how to maintain it in the future.

### An analogy

Single inference is like cooking one meal by yourself in a kitchen.
Model serving is more like running a restaurant:

- Guests line up
- The kitchen must be scheduled
- Dishes must come out consistently

---

## The most core components of serving

### Request entry point

Responsible for:

- Receiving requests
- Validating parameters
- Authenticating identity

### Queue

Responsible for:

- Buffering requests
- Smoothing traffic

### Batch processor

Responsible for combining multiple small requests to improve throughput.

### Model executor

Actually performs:

- Preprocessing
- Inference
- Postprocessing

### Version routing and health checks

Responsible for:

- Which request goes to which model version
- Whether a given instance can receive traffic

---

## First, run a minimal serving flow

This example simulates:

1. Requests entering the queue
2. The batch processor dequeuing them in batches
3. The model executor handling them together
4. Returning results by model version

```python
from collections import deque


request_queue = deque(
    [
        {"id": "req1", "text": "refund policy", "model_version": "v1"},
        {"id": "req2", "text": "invoice policy", "model_version": "v1"},
        {"id": "req3", "text": "change address", "model_version": "v2"},
        {"id": "req4", "text": "certificate instructions", "model_version": "v2"},
    ]
)


def batch_pop(queue, batch_size):
    batch = []
    while queue and len(batch) < batch_size:
        batch.append(queue.popleft())
    return batch


def model_executor(batch):
    outputs = []
    for item in batch:
        outputs.append(
            {
                "id": item["id"],
                "model_version": item["model_version"],
                "answer": f"[{item['model_version']}] processed:{item['text']}",
            }
        )
    return outputs


all_outputs = []
while request_queue:
    batch = batch_pop(request_queue, batch_size=2)
    print("batch:", batch)
    outputs = model_executor(batch)
    all_outputs.extend(outputs)

print("\noutputs:")
for item in all_outputs:
    print(item)
```

### What is the most important thing to learn from this code?

It shows the most basic operating pattern of model serving:

- Requests do not run immediately one by one
- Instead, they first enter a queue and are then scheduled by policy

### Why is batching important for serving?

Because many models achieve higher throughput when processing in batches.
If each request runs separately:

- Resource utilization may be very poor

### Why do we explicitly keep `model_version` here?

Because in real services, multiple versions often coexist.
Without a version field, gradual rollout and rollback become very difficult.

---

## What metrics should you watch most closely after launch?

### Latency

At minimum, look at:

- Average latency
- P95 latency

### Throughput

For example:

- Requests per second
- Batches per second

### Error rate

Including:

- Request failures
- Timeouts
- Internal model exceptions

### Batching efficiency

For example:

- Average batch size

If it stays too small for a long time,
it may mean the batching strategy is not really working.

---

## The easiest mistakes to make in model serving

### Mistake 1: Only looking at model inference time

Real latency usually also includes:

- Queueing
- Preprocessing
- Postprocessing
- Network overhead

### Mistake 2: Thinking bigger batches are always better

A larger batch may improve throughput,
but it may also increase single-request latency.

### Mistake 3: Replacing the production model directly without version routing

If something goes wrong, rollback becomes very painful.

---

## Summary

The most important thing in this lesson is to develop a service-oriented view:

> **Model serving is not “write an API to call the model.” It is about turning the model into a service that can be maintained sustainably, around queues, batching, versions, health checks, and metrics.**

Once you understand this layer clearly,
it will be much easier to connect the dots later when you do edge deployment and comprehensive projects.

---

## Exercises

1. Change `batch_size` in the example to `1` and `3`, and observe how the output organization changes.
2. Think about why model services must explicitly carry version information.
3. If you care more about single-request latency than total throughput, how should you tune the batching strategy?
4. Which three metrics would you prioritize after a model service goes live? Why?

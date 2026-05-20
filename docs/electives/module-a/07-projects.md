---
title: "E.A.7 Deployment Integrated Project"
sidebar_position: 7
description: "Assemble C++, optimization, inference engines, edge constraints, serving, and metrics into a small deployment portfolio project."
keywords: [deployment project, edge inference, model serving, optimization, portfolio project]
---

# E.A.7 Deployment Integrated Project

![Deployment integrated project delivery loop](/img/course/elective-deployment-project-delivery-loop-en.webp)

This project is not about training the biggest model. It is about proving that you can turn a model into a small, measurable, deployable system.

Build a simple project story:

> Lightweight image classification service with local inference, batching, metrics, and an edge-device readiness check.

## What You Need

- Python 3.10+
- No external packages
- One small model idea, real or simulated
- One target device, such as a laptop CPU, Raspberry Pi, Jetson, or cloud CPU instance

## Delivery Checklist

Your final project should show:

1. Target device and engine choice
2. Input and output examples
3. Baseline vs optimized metrics
4. Serving or batch-processing flow
5. Known failure cases
6. Reproduction commands

## Run A Project Readiness Score

Create `deployment_project_check.py`:

```python
project = {
    "name": "lightweight-image-classifier",
    "target_device": "edge-c",
    "engine": "ONNX Runtime",
    "baseline": {"latency_ms": 120, "memory_mb": 820, "accuracy": 0.904},
    "optimized": {"latency_ms": 68, "memory_mb": 430, "accuracy": 0.899},
    "evidence": ["README.md", "metrics.csv", "failure_cases.md"],
}

checks = {
    "latency_under_80": project["optimized"]["latency_ms"] < 80,
    "memory_under_512": project["optimized"]["memory_mb"] < 512,
    "accuracy_drop_ok": project["baseline"]["accuracy"] - project["optimized"]["accuracy"] <= 0.01,
    "has_failure_cases": "failure_cases.md" in project["evidence"],
}

for name, passed in checks.items():
    print(name, passed)

release_candidate = all(checks.values())
print("release_candidate:", release_candidate)
print("evidence_files:", project["evidence"])
```

Run it:

```bash
python deployment_project_check.py
```

Expected output:

```text
latency_under_80 True
memory_under_512 True
accuracy_drop_ok True
has_failure_cases True
release_candidate: True
evidence_files: ['README.md', 'metrics.csv', 'failure_cases.md']
```

This is the shape of a presentable deployment project: not just code, but evidence.

## How To Present The Project

Use this order:

1. Problem: what needs to run, where, and why.
2. Constraints: memory, latency, hardware, offline requirement.
3. Design: model format, engine, serving path.
4. Evidence: before/after metrics and failure cases.
5. Trade-off: what you did not optimize yet and why.

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
deployment_target: local inference, edge device, model server, or optimization experiment
artifact: C++ snippet, benchmark, model artifact, serving config, or deployment note
metric: latency, memory, throughput, model size, accuracy drop, or reliability
failure_check: ABI/build issue, hardware mismatch, quantization loss, or serving bottleneck
Expected_output: reproducible deployment or optimization evidence, not only theory notes
```

## Common Mistakes

- Showing only a demo interface and no metrics.
- Optimizing latency but hiding the accuracy drop.
- Claiming edge readiness without a memory or long-running test.
- Making the project too broad, such as cloud, mobile, and edge all at once.

## Practice

Add a second target device and rerun the readiness checks. Then write three README lines that explain why the chosen device and engine are reasonable.

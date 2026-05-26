---
title: "E.A.5 Edge Device Deployment"
description: "Evaluate whether a model can run on an edge device by checking memory, power, latency, and offline requirements."
sidebar:
  order: 5
head:
  - tag: meta
    attrs:
      name: keywords
      content: "edge deployment, Jetson, Raspberry Pi, memory budget, latency, offline inference"
---
![Edge deployment constraint decision map](/img/course/elective-edge-deployment-constraint-map-en.webp)

Edge deployment means the model runs near the user, camera, machine, or sensor. The main problem is not model accuracy first; it is whether the device can run the system reliably for a long time.

## What You Need

- Python 3.10+
- No external packages
- A target scenario, such as camera classification, factory inspection, or offline form reading

## The Four Checks

- **Memory**: model, runtime, input buffer, and service all need RAM.
- **Power**: a device that can run once may still overheat or throttle.
- **Latency**: some tasks need instant response; some can wait.
- **Offline mode**: if the network is unstable, the device still needs a local fallback.

## Run A Compatibility Filter

Create `edge_fit.py`:

```python
devices = [
    {"name": "edge-a", "memory_mb": 512, "power_w": 8, "offline": True},
    {"name": "edge-b", "memory_mb": 2048, "power_w": 15, "offline": False},
    {"name": "edge-c", "memory_mb": 4096, "power_w": 25, "offline": True},
]

model = {
    "name": "int8-small-classifier",
    "memory_mb": 700,
    "power_w": 10,
    "latency_ms": 65,
    "requires_offline": True,
}

for device in devices:
    reasons = []

    if device["memory_mb"] < model["memory_mb"]:
        reasons.append("memory")
    if device["power_w"] < model["power_w"]:
        reasons.append("power")
    if model["requires_offline"] and not device["offline"]:
        reasons.append("offline")

    status = "FIT" if not reasons else "CHECK " + ",".join(reasons)
    print(device["name"], status)
```

Run it:

```bash
python edge_fit.py
```

Expected output:

```text
edge-a CHECK memory,power
edge-b CHECK offline
edge-c FIT
```

Read the result from left to right: `edge-c` is not automatically the fastest or cheapest device, but it is the only one that satisfies the deployment constraints.

## Edge Review

Read an edge deployment result as a constraint table, not as a model leaderboard. A device can pass accuracy and still fail because it overheats, loses network, runs out of memory, or cannot be updated safely. That is why the compatibility filter keeps separate reasons instead of returning only one score.

When you write a project note, include the target environment: power source, network assumption, expected runtime length, input size, and how logs leave the device. These details make the difference between “the model ran once” and “the system is ready to be operated.”

## Make It More Real

Change `model["memory_mb"]` from `700` to `350` and run again. `edge-a` still fails because power is too low. This shows why edge deployment is a multi-constraint problem.

## Practical Edge Checklist

Before calling a device “ready,” verify:

1. It can start from cold boot.
2. It can run for at least 30 minutes without memory growth.
3. It handles network loss.
4. It saves enough logs for remote troubleshooting.
5. It has a simple rollback or replacement path.

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

- Picking the model first, then trying to force it onto a small device.
- Testing only one inference instead of a long-running loop.
- Assuming the device is always online.
- Forgetting that logs, caches, and input images also consume memory.

## Practice

Add `price_usd` to each device and choose the cheapest device that passes all checks. Then add a second model and compare which device works for both.

<details>
<summary>Reference implementation and walkthrough</summary>

The answer should first filter devices by constraints, then compare price only among devices that pass. A cheap device that fails memory, power, or offline requirements is not a valid deployment target.

For the second model, build a shared check such as `device["memory_mb"] >= model_a["memory_mb"] + model_b["memory_mb"]` if both models must run together, or compare each model separately if only one runs at a time. The final note should explain the trade-off: the best device is the cheapest one that still satisfies the real operating constraints.

</details>

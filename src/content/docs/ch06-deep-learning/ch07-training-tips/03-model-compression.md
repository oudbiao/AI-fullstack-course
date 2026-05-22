---
title: "6.7.4 Model Compression [Elective]"
description: "Choose quantization, pruning, or distillation from deployment constraints, then measure size, latency, and task quality."
sidebar:
  order: 3
head:
  - tag: meta
    attrs:
      name: keywords
      content: "model compression, quantization, pruning, distillation, deployment, model size"
---
:::tip[Section Overview]
Model compression is a deployment trade-off, not a magic shrink button. You compress because memory, latency, throughput, or device limits force a decision.
:::
## Learning Objectives

- Explain quantization, pruning, and distillation by what they change.
- Estimate model size from parameter count and numeric precision.
- Measure quantization error in a tiny example.
- Choose a compression path from a deployment bottleneck.
- Avoid judging compression by size alone.

---

## Start from the Deployment Bottleneck

![Model compression trade-off map](/img/course/ch06-model-compression-tradeoff-en.webp)

| Bottleneck | First method to consider | Why |
|---|---|---|
| memory too high | quantization | same parameter count, fewer bits per value |
| many redundant weights/channels | pruning | remove structure that contributes little |
| large teacher but retraining is possible | distillation | train a smaller student to imitate behavior |
| latency still high after compression | profiling first | bottleneck may be data transfer or unsupported kernels |

The important habit:

```text
measure bottleneck -> choose method -> remeasure size, latency, and metric
```

## Three Compression Paths

| Method | Changes | Typical benefit | Main risk |
|---|---|---|---|
| Quantization | numeric precision | smaller memory, sometimes faster inference | accuracy drop, hardware support issues |
| Pruning | weights, channels, or blocks | less computation if structure is actually removed | sparse speedup may not appear on all hardware |
| Distillation | training objective | smaller model with teacher-like behavior | requires retraining and teacher outputs |

Compression is not complete until the task still works after compression.

## Lab 1: Quantization Error

```python
weights = [0.12, -1.87, 3.44, -0.03]


def fake_quantize(values, scale):
    return [round(v * scale) / scale for v in values]


def mae(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


q8_like = fake_quantize(weights, scale=16)
q4_like = fake_quantize(weights, scale=4)

print("quant_error_lab")
print("original:", weights)
print("q8_like:", q8_like)
print("q4_like:", q4_like)
print("q8_mae:", round(mae(weights, q8_like), 4))
print("q4_mae:", round(mae(weights, q4_like), 4))
```

Expected output:

```text
quant_error_lab
original: [0.12, -1.87, 3.44, -0.03]
q8_like: [0.125, -1.875, 3.4375, 0.0]
q4_like: [0.0, -1.75, 3.5, 0.0]
q8_mae: 0.0106
q4_mae: 0.0825
```

More aggressive quantization usually creates more numerical error. The practical question is whether the downstream task metric still stays acceptable.

## Lab 2: Estimate Model Size

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

param_count = sum(p.numel() for p in model.parameters())

print("model_size_lab")
print("params:", param_count)

for name, bits in [("fp32", 32), ("fp16", 16), ("int8", 8), ("int4", 4)]:
    mb = param_count * bits / 8 / 1024 / 1024
    print(f"{name:>4}: {mb:.4f} MB")
```

Expected output:

```text
model_size_lab
params: 8906
fp32: 0.0340 MB
fp16: 0.0170 MB
int8: 0.0085 MB
int4: 0.0042 MB
```

![Model compression quantization and size result map](/img/course/ch06-model-compression-quant-size-result-map-en.webp)

This is an estimate for parameters only. Real deployed size can also include metadata, tokenizer files, runtime overhead, and engine-specific packaging.

## Choosing a Path

| Situation | Good first action |
|---|---|
| model does not fit in memory | try quantization first |
| model fits but latency is high | profile latency before pruning |
| most channels appear redundant | consider structured pruning |
| a smaller model must preserve behavior | distill from a teacher model |
| metric drops too much after compression | reduce compression strength or fine-tune |

For pruning, prefer structured pruning for deployment because removing whole channels or blocks is easier for hardware to exploit than random sparse weights.

For distillation, the common pattern is:

```text
teacher logits or outputs -> student learns labels + teacher behavior
```

## What to Report in a Compression Experiment

| Metric | Before | After | Why it matters |
|---|---|---|---|
| model size | required | required | did memory improve? |
| latency | required | required | did inference actually speed up? |
| throughput | useful | useful | can the service handle more requests? |
| task metric | required | required | did quality remain acceptable? |
| hardware/runtime | required | required | compression depends on deployment stack |

Never report “int8 works” without task metric and latency. Smaller is not automatically better.

## Evidence to Keep

Save compression results as a before/after report:

```text
baseline_size:
compressed_size:
baseline_latency:
compressed_latency:
baseline_metric:
compressed_metric:
runtime_hardware:
decision: keep, tune, or reject compression
```

This protects you from a common mistake: reducing file size while making the actual product slower or less accurate.

## Common Mistakes

| Mistake | Fix |
|---|---|
| compressing before measuring bottlenecks | measure memory, latency, and metric first |
| assuming quantization always speeds things up | verify hardware and runtime support |
| counting only parameter size | include tokenizer, runtime, and packaging where relevant |
| using unstructured pruning and expecting automatic speedup | benchmark on target hardware |
| ignoring accuracy after compression | compare task metric before and after |

## Exercises

1. Change `scale=16` to `scale=32` in Lab 1. Does MAE decrease?
2. Add a third Linear layer to Lab 2 and recompute model size.
3. Choose a compression strategy for a model that fits in memory but is too slow.
4. Write a before/after report template with size, latency, throughput, and metric.
5. Explain why structured pruning is usually easier to deploy than unstructured pruning.

<details>
<summary>Reference implementation and walkthrough</summary>

1. Increasing `scale` to `32` usually reduces quantization error because values are represented with finer steps. Verify with MAE instead of guessing.
2. A third `Linear` layer adds both weight and bias parameters. Recompute each layer as `in_features * out_features + out_features`.
3. If memory is acceptable but latency is too high, start with quantization, batching/runtime optimization, or distillation. Pruning helps only if the deployment runtime can exploit it.
4. A useful report compares `model_size`, `latency_p50/p95`, `throughput`, `task_metric`, hardware, batch size, and the exact compression method.
5. Structured pruning removes whole channels, heads, or blocks, so common runtimes can speed it up. Unstructured sparsity often needs special kernels to become faster.

</details>

## Key Takeaways

- Compression starts from deployment constraints.
- Quantization changes numeric precision.
- Pruning changes model structure.
- Distillation changes the training process.
- Compression is successful only if the deployed task still meets quality and latency requirements.

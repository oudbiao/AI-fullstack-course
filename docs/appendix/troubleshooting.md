---
title: "A.8 Learning Rescue for Stuck Points"
sidebar_position: 5
---

# A.8 Learning Rescue for Stuck Points

![Troubleshooting map for learning stuck points](/img/course/appendix-troubleshooting-rescue-map-en.png)

![Flowchart for minimal reproduction and asking for help](/img/course/appendix-debug-mre-help-flow-en.png)

When stuck, first turn the problem from “I cannot learn this” into “I can locate this failure.”

## First classify the problem

| Symptom | Likely problem | First move |
|---|---|---|
| `ModuleNotFoundError` | Wrong environment or missing dependency | Check Python and `pip` path |
| File not found | Wrong working directory or relative path | Print `Path.cwd()` |
| Code runs but result is strange | Input, label, or metric issue | Print samples and intermediate values |
| Training does not improve | Data, loss, learning rate, or label format | Try to overfit a tiny dataset |
| GPU memory explodes | Batch, input, or model too large | Reduce batch size first |
| Project feels too big | No minimal closed loop | Define one input, one process, one output |

## Run these checks first

```bash
python --version
which python
pip --version
pip list
pwd
ls
```

If you use NVIDIA GPU:

```bash
nvidia-smi
```

For path issues:

```python
from pathlib import Path

print(Path.cwd())
print(Path("data").exists())
```

Expected output will depend on your folder, but it should look like:

```text
/your/current/project
False
```

## Debug code in this order

1. Print the first 2 inputs and labels.
2. Print shapes, lengths, and value ranges.
3. Print one intermediate result before the model.
4. Print one model output before calculating metrics.
5. Only then change the model or parameters.

Minimal inspection example:

```python
texts = ["refund request", "invoice copy", "shipping delay"]
labels = ["support", "billing", "support"]

print("samples:", len(texts))
print("first texts:", texts[:2])
print("first labels:", labels[:2])
print("label set:", sorted(set(labels)))
```

Expected output:

```text
samples: 3
first texts: ['refund request', 'invoice copy']
first labels: ['support', 'billing']
label set: ['billing', 'support']
```

## Ask for help with a complete question

```text
What I am doing:
What I expected:
What happened:
Last 20 lines of the error:
What I already tried:
Minimal reproducible code:
```

## Minimal reproduction habit

When a project is messy, shrink it until it runs:

```python
def predict(x):
    return x * 2

data = [1, 2, 3]
preds = [predict(x) for x in data]
print(preds)
```

Expected output:

```text
[2, 4, 6]
```

Then add real logic back one layer at a time. The layer that breaks is the layer to inspect.

## Pause or keep going?

| Situation | Better action |
|---|---|
| You have tried random fixes for 30 minutes | Pause and write hypotheses |
| You cannot explain the command you are copying | Stop and inspect the environment |
| You have 1-2 clear hypotheses | Keep testing |
| You know the next observable result | Keep going |

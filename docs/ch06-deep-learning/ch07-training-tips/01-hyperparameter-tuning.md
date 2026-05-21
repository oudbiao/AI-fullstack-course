---
title: "6.7.2 Hyperparameter Tuning Strategy"
sidebar_position: 1
description: "Tune learning rate, batch size, regularization, and early stopping with controlled experiments instead of blind guessing."
keywords: [hyperparameter tuning, learning rate, batch size, regularization, experiment tracking]
---

# 6.7.2 Hyperparameter Tuning Strategy

:::tip Section Overview
Hyperparameter tuning is experiment design. Change one important thing, keep a log, compare validation evidence, then decide the next move.
:::

## Learning Objectives

- Tune in a stable order instead of changing everything at once.
- Run a small learning-rate sweep in PyTorch.
- Read validation loss, validation accuracy, and training stability together.
- Record experiment evidence in a reusable table.
- Decide when to tune learning rate, batch size, regularization, or early stopping.

---

## Use the Route First

![Deep learning tuning and diagnosis route](/img/course/ch06-training-tuning-diagnosis-route-en.webp)

The practical order:

```text
make training run -> tune learning rate -> check validation -> control overfitting -> refine locally
```

Do not start by tuning every knob. A useful tuning run should answer one question.

| Question | Parameter to try first | What to watch |
|---|---|---|
| Does the model learn at all? | learning rate | train loss trend |
| Is training unstable? | learning rate, gradient clipping, batch size | spikes or divergence |
| Is validation worse than training? | weight decay, dropout, augmentation, early stopping | generalization gap |
| Is training too slow? | batch size, model size, precision | time and memory |
| Is deployment too heavy? | architecture, pruning, quantization | latency and size |

## Lab: Run a Learning-Rate Sweep

This toy classification task is small enough to run quickly, but it shows the workflow.

Create `lr_sweep.py`:

```python
import torch
from torch import nn

torch.manual_seed(11)

X = torch.randn(240, 2)
y = ((X[:, 0] * 0.8 + X[:, 1] * -0.5) > 0).long()

train_x, val_x = X[:180], X[180:]
train_y, val_y = y[:180], y[180:]


def run(lr):
    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(40):
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_loss = loss_fn(model(train_x), train_y).item()
        val_logits = model(val_x)
        val_loss = loss_fn(val_logits, val_y).item()
        val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()

    return train_loss, val_loss, val_acc


results = []
for lr in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    train_loss, val_loss, val_acc = run(lr)
    results.append((lr, train_loss, val_loss, val_acc))

print("lr_sweep")
for lr, train_loss, val_loss, val_acc in results:
    print(
        f"lr={lr:g} "
        f"train_loss={train_loss:.3f} "
        f"val_loss={val_loss:.3f} "
        f"val_acc={val_acc:.3f}"
    )

best = min(results, key=lambda row: row[2])
print("best_lr:", best[0])
```

Run it:

```bash
python lr_sweep.py
```

Expected output:

```text
lr_sweep
lr=0.001 train_loss=0.763 val_loss=0.733 val_acc=0.450
lr=0.01 train_loss=0.675 val_loss=0.663 val_acc=0.533
lr=0.1 train_loss=0.340 val_loss=0.373 val_acc=0.967
lr=1 train_loss=0.053 val_loss=0.072 val_acc=0.983
lr=10 train_loss=0.280 val_loss=0.291 val_acc=0.883
best_lr: 1.0
```

![LR sweep output result map](/img/course/ch06-lr-sweep-result-map-en.webp)

Read it carefully:

- `0.001` and `0.01` are too slow for this budget.
- `0.1` and `1.0` learn well.
- `10.0` is worse even though it still trains, so “larger” is not automatically better.
- The best choice is based on validation loss here, not training loss.

## What to Tune Next

![Hyperparameter tuning search diagram](/img/course/hyperparameter-tuning-search-en.webp)

After a reasonable learning rate, tune in this order:

1. Batch size: adjust memory use, speed, and gradient noise.
2. Epochs and early stopping: stop when validation stops improving.
3. Weight decay and dropout: reduce overfitting.
4. Architecture size: change capacity only after the loop is stable.
5. Optimizer details: tune betas, scheduler, warmup, or momentum when needed.

The rule:

```text
global search first, local refinement later
```

## A Minimal Experiment Log

Use a log even for small projects.

```text
experiment_id:
code_version:
data_version:
seed:
lr:
batch_size:
optimizer:
weight_decay:
dropout:
epochs:
best_val_metric:
train_time:
decision:
```

Example decision text:

```text
lr=1.0 gives the best validation loss in the quick sweep.
Next: keep lr=1.0 fixed and compare batch_size=32 vs 64.
```

## Evidence to Keep

Keep one tuning decision card:

```text
question: which single variable was tested?
fixed: data split, seed, model, optimizer family, training budget
changed: learning rate values
selection_metric: validation loss or validation accuracy
best_setting: lr=1.0 in the quick sweep
next_experiment: one local refinement, not many knobs at once
```

## Diagnosis Patterns

| Pattern | Likely cause | Next experiment |
|---|---|---|
| train loss does not move | LR too low, model too small, bad labels | raise LR, inspect data, try larger model |
| train loss explodes | LR too high, gradients unstable | lower LR, add clipping |
| train good, validation poor | overfitting or leakage | add regularization, check split |
| validation improves then worsens | overfitting after best epoch | early stopping |
| results change a lot by seed | unstable training or small data | run 3 seeds and report mean/std |

## Common Mistakes

| Mistake | Fix |
|---|---|
| changing LR, batch size, optimizer, and model together | change one main variable per experiment |
| choosing by training metric | use validation metric for model selection |
| ignoring runtime | track time and memory, not only accuracy |
| trusting one lucky seed | repeat important runs with multiple seeds |
| tuning before data is clean | inspect labels, leakage, and preprocessing first |

## Exercises

1. Add `lr=0.3` and `lr=3.0` to the sweep. Which is closer to the best region?
2. Change the training budget from `40` steps to `10` steps. Does the best LR change?
3. Add a `seed` column by running each LR with two seeds.
4. Write a decision line for the next experiment after the LR sweep.
5. Explain why tuning is easier when each experiment answers one question.

<details>
<summary>Reference implementation and walkthrough</summary>

1. `lr=0.3` may be near a useful high region; `lr=3.0` is likely too aggressive. The exact answer depends on validation loss and stability.
2. With only `10` steps, a learning rate that starts fast may look best even if it later becomes unstable. Short budgets can bias the sweep.
3. A seed column reveals whether the result is stable or lucky. If two seeds disagree strongly, repeat before making a major decision.
4. A good decision line names one next experiment, such as "refine around `lr=0.1` with three seeds and 80 steps."
5. One-question experiments are easier to interpret. If you change learning rate, model size, and data at once, you cannot tell which change mattered.

</details>

## Key Takeaways

- Tuning is controlled experiment design, not guessing.
- Learning rate is usually the first knob to test.
- Validation evidence should drive decisions.
- Logs make experiments reproducible and interpretable.
- Tune broad settings first, then refine locally.

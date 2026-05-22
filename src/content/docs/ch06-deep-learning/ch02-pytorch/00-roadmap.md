---
title: "6.2.1 PyTorch Roadmap: Tensor, Autograd, Module, DataLoader, Loop"
description: "A compact PyTorch roadmap: tensors, autograd, nn.Module, Dataset/DataLoader, training loop, and debugging."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "PyTorch guide, tensor, autograd, nn.Module, DataLoader, training loop"
---
PyTorch is the framework that turns the deep learning loop into runnable code. First learn the execution order; details become easier afterward.

## Look at the Workflow First

![PyTorch chapter flowchart](/img/course/ch06-pytorch-chapter-flow-en.webp)

![NumPy to PyTorch training loop map](/img/course/ch06-numpy-to-pytorch-training-loop-map-en.webp)

```text
tensor -> model -> loss -> backward -> optimizer.step -> repeat
```

## Run Autograd Once

Create `pytorch_first_loop.py` and run it after installing `torch`.

```python
import torch

w = torch.tensor([0.0], requires_grad=True)
learning_rate = 0.2

for step in range(1, 5):
    loss = (w - 3).pow(2)
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        w.grad.zero_()
    print(step, "w=", round(w.item(), 3), "loss=", round(loss.item(), 3))
```

Expected output:

```text
1 w= 1.2 loss= 9.0
2 w= 1.92 loss= 3.24
3 w= 2.352 loss= 1.166
4 w= 2.611 loss= 0.42
```

The key PyTorch habit is visible here: compute loss, call `backward()`, update without tracking gradients, then clear old gradients.

## Learn in This Order

| Order | Read | What to practice |
|---|---|---|
| 1 | [6.2.2 sklearn to PyTorch Bridge](./00-sklearn-to-pytorch-bridge.md) | why the loop becomes explicit |
| 2 | [6.2.3 PyTorch Basics](./01-pytorch-basics.md) | tensors, dtype, shape, device |
| 3 | [6.2.4 Autograd](./02-autograd.md) | `requires_grad`, `backward`, `grad` |
| 4 | [6.2.5 nn Module](./03-nn-module.md) | model class, parameters |
| 5 | [6.2.6 Data Loading](./04-data-loading.md) | Dataset, DataLoader, batch |
| 6 | [6.2.7 Training Loop](./05-training-loop.md) | train/eval loop, loss log |
| 7 | [6.2.8 Practical Tips](./06-practical-tips.md) | shape, device, seed, debugging |
| 8 | [6.2.9 PyTorch Workshop](./07-pytorch-matplotlib-workshop.md) | run and visualize a tiny model |

## Evidence to Keep

Keep one PyTorch loop note:

```text
tensor_check: shape, dtype, device
autograd_check: loss.backward() fills gradients
module_check: named_parameters() shows trainable tensors
loader_check: one batch matches model and loss
loop_check: train/eval losses are logged separately
```

## Pass Check

You pass this roadmap when you can read a PyTorch loop and locate these five things: data batch, model output, loss, `backward()`, and optimizer update.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer connects tensors, model layers, loss, `backward()`, and optimizer updates into one training loop.
2. The evidence should include a runnable mini experiment, tensor-shape checks, and a loss or validation curve you can explain.
3. A good self-check names one failure mode such as shape mismatch, no loss decrease, overfitting, data leakage, or using Attention/Transformer words without explaining the data flow.

</details>

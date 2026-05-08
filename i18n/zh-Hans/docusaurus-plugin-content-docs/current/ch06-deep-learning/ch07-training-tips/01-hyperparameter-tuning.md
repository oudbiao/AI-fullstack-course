---
title: "6.7.2 超参数调优策略"
sidebar_position: 1
description: "用受控实验调 learning rate、batch size、正则和 early stopping，而不是盲目碰运气。"
keywords: [hyperparameter tuning, learning rate, batch size, regularization, experiment tracking]
---

# 6.7.2 超参数调优策略

:::tip 本节定位
超参数调优本质上是实验设计。一次改一个重要变量，记录结果，比较验证集证据，再决定下一步。
:::

## 学习目标

- 按稳定顺序调参，而不是一次改一堆。
- 在 PyTorch 中跑一个小型 learning-rate sweep。
- 同时阅读 validation loss、validation accuracy 和训练稳定性。
- 用可复用表格记录实验证据。
- 判断什么时候该调 learning rate、batch size、正则或 early stopping。

---

## 先看路线图

![深度学习调参与诊断路线图](/img/course/ch06-training-tuning-diagnosis-route.webp)

实操顺序：

```text
先让训练跑起来 -> 调 learning rate -> 看验证集 -> 控制过拟合 -> 局部细调
```

不要一开始就调所有旋钮。一次有用的调参实验，应该回答一个问题。

| 问题 | 优先尝试的参数 | 观察什么 |
|---|---|---|
| 模型到底能不能学？ | learning rate | train loss 趋势 |
| 训练是否不稳定？ | learning rate、gradient clipping、batch size | spike 或发散 |
| validation 比 training 差很多？ | weight decay、dropout、augmentation、early stopping | generalization gap |
| 训练太慢？ | batch size、模型大小、precision | 时间和显存 |
| 部署太重？ | 架构、pruning、quantization | latency 和 size |

## 实验：跑一个 Learning-Rate Sweep

这个 toy classification 很小，运行快，但能展示完整流程。

创建 `lr_sweep.py`：

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

运行：

```bash
python lr_sweep.py
```

预期输出：

```text
lr_sweep
lr=0.001 train_loss=0.763 val_loss=0.733 val_acc=0.450
lr=0.01 train_loss=0.675 val_loss=0.663 val_acc=0.533
lr=0.1 train_loss=0.340 val_loss=0.373 val_acc=0.967
lr=1 train_loss=0.053 val_loss=0.072 val_acc=0.983
lr=10 train_loss=0.280 val_loss=0.291 val_acc=0.883
best_lr: 1.0
```

仔细读：

- `0.001` 和 `0.01` 对这个预算太慢；
- `0.1` 和 `1.0` 学得不错；
- `10.0` 虽然还能训练，但变差了，所以不是越大越好；
- 这里按 validation loss 选，而不是按 training loss 选。

## 下一步调什么

![超参数搜索图](/img/course/hyperparameter-tuning-search.webp)

找到合理 learning rate 后，按这个顺序继续：

1. Batch size：影响显存、速度和梯度噪声。
2. Epochs 与 early stopping：验证集不再提升时停止。
3. Weight decay 与 dropout：控制过拟合。
4. 架构大小：训练循环稳定后再改容量。
5. Optimizer 细节：必要时再调 betas、scheduler、warmup 或 momentum。

规则：

```text
先做全局粗搜，再做局部细调
```

## 最小实验日志

小项目也要记录日志。

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

示例结论：

```text
lr=1.0 在 quick sweep 中验证集 loss 最好。
下一步：固定 lr=1.0，比较 batch_size=32 和 64。
```

## 诊断模式

| 现象 | 可能原因 | 下一组实验 |
|---|---|---|
| train loss 不动 | LR 太低、模型太小、标签有问题 | 提高 LR，检查数据，试大模型 |
| train loss 发散 | LR 太高、梯度不稳定 | 降低 LR，加 gradient clipping |
| train 好，validation 差 | 过拟合或泄漏 | 加正则，检查划分 |
| validation 先变好再变坏 | 最佳 epoch 后过拟合 | early stopping |
| 换 seed 后差很多 | 训练不稳定或数据太少 | 跑 3 个 seed，报 mean/std |

## 常见错误

| 错误 | 修复 |
|---|---|
| 同时改 LR、batch size、optimizer 和模型 | 每次实验只改一个主变量 |
| 按 training metric 选模型 | 用 validation metric 选 |
| 忽略运行时间 | 同时记录时间和显存 |
| 相信单个幸运 seed | 重要实验跑多个 seed |
| 数据还没清理就调参 | 先检查标签、泄漏和预处理 |

## 练习

1. 把 `lr=0.3` 和 `lr=3.0` 加入 sweep。哪个更接近最好区域？
2. 把训练预算从 `40` step 改成 `10` step。最佳 LR 会变化吗？
3. 每个 LR 跑两个 seed，并增加 `seed` 列。
4. 为 LR sweep 写一句下一步实验决策。
5. 解释为什么每个实验只回答一个问题会让调参更简单。

## 小结

- 调参是受控实验设计，不是猜。
- Learning rate 通常是第一个要测的旋钮。
- 决策应该由验证集证据驱动。
- 日志让实验可复现、可解释。
- 先粗调全局设置，再局部细调。

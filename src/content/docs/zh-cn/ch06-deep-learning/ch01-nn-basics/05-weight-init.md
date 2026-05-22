---
title: "6.1.7 权重初始化"
description: "用小型 PyTorch 实验理解 Xavier、He、PyTorch 默认初始化和常见初始化失败"
sidebar:
  order: 7
head:
  - tag: meta
    attrs:
      name: keywords
      content: "权重初始化, Xavier, Glorot, He, Kaiming, 梯度消失, 梯度爆炸"
---

# 6.1.7 权重初始化

:::tip[本节定位]
初始化决定神经网络一开始能不能带着可用信号进入训练。你通常先用 PyTorch 默认初始化，但也要知道训练异常时怎样检查 Xavier、He、全零、过小和过大的初始化。
:::
## 学习目标

- 解释为什么全零权重会破坏学习。
- 知道 Tanh/Sigmoid 常配 Xavier，ReLU 类激活常配 He。
- 在训练前运行一次信号探针。
- 在一个小型分类任务上比较初始化选择。
- 遇到早期训练不稳定时，按顺序排查，而不是随机改参数。

---

## 先看图

先别急着背公式，先看初始化要完成什么任务：

![权重初始化信号稳定图](/img/course/ch06-weight-init-signal-stability-map.webp)

从上往下读这张图：

- 前向信号不能一层层消失；
- 激活值不能一开始就大面积饱和；
- 反向梯度还要能传回来；
- 普通 `nn.Linear` 和 `nn.Conv2d` 模型，先用 PyTorch 默认值通常是好选择。

## 最小概念

神经网络训练基本上是这个循环：

1. 初始化权重；
2. 前向传播；
3. 计算损失；
4. 反向传播；
5. 优化器更新权重。

如果第 1 步就坏了，后面几步虽然还能运行，但其实是在一个很差的起点上运行。

常见失败很直观：

| 坏起点 | 会发生什么 | 你会看到什么 |
|---|---|---|
| 全零 | 神经元一直一样 | loss 不下降 |
| 太小 | 信号随深度衰减 | 深层输出接近 0 |
| 太大 | 激活饱和或爆炸 | 初始 loss 很大，梯度不稳 |
| 初始化和激活不匹配 | 非线性后的尺度不合适 | 训练慢或很脆 |

两个术语要先认识：

- `fan_in`：进入一层的输入特征数。
- `fan_out`：离开一层的输出特征数。

初始化公式会用它们来控制每一层的尺度。

## Xavier 和 He 一张表记住

第一次学不用先死背公式，先记搭配：

| 激活函数 | 常用选择 | PyTorch 函数 | 原因 |
|---|---|---|---|
| Tanh / Sigmoid | Xavier，也叫 Glorot | `nn.init.xavier_normal_` | 尽量平衡输入和输出方差 |
| ReLU / Leaky ReLU | He，也叫 Kaiming | `nn.init.kaiming_normal_` | 补偿 ReLU 把很多值变成 0 |
| 普通 PyTorch 模型但不确定 | PyTorch 默认值 | 不写手动初始化 | 适合作为第一版 baseline |

:::note[实用规则]
普通新项目不要一开始就手动初始化所有层。先用 PyTorch 默认值，确认学习率和数据流程没问题；如果信号或梯度明显异常，再检查初始化。
:::
## 实验准备

可以在 Notebook 单元格里运行，也可以保存成 `weight_init_lab.py`。

如果缺包，先安装：

```bash
pip install torch scikit-learn
```

## 实验 1：训练前检查信号

这个实验把随机数据送进 8 层网络，打印第一层和最后一层的激活统计。目标不是看准确率，而是看信号能不能穿过深层网络。

```python
import torch
import torch.nn as nn

torch.manual_seed(7)


def build_probe(activation):
    layers = []
    in_features = 32
    for _ in range(8):
        layer = nn.Linear(in_features, 128)
        layers.append(layer)
        layers.append(activation())
        in_features = 128
    return nn.Sequential(*layers)


def apply_init(model, strategy):
    for module in model:
        if isinstance(module, nn.Linear):
            if strategy == "tiny":
                nn.init.normal_(module.weight, 0.0, 0.01)
            elif strategy == "large":
                nn.init.normal_(module.weight, 0.0, 1.0)
            elif strategy == "xavier":
                nn.init.xavier_normal_(module.weight)
            elif strategy == "he":
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(module.bias)


def probe(strategy, activation_cls):
    model = build_probe(activation_cls)
    apply_init(model, strategy)
    x = torch.randn(512, 32)
    stats = []

    for layer in model:
        x = layer(x)
        if isinstance(layer, activation_cls):
            stats.append(
                {
                    "mean": x.mean().item(),
                    "std": x.std().item(),
                    "zero_ratio": (x == 0).float().mean().item(),
                    "saturated_ratio": (x.abs() > 0.98).float().mean().item(),
                }
            )

    return stats[0], stats[-1]


print("signal_probe")
for label, strategy, activation in [
    ("tiny + ReLU", "tiny", nn.ReLU),
    ("large + Tanh", "large", nn.Tanh),
    ("Xavier + Tanh", "xavier", nn.Tanh),
    ("He + ReLU", "he", nn.ReLU),
]:
    first, last = probe(strategy, activation)
    print(
        f"{label:14s} "
        f"first_std={first['std']:.4f} "
        f"last_std={last['std']:.4f} "
        f"last_zero={last['zero_ratio']:.2f} "
        f"last_saturated={last['saturated_ratio']:.2f}"
    )
```

预期输出：

```text
signal_probe
tiny + ReLU    first_std=0.0337 last_std=0.0000 last_zero=0.52 last_saturated=0.00
large + Tanh   first_std=0.9273 last_std=0.9633 last_zero=0.00 last_saturated=0.84
Xavier + Tanh  first_std=0.4872 last_std=0.2276 last_zero=0.00 last_saturated=0.00
He + ReLU      first_std=0.8304 last_std=0.6937 last_zero=0.49 last_saturated=0.19
```

这样读结果：

- `tiny + ReLU`：最后一层标准差几乎变成 0，深层信号已经衰减。
- `large + Tanh`：很多值贴近 -1 或 1，Tanh 的梯度会变弱。
- `Xavier + Tanh`：信号尺度更可控。
- `He + ReLU`：ReLU 本来就会产生很多 0，但信号还能传到深层。

## 实验 2：训练一个小分类器

现在把同样想法放进训练里比较。这是一个很小的二分类数据集，所以某些坏起点也可能被救回来。真正要观察的是初始 loss，以及全零初始化是否卡死。

```python
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

torch.manual_seed(9)

X, y = make_moons(n_samples=600, noise=0.22, random_state=9)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

train_idx, val_idx = train_test_split(
    torch.arange(len(X)),
    test_size=0.25,
    random_state=9,
    stratify=y,
)
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]


class MoonMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def apply_init(model, strategy):
    if strategy == "default":
        return

    for module in model.modules():
        if isinstance(module, nn.Linear):
            if strategy == "zeros":
                nn.init.zeros_(module.weight)
            elif strategy == "tiny":
                nn.init.normal_(module.weight, 0.0, 0.01)
            elif strategy == "large":
                nn.init.normal_(module.weight, 0.0, 1.0)
            elif strategy == "xavier":
                nn.init.xavier_normal_(module.weight)
            elif strategy == "he":
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(module.bias)


def accuracy(model, X, y):
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


def train_once(strategy):
    torch.manual_seed(9)
    model = MoonMLP()
    apply_init(model, strategy)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    start_loss = loss_fn(model(X_train), y_train).item()

    for _ in range(120):
        loss = loss_fn(model(X_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_loss = loss_fn(model(X_train), y_train).item()
    return start_loss, end_loss, accuracy(model, X_val, y_val)


print("training_probe")
for strategy in ["default", "zeros", "tiny", "large", "xavier", "he"]:
    start, end, acc = train_once(strategy)
    print(f"{strategy:8s} start_loss={start:.3f} end_loss={end:.3f} val_acc={acc:.3f}")
```

预期输出：

```text
training_probe
default  start_loss=0.671 end_loss=0.047 val_acc=0.973
zeros    start_loss=0.693 end_loss=0.693 val_acc=0.500
tiny     start_loss=0.693 end_loss=0.067 val_acc=0.973
large    start_loss=20.040 end_loss=0.068 val_acc=0.980
xavier   start_loss=0.696 end_loss=0.046 val_acc=0.980
he       start_loss=0.924 end_loss=0.053 val_acc=0.980
```

![权重初始化实验结果图](/img/course/ch06-weight-init-probe-training-result-map.webp)

重点看三件事：

- `zeros` 会卡住，因为隐藏神经元从一开始就是彼此的复制品。
- `large` 初始 loss 很大，即使这个小模型后来能恢复，也是一种警告。
- `default`、`xavier` 和 `he` 在这里都能工作，这也说明默认值适合做第一版 baseline。

## 留下的证据

保存一条初始化探针记录：

```text
错误初始化：全零会因对称性未打破而保持接近随机准确率
警告开始：大模型一开始损失非常高
可用起点：这个任务上默认 / xavier / he 初始化都能正常训练
下一次探测：把网络加深，比较哪种策略更脆弱
```

这条证据会教会真正的重点：初始化不是装饰，它决定了信号和梯度一开始是否处在可用范围。

## 排错清单

如果训练前几轮就明显不对，按这个顺序查：

1. 数据 shape 对吗？
2. 目标 dtype 对吗？`CrossEntropyLoss` 需要 `torch.long` 类型的类别标签。
3. 学习率是不是太高？
4. 激活值是不是大部分为 0、饱和、`nan` 或 `inf`？
5. 初始化和激活函数是否匹配？

不要靠猜，先用小探针：

```python
with torch.no_grad():
    sample = X_train[:32]
    out = model(sample)
    print(out.mean().item(), out.std().item(), torch.isfinite(out).all().item())
```

如果输出不是有限数，或者几乎每个值都一样，就把初始化、输入缩放和学习率一起检查。

## 练习

1. 把信号探针里的网络深度从 8 改成 20。哪种初始化最先失败？
2. 把 `MoonMLP` 里的 ReLU 改成 Tanh。Xavier 会不会更有竞争力？
3. 把 Adam 改成 `lr=0.1` 的 SGD。哪种初始化更脆？

<details>
<summary>参考实现与讲解</summary>

1. 更深的探针通常会先暴露不稳定初始化。过大或朴素随机初始化更容易让激活值或梯度爆炸/消失，往往比 He initialization 更早失败。
2. 会更有竞争力。Xavier initialization 更适合 Tanh 这类近似对称激活；如果使用 ReLU 系列激活，He initialization 通常是更稳的默认选择。
3. 使用 `SGD(lr=0.1)` 时，最脆的是激活和梯度尺度控制不好的初始化。常见现象是 loss 来回震荡、不明显下降，甚至突然发散。

</details>

## 小结

- 初始化是前向信号和反向梯度的起跑条件。
- 全零权重会破坏对称性，不要用于隐藏层。
- Xavier 适合 Tanh/Sigmoid；He 适合 ReLU 类激活。
- PyTorch 默认值通常是第一步的正确选择，但训练异常时要会用信号探针检查。

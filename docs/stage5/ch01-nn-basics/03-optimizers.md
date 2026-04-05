---
title: "1.5 梯度下降与优化器"
sidebar_position: 3
description: "掌握 SGD、Mini-batch、Momentum、Adam、AdamW 等优化器和学习率调度策略"
keywords: [优化器, SGD, Adam, AdamW, Momentum, 学习率, 学习率调度, CosineAnnealing]
---

# 梯度下降与优化器

:::tip 本节定位
在第三阶段我们学了基础梯度下降。现在深入了解深度学习中的各种优化器——**Adam 是你最常用的**，但理解背后的演化逻辑很重要。
:::

## 学习目标

- 理解批梯度下降、小批量梯度下降、随机梯度下降的区别
- 理解 Momentum 的直觉
- 🔧 掌握 Adam / AdamW 的使用
- 了解学习率调度策略

---

## 先建立一张地图

优化器这节最容易让新人学乱，因为名字很多。更好的理解顺序是：

```mermaid
flowchart LR
    A["先知道梯度是什么"] --> B["SGD：最基础的更新方式"]
    B --> C["Momentum：让更新更稳"]
    C --> D["Adam / AdamW：更常用的自适应优化器"]
    D --> E["学习率调度：让训练后期更稳"]
```

所以这节真正想解决的不是“背优化器家族谱”，而是：

- 参数到底怎么被更新
- 为什么不同优化器会有不同训练表现
- 第一次做项目时，应该先选哪个

## 这节和前两节是怎么接上的

如果你刚学完“前向传播与反向传播”，可以先这样接：

- 上一节已经解决了“梯度怎么来”
- 这一节开始解决“梯度来了以后，参数到底怎么改”

所以优化器并不是一个额外插件，而是训练闭环里的最后一环：

```mermaid
flowchart LR
    A["前向传播"] --> B["损失"]
    B --> C["反向传播得到梯度"]
    C --> D["优化器根据梯度更新参数"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#333
    style D fill:#e8f5e9,stroke:#2e7d32,color:#333
```

## 一、三种梯度下降

### 1.1 对比

| 方式 | 每次用多少数据 | 优点 | 缺点 |
|------|-------------|------|------|
| **批梯度下降（BGD）** | 全部数据 | 稳定 | 慢、内存大 |
| **随机梯度下降（SGD）** | 1 个样本 | 快、能跳出局部最优 | 噪声大、不稳定 |
| **小批量梯度下降（Mini-batch）** | 一批（32/64/128） | **兼顾速度和稳定** | 需选 batch_size |

### 1.0.1 第一次看这三种方式，最该先记什么？

不要一开始去背名词定义，先抓住这句：

> **它们的核心区别，只在于“每次更新参数时，用多少数据来估计梯度”。**

一旦这句稳住了，后面很多现象就都能解释：

- 为什么 SGD 更抖
- 为什么 BGD 更稳但更慢
- 为什么深度学习里最常见的是 mini-batch

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# 生成数据: y = 3x + 2 + noise
X = np.random.randn(200, 1)
y = 3 * X + 2 + np.random.randn(200, 1) * 0.5

def compute_loss(X, y, w, b):
    return np.mean((X * w + b - y) ** 2)

# 对比三种方式
methods = {}
for name, batch_size in [('BGD (全量)', len(X)), ('SGD (单样本)', 1), ('Mini-batch (32)', 32)]:
    w, b = 0.0, 0.0
    lr = 0.05
    losses = []
    for epoch in range(50):
        indices = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            idx = indices[start:start+batch_size]
            X_batch, y_batch = X[idx], y[idx]
            pred = X_batch * w + b
            grad_w = 2 * np.mean(X_batch * (pred - y_batch))
            grad_b = 2 * np.mean(pred - y_batch)
            w -= lr * grad_w
            b -= lr * grad_b
        losses.append(compute_loss(X, y, w, b))
    methods[name] = losses

for name, losses in methods.items():
    plt.plot(losses, label=name, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('三种梯度下降对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 二、Momentum——带惯性的下降

### 2.1 直觉

想象一个球从山坡滚下来。普通 SGD 每一步只看当前梯度方向。Momentum 让球**带上惯性**——即使遇到小坑也能滑过去。

> **v = β × v + (1-β) × gradient**
>
> **w = w - lr × v**

### 2.1.1 Momentum 到底在补 SGD 的什么短板？

最值得先记住的不是公式，而是它在补一个很实际的问题：

- SGD 每次只看当前梯度，容易左右摇摆
- Momentum 会把前几步的方向也带进来

所以你可以先把它理解成：

- SGD：只看眼前一步
- Momentum：看眼前，也保留一点前进惯性

```python
# 对比 SGD 和 Momentum
def optimize_2d(optimizer_fn, steps=100):
    """在 f(x,y) = x² + 10y² 上优化"""
    x, y = np.array(5.0), np.array(5.0)
    path = [(x, y)]
    state = {}
    for _ in range(steps):
        gx, gy = 2*x, 20*y  # 梯度
        x, y, state = optimizer_fn(x, y, gx, gy, state)
        path.append((x, y))
    return np.array(path)

def sgd(x, y, gx, gy, state, lr=0.05):
    return x - lr*gx, y - lr*gy, state

def momentum(x, y, gx, gy, state, lr=0.05, beta=0.9):
    vx = state.get('vx', 0)
    vy = state.get('vy', 0)
    vx = beta * vx + gx
    vy = beta * vy + gy
    state['vx'], state['vy'] = vx, vy
    return x - lr*vx, y - lr*vy, state

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, fn) in zip(axes, [('SGD', sgd), ('Momentum', momentum)]):
    path = optimize_2d(fn, 50)
    # 等高线
    xx, yy = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-6, 6, 100))
    zz = xx**2 + 10*yy**2
    ax.contour(xx, yy, zz, levels=20, cmap='Blues', alpha=0.5)
    ax.plot(path[:, 0], path[:, 1], 'ro-', markersize=3, linewidth=1)
    ax.set_title(name)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
plt.suptitle('SGD vs Momentum 优化路径', fontsize=13)
plt.tight_layout()
plt.show()
```

---

## 三、Adam——最常用的优化器

### 3.1 核心思想

Adam 结合了 Momentum（一阶动量）和 RMSProp（二阶动量）：
- **一阶动量 m**：梯度的移动平均（方向）
- **二阶动量 v**：梯度平方的移动平均（自适应学习率）

### 3.2 PyTorch 中使用

```python
import torch
import torch.nn as nn

# 用 PyTorch 对比不同优化器
model_configs = {
    'SGD': lambda params: torch.optim.SGD(params, lr=0.01),
    'SGD+Momentum': lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
    'Adam': lambda params: torch.optim.Adam(params, lr=0.01),
    'AdamW': lambda params: torch.optim.AdamW(params, lr=0.01, weight_decay=0.01),
}

# 简单任务: 拟合 y = sin(x)
torch.manual_seed(42)
X = torch.linspace(-3, 3, 200).unsqueeze(1)
y = torch.sin(X)

results = {}
for name, opt_fn in model_configs.items():
    model = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = opt_fn(model.parameters())
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(300):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    results[name] = losses

plt.figure(figsize=(10, 5))
for name, losses in results.items():
    plt.plot(losses, label=name, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同优化器收敛速度对比')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()
```

### 3.3 优化器选择指南

| 优化器 | 特点 | 使用场景 |
|--------|------|---------|
| **SGD** | 简单、需调学习率 | 研究实验 |
| **SGD+Momentum** | 加速收敛 | CV 经典模型 |
| **Adam** | 自适应学习率、快速收敛 | **默认首选** |
| **AdamW** | Adam + 解耦权重衰减 | **Transformer、大模型** |
| **RMSProp** | 自适应学习率 | RNN |

:::info Adam vs AdamW
Adam 把 L2 正则化混在梯度里。AdamW 把权重衰减单独做，效果更好。**现在大多数情况用 AdamW。**
:::

### 3.4 新人第一次做项目时怎么选优化器？

如果你现在还不熟，最稳的起步方式通常是：

- MLP / CNN 入门实验：先用 `Adam`
- Transformer / 更现代模型：优先考虑 `AdamW`
- 想研究更传统的优化行为：再去看 `SGD + Momentum`

先别把优化器选择想得太玄，第一轮最重要的是：

1. 模型能稳定训练
2. loss 能正常下降
3. 验证集表现别明显崩

### 3.5 为什么“学习率往往比优化器名字更重要”？

很多新人会把问题想成：

- “我是不是该把 Adam 换成更高级的优化器？”

但真实训练里，更常见的情况是：

- 优化器其实没大错
- 真正出问题的是学习率太大或太小

所以第一次排查训练不稳时，一个更稳的顺序通常是：

1. 先看学习率
2. 再看 batch size
3. 再看优化器

---

## 四、学习率调度

### 4.1 为什么需要？

固定学习率有问题：太大 → 不收敛；太小 → 太慢。**学习率调度**让学习率随训练动态调整。

### 4.2 常用策略

```python
import torch.optim.lr_scheduler as lr_scheduler

model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

schedulers = {
    'StepLR (每30步×0.1)': lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
    'CosineAnnealing': lr_scheduler.CosineAnnealingLR(optimizer, T_max=100),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, scheduler) in zip(axes, schedulers.items()):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if 'Step' in name:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    lrs = []
    for epoch in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    ax.plot(lrs, linewidth=2, color='steelblue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(name)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4.3 Warmup

先用小学习率预热几步，再逐渐增大到正常值，最后缓慢降低。**Transformer 训练的标配。**

| 策略 | 说明 | 常用场景 |
|------|------|---------|
| **StepLR** | 每 N 步乘以 γ | 简单任务 |
| **CosineAnnealing** | 余弦曲线衰减 | CNN 训练 |
| **Warmup + Cosine** | 先升后降 | **Transformer** |
| **ReduceLROnPlateau** | 验证集不降时减 | 自适应 |

### 4.4 一个新人可直接照抄的优化器选择顺序

第一次训练一个新任务时，你可以先这样试：

1. 先用 `Adam(lr=1e-3)` 或 `AdamW(lr=1e-3)`
2. 如果训练震荡，再先降学习率
3. 如果后期收敛慢，再考虑加调度器
4. 如果想更认真做对比，再去试 `SGD + Momentum`

这样通常比一开始就在一堆优化器里横跳要稳得多。

---

## 小结

| 概念 | 要点 |
|------|------|
| Mini-batch SGD | 实际训练中最常用的梯度计算方式 |
| Momentum | 给梯度加上惯性，加速收敛 |
| Adam / AdamW | 自适应学习率，**首选优化器** |
| 学习率调度 | 训练过程中动态调整学习率 |

## 这节最该带走什么

- 优化器本质上是在回答“参数怎么改”
- 学习率通常比“换优化器名字”更重要
- 第一次做项目时，先用一个稳妥默认值跑通，比追求最优更重要

如果把它再压成一句话，那就是：

> **梯度告诉你“往哪改”，优化器决定你“怎么改、改多快、改得稳不稳”。**

---

## 动手练习

### 练习 1：优化器赛马

用 `make_moons` 数据集，训练一个 MLP（PyTorch），对比 SGD、SGD+Momentum、Adam、AdamW 的收敛速度和最终准确率。

### 练习 2：学习率敏感性

用 Adam 训练同一个模型，测试学习率 0.1, 0.01, 0.001, 0.0001 的效果，画出学习曲线对比。

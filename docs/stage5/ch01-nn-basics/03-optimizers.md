---
title: "1.3 梯度下降与优化器"
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

## 一、三种梯度下降

### 1.1 对比

| 方式 | 每次用多少数据 | 优点 | 缺点 |
|------|-------------|------|------|
| **批梯度下降（BGD）** | 全部数据 | 稳定 | 慢、内存大 |
| **随机梯度下降（SGD）** | 1 个样本 | 快、能跳出局部最优 | 噪声大、不稳定 |
| **小批量梯度下降（Mini-batch）** | 一批（32/64/128） | **兼顾速度和稳定** | 需选 batch_size |

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

---

## 五、小结

| 概念 | 要点 |
|------|------|
| Mini-batch SGD | 实际训练中最常用的梯度计算方式 |
| Momentum | 给梯度加上惯性，加速收敛 |
| Adam / AdamW | 自适应学习率，**首选优化器** |
| 学习率调度 | 训练过程中动态调整学习率 |

---

## 动手练习

### 练习 1：优化器赛马

用 `make_moons` 数据集，训练一个 MLP（PyTorch），对比 SGD、SGD+Momentum、Adam、AdamW 的收敛速度和最终准确率。

### 练习 2：学习率敏感性

用 Adam 训练同一个模型，测试学习率 0.1, 0.01, 0.001, 0.0001 的效果，画出学习曲线对比。

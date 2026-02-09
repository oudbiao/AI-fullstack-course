---
title: "3.1 导数：变化率的直觉"
sidebar_position: 9
description: "理解导数的直觉含义（切线斜率=变化速度），掌握常用求导规则，用 Python 数值求导并可视化"
keywords: [导数, 微积分, 变化率, 切线斜率, Python, AI数学]
---

# 导数：变化率的直觉

:::tip 不需要你背公式
本节不会考你推导能力。核心目标是让你理解**导数 = 变化的速度**这个直觉，会用 Python 计算导数即可。后面学梯度下降时，你会发现导数就是告诉你"往哪个方向调参数能让损失变小"。
:::

## 学习目标

- 直觉理解导数 = 切线斜率 = 变化速度
- 用生活场景（速度、股价）理解导数
- 掌握常用求导规则
- 用 Python 进行数值求导和可视化

---

## 一、导数是什么？

### 1.1 生活中的"变化率"

| 场景 | 变量 | 变化率（导数） |
|------|------|--------------|
| 开车 | 距离随时间变化 | 速度（km/h） |
| 股票 | 股价随时间变化 | 涨跌速度 |
| 学习 | 分数随练习时间变化 | 学习效率 |
| AI 训练 | 损失值随训练步数变化 | 收敛速度 |

**导数 = 某个量在某一瞬间的变化速度。**

### 1.2 几何直觉：切线斜率

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 函数 f(x) = x²
def f(x):
    return x ** 2

# 在 x=1 处的切线
x0 = 1
slope = 2 * x0  # f'(x) = 2x → f'(1) = 2

x = np.linspace(-1, 3, 200)
tangent = slope * (x - x0) + f(x0)

plt.figure(figsize=(8, 6))
plt.plot(x, f(x), 'steelblue', linewidth=2, label='f(x) = x²')
plt.plot(x, tangent, 'r--', linewidth=2, label=f'切线（斜率 = {slope}）')
plt.plot(x0, f(x0), 'ro', markersize=10, zorder=5)
plt.annotate(f'x={x0}, 斜率={slope}', xy=(x0, f(x0)), 
             xytext=(x0+0.5, f(x0)+1.5), fontsize=12,
             arrowprops=dict(arrowstyle='->', color='gray'))
plt.xlim(-1, 3)
plt.ylim(-1, 8)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('导数 = 切线斜率')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**解读**：f(x) = x² 在 x=1 处的导数是 2，意思是"当 x 在 1 附近每增加一点点，f(x) 大约增加 2 倍那么多"。

### 1.3 数值求导——用 Python "近似"计算

不需要知道公式，只要能算函数值就能算导数：

**f'(x) ≈ (f(x + h) - f(x - h)) / (2h)** （h 取很小的数）

```python
def numerical_derivative(f, x, h=1e-7):
    """用中心差分法计算数值导数"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 测试：f(x) = x² 的导数应该是 2x
f = lambda x: x ** 2

for x0 in [0, 1, 2, 3]:
    approx = numerical_derivative(f, x0)
    exact = 2 * x0
    print(f"x={x0}: 数值导数={approx:.6f}, 精确导数={exact}")
```

:::tip 数值求导 vs 解析求导
- **解析求导**：用公式推导（如 (x²)' = 2x），精确但需要数学功底
- **数值求导**：用代码近似计算，简单但有微小误差
- **自动微分**（PyTorch 用的）：兼顾精确和自动化，第五阶段会学到
:::

---

## 二、常用求导规则

你不需要记住所有规则，只需要熟悉最常见的几个：

### 2.1 基本规则速查表

| 函数 | 导数 | 例子 |
|------|------|------|
| 常数 c | 0 | (5)' = 0 |
| x 的 n 次方 | n × x 的 (n-1) 次方 | (x³)' = 3x² |
| e 的 x 次方 | e 的 x 次方 | (eˣ)' = eˣ |
| ln(x) | 1/x | (ln x)' = 1/x |
| sin(x) | cos(x) | (sin x)' = cos x |

### 2.2 用 Python 验证

```python
# 验证常用导数规则
functions = [
    ("x³",      lambda x: x**3,       lambda x: 3*x**2),
    ("eˣ",      lambda x: np.exp(x),  lambda x: np.exp(x)),
    ("ln(x)",   lambda x: np.log(x),  lambda x: 1/x),
    ("sin(x)",  lambda x: np.sin(x),  lambda x: np.cos(x)),
]

print(f"{'函数':<10} {'x':<5} {'数值导数':<15} {'解析导数':<15} {'误差':<15}")
print("-" * 60)

for name, f, f_prime in functions:
    x0 = 1.0
    numerical = numerical_derivative(f, x0)
    analytical = f_prime(x0)
    error = abs(numerical - analytical)
    print(f"{name:<10} {x0:<5} {numerical:<15.8f} {analytical:<15.8f} {error:<15.2e}")
```

### 2.3 可视化：函数及其导数

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

cases = [
    ('f(x) = x²', lambda x: x**2, lambda x: 2*x),
    ('f(x) = x³', lambda x: x**3, lambda x: 3*x**2),
    ('f(x) = sin(x)', np.sin, np.cos),
    ('f(x) = eˣ', np.exp, np.exp),
]

for ax, (name, f, f_prime) in zip(axes.flat, cases):
    x = np.linspace(-2, 2, 200)
    ax.plot(x, f(x), 'steelblue', linewidth=2, label='f(x)')
    ax.plot(x, f_prime(x), 'coral', linewidth=2, linestyle='--', label="f'(x)")
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_title(name, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('函数（蓝）和导数（红）', fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 三、导数在 AI 中的角色

### 3.1 损失函数的导数 = 优化方向

```mermaid
flowchart LR
    L["损失函数 L(w)"] --> D["计算导数 dL/dw"]
    D --> U["更新参数<br/>w = w - lr × dL/dw"]
    U --> L

    style L fill:#ffebee,stroke:#c62828,color:#333
    style D fill:#e3f2fd,stroke:#1565c0,color:#333
    style U fill:#e8f5e9,stroke:#2e7d32,color:#333
```

**导数告诉你：参数应该往哪个方向调，才能让损失变小。** 这就是梯度下降的核心思想（下下节详细讲）。

### 3.2 AI 中常见函数的导数

```python
# Sigmoid 函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# ReLU 函数及其导数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.linspace(-5, 5, 200)

# Sigmoid
axes[0].plot(x, sigmoid(x), 'steelblue', linewidth=2, label='sigmoid(x)')
axes[0].plot(x, sigmoid_derivative(x), 'coral', linewidth=2, linestyle='--', label="sigmoid'(x)")
axes[0].set_title('Sigmoid 及其导数')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ReLU
axes[1].plot(x, relu(x), 'steelblue', linewidth=2, label='ReLU(x)')
axes[1].plot(x, relu_derivative(x), 'coral', linewidth=2, linestyle='--', label="ReLU'(x)")
axes[1].set_title('ReLU 及其导数')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Sigmoid 导数的问题**：在 x 远离 0 时，导数趋近 0（"梯度消失"），这就是为什么深度网络更常用 ReLU。

---

## 四、小结

| 概念 | 直觉 | Python 实现 |
|------|------|------------|
| 导数 | 函数在某点的变化速度 | `(f(x+h) - f(x-h)) / (2h)` |
| 切线斜率 | 导数的几何含义 | 画切线可视化 |
| 常用规则 | 幂函数、指数、对数、三角函数 | 用数值导数验证 |
| AI 中的角色 | 导数指示优化方向 | 梯度下降的基础 |

:::info 连接后续
- **下一节**：偏导数与梯度——多个变量时的"方向导数"
- **3.3 节**：梯度下降——用导数一步步优化模型
- **第五阶段**：PyTorch 的 `autograd` 自动帮你算导数（自动微分）
:::

---

## 动手练习

### 练习 1：数值求导

用 `numerical_derivative` 函数计算以下函数在 x=2 处的导数，和精确值对比：
1. f(x) = 3x² + 2x - 1
2. f(x) = 1/x
3. f(x) = x × sin(x)

### 练习 2：画导数图

画出 f(x) = x³ - 3x 和它的导数 f'(x) = 3x² - 3 在 [-3, 3] 范围内的图形。观察：f'(x) = 0 的地方（x = ±1），对应 f(x) 的什么特征？

### 练习 3：Sigmoid 梯度消失

画出 sigmoid 的导数图，找出导数最大值是多少，在什么位置。解释为什么这会导致"梯度消失"问题。

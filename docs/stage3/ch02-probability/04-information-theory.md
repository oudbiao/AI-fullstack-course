---
title: "2.4 信息论基础"
sidebar_position: 8
description: "理解信息量、熵、交叉熵和 KL 散度的直觉含义，理解为什么交叉熵是分类任务的损失函数"
keywords: [信息论, 熵, 交叉熵, KL散度, 损失函数, AI数学]
---

# 信息论基础

:::tip 为什么学信息论？
你在训练分类模型时用的 `CrossEntropyLoss`（交叉熵损失），名字里就有"熵"。信息论告诉你**这个损失函数到底在度量什么**，以及为什么它对分类任务这么有效。
:::

## 学习目标

- 理解信息量和熵——不确定性的度量
- 理解交叉熵——衡量两个分布的差异
- 理解 KL 散度——一个分布到另一个分布的"距离"
- 用 Python 计算和可视化

---

## 一、信息量——"惊讶程度"

### 1.1 直觉

一条消息包含的**信息量** = 它有多**出人意料**。

- "太阳从东边升起" → 信息量 ≈ 0（完全不意外）
- "今天北京下雪了"（夏天） → 信息量很大（非常意外）
- "今天北京下雪了"（冬天） → 信息量中等

**概率越低的事件，发生时带来的信息量越大。**

### 1.2 数学定义

**信息量 = -log2(概率)**

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 不同概率对应的信息量
probs = np.linspace(0.01, 1, 100)
info = -np.log2(probs)

plt.figure(figsize=(8, 5))
plt.plot(probs, info, color='steelblue', linewidth=2)
plt.xlabel('事件概率')
plt.ylabel('信息量（比特）')
plt.title('信息量 = -log₂(概率)')
plt.grid(True, alpha=0.3)

# 标注几个关键点
for p, label in [(1.0, '必然事件'), (0.5, '抛硬币'), (0.01, '罕见事件')]:
    i = -np.log2(p)
    plt.annotate(f'{label}\np={p}, info={i:.1f}bit', 
                 xy=(p, i), fontsize=10,
                 xytext=(p+0.15, i+0.5),
                 arrowprops=dict(arrowstyle='->', color='gray'))

plt.show()
```

| 事件概率 | 信息量 | 直觉 |
|---------|--------|------|
| 1.0 | 0 bit | 必然发生，没有信息 |
| 0.5 | 1 bit | 抛一次硬币，1 比特信息 |
| 0.25 | 2 bit | 猜中一个两位二进制数 |
| 0.01 | 6.64 bit | 很意外，信息量大 |

---

## 二、熵——平均不确定性

### 2.1 直觉

**熵（Entropy）= 一个分布的"平均信息量"= 系统的"平均不确定性"。**

```mermaid
flowchart LR
    A["确定性高<br/>熵低"] --> E["例：99% 晴天<br/>几乎不用猜"]
    B["不确定性高<br/>熵高"] --> F["例：50% 晴 50% 雨<br/>完全猜不到"]

    style A fill:#e8f5e9,stroke:#2e7d32,color:#333
    style B fill:#ffebee,stroke:#c62828,color:#333
```

### 2.2 公式与计算

**H(X) = -Σ p(x) × log2(p(x))**

```python
def entropy(probs):
    """计算熵（以比特为单位）"""
    probs = np.array(probs)
    # 避免 log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# 例 1：公平硬币（最大不确定性）
h1 = entropy([0.5, 0.5])
print(f"公平硬币的熵: {h1:.3f} bit")  # 1.0

# 例 2：不公平硬币
h2 = entropy([0.9, 0.1])
print(f"不公平硬币(0.9, 0.1)的熵: {h2:.3f} bit")  # 0.469

# 例 3：必然事件（无不确定性）
h3 = entropy([1.0, 0.0])
print(f"必然事件的熵: {h3:.3f} bit")  # 0.0

# 例 4：公平骰子
h4 = entropy([1/6]*6)
print(f"公平骰子的熵: {h4:.3f} bit")  # 2.585
```

### 2.3 可视化：硬币的熵随 p 变化

```python
p_values = np.linspace(0.001, 0.999, 1000)
entropies = [-p * np.log2(p) - (1-p) * np.log2(1-p) for p in p_values]

plt.figure(figsize=(8, 5))
plt.plot(p_values, entropies, color='steelblue', linewidth=2)
plt.xlabel('正面概率 p')
plt.ylabel('熵 H (bit)')
plt.title('二元分布的熵：p=0.5 时最大（最不确定）')
plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='p=0.5（最大熵）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**关键洞察**：p = 0.5 时熵最大（最不确定），p = 0 或 1 时熵为 0（完全确定）。

### 2.4 熵在 AI 中的应用

| 应用 | 说明 |
|------|------|
| 决策树 | 用**信息增益**（熵的减少量）选择最佳分割特征 |
| 模型输出 | 分类模型输出概率分布的熵越低，模型越"自信" |
| 数据压缩 | 熵是数据压缩的理论下限 |
| 语言模型 | 困惑度(Perplexity) = 2^(交叉熵)，衡量模型好坏 |

---

## 三、交叉熵——衡量"预测有多准"

### 3.1 直觉

**交叉熵 = 用分布 Q 来编码分布 P 的数据，平均每个样本需要多少比特。**

如果 Q 和 P 完全一样 → 交叉熵 = P 的熵（最小值）
如果 Q 和 P 差异大 → 交叉熵远大于 P 的熵

### 3.2 公式与计算

**H(P, Q) = -Σ p(x) × log2(q(x))**

```python
def cross_entropy(p, q):
    """计算交叉熵"""
    p, q = np.array(p), np.array(q)
    # 避免 log(0)
    q = np.clip(q, 1e-10, 1)
    return -np.sum(p * np.log2(q))

# 真实分布 P
P = [0.7, 0.2, 0.1]  # 三分类问题

# 预测分布 Q（不同质量的预测）
Q_good = [0.65, 0.25, 0.10]   # 好预测
Q_bad  = [0.33, 0.33, 0.34]   # 差预测（均匀猜）
Q_wrong = [0.1, 0.1, 0.8]     # 错误预测

print(f"P 的熵:        {entropy(P):.4f}")
print(f"好预测交叉熵:   {cross_entropy(P, Q_good):.4f}")
print(f"差预测交叉熵:   {cross_entropy(P, Q_bad):.4f}")
print(f"错误预测交叉熵: {cross_entropy(P, Q_wrong):.4f}")
```

### 3.3 交叉熵作为损失函数

在分类任务中，**最小化交叉熵 = 让模型预测尽可能接近真实分布**。

```python
# 二分类的交叉熵损失
def binary_cross_entropy(y_true, y_pred):
    """二分类交叉熵（和 PyTorch 的 BCELoss 等价）"""
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

# 真实标签
y_true = np.array([1, 0, 1, 1, 0])

# 不同质量的预测
predictions = {
    "完美预测":   np.array([1.0, 0.0, 1.0, 1.0, 0.0]),
    "好预测":     np.array([0.9, 0.1, 0.8, 0.9, 0.2]),
    "差预测":     np.array([0.6, 0.4, 0.6, 0.6, 0.4]),
    "完全错误":   np.array([0.1, 0.9, 0.1, 0.1, 0.9]),
}

for name, y_pred in predictions.items():
    loss = binary_cross_entropy(y_true, y_pred)
    print(f"{name:10s} → 交叉熵损失 = {loss:.4f}")
```

输出：
```
完美预测   → 交叉熵损失 ≈ 0.0000
好预测     → 交叉熵损失 ≈ 0.1643
差预测     → 交叉熵损失 ≈ 0.6365
完全错误   → 交叉熵损失 ≈ 2.3026
```

### 3.4 可视化：预测准确度 vs 损失

```python
# 真实标签 y=1 时，预测值 p 和损失的关系
p_pred = np.linspace(0.01, 0.99, 100)

# 当 y=1 时，loss = -log(p)
loss_y1 = -np.log(p_pred)

# 当 y=0 时，loss = -log(1-p)
loss_y0 = -np.log(1 - p_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(p_pred, loss_y1, color='steelblue', linewidth=2)
axes[0].set_xlabel('模型预测 P(y=1)')
axes[0].set_ylabel('损失')
axes[0].set_title('真实标签 y=1 时的损失\n预测越接近 1，损失越小')
axes[0].grid(True, alpha=0.3)

axes[1].plot(p_pred, loss_y0, color='coral', linewidth=2)
axes[1].set_xlabel('模型预测 P(y=1)')
axes[1].set_ylabel('损失')
axes[1].set_title('真实标签 y=0 时的损失\n预测越接近 0，损失越小')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**解读**：交叉熵损失有一个很好的性质——当预测错误时（比如 y=1 但模型输出 0.01），损失会**急剧增大**，对错误预测施加强烈惩罚。

---

## 四、KL 散度——两个分布的"距离"

### 4.1 直觉

**KL 散度（KL Divergence）= 用分布 Q 替代真实分布 P，会"多花"多少比特。**

**KL(P || Q) = 交叉熵(P, Q) - 熵(P)**

```python
def kl_divergence(p, q):
    """计算 KL 散度"""
    p, q = np.array(p), np.array(q)
    q = np.clip(q, 1e-10, 1)
    p = np.clip(p, 1e-10, 1)
    return np.sum(p * np.log2(p / q))

P = [0.7, 0.2, 0.1]
Q1 = [0.65, 0.25, 0.10]  # 接近 P
Q2 = [0.33, 0.33, 0.34]  # 远离 P

print(f"KL(P || Q1): {kl_divergence(P, Q1):.4f} (Q1 接近 P)")
print(f"KL(P || Q2): {kl_divergence(P, Q2):.4f} (Q2 远离 P)")
print(f"KL(P || P):  {kl_divergence(P, P):.4f} (P 和自己)")
```

### 4.2 KL 散度的性质

| 性质 | 说明 |
|------|------|
| 非负性 | KL(P \|\| Q) ≥ 0，等号当且仅当 P = Q |
| 不对称 | KL(P \|\| Q) ≠ KL(Q \|\| P)，不是真正的"距离" |
| P=Q 时为 0 | 两个分布完全一样时，KL 散度为 0 |

```python
# 验证不对称性
P = [0.7, 0.2, 0.1]
Q = [0.33, 0.33, 0.34]

print(f"KL(P || Q) = {kl_divergence(P, Q):.4f}")
print(f"KL(Q || P) = {kl_divergence(Q, P):.4f}")
print("两者不相等！")
```

### 4.3 可视化：KL 散度随分布差异变化

```python
# 二元分布：P = [0.8, 0.2]，让 Q 从 [0.01, 0.99] 到 [0.99, 0.01] 变化
p = 0.8
q_values = np.linspace(0.01, 0.99, 200)

kl_values = [kl_divergence([p, 1-p], [q, 1-q]) for q in q_values]

plt.figure(figsize=(8, 5))
plt.plot(q_values, kl_values, color='steelblue', linewidth=2)
plt.axvline(x=p, color='red', linestyle='--', label=f'q = p = {p}（KL=0）')
plt.xlabel('q 的值')
plt.ylabel('KL(P || Q)')
plt.title(f'KL 散度：P=[{p}, {1-p}]，Q=[q, 1-q]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 4.4 KL 散度在 AI 中的应用

| 应用 | 说明 |
|------|------|
| VAE | 让隐变量的分布接近标准正态分布：KL(q(z\|x) \|\| N(0,1)) |
| 知识蒸馏 | 让小模型的输出分布接近大模型：最小化 KL 散度 |
| RLHF | 限制微调后的模型不要偏离原始模型太远 |
| 策略优化 | PPO 算法中限制策略更新幅度 |

:::tip KL 散度在 RLHF 中的关键作用
大语言模型微调（RLHF）时，需要在"满足人类偏好"和"不偏离原始模型太远"之间平衡。KL 散度就是那个"不偏离太远"的约束。
:::

---

## 五、三者的关系

```mermaid
flowchart TD
    H["熵 H(P)<br/>P 自身的不确定性<br/>信息的下限"]
    CE["交叉熵 H(P,Q)<br/>用 Q 编码 P 的数据<br/>= 熵 + KL 散度"]
    KL["KL 散度 KL(P||Q)<br/>Q 比 P 多花的比特<br/>= 交叉熵 - 熵"]

    CE --> H
    CE --> KL
    H -.->|"H(P,Q) = H(P) + KL(P||Q)"| CE

    style H fill:#e3f2fd,stroke:#1565c0,color:#333
    style CE fill:#fff3e0,stroke:#e65100,color:#333
    style KL fill:#e8f5e9,stroke:#2e7d32,color:#333
```

**核心关系：交叉熵 = 熵 + KL 散度**

```python
P = [0.7, 0.2, 0.1]
Q = [0.5, 0.3, 0.2]

h = entropy(P)
ce = cross_entropy(P, Q)
kl = kl_divergence(P, Q)

print(f"熵 H(P):        {h:.4f}")
print(f"交叉熵 H(P,Q):  {ce:.4f}")
print(f"KL 散度:        {kl:.4f}")
print(f"H(P) + KL =     {h + kl:.4f}")  # = 交叉熵 ✓
```

:::info 为什么分类用交叉熵而不是 KL 散度？
因为在训练时，真实分布 P 是固定的（标签不变），所以 H(P) 是常数。最小化交叉熵 H(P,Q) 等价于最小化 KL(P||Q)。但交叉熵的计算更简单——不需要知道 H(P)。
:::

---

## 六、小结

| 概念 | 直觉 | 值域 |
|------|------|------|
| 信息量 | 一个事件有多"意外" | ≥ 0 |
| 熵 | 一个分布有多"不确定" | ≥ 0 |
| 交叉熵 | 预测分布和真实分布差多少 | ≥ H(P) |
| KL 散度 | 两个分布的"距离" | ≥ 0 |

:::info 本章回顾
概率与统计四节课，你学到了：
1. **概率基础**：条件概率、贝叶斯定理——用证据更新信念
2. **概率分布**：正态分布无处不在，中心极限定理
3. **统计推断**：MLE 就是交叉熵损失的来源，MAP 就是正则化
4. **信息论**（本节）：熵度量不确定性，交叉熵是分类损失，KL 散度约束分布差异

这些概念会在机器学习、深度学习、大语言模型中反复出现。

**🔀 融合学习跳转**：建议现在就去看**第四阶段·2.2 逻辑回归 + 2.3 决策树**——用刚学的概率和信息论知识理解分类算法的损失函数和信息增益。
:::

---

## 动手练习

### 练习 1：计算熵

计算以下分布的熵，解释哪个最"不确定"：
1. [0.25, 0.25, 0.25, 0.25]（四面骰子）
2. [0.97, 0.01, 0.01, 0.01]（几乎确定）
3. [0.4, 0.3, 0.2, 0.1]（不均匀）

### 练习 2：交叉熵损失

三分类问题：真实标签是第 2 类（one-hot: [0, 1, 0]），模型输出 softmax 概率为 [0.1, 0.7, 0.2]。计算交叉熵损失。如果模型输出变成 [0.05, 0.9, 0.05]，损失会怎么变？

### 练习 3：可视化 KL 散度

画一张图，展示当真实分布 P = [0.6, 0.3, 0.1] 固定时，让 Q 沿着某个参数变化（如 q1 从 0.1 到 0.9），KL(P||Q) 的变化曲线。

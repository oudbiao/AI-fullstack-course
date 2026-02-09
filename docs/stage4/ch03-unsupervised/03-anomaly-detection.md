---
title: "3.3 异常检测"
sidebar_position: 9
description: "掌握统计方法（Z-score、IQR）、Isolation Forest、One-Class SVM、LOF 等异常检测技术"
keywords: [异常检测, Isolation Forest, One-Class SVM, LOF, Z-score, IQR, 离群点]
---

# 异常检测

:::tip 本节定位
异常检测用于发现数据中**"不正常"的样本**——信用卡欺诈、网络入侵、设备故障等。与分类不同，异常通常是稀有的，没有足够的标签，所以需要特殊的算法。
:::

## 学习目标

- 掌握统计方法异常检测（Z-score、IQR）
- 掌握 Isolation Forest
- 了解 One-Class SVM
- 了解 LOF（局部离群因子）
- 能对比选择合适的方法

---

## 一、异常检测概述

### 1.1 什么是异常？

**异常（Anomaly / Outlier）**= 与大多数数据显著不同的样本。

```mermaid
flowchart LR
    D["数据集"] --> N["正常数据<br/>（99%）"]
    D --> A["异常数据<br/>（1%）"]
    A --> A1["欺诈交易"]
    A --> A2["设备故障"]
    A --> A3["网络攻击"]

    style N fill:#e8f5e9,stroke:#2e7d32,color:#333
    style A fill:#ffebee,stroke:#c62828,color:#333
```

| 应用场景 | 正常数据 | 异常数据 |
|---------|---------|---------|
| 信用卡欺诈 | 正常消费 | 盗刷交易 |
| 工业监控 | 设备正常运行 | 设备即将故障 |
| 网络安全 | 正常流量 | DDoS 攻击 |
| 医疗诊断 | 健康指标正常 | 疾病信号 |

### 1.2 为什么不用分类器？

| 问题 | 说明 |
|------|------|
| **数据不平衡** | 异常样本可能只有 0.1% |
| **缺乏标签** | 很多异常事先不知道长什么样 |
| **异常多变** | 新型欺诈手段不断出现 |

### 1.3 生成演示数据

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 正常数据：二维高斯分布
n_normal = 300
X_normal = np.random.randn(n_normal, 2) * 1.5 + [5, 5]

# 异常数据：随机散布
n_anomaly = 15
X_anomaly = np.random.uniform(0, 12, (n_anomaly, 2))

# 合并
X_all = np.vstack([X_normal, X_anomaly])
y_true = np.array([1] * n_normal + [-1] * n_anomaly)  # 1=正常, -1=异常

plt.figure(figsize=(8, 6))
plt.scatter(X_normal[:, 0], X_normal[:, 1], s=20, alpha=0.6, label='正常', color='steelblue')
plt.scatter(X_anomaly[:, 0], X_anomaly[:, 1], s=80, marker='x', color='red',
            linewidths=2, label='异常')
plt.title('异常检测演示数据')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 二、统计方法

### 2.1 Z-score 方法

**原理**：假设数据服从正态分布，距离均值超过 N 个标准差的样本视为异常。

> **Z = (x - μ) / σ**

通常 |Z| > 3 被视为异常（99.7% 的数据在 3σ 以内）。

```python
from scipy import stats

# 一维示例
data_1d = np.concatenate([np.random.randn(200) * 2 + 10, [25, -5, 30]])

z_scores = np.abs(stats.zscore(data_1d))
threshold = 3
anomalies = z_scores > threshold

plt.figure(figsize=(10, 4))
plt.scatter(range(len(data_1d)), data_1d, c=['red' if a else 'steelblue' for a in anomalies],
            s=20, alpha=0.7)
plt.axhline(y=data_1d[~anomalies].mean() + threshold * data_1d[~anomalies].std(),
            color='orange', linestyle='--', label='+3σ')
plt.axhline(y=data_1d[~anomalies].mean() - threshold * data_1d[~anomalies].std(),
            color='orange', linestyle='--', label='-3σ')
plt.title(f'Z-score 异常检测（发现 {anomalies.sum()} 个异常）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2.2 IQR 方法

**原理**：基于四分位数，不依赖正态分布假设。

- IQR = Q3 - Q1（四分位距）
- 异常：x < Q1 - 1.5×IQR 或 x > Q3 + 1.5×IQR

```python
# IQR 方法
Q1 = np.percentile(data_1d, 25)
Q3 = np.percentile(data_1d, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
anomalies_iqr = (data_1d < lower) | (data_1d > upper)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 箱线图
axes[0].boxplot(data_1d, vert=False)
axes[0].set_title('箱线图（自动标记异常值）')

# 散点图
axes[1].scatter(range(len(data_1d)), data_1d,
                c=['red' if a else 'steelblue' for a in anomalies_iqr], s=20, alpha=0.7)
axes[1].axhline(y=upper, color='orange', linestyle='--', label=f'上界={upper:.1f}')
axes[1].axhline(y=lower, color='orange', linestyle='--', label=f'下界={lower:.1f}')
axes[1].set_title(f'IQR 异常检测（发现 {anomalies_iqr.sum()} 个异常）')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2.3 Z-score vs IQR

| | Z-score | IQR |
|---|---------|-----|
| 假设 | 正态分布 | 无分布假设 |
| 鲁棒性 | 受极端值影响 | 更鲁棒 |
| 适用 | 大致正态的数据 | 任何分布 |
| 阈值 | 通常 3σ | 1.5×IQR |

---

## 三、Isolation Forest

### 3.1 原理

Isolation Forest（孤立森林）的思路非常巧妙：

**异常点更容易被"孤立"——只需很少的分割就能把它隔开。**

```mermaid
flowchart TD
    A["随机选一个特征"] --> B["随机选一个分割值"]
    B --> C["将数据分成两半"]
    C --> D{"数据被孤立了？"}
    D -->|"是"| E["路径很短 → 可能是异常"]
    D -->|"否"| A

    style E fill:#ffebee,stroke:#c62828,color:#333
```

| 概念 | 说明 |
|------|------|
| 路径长度 | 从根节点到叶节点的步数 |
| 异常分数 | 路径越短 → 分数越高 → 越可能是异常 |
| 正常点 | 被"正常"数据包围，需要更多次分割才能孤立 |

### 3.2 sklearn 实现

```python
from sklearn.ensemble import IsolationForest

# 训练 Isolation Forest
iso = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # 预估异常比例
    random_state=42
)
y_pred_iso = iso.fit_predict(X_all)  # 1=正常, -1=异常

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 预测结果
colors_pred = ['red' if p == -1 else 'steelblue' for p in y_pred_iso]
axes[0].scatter(X_all[:, 0], X_all[:, 1], c=colors_pred, s=20, alpha=0.7)
n_detected = (y_pred_iso == -1).sum()
axes[0].set_title(f'Isolation Forest 检测结果\n（检测到 {n_detected} 个异常）')

# 异常分数热图
xx, yy = np.meshgrid(np.linspace(-2, 14, 200), np.linspace(-2, 14, 200))
Z = iso.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
axes[1].contourf(xx, yy, Z, levels=20, cmap='RdBu')
axes[1].scatter(X_all[:, 0], X_all[:, 1], c=colors_pred, s=20, edgecolors='white', linewidth=0.5)
axes[1].set_title('异常分数热图\n（蓝=正常，红=异常）')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 评估
from sklearn.metrics import classification_report
print("Isolation Forest 评估:")
print(classification_report(y_true, y_pred_iso, target_names=['异常(-1)', '正常(1)']))
```

### 3.3 关键参数

| 参数 | 说明 | 推荐 |
|------|------|------|
| `n_estimators` | 树的数量 | 100（默认） |
| `contamination` | 异常比例的估计 | 根据业务设定 |
| `max_samples` | 每棵树的采样数 | 'auto' 或 256 |
| `max_features` | 每棵树使用的特征比例 | 1.0（默认） |

---

## 四、One-Class SVM

### 4.1 原理

**只用正常数据训练**，学习一个"正常"的边界。落在边界外的就是异常。

```python
from sklearn.svm import OneClassSVM

# One-Class SVM
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)  # nu ≈ 异常比例
y_pred_svm = ocsvm.fit_predict(X_all)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors_svm = ['red' if p == -1 else 'steelblue' for p in y_pred_svm]
axes[0].scatter(X_all[:, 0], X_all[:, 1], c=colors_svm, s=20, alpha=0.7)
n_detected = (y_pred_svm == -1).sum()
axes[0].set_title(f'One-Class SVM 检测结果\n（检测到 {n_detected} 个异常）')

# 决策边界
Z_svm = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)
axes[1].contourf(xx, yy, Z_svm, levels=20, cmap='RdBu')
axes[1].contour(xx, yy, Z_svm, levels=[0], colors='black', linewidths=2)
axes[1].scatter(X_all[:, 0], X_all[:, 1], c=colors_svm, s=20, edgecolors='white', linewidth=0.5)
axes[1].set_title('One-Class SVM 决策边界')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4.2 关键参数

| 参数 | 说明 |
|------|------|
| `kernel` | 核函数（'rbf' 最常用） |
| `nu` | 异常比例的上界（0 到 1） |
| `gamma` | RBF 核参数（'auto' 或 'scale'） |

---

## 五、LOF（局部离群因子）

### 5.1 原理

LOF（Local Outlier Factor）通过比较**一个点与其邻居的密度**来判断异常。

- 正常点：周围邻居密度和自己差不多
- 异常点：周围邻居密度比自己高得多（自己在"低密度区"）

LOF 的优势：**能检测到"局部异常"**——在不同密度的区域中也能工作。

### 5.2 sklearn 实现

```python
from sklearn.neighbors import LocalOutlierFactor

# LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred_lof = lof.fit_predict(X_all)

# 可视化
fig, ax = plt.subplots(figsize=(8, 6))
colors_lof = ['red' if p == -1 else 'steelblue' for p in y_pred_lof]

# LOF 分数（绝对值越大越异常）
lof_scores = -lof.negative_outlier_factor_
sizes = 20 + (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min()) * 200

ax.scatter(X_all[:, 0], X_all[:, 1], c=colors_lof, s=sizes, alpha=0.6,
           edgecolors='white', linewidth=0.5)
n_detected = (y_pred_lof == -1).sum()
ax.set_title(f'LOF 检测结果（圆越大 = 越异常，检测到 {n_detected} 个）')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 5.3 关键参数

| 参数 | 说明 | 推荐 |
|------|------|------|
| `n_neighbors` | 考虑的邻居数 | 20（默认） |
| `contamination` | 异常比例 | 'auto' 或手动设定 |

---

## 六、方法综合对比

```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

methods = {
    'Isolation Forest': IsolationForest(contamination=0.05, random_state=42),
    'One-Class SVM': OneClassSVM(nu=0.05, kernel='rbf', gamma='auto'),
    'LOF': LocalOutlierFactor(n_neighbors=20, contamination=0.05),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, model) in zip(axes, methods.items()):
    y_pred = model.fit_predict(X_all)
    colors = ['red' if p == -1 else 'steelblue' for p in y_pred]
    ax.scatter(X_all[:, 0], X_all[:, 1], c=colors, s=20, alpha=0.7)
    n_anomalies = (y_pred == -1).sum()
    ax.set_title(f'{name}\n检测到 {n_anomalies} 个异常')
    ax.grid(True, alpha=0.3)

plt.suptitle('三种异常检测方法对比', fontsize=13)
plt.tight_layout()
plt.show()
```

| | 统计方法 | Isolation Forest | One-Class SVM | LOF |
|---|---------|-----------------|---------------|-----|
| 原理 | 距离均值/四分位 | 随机分割隔离 | 学习正常边界 | 局部密度比较 |
| 速度 | 最快 | 快 | 中等 | 中等 |
| 高维数据 | 差 | 好 | 好 | 中等 |
| 局部异常 | 检测不到 | 一般 | 一般 | 好 |
| 大数据 | 好 | 好 | 差 | 中等 |
| 适用场景 | 简单、低维 | 通用首选 | 有核函数需求 | 密度不均匀 |

---

## 七、实战：信用卡欺诈检测模拟

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 模拟高度不平衡的数据
X_cc, y_cc = make_classification(
    n_samples=5000, n_features=10, n_informative=5,
    n_redundant=3, n_classes=2,
    weights=[0.97, 0.03],  # 97% 正常，3% 异常
    random_state=42
)

print(f"正常样本: {(y_cc == 0).sum()}")
print(f"异常样本: {(y_cc == 1).sum()}")

# 标准化
scaler = StandardScaler()
X_cc_scaled = scaler.fit_transform(X_cc)

# 用 Isolation Forest 检测
iso = IsolationForest(contamination=0.05, random_state=42)
y_pred = iso.fit_predict(X_cc_scaled)

# 转换标签格式（iso: 1=正常,-1=异常 → 0=正常,1=异常）
y_pred_binary = (y_pred == -1).astype(int)

print("\n检测结果:")
print(classification_report(y_cc, y_pred_binary, target_names=['正常', '异常']))

# 混淆矩阵
cm = confusion_matrix(y_cc, y_pred_binary)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['正常', '异常'])
ax.set_yticklabels(['正常', '异常'])
ax.set_xlabel('预测')
ax.set_ylabel('真实')
ax.set_title('Isolation Forest 欺诈检测混淆矩阵')

for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=16)

plt.colorbar(im)
plt.tight_layout()
plt.show()
```

---

## 八、如何选择方法？

```mermaid
flowchart TD
    Q["异常检测需求"] --> Q1{"数据维度？"}
    Q1 -->|"低维（1-3）"| STAT["统计方法<br/>Z-score / IQR"]
    Q1 -->|"中高维"| Q2{"数据量？"}
    Q2 -->|"大"| ISO["Isolation Forest<br/>（通用首选）"]
    Q2 -->|"中等"| Q3{"密度均匀？"}
    Q3 -->|"是"| ISO2["Isolation Forest"]
    Q3 -->|"否"| LOF["LOF<br/>（局部密度检测）"]

    style ISO fill:#e8f5e9,stroke:#2e7d32,color:#333
    style ISO2 fill:#e8f5e9,stroke:#2e7d32,color:#333
    style LOF fill:#fff3e0,stroke:#e65100,color:#333
```

:::tip 实用建议
1. **第一选择**：Isolation Forest（通用、快速、效果好）
2. **低维简单场景**：Z-score 或 IQR 就够了
3. **密度不均匀**：LOF
4. **设定 contamination**：根据业务知识估算异常比例
:::

---

## 九、小结

| 要点 | 说明 |
|------|------|
| 统计方法 | Z-score（正态假设）、IQR（无假设） |
| Isolation Forest | 随机分割隔离，通用首选 |
| One-Class SVM | 学习正常边界，核函数灵活 |
| LOF | 局部密度比较，能检测局部异常 |
| contamination | 大多数方法需要估算异常比例 |

:::info 连接后续
- **第 4 章**：模型评估——如何科学评估异常检测效果
- **第 5 章**：特征工程——为异常检测准备更好的特征
:::

---

## 动手练习

### 练习 1：IQR vs Z-score

生成一个包含异常值的一维数据（混合高斯 + 均匀噪声），分别用 Z-score 和 IQR 检测异常，对比结果差异。

### 练习 2：Isolation Forest 调参

用演示数据，尝试不同的 `contamination` 值（0.01, 0.05, 0.1, 0.2），观察检测结果的变化。画出 4 张子图对比。

### 练习 3：真实数据集

用 sklearn 的 `fetch_kddcup99`（网络入侵检测数据集的子集，可用 `subset='SA'`）做异常检测。对比 Isolation Forest 和 LOF 的效果。

### 练习 4：多方法融合

用相同的数据，分别用 Isolation Forest、One-Class SVM、LOF 检测异常，然后用"多数投票"（至少 2 个方法标为异常才算异常）融合结果。对比单个方法和融合方法的效果。

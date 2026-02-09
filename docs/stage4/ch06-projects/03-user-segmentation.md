---
title: "6.3 项目：用户分群分析"
sidebar_position: 21
description: "完整的聚类项目实战：RFM 模型、多种聚类算法对比、聚类结果解释与业务建议"
keywords: [用户分群, 聚类, RFM, K-Means, 降维, 客户价值分析]
---

# 项目三：用户分群分析（聚类问题）

:::tip 项目定位
用户分群是**无监督学习最常见的商业应用**。本项目用 RFM 模型构建客户特征，通过聚类发现不同价值的客户群体，并给出营销建议。
:::

## 项目概览

| 信息 | 说明 |
|------|------|
| 任务类型 | 聚类（无监督） |
| 方法 | RFM 模型 + K-Means |
| 评估指标 | 轮廓系数 |
| 涉及技能 | 特征构造、标准化、降维、聚类、业务解读 |

---

## Step 1：生成 RFM 数据

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_customers = 1000

df = pd.DataFrame({
    'customer_id': range(1, n_customers + 1),
    'recency': np.random.exponential(30, n_customers).astype(int) + 1,       # 最近一次购买距今天数
    'frequency': np.random.poisson(5, n_customers) + 1,                       # 购买频次
    'monetary': np.random.exponential(200, n_customers).round(2) + 10,        # 总消费金额
})

print(df.describe())
```

### RFM 简介

| 指标 | 含义 | 值大的含义 |
|------|------|-----------|
| **R**ecency | 最近一次购买距今天数 | 越小越好（最近买过） |
| **F**requency | 购买频次 | 越大越好（常客） |
| **M**onetary | 总消费金额 | 越大越好（高消费） |

---

## Step 2：特征标准化与聚类

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = ['recency', 'frequency', 'monetary']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# 肘部法选 K
inertias = []
sil_scores = []
K_range = range(2, 9)

from sklearn.metrics import silhouette_score
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inertia')
axes[0].set_title('肘部法')

axes[1].plot(K_range, sil_scores, 'ro-')
axes[1].set_xlabel('K')
axes[1].set_ylabel('轮廓系数')
axes[1].set_title('轮廓系数法')

plt.tight_layout()
plt.show()
```

---

## Step 3：聚类与可视化

```python
from sklearn.decomposition import PCA

# 选择最佳 K
best_k = K_range[np.argmax(sil_scores)]
print(f"最佳 K: {best_k}, 轮廓系数: {max(sil_scores):.4f}")

km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = km.fit_predict(X_scaled)

# PCA 降维可视化
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['cluster'], cmap='Set2', s=15, alpha=0.6)
plt.colorbar(scatter, label='聚类')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('用户分群（PCA 投影）')
plt.show()
```

---

## Step 4：聚类结果解读

```python
# 各群体的 RFM 统计
cluster_summary = df.groupby('cluster')[features].mean().round(1)
cluster_summary['客户数'] = df.groupby('cluster').size()
print(cluster_summary)

# 雷达图
fig, axes = plt.subplots(1, best_k, figsize=(4*best_k, 4), subplot_kw=dict(polar=True))
if best_k == 1:
    axes = [axes]

angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]

# 归一化到 [0, 1] 以便比较
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
radar_data = mms.fit_transform(cluster_summary[features])
# recency 越小越好，翻转
radar_data[:, 0] = 1 - radar_data[:, 0]

for i, ax in enumerate(axes):
    values = radar_data[i].tolist() + [radar_data[i][0]]
    ax.fill(angles, values, alpha=0.25)
    ax.plot(angles, values, linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['近期活跃', '购买频次', '消费金额'])
    ax.set_title(f'群体 {i}', pad=15)

plt.suptitle('各群体 RFM 雷达图', y=1.02, fontsize=13)
plt.tight_layout()
plt.show()
```

---

## Step 5：业务建议

```python
# 根据聚类特征给出标签和建议
print("\n=== 业务建议 ===")
for i in range(best_k):
    row = cluster_summary.loc[i]
    label = ""
    suggestion = ""
    if row['recency'] < cluster_summary['recency'].median() and row['monetary'] > cluster_summary['monetary'].median():
        label = "高价值活跃客户"
        suggestion = "VIP 服务、专属优惠，保持忠诚度"
    elif row['recency'] > cluster_summary['recency'].median() and row['monetary'] > cluster_summary['monetary'].median():
        label = "高价值流失风险"
        suggestion = "召回营销、个性化推荐"
    elif row['frequency'] > cluster_summary['frequency'].median():
        label = "高频低消客户"
        suggestion = "提升客单价、交叉销售"
    else:
        label = "低活跃客户"
        suggestion = "低成本触达、优惠券激活"

    print(f"  群体 {i} ({int(row['客户数'])} 人): {label}")
    print(f"    建议: {suggestion}")
```

---

## 项目检查清单

- [ ] 构建 RFM 特征
- [ ] 标准化 + 肘部法/轮廓系数选 K
- [ ] PCA 降维可视化聚类结果
- [ ] 分析各群体的 RFM 特征
- [ ] 给出可执行的业务建议

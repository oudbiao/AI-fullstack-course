---
title: "6.2 项目：房价预测"
sidebar_position: 19
description: "完整的回归项目实战：EDA、特征工程、多模型对比、模型融合与调优"
keywords: [房价预测, 回归, EDA, 特征工程, XGBoost, 模型融合, Kaggle]
---

# 项目一：房价预测（回归问题）

:::tip 项目定位
这是你的**第一个完整 ML 回归项目**。从数据探索到模型部署，走完完整流程。使用 sklearn 内置的加州房价数据集。
:::

## 项目概览

| 信息 | 说明 |
|------|------|
| 任务类型 | 回归 |
| 数据集 | California Housing（sklearn 内置） |
| 评估指标 | RMSE、R² |
| 涉及技能 | EDA、特征工程、多模型对比、调参 |

---

## 先建立一张地图

这个项目最适合拿来练“一个回归项目到底应该怎么长出来”。

```mermaid
flowchart LR
    A["看数据和目标分布"] --> B["先做线性回归 baseline"]
    B --> C["做少量特征工程"]
    C --> D["对比树模型 / GBDT"]
    D --> E["调参"]
    E --> F["残差分析和解释"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#333
    style F fill:#e8f5e9,stroke:#2e7d32,color:#333
```

如果你第一次做回归项目，就按这条线走，通常最稳。

## 这题你真正要练什么

这个项目不只是“把回归模型跑通”，更重要的是练这 4 件事：

1. 从数据探索里找到有用线索
2. 先立一个简单 baseline
3. 通过特征工程和模型对比提升效果
4. 用误差分析解释模型哪里做得好、哪里做得差

## 推荐推进顺序

更适合新人的顺序通常是：

1. 先做一个最简单的线性回归 baseline
2. 再做基本特征工程
3. 再上树模型或 GBDT
4. 最后才做调参

如果一开始就直接上复杂模型，你通常会失去对问题本身的感觉。

## 第一版最重要的目标，不是高分

这一题第一版最重要的目标其实只有两个：

1. 确认这个问题用回归建模是通的
2. 建一个后面所有改进都能对照的 baseline

也就是说，第一版做得“简单但完整”，比一开始就做得“复杂但说不清楚”更有价值。

## Step 1：数据加载与探索

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# 加载数据
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target  # 房价中位数（万美元）

print(f"数据形状: {df.shape}")
print(df.describe())

# 目标分布
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['MedHouseVal'].hist(bins=50, ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('房价分布')
axes[0].set_xlabel('房价中位数（万美元）')

# 相关性
corr = df.corr()['MedHouseVal'].drop('MedHouseVal').sort_values()
corr.plot.barh(ax=axes[1], color='coral')
axes[1].set_title('各特征与房价的相关性')
plt.tight_layout()
plt.show()
```

### Step 1.1 这一步最该问什么

第一次看数据时，不要只急着画图。先问这几个问题：

- 目标值分布是否偏斜
- 有没有特别明显的异常值区间
- 哪些特征可能和价格强相关
- 哪些特征很可能只是弱信号

这几问会直接决定你后面：

- 先上什么 baseline
- 特征工程优先做哪几项
- 误差分析该切哪些维度看

---

## Step 2：特征工程

```python
# 构造新特征
df['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
df['bedrooms_ratio'] = df['AveBedrms'] / df['AveRooms']
df['population_per_household'] = df['Population'] / df['HouseAge']

# 准备数据
from sklearn.model_selection import train_test_split

feature_cols = [c for c in df.columns if c != 'MedHouseVal']
X = df[feature_cols]
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
```

### Step 2.1 第一次做特征工程时，为什么要克制一点

回归项目里最常见的错误之一，就是一口气构造太多特征，最后自己也说不清到底哪个有效。  
更稳的做法是：

- 先只加 2~3 个最有解释力的新特征
- 每次加完都和 baseline 对比
- 如果没有明显收益，就不要因为“看起来高级”而硬留

---

## Step 3：多模型对比

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

models = {
    '线性回归': make_pipeline(StandardScaler(), LinearRegression()),
    'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
    'GBDT': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'R²': r2}
    print(f"{name:10s} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

# 可视化
fig, ax = plt.subplots(figsize=(8, 5))
names = list(results.keys())
r2s = [v['R²'] for v in results.values()]
bars = ax.bar(names, r2s, color=['steelblue', 'coral', 'seagreen', 'gold'], alpha=0.8)
for bar, score in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{score:.4f}', ha='center')
ax.set_ylabel('R²')
ax.set_title('模型 R² 对比')
ax.grid(axis='y', alpha=0.3)
plt.show()
```

### Step 3.1 模型对比时最该看什么

不要只看哪个 `R²` 最高。更稳的对比方式是同时看：

- `RMSE` 有没有明显下降
- 模型复杂度是不是高了很多
- 可解释性有没有明显变差

第一次做回归项目时，最值得珍惜的不是“最高分模型”，而是：

- 你知道为什么它比 baseline 好
- 你知道好在什么地方

---

## Step 4：模型调优

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 20),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.7, 0.3),
}

rs = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_dist, n_iter=30, cv=5,
    scoring='neg_root_mean_squared_error',
    random_state=42, n_jobs=-1
)
rs.fit(X_train, y_train)

y_pred_best = rs.predict(X_test)
print(f"调优后 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.4f}")
print(f"调优后 R²: {r2_score(y_test, y_pred_best):.4f}")
print(f"最佳参数: {rs.best_params_}")
```

---

## Step 5：结果分析

```python
# 预测 vs 实际
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test, y_pred_best, s=5, alpha=0.3)
axes[0].plot([0, 5], [0, 5], 'r--')
axes[0].set_xlabel('实际房价')
axes[0].set_ylabel('预测房价')
axes[0].set_title('预测 vs 实际')

# 特征重要性
importance = rs.best_estimator_.feature_importances_
sorted_idx = np.argsort(importance)
axes[1].barh(range(len(sorted_idx)), importance[sorted_idx], color='coral')
axes[1].set_yticks(range(len(sorted_idx)))
axes[1].set_yticklabels(np.array(feature_cols)[sorted_idx])
axes[1].set_title('特征重要性')

plt.tight_layout()
plt.show()
```

### Step 5.1 残差分析比最终分数更能体现你会不会做项目

很多人做到这里就停在 `RMSE` 和 `R²`。但真正能拉开项目质量的，往往是残差分析：

- 哪些价格区间误差特别大
- 模型是不是系统性低估高价房
- 哪些区域特征组合最容易预测错

这一步会直接决定你下一轮到底该：

- 补特征
- 换模型
- 还是重新看数据切分方式

---

## 一个很适合新人的最小复盘表

你可以直接做这样一张表：

| 版本 | 做了什么改动 | RMSE | R² | 我的判断 |
|---|---|---|---|---|
| baseline | 线性回归 | - | - | 先建立下界 |
| v2 | 加 2~3 个特征 | - | - | 看特征工程是否真有帮助 |
| v3 | 换树模型 / GBDT | - | - | 看非线性模型是否更合适 |

这张表会让你的项目从“跑过代码”变成“有清楚迭代过程”。

## 如果继续把这个项目往上做，最值得补什么

更值得优先补的通常是：

1. 残差分布分析
2. 不同区域 / 房价区间上的误差对比
3. baseline 到最优模型的完整版本演化说明

这样项目会更像一个真正做过建模和复盘的回归作品。

## 项目交付时最好补上的内容

- 一张“真实值 vs 预测值”的图
- 一段对误差来源的说明
- 一份 baseline 和改进版的对比表
- 一段“如果继续做，我会优先改什么”的总结

## 做成作品集时，最值得展示什么

如果你想把这题做成作品集页，最值得展示的不是一长串模型名字，而是：

1. 你的 baseline 是什么
2. 你做了哪一轮最有效的改进
3. 改进前后 `RMSE / R²` 怎么变了
4. 你通过残差分析发现了什么
5. 你下一步准备怎么继续提升

---

## 项目检查清单

- [ ] 完成 EDA：分布、相关性、缺失值
- [ ] 特征工程：构造至少 2 个新特征
- [ ] 至少对比 3 种模型
- [ ] 对最佳模型做超参数调优
- [ ] 残差分析和特征重要性分析

---
title: "6.1 项目：房价预测"
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

---

## 项目检查清单

- [ ] 完成 EDA：分布、相关性、缺失值
- [ ] 特征工程：构造至少 2 个新特征
- [ ] 至少对比 3 种模型
- [ ] 对最佳模型做超参数调优
- [ ] 残差分析和特征重要性分析

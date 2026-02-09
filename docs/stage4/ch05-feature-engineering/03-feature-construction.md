---
title: "5.3 特征构造"
sidebar_position: 16
description: "掌握多项式特征、交互特征、时间特征提取、统计特征和领域知识驱动的特征设计"
keywords: [特征构造, 多项式特征, 交互特征, 时间特征, 统计特征, 特征工程]
---

# 特征构造

:::tip 本节定位
特征构造是从已有数据中**创造新特征**，往往是提升模型效果最有效的手段。Kaggle 竞赛的胜负，常常取决于谁构造了更好的特征。
:::

## 学习目标

- 掌握多项式特征与交互特征
- 掌握时间特征提取
- 掌握统计特征（分组统计）
- 理解领域知识驱动的特征设计

---

## 一、多项式特征与交互特征

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd

# 原始特征
X = np.array([[2, 3], [4, 5]])
feature_names = ['x1', 'x2']

# 二阶多项式（包含交互项）
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print("原始特征:", feature_names)
print("多项式特征:", poly.get_feature_names_out(feature_names))
print(f"特征数: {X.shape[1]} → {X_poly.shape[1]}")
```

| 原始 | 生成 | 说明 |
|------|------|------|
| x1, x2 | x1², x2² | 二次项 |
| x1, x2 | x1×x2 | 交互项 |

:::warning 注意
多项式特征会让特征数**爆炸式增长**。10 个特征的 3 阶多项式会产生 286 个特征。通常只用 `degree=2`，并配合特征选择。
:::

---

## 二、时间特征提取

```python
# 从日期中提取丰富的特征
dates = pd.date_range('2024-01-01', periods=100, freq='D')
df_time = pd.DataFrame({'date': dates})

df_time['year'] = df_time['date'].dt.year
df_time['month'] = df_time['date'].dt.month
df_time['day'] = df_time['date'].dt.day
df_time['dayofweek'] = df_time['date'].dt.dayofweek     # 0=周一, 6=周日
df_time['is_weekend'] = df_time['dayofweek'].isin([5, 6]).astype(int)
df_time['quarter'] = df_time['date'].dt.quarter
df_time['day_of_year'] = df_time['date'].dt.dayofyear

print(df_time.head(10))
```

| 提取特征 | 适用场景 |
|---------|---------|
| 年/月/日 | 趋势和季节性 |
| 星期几/是否周末 | 消费行为差异 |
| 小时/分钟 | 日内模式 |
| 季度 | 季度性业务分析 |
| 距某事件的天数 | 节假日效应 |

---

## 三、统计特征（分组统计）

```python
import seaborn as sns

df = sns.load_dataset('tips')

# 基于分组的统计特征
df['avg_tip_by_day'] = df.groupby('day')['tip'].transform('mean')
df['max_bill_by_time'] = df.groupby('time')['total_bill'].transform('max')
df['tip_pct'] = df['tip'] / df['total_bill']
df['bill_rank_in_day'] = df.groupby('day')['total_bill'].rank(pct=True)

print(df[['day', 'total_bill', 'tip', 'avg_tip_by_day', 'tip_pct', 'bill_rank_in_day']].head(10))
```

| 统计类型 | 示例 | 场景 |
|---------|------|------|
| 分组均值 | 每天的平均消费 | 同组对比 |
| 分组计数 | 每个用户的订单数 | 活跃度 |
| 排名/百分位 | 消费在同组中的排名 | 相对位置 |
| 差值/比例 | 小费/账单比例 | 派生指标 |

---

## 四、领域知识驱动

**好的特征往往来自对业务的理解：**

| 领域 | 原始特征 | 构造特征 |
|------|---------|---------|
| 电商 | 总消费、订单数 | 客单价 = 总消费/订单数 |
| 房产 | 面积、房间数 | 每房间面积 = 面积/房间数 |
| 金融 | 收入、负债 | 负债比 = 负债/收入 |
| 用户 | 注册时间、最后登录 | 沉默天数 = 今天 - 最后登录 |

```python
# 房价数据的领域特征示例
np.random.seed(42)
house = pd.DataFrame({
    'area': np.random.uniform(50, 200, 100),
    'rooms': np.random.randint(1, 6, 100),
    'floor': np.random.randint(1, 30, 100),
    'age': np.random.randint(0, 30, 100),
})

# 领域特征
house['area_per_room'] = house['area'] / house['rooms']
house['is_new'] = (house['age'] <= 5).astype(int)
house['is_high_floor'] = (house['floor'] >= 15).astype(int)

print(house.head())
```

---

## 五、小结

| 方法 | 说明 | 要点 |
|------|------|------|
| 多项式/交互 | 自动生成高阶和组合特征 | 注意特征爆炸 |
| 时间特征 | 从日期中提取周期信息 | 周几、月份、是否假日 |
| 统计特征 | 分组聚合生成相对指标 | transform 保持行数不变 |
| 领域知识 | 基于业务理解构造 | 最有效但依赖经验 |

---

## 动手练习

### 练习 1：Titanic 特征构造

在 Titanic 数据集上构造：家庭大小（sibsp+parch+1）、是否独自旅行、票价分段、姓名中的称谓。观察对模型的提升。

### 练习 2：时间序列特征

生成一年的日期数据，提取所有时间特征（月、周、季度、是否工作日），用柱状图展示不同特征的分布。

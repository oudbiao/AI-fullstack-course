---
title: "5.2 特征预处理"
sidebar_position: 15
description: "掌握缺失值处理、标准化与归一化、类别特征编码和异常值处理"
keywords: [缺失值, 标准化, 归一化, One-Hot, Label Encoding, Target Encoding, 异常值]
---

# 特征预处理

:::tip 本节定位
原始数据很少能直接喂给模型——缺失值、不同尺度、类别文本都需要先处理。本节是特征工程中**最基础也最常用**的技能。
:::

## 学习目标

- 掌握缺失值处理策略
- 掌握标准化和归一化
- 掌握类别特征编码
- 掌握异常值处理

---

## 一、缺失值处理

### 1.1 常用策略

| 策略 | 适用 | sklearn 类 |
|------|------|-----------|
| 删除含缺失的行 | 缺失比例很小 | `dropna()` |
| 均值/中位数填充 | 数值特征 | `SimpleImputer(strategy='mean')` |
| 众数填充 | 类别特征 | `SimpleImputer(strategy='most_frequent')` |
| 常数填充 | 业务含义明确 | `SimpleImputer(strategy='constant')` |
| KNN 填充 | 特征间有关联 | `KNNImputer` |

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
import seaborn as sns

df = sns.load_dataset('titanic')

# 查看缺失情况
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("缺失值:\n", missing)

# 数值特征：中位数填充
num_imputer = SimpleImputer(strategy='median')
df['age'] = num_imputer.fit_transform(df[['age']])

# 类别特征：众数填充
cat_imputer = SimpleImputer(strategy='most_frequent')
df['embarked'] = cat_imputer.fit_transform(df[['embarked']]).ravel()

print(f"\n填充后缺失: {df[['age', 'embarked']].isnull().sum().sum()}")
```

---

## 二、标准化与归一化

### 2.1 对比

| 方法 | 公式 | 结果范围 | 适用 |
|------|------|---------|------|
| **StandardScaler** | `(x - mean) / std` | 均值0, 标准差1 | SVM、逻辑回归、KNN |
| **MinMaxScaler** | `(x - min) / (max - min)` | [0, 1] | 神经网络 |
| **RobustScaler** | `(x - median) / IQR` | 无固定范围 | 有异常值时 |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt

data = df[['age', 'fare']].values

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
scalers = [
    ('原始数据', None),
    ('StandardScaler', StandardScaler()),
    ('MinMaxScaler', MinMaxScaler()),
    ('RobustScaler', RobustScaler()),
]

for ax, (name, scaler) in zip(axes, scalers):
    if scaler:
        scaled = scaler.fit_transform(data)
    else:
        scaled = data
    ax.scatter(scaled[:, 0], scaled[:, 1], s=10, alpha=0.5, color='steelblue')
    ax.set_xlabel('age')
    ax.set_ylabel('fare')
    ax.set_title(name)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 三、类别特征编码

### 3.1 Label Encoding

把类别转成数字（适合有序特征和树模型）。

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['sex_encoded'] = le.fit_transform(df['sex'])
print(f"编码映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")
```

### 3.2 One-Hot Encoding

把类别展开成多列 0/1（适合无序特征和线性模型）。

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' 避免多重共线性
embarked_encoded = ohe.fit_transform(df[['embarked']])
print(f"One-Hot 列名: {ohe.get_feature_names_out(['embarked'])}")
print(f"前 3 行:\n{embarked_encoded[:3]}")
```

### 3.3 编码选择指南

| 编码方式 | 适用特征 | 适用模型 |
|---------|---------|---------|
| Label Encoding | 有序类别 | 树模型 |
| One-Hot Encoding | 无序类别（类别少） | 线性模型、神经网络 |
| `pd.get_dummies` | 快速 One-Hot | 探索阶段 |

---

## 四、异常值处理

### 4.1 IQR 方法

```python
def remove_outliers_iqr(df, column, factor=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    mask = (df[column] >= lower) & (df[column] <= upper)
    return df[mask], (~mask).sum()

df_clean, n_removed = remove_outliers_iqr(df, 'fare')
print(f"fare 异常值: {n_removed} 个, 剩余: {len(df_clean)} 行")
```

### 4.2 处理策略

| 策略 | 说明 |
|------|------|
| **删除** | 明显错误的数据 |
| **截断（Clipping）** | 超出范围的值设为上下界 |
| **分箱** | 转为类别特征 |
| **保留** | 树模型对异常值不敏感 |

---

## 五、小结

| 预处理 | 方法 | 要点 |
|--------|------|------|
| 缺失值 | 填充 / 删除 | 训练集 fit，测试集 transform |
| 标准化 | StandardScaler / MinMaxScaler | 线性模型必须做 |
| 编码 | LabelEncoder / OneHotEncoder | 根据模型和特征类型选择 |
| 异常值 | IQR / 截断 / 保留 | 看业务和模型类型 |

---

## 动手练习

### 练习 1：Titanic 完整预处理

对 Titanic 数据集做完整预处理：填充缺失值、编码类别特征、标准化数值特征，最后训练逻辑回归看效果。

### 练习 2：缩放方法对比

用 Wine 数据集，对比不做缩放、StandardScaler、MinMaxScaler 对 SVM 和逻辑回归准确率的影响。

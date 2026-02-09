---
title: "2.5 数组变形与操作"
sidebar_position: 6
description: "掌握数组的 reshape、拼接、分割和转置操作"
---

# 数组变形与操作

## 学习目标

- 掌握 reshape、flatten、ravel 等变形操作
- 学会数组的拼接（concatenate、stack、hstack、vstack）
- 学会数组的分割（split、hsplit、vsplit）
- 理解转置和轴交换

---

## reshape：改变形状

`reshape` 是最常用的变形操作——在**不改变数据**的前提下改变数组的形状。

### 基本用法

```python
import numpy as np

arr = np.arange(12)    # [ 0  1  2  3  4  5  6  7  8  9 10 11]
print(arr.shape)       # (12,)

# 变成 3 行 4 列
m1 = arr.reshape(3, 4)
print(m1)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# 变成 4 行 3 列
m2 = arr.reshape(4, 3)
print(m2)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

# 变成 2×2×3 的三维数组
m3 = arr.reshape(2, 2, 3)
print(m3)
# [[[ 0  1  2]
#   [ 3  4  5]]
#  [[ 6  7  8]
#   [ 9 10 11]]]
```

:::caution 元素总数必须一致
reshape 前后的元素总数必须相同，否则报错：

```python
arr = np.arange(12)    # 12 个元素
arr.reshape(3, 5)      # ❌ 报错！3 × 5 = 15 ≠ 12
arr.reshape(3, 4)      # ✅ 3 × 4 = 12
```
:::

### 用 -1 自动计算

`-1` 表示"让 NumPy 自动计算这个维度"：

```python
arr = np.arange(12)

# 我想要 3 行，列数你帮我算
m1 = arr.reshape(3, -1)    # 自动算出 4 列
print(m1.shape)             # (3, 4)

# 我想要 4 列，行数你帮我算
m2 = arr.reshape(-1, 4)    # 自动算出 3 行
print(m2.shape)             # (3, 4)

# 变成一列（列向量）
col = arr.reshape(-1, 1)
print(col.shape)            # (12, 1)
```

:::tip -1 只能用一次
reshape 中最多只有一个维度可以是 -1。因为只有一个未知数才能算出来。

```python
arr.reshape(-1, -1)  # ❌ 报错！不能有两个 -1
```
:::

---

## flatten 和 ravel：展平数组

把多维数组变回一维：

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# flatten：返回拷贝（修改不影响原数组）
flat = matrix.flatten()
print(flat)          # [1 2 3 4 5 6]
flat[0] = 99
print(matrix[0, 0])  # 1  ← 原数组不变

# ravel：返回视图（修改会影响原数组）
rav = matrix.ravel()
print(rav)           # [1 2 3 4 5 6]
rav[0] = 99
print(matrix[0, 0])  # 99  ← 原数组也变了！
```

| 方法 | 返回类型 | 修改是否影响原数组 | 速度 |
|------|---------|------------------|------|
| `flatten()` | 拷贝 | 不影响 | 较慢（要复制数据） |
| `ravel()` | 视图 | 影响 | 较快（不复制） |
| `reshape(-1)` | 视图 | 影响 | 较快 |

---

## 数组拼接

### concatenate：通用拼接

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 一维拼接
c = np.concatenate([a, b])
print(c)  # [1 2 3 4 5 6]
```

二维拼接需要指定方向（axis）：

```python
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])

# axis=0：上下拼接（行数增加）
v = np.concatenate([m1, m2], axis=0)
print(v)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# axis=1：左右拼接（列数增加）
h = np.concatenate([m1, m2], axis=1)
print(h)
# [[1 2 5 6]
#  [3 4 7 8]]
```

### vstack 和 hstack：快捷拼接

```python
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])

# vstack = vertical stack = 上下拼接 = concatenate(axis=0)
print(np.vstack([m1, m2]))
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# hstack = horizontal stack = 左右拼接 = concatenate(axis=1)
print(np.hstack([m1, m2]))
# [[1 2 5 6]
#  [3 4 7 8]]
```

### stack：创建新维度

`stack` 和 `concatenate` 的区别是——`stack` 会**增加一个维度**：

```python
a = np.array([1, 2, 3])   # shape: (3,)
b = np.array([4, 5, 6])   # shape: (3,)

# stack 沿新维度堆叠
s0 = np.stack([a, b], axis=0)   # 相当于"横着放"
print(s0)
# [[1 2 3]
#  [4 5 6]]
print(s0.shape)  # (2, 3)

s1 = np.stack([a, b], axis=1)   # 相当于"竖着放"
print(s1)
# [[1 4]
#  [2 5]
#  [3 6]]
print(s1.shape)  # (3, 2)
```

### 拼接方法总结

| 函数 | 作用 | 维度变化 |
|------|------|---------|
| `np.concatenate()` | 沿已有轴拼接 | 维度不变，某个轴变长 |
| `np.vstack()` | 上下拼接 | 行数增加 |
| `np.hstack()` | 左右拼接 | 列数增加 |
| `np.stack()` | 沿新轴堆叠 | 增加一个维度 |

---

## 数组分割

### split：均匀分割

```python
arr = np.arange(12)   # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# 均匀分成 3 份
parts = np.split(arr, 3)
print(parts[0])   # [0 1 2 3]
print(parts[1])   # [4 5 6 7]
print(parts[2])   # [8 9 10 11]

# 按指定位置分割
parts2 = np.split(arr, [3, 7])  # 在索引 3 和 7 处切
print(parts2[0])  # [0 1 2]
print(parts2[1])  # [3 4 5 6]
print(parts2[2])  # [7 8 9 10 11]
```

### 二维分割

```python
matrix = np.arange(16).reshape(4, 4)
print(matrix)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]

# vsplit：上下分割
top, bottom = np.vsplit(matrix, 2)
print(top)
# [[0 1 2 3]
#  [4 5 6 7]]

# hsplit：左右分割
left, right = np.hsplit(matrix, 2)
print(left)
# [[ 0  1]
#  [ 4  5]
#  [ 8  9]
#  [12 13]]
```

---

## 转置与轴交换

### 二维转置

转置就是**行变列，列变行**：

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(matrix.shape)  # (2, 3)

# 转置
t = matrix.T
print(t)
# [[1 4]
#  [2 5]
#  [3 6]]
print(t.shape)  # (3, 2)

# 也可以用 transpose
t2 = matrix.transpose()
print(np.array_equal(t, t2))  # True
```

### 添加维度：np.newaxis 和 expand_dims

有时候我们需要给数组增加一个维度（比如把行向量变成列向量）：

```python
arr = np.array([1, 2, 3])      # shape: (3,)

# 方法 1：np.newaxis
row = arr[np.newaxis, :]        # shape: (1, 3) 行向量
col = arr[:, np.newaxis]        # shape: (3, 1) 列向量
print(row)  # [[1 2 3]]
print(col)
# [[1]
#  [2]
#  [3]]

# 方法 2：np.expand_dims
row2 = np.expand_dims(arr, axis=0)   # 在 axis=0 处添加维度 → (1, 3)
col2 = np.expand_dims(arr, axis=1)   # 在 axis=1 处添加维度 → (3, 1)

# 方法 3：reshape
row3 = arr.reshape(1, -1)   # (1, 3)
col3 = arr.reshape(-1, 1)   # (3, 1)
```

### 压缩维度：squeeze

去掉大小为 1 的维度：

```python
arr = np.array([[[1, 2, 3]]])
print(arr.shape)          # (1, 1, 3)

squeezed = arr.squeeze()
print(squeezed.shape)     # (3,)
print(squeezed)           # [1 2 3]
```

---

## 实战：数据重组

```python
import numpy as np

# 场景：你有 12 个月的销售数据（一维）
monthly_sales = np.array([
    120, 135, 150, 180, 200, 210,
    195, 188, 220, 250, 280, 310
])

# 重组成 4 个季度 × 3 个月
quarterly = monthly_sales.reshape(4, 3)
print("季度数据:")
print(quarterly)
# [[120 135 150]    Q1
#  [180 200 210]    Q2
#  [195 188 220]    Q3
#  [250 280 310]]   Q4

# 每季度总销售额
q_totals = quarterly.sum(axis=1)
quarters = ["Q1", "Q2", "Q3", "Q4"]
for q, total in zip(quarters, q_totals):
    print(f"  {q}: {total}")

# 上半年 vs 下半年
first_half, second_half = np.vsplit(quarterly, 2)
print(f"\n上半年总额: {first_half.sum()}")
print(f"下半年总额: {second_half.sum()}")
```

---

## 小结

| 操作 | 函数 | 说明 |
|------|------|------|
| 改变形状 | `reshape()` | 元素总数不变，改变维度排列 |
| 展平 | `flatten()` / `ravel()` | 多维变一维 |
| 拼接 | `concatenate()` / `vstack()` / `hstack()` | 多个数组合并 |
| 堆叠 | `stack()` | 合并并增加一个维度 |
| 分割 | `split()` / `vsplit()` / `hsplit()` | 一个数组拆成多个 |
| 转置 | `.T` / `transpose()` | 行列互换 |
| 增加维度 | `np.newaxis` / `expand_dims()` | 添加 size=1 的维度 |
| 压缩维度 | `squeeze()` | 去掉 size=1 的维度 |

---

## 动手练习

### 练习 1：reshape 练习

```python
arr = np.arange(24)

# 1. 变成 4×6 的矩阵
# 2. 变成 2×3×4 的三维数组
# 3. 变成 6 行（列数自动计算）
# 4. 把 (2,3,4) 数组展平回一维
```

### 练习 2：拼接与分割

```python
# 有 3 个班的成绩数据
class_a = np.array([[85, 90], [78, 82], [92, 88]])   # 3 人 × 2 科
class_b = np.array([[76, 80], [95, 91], [83, 87]])   # 3 人 × 2 科
class_c = np.array([[88, 92], [71, 75], [90, 85]])   # 3 人 × 2 科

# 1. 把 3 个班的成绩合并成一个 9×2 的矩阵
# 2. 如果有第 3 科成绩需要补充，怎么拼接？
extra_scores = np.array([[70], [65], [80], [75], [90], [85], [78], [72], [88]])
# 3. 把合并后的 9×3 矩阵按每 3 人分割回 3 组
```

### 练习 3：数据重组

```python
# 一年 365 天的温度数据（假数据）
np.random.seed(42)
daily_temps = np.random.uniform(low=-5, high=38, size=360)  # 取 360 天方便分割

# 1. 重组成 12 个月 × 30 天
# 2. 计算每月平均温度
# 3. 找出最热和最冷的月份
# 4. 计算上半年和下半年的平均温度差
```

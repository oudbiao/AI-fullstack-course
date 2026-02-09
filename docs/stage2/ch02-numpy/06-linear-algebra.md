---
title: "2.6 线性代数基础操作"
sidebar_position: 7
description: "用 NumPy 进行矩阵乘法、求逆、求特征值等线性代数运算"
---

# 线性代数基础操作

## 学习目标

- 掌握矩阵乘法的三种写法（dot、matmul、@）
- 了解逆矩阵、行列式、特征值的含义和计算
- 学会用 `numpy.linalg` 模块进行线性代数运算
- 理解线性代数在 AI 中的重要性

---

## 为什么要学线性代数？

你可能觉得"线性代数"听起来很数学、很抽象。但在 AI 领域，它是**最核心**的数学基础：

| AI 场景 | 线性代数的角色 |
|---------|--------------|
| 神经网络 | 每一层的运算就是矩阵乘法 |
| 推荐系统 | 用户-商品矩阵分解 |
| 图像处理 | 一张图片就是一个矩阵 |
| 词向量 | 每个词是一个向量，相似度 = 点积 |
| 降维 | PCA 就是求特征值和特征向量 |

现在先用 NumPy 操作一下这些概念，建立直觉。第三阶段（AI 数学基础）会更深入地讲解原理。

---

## 矩阵乘法

### 元素乘法 vs 矩阵乘法

这是新手最容易搞混的地方：

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 元素乘法（逐位相乘）
print(A * B)
# [[ 5 12]
#  [21 32]]
# 计算过程：1×5=5, 2×6=12, 3×7=21, 4×8=32

# 矩阵乘法
print(A @ B)
# [[19 22]
#  [43 50]]
# 计算过程：
# [1×5+2×7, 1×6+2×8]   = [19, 22]
# [3×5+4×7, 3×6+4×8]   = [43, 50]
```

### 矩阵乘法的三种写法

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 方法 1：@ 运算符（推荐，最简洁）
C1 = A @ B

# 方法 2：np.matmul
C2 = np.matmul(A, B)

# 方法 3：np.dot
C3 = np.dot(A, B)

# 三种方法结果完全一样
print(np.array_equal(C1, C2))  # True
print(np.array_equal(C2, C3))  # True
```

:::tip 推荐使用 @
在 Python 3.5+ 中，`@` 运算符是最推荐的矩阵乘法写法，简洁直观。
:::

### 矩阵乘法的规则

两个矩阵能相乘的条件：**前面的列数 = 后面的行数**。

```python
# (2, 3) @ (3, 4) → (2, 4)  ✅ 3 == 3
A = np.ones((2, 3))
B = np.ones((3, 4))
C = A @ B
print(C.shape)   # (2, 4)

# (2, 3) @ (2, 4) → ❌ 报错！3 ≠ 2
# A = np.ones((2, 3))
# B = np.ones((2, 4))
# C = A @ B  # ValueError!
```

记忆口诀：`(m, n) @ (n, p) → (m, p)`

### 向量点积

一维数组的 `@` 或 `np.dot` 计算的是**点积**（内积）：

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 点积 = 1×4 + 2×5 + 3×6 = 32
print(a @ b)        # 32
print(np.dot(a, b)) # 32
```

点积在 AI 中非常重要——后面学到**余弦相似度**和**注意力机制**时都会用到。

---

## numpy.linalg 模块

NumPy 的 `linalg` 子模块提供了完整的线性代数功能：

### 逆矩阵

矩阵的逆满足 `A × A⁻¹ = 单位矩阵`：

```python
A = np.array([[1, 2], [3, 4]])

# 求逆矩阵
A_inv = np.linalg.inv(A)
print(A_inv)
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# 验证：A × A_inv ≈ 单位矩阵
print(A @ A_inv)
# [[1.0000000e+00 0.0000000e+00]
#  [8.8817842e-16 1.0000000e+00]]
# 对角线是 1，其余接近 0（浮点精度误差）
```

:::caution 不是所有矩阵都有逆
只有**方阵**（行数=列数）且**行列式不为 0** 的矩阵才有逆。

```python
# 奇异矩阵（行列式为 0）没有逆
singular = np.array([[1, 2], [2, 4]])  # 第二行是第一行的 2 倍
# np.linalg.inv(singular)  # LinAlgError: Singular matrix
```
:::

### 行列式

行列式是一个标量值，表示矩阵的"缩放因子"：

```python
A = np.array([[1, 2], [3, 4]])
det = np.linalg.det(A)
print(f"行列式: {det:.1f}")   # -2.0

# 2×2 矩阵的行列式 = ad - bc
# [[a, b], [c, d]] → 1×4 - 2×3 = -2
```

### 特征值和特征向量

特征值和特征向量是矩阵的"DNA"——揭示矩阵的内在性质：

```python
A = np.array([[4, 2], [1, 3]])

# 求特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"特征值: {eigenvalues}")      # [5. 2.]
print(f"特征向量:\n{eigenvectors}")
# [[ 0.894 -0.707]
#  [ 0.447  0.707]]
```

:::info 特征值的直觉
如果把矩阵想象成一种"变换"（比如旋转、拉伸），那么：
- **特征向量** = 变换后**方向不变**的向量
- **特征值** = 该方向上被**拉伸的倍数**

这个概念在后面学 PCA 降维时会非常有用——PCA 本质上就是找到数据"变化最大"的方向（最大特征值对应的特征向量）。
:::

### 解线性方程组

```
解方程：
2x + y = 5
x + 3y = 7
```

用矩阵形式：`Ax = b`

```python
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 7])

# 解方程
x = np.linalg.solve(A, b)
print(f"x = {x[0]:.2f}, y = {x[1]:.2f}")  # x = 1.60, y = 1.80

# 验证
print(A @ x)   # [5. 7.]  ← 等于 b，说明解正确
```

---

## 其他实用操作

### 范数（向量的长度）

```python
v = np.array([3, 4])

# L2 范数（欧几里得距离）
l2 = np.linalg.norm(v)
print(f"L2 范数: {l2}")   # 5.0  (3² + 4² = 25, √25 = 5)

# L1 范数（绝对值之和）
l1 = np.linalg.norm(v, ord=1)
print(f"L1 范数: {l1}")   # 7.0  (|3| + |4| = 7)

# 矩阵的范数
M = np.array([[1, 2], [3, 4]])
print(f"矩阵 Frobenius 范数: {np.linalg.norm(M):.2f}")  # 5.48
```

### 矩阵的秩

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
rank = np.linalg.matrix_rank(A)
print(f"矩阵的秩: {rank}")  # 2（不是满秩，因为第三行 = 第一行×(-1) + 第二行×2）
```

### 常用函数速查

| 函数 | 作用 | 示例 |
|------|------|------|
| `A @ B` | 矩阵乘法 | `np.array([[1,2],[3,4]]) @ np.eye(2)` |
| `np.linalg.inv(A)` | 逆矩阵 | |
| `np.linalg.det(A)` | 行列式 | |
| `np.linalg.eig(A)` | 特征值和特征向量 | |
| `np.linalg.solve(A, b)` | 解方程 Ax=b | |
| `np.linalg.norm(v)` | 范数 | |
| `np.linalg.matrix_rank(A)` | 矩阵的秩 | |
| `A.T` | 转置 | |
| `np.trace(A)` | 迹（对角线之和） | |

---

## 实战：计算余弦相似度

余弦相似度是 AI 中衡量两个向量"相似程度"的常用方法。后面在词向量、推荐系统、RAG 中都会反复使用。

公式：`cos(θ) = (a · b) / (||a|| × ||b||)`

```python
import numpy as np

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    dot_product = a @ b                         # 点积
    norm_a = np.linalg.norm(a)                  # a 的长度
    norm_b = np.linalg.norm(b)                  # b 的长度
    return dot_product / (norm_a * norm_b)

# 示例：比较用户兴趣
# 维度代表：[科技, 体育, 音乐, 电影, 美食]
user_a = np.array([5, 1, 3, 4, 2])   # 喜欢科技和电影
user_b = np.array([4, 2, 3, 5, 1])   # 也喜欢科技和电影
user_c = np.array([1, 5, 2, 1, 4])   # 喜欢体育和美食

print(f"A 和 B 的相似度: {cosine_similarity(user_a, user_b):.4f}")  # 0.9631 很相似
print(f"A 和 C 的相似度: {cosine_similarity(user_a, user_c):.4f}")  # 0.5528 不太像
print(f"B 和 C 的相似度: {cosine_similarity(user_b, user_c):.4f}")  # 0.5025 不太像
```

---

## 小结

| 概念 | 说明 | NumPy 函数 |
|------|------|-----------|
| 矩阵乘法 | `(m,n) @ (n,p) → (m,p)` | `A @ B` 或 `np.matmul` |
| 逆矩阵 | `A × A⁻¹ = I` | `np.linalg.inv()` |
| 行列式 | 矩阵的缩放因子 | `np.linalg.det()` |
| 特征值/向量 | 矩阵的"DNA" | `np.linalg.eig()` |
| 解方程 | 解 Ax = b | `np.linalg.solve()` |
| 范数 | 向量的长度 | `np.linalg.norm()` |

:::tip 学到什么程度就够？
在本阶段，你只需要：
1. **会用** NumPy 的线性代数函数
2. **知道**矩阵乘法、逆矩阵、特征值大概是什么意思
3. **能算**余弦相似度

深入的数学理解会在第三阶段（AI 数学基础）中系统学习。现在先建立代码直觉就好。
:::

---

## 动手练习

### 练习 1：矩阵乘法

```python
# 某商店 3 种商品的单价
prices = np.array([10, 25, 8])   # [苹果, 牛排, 面包]

# 3 位顾客的购买数量
quantities = np.array([
    [3, 1, 2],    # 顾客 1: 3苹果 + 1牛排 + 2面包
    [0, 2, 5],    # 顾客 2
    [5, 0, 3]     # 顾客 3
])

# 用矩阵乘法计算每位顾客的消费总额
# totals = ?
```

### 练习 2：解方程

```python
# 解方程组：
# 3x + 2y - z = 1
# x - y + 2z = 5
# 2x + 3y - z = 0
#
# 提示：写成 Ax = b 的形式
```

### 练习 3：余弦相似度应用

```python
# 假设有 5 部电影的特征向量
# 维度代表：[动作, 喜剧, 爱情, 科幻, 恐怖]
movies = {
    "复仇者联盟": np.array([5, 2, 1, 4, 0]),
    "泰囧":       np.array([1, 5, 2, 0, 0]),
    "泰坦尼克号": np.array([1, 0, 5, 0, 1]),
    "星际穿越":   np.array([3, 0, 2, 5, 0]),
    "釜山行":     np.array([4, 0, 1, 1, 5]),
}

# 用余弦相似度找出和"复仇者联盟"最相似的电影
# 提示：计算"复仇者联盟"和其他每部电影的余弦相似度
```

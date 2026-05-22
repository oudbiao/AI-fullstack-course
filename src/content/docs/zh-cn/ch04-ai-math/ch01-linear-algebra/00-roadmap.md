---
title: "4.1.1 线性代数路线图：数据是向量，批量是矩阵"
description: "面向 AI 的紧凑版线性代数路线图：向量、矩阵、点积、特征值和变换。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "线性代数指南, AI 数学指南, 向量, 矩阵, 特征值, PCA"
---
线性代数是 AI 表示数据和变换数据的语言。不要从背证明开始，先看每个对象在代码里做什么。

## 先看地图

![线性代数学习地图](/img/course/ch04-linear-algebra-roadmap-vertical.webp)

本小章流向是：

![线性代数章节流程](/img/course/ch04-linear-algebra-chapter-flow.webp)

| 概念 | 在 AI 里的第一层意思 |
|---|---|
| 向量 | 一个对象写成一串数字 |
| 矩阵 | 多个向量叠在一起，或表示一种变换 |
| 点积 | 对应位置相乘后求和 |
| 矩阵乘法 | 一次做很多个点积 |
| 特征值/特征向量 | 重要方向，是理解 PCA 的入口 |

## 跑最小闭环

创建 `linear_algebra_first_loop.py`，安装 `numpy` 后运行。

```python
import numpy as np

student = np.array([90, 85, 92])
students = np.array(
    [
        [90, 85, 92],
        [70, 88, 75],
        [95, 91, 89],
    ]
)
weights = np.array([0.4, 0.2, 0.4])

single_score = student @ weights
all_scores = students @ weights

print("student_vector:", student)
print("matrix_shape:", students.shape)
print("single_score:", round(single_score, 2))
print("all_scores:", all_scores.round(2))
```

预期输出：

```text
student_vector: [90 85 92]
matrix_shape: (3, 3)
single_score: 89.8
all_scores: [89.8 75.6 91.8]
```

如果误用 `*` 而不是 `@`，得到的是逐元素相乘，不是加权得分。这是新手最值得先分清的地方。

## 按这个顺序学

| 顺序 | 阅读 | 先抓住什么 |
|---|---|---|
| 1 | [4.1.2 向量](./01-vectors.md) | 对象 -> 向量、长度、点积、余弦相似度 |
| 2 | [4.1.3 矩阵](./02-matrices.md) | 批量数据、矩阵乘法、`X @ W + b` |
| 3 | [4.1.4 特征值与特征向量](./03-eigenvalues.md) | 特殊方向、PCA 直觉 |
| 4 | [4.1.5 向量空间](./04-vector-spaces.md) | 基、维度、线性变换 |

## 通过标准

能解释为什么一个样本是向量、一批样本是矩阵、`@` 在做什么，以及这些概念为什么会出现在 RAG 相似度、PCA 和神经网络层里，就算通过。


<details>
<summary>检查思路与讲解</summary>

- 线性代数路线通过的标志是：你能把 `X @ W` 同时读成 shape 运算和一批点积。
- 证据至少保留一个向量相似度例子、一个矩阵变换例子、一个 PCA 或特征向量图、一个 SVD 或 rank 检查。
- 重点不是公式写得漂亮，而是说清方向、长度、维度或冗余信息发生了什么变化。

</details>


## 留下的证据

学完这一页，至少保留这张证据卡：

```text
数学对象：向量、矩阵、特征值、基或向量空间概念
数值示例：用于计算它的简单数字或 NumPy 片段
可视化或输出：形状、变换后的点、相似度分数、特征方向或投影
AI 关联：这里出现在 embeddings、批次、PCA、神经层或注意力中
期望产出：计算过程，以及一句把它和 AI 操作联系起来的话
```

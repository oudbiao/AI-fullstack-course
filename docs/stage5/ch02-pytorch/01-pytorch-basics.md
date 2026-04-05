---
title: "2.3 PyTorch 基础"
sidebar_position: 1
description: "从张量、形状、索引、广播到和 NumPy 的关系，打牢 PyTorch 的第一层基础。"
keywords: [PyTorch, tensor, 张量, shape, broadcasting, numpy]
---

# PyTorch 基础

## 学习目标

- 理解什么是 `Tensor`
- 掌握张量的创建、形状、数据类型和常用运算
- 理解 PyTorch 和 NumPy 的关系
- 能独立读懂最基本的张量操作代码

---

## 先建立一张地图

这一节不要把它看成“PyTorch 语法表”，更适合的理解顺序是：

```mermaid
flowchart LR
    A["真实数据"] --> B["变成 Tensor"]
    B --> C["看 shape 和 dtype"]
    C --> D["做索引、变形、运算"]
    D --> E["后面送进模型"]
```

也就是说，这一节真正要打稳的是：

- 你能不能把数据装进 PyTorch
- 你能不能看懂张量形状
- 你能不能安全地做最基础的运算

## 这节和第四阶段、NumPy 是怎么接上的

如果你是从第四阶段过来，可以先把这一节理解成：

- 第四阶段里 `X`、`y`、矩阵乘法这些东西，到这里都还在
- 只是现在它们要进入一个更适合深度学习训练的容器：`Tensor`

如果你熟悉 NumPy，也可以先这样记：

- `Tensor` 很像 `ndarray`
- 但它还能上 GPU，还能参与自动求导

所以这一节不是在学“全新数学”，而是在学：

- 同样的数据对象，换一种更适合训练神经网络的表达方式

## 一、张量到底是什么？

最实用的理解方式是：

> **张量 = 能在 CPU / GPU 上计算的多维数组**

如果你学过 NumPy，可以先把它想成“升级版 `ndarray`”：

- 能做数值运算
- 能放到 GPU 上
- 能参与自动求导

类比一下：

| 概念 | 类比 |
|---|---|
| 标量（0 维） | 一个数字 |
| 向量（1 维） | 一排数字 |
| 矩阵（2 维） | 一张表 |
| 张量（更高维） | 一叠表 / 一批图片 / 一段视频 |

在深度学习里，几乎所有数据最后都会变成张量：

- 一张灰度图：`[高度, 宽度]`
- 一张彩色图：`[通道, 高度, 宽度]`
- 一批图片：`[批大小, 通道, 高度, 宽度]`
- 一批句子的词向量：`[批大小, 序列长度, 向量维度]`

### 1.1 第一次看张量时，最该先问什么？

先不要急着问 API，先问这三个问题：

1. 它装的是什么数据？
2. 每一维分别代表什么？
3. 后面这份数据会被送到哪一层？

这会让你从一开始就把“形状”和“语义”绑在一起。

---

## 二、创建张量

:::info 运行环境
下面的代码可以直接运行。若本地未安装：

```bash
pip install torch
```
:::

```python
import torch

# 从 Python 列表创建
scores = torch.tensor([88, 92, 76, 95])
print(scores)

# 指定数据类型
prices = torch.tensor([12.5, 19.9, 8.8], dtype=torch.float32)
print(prices.dtype)

# 常见的初始化方式
zeros = torch.zeros((2, 3))
ones = torch.ones((2, 3))
randn = torch.randn((2, 3))
arange = torch.arange(0, 10, 2)

print("zeros:\n", zeros)
print("ones:\n", ones)
print("randn:\n", randn)
print("arange:", arange)
```

---

## 三、形状、维度和数据类型

初学深度学习，最容易卡住的不是公式，而是**形状（shape）**。

你可以把 `shape` 理解成“这个数据盒子有几层、每层装多少个元素”。

```python
import torch

X = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

print("张量:\n", X)
print("shape:", X.shape)       # torch.Size([2, 3])
print("ndim:", X.ndim)         # 2 维
print("dtype:", X.dtype)       # float32
print("元素总数:", X.numel())   # 6
```

### 一个非常重要的习惯

写模型前，先问自己：

1. 这个张量每一维是什么意思？
2. 当前形状对不对？
3. 下一层会期待什么形状？

很多训练报错，本质上都是 shape 不匹配。

### 3.1 一个更稳的“看张量四步法”

以后你拿到任何张量，都建议先做这四步检查：

1. 看 `shape`
2. 看 `dtype`
3. 想清每一维的语义
4. 想清下一步要做什么运算

比如：

```python
print(X.shape, X.dtype)
print("meaning: [batch, features]")
```

这个习惯会帮你少掉非常多莫名其妙的报错。

### 一个新人最该养成的记录方式

建议你第一次接触 PyTorch 时，看到张量就顺手记一句：

```python
print("shape:", X.shape, "| meaning: [batch, features]")
```

把“形状”和“语义”一起写出来，会比单看 `torch.Size(...)` 清楚很多。

---

## 四、索引、切片、变形

```python
import torch

X = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("第 0 行:", X[0])
print("第 1 行第 2 列:", X[1, 2])
print("前两行:\n", X[:2])
print("第 2 列:", X[:, 1])

# 变形
flat = X.reshape(9)
grid = flat.reshape(3, 3)

print("拉平:", flat)
print("重新变回 3x3:\n", grid)
```

### `reshape` 的直觉

就像你把一盒积木重新摆放：

- 元素个数不能变
- 只是换了一种组织方式

### 4.1 `reshape` 时新人最容易踩的坑

最常见的误区是：

- 以为 `reshape` 会改变数据内容

其实它通常只是在改变“怎么看这批元素”。  
所以更稳的习惯是每次 `reshape` 后都问一句：

- 现在每一维是什么意思？

---

## 五、张量运算

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("加法:", a + b)
print("减法:", a - b)
print("逐元素乘法:", a * b)
print("平方:", a ** 2)
print("求和:", a.sum())
print("均值:", a.mean())
```

### 矩阵乘法

深度学习里最常见的运算之一就是矩阵乘法：

```python
import torch

X = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])

W = torch.tensor([[2.0, 0.0],
                  [0.0, 2.0]])

Y = X @ W
print(Y)
```

这和你在第三阶段学的线性代数是同一件事。  
神经网络里的很多层，本质上都是“张量做线性变换，再过一个非线性函数”。

### 5.1 看到 `@` 时，脑子里最好立刻跳出的是什么？

最值得先跳出来的是：

- 这通常不是普通算术
- 这是“把一批输入按权重重新组合”

也就是说，一旦你在网络代码里看到：

```python
X @ W
```

就可以先理解成：

- 当前这一层正在把输入变成新的表示

---

## 六、广播机制

广播是 PyTorch 里一个特别省代码的机制。

它的直觉是：

> “如果两个张量形状不完全一样，但差得不多，PyTorch 会自动帮你扩展。”

```python
import torch

scores = torch.tensor([
    [80.0, 85.0, 90.0],
    [70.0, 75.0, 88.0]
])

bonus = torch.tensor([5.0, 5.0, 5.0])

print(scores + bonus)
```

这里 `bonus` 的 shape 是 `[3]`，`scores` 的 shape 是 `[2, 3]`。  
PyTorch 会自动把 `bonus` 当成每一行都加一次。

### 广播的常见用法

- 给一批样本统一加偏置
- 对图像做归一化
- 对 batch 中每个特征做缩放

### 6.1 广播为什么既方便又危险？

方便是因为它很省代码。  
危险是因为：

- 有时候代码能跑
- 但广播的方向不是你以为的那个方向

所以广播场景里，最稳的习惯是：

- 先写出两个张量的 shape
- 再明确“谁在被扩展”

---

## 七、和 NumPy 互转

NumPy 和 PyTorch 的关系非常近，所以互转很常见。

```python
import numpy as np
import torch

arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
tensor = torch.from_numpy(arr)

print("NumPy -> Tensor:\n", tensor)

back_to_numpy = tensor.numpy()
print("Tensor -> NumPy:\n", back_to_numpy)
```

### 什么时候用 NumPy，什么时候用 PyTorch？

- 数据分析、传统数值实验：NumPy 很方便
- 需要训练神经网络、自动求导、GPU：PyTorch 更合适

---

## 八、一个小例子：计算学生总成绩和平均分

这个例子没有“深度学习味”，但非常适合练张量思维。

```python
import torch

# 3 个学生，4 门课
scores = torch.tensor([
    [85.0, 92.0, 78.0, 90.0],
    [76.0, 88.0, 91.0, 84.0],
    [93.0, 87.0, 89.0, 95.0]
])

student_totals = scores.sum(dim=1)
student_means = scores.mean(dim=1)
subject_means = scores.mean(dim=0)

print("每位学生总分:", student_totals)
print("每位学生平均分:", student_means)
print("每门课程平均分:", subject_means)
```

这里你已经用到了张量最重要的思维之一：  
**“沿着哪一维做计算？”**

- `dim=1` 表示按行聚合
- `dim=0` 表示按列聚合

---

## 九、初学者最容易犯的错

### 1. 忽略 shape

很多人只看数字，不看张量形状。  
结果是代码“看起来像对的”，一运行就维度报错。

### 2. 把 `*` 当成矩阵乘法

在 PyTorch 里：

- `*` 是逐元素乘法
- `@` 才是矩阵乘法

### 3. 不清楚 dtype

有些模型需要 `float32`，标签有时又要 `long`。  
类型不对，损失函数可能直接报错。

### 4. 只看值，不看“值的意义”

最常见的初学者问题不是不会写代码，而是：

- 张量打印出来了
- 但不知道这一维代表 batch、特征、通道，还是类别

一旦语义没跟上，后面 `Linear`、`Conv`、`Loss` 都会开始混乱。

---

## 小结

这一节最重要的不是记住多少 API，而是建立三个基本反应：

1. 看到数据先看 `shape`
2. 看到运算先区分“逐元素”还是“矩阵乘法”
3. 知道深度学习里的输入、参数、输出，本质上都是张量

接下来我们就要让这些张量“自己知道该往哪里改”了，这就是自动求导。

## 这节最该带走什么

如果只带走一句话，我希望你记住：

> **PyTorch 基础这一节真正要练的，不是语法熟练度，而是你能不能把“张量的 shape、语义和运算方式”稳稳对上。**

因为后面大多数深度学习代码问题，最后都会落回这三件事：

- shape
- dtype
- 运算含义

---

## 练习

1. 新建一个形状为 `(2, 3, 4)` 的随机张量，并打印它的 `shape`、`ndim`、`numel()`。
2. 创建一个 `3x3` 张量，把它 reshape 成 `1x9` 和 `9x1`。
3. 自己构造两个可以相乘的矩阵，用 `@` 试一次矩阵乘法。

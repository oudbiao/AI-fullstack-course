---
title: "1.4 线性判别分析"
sidebar_position: 15
description: "从“类内更紧、类间更远”的目标出发，理解 LDA 为什么既能做分类，也常被当作监督式降维方法。"
keywords: [LDA, linear discriminant analysis, dimensionality reduction, classification, classic ML]
---

# 线性判别分析

:::tip 本节定位
LDA 很容易和别的缩写混掉，  
也很容易被误解成“又一个线性分类器”。

更准确的理解是：

> **LDA 关心的是怎样找到一个投影方向，让同类样本更聚、不同类样本更分开。**

所以它既能分类，也能作为一种监督式降维方法来看。
:::

## 学习目标

- 理解 LDA 的核心目标：类内紧凑、类间分离
- 理解它和普通线性分类器的差别
- 通过可运行示例看懂 LDA 的降维与分类效果
- 建立何时尝试 LDA 的基本判断

---

## 一、LDA 在解决什么问题？

### 1.1 不只是“分开类别”

LDA 的目标更具体：

- 同一个类别内部尽量聚在一起
- 不同类别之间尽量拉开

### 1.2 为什么这比普通线性切分更有意思？

因为它不只是找一条边界，  
还在找一个“更有判别力的表示空间”。

这意味着它除了能分类，还能做：

- 监督式降维

### 1.3 一个类比

如果说 PCA 更像：

- 找最能解释整体变化的方向

那 LDA 更像：

- 找最有利于区分类别的方向

---

## 二、LDA 为什么常被当作“带标签的降维”？

### 2.1 因为它用到了类别标签

PCA 不关心类别，只看整体方差。  
LDA 会明确利用标签去问：

- 哪个方向最利于分类？

### 2.2 所以它很适合什么场景？

适合：

- 你已经有监督标签
- 想做更有判别力的低维表示
- 或者想做一个轻量线性分类器

---

## 三、先跑一个最小可运行示例

这个例子会同时做两件事：

1. 用 LDA 分类
2. 把数据投影到更低维空间

```python
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [2.0, 1.5],
    [4.0, 5.0],
    [4.5, 4.8],
    [5.0, 4.5],
])
y = np.array([0, 0, 0, 1, 1, 1])

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X, y)

pred = lda.predict([[1.4, 1.9], [4.8, 4.6]])
projection = lda.transform(X)

print("predictions:", pred.tolist())
print("projection shape:", projection.shape)
print("projected values:", projection.ravel().round(3).tolist())
```

### 3.1 这段代码为什么比单纯 `predict` 更有价值？

因为它让你同时看到：

- 分类输出
- 投影后的低维表示

这正好体现了 LDA 的双重价值：

- 能分类
- 也能做监督式降维

### 3.2 为什么 `n_components=1`？

因为当前只有两类。  
在这种情况下，LDA 最多只能投到：

- 1 个判别方向

这也是它和类别数相关的一个特点。

---

## 四、LDA 和 SVM / Logistic Regression 有什么不同？

### 4.1 和 SVM 的差别

SVM 更强调：

- 间隔最大化

LDA 更强调：

- 类内方差小
- 类间均值差异大

### 4.2 和 Logistic Regression 的差别

Logistic Regression 更像在学：

- 条件概率边界

LDA 更像先假设数据分布，再找更有区分力的方向。

### 4.3 为什么这值得学？

因为它让你看到：

- 经典模型并不是只有一种“线性分类”思路

---

## 五、LDA 适合什么时候试？

### 5.1 数据量不大，但类别结构比较清楚

LDA 在这类场景里可能很有用。

### 5.2 你需要更可解释的低维表示

例如：

- 先投影再可视化
- 先投影再接简单分类器

### 5.3 不太适合的情况

如果类别边界特别复杂、明显非线性，  
LDA 往往就会比较吃力。

---

## 六、最常见误区

### 6.1 误区一：LDA 就只是另一个分类器

不完整。  
它的“判别式表示”价值同样重要。

### 6.2 误区二：有标签时就一定比 PCA 好

也不一定。  
看任务目标和数据分布。

### 6.3 误区三：LDA 和主题模型里的 LDA 是一回事

不是。  
这里的 LDA 是：

- Linear Discriminant Analysis

不是主题模型里的：

- Latent Dirichlet Allocation

---

## 小结

这节最重要的是建立一个判断：

> **LDA 的核心价值在于利用标签找到更有判别力的投影方向，因此它既能做轻量分类，也能做监督式降维。**

一旦把这层理解清楚，它就不再只是一个容易混淆的缩写。

---

## 练习

1. 把示例中的数据再加一个新类别，看看 `n_components` 会发生什么变化。
2. 想一想：为什么说 LDA 更像“带标签的降维”？
3. 如果类别边界非常弯曲、非线性明显，你还会优先试 LDA 吗？为什么？
4. 用自己的话解释：LDA 和 PCA 最大的区别是什么？

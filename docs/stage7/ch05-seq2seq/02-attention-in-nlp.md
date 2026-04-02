---
title: "5.2 NLP 中的注意力机制"
sidebar_position: 14
description: "从 encoder-decoder 的信息瓶颈讲起，理解注意力如何让生成端动态查看输入不同位置。"
keywords: [attention, seq2seq, encoder decoder, alignment, NLP]
---

# NLP 中的注意力机制

:::tip 本节定位
上一节我们提到 Seq2Seq 的一个典型问题：

- 输入被压成一个固定向量后，长句子信息容易丢失

注意力机制解决的正是这个瓶颈：

> **生成每个输出词时，不必只靠一个固定表示，而是可以重新查看输入序列里最相关的位置。**
:::

## 学习目标

- 理解注意力机制出现的背景
- 理解“对齐”和“加权聚合”的核心直觉
- 通过可运行示例理解注意力权重和上下文向量
- 建立注意力和后续 Transformer 之间的连接感

---

## 一、为什么 Seq2Seq 需要注意力？

### 1.1 固定长度编码容易丢信息

如果输入是：

- 很长的句子
- 复杂的段落

只把它压成一个固定向量，  
解码器后面会很吃力。

### 1.2 解码器在不同时间步关注点应该不同

例如翻译时：

- 生成第 1 个词时关注输入前半
- 生成后面的词时关注输入别的位置

所以“整个输出过程只看同一个向量”并不自然。

### 1.3 注意力的核心直觉

每次生成输出时，  
都根据当前解码状态去问：

- 输入序列里谁和我最相关？

然后把相关位置加权聚合成当前上下文。

---

## 二、先跑一个最小注意力示例

```python
import math

encoder_states = [
    [1.0, 0.0],
    [0.5, 0.5],
    [0.0, 1.0],
]

query = [0.7, 0.3]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(values):
    exps = [math.exp(v) for v in values]
    total = sum(exps)
    return [round(v / total, 4) for v in exps]


scores = [dot(state, query) for state in encoder_states]
weights = softmax(scores)

context = [0.0, 0.0]
for w, state in zip(weights, encoder_states):
    context = [context[i] + w * state[i] for i in range(len(context))]

print("scores :", [round(x, 4) for x in scores])
print("weights:", weights)
print("context:", [round(x, 4) for x in context])
```

### 2.1 这段代码最该看什么？

三步最关键：

1. `query` 和每个 `encoder_state` 打分
2. `softmax` 得到注意力权重
3. 用权重对 encoder states 做加权平均

### 2.2 为什么这已经体现了注意力本质？

因为它回答了两个核心问题：

- 该看谁
- 看多少

这就是注意力最重要的直觉。

---

## 三、注意力为什么会显著改善 Seq2Seq？

### 3.1 它缓解了信息瓶颈

输入不再只能通过一个固定向量传递给解码器。

### 3.2 它让输入输出对齐更自然

很多翻译任务本来就有“某个输出词大致对应输入哪些词”的结构。  
注意力让这种对齐更容易学到。

### 3.3 这也是从经典 Seq2Seq 走向 Transformer 的桥梁

后面 Transformer 把注意力推广得更彻底，  
但这节的直觉基础是一样的。

---

## 四、最容易踩的坑

### 4.1 误区一：注意力只是一个加权平均小技巧

不止。  
它改变了模型如何访问输入信息。

### 4.2 误区二：有注意力就再也不会丢信息

不是。  
长序列仍然会有难题，只是瓶颈被显著缓解。

### 4.3 误区三：注意力就是 Transformer

注意力是更大的概念，Transformer 是在其上发展出的完整架构。

---

## 小结

这节最重要的是建立一个桥接直觉：

> **注意力机制让解码器在生成每一步时都能重新查看输入序列里最相关的位置，从而缓解固定编码向量的信息瓶颈。**

只要这层理解清楚，你后面再学 Transformer 的自注意力就会轻松很多。

---

## 练习

1. 改一改 `query`，看看注意力权重会如何变化。
2. 用自己的话解释：为什么 Seq2Seq 会需要“动态看输入”而不是只看固定向量？
3. `weights` 为什么要经过 softmax？
4. 想一想：这节的注意力和后面 Transformer 的自注意力，核心相同点是什么？

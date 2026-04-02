---
title: "1.2 词嵌入与语义表示"
sidebar_position: 2
description: "从 one-hot 到 dense vector，再到句子表示与上下文化表示，理解模型为什么能把“语义相近”变成向量空间里的距离关系。"
keywords: [embedding, semantic representation, cosine similarity, sentence embedding, contextual embedding]
---

# 词嵌入与语义表示

:::tip 本节定位
Tokenizer 解决的是：

- 文本怎么切

Embedding 解决的是：

- 切出来的 token 怎么变成有语义的向量

很多人刚接触 embedding 时，会把它理解成：

- 给每个词分配一串数字

这还不够。  
真正关键的是：

> **这些数字不是随便填的，它们会逐渐形成一个“语义空间”，让相近词、相近句子在空间里彼此靠近。**
:::

## 学习目标

- 理解 one-hot 和 dense embedding 的根本差异
- 理解为什么相似语义可以体现在向量距离上
- 理解词向量、句向量和上下文化表示的层层递进
- 通过可运行示例看懂 embedding 如何支持相似度计算

---

## 一、为什么不能直接用 one-hot 表示词？

### 1.1 one-hot 很干净，但它不表达语义关系

假设词表里有四个词：

- `refund`
- `return`
- `password`
- `banana`

one-hot 表示会像这样：

- `refund` -> `[1, 0, 0, 0]`
- `return` -> `[0, 1, 0, 0]`
- `password` -> `[0, 0, 1, 0]`

问题在于：

- `refund` 和 `return` 语义很近
- `refund` 和 `banana` 语义很远

但在 one-hot 空间里，它们彼此一样“远”。

这意味着：

> **one-hot 能区分身份，但不会表达相似性。**

### 1.2 Dense embedding 的核心价值

embedding 想做的事情是：

- 让语义相近的词，向量也相近

例如：

- `refund` 和 `return`
- `reset` 和 `recover`

它们可以在向量空间里靠得更近。

这就是 embedding 真正重要的地方：

- 不只是编码
- 而是表示

### 1.3 一个类比：给词放进一张地图

你可以把 embedding 想成地图坐标。

- one-hot 更像身份证号，只能区分人
- embedding 更像地图位置，不仅能区分，还能看谁离谁近

一旦有了这张语义地图，  
模型就能更容易发现：

- 哪些词经常在类似上下文里出现
- 哪些句子表达的是相近意思

---

## 二、词向量为什么会有语义？

### 2.1 因为它们是在上下文里被学出来的

embedding 不是人工规定的。  
它通常是在训练过程中慢慢学出来的。

如果两个词经常出现在相似上下文里，  
模型就会倾向于把它们学成相近向量。

这就是经典的分布式假设：

> **词的意义，很大程度上由它出现的上下文决定。**

### 2.2 语义相近不代表完全同义

向量靠近只说明：

- 用法相近
- 上下文分布接近

并不等于：

- 完全可以互换

例如：

- `doctor` 和 `hospital`

可能也会靠近，因为它们经常一起出现。  
所以 embedding 里的“近”，更多是分布意义上的近。

### 2.3 从词到句，表示可以继续往上聚合

当你把多个 token 向量组合起来，  
就可以得到：

- 短语向量
- 句子向量
- 段落向量

这也是为什么 embedding 不只用来做词相似度，  
还会被广泛用在：

- 检索
- 聚类
- 分类
- RAG

---

## 三、先跑一个真正有语义对比意义的示例

下面这段代码会做三件事：

1. 给几个词指定一组小型 embedding
2. 计算词之间的余弦相似度
3. 用“平均 token 向量”的方法得到句向量，再比较句子相似度

```python
from math import sqrt

embeddings = {
    "refund": [0.90, 0.80, 0.10],
    "return": [0.88, 0.78, 0.12],
    "password": [0.10, 0.20, 0.95],
    "reset": [0.12, 0.18, 0.92],
    "order": [0.75, 0.70, 0.15],
    "banana": [0.05, 0.95, 0.10],
}


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def sentence_embedding(tokens, embedding_table):
    valid = [embedding_table[token] for token in tokens if token in embedding_table]
    dim = len(valid[0])
    return [sum(vec[i] for vec in valid) / len(valid) for i in range(dim)]


print("refund vs return  :", round(cosine(embeddings["refund"], embeddings["return"]), 3))
print("refund vs password:", round(cosine(embeddings["refund"], embeddings["password"]), 3))

query_a = ["reset", "password"]
query_b = ["password", "reset"]
query_c = ["refund", "order"]

vec_a = sentence_embedding(query_a, embeddings)
vec_b = sentence_embedding(query_b, embeddings)
vec_c = sentence_embedding(query_c, embeddings)

print("query_a vs query_b:", round(cosine(vec_a, vec_b), 3))
print("query_a vs query_c:", round(cosine(vec_a, vec_c), 3))
```

### 3.1 这段代码在说明什么？

它说明了两个层次的事情：

第一层：

- `refund` 和 `return` 这种相近词，在向量空间里也更近

第二层：

- 把 token 向量聚合后，句子也能在向量空间里比较相似度

这就是 embedding 为什么能支持语义检索和召回。

### 3.2 为什么 `query_a` 和 `query_b` 会非常接近？

因为它们只是词序不同，  
平均向量后得到的表示基本一致。

这同时也暴露了简单平均法的局限：

- 它几乎不关心顺序

所以早期静态句向量虽然有用，  
但表达能力有限。

### 3.3 为什么这段代码仍然是有价值的？

因为它抓住了 embedding 最本质的直觉：

> **“语义近”可以变成“向量近”。**

后面再复杂的句向量模型、双塔检索模型、LLM embedding API，  
本质上仍然在利用这一点。

---

## 四、从词向量到上下文化表示

### 4.1 早期 embedding：一个词通常只有一个固定向量

例如传统词向量里：

- `bank`

不管是在：

- river bank
- bank account

通常都是同一个向量。

这就会带来歧义问题。

### 4.2 上下文化表示：同一个词在不同语境里可以变

到了 Transformer 时代，  
词的表示不再完全固定，而是会根据上下文变化。

也就是说：

- `bank` 在金融语境里的向量
- `bank` 在河岸语境里的向量

可以不同。

这就是上下文化表示最重要的进步之一。

### 4.3 一个简单的上下文模拟

下面这个例子不是真正的 Transformer，  
但它能帮你先建立“同词不同向量”的直觉。

```python
base_bank = [0.50, 0.50, 0.50]
finance_context = [0.30, -0.10, 0.20]
river_context = [-0.20, 0.25, -0.10]

bank_in_finance = [a + b for a, b in zip(base_bank, finance_context)]
bank_in_river = [a + b for a, b in zip(base_bank, river_context)]

print("bank in finance:", [round(x, 2) for x in bank_in_finance])
print("bank in river  :", [round(x, 2) for x in bank_in_river])
```

它的意义不是模拟真实模型，  
而是帮你记住：

- 静态 embedding：一个词一个向量
- 上下文化表示：同一个词会随上下文变化

---

## 五、Embedding 在真实项目里有什么用？

### 5.1 检索和 RAG

把问题和文档都编码成向量后，  
就可以做：

- 相似度召回

这正是很多 RAG 系统的基础。

### 5.2 语义聚类和去重

如果两段文本向量很近，  
往往表示它们语义也相近。

这可以用于：

- 文本聚类
- FAQ 合并
- 近重复检测

### 5.3 作为下游任务输入特征

很多分类、匹配、排序任务也会先把文本变成 embedding，  
再在其上训练头部或做相似度打分。

---

## 六、Embedding 最容易被误解的地方

### 6.1 误区一：向量接近就一定是同义词

不一定。  
它更可能意味着：

- 分布相近
- 用法相关

### 6.2 误区二：句向量越简单越好

平均词向量虽然直观，  
但很容易丢掉：

- 顺序
- 否定
- 长距离依赖

### 6.3 误区三：有 embedding 就等于理解了语言

embedding 只是表示层，  
真正的理解还需要：

- 上下文建模
- 任务目标
- 训练数据

---

## 小结

这节最重要的不是记住几个相似度公式，  
而是建立这样一个判断：

> **Embedding 的核心价值，是把离散的 token 变成一个可比较、可组合、能反映语义关系的向量空间。**

只要你抓住这条主线，  
后面再学：

- 句向量
- 检索模型
- RAG
- 上下文化表示

就会自然很多。

---

## 练习

1. 把示例里的词向量自己改一改，观察哪些词会变得更近或更远。
2. 为什么说平均词向量能建立第一层直觉，但不适合表达所有语义现象？
3. 用自己的话解释：静态 embedding 和上下文化表示最大的差别是什么？
4. 想一想：如果你在做 FAQ 检索，embedding 最先能帮你解决什么问题？

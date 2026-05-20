---
title: "11.2.1 表示学习路线图：用向量表达语义"
sidebar_position: 0
description: "表示学习的简短实操路线：比较稀疏特征、词向量、上下文向量和语言模型表示。"
keywords: [表示学习指南, 词向量, 上下文化表示, 语言模型]
---

# 11.2.1 表示学习路线图：用向量表达语义

表示学习关心的是：文本如何变成带语义的数字，而不仅仅是编号。

## 先看表示路径

![NLP 表示学习章节学习顺序图](/img/course/ch11-embeddings-chapter-flow.webp)

![Embedding 语义空间图](/img/course/embedding-semantic-space.webp)

![上下文向量对比图](/img/course/contextual-embedding-comparison.webp)

这条路径从稀疏词身份，到词向量，到上下文向量，再到学习更广泛语言模式的语言模型。

## 跑一个相似度检查

```python
vectors = {
    "cat": [1.0, 0.8],
    "dog": [0.9, 0.7],
    "car": [0.1, 0.2],
}

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

print("cat_dog:", round(dot(vectors["cat"], vectors["dog"]), 2))
print("cat_car:", round(dot(vectors["cat"], vectors["car"]), 2))
```

预期输出：

```text
cat_dog: 1.46
cat_car: 0.26
```

这是玩具分数，但体现了核心思想：语义接近的文本，应该更容易被模型比较。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 词向量 | 解释语义接近等于向量接近 |
| 2 | 上下文化表示 | 解释同一个词为什么会有不同含义 |
| 3 | 语言模型 | 把表示学习连接到 next-token 或 masked prediction |

## 通过标准

如果你能比较稀疏特征、词向量和上下文向量，并解释表示质量为什么影响分类、检索和 RAG，就通过了本章。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
representation: BoW, TF-IDF, static embedding, contextual embedding, or language-model score
comparison: nearest text, similarity score, or next-token/log-prob style output
interpretation: what the representation captures and what it misses
failure_check: polysemy, domain mismatch, short text, tokenization, or semantic drift
Expected_output: small comparison table with at least one surprising result
```

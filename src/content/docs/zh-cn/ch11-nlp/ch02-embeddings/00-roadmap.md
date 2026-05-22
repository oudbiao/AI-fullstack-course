---
title: "11.2.1 表示学习路线图：用向量表达语义"
description: "表示学习的简短实操路线：比较稀疏特征、词向量、上下文向量和语言模型表示。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "表示学习指南, 词向量, 上下文化表示, 语言模型"
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

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要从文本单元和输出类型说起：token、span、句子标签、序列、embedding 或生成文本。
2. 证据应包含小样本、模型或 pipeline 选择、评价指标，以及至少一个被检查过的错误案例。
3. 自检时要能区分预处理问题和模型问题，例如分词错误、标签歧义、数据不平衡或生成幻觉。

</details>


## 留下的证据

学完这一页，至少保留这张证据卡：

```text
表示：BoW、TF-IDF、静态 embedding、上下文 embedding，或语言模型分数
比较：最近文本、相似度分数或下一 token/log-prob 风格输出
解释: 该表示捕捉了什么，以及遗漏了什么
失败检查：一词多义、领域不匹配、文本过短、分词问题或语义漂移
期望产出：至少有一个意外结果的小型对比表
```

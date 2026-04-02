---
title: "1.3 文本表示方法"
sidebar_position: 3
description: "从 one-hot、词袋模型、TF-IDF 到向量相似度，理解文本如何被转换成机器可计算的数字。"
keywords: [文本表示, one-hot, BoW, TF-IDF, cosine similarity, embedding]
---

# 文本表示方法

## 学习目标

完成本节后，你将能够：

- 理解为什么文本必须先表示成数字
- 掌握 one-hot、词袋模型、TF-IDF 的基本思路
- 会用纯 Python 写一个简单的文本向量化示例
- 理解传统表示方法和 embedding 的差异

---

## 一、为什么文本必须先数值化？

模型不会直接理解“我喜欢这门课”这几个字。  
它只能处理数字。

所以 NLP 里有一个绕不过去的步骤：

> **把文本变成向量。**

类比一下：

- 人读句子，看到的是语义
- 机器读句子，先看到的是一串数字坐标

---

## 二、one-hot：最朴素的表示法

假设词表只有 4 个词：

```python
["i", "love", "nlp", "python"]
```

那每个词都可以用一个只有一个位置为 1 的向量表示：

| 词 | 向量 |
|---|---|
| i | `[1, 0, 0, 0]` |
| love | `[0, 1, 0, 0]` |
| nlp | `[0, 0, 1, 0]` |
| python | `[0, 0, 0, 1]` |

它的优点是简单。  
缺点也很明显：

- 维度会很高
- 词和词之间没有语义距离

比如 `love` 和 `like` 在 one-hot 空间里并不会更接近。

---

## 三、词袋模型（Bag of Words）

词袋模型的核心思想非常朴素：

> 不看顺序，只看每个词出现了几次。

下面给一个直接可运行的实现。

```python
from collections import Counter

docs = [
    "i love nlp",
    "i love python",
    "python love me"
]

tokenized_docs = [doc.split() for doc in docs]
vocab = sorted(set(token for doc in tokenized_docs for token in doc))
vocab_index = {word: idx for idx, word in enumerate(vocab)}

def to_bow_vector(tokens):
    vector = [0] * len(vocab)
    counts = Counter(tokens)
    for word, count in counts.items():
        vector[vocab_index[word]] = count
    return vector

print("词表:", vocab)
for doc, tokens in zip(docs, tokenized_docs):
    print(doc, "->", to_bow_vector(tokens))
```

你会发现，句子已经被变成了固定长度向量。

---

## 四、BoW 的优点和局限

### 优点

- 简单
- 易解释
- 很适合入门和小数据任务

### 局限

- 不考虑词序
- 无法表达上下文
- 语义相近的词不会自动接近

例如：

- “狗咬人”
- “人咬狗”

BoW 看起来可能很像，但意思完全不同。

---

## 五、TF-IDF：给“更有信息量”的词更高权重

词袋模型只数次数，但高频词不一定有信息量。

比如：

- 在所有文档里都常见的词，区分力很弱
- 只在少数文档里高频出现的词，往往更有区分力

这就是 TF-IDF 的直觉。

### 公式直觉版

- `TF`：这个词在当前文档里出现得多不多
- `IDF`：这个词在整个语料里稀不稀有

所以：

> **一个词在当前文档很重要、但在全语料又不那么常见，它的 TF-IDF 就会高。**

---

## 六、纯 Python 实现一个简单 TF-IDF

```python
import math
from collections import Counter

docs = [
    "python is great for data analysis",
    "python is great for machine learning",
    "basketball is a great sport"
]

tokenized_docs = [doc.split() for doc in docs]
vocab = sorted(set(token for doc in tokenized_docs for token in doc))

def compute_idf(tokenized_docs, vocab):
    n_docs = len(tokenized_docs)
    idf = {}
    for word in vocab:
        df = sum(1 for doc in tokenized_docs if word in doc)
        idf[word] = math.log((n_docs + 1) / (df + 1)) + 1
    return idf

idf = compute_idf(tokenized_docs, vocab)

def to_tfidf(tokens, vocab, idf):
    counts = Counter(tokens)
    total = len(tokens)
    vector = []
    for word in vocab:
        tf = counts[word] / total
        vector.append(round(tf * idf[word], 4))
    return vector

print("词表:", vocab)
for doc, tokens in zip(docs, tokenized_docs):
    print(doc)
    print(to_tfidf(tokens, vocab, idf))
```

---

## 七、相似度：两个句子有多像？

当文本被向量化后，就可以计算相似度。

最常见的是 **余弦相似度**。

直觉上，你可以把它理解成：

> 两个向量“朝向”有多接近。

```python
import math
from collections import Counter

docs = [
    "i love python",
    "i love coding",
    "weather is sunny"
]

tokenized_docs = [doc.split() for doc in docs]
vocab = sorted(set(token for doc in tokenized_docs for token in doc))
vocab_index = {word: idx for idx, word in enumerate(vocab)}

def to_bow(tokens):
    vector = [0] * len(vocab)
    counts = Counter(tokens)
    for word, count in counts.items():
        vector[vocab_index[word]] = count
    return vector

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

vec1 = to_bow(tokenized_docs[0])
vec2 = to_bow(tokenized_docs[1])
vec3 = to_bow(tokenized_docs[2])

print("句子1 vs 句子2:", round(cosine_similarity(vec1, vec2), 4))
print("句子1 vs 句子3:", round(cosine_similarity(vec1, vec3), 4))
```

通常你会看到：

- “i love python” 和 “i love coding” 更相似
- 和 “weather is sunny” 相似度更低

---

## 八、Embedding 又解决了什么？

前面的 one-hot、BoW、TF-IDF 都属于“传统表示法”。  
它们有个共同问题：

> 不太会自动理解语义。

Embedding 的核心思想是：

> **让语义相近的词，在向量空间里也更接近。**

比如：

- `king`
- `queen`
- `man`
- `woman`

在好的 embedding 空间里，它们会形成更有规律的几何关系。

这就是为什么后面的词向量、BERT、Transformer 会那么重要。

---

## 九、传统表示法还有用吗？

有，而且非常有用。

在很多场景里：

- 数据量不大
- 模型需要可解释
- 任务比较简单

BoW / TF-IDF 依然是很好的起点。

所以你不该把它们看成“过时内容”，而是看成：

> NLP 表示学习的起点。

---

## 十、初学者常见误区

### 1. 以为词向量一出来，BoW 就没用了

不是。  
很多简单任务里，TF-IDF + 线性模型依然很好用。

### 2. 只背概念，不亲手把句子变成向量

建议你一定要亲手跑一次上面的代码。  
否则“文本数值化”这件事很容易一直悬在空中。

### 3. 把“高维”误解成“高级”

维度高不等于表示更好，关键看是否携带有效信息。

---

## 小结

这节课的核心是：

1. 机器不能直接吃文字，必须先数值化
2. one-hot、BoW、TF-IDF 是最基础的文本表示方法
3. embedding 解决的是更深层的语义表示问题

后面你学词向量、BERT、GPT，本质上都是在学习“更高级的文本表示”。

---

## 练习

1. 把 `docs` 换成你自己的 3 句话，重新生成词表和 BoW 向量。
2. 给 TF-IDF 例子再加一条和 `python` 完全无关的句子，观察权重变化。
3. 思考：为什么 BoW 不关心词序会带来问题？

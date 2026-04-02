---
title: "7.2 项目：文本摘要系统"
sidebar_position: 22
description: "从抽取式摘要、句子打分到系统评估，走通一个小型文本摘要项目的完整闭环。"
keywords: [text summarization, extractive summarization, sentence scoring, TF-IDF, NLP project]
---

# 项目：文本摘要系统

:::tip 本节定位
文本摘要是一个非常适合做项目的 NLP 任务，因为它会逼着你同时思考：

- 什么叫“关键信息”
- 怎样把长文本压缩
- 怎样判断摘要好不好

这一节我们先做一个**抽取式摘要系统**，用更可解释的方式把整条链路走通。
:::

## 学习目标

- 理解文本摘要任务的核心目标与常见路线
- 学会把一篇文本切成句子并进行句子打分
- 用 TF-IDF 做一个最小可运行的抽取式摘要系统
- 理解摘要项目该怎样评估与迭代

---

## 一、文本摘要系统到底在做什么？

### 1.1 摘要不是“缩短文本”这么简单

真正的摘要系统至少要做两件事：

1. 找出原文里最重要的信息
2. 用更短的形式表达这些信息

所以摘要任务的难点不是“删字”，而是：

> **删掉不重要的信息，但保住关键内容。**

### 1.2 先分清两种摘要路线

| 路线 | 做法 |
|---|---|
| 抽取式摘要 | 直接从原文中挑关键句 |
| 生成式摘要 | 让模型重新写一个更短版本 |

对于当前阶段，最适合先学的是：

> **抽取式摘要。**

因为它：

- 更稳定
- 更可解释
- 更容易评估

---

## 二、先准备一段可以直接实验的文本

```python
article = """
人工智能课程的学习路径通常分为基础阶段和进阶阶段。
基础阶段包括 Python 编程、数据分析和机器学习。
当学习者掌握了这些内容之后，才能更稳地进入深度学习和大模型应用开发。
很多人一开始就想直接学大模型，但往往因为基础不牢而很快卡住。
如果学习目标是做 AI 应用工程，理解数据处理、模型训练和系统部署都很重要。
""".strip()

print(article)
```

### 2.2 为什么先用短文本？

因为摘要系统的教学重点，不是大语料处理，而是先让你看清：

- 句子怎么拆
- 重要性怎么评
- 结果怎么选

---

## 三、第一步：切句

### 3.1 为什么切句是第一步？

抽取式摘要通常是“从句子里挑几句”。  
所以第一件事自然是先把文章切成句子。

```python
import re

def split_sentences(text):
    parts = re.split(r"[。！？\\n]+", text)
    return [p.strip() for p in parts if p.strip()]

sentences = split_sentences(article)
for i, s in enumerate(sentences, start=1):
    print(i, s)
```

### 3.2 这一步为什么值得重视？

因为切句切不好，后面的摘要单元就会错。  
真实项目里，切句和分段质量经常直接影响摘要效果。

---

## 四、第二步：给句子打分

### 4.1 一个最简单的思路

如果一个句子里包含很多“全篇高频但又有区分度的词”，它往往更重要。

这就是 TF-IDF 非常适合拿来做摘要 baseline 的原因。

### 4.2 可运行示例：句子打分

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

sentences = split_sentences(article)

vectorizer = TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b")
X = vectorizer.fit_transform(sentences)

# 每句的重要性分数：TF-IDF 向量和
scores = np.asarray(X.sum(axis=1)).ravel()

for sent, score in zip(sentences, scores):
    print(round(float(score), 4), "->", sent)
```

### 4.3 为什么这能工作？

因为摘要本质上就是在估计：

> 哪些句子最能代表全文主题。 

TF-IDF 不是完美方法，但对入门项目非常有价值，因为它：

- 简单
- 稳定
- 可解释

---

## 五、第三步：选出 top-k 句子组成摘要

### 5.1 先按分数选句子

```python
import numpy as np

def extract_summary(text, top_k=2):
    sentences = split_sentences(text)
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b")
    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()

    top_idx = scores.argsort()[::-1][:top_k]
    top_idx = sorted(top_idx)  # 恢复原文顺序

    summary = "。".join(sentences[i] for i in top_idx) + "。"
    return summary, scores

summary, scores = extract_summary(article, top_k=2)
print("摘要：")
print(summary)
```

### 5.2 为什么要恢复原文顺序？

因为如果你只按分数从高到低拼接，摘要可能会乱序。  
而摘要虽然是缩写，但最好还能保持原文阅读顺序。

---

## 六、把整个项目包装成一个完整流程

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

def split_sentences(text):
    parts = re.split(r"[。！？\\n]+", text)
    return [p.strip() for p in parts if p.strip()]

def summarize(text, top_k=2):
    sentences = split_sentences(text)
    if len(sentences) <= top_k:
        return text

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b")
    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()

    top_idx = scores.argsort()[::-1][:top_k]
    top_idx = sorted(top_idx)

    return "。".join(sentences[i] for i in top_idx) + "。"

article = """
人工智能课程的学习路径通常分为基础阶段和进阶阶段。
基础阶段包括 Python 编程、数据分析和机器学习。
当学习者掌握了这些内容之后，才能更稳地进入深度学习和大模型应用开发。
很多人一开始就想直接学大模型，但往往因为基础不牢而很快卡住。
如果学习目标是做 AI 应用工程，理解数据处理、模型训练和系统部署都很重要。
""".strip()

print("原文：")
print(article)
print("\n摘要：")
print(summarize(article, top_k=2))
```

现在你已经有一个真正能工作的最小摘要系统了。

---

## 七、怎样评估一个摘要系统？

### 7.1 最简单的人工评估问题

你可以先问自己：

- 有没有覆盖最重要信息？
- 有没有明显漏掉主线？
- 读起来顺不顺？

### 7.2 一个很实用的项目评估思路

即使没有复杂指标，也可以先准备一组文章，人工看：

- top-1 摘要是否过短
- top-2 是否覆盖更完整
- 哪些文章容易失败

### 7.3 真实项目还会继续上什么？

更进一步可以考虑：

- ROUGE
- BERTScore
- 人工偏好评价

但在入门项目里，先把 baseline 跑稳更重要。

---

## 八、抽取式摘要的局限

### 8.1 它不会真正“改写”

它只能从原文里挑句子，不能写出更流畅的新句子。

### 8.2 它容易选出重复信息

如果多句说的是类似意思，系统可能会都选进来。

### 8.3 它不一定擅长长距离整合

有些文章真正重要的信息分散在几处，简单句子打分未必能很好整合。

所以抽取式摘要更像：

> 一个稳定、可解释的 baseline。 

---

## 九、这个项目怎样继续升级？

下一步你可以尝试：

1. 加入去重逻辑，避免摘要句子内容太相近
2. 加入位置权重，让开头句适当更容易被选中
3. 替换成 embedding 句向量打分
4. 再进一步升级到生成式摘要

也就是说：

> 这个项目是你走向更复杂摘要系统的踏脚石。 

---

## 十、小结

这一节最重要的不是“做出一个会压缩文本的函数”，而是理解：

> **摘要系统的核心，是先定义什么叫重要信息，再用一个可解释的方法把这些信息留下来。**

抽取式摘要不是最终答案，但它是非常好的项目起点。

---

## 练习

1. 把 `top_k` 从 2 改成 3，比较摘要变化。
2. 自己换一段短文章，看看这个摘要器会选哪些句子。
3. 想一想：为什么恢复原文顺序对摘要可读性很重要？
4. 如果两句话内容高度重复，你会怎样修改这个系统避免都选进去？

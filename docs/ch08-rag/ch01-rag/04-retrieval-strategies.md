---
title: "1.5 检索策略"
sidebar_position: 4
description: "理解关键词检索、向量检索、混合检索、重排和查询改写等常见策略，知道怎样把“找资料”这一步做得更准。"
keywords: [retrieval, hybrid search, rerank, query rewrite, dense retrieval, sparse retrieval]
---

# 检索策略

![Hybrid Search 与 Rerank 流程图](/img/course/hybrid-search-rerank-flow.png)

## 学习目标

完成本节后，你将能够：

- 理解检索策略为什么直接决定 RAG 质量
- 分清关键词检索、向量检索和混合检索
- 理解 rerank、query rewrite 这些常见增强手段
- 用一个可运行例子体验混合检索的思路

---

## 一、检索不是“只有一个 top-k”

### 1.1 关键词检索：适合找明确词项

关键词检索更像“查目录”。

它擅长：

- 精确术语
- 产品名
- 报错码
- 法条编号

例如用户问：

> “错误码 403 是什么？”

这种场景关键词检索往往很强。

### 1.2 向量检索：适合找语义相近内容

向量检索更像“按意思找相似”。

它擅长：

- 同义表达
- 改写后的问题
- 用户口语化提问

例如：

> “怎么退课？”

和：

> “课程购买后 7 天内可申请退款”

虽然词不一样，但向量检索有机会把它们连起来。

---

## 二、为什么很多项目最后都走向混合检索？

### 2.1 因为关键词和语义各有盲区

只用关键词：

- 容易漏掉语义近但措辞不同的内容

只用向量：

- 有时会忽略特别关键的专有词

所以很多系统会做：

> **关键词分数 + 向量分数 = 混合分数**

### 2.2 这很像“同时看字面和意思”

人类找资料时也会这样：

- 先看有没有明确关键词
- 再判断是不是在说同一件事

混合检索就是把这两种判断合起来。

---

## 三、一个最小可运行的混合检索示例

下面这个例子里：

- `keyword_score` 模拟关键词匹配
- `vector_score` 模拟语义相似度
- 最后把两者做加权组合

```python
import math
import re
from collections import Counter
import numpy as np

docs = [
    {
        "id": "d1",
        "text": "课程购买后 7 天内可申请退款",
        "vector": np.array([0.95, 0.10, 0.05])
    },
    {
        "id": "d2",
        "text": "完成所有项目并通过测试后可获得证书",
        "vector": np.array([0.10, 0.95, 0.10])
    },
    {
        "id": "d3",
        "text": "建议先学 Python，再学机器学习和深度学习",
        "vector": np.array([0.20, 0.30, 0.95])
    }
]

query = "怎么申请退课退款"
query_vector = np.array([0.90, 0.10, 0.10])

def tokenize(text):
    return re.findall(r"[\\w\\u4e00-\\u9fff]+", text.lower())

def keyword_score(query, text):
    q = Counter(tokenize(query))
    t = Counter(tokenize(text))
    return sum(min(q[k], t[k]) for k in q)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

results = []
for doc in docs:
    kw = keyword_score(query, doc["text"])
    vec = cosine_similarity(query_vector, doc["vector"])
    hybrid = 0.4 * kw + 0.6 * vec
    results.append((hybrid, kw, vec, doc["id"], doc["text"]))

for hybrid, kw, vec, doc_id, text in sorted(results, reverse=True):
    print(doc_id, "hybrid=", round(hybrid, 4), "kw=", kw, "vec=", round(vec, 4), "->", text)
```

这个例子虽然简化，但已经很接近真实系统的核心思路。

---

## 四、Rerank：先粗召回，再精排序

### 4.1 为什么要 rerank？

很多系统不会一开始就追求“第一次就排准”，而是：

1. 先用较便宜的方法召回一批候选
2. 再用更强但更贵的方法重排

这就叫 rerank。

### 4.2 一个直觉比喻

像找工作时：

- 第一轮按关键词筛简历
- 第二轮再认真看候选人是否真的合适

RAG 也是一样。

---

## 五、Query Rewrite：用户问题往往不够“适合检索”

### 5.1 用户提问不一定是好检索词

用户可能会说：

> “我这个情况还能退吗？”

但知识库里写的是：

> “购买后 7 天内且学习进度低于 20% 可退款”

这时系统常常会先把问题改写得更适合检索。

### 5.2 一个玩具版查询改写

```python
def rewrite_query(query):
    replacements = {
        "退课": "退款",
        "退掉课程": "退款",
        "拿证": "证书",
        "毕业证": "证书"
    }
    new_query = query
    for old, new in replacements.items():
        new_query = new_query.replace(old, new)
    return new_query

queries = ["怎么退课", "我想拿证", "退掉课程可以吗"]

for q in queries:
    print(q, "->", rewrite_query(q))
```

真实系统里，query rewrite 可能由 LLM 来完成。

---

## 六、还有哪些常见检索增强策略？

### 6.1 Multi-query

把一个问题改写成多个等价问法，再分别检索，合并结果。

### 6.2 Metadata filter

先按业务条件缩小范围，再做语义检索。

### 6.3 Parent-child retrieval

先检索小 chunk，再回到更大块或原文段落。

### 6.4 Self-query retrieval

让模型自动判断需要哪些过滤条件和检索字段。

---

## 七、怎么选检索策略？

### 7.1 如果你有大量专有名词

更要重视：

- 关键词检索
- 混合检索
- 元数据过滤

### 7.2 如果用户表达特别口语化

更要重视：

- 向量检索
- query rewrite
- rerank

### 7.3 如果知识库结构化程度高

可以考虑：

- 先路由
- 再定向检索
- 最后重排

---

## 八、如果你的目标是“知识库驱动的课件生成助手”，检索策略该怎么想？

这类项目里，检索不只是“找到相关内容”，  
而是更像在做两层选择：

1. 先决定从内部资料找，还是从外部资料补
2. 再决定要找知识点、例题，还是练习题

所以很适合先把检索条件写成这种样子：

| 条件 | 它在帮你控制什么 |
|---|---|
| `topic` | 当前主题 |
| `content_type` | 概念 / 例题 / 练习 |
| `source_origin` | 内部资料 / 外部资料 |
| `grade` | 年级或适用对象 |

你可以先把这条线记成一句话：

> **课件生成项目里的检索，不只是“找相关”，而是“按栏目找对资料”。**

一个最小过滤示例可以先写成：

```python
items = [
    {"topic": "折扣应用题", "content_type": "concept", "source_origin": "internal", "text": "折扣 = 原价 × 折扣率"},
    {"topic": "折扣应用题", "content_type": "example", "source_origin": "internal", "text": "商品原价 100 元，打 8 折后是多少元？"},
    {"topic": "折扣应用题", "content_type": "note", "source_origin": "external", "text": "外部资料补充：折扣常见误区。"},
]

hits = [
    x for x in items
    if x["topic"] == "折扣应用题" and x["content_type"] in {"concept", "example"}
]

for hit in hits:
    print(hit)
```

这个例子特别适合新人，因为它会让你先看到：

- metadata filter 往往比“换更大的模型”更先见效

---

## 九、初学者常见误区

### 9.1 只测向量检索，不测关键词检索

很多企业场景里，关键词检索并不弱，甚至是基础盘。

### 9.2 检索策略一开始就做太复杂

建议先从：

1. 一个 baseline
2. 一个明确评估集
3. 一次只改一个策略

开始。

### 9.3 只看召回，不看最终回答

检索分数高，不代表最终答案一定更好。  
因为生成阶段也会影响表现。

---

## 检索策略调参表

调检索时，不要只说“效果不好”，要把现象映射到可以调整的杠杆。

| 现象 | 优先调整 | 为什么 |
|---|---|---|
| 明确术语、报错码搜不到 | 增加关键词检索或混合检索 | 向量检索可能弱化精确词 |
| 用户口语化问题搜不到 | query rewrite、multi-query、向量检索 | 用户表达和文档表达不一致 |
| top-k 里相关内容排得靠后 | rerank | 粗召回能找到，但排序不够准 |
| 检索结果主题对但版本错 | metadata filter | 需要按版本、日期、来源缩小范围 |
| 答案需要跨多个片段 | parent-child retrieval 或更合理的 chunk | 小 chunk 命中但上下文不足 |

这张表适合和评估集一起使用。每次只改一个策略，然后记录 Hit@k、MRR、引用质量和失败样本变化。

## 一个检索实验记录模板

| 实验 | 策略 | top-k | 是否 rerank | 结果 | 结论 |
|---|---|---:|---|---|---|
| baseline | 关键词 | 3 | 否 | 能命中精确词，漏掉同义问法 | 适合报错和术语 |
| exp-1 | 向量 | 3 | 否 | 同义问法更好，专有词有时不准 | 需要保留关键词通道 |
| exp-2 | 混合 | 5 | 是 | 整体最好，但延迟增加 | 可作为标准版本 |

检索优化的关键不是一次找到完美策略，而是让每次改动都有记录、有指标、有失败样本。

## 小结

这节课最关键的认识是：

> RAG 的“找资料”不是机械步骤，而是一个可以不断设计和优化的系统环节。

很多时候，检索策略升级带来的收益，比换一个更大的模型还直接。

---

## 练习

1. 修改混合检索示例里的权重，比较关键词权重更高和向量权重更高时排序变化。
2. 给文档再加一条包含“退课”字样的句子，观察关键词检索优势。
3. 自己设计一个更丰富的 `rewrite_query()` 规则表。

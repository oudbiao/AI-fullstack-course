---
title: "10.1 项目：智能研究助手"
sidebar_position: 54
description: "围绕检索、阅读、引用和结构化总结，建立一个研究助手 Agent 的作品级项目闭环。"
keywords: [research assistant, citation, retrieval, summary, agent project, RAG]
---

# 项目：智能研究助手

:::tip 本节定位
研究助手项目很适合作为 Agent 作品集，不是因为它看起来高级，而是因为它天然要求系统同时做好：

- 检索
- 阅读
- 总结
- 引用追踪

只要其中一环松掉，结果就会变得“不可信”。  
所以它非常适合练“可信 Agent”这条主线。
:::

## 学习目标

- 学会把研究助手项目范围定得清楚
- 学会把“检索 -> 阅读 -> 总结 -> 引用”串成闭环
- 学会定义这个项目最关键的评估标准
- 学会把它包装成一个有说服力的作品集项目

---

## 一、先把项目范围定窄

一个适合练手的研究助手项目，建议先做成：

- 给定主题
- 检索若干文档
- 输出结构化摘要
- 每条摘要带来源

而不是一上来就做：

- 自动写论文
- 自动做综述

### 为什么？

因为对研究助手来说，“可信”比“花哨”更重要。

---

## 二、作品级研究助手最小闭环长什么样？

1. 输入主题或问题
2. 检索候选资料
3. 选出最相关资料
4. 生成结构化摘要
5. 给出每条摘要来源
6. 做错误分析和回归集

只要这 6 步清楚，这个项目就很有作品集价值。

## 三、推荐推进顺序

对新人来说，更稳的顺序通常是：

1. 先把主题范围收窄
2. 再做最简单检索 baseline
3. 再补结构化总结
4. 最后再补引用校验和失败案例展示

这样你才更容易把“可信研究助手”做成一个清楚闭环。

---

## 四、先看一个最小研究助手示例

下面这个例子会做三件事：

1. 用关键词匹配模拟检索
2. 生成结构化总结
3. 给每条总结附来源

```python
docs = [
    {
        "id": "d1",
        "title": "RAG improves factual grounding",
        "text": "RAG can improve factual grounding by retrieving external evidence.",
        "keywords": {"rag", "retrieval", "grounding", "evidence"},
    },
    {
        "id": "d2",
        "title": "Long context still struggles with precision",
        "text": "Long context models may still miss key details without retrieval or re-ranking.",
        "keywords": {"long", "context", "retrieval", "ranking"},
    },
    {
        "id": "d3",
        "title": "Citations increase user trust",
        "text": "Users trust generated summaries more when each claim is tied to an explicit source.",
        "keywords": {"citation", "trust", "summary", "source"},
    },
]


def retrieve(query, top_k=2):
    query_terms = set(query.lower().split())
    scored = []
    for doc in docs:
        score = len(query_terms & doc["keywords"])
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:top_k] if score > 0]


def summarize_with_citations(query):
    hits = retrieve(query, top_k=2)
    bullets = []
    for doc in hits:
        bullets.append(
            {
                "claim": doc["text"],
                "source_id": doc["id"],
                "source_title": doc["title"],
            }
        )
    return bullets


query = "rag retrieval citation trust"
result = summarize_with_citations(query)
for item in result:
    print(item)
```

### 4.1 这个例子为什么比“项目骨架 dataclass”更有价值？

因为它已经体现出研究助手最关键的产品特征：

- 结果不是一段黑盒总结
- 每条结论都能回到来源

### 4.2 为什么引用是这类项目的命门？

因为没有来源，用户很难区分：

- 这是系统真的从文档里读出来的
- 还是模型自己编的

---

## 五、这个项目最该怎么评估？

### 5.1 检索质量

例如：

- 命中的文档是否真的相关

### 5.2 总结质量

例如：

- 是否覆盖关键点
- 是否过度概括

### 5.3 引用准确性

这是研究助手特别重要的一层：

- 每条 claim 是否真的能在引用来源里找到支持

### 5.4 一个最小评估数据结构

```python
eval_cases = [
    {
        "query": "rag retrieval grounding",
        "expected_source_ids": {"d1", "d2"},
    },
    {
        "query": "citation trust summary",
        "expected_source_ids": {"d3"},
    },
]

for case in eval_cases:
    hit_ids = {item["source_id"] for item in summarize_with_citations(case["query"])}
    print({
        "query": case["query"],
        "hit_ids": hit_ids,
        "overlap": hit_ids & case["expected_source_ids"],
    })
```

---

## 六、最容易踩的坑

### 5.1 检索对了，但总结丢了关键点

### 5.2 总结看起来通顺，但来源对不上

### 5.3 项目只展示一段“看起来很聪明”的回答

研究助手最值得展示的其实是：

- 查询词
- 检索结果
- 摘要条目
- 引用来源

这条完整 trace。

---

## 七、怎么把它打磨成作品级项目？

### 7.1 页面上分四栏展示

- Query
- Retrieved sources
- Structured summary
- Citations

### 7.2 准备 5~10 个固定评估问题

这样你可以稳定展示：

- before / after
- 检索策略变更
- 总结策略改进

### 7.3 单独列失败案例

例如：

- 检索到不相关文档
- 正确文档被漏掉
- 总结 claim 与引用不一致

---

## 八、小结

这节最重要的是建立一个作品级判断：

> **研究助手项目真正的亮点，不是“会总结”，而是“能把检索、总结和引用组织成可信、可追踪、可复核的输出”。**

只要这点立住，这个项目就会很像一个成熟的 Agent 作品。

## 九、项目交付时最好补上的内容

- 一张查询到引用的流程图
- 一组检索结果与最终总结并排展示
- 一组引用对不上或总结漏点的失败案例
- 一段你对“可信输出”怎么定义的说明

---

## 练习

1. 给示例再加一篇文档，让某个 query 出现“相关文档竞争”。
2. 想一想：为什么研究助手里“引用准确性”比普通问答更关键？
3. 如果某条总结很好看但来源对不上，你会算它成功吗？为什么？
4. 如果你把这个项目做成作品集，首页最该展示哪 4 块？

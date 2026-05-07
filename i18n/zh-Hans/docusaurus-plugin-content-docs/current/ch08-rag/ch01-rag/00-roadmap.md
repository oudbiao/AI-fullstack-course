---
title: "8.1.1 RAG 路线图：文档、检索、回答"
sidebar_position: 0
description: "RAG 的简短实操路线：把文档变成可检索分块，检索证据，带引用回答，并评估失败。"
keywords: [RAG 指南, 检索增强生成, 向量数据库, 文档分块, 重排序, RAG 评估]
---

# 8.1.1 RAG 路线图：文档、检索、回答

RAG 解决的是一个很实际的问题：模型不知道所有最新、私有或带来源要求的事实，所以应用必须先检索证据，再让模型回答。

## 8.1.1.1 先看 RAG 流水线

![RAG 在大模型应用中的位置桥接图](/img/course/ch08-rag-position-bridge.png)

![RAG 核心章节学习顺序图](/img/course/ch08-rag-core-chapter-flow.png)

![RAG 从资料到回答的流水线图](/img/course/ch08-rag-data-to-answer-pipeline.png)

核心闭环是：加载文档、切分 chunk、添加 metadata、embedding、检索、重排序、组装上下文、回答、引用来源、评估。

## 8.1.1.2 跑一个极小检索检查

这还不是向量数据库，而是检索习惯的离线迷你版：给分块打分，打印来源，确认证据是否匹配问题。

```python
chunks = [
    {"source": "rag.md", "text": "RAG retrieves source chunks before the model answers."},
    {"source": "eval.md", "text": "Citations let users verify whether an answer is grounded."},
    {"source": "deploy.md", "text": "Deployment exposes the model through a stable API."},
]

query = "why do RAG answers need citations"
query_terms = set(query.lower().split())

def score(chunk):
    words = set(chunk["text"].lower().replace(".", "").split())
    return len(query_terms & words)

for chunk in sorted(chunks, key=score, reverse=True)[:2]:
    print(chunk["source"], score(chunk))
```

预期输出：

```text
rag.md 2
eval.md 1
```

如果排在最前面的来源不相关，不要先改最终 Prompt。先检查文档解析、分块、metadata 和检索覆盖率。

## 8.1.1.3 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | RAG 基础 | 画出问题 → 证据 → 回答闭环 |
| 2 | 文档处理 | 产出带来源和 metadata 的 chunks |
| 3 | 向量数据库 | 解释 embedding、向量记录和相似度搜索 |
| 4 | 检索策略 | 比较关键词、向量、混合、过滤和重排序 |
| 5 | 优化与高级 RAG | 排查召回差、排序差和上下文弱的问题 |
| 6 | RAG 评估 | 测试回答正确性、引用支撑和无答案处理 |

## 8.1.1.4 通过标准

如果你能为至少 10 个固定问题构建一个最小知识库问答闭环，并打印检索分块、回答文本和来源引用，就通过了本章。

本章出口小项目是课程知识库助手：准备 3 到 5 篇 Markdown 文档、top-k 检索输出、来源显示和一张简单评估表。

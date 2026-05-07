---
title: "8 LLM 应用开发与 RAG"
sidebar_position: 0
description: "构建可操作的 RAG 应用闭环：文档、切块、检索、引用、评估、API 封装和工程日志。"
keywords: [LLM应用, RAG, Prompt Engineering, LangChain, 向量数据库, 大模型部署]
---

# 8 LLM 应用开发与 RAG

![LLM 应用与 RAG 主视觉](/img/course/ch08-rag-engineering.png)

第 7 章讲清楚大模型怎样生成文本。第 8 章把模型接成真实应用：**连接文档、检索证据、带引用回答、记录失败，并用评估集持续改进。**

可以把 RAG 理解成“回答前先读资料”。当答案必须来自课程笔记、公司文档、产品手册或私有知识库时，模型不应该只凭记忆猜。

## 先看 RAG 应用闭环

![RAG 应用闭环](/img/course/ch08-rag-app-loop.svg)

整章围绕这条闭环学习。

| 层 | 负责什么 | 应该打印或保存什么 |
|---|---|---|
| 知识层 | 解析文档、清洗文本、切块、保留 metadata | `chunks.jsonl`、来源、章节、页码、版本 |
| 检索层 | 找出与问题最相关的片段 | query、top-k 片段、分数、来源 ID |
| 生成层 | 让 LLM 只基于检索上下文回答 | 最终 Prompt、答案、引用、无法回答原因 |
| 应用层 | 封装成 CLI、API、聊天界面或内部工具 | 请求、响应、错误处理、用户反馈 |
| 运维层 | 持续比较质量、成本、延迟和失败 | 评估集、日志、token 成本、耗时、失败样本 |

## 学习顺序与任务表

完整工作坊放在基础之后。先看见检索链路，再逐步替换更强组件。

| 步骤 | 阅读内容 | 要动手做什么 | 留下什么证据 |
|---|---|---|---|
| 8.1 | RAG 基础、文档处理、检索、评估 | 做一个最小“文档到答案”闭环 | chunks、top-k 输出、带引用答案 |
| 8.2 | 部署与统一 API | 理解云 API、本地模型、统一调用层 | 一份调用笔记或配置对比 |
| 8.3 | LLM 应用开发 | 给 RAG 闭环加 API、工具、对话或文档解析 | 请求/响应样例和错误路径 |
| 8.4 | 工程实践 | 加异步、日志、监控、API 设计或 Docker 说明 | 日志、配置、部署清单 |
| 8.5 | 阶段项目 | 运行 [8.5.6 实操：第 8 章 RAG 应用完整工作坊](./ch05-projects/05-stage-hands-on-workshop.md) | 工作坊输出、一个新增文档、一个新增评估问题 |

## 第一个可运行循环：不用框架的 Tiny RAG

在 LangChain、LlamaIndex 或向量数据库之前，先跑最小链路。目标不是检索器很强，而是看清每一步。

新建 `ch08_tiny_rag.py`，用 Python 3.10 或更新版本运行。

```python
import re

docs = [
    {
        "id": "ragops",
        "source": "study-guide.md#ragops",
        "text": "A RAG app needs an evaluation set with fixed questions, expected sources, ideal answers, and failure labels.",
    },
    {
        "id": "chunking",
        "source": "rag-basics.md#chunking",
        "text": "A RAG app splits documents into chunks and keeps source metadata so answers can cite evidence.",
    },
    {
        "id": "agentops",
        "source": "agent-guide.md#trace",
        "text": "Agent systems record tool calls, observations, permissions, and recovery steps.",
    },
]

question = "Why does a RAG app need an evaluation set?"
STOPWORDS = {"a", "an", "the", "why", "does", "with", "and", "so", "can", "be"}


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff\u3040-\u30ff]+", text.lower())) - STOPWORDS


query_tokens = tokenize(question)
ranked = sorted(
    (
        (len(query_tokens & tokenize(doc["text"])), doc)
        for doc in docs
    ),
    key=lambda item: item[0],
    reverse=True,
)

print("question:", question)
print("top chunks:")
for score, doc in ranked[:2]:
    print(f"- {doc['id']} score={score} source={doc['source']}")

best = ranked[0][1]
answer = (
    "Use a fixed evaluation set so every RAG change can be compared "
    f"against the same questions and expected sources. [{best['source']}]"
)
print("answer:", answer)
```

预期输出：

```text
question: Why does a RAG app need an evaluation set?
top chunks:
- ragops score=4 source=study-guide.md#ragops
- chunking score=2 source=rag-basics.md#chunking
answer: Use a fixed evaluation set so every RAG change can be compared against the same questions and expected sources. [study-guide.md#ragops]
```

操作提示：新增一段文档、提出一个新问题，并在看最终答案前先打印 top-k 片段。如果证据错了，答案就不能信。

## 调试回答不好的 RAG

![RAG 调试阶梯](/img/course/ch08-rag-debug-ladder.svg)

答案不好时，先定位失败层，再考虑换模型。

| 现象 | 先打印什么 | 可能修复 |
|---|---|---|
| 答案没有来源 | 最终 Prompt 和召回片段 | 在 chunk 中保留来源 ID，并强制引用 |
| 原文有答案但检索不到 | 原文关键词搜索和切块文本 | 调整 chunk 大小、补关键词、使用混合检索 |
| 召回很多但最好片段不在前面 | top-k 分数和人工相关性标注 | 加重排或规则过滤 |
| 答案使用旧信息 | 文档版本和索引构建时间 | 重建索引并加入回归测试 |
| 不知道优化有没有变好 | 同一组问题的前后答案 | 建立固定评估集 |

## 常见错误

- 以为“接了向量数据库”就等于 RAG 完成。RAG 质量还取决于文档、切块、排序、Prompt、引用和评估。
- 还没理解链路就上框架。能打印 query、chunks、prompt、answer、source 之后，框架才更好学。
- 检索为空还让模型硬答。可用的 RAG 应用必须能说“资料中没有足够依据”。
- 忘记 metadata。没有来源、页码、章节和版本，引用和排障都会变弱。
- 凭感觉优化。每次改切块、检索、重排或 Prompt，都要用同一组评估问题比较。

## 通关检查

进入第 9 章前，你应该能做到：

- 解释 RAG 为什么能解决私有、新鲜、可引用知识问题；
- 运行 Tiny RAG 脚本，并在看答案前检查 top-k 片段；
- 创建带来源 metadata 的 chunk，并在答案中引用来源；
- 区分文档、切块、检索、生成、引用和部署失败；
- 跑通第 8 章完整工作坊，新增一个文档、新增一个评估问题，并在 README 中记录结果。

可打印清单见 [8.0 学习检查表](./study-guide.md)。如果想直接做项目，从 [8.5.6 实操：第 8 章 RAG 应用完整工作坊](./ch05-projects/05-stage-hands-on-workshop.md) 开始。

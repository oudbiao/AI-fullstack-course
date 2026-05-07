---
title: "7.2.1 LLM 概览路线图：能力、成本、产品适配"
sidebar_position: 0
description: "紧凑版 LLM 概览路线图：发展历史、核心概念、产业图谱和第一次 API 调用工作台。"
keywords: [LLM 概览, 大语言模型, 模型能力, LLM 应用, API 调用]
---

# 7.2.1 LLM 概览路线图：能力、成本、产品适配

LLM 概览不是模型名称清单，而是帮你判断大模型能做什么、代价是什么，以及什么时候该用 Prompt、RAG、Agent 或微调。

## 7.2.1.1 先看能力栈

![LLM 概览章节关系图](/img/course/ch07-llm-overview-chapter-flow.png)

![大模型能力栈与应用生态图](/img/course/ch07-llm-capability-stack.png)

| 路线 | 适合什么时候 |
|---|---|
| prompt | 模型本身已经足够懂，任务简单 |
| RAG | 私有或会变化的知识需要引用 |
| Agent | 模型需要用工具或分步骤行动 |
| 微调 | 行为、风格、格式需要长期适配 |

## 7.2.1.2 跑一次路线判断

```python
request = {
    "needs_private_docs": True,
    "needs_tool_action": False,
    "needs_repeated_style": False,
}

if request["needs_tool_action"]:
    route = "Agent"
elif request["needs_private_docs"]:
    route = "RAG"
elif request["needs_repeated_style"]:
    route = "fine-tuning"
else:
    route = "prompt"

print("recommended_route:", route)
```

预期输出：

```text
recommended_route: RAG
```

这不是完整架构决策，只是在训练习惯：选择能解决实际产品问题的最小路线。

## 7.2.1.3 按这个顺序学

| 顺序 | 阅读 | 留下什么 |
|---|---|---|
| 1 | [7.2.2 发展历史](./01-development-history.md) | 为什么 scaling 和指令微调重要 |
| 2 | [7.2.3 核心概念](./02-core-concepts.md) | context、token、temperature、延迟、成本 |
| 3 | [7.2.4 产业图谱](./03-industry-landscape.md) | 模型/供应商选择记录 |
| 4 | [7.2.5 LLM 调用工作台](./04-llm-call-workbench.md) | 一条请求/响应记录 |

## 7.2.1.4 通过标准

能从能力、上下文、成本、延迟、数据隐私和路线适配解释一次模型选择，就算通过。

---
title: "9.4.1 记忆路线图：写入、检索、遗忘"
sidebar_position: 0
description: "Agent 记忆的简短实操路线：判断什么值得记，检索合适上下文，更新过期事实，并避免记忆污染。"
keywords: [记忆系统概览, Agent 记忆, 短期记忆, 长期记忆, 情节记忆]
---

# 9.4.1 记忆路线图：写入、检索、遗忘

记忆不是为了让 Agent 看起来像人，而是为了服务任务：减少重复沟通、保留有用上下文、复用经验，并避免过期信息或隐私泄露。

## 先看记忆闭环

![Agent 记忆系统分层图](/img/course/agent-memory-system.webp)

![Agent 记忆系统章节学习顺序图](/img/course/ch09-memory-chapter-flow.webp)

![Agent 记忆写入与检索闭环图](/img/course/ch09-memory-write-retrieve-loop.webp)

核心决策不是“全部保存”，而是什么该保存、何时检索、何时更新、何时遗忘。

## 跑一个记忆写入过滤器

只有稳定偏好和可复用事实才适合进入长期记忆。

```python
events = [
    {"type": "preference", "text": "prefers short examples"},
    {"type": "temporary", "text": "debugging one local error"},
    {"type": "fact", "text": "project uses Python"},
]

memory = []
for event in events:
    if event["type"] in {"preference", "fact"}:
        memory.append(event["text"])

print("saved:", memory)
print("count:", len(memory))
```

预期输出：

```text
saved: ['prefers short examples', 'project uses Python']
count: 2
```

如果一条记忆不有用、不新鲜、没权限或检索不到，它可能比没有记忆更伤害 Agent。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 记忆概览 | 区分上下文窗口、短期记忆、长期记忆 |
| 2 | 短期记忆 | 跟踪跨轮次的当前任务状态 |
| 3 | 长期记忆 | 保存稳定偏好、事实和项目背景 |
| 4 | 情节记忆与程序记忆 | 区分发生过什么和下次怎么做 |
| 5 | 记忆工程 | 设计写入、检索、更新、过期和删除规则 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
memory_type: short-term, long-term, episodic, or procedural
write_rule: when memory is created or updated
retrieve_rule: query, relevance, recency, and permission check
failure_check: stale memory, privacy leak, contradiction, or over-retrieval
cleanup_action: summarize, merge, expire, delete, or ask for confirmation
```

## 通过标准

如果你能解释为什么“记更多”不等于“表现更好”，就通过了本章。

本章出口小项目是一套学习规划助手记忆规则：什么保存、什么确认、什么临时保留、什么删除。

<details>
<summary>参考答案与讲解</summary>

1. 合格答案要描述 agent 循环：目标、计划、工具调用、观察结果、记忆或状态更新，以及停止条件。
2. 证据应包含另一个开发者可以检查的 trace，而不只是最终回答。
3. 自检时要能说出一个安全或可靠性控制，例如工具 schema、权限边界、重试、评估用例或人工复核点。

</details>

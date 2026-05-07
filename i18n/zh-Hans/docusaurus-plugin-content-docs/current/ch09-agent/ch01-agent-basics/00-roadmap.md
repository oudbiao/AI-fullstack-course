---
title: "9.1.1 Agent 基础路线图：目标、状态、行动"
sidebar_position: 0
description: "Agent 基础的简短实操路线：区分 Agent、聊天机器人和工作流，然后构建最小目标-状态-行动闭环。"
keywords: [Agent 指南, 智能体指南, Agent 系统架构, 工具调用, Agent 闭环]
---

# 9.1.1 Agent 基础路线图：目标、状态、行动

Agent 不是一个模型名称，而是一种系统模式：围绕目标组织模型、工具、状态、记忆和反馈，让系统能持续推进任务。

## 9.1.1.1 先看单 Agent 闭环

![Agent 基础位置桥接图](/img/course/ch09-basics-position-bridge.png)

![Agent 基础章节学习顺序图](/img/course/ch09-basics-chapter-flow.png)

![单 Agent 执行闭环图](/img/course/ch09-basics-execution-loop.png)

普通聊天机器人回答一次，工作流执行固定步骤。Agent 可以计划、行动、观察、更新状态，并在目标未完成时继续。

## 9.1.1.2 跑一个极小 Agent 状态闭环

这个脚本还不会调用模型，但会展示 Agent 可调试前至少需要哪些状态。

```python
goal = "summarize RAG citation rules"
state = {"steps": [], "done": False}

for action in ["plan", "search_docs", "summarize"]:
    state["steps"].append(action)

state["done"] = True

print("goal:", goal)
print("steps:", " -> ".join(state["steps"]))
print("done:", state["done"])
```

预期输出：

```text
goal: summarize RAG citation rules
steps: plan -> search_docs -> summarize
done: True
```

如果一个 Demo 不能展示目标、状态、行动、观察和停止条件，先把它称为大模型应用，而不是 Agent。

## 9.1.1.3 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 什么是 Agent | 比较聊天机器人、工作流、RAG 应用和 Agent |
| 2 | 发展历史 | 理解为什么 LLM 让 Agent 系统重新升温 |
| 3 | 能力等级 | 把回答、检索、工具、规划、记忆、协作放到同一条能力阶梯 |
| 4 | 系统架构 | 画出目标、状态、规划器、工具、记忆、观察和执行器 |
| 5 | RL 到 Agent 的突破 | 连接行动、奖励、反馈和规划 |

## 9.1.1.4 通过标准

如果你能画出一个单 Agent 闭环，并解释为什么单 Agent 稳定性要先于多 Agent 协作，就通过了本章。

本章出口小项目是一份研究助手 Agent trace：一个目标、一个计划、至少一个工具决策、一次观察、一个停止条件和一个最终回答。

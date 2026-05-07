---
title: "9.0 学习检查表：AI Agent 与智能体系统"
sidebar_position: 1
description: "第 9 章的简短检查表：Agent 闭环、工具 schema、trace、安全边界、评估和作品集证据。"
keywords: [Agent 检查表, AI Agent 学习, ReAct, MCP, 工具调用, Agent 评估]
---

# 9.0 学习检查表：AI Agent 与智能体系统

这页当成可打印检查表使用。需要完整讲解时，回到 [第 9 章入口页](./index.md)。

![Agent trace 证据包](/img/course/ch09-agent-trace-pack.svg)

## 两小时快速通读

| 时间 | 做什么 | 能说出这句话就停 |
|---|---|---|
| 20 分钟 | 看入口页的执行闭环 | “Agent 是 goal-state-tool-observation 循环。” |
| 25 分钟 | 运行 trace 脚本 | “我能回放每个动作和观察。” |
| 25 分钟 | 浏览 9.1 和 9.2 | “我能区分 Agent、工作流、RAG、ReAct、Plan-and-Execute。” |
| 25 分钟 | 浏览 9.3 工具安全 | “工具 schema 和权限比花哨 Prompt 更重要。” |
| 25 分钟 | 阅读边界选择图 | “我知道什么时候不该用 Agent。” |

## 必须留下的证据

| 证据 | 最小版本 |
|---|---|
| `tools_schema.md` | 1～2 个工具，写清名称、用途、参数、返回值、错误和风险等级 |
| `agent_traces.jsonl` | 至少三次运行，记录 goal、step、action、input、observation、result |
| `safety_boundary.md` | 最大步数、工具白名单、被拦截动作、人工确认规则 |
| `failure_cases.md` | 至少三个失败：选错工具、参数错误、循环、权限拦截、不支持的回答 |
| `eval_tasks.csv` | 3～5 个固定任务，包含期望结果和成功标准 |
| `README.md` | 运行命令、trace 示例、安全样例、评估结果、限制 |

## 离章问题

- 你能说明 Agent 和普通 LLM 应用的区别吗？
- 你能展示一条 trace，并解释每次工具调用为什么发生吗？
- 你能拦截高风险或不在白名单里的工具吗？
- 你能定义停止条件和最大步数吗？
- 你能解释为什么多 Agent 应该在单 Agent 可靠之后再做吗？

如果答案都是可以，就继续下一方向：部署、多模态 Agent，或课程最终项目。

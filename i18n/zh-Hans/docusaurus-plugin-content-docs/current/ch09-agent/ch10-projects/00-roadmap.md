---
title: "9.10.1 项目路线图：构建可追踪 Agent"
sidebar_position: 0
description: "第 9 章项目的简短实操路线：构建包含目标、计划、工具、记忆、trace、安全、评估和部署证据的 Agent 作品集项目。"
keywords: [Agent 项目指南, 研究助手, 数据分析 Agent, 多 Agent 项目, Agent 作品集]
---

# 9.10.1 项目路线图：构建可追踪 Agent

Agent 项目作品集应该展示可追踪执行闭环，而不是只展示一个最终模型回答。

## 先看项目闭环

![Agent 综合项目路线图](/img/course/ch09-projects-route-map.png)

![Agent 项目学习顺序图](/img/course/ch09-project-learning-order-map.png)

![Agent 项目交付闭环图](/img/course/ch09-project-delivery-loop.png)

闭环是：目标、计划、工具调用、观察、状态更新、失败处理、停止决策、最终输出、评估。

## 跑一个 Agent 证据检查

在称为作品集项目之前先跑这个检查。

```python
project = {
    "goal_defined": True,
    "trace_saved": True,
    "tool_logs": True,
    "failure_case": True,
    "eval_tasks": 10,
}

ready = (
    project["goal_defined"]
    and project["trace_saved"]
    and project["tool_logs"]
    and project["failure_case"]
    and project["eval_tasks"] >= 5
)

print("portfolio_ready:", ready)
print("evidence:", "goal, trace, tools, failure, eval")
```

预期输出：

```text
portfolio_ready: True
evidence: goal, trace, tools, failure, eval
```

如果这里输出 `False`，先补证据，再增加更多 Agent 角色。

## 按这个顺序学

| 步骤 | 项目 | 真正训练的能力 |
|---|---|---|
| 1 | 研究助手 | 检索、引用、总结、可信输出 |
| 2 | 数据分析 Agent | Python 工具调用、表格分析、图表、解释 |
| 3 | 多 Agent 开发团队 | 角色分工、交接、评审闭环、合并负责人 |
| 4 | 实操工作坊 | 最小可追踪单 Agent 基线 |

扩展项目前，先运行 [9.10.5 实操：构建可追踪单 Agent 助手](./04-stage-hands-on-workshop.md)。

## 项目交付物标准

| 交付物 | 最低要求 | 更强的作品集版本 |
|---|---|---|
| README | 目标、运行命令、依赖、示例 | 增加架构、取舍、成本、安全和复盘 |
| 架构 | 模型、工具、记忆、状态、评估、安全 | 增加部署边界和人工交接 |
| 工具清单 | 可调用工具、输入/输出 schema、失败情况 | 增加权限规则和沙箱说明 |
| 执行 trace | 计划、行动、观察、重规划、停止 | 增加可重放 JSONL 日志 |
| 失败案例 | 至少 1 个真实失败 | 增加 3 个案例，包含原因、修复、回归检查 |
| 评估集 | 固定任务和通过/失败规则 | 增加基线、指标和对比实验 |
| 部署说明 | 如何本地运行 | 增加 API 入口、环境变量、监控和回滚 |

## 通过标准

如果另一个开发者能重放你的 Agent 运行，检查每次工具调用和观察，理解它为什么停止，并看到至少一个失败分析，就通过了本章。

基础版可以是单 Agent 项目。只有当 trace 和评估闭环稳定后，再加入记忆、MCP、多 Agent 协作或部署。

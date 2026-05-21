---
title: "9.8.1 评估与安全路线图：评分、防护、追踪"
sidebar_position: 0
description: "Agent 评估与安全的简短实操路线：评估结果和过程，加入 guardrails，记录 trace，并复盘风险。"
keywords: [Agent 评估指南, Agent 安全指南, Guardrails, 可观测性, Agent 风险]
---

# 9.8.1 评估与安全路线图：评分、防护、追踪

Agent 不能只是能跑。你还必须知道它是否成功、过程是否安全、失败发生在哪里。

## 先看防护栈

![Agent guardrails 分层图](/img/course/agent-guardrails-layers.webp)

![Agent 评估与安全章节学习流程图](/img/course/ch09-eval-safety-chapter-flow.webp)

![Agent 风险调试闭环图](/img/course/ch09-agent-risk-debug-loop.webp)

评估告诉你系统是否有效，安全告诉你系统允许做什么，可观测性告诉你哪里出了问题。

## 跑一个上线评分卡检查

同时评估最终输出和执行过程。

```python
run = {
    "task_success": True,
    "tool_error": False,
    "permission_confirmed": True,
    "trace_saved": True,
    "cost_usd": 0.08,
}

launch_ok = (
    run["task_success"]
    and not run["tool_error"]
    and run["permission_confirmed"]
    and run["trace_saved"]
    and run["cost_usd"] < 0.10
)

print("launch_ok:", launch_ok)
print("scorecard:", "task, tools, safety, trace, cost")
```

预期输出：

```text
launch_ok: True
scorecard: task, tools, safety, trace, cost
```

一个流畅的最终回答不是足够证据。要保留可重放任务和过程 trace。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 评估方法 | 区分结果评估和过程评估 |
| 2 | 基准测试 | 把公开基准测试当参考，而不是产品替代品 |
| 3 | 安全与对齐 | 识别 prompt injection、越权、泄漏、幻觉 |
| 4 | Guardrails | 加入输入过滤、输出校验、权限和人工确认 |
| 5 | 可观测性 | 保存日志、追踪、错误、延迟、成本和失败原因 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
eval_cases: fixed tasks and expected safe behavior
scorecard: task success, tool correctness, trace quality, safety
guardrail: policy, permission, validation, or human confirmation
failure_check: unsafe tool use, prompt injection, hidden state, or unobserved action
next_action: add case, guardrail, log, rollback, or refusal path
```

## 通过标准

如果每次 Agent 运行都能通过目标、计划、工具调用、观察、最终回答、安全规则、成本和失败原因进行复盘，就通过了本章。

本章出口小项目是 10 到 20 个任务的评估集，以及至少 3 条安全规则。

<details>
<summary>参考答案与讲解</summary>

1. 合格答案要描述 agent 循环：目标、计划、工具调用、观察结果、记忆或状态更新，以及停止条件。
2. 证据应包含另一个开发者可以检查的 trace，而不只是最终回答。
3. 自检时要能说出一个安全或可靠性控制，例如工具 schema、权限边界、重试、评估用例或人工复核点。

</details>

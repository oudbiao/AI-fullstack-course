---
title: "9.2.1 推理路线图：计划、行动、检查"
sidebar_position: 0
description: "Agent 推理与规划的简短实操路线：建立中间步骤，选择行动，监控进度，并评估失败。"
keywords: [Agent 推理指南, ReAct, Plan-and-Execute, 规划]
---

# 9.2.1 推理路线图：计划、行动、检查

Agent 推理不是更长的回答，而是生成可用中间步骤、决定下一步做什么，并检查计划是否还有效。

## 先看规划闭环

![Agent 推理与规划章节学习顺序图](/img/course/ch09-reasoning-chapter-flow.webp)

![计划执行监控重规划图](/img/course/ch09-plan-execute-monitor-replan-map.webp)

![推理状态检查点图](/img/course/ch09-reasoning-state-checkpoint-map.webp)

核心习惯是：规划一步、行动、观察结果、记录状态检查点，并在情况变化时重新规划。

## 跑一个计划检查表

加工具之前先显式写出步骤。不能打印出来的计划，很难检查。

```python
task = "prepare a cited RAG demo answer"
plan = ["inspect question", "retrieve sources", "draft answer", "check citations"]

print("task:", task)
for index, step in enumerate(plan, start=1):
    print(f"{index}. {step}")
print("checkpoint:", plan[-1])
```

预期输出：

```text
task: prepare a cited RAG demo answer
1. inspect question
2. retrieve sources
3. draft answer
4. check citations
checkpoint: check citations
```

好的规划应该可见，并让失败更容易定位，而不是把问题藏在最后一段话里。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | LLM 推理能力 | 区分知道答案和推导路径 |
| 2 | 链式推理 | 建立中间状态和自检点 |
| 3 | ReAct | 交替进行思考、行动、观察和下一步 |
| 4 | Plan-and-Execute | 任务变大时分离规划和执行 |
| 5 | 高级规划 | 处理依赖、优先级、回滚和重规划 |
| 6 | 推理评估 | 评估最终结果、路径质量和失败类型 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
任务目标：Agent 想要解决什么
计划或轨迹：推理步骤、计划、ReAct 轨迹或执行图
观察：每次操作后发生了什么变化
失败检查：虚构步骤、过时观察、循环或未经验证的结论
评估动作：与期望结果对比并修正计划
```

## 通过标准

如果你能说明一个计划为什么失败：拆解差、工具选错、观察过期、缺少检查点或最终验证太弱，就通过了本章。

本章出口小项目是一个可见推理 trace：包含计划步骤、观察、重规划和最终回答。

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要描述 agent 循环：目标、计划、工具调用、观察结果、记忆或状态更新，以及停止条件。
2. 证据应包含另一个开发者可以检查的 trace，而不只是最终回答。
3. 自检时要能说出一个安全或可靠性控制，例如工具 schema、权限边界、重试、评估用例或人工复核点。

</details>

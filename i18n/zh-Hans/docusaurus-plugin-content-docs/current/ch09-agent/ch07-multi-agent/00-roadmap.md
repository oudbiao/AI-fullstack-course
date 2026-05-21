---
title: "9.7.1 多 Agent 路线图：角色、消息、负责人"
sidebar_position: 0
description: "多 Agent 系统的简短实操路线：只在有价值时拆分角色，定义消息合约，控制协作成本，并保留最终负责人。"
keywords: [Multi-Agent 指南, 协作系统, Agent 通信, Agent 协调, 多智能体]
---

# 9.7.1 多 Agent 路线图：角色、消息、负责人

多 Agent 是分工协作机制，不是几个聊天机器人互相聊天。只有当角色分离、并行处理、交叉检查或专家协作的收益超过协调成本时才值得使用。

## 先看协作成本

![多 Agent 协作消息流图](/img/course/multi-agent-message-flow.webp)

![多 Agent 章节学习顺序图](/img/course/ch09-multi-agent-chapter-flow.webp)

![多 Agent 协作与协调图](/img/course/ch09-multi-agent-coordination-map.webp)

关键问题是：拆分工作的收益，是否超过消息、重复上下文、冲突和最终合并的成本？

## 跑一个角色边界检查

每个角色都要有一个职责和一个产出。最终决策必须有一个负责人。

```python
agents = {
    "researcher": "collect evidence",
    "editor": "rewrite content",
    "reviewer": "check beginner clarity",
}

final_owner = "reviewer"

print("agent_count:", len(agents))
for name, job in agents.items():
    print(f"{name}: {job}")
print("final_owner:", final_owner)
```

预期输出：

```text
agent_count: 3
researcher: collect evidence
editor: rewrite content
reviewer: check beginner clarity
final_owner: reviewer
```

如果两个角色产出相同，就合并它们。如果没有最终负责人，系统会漂移。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 何时使用多 Agent | 写出什么时候单 Agent 更好 |
| 2 | 常见模式 | 比较主管-执行者、流水线、辩论、专家委员会 |
| 3 | 通信 | 定义消息格式、共享状态和交接规则 |
| 4 | 协调 | 追踪负责人、队列、冲突规则和聚合方式 |
| 5 | 实战与风险 | 衡量成本、循环、重复工作和角色越权 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
roles: owner, worker, reviewer, or specialist responsibilities
message_contract: artifact, request, response, and handoff state
coordination: routing, task split, conflict resolution, and final owner
failure_check: duplicated work, lost context, no accountable owner, or message loop
eval_action: compare multi-agent result against single-agent baseline
```

## 通过标准

如果一个 2 到 3 个 Agent 的演示有可追踪输入、输出、交接、最终负责人，并能说明为什么它优于单 Agent，就通过了本章。

<details>
<summary>参考答案与讲解</summary>

1. 合格答案要描述 agent 循环：目标、计划、工具调用、观察结果、记忆或状态更新，以及停止条件。
2. 证据应包含另一个开发者可以检查的 trace，而不只是最终回答。
3. 自检时要能说出一个安全或可靠性控制，例如工具 schema、权限边界、重试、评估用例或人工复核点。

</details>

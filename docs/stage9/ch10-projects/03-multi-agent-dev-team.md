---
title: "10.3 项目：多 Agent 开发团队【选修】"
sidebar_position: 56
description: "围绕 planner、coder、reviewer 和 tester 四类角色，建立一个多 Agent 开发团队项目的最小闭环。"
keywords: [multi-agent dev team, coder agent, reviewer agent, tester agent, project]
---

# 项目：多 Agent 开发团队【选修】

:::tip 本节定位
多 Agent 开发团队是很典型的高级项目题材，  
因为它天然能展示：

- 分工
- 协作
- 状态传递
- 评审闭环

但它也特别容易做成“看起来复杂、实际上空转”的 demo。  
所以这节重点是把最小闭环先定清楚。
:::

## 学习目标

- 学会定义一个多 Agent 开发团队项目的最小角色分工
- 理解多角色协作时最关键的交接信息是什么
- 建立一个可展示的项目骨架
- 理解为什么多 Agent 项目特别依赖边界和协议

---

## 一、最小角色集通常够用

建议先只做：

- planner
- coder
- reviewer
- tester

这已经足够展示：

- 任务拆解
- 实现
- 复审
- 验证

---

## 二、项目骨架示例

```python
from dataclasses import dataclass, field


@dataclass
class MultiAgentProject:
    name: str
    roles: list
    artifacts: list
    metrics: list
    risks: list = field(default_factory=list)


project = MultiAgentProject(
    name="multi_agent_dev_team",
    roles=["planner", "coder", "reviewer", "tester"],
    artifacts=["task_plan", "patch", "review_notes", "test_report"],
    metrics=["task_completion", "review_quality", "test_pass_rate"],
    risks=["角色边界模糊", "重复劳动", "交接信息缺失"],
)

print(project)
```

---

## 三、为什么多 Agent 项目特别依赖协议？

因为一旦交接信息不清楚，就会出现：

- planner 说不明白任务
- coder 改错方向
- reviewer 不知道该查什么
- tester 覆盖不到关键路径

所以多 Agent 项目最重要的往往不是角色数量，而是：

- 边界
- 输入输出格式
- 交接协议

---

## 四、真实项目里还应该补什么？

### 4.1 角色输入输出协议

例如：

- planner 输出任务清单
- coder 输出 patch
- reviewer 输出 review note
- tester 输出 test report

如果这几类工件不清楚，系统很容易空转。

### 4.2 状态看板

多 Agent 系统特别适合展示一个共享状态面板，例如：

- 当前任务阶段
- 谁在工作
- 哪一步失败

### 4.3 失败恢复策略

例如：

- reviewer 打回后 coder 是否重做
- tester 失败后是否回到 planner

### 4.4 项目展示建议

最有说服力的通常不是角色数量，  
而是：

- 一条完整任务 trace
- 明确的交接工件
- 清楚的失败回退逻辑

---

## 五、小结

这节最重要的是建立一个项目判断：

> **多 Agent 开发团队项目最有价值的地方，不是“角色越多越炫”，而是能否把分工、交接和验证做成稳定闭环。**

---

## 练习

1. 给这个项目再加一个 `ops_agent`，思考它应该负责什么。
2. 为什么多 Agent 项目里“角色边界模糊”会特别危险？
3. 想一想：reviewer 和 tester 的区别为什么必须明确？
4. 你会如何把这个项目做成一个清楚的演示页面？

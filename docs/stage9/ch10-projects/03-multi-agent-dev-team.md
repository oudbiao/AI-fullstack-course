---
title: "10.4 项目：多 Agent 开发团队【选修】"
sidebar_position: 56
description: "围绕 planner、coder、reviewer 和 tester 四类角色，建立一个多 Agent 开发团队项目的作品级最小闭环。"
keywords: [multi-agent dev team, planner, coder, reviewer, tester, project]
---

# 项目：多 Agent 开发团队【选修】

:::tip 本节定位
多 Agent 开发团队项目很容易流于表演：

- 角色很多
- 对话很多
- 但结果并不稳定

所以真正有价值的，不是角色数量，而是：

> **任务有没有被稳定拆分、交接是否清楚、失败后能不能回退。**

这节课会把一个“作品级最小闭环”讲出来。
:::

## 学习目标

- 学会定义一个多 Agent 开发团队的最小角色集
- 理解角色之间最关键的交接工件是什么
- 建立一个可展示、可验证的多 Agent 项目骨架
- 理解为什么协议和状态比“多说几轮话”更重要

---

## 一、最小角色集为什么通常就够了？

一个很稳的最小闭环通常只需要：

- planner
- coder
- reviewer
- tester

这四类角色已经足够展示：

- 拆任务
- 实现
- 复审
- 验证

如果一开始就把角色堆到很多，  
系统很容易看起来很忙，但其实在空转。

---

## 二、先跑一个角色工件交接示例

这个例子不会真的改代码，  
但它会把最关键的“交接工件”结构跑出来。

```python
from dataclasses import dataclass


@dataclass
class TaskPlan:
    goal: str
    files_to_change: list
    acceptance_test: str


@dataclass
class Patch:
    summary: str
    changed_files: list


@dataclass
class ReviewNote:
    approved: bool
    issues: list


@dataclass
class TestReport:
    passed: bool
    cases: list


plan = TaskPlan(
    goal="修复退款页面金额显示错误",
    files_to_change=["refund.py", "test_refund.py"],
    acceptance_test="输入 100 元和 8 折，结果应为 80 元",
)

patch = Patch(
    summary="修复折扣计算逻辑，并补充测试",
    changed_files=["refund.py", "test_refund.py"],
)

review = ReviewNote(
    approved=False,
    issues=["变量命名不清晰", "边界条件测试不完整"],
)

test_report = TestReport(
    passed=False,
    cases=["test_discount_basic", "test_discount_zero"],
)

print(plan)
print(patch)
print(review)
print(test_report)
```

### 2.1 这个例子最关键的地方是什么？

它说明多 Agent 项目真正应该展示的，不是纯聊天记录，  
而是：

- 交接工件
- 任务状态
- 结果验证

### 2.2 为什么工件比对话更重要？

因为工件才是后续角色真正依赖的输入。  
如果只看对话，很难判断系统是不是能稳定协作。

---

## 三、一个最小工作流闭环

下面把四个角色串成一条最小流程：

```python
def planner(goal):
    return TaskPlan(
        goal=goal,
        files_to_change=["refund.py", "test_refund.py"],
        acceptance_test="输入 100 元和 8 折，结果应为 80 元",
    )


def coder(plan):
    return Patch(
        summary=f"根据任务目标实现: {plan.goal}",
        changed_files=plan.files_to_change,
    )


def reviewer(patch):
    if "test_refund.py" not in patch.changed_files:
        return ReviewNote(approved=False, issues=["缺少测试文件改动"])
    return ReviewNote(approved=True, issues=[])


def tester(review_note):
    if not review_note.approved:
        return TestReport(passed=False, cases=["review_failed"])
    return TestReport(passed=True, cases=["test_discount_basic", "test_discount_zero"])


goal = "修复退款页面金额显示错误"
plan = planner(goal)
patch = coder(plan)
review = reviewer(patch)
test_report = tester(review)

print(plan)
print(patch)
print(review)
print(test_report)
```

### 3.1 为什么这个闭环已经很像真实项目？

因为它体现了多 Agent 项目最关键的 3 个点：

1. 角色分工
2. 明确工件交接
3. 基于评审与测试的回路

### 3.2 如果 reviewer 不通过，为什么 tester 就不该继续？

这说明多 Agent 系统不是“人人都并行做自己的”，  
而是要尊重：

- 阶段依赖
- 交接质量

---

## 四、作品级项目最该展示什么？

### 4.1 一条完整任务 trace

例如：

- 任务目标
- plan
- patch
- review issues
- test report

### 4.2 一次失败回退

这会非常有说服力。  
例如：

- reviewer 打回
- coder 二次修复
- tester 重新验证

### 4.3 清楚的角色边界

作品集里要能回答：

- 为什么要这 4 个角色
- 每个角色输入和输出是什么

---

## 五、最容易踩的坑

### 5.1 角色很多但边界不清

这会让系统看起来复杂，  
实际上只是重复劳动。

### 5.2 没有共享状态或统一工件格式

这样角色之间很难稳定交接。

### 5.3 只展示成功路径

一个好的多 Agent 项目更该展示：

- 失败后如何回退
- 哪一步最容易出问题

---

## 小结

这节最重要的是建立一个作品级判断：

> **多 Agent 开发团队项目真正有价值的地方，不是角色越多越炫，而是能否把任务拆解、工件交接和失败回退组织成稳定闭环。**

只要这条闭环立住，这个项目会非常适合展示你对多 Agent 系统的真正理解。

---

## 练习

1. 给工作流加一个 `ops_agent`，思考它应该接在哪一步。
2. 想一想：为什么多 Agent 项目里“统一工件格式”比“角色会聊天”更重要？
3. 如果 reviewer 经常打回 patch，应该优先优化哪一层？
4. 如果把这个项目做成 demo 页面，你最想展示哪一条完整 trace？

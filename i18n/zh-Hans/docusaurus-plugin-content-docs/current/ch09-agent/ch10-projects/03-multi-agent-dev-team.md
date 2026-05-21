---
title: "9.10.4 项目：多 Agent 开发团队【选修】"
sidebar_position: 56
description: "围绕 planner、coder、reviewer 和 tester 四类角色，建立一个多 Agent 开发团队项目的作品级最小闭环。"
keywords: [multi-agent dev team, planner, coder, reviewer, tester, project]
---

# 9.10.4 项目：多 Agent 开发团队【选修】

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

## 最小角色集为什么通常就够了？

一个很稳的最小闭环通常只需要：

- 规划者
- 开发者（编码者）
- 审核者
- 测试者

这四类角色已经足够展示：

- 拆任务
- 实现
- 复审
- 验证

如果一开始就把角色堆到很多，
系统很容易看起来很忙，但其实在空转。

---

## 先跑一个角色工件交接示例

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
    goal="修复退款页面状态标签不一致问题",
    files_to_change=["status.py", "test_status.py"],
    acceptance_test="输入 '  OPEN ' 时，标准化结果应为 'open'",
)

patch = Patch(
    summary="修复状态标准化逻辑，并补充测试",
    changed_files=["status.py", "test_status.py"],
)

review = ReviewNote(
    approved=False,
    issues=["变量命名不清晰", "边界条件测试不完整"],
)

test_report = TestReport(
    passed=False,
    cases=["test_status_normalize_basic", "test_status_normalize_empty"],
)

print(plan)
print(patch)
print(review)
print(test_report)
```

预期输出：

```text
TaskPlan(goal='修复退款页面状态标签不一致问题', files_to_change=['status.py', 'test_status.py'], acceptance_test="输入 '  OPEN ' 时，标准化结果应为 'open'")
Patch(summary='修复状态标准化逻辑，并补充测试', changed_files=['status.py', 'test_status.py'])
ReviewNote(approved=False, issues=['变量命名不清晰', '边界条件测试不完整'])
TestReport(passed=False, cases=['test_status_normalize_basic', 'test_status_normalize_empty'])
```

![多 Agent 工件交接结果图](/img/course/ch09-multi-agent-artifact-handoff-anatomy-result-map.webp)

### 这个例子最关键的地方是什么？

它说明多 Agent 项目真正应该展示的，不是纯聊天记录，
而是：

- 交接工件
- 任务状态
- 结果验证

### 为什么工件比对话更重要？

因为工件才是后续角色真正依赖的输入。
如果只看对话，很难判断系统是不是能稳定协作。

---

## 一个最小工作流闭环

继续在同一个文件或 Python 会话里运行，因为下面这段会复用上一个示例里的 dataclass。

下面把四个角色串成一条最小流程：

```python
def planner(goal):
    return TaskPlan(
        goal=goal,
        files_to_change=["status.py", "test_status.py"],
        acceptance_test="输入 '  OPEN ' 时，标准化结果应为 'open'",
    )


def coder(plan):
    return Patch(
        summary=f"根据任务目标实现: {plan.goal}",
        changed_files=plan.files_to_change,
    )


def reviewer(patch):
    if "test_status.py" not in patch.changed_files:
        return ReviewNote(approved=False, issues=["缺少测试文件改动"])
    return ReviewNote(approved=True, issues=[])


def tester(review_note):
    if not review_note.approved:
        return TestReport(passed=False, cases=["review_failed"])
    return TestReport(passed=True, cases=["test_status_normalize_basic", "test_status_normalize_empty"])


goal = "修复退款页面状态标签不一致问题"
plan = planner(goal)
patch = coder(plan)
review = reviewer(patch)
test_report = tester(review)

print(plan)
print(patch)
print(review)
print(test_report)
```

预期输出：

```text
TaskPlan(goal='修复退款页面状态标签不一致问题', files_to_change=['status.py', 'test_status.py'], acceptance_test="输入 '  OPEN ' 时，标准化结果应为 'open'")
Patch(summary='根据任务目标实现: 修复退款页面状态标签不一致问题', changed_files=['status.py', 'test_status.py'])
ReviewNote(approved=True, issues=[])
TestReport(passed=True, cases=['test_status_normalize_basic', 'test_status_normalize_empty'])
```

![多 Agent 开发团队工件 追踪 结果图](/img/course/ch09-multi-agent-dev-team-artifact-trace-result-map.webp)

:::tip 读结果
把输出当成一条工件链来读：`TaskPlan` 定义目标和验收规则，`Patch` 同时改实现和测试文件，`ReviewNote` 是关卡，`TestReport` 是最后证据。
:::

### 为什么这个闭环已经很像真实项目？

因为它体现了多 Agent 项目最关键的 3 个点：

1. 角色分工
2. 明确工件交接
3. 基于评审与测试的回路

### 如果审核者不通过，为什么测试者就不该继续？

这说明多 Agent 系统不是“人人都并行做自己的”，
而是要尊重：

- 阶段依赖
- 交接质量

![多 Agent 开发团队交付闭环图](/img/course/ch09-multi-agent-dev-team-delivery-map.webp)

:::tip 读图提示
这张图强调“角色数量不是重点，工件交接才是重点”：规划者产出计划，编码者产出补丁，审核者产出问题清单，测试者产出测试报告，失败后回到对应角色修复。
:::

---

## 作品级项目最该展示什么？

### 一条完整任务追踪

例如：

- 任务目标
- 计划（plan）
- 补丁（patch）
- 审核问题
- 测试报告

### 一次失败回退

这会非常有说服力。
例如：

- 审核者打回
- 编码者 二次修复
- 测试者重新验证

### 清楚的角色边界

作品集里要能回答：

- 为什么要这 4 个角色
- 每个角色输入和输出是什么

---

## 最容易踩的坑

### 角色很多但边界不清

这会让系统看起来复杂，
实际上只是重复劳动。

### 没有共享状态或统一工件格式

这样角色之间很难稳定交接。

### 只展示成功路径

一个好的多 Agent 项目更该展示：

- 失败后如何回退
- 哪一步最容易出问题

---

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
项目目标：智能体应完成什么，以及必须不做什么
基线：在加入高级功能前的单智能体循环
追踪包：目标、计划、tool 调用、观察、记忆、评估
失败日志：一次失败或不安全的运行及其根因
交付物：README、运行命令、trace 截图/日志、下一步
```

## 小结

这节最重要的是建立一个作品级判断：

> **多 Agent 开发团队项目真正有价值的地方，不是角色越多越炫，而是能否把任务拆解、工件交接和失败回退组织成稳定闭环。**

只要这条闭环立住，这个项目会非常适合展示你对多 Agent 系统的真正理解。

---



## 版本路线建议

| 版本 | 目标 | 交付重点 |
|---|---|---|
| 基础版 | 跑通最小闭环 | 能输入、能处理、能输出，并保留一组示例 |
| 标准版 | 形成可展示项目 | 增加配置、日志、错误处理、README 和截图 |
| 挑战版 | 接近作品集质量 | 增加评估、对比实验、失败样本分析和下一步路线 |

建议先完成基础版，不要一开始就追求大而全。每提升一个版本，都要把“新增了什么能力、怎么验证、还有什么问题”写进 README。

## 练习

1. 给工作流加一个 `ops_agent`，思考它应该接在哪一步。
2. 想一想：为什么多 Agent 项目里“统一工件格式”比“角色会聊天”更重要？
3. 如果审核者经常打回 patch，应该优先优化哪一层？
4. 如果把这个项目做成演示页面，你最想展示哪一条完整 追踪？

<details>
<summary>项目交付参考与讲解</summary>

1. 可以把 `ops_agent` 放在 implementation 之后、final release review 之前。它负责检查运行命令、环境变量、日志、rollback notes 和部署风险。
2. unified artifact format 重要，是因为 Agent 协作需要稳定输入输出。只靠聊天很难测试、复盘、diff，也难交接给另一个 Agent。
3. 如果 reviewer 经常拒绝 patch，先优化 task specification 和 acceptance criteria，再检查 coder context、test feedback，以及 review comments 是否可执行。
4. 好的演示轨迹应展示 需求 -> 计划 -> 补丁 -> 测试结果 -> 审核拒绝或通过 -> 修订 -> 最终产物。这个轨迹会让协作结构变得可见。

</details>

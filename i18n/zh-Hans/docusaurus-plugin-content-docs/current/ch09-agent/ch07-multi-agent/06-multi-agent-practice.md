---
title: "9.7.7 实战：多 Agent 协作系统"
sidebar_position: 43
description: "从任务输入、角色划分、状态流转到结果汇总，完整搭一个最小多 Agent 协作系统。"
keywords: [multi-agent project, planner, retriever, writer, reviewer, workflow, collaboration]
---

# 9.7.7 实战：多 Agent 协作系统

![多 Agent 协作实操运行图](/img/course/ch09-multi-agent-collaboration-run-map.webp)

:::tip 本节定位
这一节是本章的收口项目。
前面你已经学过：

- 架构模式
- 通信
- 任务分配
- 协作模式
- 挑战与解决

现在要做的就是把这些真正拼起来，形成一个最小但完整的多 Agent 系统。
:::

## 学习目标

- 搭建一个最小多 Agent 协作闭环
- 学会让规划者、检索者、撰写者、审核者各司其职
- 看懂任务状态如何在多个角色之间流转
- 理解这个项目和单 Agent 系统相比真正多了什么

---

## 先定义项目目标

我们做一个最小研究型多 Agent 系统：

用户输入：

> “请帮我总结退款政策的关键条件。”

系统内部角色：

- 规划者：拆任务
- 检索者：找资料
- 撰写者：写总结
- 审核者：检查结果

这个任务之所以合适，是因为它天然能拆工，而且每个角色职责很清楚。

---

## 先准备资料库

```python
knowledge_base = {
    "退款政策": "课程购买后 7 天内且学习进度低于 20% 可申请退款。",
    "证书政策": "完成所有必修项目并通过测试后可获得结业证书。",
    "学习顺序": "建议先学 Python、数据分析、机器学习，再进入深度学习与大模型阶段。"
}

print(knowledge_base)
```

预期输出：

```text
{'退款政策': '课程购买后 7 天内且学习进度低于 20% 可申请退款。', '证书政策': '完成所有必修项目并通过测试后可获得结业证书。', '学习顺序': '建议先学 Python、数据分析、机器学习，再进入深度学习与大模型阶段。'}
```

这就是系统要操作的最小知识来源。

---

## 定义四个 Agent

### 规划者

```python
def planner_agent(user_query):
    if "退款" in user_query:
        return ["检索退款政策", "整理关键条件", "撰写总结", "审核输出"]
    return ["检索相关资料", "撰写总结", "审核输出"]
```

### 检索者

```python
def retriever_agent(task):
    if "退款政策" in task:
        return knowledge_base["退款政策"]
    return "未找到资料"
```

### 撰写者

```python
def writer_agent(evidence):
    return f"总结：{evidence}"
```

### 审核者

```python
def reviewer_agent(draft):
    if "7 天内" in draft and "20%" in draft:
        return {"approved": True, "comment": "关键信息完整"}
    return {"approved": False, "comment": "缺少关键条件"}
```

---

## 把它们串起来

### 一个最小多 Agent 协作流程

请接着上面的知识库和四个 Agent 函数，在同一个 Python 文件或同一个解释器会话里运行。

```python
def multi_agent_system(user_query):
    state = {
        "query": user_query,
        "plan": [],
        "evidence": None,
        "draft": None,
        "review": None
    }

    # 1. 规划
    state["plan"] = planner_agent(user_query)

    # 2. 检索
    state["evidence"] = retriever_agent(state["plan"][0])

    # 3. 写作
    state["draft"] = writer_agent(state["evidence"])

    # 4. 审核
    state["review"] = reviewer_agent(state["draft"])

    return state

result = multi_agent_system("请帮我总结退款政策的关键条件。")
for k, v in result.items():
    print(k, "->", v)
```

预期输出：

```text
query -> 请帮我总结退款政策的关键条件。
plan -> ['检索退款政策', '整理关键条件', '撰写总结', '审核输出']
evidence -> 课程购买后 7 天内且学习进度低于 20% 可申请退款。
draft -> 总结：课程购买后 7 天内且学习进度低于 20% 可申请退款。
review -> {'approved': True, 'comment': '关键信息完整'}
```

![多 Agent 状态交接结果图](/img/course/ch09-multi-agent-state-handoff-result-map.webp)

### 这段代码已经说明了什么？

它已经说明：

- 多 Agent 不是简单多个函数
- 关键在状态流转
- 每个角色只负责自己那一段

这就是一个真正的最小多 Agent 系统。

---

## 让系统更像真实工作流

### 如果审核者不通过怎么办？

真实系统里，review 不通过后，通常不会直接结束。
更合理的做法是：

- 把评审意见回传给撰写者
- 再修一版

### 一个带修订的小例子

继续在同一个文件或会话中运行，确保 `multi_agent_system` 和前面的 Agent 函数已经定义。

```python
def reviser_agent(draft, review):
    if review["approved"]:
        return draft
    return draft + " 补充说明：退款还要求学习进度低于 20%。"

state = multi_agent_system("请帮我总结退款政策的关键条件。")
final_output = reviser_agent(state["draft"], state["review"])

print("draft :", state["draft"])
print("review:", state["review"])
print("final :", final_output)
```

预期输出：

```text
draft : 总结：课程购买后 7 天内且学习进度低于 20% 可申请退款。
review: {'approved': True, 'comment': '关键信息完整'}
final : 总结：课程购买后 7 天内且学习进度低于 20% 可申请退款。
```

这一步很重要，因为它体现了：

> 多 Agent 系统的价值，不只是分工，还在于角色之间能形成迭代闭环。

---

## 加入更明确的任务日志

### 为什么项目里一定要有 追踪？

如果系统答错了，你至少得知道：

- 规划者怎么拆的
- 检索者找到了什么
- 撰写者写了什么
- 审核者为什么没拦住

### 一个最小 追踪 版本

继续在同一个文件或会话中运行，确保四个 Agent 函数已经定义。

```python
def traced_multi_agent_system(user_query):
    trace = []

    plan = planner_agent(user_query)
    trace.append({"agent": "planner", "output": plan})

    evidence = retriever_agent(plan[0])
    trace.append({"agent": "retriever", "output": evidence})

    draft = writer_agent(evidence)
    trace.append({"agent": "writer", "output": draft})

    review = reviewer_agent(draft)
    trace.append({"agent": "reviewer", "output": review})

    return trace

for step in traced_multi_agent_system("请帮我总结退款政策的关键条件。"):
    print(step)
```

预期输出：

```text
{'agent': 'planner', 'output': ['检索退款政策', '整理关键条件', '撰写总结', '审核输出']}
{'agent': 'retriever', 'output': '课程购买后 7 天内且学习进度低于 20% 可申请退款。'}
{'agent': 'writer', 'output': '总结：课程购买后 7 天内且学习进度低于 20% 可申请退款。'}
{'agent': 'reviewer', 'output': {'approved': True, 'comment': '关键信息完整'}}
```

这个 trace 就是后面你调试和评估系统的重要基础。

---

## 为什么这个系统比单 Agent 更值得学？

### 因为它把问题拆开了

单 Agent 往往是一口气：

- 理解任务
- 检索
- 总结
- 自我检查

而多 Agent 把这些动作拆开后，你更容易：

- 观察每一层
- 替换其中一层
- 找到哪一层出错

### 但它也更贵、更复杂

所以真正的工程判断不是：

> 多 Agent 一定更高级

而是：

> 这个任务值不值得为“更可拆、可控”付出额外复杂度。

---

## 这个项目怎样继续升级？

你可以继续往上加：

1. 更真实的检索器
2. 多任务路由
3. 异步通信
4. 冲突裁决机制
5. 失败重试

如果再继续做大，它就会越来越接近真实的多 Agent 产品系统。

---

## 初学者最常踩的坑

### 把所有角色都写得差不多

这样最后只是“多个名字不同的同一种 Agent”。

### 没有共享状态或 追踪

一旦出错就很难查。

### 项目看起来热闹，但每个角色并没有真正分工

这是很多多 Agent 演示最常见的问题。

---

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
roles: owner, worker, reviewer, or specialist responsibilities
message_contract: artifact, request, response, and handoff state
coordination: routing, task split, conflict resolution, and final owner
failure_check: duplicated work, lost context, no accountable owner, or message loop
eval_action: compare multi-agent result against single-agent baseline
```

## 小结

这一节最重要的不是写出四个函数，而是理解：

> **多 Agent 项目的核心，是让每个角色围绕状态流转承担不同责任，并最终收敛成一个可解释、可迭代的工作流。**

这才是多 Agent 真正比单 Agent 更有价值的地方。

---

## 练习

1. 给这个系统再加一个 `fact_checker_agent`，专门核查数字条件。
2. 让 `planner_agent` 针对“证书政策”也能产出不同计划。
3. 想一想：如果审核者一直不通过，系统应该怎样限制修订轮数？
4. 用自己的话解释：为什么说多 Agent 项目真正重要的是“状态流转”，而不是“角色数量”？

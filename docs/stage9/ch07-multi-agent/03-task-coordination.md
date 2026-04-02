---
title: "7.3 任务分配与协调"
sidebar_position: 40
description: "从拆任务、分配角色、同步状态到冲突解决，理解多 Agent 系统怎样真正把活分出去并收回来。"
keywords: [task coordination, task assignment, multi-agent, scheduling, conflict resolution]
---

# 任务分配与协调

:::tip 本节定位
上一节讲了通信，说明“信息怎样传”。  
这一节要解决的是另一个更棘手的问题：

> **任务到底该怎么拆、怎么分、怎么收？**

如果分配得不好，多 Agent 系统即使会通信，也只会低效甚至互相打架。
:::

## 学习目标

- 理解多 Agent 中任务分配为什么是核心问题
- 分清静态分配、动态分配、能力路由这几种常见方式
- 理解协调里最常见的冲突和解决方式
- 看懂一个小型任务调度示例

---

## 一、为什么多 Agent 最难的不只是“多”

### 1.1 多 Agent 最大风险：不是没人干活，而是大家都干不对

多 Agent 系统里常见失败不只是：

- 没人做

更常见的是：

- 两个人重复做
- 错的人接了任务
- 任务顺序错了
- 结果回不来

所以真正的重点是：

> **怎样让对的人，在对的时机，做对的事。**

### 1.2 一个生活类比

像做一个小项目：

- 有人负责找资料
- 有人负责写代码
- 有人负责评审

如果分配乱了，再聪明的人也会低效。

---

## 二、最常见的三种任务分配方式

### 2.1 静态分配

任务和角色提前写死。

例如：

- 检索一定给 retriever
- 写作一定给 writer

优点：

- 稳定
- 易调试

缺点：

- 灵活性差

### 2.2 动态分配

系统根据当前任务内容决定交给谁。

例如：

- 法律问题交给 legal_agent
- 技术问题交给 tech_agent

优点：

- 更灵活

缺点：

- 路由错了就会连锁出错

### 2.3 能力路由

不是按名字分，而是按能力特征分：

- 谁更适合检索？
- 谁更适合总结？
- 谁更适合审查？

这更像“按岗位能力派活”。

---

## 三、一个最小任务分配示例

```python
agents = {
    "researcher": {"skills": ["search", "retrieve"]},
    "writer": {"skills": ["write", "summarize"]},
    "reviewer": {"skills": ["review", "critique"]}
}

tasks = [
    {"name": "查资料", "skill": "search"},
    {"name": "写总结", "skill": "write"},
    {"name": "做评审", "skill": "review"}
]

def assign_task(task, agents):
    for agent_name, profile in agents.items():
        if task["skill"] in profile["skills"]:
            return agent_name
    return None

for task in tasks:
    print(task["name"], "->", assign_task(task, agents))
```

### 3.2 这段代码在教你什么？

它在教你一个非常重要的抽象：

> 任务分配不是随机发活，而是“任务需求”和“Agent 能力”之间的匹配。 

---

## 四、任务协调不只是分配，还包括顺序控制

### 4.1 有些任务不能并行

例如：

1. 先查资料
2. 再写总结
3. 最后再评审

如果顺序反了，系统就会乱。

### 4.2 一个最小顺序调度示例

```python
dependencies = {
    "retrieve": [],
    "write": ["retrieve"],
    "review": ["write"]
}

done = set()
execution_order = []

while len(done) < len(dependencies):
    for task, need in dependencies.items():
        if task not in done and all(n in done for n in need):
            done.add(task)
            execution_order.append(task)

print(execution_order)
```

输出会是：

```text
['retrieve', 'write', 'review']
```

这就是多 Agent 协调里很重要的一层：  
**不仅知道谁做，还要知道先后顺序。**

---

## 五、任务协调里最常见的冲突

### 5.1 重复劳动

两个 Agent 都去做同一件事。

### 5.2 结论冲突

一个 Agent 说“可以退款”，另一个说“不可以退款”。

### 5.3 状态不同步

writer 还以为资料没找到，但 retriever 其实已经返回了。

### 5.4 为什么这些问题很常见？

因为多 Agent 本质上就是“分布式系统的小型版”。  
只要一旦分工，就会出现：

- 同步
- 冲突
- 收敛

这些问题。

---

## 六、一个带冲突解决思路的小例子

```python
results = {
    "agent_a": {"decision": "approve", "confidence": 0.7},
    "agent_b": {"decision": "reject", "confidence": 0.9}
}

def resolve_conflict(results):
    best_agent = max(results.items(), key=lambda x: x[1]["confidence"])
    return {
        "final_decision": best_agent[1]["decision"],
        "source": best_agent[0]
    }

print(resolve_conflict(results))
```

### 6.2 为什么这只是最小版？

真实系统里，冲突解决可能会用：

- 置信度
- 投票
- reviewer 裁决
- supervisor 最终拍板

但你至少要先意识到：

> 多 Agent 一定会有冲突，冲突不是异常，而是常态。 

---

## 七、任务协调和通信有什么关系？

通信解决的是：

- 信息怎么传

协调解决的是：

- 任务怎么排
- 谁负责什么
- 出现冲突怎么收敛

所以可以记成：

- 通信更像“线路”
- 协调更像“调度”

两者缺一不可。

---

## 八、真实系统里常见的协调策略

### 8.1 中心调度型

由 supervisor 统一决定任务流转。

优点：

- 最容易管控

### 8.2 分布协商型

Agent 之间互相提议、协商。

优点：

- 灵活

缺点：

- 难调

### 8.3 半中心型

大方向由 supervisor 控制，细节由 worker 自主。

这在实际工程里往往是个比较平衡的选择。

---

## 九、初学者最常踩的坑

### 9.1 只分工，不设计收尾

任务做了一半没人负责收尾，是非常常见的问题。

### 9.2 只设计 happy path

一旦有 Agent 超时、失败、冲突，系统就乱了。

### 9.3 以为“更多 Agent = 更高效率”

如果协调做不好，更多 Agent 只会带来更多管理成本。

---

## 小结

这一节最重要的不是把任务“分出去”，而是理解：

> **任务分配与协调的核心，是让任务、角色、顺序和冲突处理形成一个能收敛的系统。**

这才是多 Agent 从“看起来热闹”走向“真正高效协作”的关键。

---

## 练习

1. 给任务分配示例再加一个 `planner` Agent，并让它决定执行顺序。
2. 设计一个“retrieve -> write -> review -> revise” 的协调流程。
3. 想一想：如果两个 Agent 结论冲突，你更倾向于投票、置信度裁决，还是 reviewer 拍板？为什么？
4. 用自己的话解释：为什么说多 Agent 协调本质上很像一个小型任务调度系统？

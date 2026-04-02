---
title: "7.2 Agent 间通信"
sidebar_position: 39
description: "从消息格式、同步与异步、共享状态到失败重试，系统理解多 Agent 之间到底怎样通信。"
keywords: [multi-agent communication, message passing, event bus, shared state, async, protocol]
---

# Agent 间通信

:::tip 本节定位
如果说上一节是在回答“这些 Agent 应该怎样分工”，那这一节就在回答：

> **分好工以后，它们到底怎样把信息传来传去？**

很多多 Agent 系统最后出问题，不是因为个体 Agent 不够聪明，而是因为通信设计太弱。
:::

## 学习目标

- 理解多 Agent 通信为什么是系统成败关键
- 分清消息传递、共享状态、事件总线三类常见通信方式
- 看懂一个最小事件总线示例
- 理解同步通信和异步通信的工程差别

---

## 一、为什么通信会成为多 Agent 系统的核心问题？

### 1.1 多 Agent 最大的风险不是“不会干活”，而是“互相没对齐”

即使每个 Agent 单独都很强，系统也可能因为通信设计差而出问题：

- 重复劳动
- 消息丢失
- 信息理解不一致
- 任务已经完成还在继续讨论

### 1.2 一个很直观的类比

多 Agent 很像一个小团队协作：

- 分工只是第一步
- 真正决定效率的，往往是开会、交接、同步、回传这些沟通机制

这就是为什么通信不是“附属模块”，而是核心结构。

---

## 二、最常见的三种通信方式

### 2.1 直接消息传递（message passing）

一个 Agent 明确给另一个 Agent 发消息。

优点：

- 简单
- 清晰
- 好追踪

缺点：

- Agent 之间耦合比较强

### 2.2 共享状态（shared state / blackboard）

所有 Agent 都往一个共享工作区写入和读取信息。

优点：

- 不需要每次显式点对点发送
- 很适合多方协同观察同一个任务状态

缺点：

- 更容易写乱
- 权限和冲突更难控

### 2.3 事件总线（event bus）

Agent 不一定直接知道彼此，而是把消息发到总线，由订阅者接收。

优点：

- 更解耦
- 更适合复杂系统

缺点：

- 调试更复杂

---

## 三、先看最简单的点对点消息传递

### 3.1 一个最小示例

```python
message = {
    "from": "planner",
    "to": "worker",
    "type": "task_assignment",
    "content": "请整理退款政策的关键条件"
}

print(message)
```

### 3.2 为什么这已经很重要？

因为它把通信的几个关键元素都显式化了：

- 谁发的
- 发给谁
- 消息类型
- 消息内容

这比“随便传一段自然语言”稳很多。

---

## 四、消息格式为什么要标准化？

### 4.1 一个坏消息格式

```python
bad_message = "帮我做这个任务"
print(bad_message)
```

问题在于：

- 不知道谁发的
- 不知道任务类型
- 不知道上下文
- 不知道下一步怎么处理

### 4.2 一个更稳的消息结构

```python
good_message = {
    "from": "planner",
    "to": "researcher",
    "type": "search_request",
    "task_id": "task_001",
    "payload": {
        "query": "退款政策"
    }
}

print(good_message)
```

这才更像一个能进入系统流水线的消息。

---

## 五、一个最小事件总线示例

### 5.1 可运行代码

```python
from collections import defaultdict

class EventBus:
    def __init__(self):
        self.handlers = defaultdict(list)

    def subscribe(self, event_type, handler):
        self.handlers[event_type].append(handler)

    def publish(self, event_type, payload):
        for handler in self.handlers[event_type]:
            handler(payload)

def planner_handler(payload):
    print("[planner] 收到结果:", payload)

def worker_handler(payload):
    print("[worker] 收到任务:", payload)
    result = {
        "task_id": payload["task_id"],
        "summary": f"已完成对 {payload['query']} 的检索"
    }
    bus.publish("task_done", result)

bus = EventBus()
bus.subscribe("task_assignment", worker_handler)
bus.subscribe("task_done", planner_handler)

bus.publish("task_assignment", {
    "task_id": "task_001",
    "query": "退款政策"
})
```

### 5.2 这段代码真正教了什么？

它教你：

- 通信不一定非要点对点耦合
- 可以通过事件类型来解耦
- 完成消息和结果消息可以走同一套基础设施

这已经非常接近真实系统的通信主线了。

---

## 六、共享状态：什么时候更适合？

### 6.1 一个很典型的场景

如果多个 Agent 都围绕同一个任务工作，比如：

- planner 写计划
- retriever 写资料
- writer 生成草稿
- reviewer 写评审意见

这时很多信息都可以放在共享工作区里。

### 6.2 一个最小示例

```python
shared_state = {
    "goal": "完成退款政策总结",
    "plan": [],
    "evidence": [],
    "draft": None,
    "review": None
}

# planner
shared_state["plan"] = ["查政策", "整理要点", "输出总结"]

# retriever
shared_state["evidence"].append("购买后 7 天内且学习进度低于 20% 可退款")

# writer
shared_state["draft"] = "退款条件包括时间限制和学习进度限制。"

print(shared_state)
```

### 6.3 这种方式的优缺点

优点：

- 大家都能看同一块黑板
- 状态更集中

缺点：

- 谁能写什么要控制
- 很容易出现覆盖冲突

---

## 七、同步通信和异步通信怎么理解？

### 7.1 同步通信

一个 Agent 发出请求后，要等对方回复，自己才能继续。

优点：

- 简单
- 容易理解

缺点：

- 容易堵塞

### 7.2 异步通信

发出消息后先继续做别的事，等对方完成后再回来处理结果。

优点：

- 更灵活
- 更适合复杂系统和高并发

缺点：

- 状态管理更复杂

### 7.3 一个很实用的工程直觉

如果你的任务链很短、流程很清楚，先同步。  
如果任务很长、等待时间不稳定，再考虑异步。

---

## 八、Agent 间通信最常见的失败点

### 8.1 消息格式不统一

今天叫 `task_id`，明天叫 `id`，后天叫 `job_id`，系统会越来越乱。

### 8.2 消息发出去了，但没人处理

这是事件系统里很常见的问题：

- 发布了
- 但没有订阅者

### 8.3 多个 Agent 理解同一条消息的方式不同

例如：

- 一个 Agent 觉得这是“检索请求”
- 另一个 Agent 觉得这是“总结请求”

这就会导致系统跑偏。

### 8.4 没有超时和重试

如果一个 Agent 卡住，整个系统可能就一直等下去。

---

## 九、真实系统里怎样让通信更稳？

### 9.1 统一消息协议

至少统一：

- `from`
- `to`
- `type`
- `task_id`
- `payload`

### 9.2 统一状态追踪

每条任务最好都有唯一 ID，便于：

- 追踪完整链路
- 回放
- 排错

### 9.3 统一超时和失败策略

例如：

- 超时自动回退
- 失败转人工
- 多次重试后终止

---

## 十、小结

这一节最重要的不是记住“消息传递、事件总线、共享状态”这些词，而是理解：

> **多 Agent 通信的关键，不只是把消息发出去，而是让消息结构稳定、责任清晰、失败可控。**

只有通信层做稳了，多 Agent 系统才不会因为“组织混乱”而把模型能力浪费掉。

---

## 练习

1. 给事件总线示例再加一个 `reviewer_handler`，让它订阅 `task_done`。
2. 设计一份你自己的统一消息协议，至少包含 `type`、`task_id` 和 `payload`。
3. 想一想：什么时候你会更倾向于共享状态，而不是点对点消息？
4. 用自己的话解释：为什么多 Agent 系统里，通信设计往往和任务分工同样重要？

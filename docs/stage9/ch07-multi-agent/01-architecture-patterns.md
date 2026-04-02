---
title: "7.1 多 Agent 架构模式"
sidebar_position: 38
description: "从监督者模式、流水线模式、群体协作模式到评审模式，理解多 Agent 系统为什么要这样分工。"
keywords: [multi-agent, supervisor, pipeline, reviewer pattern, agent architecture, collaboration]
---

# 多 Agent 架构模式

:::tip 本节定位
很多人第一次做多 Agent 时，最容易犯的错就是：

> “既然一个 Agent 不够，那就多开几个。”

但真正的问题不是“数量”，而是：

> **这些 Agent 怎样分工、怎样组织、怎样协作。**

这就是多 Agent 架构模式要解决的核心。
:::

## 学习目标

- 理解什么时候真的需要多 Agent
- 分清几种最常见的多 Agent 架构模式
- 看懂 supervisor、pipeline、reviewer 等模式的优缺点
- 用一个小型示例感受不同模式的组织方式

---

## 一、为什么不是所有任务都需要多 Agent？

### 1.1 多 Agent 不是默认升级路线

如果一个任务本来就能由单 Agent 稳定完成，多 Agent 往往只会增加：

- 通信成本
- 调试难度
- 失败路径

所以更稳妥的原则通常是：

> **先把单 Agent 做稳，再考虑是否真的需要拆成多 Agent。**

### 1.2 那什么时候值得上多 Agent？

通常是这些情况：

- 任务明显可以拆工
- 子任务类型差异很大
- 一个 Agent 同时承担所有职责太乱
- 需要独立的规划、执行、评审角色

这时多 Agent 才真正有意义。

---

## 二、先看最常见的几种模式

### 2.1 Supervisor-Worker 模式

一个监督者（supervisor）负责：

- 拆任务
- 分配任务
- 汇总结果

其他 worker 负责具体执行。

这是最常见也最好理解的模式之一。

### 2.2 Pipeline 模式

每个 Agent 只负责固定阶段：

1. 检索
2. 分析
3. 写作

它更像流水线。

### 2.3 Reviewer 模式

一个 Agent 负责生成，另一个专门负责检查或评审。

这在代码、文档、报告生成里特别常见。

### 2.4 Group / Peer 模式

多个 Agent 相对平等地协商。

这种模式更灵活，但也更难控。

---

## 三、Supervisor-Worker：最值得先学的模式

### 3.1 为什么它很常见？

因为它最符合很多现实团队结构：

- 项目经理 / 组长负责拆任务
- 执行同学负责具体工作

### 3.2 一个最小可运行示例

```python
tasks = ["检索资料", "整理要点", "撰写总结"]
workers = {
    "researcher": "负责找资料",
    "analyst": "负责整理信息",
    "writer": "负责生成最终文本"
}

assignment = {
    "检索资料": "researcher",
    "整理要点": "analyst",
    "撰写总结": "writer"
}

for task in tasks:
    worker = assignment[task]
    print(f"{worker} <- {task} ({workers[worker]})")
```

### 3.3 它的优点和缺点

优点：

- 分工清楚
- 更容易控制
- 更容易观察谁哪一步出问题

缺点：

- supervisor 可能成为瓶颈
- 如果拆任务质量差，后面全都会受影响

---

## 四、Pipeline 模式：像工厂流水线一样协作

### 4.1 它和 supervisor 模式的区别

supervisor 模式强调“一个中心调度”。  
pipeline 模式更强调“任务按固定阶段流动”。

例如：

1. Retriever Agent 找资料
2. Filter Agent 过滤噪声
3. Writer Agent 生成答案

### 4.2 一个最小示例

```python
def retriever(query):
    return {"docs": ["退款政策", "证书说明"], "query": query}

def filter_agent(data):
    return {"docs": [doc for doc in data["docs"] if "退款" in doc], "query": data["query"]}

def writer(data):
    if not data["docs"]:
        return "未找到足够相关的信息。"
    return f"基于 {data['docs']}，生成最终回答。"

query = "退款政策是什么"
step1 = retriever(query)
step2 = filter_agent(step1)
step3 = writer(step2)

print(step1)
print(step2)
print(step3)
```

### 4.3 它适合什么？

适合：

- 阶段固定
- 顺序清晰
- 每一层职责非常明确

不太适合：

- 需要频繁回头修改计划
- 需要大量灵活协商

---

## 五、Reviewer 模式：生成和检查分离

### 5.1 为什么这个模式很实用？

很多任务里，“生成”和“评审”天然就是两种不同能力。

例如：

- 代码编写 vs 代码审查
- 报告撰写 vs 事实核查
- 答案生成 vs 风险审查

### 5.2 一个可运行示例

```python
def writer_agent(topic):
    return f"关于 {topic} 的初稿：课程购买后 7 天内可退款。"

def reviewer_agent(text):
    if "7 天内" in text:
        return {"approved": True, "comment": "关键信息已覆盖"}
    return {"approved": False, "comment": "缺少核心时间条件"}

draft = writer_agent("退款政策")
review = reviewer_agent(draft)

print("draft :", draft)
print("review:", review)
```

### 5.3 为什么这个模式好用？

因为它可以把“生成质量”和“检查质量”拆开管理。

这在高风险任务里尤其有价值。

---

## 六、Peer / Group 模式：多个 Agent 平等协作

### 6.1 看起来很自由，但也更难控

在这种模式里，多个 Agent 都能提议、争论、补充。

优点：

- 灵活
- 容易激发多种方案

缺点：

- 容易重复劳动
- 容易跑题
- 更难收敛

### 6.2 什么时候考虑它？

比较适合：

- 头脑风暴
- 方案比较
- 多视角分析

但对很多工程系统来说，它未必是最稳的起点。

---

## 七、一个很重要的问题：谁负责收尾？

不管你用哪种模式，都必须回答这个问题：

> 最终由谁来决定“任务完成了”？ 

如果这个问题没设计好，很容易出现：

- 大家都在做事，但没人收尾
- 多个 Agent 互相来回发消息
- 任务迟迟不结束

这也是为什么很多多 Agent 系统，最后还是会有一个“最终裁决者”。

---

## 八、多 Agent 架构的选择逻辑

### 8.1 如果任务阶段固定

优先考虑：

- Pipeline 模式

### 8.2 如果任务需要中心拆分和调度

优先考虑：

- Supervisor-Worker 模式

### 8.3 如果任务需要强评审和复核

优先考虑：

- Writer-Reviewer 模式

### 8.4 如果任务本身就是多视角讨论

才考虑：

- Peer / Group 模式

所以最重要的不是“哪种模式更高级”，而是：

> **哪种模式更匹配你的任务形状。**

---

## 九、初学者最常踩的坑

### 9.1 把多 Agent 当成“多开几个模型就行”

真正难的是架构，不是数量。

### 9.2 一上来就选最自由的协作模式

自由越高，调试和收敛难度通常也越高。

### 9.3 没有明确的结束条件

这是很多多 Agent demo 看起来聪明，但实际跑起来容易死循环的根源。

---

## 小结

这一节最重要的不是背模式名字，而是理解：

> **多 Agent 架构模式的核心，是把任务拆成合适的角色和协作关系，而不是简单增加参与者数量。**

选对架构模式，系统会更稳、更可控；  
选错了，复杂度会比收益增长得更快。

---

## 练习

1. 用自己的话解释 supervisor、pipeline、reviewer 三种模式的区别。
2. 想一想：如果你要做“自动研究报告”，哪种模式最适合先落地？为什么？
3. 设计一个“检索 -> 写作 -> 审核”的三 Agent 流水线。
4. 思考：为什么说多 Agent 架构首先是组织问题，而不是模型数量问题？

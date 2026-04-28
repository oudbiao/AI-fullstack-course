---
title: "6.3 LangChain / LangGraph"
sidebar_position: 31
description: "从链式组件组合到显式状态图，真正理解 LangChain 和 LangGraph 各自在帮你抽象什么。"
keywords: [LangChain, LangGraph, chain, graph, stateful workflow, agent framework]
---

# LangChain / LangGraph

:::tip 本节定位
很多人第一次接触 Agent 框架时，会把 LangChain 和 LangGraph 混在一起。  
但从工程视角看，它们解决的重心并不完全一样：

- LangChain 更像“把常见组件串起来”
- LangGraph 更像“把复杂状态流画成图”

理解这两者的差别，比背接口重要得多。
:::

## 学习目标

- 理解 LangChain 更适合哪些场景
- 理解 LangGraph 为什么更适合复杂状态流
- 看懂“链式抽象”和“图式抽象”的真实差别
- 学会判断什么时候应该从链式流程升级到图式流程

---

## 一、为什么会先有 LangChain，后来又有 LangGraph？

### 1.1 早期需求通常是线性的

很多 LLM 应用一开始都长这样：

1. 接收输入
2. 改写 query
3. 检索文档
4. 调模型回答

这类流程很像一条直线：

> 上一步输出，喂给下一步。

在这种情况下，“链（chain）”是一种很自然的抽象。

### 1.2 但系统很快就会变复杂

一旦你开始有这些需求：

- 如果检索为空怎么办？
- 如果回答不可信要不要重试？
- 要不要先工具调用，再回到问答？
- 某些情况要人工确认

流程就不再是直线了，而会变成：

- 有分支
- 有状态
- 有回路

这时你需要的就不只是“把步骤串起来”，而是：

> **显式表示状态和边。**

这也是 LangGraph 之所以重要的根源。

---

## 二、先理解 LangChain：它到底在抽象什么？

### 2.1 它最适合的，是“组件管道”

LangChain 的典型长处在于把这些东西接起来：

- prompt 模板
- 模型调用
- 输出解析
- 检索模块
- 工具模块

你可以把它理解成：

> 一个偏组件编排层的框架。

它很像把“提示词、模型、检索器、解析器”这些零件做成了更容易拼装的积木。

### 2.2 一个最小链式思维示例

下面这个例子虽然不用真实 LangChain，但它已经有 LangChain 风格。

```python
class SimpleChain:
    def __init__(self, steps):
        self.steps = steps

    def run(self, value):
        for step in self.steps:
            value = step(value)
        return value

def normalize_query(text):
    return text.strip().lower()

def retrieve_docs(query):
    if "退款" in query:
        return {"query": query, "docs": ["课程购买后 7 天内可退款"]}
    return {"query": query, "docs": []}

def format_answer(payload):
    if payload["docs"]:
        return f"根据资料：{payload['docs'][0]}"
    return "没有找到相关资料。"

chain = SimpleChain([
    normalize_query,
    retrieve_docs,
    format_answer
])

print(chain.run("  退款政策是什么？ "))
```

### 2.3 这个例子最该你记住什么？

它在表达一个非常清楚的思想：

> **每一步只做一件事，整个系统通过串联步骤完成任务。**

这正是 LangChain 最容易让人上手的地方。

---

## 三、LangChain 什么时候会很好用？

### 3.1 适合这些情况

- 流程基本线性
- 分支不多
- 重点是把几个模块组合起来
- 你想快速搭一个原型

典型例子：

- FAQ 检索问答
- 文本抽取
- 检索后生成
- 单工具增强问答

### 3.2 它的优点

- 上手快
- 组件生态丰富
- 很适合把“小模块”先拼起来

### 3.3 它容易在哪些地方开始吃力？

当你开始需要：

- 长状态链
- 多个判断分支
- 节点回跳
- 显式中间状态管理

这时链式思维会越来越勉强。

---

## 四、再理解 LangGraph：它为什么更像“状态机”？

### 4.1 LangGraph 的重点不只是节点，而是状态

LangGraph 最重要的视角不是：

- 下一步调哪个组件

而是：

- 当前状态是什么
- 这个状态应该走向哪一条边
- 节点执行后状态怎样更新

你可以先把它理解成：

> **带状态的工作流图。**

### 4.2 一个最小图式示例

```python
def plan_node(state):
    if "退款" in state["query"]:
        state["next"] = "retrieve"
    else:
        state["next"] = "fallback"
    return state

def retrieve_node(state):
    state["docs"] = ["课程购买后 7 天内可退款"]
    state["next"] = "answer"
    return state

def answer_node(state):
    state["answer"] = f"根据资料：{state['docs'][0]}"
    state["next"] = None
    return state

def fallback_node(state):
    state["answer"] = "当前没有找到匹配的流程。"
    state["next"] = None
    return state

nodes = {
    "plan": plan_node,
    "retrieve": retrieve_node,
    "answer": answer_node,
    "fallback": fallback_node
}

state = {"query": "退款政策是什么", "next": "plan"}

while state["next"] is not None:
    current = state["next"]
    state = nodes[current](state)
    print(current, "->", state)
```

### 4.3 这段代码和链式示例最大的差别在哪？

在链式系统里：

- 下一步通常是固定的

在图式系统里：

- 下一步是由当前状态决定的

这就是图工作流最本质的优势。

---

## 五、什么时候你应该从 LangChain 思维切到 LangGraph 思维？

### 5.1 一个很实用的判断标准

如果你画流程图时，发现它已经不是一条直线，而是：

- 有明显分支
- 有失败回退
- 有循环
- 有“根据中间结果决定下一步”

那通常就该开始考虑图式抽象了。

### 5.2 一个信号非常明显的坏味道

如果你的代码开始变成：

```python
if ...
    if ...
        if ...
            while ...
```

并且这些判断全都围绕中间状态，那往往说明：

> 你已经不再只是“链式应用”，而是在做状态图系统。 

---

## 六、为什么很多团队会同时提到 LangChain 和 LangGraph？

因为现实系统往往并不是“二选一”。

### 6.1 一个很常见的组合方式

- LangChain 风格负责封装：
  - prompt
  - retriever
  - parser

- LangGraph 风格负责控制：
  - 状态迁移
  - 分支
  - 重试
  - 人工确认节点

所以很多时候它们更像：

> 组件层 + 工作流层。 

而不是完全冲突的两派。

---

## 七、一个真实工程里的选择建议

### 7.1 如果你现在要做的是：

- 单轮 FAQ
- 简单 RAG
- 几步固定流程

那先用链式思维通常很够。

### 7.2 如果你现在要做的是：

- 多步 Agent
- 带工具回路
- 有人工确认节点
- 明显依赖中间状态

那图式思维会更稳。

### 7.3 不要一开始就为了“高级感”上图

这也是一个很重要的判断。  
图式抽象更强，但也会带来：

- 更高学习成本
- 更多结构设计工作

复杂度不高时，链式方式反而更清楚。

---

## 八、初学者最常踩的坑

### 8.1 还没理解任务结构，就先学一堆框架 API

这会让你最后学会的只是“框架语法”，不是系统设计。

### 8.2 用链去硬撑图

系统越写越多 `if/else`，但还不愿意切到图式抽象。

### 8.3 一上来就上图式框架

明明需求很简单，却先把系统做得很重。

---

## 小结

这一节最重要的不是记住框架名，而是建立这个判断：

> **LangChain 更像把常见组件串起来，LangGraph 更像把复杂状态流显式画出来。**

当你开始用“任务结构”而不是“框架热度”去判断它们时，选型就会稳很多。

---

## 练习

1. 画一下你自己的一个 Agent 系统，判断它更像链还是图。
2. 把链式示例加一个“如果没找到文档就改写 query 再查一次”的逻辑，看看它会不会开始变乱。
3. 用自己的话解释：为什么图式抽象比链式抽象更适合有状态回路的系统？
4. 想一想：在什么情况下，链式方式其实比图式方式更合适？

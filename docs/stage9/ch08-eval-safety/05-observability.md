---
title: "8.5 Agent 可观测性"
sidebar_position: 48
description: "从日志、trace、指标和回放讲起，理解没有可观测性就几乎不可能稳定迭代 Agent 系统。"
keywords: [observability, trace, metrics, logs, replay, agent]
---

# Agent 可观测性

:::tip 本节定位
Agent 系统如果没有可观测性，很多问题会变成：

- 看起来怪怪的
- 但不知道哪一步怪

这节的核心就是：

> **让系统内部过程可以被看见、被定位、被回放。**
:::

## 学习目标

- 理解日志、trace、指标三者区别
- 理解为什么 Agent 特别需要轨迹级观测
- 通过可运行示例建立最小 trace 记录器
- 建立“没有观测就无法迭代”的工程意识

---

## 一、为什么 Agent 比普通接口更需要可观测性？

因为一次请求后面可能包含：

- 多轮推理
- 多个工具
- 中间状态变化

如果只看最终输出，很难知道问题发生在哪一步。

---

## 二、最常见的三类观测对象

### 1. 日志

回答：

- 发生了什么事件

### 2. 指标

回答：

- 整体趋势怎样

### 3. Trace

回答：

- 这次请求完整链路怎么走的

---

## 三、先跑一个最小 trace 记录器

```python
trace = []


def record(stage, payload):
    trace.append({"stage": stage, "payload": payload})


record("input", {"query": "退款规则是什么"})
record("tool_select", {"tool": "search_policy"})
record("tool_result", {"result": "7天内可退"})
record("final_answer", {"answer": "课程购买后 7 天内可退款"})

for item in trace:
    print(item)
```

### 3.1 这个例子最重要的地方是什么？

它说明可观测性不是只记录报错，  
而是把整条链都记下来。

---

## 四、最常见误区

### 1. 只打日志，不打 trace

### 2. 指标太少，无法定位趋势

### 3. 不保留请求级回放信息

---

## 小结

这节最重要的是建立一个判断：

> **Agent 可观测性的核心，是让“这次请求发生了什么”可以被完整重建。**

---

## 练习

1. 给示例再加一个 `latency_ms` 字段。
2. 为什么说 trace 对 Agent 特别重要？
3. 想一想：日志和指标分别更适合回答什么问题？
4. 如果线上只剩最终答案日志，会给排障带来什么困难？

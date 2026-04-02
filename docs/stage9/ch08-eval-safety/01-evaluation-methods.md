---
title: "8.1 Agent 评估方法"
sidebar_position: 44
description: "从任务成功率、轨迹质量、工具使用质量到人工评估，理解 Agent 评估为什么必须是多维度的。"
keywords: [agent evaluation, success rate, trajectory quality, tool usage, eval]
---

# Agent 评估方法

:::tip 本节定位
Agent 最容易让人误判的一点是：

- 看起来很聪明

但“看起来像在思考”不等于：

- 真正完成了任务
- 真正有效率
- 真正可上线

所以 Agent 评估不能只看最终回答，而必须看完整任务链路。
:::

## 学习目标

- 理解 Agent 评估为什么比普通 QA 更复杂
- 了解常见的结果指标、过程指标和成本指标
- 学会构建一个最小多维评估表
- 建立“评估驱动迭代”的工程习惯

---

## 一、为什么 Agent 评估不能只看最终答案？

因为 Agent 往往还会经历：

- 推理步骤
- 工具调用
- 状态更新
- 决策和停止

两个系统即使都答对了，  
也可能有巨大差异：

- 一个 2 步完成
- 一个 9 步还乱调工具

这在工程上差别非常大。

---

## 二、Agent 最常见的评估维度

### 1. 结果质量

例如：

- 任务成功率
- exact match
- 正确率

### 2. 过程质量

例如：

- 是否漏步骤
- 是否逻辑矛盾
- 是否重复行动

### 3. 工具使用质量

例如：

- 工具选择是否合理
- 参数是否正确
- 是否有无用调用

### 4. 成本与效率

例如：

- 平均步数
- 平均耗时
- token 成本

---

## 三、先跑一个最小多维评估示例

```python
traces = [
    {"success": True, "steps": 3, "tool_calls": 1, "cost": 0.02},
    {"success": False, "steps": 5, "tool_calls": 3, "cost": 0.05},
    {"success": True, "steps": 2, "tool_calls": 1, "cost": 0.015},
]


def evaluate(traces):
    n = len(traces)
    success_rate = sum(item["success"] for item in traces) / n
    avg_steps = sum(item["steps"] for item in traces) / n
    avg_tools = sum(item["tool_calls"] for item in traces) / n
    avg_cost = sum(item["cost"] for item in traces) / n

    return {
        "success_rate": round(success_rate, 4),
        "avg_steps": round(avg_steps, 4),
        "avg_tool_calls": round(avg_tools, 4),
        "avg_cost": round(avg_cost, 4),
    }


print(evaluate(traces))
```

### 3.1 这个例子最关键的点

它说明 Agent 评估天然是：

- 多目标

只看一个分数，很容易误判系统质量。

---

## 四、最常见误区

### 1. 只看成功率

### 2. 只看人工主观感觉

### 3. 不记录 trace 就做优化

---

## 小结

这节最重要的是建立一个判断：

> **Agent 评估必须同时看结果、过程、工具使用和成本，否则你很容易优化错方向。**

---

## 练习

1. 给示例再加一个 `latency` 字段，把它也纳入评估。
2. 为什么说两个都“答对”的 Agent，工程价值可能差很多？
3. 想一想：如果一个系统成功率高但平均步数很长，你会怎么判断？
4. 你会为自己的 Agent 增加哪一个评估维度？为什么？

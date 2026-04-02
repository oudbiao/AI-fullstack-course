---
title: "10.2 项目：数据分析 Agent"
sidebar_position: 55
description: "围绕读取数据、统计分析、画图和解释结果，建立一个数据分析 Agent 的最小项目闭环。"
keywords: [data analysis agent, pandas, plotting, insight generation, agent project]
---

# 项目：数据分析 Agent

:::tip 本节定位
数据分析 Agent 类项目很适合展示：

- 工具调用
- 推理
- 结果解释

因为它不只是“算一堆数”，  
还要把数变成用户能理解的结论。
:::

## 学习目标

- 学会定义一个数据分析 Agent 的最小范围
- 学会把数据读取、分析和解释串起来
- 理解这类项目为什么特别适合展示多步工具协作
- 通过项目骨架建立作品集思路

---

## 一、项目最小范围

建议先做：

- 读取一个小表
- 算几个统计量
- 输出一段总结

而不是一开始就做：

- 全自动 BI 平台

---

## 二、项目骨架示例

```python
from dataclasses import dataclass, field


@dataclass
class DataAgentProject:
    name: str
    modules: list
    outputs: list
    metrics: list
    risks: list = field(default_factory=list)


project = DataAgentProject(
    name="data_analysis_agent",
    modules=["load_table", "compute_stats", "plot", "write_insight"],
    outputs=["summary", "chart", "insights"],
    metrics=["analysis_accuracy", "latency", "tool_success_rate"],
    risks=["误解字段含义", "统计口径不一致", "图表误导"],
)

print(project)
```

---

## 三、真实项目里还应该补什么？

### 3.1 数据模式说明

至少要写清：

- 字段有哪些
- 哪些字段能直接分析
- 哪些字段需要清洗

### 3.2 工具链闭环

数据分析 Agent 真正的亮点通常是：

- 读表
- 算统计
- 画图
- 解释

这几个步骤是否真的衔接顺畅。

### 3.3 可复核性

这类项目特别适合展示：

- 原始输入
- 中间统计结果
- 最终洞察文本

这样别人能看出“结论是不是从数据里来的”。

### 3.4 典型失败

建议至少展示：

- 字段理解错误
- 统计口径错误
- 图表误读

---

## 四、小结

这节最重要的是建立一个项目判断：

> **数据分析 Agent 的价值，不只是调用分析工具，而是把分析过程和结果解释串成一个可复核的闭环。**

---

## 练习

1. 给这个项目再加一个“异常检测”模块。
2. 为什么数据分析 Agent 比普通聊天更适合展示多工具协作？
3. 想一想：如果字段含义理解错了，后续结论会怎样被连锁带偏？
4. 你会如何展示这个项目的“可复核性”？

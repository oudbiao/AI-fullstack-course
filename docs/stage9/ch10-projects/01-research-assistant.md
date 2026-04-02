---
title: "10.1 项目：智能研究助手"
sidebar_position: 54
description: "围绕检索、阅读、总结和引用组织，建立一个研究助手 Agent 的最小项目闭环。"
keywords: [research assistant, RAG, summarization, citation, agent project]
---

# 项目：智能研究助手

:::tip 本节定位
研究助手类项目很适合展示 Agent 能力，因为它天然结合了：

- 检索
- 阅读
- 总结
- 引用组织

这节的重点不是做一个巨大的学术系统，而是做一个最小但完整的项目闭环。
:::

## 学习目标

- 学会定义一个研究助手 Agent 的最小范围
- 学会把检索和总结串成项目骨架
- 理解引用和来源追踪为什么重要
- 通过项目结构建立 Agent 作品集思路

---

## 一、项目最小范围怎么定？

建议先只做：

- 给定一个主题
- 检索若干资料
- 输出结构化总结和来源

而不是一上来做：

- 全自动论文写作

---

## 二、先跑一个项目骨架示例

```python
from dataclasses import dataclass, field


@dataclass
class AgentProject:
    name: str
    modules: list
    outputs: list
    metrics: list
    risks: list = field(default_factory=list)


project = AgentProject(
    name="research_assistant",
    modules=["retrieve", "read", "summarize", "cite"],
    outputs=["summary", "bullet_points", "citations"],
    metrics=["answer_quality", "citation_accuracy", "latency"],
    risks=["引用丢失", "检索偏差", "总结幻觉"],
)

print(project)
```

### 2.1 这个示例的意义

它逼你先说清：

- 功能边界
- 输出结构
- 风险

这对 Agent 项目尤其重要。

---

## 三、真实项目里还应该补什么？

### 3.1 检索质量

至少要明确：

- 检索源是什么
- top-k 取多少
- 引用是否来自真实命中文档

### 3.2 引用格式

研究助手项目里，引用不是装饰。  
它直接决定：

- 结果是否可追溯
- 用户是否愿意相信总结

### 3.3 错误分析

特别值得单独展示：

- 检索错导致的总结错
- 引用丢失
- 文本总结幻觉

### 3.4 项目展示建议

这类项目最值得展示的是：

- 输入主题
- 检索到的来源
- 最终总结
- 每条总结对应的引用

---

## 四、小结

这节最重要的是建立一个项目判断：

> **研究助手项目的核心，不是会不会“看起来很聪明”，而是能不能把检索、总结和引用组织成可信的输出。**

---

## 练习

1. 给这个项目再加一个“对比不同来源观点”的模块。
2. 为什么研究助手特别强调引用？
3. 想一想：如果没有来源追踪，这个项目最大的风险是什么？
4. 你会怎样把这个项目做成作品集？

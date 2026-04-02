---
title: "8.2 项目：文本情感分析"
sidebar_position: 2
description: "围绕一个最小情感分析项目，走完标签设计、基线模型、错误分析和部署思路。"
keywords: [sentiment analysis project, text classification, labels, baseline, NLP]
---

# 项目：文本情感分析

:::tip 本节定位
情感分析项目非常适合作为 NLP 入门作品，  
因为它把很多基础能力自然串起来：

- 文本预处理
- 文本表示
- 分类
- 错误分析

同时任务边界也相对清楚。
:::

## 学习目标

- 学会给情感分析任务设计标签边界
- 学会做一个可解释 baseline
- 理解为什么错误分析比总分更重要
- 通过项目闭环练习 NLP 基础能力

---

## 一、项目题目应该怎么定？

建议不要一开始就做特别复杂的多级情绪体系。  
更稳妥的是先从：

- positive
- negative

两类开始。

原因很简单：

- 标签更稳
- 错误更容易分析
- baseline 更容易建立

---

## 二、一个最小情感分析项目包含什么？

### 1. 数据

- 评论文本
- 情感标签

### 2. 基线模型

- 词袋 / TF-IDF + 逻辑回归

### 3. 评估

- accuracy
- 典型错误案例

### 4. 部署展示

- 最小推理接口

---

## 三、先跑一个最小项目骨架示例

```python
from dataclasses import dataclass, field


@dataclass
class NLPProjectPlan:
    name: str
    labels: list
    modules: list
    metrics: list
    risks: list = field(default_factory=list)


plan = NLPProjectPlan(
    name="sentiment_analysis_system",
    labels=["positive", "negative"],
    modules=["preprocess", "vectorize", "train", "evaluate", "serve"],
    metrics=["accuracy", "error_cases"],
    risks=["讽刺反语", "否定词处理", "训练集标签不一致"],
)

print(plan)
```

### 3.1 为什么这一步很重要？

因为项目不是“想到什么就加什么”，  
而是先把：

- 标签
- 模块
- 指标
- 风险

讲清楚。

---

## 四、情感分析项目里最容易出错的地方

### 4.1 否定词

例如：

- “不差”
- “不推荐”

### 4.2 反讽

例如：

- “真是太棒了，又崩了”

### 4.3 标签标准不统一

如果标注者对“中性偏正”和“正面”理解不同，  
模型会很难稳定。

---

## 五、小结

这节最重要的是建立一个 NLP 项目习惯：

> **情感分析项目最先要做的是把标签边界、基线模型和错误类型分析清楚，而不是一开始就追最复杂模型。**

---

## 练习

1. 设计 10 条评论并给出二分类标签。
2. 想一想：如果模型总在否定句上出错，你会先改预处理还是改模型？为什么？
3. 为什么情感分析项目特别适合练“错误分析”？
4. 如果要扩成三分类（正/负/中性），你最担心什么问题？

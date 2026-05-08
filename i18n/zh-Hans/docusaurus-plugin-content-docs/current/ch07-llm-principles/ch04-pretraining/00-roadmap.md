---
title: "7.4.1 预训练路线图：数据、目标、工程"
sidebar_position: 0
description: "紧凑版预训练路线图：数据治理、next-token 目标、工程流水线、污染和评估。"
keywords: [LLM 预训练, 训练数据, next token prediction, 数据治理, 预训练工程]
---

# 7.4.1 预训练路线图：数据、目标、工程

预训练让模型先学到广泛语言模式。工程视角是：清理数据，选择目标，大规模训练，追踪风险。

## 先看预训练三角

![预训练章节关系图](/img/course/ch07-pretraining-chapter-flow.webp)

![预训练数据、目标与工程三角图](/img/course/ch07-pretraining-data-objective-engineering-map.webp)

| 部分 | 先问的问题 |
|---|---|
| 数据 | 哪些文本进入训练，哪些必须过滤？ |
| 目标 | 哪个预测任务产生学习信号？ |
| 工程 | 规模、checkpoint、日志和失败如何处理？ |
| 评估 | 模型能做什么，哪里会失败？ |

## 创建 next-token 样本

```python
tokens = ["AI", "learns", "from", "text"]
pairs = list(zip(tokens[:-1], tokens[1:]))

for source, target in pairs:
    print(f"{source} -> {target}")
```

预期输出：

```text
AI -> learns
learns -> from
from -> text
```

这个小例子就是 next-token prediction 的形状。真实预训练会把它扩展到海量文本，并配合严格的数据治理。

## 按这个顺序学

| 顺序 | 阅读 | 先抓住什么 |
|---|---|---|
| 1 | [7.4.2 预训练数据](./01-pretraining-data.md) | 来源、过滤、去重、污染 |
| 2 | [7.4.3 预训练方法](./02-pretraining-methods.md) | next-token prediction、loss、scaling |
| 3 | [7.4.4 预训练工程](./03-pretraining-engineering.md) | 分布式训练、checkpoint、监控 |

## 通过标准

能解释数据、目标和工程分别如何影响最终模型，并知道数据污染为什么会让评估误导人，就算通过。

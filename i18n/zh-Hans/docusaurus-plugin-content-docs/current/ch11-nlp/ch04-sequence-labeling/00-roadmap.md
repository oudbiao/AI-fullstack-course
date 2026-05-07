---
title: "11.4.1 序列标注路线图：每个 Token 一个标签"
sidebar_position: 0
description: "序列标注的简短实操路线：理解 BIO 标签、NER、HMM/CRF 历史、BiLSTM-CRF 和 span 级评估。"
keywords: [序列标注指南, NER, BiLSTM-CRF]
---

# 11.4.1 序列标注路线图：每个 Token 一个标签

序列标注为每个 token 预测一个标签。NER、分词、词性标注和槽位填充都属于这个思路。

## 11.4.1.1 先看标签路径

![序列标注章节学习流程图](/img/course/ch11-sequence-labeling-chapter-flow.png)

![HMM CRF 序列历史图](/img/course/ch11-hmm-crf-sequence-history-map.png)

![BiLSTM CRF 标签路径图](/img/course/ch11-bilstm-crf-label-path-map.png)

关键输出不是一个句子标签，而是对齐 token 的标签，例如 `B-PER`、`I-PER` 和 `O`。

## 11.4.1.2 跑一个 BIO 标签检查

```python
tokens = ["Ada", "Lovelace", "wrote", "notes"]
tags = ["B-PER", "I-PER", "O", "O"]

for token, tag in zip(tokens, tags):
    print(token, tag)
```

预期输出：

```text
Ada B-PER
Lovelace I-PER
wrote O
notes O
```

如果分词变化，标签必须仍然对齐。很多序列标注 bug 本质上是对齐 bug。

## 11.4.1.3 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | HMM/CRF 历史 | 理解序列约束和标签转移 |
| 2 | NER 与 BIO | 创建 token 级标签和实体 span |
| 3 | BiLSTM-CRF | 连接上下文特征和合法标签路径 |
| 4 | 项目实战 | 评估 precision、recall、F1、边界错误 |

## 11.4.1.4 通过标准

如果你能检查 token/tag 对齐，并解释一个边界错误或非法标签转移，就通过了本章。

---
title: "4.0 学前导读：序列标注这一章到底在学什么"
sidebar_position: 0
description: "先建立序列标注章的学习地图：NER、BiLSTM-CRF 和项目实践是怎样围绕词级标签任务展开的。"
keywords: [序列标注导读, NER, BiLSTM-CRF]
---

# 学前导读：序列标注这一章到底在学什么

这一章解决的是：

> **不是给整句一个标签，而是给序列中的每个位置一个标签。**

## 这一章的主线

```mermaid
flowchart LR
    A["NER 任务直觉"] --> B["BiLSTM-CRF"]
    B --> C["NER 项目实践"]
```

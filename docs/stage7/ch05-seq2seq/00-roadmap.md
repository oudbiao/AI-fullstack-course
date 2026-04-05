---
title: "5.1 学前导读：Seq2Seq 与注意力这一章到底在学什么"
sidebar_position: 0
description: "先建立 Seq2Seq 章节的学习地图：编码器-解码器、注意力和机器翻译任务是怎样衔接的。"
keywords: [Seq2Seq导读, attention导读, 机器翻译]
---

# 学前导读：Seq2Seq 与注意力这一章到底在学什么

这一章解决的是：

> **当输入和输出都是序列时，模型怎样把一段文本变成另一段文本。**

## 这一章的主线

```mermaid
flowchart LR
    A["Encoder-Decoder"] --> B["注意力机制"]
    B --> C["机器翻译实践"]
```

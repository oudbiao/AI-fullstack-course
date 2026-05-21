---
title: "12.1.1 多模态路线图：编码、对齐、使用"
sidebar_position: 0
description: "多模态基础章的简明实操路线图：把图片和文本变成结构化观察，标记不确定性，并接入后续工作流。"
keywords: [多模态入门, 对齐, 融合, 视觉语言模型, 多模态应用]
---

# 12.1.1 多模态路线图：编码、对齐、使用

多模态 AI 不是简单地“上传图片聊天”。真正有用的系统会把图片、文本、音频或视频变成结构化观察，再和任务对齐，最后进入检索、审核、创作或自动化流程。

## 先看流程图

![多模态基础章节学习流程图](/img/course/ch12-multimodal-chapter-flow.webp)

![多模态对齐与融合图](/img/course/multimodal-alignment-fusion.webp)

![多模态系统主干图](/img/course/ch12-multimodal-system-backbone.webp)

先养成一个习惯：输入是什么模态，看到了哪些证据，哪些地方不确定，结构化结果下一步要去哪里。

## 跑一个模拟视觉记录

```python
import json

visible_text = ["RAG", "Embedding", "Vector DB"]
record = {
    "source": "rag-slide.png",
    "modalities": ["image", "text"],
    "visible_text": visible_text,
    "next_step": "send extracted text to retrieval index",
    "uncertainty": ["small footer text is unreadable"],
}

print(json.dumps(record, indent=2))
```

预期输出：

```text
{
  "source": "rag-slide.png",
  "modalities": [
    "image",
    "text"
  ],
  "visible_text": [
    "RAG",
    "Embedding",
    "Vector DB"
  ],
  "next_step": "send extracted text to retrieval index",
  "uncertainty": [
    "small footer text is unreadable"
  ]
}
```

即使还没有接真实视觉模型，这个小记录也足够你先练产品里的数据结构。

## 按这个顺序学

| 步骤 | 阅读内容 | 练习产物 |
|---|---|---|
| 1 | 模态与表示 | 列出图片、文本、音频、视频输入及其结构化字段 |
| 2 | 对齐与融合 | 解释图片证据如何连接到文本任务 |
| 3 | 多模态应用 | 做一个截图或文档理解记录 |

## 通过标准

你能把一张图片或截图转成结构化文本，标记不确定性，并解释结果如何进入 RAG、审核或 Agent 工作流，就算通过本章。

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要说清涉及哪些模态、输入输出契约是什么，以及文字、图像、音频或视频证据如何对齐。
2. 证据应包含真实媒体产物或 trace，并附上质量、安全和失败案例说明。
3. 自检时要能判断任务需要的是生成、理解、检索、工具编排还是人工复核，而不是把所有多模态问题都当成同一种 demo。

</details>


## 留下的证据

学完这一页，至少保留这张证据卡：

```text
源资产：带版本/来源说明的图像、截图、PDF、音频、视频或文本输入
结构化记录：可见文本、对象、区域、时间戳、转写文本或不确定性
融合结果：答案、检索记录、路由决策或多模态特征比较
失败检查：缺少来源、OCR 错误、对齐错误、不确定性或论断无依据
期望产出：可供后续引用或复查的结构化记录
```

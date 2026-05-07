---
sidebar_position: 0
title: "30 分钟 AI 快速体验"
description: "给新人准备的极短 AI 初体验：先看见输入、模型、输出，再继续准备环境。"
keywords: [AI 快速体验, Google Colab, 图像识别, 文本生成, 图像生成, AI 入门]
---

# 30 分钟 AI 快速体验

![30 分钟 AI 快速体验闭环](/img/course/intro-quick-experience-loop.png)

**先感受闭环：**输入 -> 模型 -> 输出。现在不用背术语。

## 最快无代码体验

打开你能访问的任意 AI 聊天或图像工具，输入：

```text
用一个比喻给新手解释 RAG。
```

再把“新手”改成“开发者”，比较两次输出有什么不同。

## 可选 Colab 体验

打开 [Google Colab](https://colab.research.google.com)，新建 Notebook，运行：

```python
!pip install transformers torch pillow requests -q

from transformers import pipeline
from PIL import Image
import io
import requests

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"
image = Image.open(io.BytesIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).content))

for row in classifier(image)[:3]:
    print(f"{row['label']:30s} {row['score']:.1%}")
```

预期形状：

```text
Labrador retriever              95.6%
golden retriever                1.0%
kuvasz                          0.5%
```

## 留下一条笔记

这里的 AI 不神秘：你给输入，训练好的模型处理它，你检查输出。下一步去准备最小环境。

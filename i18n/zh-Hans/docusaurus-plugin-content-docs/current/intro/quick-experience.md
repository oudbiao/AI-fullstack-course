---
sidebar_position: 0
title: "30 分钟 AI 快速体验"
description: "在正式学习前，用图像识别、文本生成和图像生成三个小体验快速感受 AI。"
keywords: [AI 快速体验, Google Colab, 图像识别, 文本生成, 图像生成, AI 入门]
---

# 30 分钟 AI 快速体验

![30 分钟 AI 快速体验闭环](/img/course/intro-quick-experience-loop.png)

**目标：**先跑三个极小 AI 例子，再学习理论。

**需要：**Google Colab 和浏览器。不需要本地安装。

## 1. 你会看到什么

| 体验 | 你做什么 | 后面在哪学 |
|---|---|---|
| 图像识别 | 给图片，得到标签 | 深度学习与视觉 |
| 文本生成 | 给句子开头，得到续写 | Transformer 与大模型 |
| 图像生成 | 给提示词，得到图片 | AIGC 与多模态 |

## 2. 图像识别

打开 [Google Colab](https://colab.research.google.com)，新建 Notebook，运行两个代码单元。

单元 1：

```python
!pip install transformers torch pillow requests -q
```

单元 2：

```python
from transformers import pipeline
from PIL import Image
import io
import requests

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
resp.raise_for_status()
image = Image.open(io.BytesIO(resp.content))

for row in classifier(image)[:3]:
    print(f"{row['label']:30s} confidence: {row['score']:.1%}")
```

预期形状：

```text
Labrador retriever              confidence: 95.6%
golden retriever                confidence: 1.0%
kuvasz                          confidence: 0.5%
```

你的数字可能不同。关键点是：训练好的模型可以给一张你没训练过的图片贴标签。

## 3. 文本生成

再加一个代码单元：

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator(
    "The future of artificial intelligence is",
    max_length=60,
    num_return_sequences=1,
)
print(result[0]["generated_text"])
```

GPT-2 很旧也很小；这里使用它，是因为它能在免费 Notebook 里快速运行。核心思想仍然有用：语言模型会预测接下来可能出现的 token。

## 4. 图像生成

打开你能访问的任意图像生成工具，输入：

```text
a small robot reading a book in a warm library, digital art
```

把 `library` 改成 `spaceship` 再生成一次。你刚刚体验了提示词控制。

## 到这里就够了

只记三件事：

| 信号 | 含义 |
|---|---|
| 识别 | AI 把输入映射成标签 |
| 生成 | AI 续写或创造内容 |
| 提示词控制 | 你的措辞会改变结果 |

下一步打开能力地图，先把它当图来看。

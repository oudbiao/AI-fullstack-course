---
sidebar_position: 0
title: "30 分钟 AI 快速体验"
description: "正式学习前，先用三个小例子体验图像识别、文本生成和图像生成。"
keywords: [AI 快速体验, Google Colab, 图像识别, 文本生成, 图像生成, AI 入门]
---

# 30 分钟 AI 快速体验

![30 分钟 AI 快速体验闭环](/img/course/intro-quick-experience-loop.png)

**目标：** 在学理论前，先跑通三个很小的 AI 例子。

**需要：** 一个可打开 Colab 的 Google 账号。不需要本地安装。

现在不要背术语。复制、运行、看输出，再留下一个问题，后面学习时回来解答。

## 你会体验什么

| 体验 | 会发生什么 | 后面会在哪学 |
| --- | --- | --- |
| 图像识别 | 模型给图片贴标签 | 深度学习和计算机视觉 |
| 文本生成 | 模型接着一句话往下写 | Transformer 和大语言模型 |
| 图像生成 | 模型根据 prompt 画图 | AIGC 和多模态 AI |

## 1. 在 Colab 里做图像识别

打开 [Google Colab](https://colab.research.google.com)，新建 notebook，先运行：

```python
!pip install transformers torch pillow requests -q
```

再新建一个代码单元：

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

results = classifier(image)

print("AI thinks this image is:")
for row in results[:3]:
    print(f"{row['label']:30s} confidence: {row['score']:.1%}")
```

输出大致会像这样：

```text
AI thinks this image is:
Labrador retriever              confidence: 95.6%
golden retriever                confidence: 1.0%
kuvasz                          confidence: 0.5%
```

置信度可能不同。重点是：模型已经从大量图片中学过视觉模式，所以能识别一张你没有亲自标注过的新图。

## 2. 文本生成

新建代码单元：

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "The future of artificial intelligence is"
result = generator(prompt, max_length=80, num_return_sequences=1)

print(result[0]["generated_text"])
```

GPT-2 是较旧、较小的模型。这里使用它只是因为它能在免费 notebook 里快速运行。核心直觉仍然有用：语言模型会不断预测下一个最可能的 token。

## 3. 不写代码体验图像生成

打开你能使用的任意图像生成工具，输入：

```text
a small robot reading a book in a warm library, digital art
```

把 `library` 改成 `spaceship`，再生成一次。你会看到 prompt 控制的第一层含义：文字会变成模型生成时的约束。

## 到这里就够了

你已经看到后面课程会反复出现的三个信号：

| 信号 | 在课程里的含义 |
| --- | --- |
| Recognition | 模型能把输入映射到标签 |
| Generation | 模型能续写或创造内容 |
| Prompt control | 你的措辞会改变输出 |

下一篇能力地图先看图就好。细节会在你做小项目时逐步变清楚。

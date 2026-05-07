---
sidebar_position: 0
title: "0.1 30分 AI クイック体験"
description: "初心者向けの短いAI初回体験です。入力、モデル、出力を見てから環境準備へ進みます。"
keywords: [AIクイック体験, Google Colab, 画像認識, テキスト生成, 画像生成, AI入門]
---

# 0.1 30分 AI クイック体験

![30分AIクイック体験ループ](/img/course/intro-quick-experience-loop-ja.png)

**まず流れを感じます：**入力 -> モデル -> 出力。用語の暗記はまだ不要です。

## いちばん速いノーコード体験

使える AI チャットまたは画像ツールを開き、次を入力します。

```text
初心者に RAG をたとえ話で説明して。
```

次に「初心者」を「開発者」に変えて、出力の違いを見ます。

## 任意の Colab 体験

[Google Colab](https://colab.research.google.com) を開き、新しい Notebook で実行します。

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

出力の形：

```text
Labrador retriever              95.6%
golden retriever                1.0%
kuvasz                          0.5%
```

## メモを1つ残す

ここでの AI は魔法ではありません。入力を渡し、学習済みモデルが処理し、出力を確認します。次は最小環境を準備します。

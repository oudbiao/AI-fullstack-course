---
sidebar_position: 0
title: "30分 AI クイック体験"
description: "本格的な学習の前に、画像認識、テキスト生成、画像生成を短く体験します。"
keywords: [AIクイック体験, Google Colab, 画像認識, テキスト生成, 画像生成, AI入門]
---

# 30分 AI クイック体験

![30分AIクイック体験ループ](/img/course/intro-quick-experience-loop-ja.png)

**目標：**理論に入る前に、3つの小さなAI例を動かします。

**必要なもの：**Google Colab とブラウザ。ローカル環境は不要です。

## 1. 何を見るか

| 体験 | やること | 後で学ぶ章 |
|---|---|---|
| 画像認識 | 画像を渡してラベルを得る | 深層学習とビジョン |
| テキスト生成 | 文の始まりを渡して続きを得る | Transformer と LLM |
| 画像生成 | Prompt を渡して画像を得る | AIGC とマルチモーダル |

## 2. 画像認識

[Google Colab](https://colab.research.google.com) を開き、新しい Notebook で2つのセルを実行します。

セル 1：

```python
!pip install transformers torch pillow requests -q
```

セル 2：

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

出力の形：

```text
Labrador retriever              confidence: 95.6%
golden retriever                confidence: 1.0%
kuvasz                          confidence: 0.5%
```

数値は変わることがあります。大事なのは、学習済みモデルが、あなたが訓練していない画像にもラベルを付けられることです。

## 3. テキスト生成

もう1つセルを追加します。

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

GPT-2 は古く小さいモデルです。ここでは無料 Notebook で速く動かすために使います。考え方は今も同じで、言語モデルは次に来そうな token を予測します。

## 4. 画像生成

使える画像生成ツールを開き、次を入力します。

```text
a small robot reading a book in a warm library, digital art
```

`library` を `spaceship` に変えてもう一度生成します。これで Prompt による制御を体験できます。

## ここで終了

覚えるのは3つだけです。

| 信号 | 意味 |
|---|---|
| 認識 | AI が入力をラベルに変える |
| 生成 | AI が続きや新しい内容を作る |
| Prompt 制御 | 言葉の選び方で結果が変わる |

次は能力マップを、まず画像として眺めてください。

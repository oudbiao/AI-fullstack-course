---
sidebar_position: 0
title: "30 分 AI クイック体験"
description: "本格的に学ぶ前に、画像認識、テキスト生成、画像生成を短く体験します。"
keywords: [AI クイック体験, Google Colab, 画像認識, テキスト生成, 画像生成, AI 入門]
---

# 30 分 AI クイック体験

![30 分 AI クイック体験ループ](/img/course/intro-quick-experience-loop-ja.png)

**目的：** 理論に入る前に、3 つの小さな AI 例を動かします。

**必要なもの：** Colab を開ける Google アカウント。ローカルへのインストールは不要です。

今は用語を覚えなくて大丈夫です。コピーして実行し、出力を見て、あとで答えたい疑問を 1 つ残してください。

## 何を試すか

| 体験 | 何が起きるか | 後で学ぶ場所 |
| --- | --- | --- |
| 画像認識 | モデルが画像にラベルを付ける | 深層学習とコンピュータビジョン |
| テキスト生成 | モデルが文の続きを書く | Transformer と大規模言語モデル |
| 画像生成 | モデルが prompt から画像を作る | AIGC とマルチモーダル AI |

## 1. Colab で画像認識

[Google Colab](https://colab.research.google.com) を開き、新しい notebook を作って、まず実行します。

```python
!pip install transformers torch pillow requests -q
```

次に、2 つ目のコードセルを作ります。

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

出力はおおよそ次の形になります。

```text
AI thinks this image is:
Labrador retriever              confidence: 95.6%
golden retriever                confidence: 1.0%
kuvasz                          confidence: 0.5%
```

confidence の値は変わることがあります。大事なのは、モデルが大量の画像から視覚パターンを学んでいるため、あなたがラベル付けしていない新しい画像も認識できる、という点です。

## 2. テキスト生成

新しいセルを追加します。

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "The future of artificial intelligence is"
result = generator(prompt, max_length=80, num_return_sequences=1)

print(result[0]["generated_text"])
```

GPT-2 は古くて小さいモデルです。ここでは、無料 notebook で素早く動くために使っています。重要な直感は今でも役に立ちます。言語モデルは、次に来そうな token を予測し続けます。

## 3. コードなしで画像生成

使える画像生成ツールを開き、次の prompt を入力します。

```text
a small robot reading a book in a warm library, digital art
```

`library` を `spaceship` に変えて、もう一度生成してください。これが prompt control の最初の感覚です。入力した言葉が、モデル生成時の制約になります。

## ここで完了

このコースで何度も出てくる 3 つの信号を見ました。

| 信号 | コースでの意味 |
| --- | --- |
| Recognition | モデルは入力をラベルに対応づけられる |
| Generation | モデルは続きを書いたり、新しい内容を作ったりできる |
| Prompt control | 言い方を変えると出力が変わる |

次の能力マップは、まず図として見てください。細部は、小さなプロジェクトを作る中で自然に分かってきます。

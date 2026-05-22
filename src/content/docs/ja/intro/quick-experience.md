---
title: "0.1 30分 AI エンジニアリング体験"
description: "短いAIエンジニアリング初回体験です。入力変更を比較し、モデル出力を観察し、セットアップ前に証拠を残します。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AIエンジニアリング体験, AIクイック体験, Google Colab, 画像認識, テキスト生成, AI入門"
---
![30分AIクイック体験ループ](/img/course/intro-quick-experience-loop-ja.webp)

**まず流れを感じます：**入力 -> モデル -> 出力 -> 確認。用語の暗記はまだ不要です。

キャリア転換で最初に身につける習慣は、「AIに驚くこと」ではありません。条件を1つ変え、出力を比較し、他の人が確認できる証拠を記録することです。

## いちばん速いノーコード体験

使える AI チャットまたは画像ツールを開き、次を入力します。

```text
初心者に RAG をたとえ話で説明して。
```

次に「初心者」を「開発者」に変えて、出力の違いを見ます。目的は AI が賢いかどうかを判定することではありません。小さな入力変更が、構成、語彙、例、自信の出し方をどう変えるかを見ることです。

| 変更するもの | 確認するもの |
|---|---|
| 読者：`初心者` -> `開発者` | 例と語彙が変わるか |
| 制約：`80字以内` を追加 | 長さと焦点を守るか |
| 形式：`3つの箇条書き` を追加 | 出力が読みやすくなるか |
| 証拠：`限界を1つ含めて` を追加 | 保証できないことを説明するか |

この小さな比較が、講座全体の最初の習慣です。1回の出力だけを見ないでください。条件を1つ変え、比較し、なぜ良いのかを書いてより良い結果を残します。

## 任意の Colab 体験

[Google Colab](https://colab.research.google.com) を開き、新しい Notebook で実行します。

```python
!pip install transformers torch pillow requests -q

from transformers import pipeline
from PIL import Image
import io
import requests

classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    use_fast=True,
)

image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
response = requests.get(image_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
response.raise_for_status()

content_type = response.headers.get("Content-Type", "")
if "image" not in content_type:
    raise ValueError(f"Expected an image response, got {content_type}: {response.text[:120]}")

image = Image.open(io.BytesIO(response.content)).convert("RGB")

for row in classifier(image)[:3]:
    print(f"{row['label']:30s} {row['score']:.1%}")
```

出力の形：

```text
tabby, tabby cat                27.4%
tiger cat                       27.2%
Egyptian cat                    14.0%
```

数値は環境によって変わります。重要なのは、ラベルと信頼度が順位付きで出る形です。
HTTP 状態と `Content-Type` を先に確認するため、URL が HTML やエラーページを返した場合も、PIL の `UnidentifiedImageError` ではなく読みやすいエラーになります。

## 結果を読む

| 初学者の問い | 実用的な答え | 深い信号 |
|---|---|---|
| 入力は何か | URL から取得した1枚の画像 | 実システムでは種類、サイズ、出所、プライバシーを確認する |
| モデルは何か | 事前学習済み画像分類器 | 学習設定にあるラベルしか扱えない |
| 出力は何か | 上位ラベルとスコア | 高スコアは事実の証明ではなく、モデルの信頼度 |
| どこで失敗するか | download、install、model loading で失敗し得る | 信頼できる AI 作業にはログ、代替手段、再現可能な環境が必要 |

Colab が失敗しても、ここで1日使い切らないでください。エラーを保存し、ノーコード体験を先に終え、第1章で terminal、Python、環境を学んでから戻ります。

## エンジニアリングメモを1つ残す

短いメモを作り、5 行だけ書きます。

```text
成果物名：
試した入力：
観察した出力：
制御した変更：
証拠としての価値：
```

ここでの AI は魔法ではありません。入力を渡し、学習済みモデルが処理し、出力を確認します。経験者は、依存関係のインストール時間、モデル download、入力検証、モデルの限界、出力形式、証拠の残し方という隠れた工程にも注意してください。次は最小環境を準備します。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
ミニアプリ：完成した最小実行可能な AI または自動化デモ
入力と出力：サンプル入力、表示出力、スクリーンショット、またはログ
比較記録: 入力変更を1つ選び、それによる出力差分を書く
概念リンク: 後の章で隠れた仕組みを説明している箇所はどれか
失敗確認：APIキー、依存関係、ネットワーク、プロンプト、または出力形式の問題
期待される成果：小さいが検証可能な AI の振る舞い記録
```

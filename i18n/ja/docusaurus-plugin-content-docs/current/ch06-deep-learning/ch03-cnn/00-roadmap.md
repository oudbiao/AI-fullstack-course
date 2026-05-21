---
title: "6.3.1 CNN ロードマップ：画像を特徴マップへ変える"
sidebar_position: 0
description: "短い CNN ロードマップです。畳み込み、チャネル、特徴マップ、古典的構成、転移学習、画像分類実践を扱います。"
keywords: [CNN ガイド, 畳み込み, 画像分類, 転移学習, 特徴マップ]
---

# 6.3.1 CNN ロードマップ：画像を特徴マップへ変える

CNN は局所的な視覚パターンを学びます。画像を1行に平坦化するのではなく、小さな領域をスキャンして特徴マップを作ります。

## まず画像フローを見る

![CNN 章関係図](/img/course/ch06-cnn-chapter-flow-ja.webp)

![CNN 受容野成長マップ](/img/course/ch06-cnn-receptive-field-growth-map-ja.webp)

| 概念 | 最初の意味 |
|---|---|
| channel | 色または学習された特徴次元 |
| kernel | 小さなスライドフィルタ |
| feature map | フィルタが画像を走査した後の出力 |
| pooling / stride | 空間サイズを小さくする |
| transfer learning | 学習済みの視覚 backbone を再利用する |

## 畳み込みを一度動かす

`cnn_first_loop.py` を作り、`torch` をインストールしてから実行します。

```python
import torch

image = torch.randn(1, 3, 32, 32)
conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
features = conv(image)

print("input_shape:", tuple(image.shape))
print("feature_shape:", tuple(features.shape))
```

出力：

```text
input_shape: (1, 3, 32, 32)
feature_shape: (1, 8, 32, 32)
```

形は `[batch, channels, height, width]` と読みます。畳み込みは `3` 入力チャネルを `8` 個の学習特徴チャネルに変えました。

## この順番で学ぶ

| 順番 | 読む | 練習すること |
|---|---|---|
| 1 | [6.3.2 畳み込み基礎](./01-convolution-basics.md) | kernel、stride、padding、channel |
| 2 | [6.3.3 CNN 構造](./02-cnn-structure.md) | 畳み込みブロック、pooling、分類ヘッド |
| 3 | [6.3.4 古典的アーキテクチャ](./03-classic-architectures.md) | LeNet、AlexNet、VGG、ResNet の直感 |
| 4 | [6.3.5 転移学習](./04-transfer-learning.md) | backbone 凍結、fine-tuning |
| 5 | [6.3.6 画像分類実践](./05-image-classification-practice.md) | データセット、学習、予測例 |

## 残す証拠

CNN shape メモを 1 つ残します。

```text
入力形状：[batch, channels, height, width]
畳み込み出力: out_channels が新しい特徴マップになる
空間変化: stride/padding/pooling により高さと幅が変わる
分類器の橋渡し：conv の特徴は最終的に class logits になる
転移学習の選択：まず freeze し、validation が改善する場合のみ微調整 する
```

## 合格ライン

入力画像形状と特徴マップ形状の間で何が変わったか、そして小規模データセットで学習済み CNN backbone が役立つ理由を説明できれば合格です。

<details>
<summary>確認の考え方と解説</summary>

1. 合格レベルの答えでは、tensor、model layer、loss、`backward()`、optimizer update を1つの学習ループとしてつなげます。
2. 証拠には、動く小さな実験、tensor shape の確認、説明できる loss または validation curve を含めます。
3. shape mismatch、loss が下がらない、過学習、data leakage、Attention/Transformer の data flow を説明できない、といった失敗例を1つ言えればよいです。

</details>

---
title: "6.6.1 生成モデルロードマップ：サンプリング、デコード、レビュー"
sidebar_position: 0
description: "短い生成モデルロードマップです。潜在ベクトル、GAN、VAE、生成出力、評価習慣を扱います。"
keywords: [生成モデルガイド, GAN, VAE, latent vector, deep learning]
---

# 6.6.1 生成モデルロードマップ：サンプリング、デコード、レビュー

生成モデルはラベルを予測するだけでなく、新しいサンプルを作ります。実用上の流れは、潜在コードをサンプリングし、デコードし、出力をレビューし、バージョン比較することです。

## まず生成フローを見る

![生成モデル章関係図](/img/course/ch06-generative-chapter-flow-ja.webp)

![GAN 敵対的バランスマップ](/img/course/ch06-gan-adversarial-balance-map-ja.webp)

| 概念 | 最初の意味 |
|---|---|
| latent vector | 生成に使うコンパクトな隠れ入力 |
| decoder / generator | 潜在コードを出力へ変える |
| discriminator | GAN で本物か生成物かを判定する |
| VAE | より滑らかな潜在空間を学ぶ |
| review | 生成結果にも人と指標の確認が必要 |

## 小さな decoder を一度動かす

`generative_first_loop.py` を作り、`torch` をインストールしてから実行します。

```python
import torch

torch.manual_seed(0)
latent = torch.randn(2, 4)
decoder = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Tanh())
generated = decoder(latent)

print("latent_shape:", tuple(latent.shape))
print("generated_shape:", tuple(generated.shape))
print("value_range:", round(generated.min().item(), 3), round(generated.max().item(), 3))
```

出力：

```text
latent_shape: (2, 4)
generated_shape: (2, 6)
value_range: -0.863 0.695
```

これはまだ本物の生成器ではありません。小さな latent vector をより大きな出力へデコードできる、という形の直感を見ています。

![小さな decoder の実行結果図](/img/course/ch06-generative-tiny-decoder-result-map-ja.webp)

## この順番で学ぶ

| 順番 | 読む | まず見ること |
|---|---|---|
| 1 | [6.6.2 GAN](./01-gan.md) | generator、discriminator、敵対的バランス |
| 2 | [6.6.3 VAE](./02-vae.md) | encoder、decoder、潜在空間 |

## 残す証拠

生成レビュー メモを 1 つ残します。

```text
latent_input: random or encoded compact vector
decoder_output: generated sample or reconstruction
review_needed: generation quality is not proven by loss alone
gan_focus: realism and diversity can fight each other
vae_focus: reconstruction and latent smoothness trade off
```

## 合格ライン

ラベル予測とサンプル生成の違いを説明し、生成結果を盲信せずレビューが必要な理由を言えれば合格です。

<details>
<summary>参考解答と解説</summary>

1. 合格レベルの答えでは、tensor、model layer、loss、`backward()`、optimizer update を1つの学習ループとしてつなげます。
2. 証拠には、動く小さな実験、tensor shape の確認、説明できる loss または validation curve を含めます。
3. shape mismatch、loss が下がらない、過学習、data leakage、Attention/Transformer の data flow を説明できない、といった失敗例を1つ言えればよいです。

</details>

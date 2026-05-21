---
title: "6.1.1 ニューラルネットワークロードマップ：線形層、活性化、損失、更新"
sidebar_position: 0
description: "短いニューラルネットワーク基礎ロードマップです。ニューロン、活性化、順伝播、損失、逆伝播、オプティマイザ、正則化を扱います。"
keywords: [ニューラルネットワークガイド, 深層学習基礎, 活性化関数, バックプロパゲーション, オプティマイザ]
---

# 6.1.1 ニューラルネットワークロードマップ：線形層、活性化、損失、更新

ニューラルネットワークは魔法ではありません。層はまず重み付き和を計算し、活性化で信号の形を変え、学習では重みを調整して loss を下げます。

## まず流れを見る

![ニューラルネットワーク基礎章関係図](/img/course/ch06-nn-basics-chapter-flow-ja.webp)

このループを覚えます。

```text
入力 -> 重み付き和 -> 活性化 -> loss -> 勾配 -> 重み更新
```

| 用語 | 最初の意味 |
|---|---|
| ニューロン | 重み付き和とバイアス |
| 活性化 | ReLU などの非線形変化 |
| 順伝播 | 予測を計算する |
| 逆伝播 | 誤差への責任を計算する |
| オプティマイザ | 勾配で重みを更新する |

## ニューロンを1つ動かす

`nn_first_loop.py` を作り、`torch` をインストールしてから実行します。

```python
import torch

x = torch.tensor([[1.0, -2.0, 3.0]])
weights = torch.tensor([[0.5], [-1.0], [0.25]])
bias = torch.tensor([0.1])

linear_output = x @ weights + bias
activated = torch.relu(linear_output)

print("linear_output:", round(linear_output.item(), 3))
print("relu_output:", round(activated.item(), 3))
```

出力：

```text
linear_output: 3.35
relu_output: 3.35
```

線形出力が負なら、ReLU はそれを `0` にします。この小さなゲートによって、多層ネットワークは非線形パターンを表せます。

## この順番で学ぶ

| 順番 | 読む | まず見ること |
|---|---|---|
| 1 | [6.1.2 ML から DL へ](./00-ml-to-dl-bridge.md) | sklearn の後に何が変わるか |
| 2 | [6.1.3 ニューロンと活性化](./01-neurons-activation.md) | 重み付き和、バイアス、ReLU |
| 3 | [6.1.4 順伝播と逆伝播](./02-forward-backward.md) | 予測、loss、勾配 |
| 4 | [6.1.5 オプティマイザ](./03-optimizers.md) | SGD、Momentum、Adam の直感 |
| 5 | [6.1.6 正則化](./04-regularization.md) | 過学習を抑える |
| 6 | [6.1.7 重み初期化](./05-weight-init.md) | 安定した開始点 |
| 7 | [6.1.8 任意の歴史背景](./06-history-breakthroughs.md) | backprop、CNN、RNN、Attention、Transformer がなぜ現れたか |

## 残す証拠

6.1 の終わりに、次の 4 行メモを残します。

```text
one_layer: input @ weights + bias
nonlinearity: activation lets stacked layers model curved patterns
training: forward -> loss -> backward -> optimizer step
debug_first: check shape, loss, gradient, update
```

このメモは、後で PyTorch、CNN、RNN、Transformer を読むときの小さな地図になります。

## 合格ライン

1つの層を `input @ weights + bias` として説明し、活性化が何をするかを言え、loss、勾配、オプティマイザを1つの学習ループとしてつなげられれば合格です。

<details>
<summary>参考解答と解説</summary>

1. 合格レベルの答えでは、tensor、model layer、loss、`backward()`、optimizer update を1つの学習ループとしてつなげます。
2. 証拠には、動く小さな実験、tensor shape の確認、説明できる loss または validation curve を含めます。
3. shape mismatch、loss が下がらない、過学習、data leakage、Attention/Transformer の data flow を説明できない、といった失敗例を1つ言えればよいです。

</details>

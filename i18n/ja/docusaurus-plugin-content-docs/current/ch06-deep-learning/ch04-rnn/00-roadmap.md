---
title: "6.4.1 RNN ロードマップ：系列を順番に処理する"
sidebar_position: 0
description: "短い RNN ロードマップです。系列入力、隠れ状態、RNN、LSTM、GRU、系列実践を扱います。"
keywords: [RNN ガイド, 系列モデル, LSTM, GRU, hidden state]
---

# 6.4.1 RNN ロードマップ：系列を順番に処理する

RNN は順序のあるデータに向いています。テキスト、時系列、クリック列、センサー値など、前のステップが後のステップに影響する入力です。

## まず系列フローを見る

![RNN 系列モデル章関係図](/img/course/ch06-rnn-chapter-flow-ja.png)

![RNN 隠れ状態ローリングメモリマップ](/img/course/ch06-rnn-hidden-state-rolling-memory-map-ja.png)

| 概念 | 最初の意味 |
|---|---|
| sequence length | 時間ステップの数 |
| input size | 各ステップの特徴数 |
| hidden state | 流れていく記憶 |
| LSTM / GRU | ゲート付き記憶制御 |
| batch first | `[batch, seq_len, features]` の形 |

## GRU の形を一度確認する

`rnn_first_loop.py` を作り、`torch` をインストールしてから実行します。

```python
import torch

sequence = torch.randn(2, 3, 5)
gru = torch.nn.GRU(input_size=5, hidden_size=4, batch_first=True)
outputs, hidden = gru(sequence)

print("sequence_shape:", tuple(sequence.shape))
print("outputs_shape:", tuple(outputs.shape))
print("hidden_shape:", tuple(hidden.shape))
```

出力：

```text
sequence_shape: (2, 3, 5)
outputs_shape: (2, 3, 4)
hidden_shape: (1, 2, 4)
```

2つの系列、各3ステップ、各ステップ5特徴と読みます。GRU はサイズ `4` の隠れ表現を返します。

## この順番で学ぶ

| 順番 | 読む | 練習すること |
|---|---|---|
| 1 | [6.4.2 RNN 基礎](./01-rnn-basics.md) | 系列入力、隠れ状態、形 |
| 2 | [6.4.3 LSTM と GRU](./02-lstm-gru.md) | ゲート、長期依存、記憶制御 |
| 3 | [6.4.4 系列モデリング実践](./03-sequence-practice.md) | スライディングウィンドウ、train/eval ループ |

## 合格ライン

`[batch, seq_len, features]` を読め、hidden state を流れていく記憶として説明でき、LSTM/GRU が長期依存のために導入されたことを説明できれば合格です。

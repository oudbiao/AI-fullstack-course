---
title: "6 深層学習と Transformer 基礎"
sidebar_position: 0
description: "実用的な深層学習ループを学びます：テンソル、モデル、損失、逆伝播、optimizer、曲線、CNN/RNN/Transformer、小プロジェクト。"
keywords: [深層学習, PyTorch, ニューラルネットワーク, CNN, RNN, Transformer, Attention]
---

# 6 深層学習と Transformer 基礎

![深層学習と Transformer のメインビジュアル](/img/course/ch06-deep-learning-ja.png)

第 6 章の目的は 1 つです。モデルが**損失、勾配、反復する学習ステップ**によってどう学ぶのかを理解することです。

## 6.0.1 まず学習ループを見る

![深層学習トレーニングループのメイン図](/img/course/ch06-training-loop-backbone-ja.png)

先に図を見てください。深層学習の学習コードの多くは、この流れです。

```text
batch データ -> モデル forward -> loss -> 勾配 backward -> optimizer step -> 曲線
```

最初から大きなモデルを追わないでください。まず小さなモデルを学習させ、何が起きたかを記録し、なぜ改善または失敗したかを説明します。

## 6.0.2 学習順序とタスクリスト

この表を、本章の学習ガイド兼タスクリストとして使います。

| ページ | 手を動かすこと | 残す証拠 |
|---|---|---|
| [6.1 ニューラルネットワーク基礎](ch01-nn-basics/00-roadmap.md) | ニューロン、活性化、forward/backward、optimizer、正則化、初期化を理解する | 手書きの学習ループ説明 |
| [6.1.2 深層学習の歴史](ch01-nn-basics/06-history-breakthroughs.md) | 任意の背景：backprop、CNN、RNN、Attention、Transformer がなぜ現れたかを軽く読む | 「この構造がある理由」のメモ |
| [6.2 PyTorch](ch02-pytorch/00-roadmap.md) | tensor、autograd、`nn.Module`、Dataset、DataLoader、最小学習ループを練習する | 実行できる PyTorch スクリプト |
| [6.3 CNN](ch03-cnn/00-roadmap.md) | 画像分類でデータ形状、畳み込み、プーリング、転移学習をつなげる | shape メモと画像分類の実行結果 |
| [6.4 RNN](ch04-rnn/00-roadmap.md) | 系列データに記憶が必要な理由、LSTM/GRU が Transformer 前に解いた問題を理解する | 系列モデルメモ |
| [6.5 Transformer](ch05-transformer/00-roadmap.md) | Query、Key、Value、self-attention、位置エンコーディング、Transformer block を学ぶ | attention の入出力図 |
| [6.6 生成モデル](ch06-generative/00-roadmap.md) と [6.7 学習テクニック](ch07-training-tips/00-roadmap.md) | 学習ループが安定してから拡張として扱う | チューニングまたは診断メモ |
| [6.8 プロジェクト](ch08-projects/00-roadmap.md) と [6.8.5 ワークショップ](ch08-projects/04-hands-on-dl-workshop.md) | 画像、感情分析、生成プロジェクトの前に PyTorch 証拠パックを作る | ログ、曲線、checkpoint、shape trace、README |

本章でよく使う用語：

| 用語 | 意味 |
|---|---|
| `tensor` | PyTorch が使う多次元配列 |
| `forward` | データがモデルを通り、予測を作る流れ |
| `loss` | 予測誤差を測る数値 |
| `backward` | loss から勾配を計算すること |
| `optimizer` | 勾配を使ってパラメータを更新するもの |
| `epoch` | 学習データ全体を 1 回見ること |
| `batch` | 一度に処理する小さなサンプル群 |

## 6.0.3 最初の実行ループ

PyTorch がなければ公式セレクタで先にインストールします。PyTorch が使える状態で、次の小さな学習ループを実行してください。

```python
import torch
from torch import nn

torch.manual_seed(42)
x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
y = torch.tensor([[0.0], [2.0], [4.0], [6.0]])

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(20):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch in {0, 1, 5, 19}:
        print(epoch, round(loss.item(), 4))
```

期待される形：

```text
0 ...
1 ...
5 ...
19 ...
```

具体的な数値は環境で変わることがありますが、loss はおおむね下がるはずです。下がれば、学習ループが動いていることを確認できています。

## 6.0.4 よくある失敗

| 症状 | 最初に確認すること | よくある修正 |
|---|---|---|
| shape mismatch | 入力 shape、batch 次元、出力クラス数 | 各層で tensor shape を表示する |
| loss が下がらない | 学習率、ラベル、正規化、損失関数 | まず小さな batch を過学習できるか試す |
| 学習は良いが検証が悪い | 過学習またはデータ分割の問題 | 検証曲線、データ拡張、正則化、early stopping を使う |
| メモリ不足 | batch サイズ、画像サイズ、モデルサイズ | batch/解像度を下げる、軽いモデルにする |
| Transformer が抽象的 | Q/K/V と系列長 | コード前に attention 表を描く |

## 6.0.5 通過チェック

次の 5 つに答えられたら、第 7 章へ進めます。

- `forward`、`loss.backward()`、`optimizer.step()` はそれぞれ何をしますか？
- Dataset と DataLoader はそれぞれ何を解決しますか？
- 学習曲線と検証曲線は、過学習をどう示しますか？
- Attention はなぜ文脈を扱えますか？
- Transformer は後の大規模モデルとどうつながりますか？

印刷用のチェックリストが必要なときは、[6.0 学習ガイドとタスクリスト](./study-guide.md) を使ってください。後の LLM、RAG、多モーダルモデルは、すべてこの表現学習の考え方の上にあります。

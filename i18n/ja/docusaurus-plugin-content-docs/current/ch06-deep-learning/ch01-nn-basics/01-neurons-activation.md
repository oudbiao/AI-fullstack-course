---
title: "6.1.3 ニューロンから多層パーセプトロンへ"
sidebar_position: 3
description: "手を動かして学ぶニューラルネットワーク入門：1 つのニューロン、活性化関数、パーセプトロンの限界、XOR、実行できる PyTorch MLP"
keywords: [ニューロン, パーセプトロン, 活性化関数, ReLU, Sigmoid, Tanh, MLP, PyTorch, XOR]
---

# 6.1.3 ニューロンから多層パーセプトロンへ

![ニューロンから MLP への構造図](/img/course/mlp-neuron-activation-ja.webp)

:::tip この節の概要
ニューラルネットワークは、重み付きスコアを計算し、非線形の活性化を通し、その単位を層として積み重ねる、という単純な考えから始まります。
:::

## 作るもの

この節では、小さな PyTorch 実験を実行します。

- 人工ニューロンを手で計算する；
- `sigmoid` と `ReLU` を比較する；
- 小さな MLP で XOR を解く；
- 1 つの線形層だけでは足りない理由を説明する。

中心となる流れは次です。

```text
features -> weighted sum z -> activation a -> layer -> multilayer network
```

![ニューロンの線形スコアと活性化ゲート図](/img/course/ch06-neuron-linear-activation-gate-ja.webp)

## 最小限の歴史

パーセプトロンが人々を興奮させたのは、機械がデータからルールを学べることを示したからです。その後、XOR のような単純な非線形パターンを単層パーセプトロンが解けないことが明らかになりました。

この歴史から学ぶべきことは次です。

> ニューロン自体は単純。表現力を生むのは、非線形活性化を持つ層の積み重ね。

![XOR における単層パーセプトロンの限界図](/img/course/ch06-xor-single-layer-limit-map-ja.webp)

## セットアップ

```bash
python -m pip install -U torch
```

コードでは安定した PyTorch API を使います：`torch.Tensor`、`nn.Module`、`nn.Sequential`、`nn.Linear`、活性化関数、loss、optimizer です。

## 完全な実験を実行する

`neuron_mlp_lab.py` を作成します。

```python
import torch
import torch.nn as nn


torch.manual_seed(42)

x = torch.tensor([[0.8, 0.3, 0.5]])
w = torch.tensor([[0.2], [-0.4], [0.6]])
b = torch.tensor([0.1])
z = x @ w + b
print("single_neuron")
print("z=", round(float(z.item()), 3))
print("sigmoid=", round(float(torch.sigmoid(z).item()), 3))
print("relu=", round(float(torch.relu(z).item()), 3))

xor_x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
xor_y = torch.tensor([[0.], [1.], [1.], [0.]])


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


model = TinyMLP()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for step in range(2000):
    pred = model(xor_x)
    loss = loss_fn(pred, xor_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    prob = model(xor_x)
    pred = (prob >= 0.5).float()
print("xor_mlp")
for row, p, y_hat in zip(xor_x.tolist(), prob.squeeze().tolist(), pred.squeeze().tolist()):
    print(f"x={row} prob={p:.3f} pred={int(y_hat)}")
print("final_loss=", round(float(loss.item()), 4))
```

実行します。

```bash
python neuron_mlp_lab.py
```

期待される出力：

```text
single_neuron
z= 0.44
sigmoid= 0.608
relu= 0.44
xor_mlp
x=[0.0, 0.0] prob=0.000 pred=0
x=[0.0, 1.0] prob=1.000 pred=1
x=[1.0, 0.0] prob=1.000 pred=1
x=[1.0, 1.0] prob=0.000 pred=0
final_loss= 0.0001
```

![ニューロンと XOR 実験結果図](/img/course/ch06-neuron-xor-run-result-map-ja.webp)

## 1 つのニューロンを読む

最初の部分は次を計算しています。

```text
z = x @ w + b
```

出力では：

```text
z= 0.44
sigmoid= 0.608
relu= 0.44
```

重み付きスコア `z` はまだ線形です。活性化関数が、信号をどう次へ渡すかを変えます。

| 活性化 | すること | よく使う場面 |
|---|---|---|
| `Sigmoid` | `0-1` に押し込む | 二値分類の確率出力 |
| `Tanh` | `-1` から `1` に押し込む | 小さなデモ、一部の系列モデル |
| `ReLU` | 正の値を残し、負の値を 0 にする | 隠れ層の一般的な既定選択 |

## 活性化関数が重要な理由

線形層だけを積み重ねても、全体としては 1 つの大きな線形層と等価です。非線形活性化があるから、層を重ねたネットワークは曲がった境界を表現できます。

そのため、この MLP は次を使います。

```python
nn.Linear(2, 4),
nn.Tanh(),
nn.Linear(4, 1),
nn.Sigmoid(),
```

隠れ層の `Tanh` が非線形の表現力を与えます。最後の `Sigmoid` は、二値分類向けの確率らしい値に変換します。

## XOR が古典的なテストである理由

XOR は 4 行だけです。

| x1 | x2 | y |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

直線 1 本では、このラベルを分けられません。だから単層パーセプトロンは失敗します。小さな MLP が成功するのは、最終判断の前に中間の隠れ特徴を作れるからです。

## 残す証拠

次の小さな結果カードを残します。

```text
single_neuron: z = x @ w + b, activation changes the signal
xor_result: [0, 1, 1, 0] recovered by a tiny MLP
core_reason: nonlinear hidden layers create intermediate features
failure_probe: remove hidden activation and compare final_loss
```

重要なのは、toy model が 4 行を覚えたことではありません。非線形性によって、層を重ねたモデルが表現できる形が変わることです。

## よくあるトラブル

| 症状 | よくある原因 | 修正 |
|---|---|---|
| loss が下がらない | 学習率が高すぎる/低すぎる、loss の組み合わせが違う | LR を下げ、出力活性化と loss の組み合わせを確認する |
| 確率がすべて 0.5 付近 | モデルが学習していない | 長く訓練し、勾配を見て、hidden size を変える |
| output shape エラー | target shape と prediction が違う | この二値例では target を `[batch, 1]` にする |
| `nan` が出る | 学習が不安定 | 学習率を下げ、入力を確認する |
| 訓練データは解けるが実データで弱い | 訓練データを記憶している | train/validation split と正則化を使う |

## 練習

1. 隠れユニットを `4` から `2` に変えてください。XOR は安定して学習できますか？
2. `nn.Tanh()` を `nn.ReLU()` に置き換えてください。結果は変わりますか？
3. 200 step ごとに loss を表示し、学習曲線を見てください。
4. 隠れ層の活性化関数を外し、なぜ弱くなるか説明してください。
5. 隠れ層をもう 1 つ追加し、final loss を比較してください。

## 合格チェック

次を説明できれば、この節はクリアです。

- ニューロンは `x @ w + b` を計算し、その後に活性化を適用する；
- 活性化関数は非線形性を加える；
- 単層パーセプトロンは XOR を解けない；
- MLP は層を積み重ねて中間特徴を作る；
- PyTorch モデルは通常 `nn.Module`、loss、optimizer、`backward()`、`step()` を組み合わせる。

---
title: "6.2.2 つなぎ目：sklearn から PyTorch へ"
sidebar_position: 0
description: "最小限の動く例で sklearn と PyTorch の考え方の違いを理解し、従来の機械学習から深層学習フレームワークへの橋をかける。"
keywords: [sklearn, PyTorch, 深層学習入門, 訓練ループ, テンソル, 勾配降下法]
---

# 6.2.2 sklearn から PyTorch へ

:::tip この節の位置づけ
`scikit-learn` がオートマ車なら、`PyTorch` はマニュアル車のようなものです。

- `scikit-learn` は、たくさんの細かい部分をまとめて隠してくれます
- `PyTorch` では、モデル、損失関数、勾配、訓練の流れを自分で制御します

この節を学ぶと、あなたがどこで「ギアを切り替えている」のかがわかります。
:::

## 学習目標

- `sklearn` と `PyTorch` の役割の違いを理解する
- データ、モデル、損失関数、最適化器、訓練ループの全体像をつかむ
- 最小の例で `sklearn` と `PyTorch` の両方を動かす
- なぜ深層学習には PyTorch のような「より低レベル」なフレームワークが必要なのかを理解する

---

## 一、なぜ sklearn を学んだあとで PyTorch も学ぶの？

第 5 章では、すでに `scikit-learn` を使いました。

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

この体験はとても快適ですが、その分いろいろなものが「見えないところ」に隠れています。

| あなたがすること | sklearn がしてくれること |
|---|---|
| モデルを選ぶ | パラメータの構造を定義する |
| `fit()` を呼ぶ | 順伝播、損失の計算、勾配の計算、パラメータ更新を自動で行う |
| `predict()` を呼ぶ | 推論を自動で行う |

一方、PyTorch では、これらの手順を分けて書きます。

| 手順 | 自分で扱う内容 |
|---|---|
| データを準備する | データを `Tensor` に変換する |
| モデルを定義する | `nn.Module` や `nn.Sequential` でネットワークを書く |
| 損失関数を定義する | たとえば `nn.MSELoss()` |
| 最適化器を定義する | たとえば `torch.optim.SGD()` |
| 訓練ループを書く | `forward -> loss -> backward -> step` |

見た目は少し面倒ですが、その代わりに次のような利点があります。

- どんなネットワーク構造でも作れる
- 訓練プロセスの各ステップを自分で制御できる
- CNN、RNN、Transformer、大規模モデルの微調整など、`sklearn` では対応しにくいことができる

---

## 二、両者を1枚の図で見る

![sklearn から PyTorch へのギアチェンジ図](/img/course/ch06-sklearn-to-pytorch-shift-map-ja.webp)

- `sklearn` では、この流れの多くが `fit()` の中にまとめられています
- `PyTorch` では、この流れがそのまま全部見えます

つまり PyTorch の学習で大事なのは「API が増えること」ではなく、  
**モデル訓練の内部構造に本当に触れ始めること**です。

---

## 三、最小の比較実験

:::info 実行環境
以下のコードはそのまま実行できます。もしローカルで依存関係がまだ入っていないなら、次を実行してください。

```bash
pip install numpy scikit-learn torch
```
:::

最もシンプルな線形回帰をやってみましょう。学習時間から試験点数を予測します。

### sklearn で訓練する

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# 学習時間（時間）
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)

# 対応する点数
y = np.array([52.0, 59.0, 66.0, 73.0, 80.0], dtype=np.float32)

sk_model = LinearRegression()
sk_model.fit(X, y)

print("sklearn の切片:", round(float(sk_model.intercept_), 2))
print("sklearn の重み:", round(float(sk_model.coef_[0]), 2))
print("6時間勉強したときの予測点数:", round(float(sk_model.predict([[6.0]])[0]), 2))
```

期待される出力：

```text
sklearn の切片: 45.0
sklearn の重み: 7.0
6時間勉強したときの予測点数: 87.0
```

きれいな直線モデルが得られ、流れもとてもスムーズです。`fit()` がすでに `score = 7 * hours + 45` という直線を見つけています。

### PyTorch で同じタスクを訓練する

```python
import torch
from torch import nn

torch.manual_seed(42)

# 1. データをテンソルに変換する
X_torch = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_torch = torch.tensor([[52.0], [59.0], [66.0], [73.0], [80.0]])

# 2. モデルを定義する: 1つの線形層 y = wx + b
model = nn.Linear(in_features=1, out_features=1)

# 3. 損失関数を定義する
loss_fn = nn.MSELoss()

# 4. 最適化器を定義する
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 5. 訓練ループ
for epoch in range(1000):
    pred = model(X_torch)                  # forward
    loss = loss_fn(pred, y_torch)          # 損失を計算する

    optimizer.zero_grad()                  # 以前の勾配を消す
    loss.backward()                        # backward
    optimizer.step()                       # パラメータを更新する

    if epoch % 200 == 0:
        print(f"epoch={epoch:4d}, loss={loss.item():.4f}")

weight = model.weight.item()
bias = model.bias.item()
pred_6 = model(torch.tensor([[6.0]])).item()

print("PyTorch の切片:", round(bias, 2))
print("PyTorch の重み:", round(weight, 2))
print("6時間勉強したときの予測点数:", round(pred_6, 2))
```

期待される出力：

```text
epoch=   0, loss=4031.2007
epoch= 200, loss=72.9774
epoch= 400, loss=18.8304
epoch= 600, loss=4.8588
epoch= 800, loss=1.2537
PyTorch の切片: 43.67
PyTorch の重み: 7.37
6時間勉強したときの予測点数: 87.88
```

![sklearn と PyTorch の出力比較図](/img/course/ch06-sklearn-pytorch-result-comparison-map-ja.webp)

この図は上から下へ読みます。

- `sklearn` はこの小さなデータでは直線をそのまま見つけ、`87.0` を予測する
- `PyTorch` はランダムなパラメータから始まり、訓練ループで loss を下げながら同じ直線に近づく
- 大事なのはどちらが上級かではなく、訓練の中身をどこまで見て制御できるか

---

## 四、ここで本当に新しく学んだことは何？

上の PyTorch コードは `sklearn` より長いですが、深層学習の最重要な 5 つの要素を見える形にしています。

| 要素 | たとえ | 役割 |
|---|---|---|
| データ | 食材 | モデルが加工する入力 |
| モデル | 料理人 | 入力をどう出力に変えるかを決める |
| 損失関数 | 採点表 | モデルの出来を判断する |
| 最適化器 | 調整係 | 誤差をもとにパラメータを変える |
| 訓練ループ | 毎日の振り返り | 試行錯誤を繰り返して精度を上げる |

これから CNN、Transformer、RAG の微調整、ローカルモデルの訓練を学んでも、根本はこの 5 つです。  
違うのは、モデルの構造がもっと複雑になるだけです。

---

## 五、いつ sklearn を使い、いつ PyTorch に切り替える？

### `sklearn` のほうが向いている場面

- 表形式データが中心
- モデルが線形回帰、ロジスティック回帰、決定木、ランダムフォレスト、XGBoost のようなもの
- 速くモデル化して、調整したいことを優先したい

### `PyTorch` のほうが向いている場面

- 画像、音声、テキストなどの非構造化データ
- ネットワーク構造を自分で作りたい
- GPU で訓練したい
- 事前学習済みモデルを微調整したい
- 訓練の細かい部分を自分で制御したい

一言で覚えるなら：

> `sklearn` は「従来の機械学習を効率よく使うこと」が得意で、`PyTorch` は「深層学習を柔軟に組み立てること」が得意です。

---

## 六、よくある誤解

### 誤解 1：PyTorch はただの別のモデリングライブラリ

違います。PyTorch はむしろ「深層学習を組み立てるためのプラットフォーム」に近いです。  
単にモデルを呼び出すのではなく、訓練システム全体を組み立てます。

### 誤解 2：PyTorch は sklearn より上位だから、これからは全部 PyTorch を使えばいい

これも違います。実務で大事なのは、**適切な道具を選ぶこと**です。  
表形式データの多くでは、`sklearn` や木系モデルが今でも第一候補です。

### 誤解 3：訓練ループが書ければ、深層学習を理解したことになる

訓練ループは外側の枠にすぎません。さらに次のことも理解する必要があります。

- テンソルと自動微分
- `nn.Module`
- データ読み込み
- モデルのデバッグ
- 訓練の安定性と評価方法

これらは、この章の次の節で順番に補っていきます。

---

## 七、この章のあとでできるようになってほしいこと

この小節を学んだら、少なくとも次の 3 つの質問に答えられるようになってほしいです。

1. `sklearn.fit()` は、どんな手順を隠してくれているのか？
2. なぜ PyTorch の訓練では、損失関数と最適化器が必ず必要なのか？
3. なぜ「モデル + 損失 + 最適化器 + 訓練ループ」が、その後のすべての深層学習コースの共通構造になるのか？

この 3 つを自分の言葉で説明できれば、橋はもうかかっています。

---

## 残す証拠

左右比較のメモを保存します。

```text
sklearn: fit() hides parameter updates
pytorch: I write model, loss, backward, optimizer step
same_goal: minimize error and validate on held-out data
new_responsibility: inspect shape, gradient, device, and checkpoint
```

大事なのは、PyTorch が「より高度」だということではありません。PyTorch は訓練の仕組みを見えるようにし、カスタム深層学習システムを作れるようにする道具です。

## 練習

1. 上の例の学習時間と点数を自分のデータに変えて、それぞれ `sklearn` と `PyTorch` で 1 回ずつ訓練してみましょう。
2. `PyTorch` の学習率を `0.01` から `0.1` と `0.001` に変えて、損失の下がり方の違いを観察しましょう。
3. 100 回ごとの `weight` と `bias` を出力して、パラメータがどのように少しずつ答えへ近づくか見てみましょう。

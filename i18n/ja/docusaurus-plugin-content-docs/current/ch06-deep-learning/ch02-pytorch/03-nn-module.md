---
title: "6.2.5 nn モジュール"
sidebar_position: 3
description: "nn.Module、nn.Linear、nn.Sequential を使ってモデルを整理し、forward とパラメータ管理を理解する。"
keywords: [nn.Module, nn.Linear, nn.Sequential, forward, parameters, PyTorch]
---

# 6.2.5 nn モジュール

## 学習目標

- PyTorch がなぜ `nn.Module` でモデルをまとめるのかを理解する
- `nn.Linear`、`nn.ReLU`、`nn.Sequential` を使えるようになる
- いちばんシンプルな自作ネットワークを自分で書けるようになる
- `forward()`、`parameters()`、`train()`、`eval()` の役割を理解する

---

## まずは地図を作ろう

この節でいちばん大事なのは、クラス名をたくさん覚えることではなく、次の点をはっきり理解することです。

![nn.Module パラメータ整理フロー図](/img/course/ch06-nn-module-parameter-flow-ja.png)

つまり、この節で本当に解決するのは：

- モデル構造をどうやって「学習可能なオブジェクト」としてまとめるか

## この節は前後の内容とどうつながるのか

- 前の節の `autograd` で「勾配がどうやって生まれるか」は解決済み
- この節では「それらのパラメータがどこにあり、どう一元管理されるか」を学ぶ
- 次の節の `DataLoader` で「データをどうやってまとめて入力するか」を学ぶ

つまり、この節は学習ループのための「モデル側」を準備する部分です。

## なぜ `nn.Module` が必要なのか？

Tensor が「データの箱」だとすると、`nn.Module` は「モデルの箱」です。

これが、いろいろな要素をまとめてくれます。

- ネットワーク層
- パラメータ
- 順伝播の処理
- 学習 / 評価モードの切り替え

たとえば、こんなイメージです。

| コンポーネント | たとえ |
|---|---|
| `Tensor` | 1 枚のレンガ |
| `nn.Linear` | ひとつの標準パーツ |
| `nn.Module` | 組み立て可能な機械 |

`nn.Module` がなければ、ネットワークは自分で全部書けますが、とても散らかりやすくなります。  
`nn.Module` があると、モデルはレゴブロックのように一段ずつ組み立てられます。

### 初心者向けの直感：`nn.Module` は「モデルの入れ物」

まずはこう理解して大丈夫です。`nn.Module` には次のものが入ります。

- ネットワーク層
- パラメータ
- 順伝播ロジック
- 学習 / 評価モード

だからこそ、後でたくさんの場所で `model` という 1 つのオブジェクトを渡すだけで、次のことができます。

- 順伝播
- パラメータ更新
- 保存と読み込み

---

## いちばんよく使う層：`nn.Linear`

線形層がしていることはこれです。

> `y = xW + b`

```python
import torch
from torch import nn

layer = nn.Linear(in_features=3, out_features=2)

x = torch.tensor([[1.0, 2.0, 3.0]])
y = layer(x)

print("出力:", y)
print("weight shape:", layer.weight.shape)
print("bias shape:", layer.bias.shape)
```

ここで形状を理解することが大事です。

- 入力は `[1, 3]` で、1 個のサンプルに 3 個の特徴量がある
- 出力は `[1, 2]` で、2 個の出力値に変換される

### `nn.Linear(in, out)` を見たら、まず何を思い浮かべるべき？

まず思い浮かべるべきなのは：

- これは「謎の変換」ではない
- 各サンプルを `in` 次元から `out` 次元へ写像している

なので、線形層をいちばん実用的に理解するときは、次のように考えます。

- 入力空間を新しい特徴空間に再エンコードしている

---

## `nn.Sequential` でネットワークを手早く組む

モデルがシンプルなら、層を順番につないでいけます。

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

x = torch.tensor([[1.0, 2.0]])
pred = model(x)

print(pred)
```

このコードは、次の流れを表します。

1. 2 個の特徴量を入力する
2. まず 4 次元の隠れ層に写像する
3. `ReLU` で活性化する
4. 最後に 1 個の値を出力する

これでもう、最小版の多層パーセプトロンです。

---

## 自分でモデルクラスを定義する

モデルが少し複雑になったら、`nn.Module` を継承するのがおすすめです。

```python
import torch
from torch import nn

class ScorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ScorePredictor()

x = torch.tensor([
    [3.0, 4.0],   # 学習時間、宿題の完了数
    [5.0, 8.0]
])

print(model(x))
```

### `__init__()` と `forward()` はそれぞれ何をするのか？

| メソッド | 役割 |
|---|---|
| `__init__()` | 層とサブモジュールを定義する |
| `forward()` | 「データがどう流れるか」を定義する |

ひとことで覚えるなら：

- `__init__` は「機械を組み立てる」
- `forward` は「機械がどう動くか」を決める

### なぜ `forward()` にはデータの流れだけを書いて、学習処理は書かないのか？

それは、学習処理は別の役割だからです。  
`forward()` の責務はとてもシンプルです。

- 入力を受け取る
- 出力を返す

一方で、次のようなものは `forward()` の仕事ではありません。

- loss
- backward
- optimizer.step

この役割分担をはっきりさせることは、あとで大きなモデルのコードを読むときにとても重要です。

---

## モデルのパラメータはどう管理されるのか？

`nn.Module` の大きな利点の 1 つは、  
自分で定義した層が自動で登録され、パラメータも自動で `model.parameters()` に現れることです。

```python
import torch
from torch import nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = TinyNet()

for name, param in model.named_parameters():
    print(name, param.shape)
```

だからこそ、オプティマイザはこう書けます。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### なぜ `model.parameters()` がそんなに重要なのか？

それは、「モデルは複数のパラメータの集まりだ」という事実を 1 つにまとめてくれるからです。

つまり、オプティマイザは実際には、何層あるかや構造の種類を気にしていません。いちばん大事なのは：

- どのパラメータを更新するのか

`nn.Module` は、その情報を自動で整理してくれます。

モデルが学習すべきパラメータをすべてまとめて持っているからです。

---

## `train()` と `eval()` とは？

初学者の多くは、次のように考えがちです。

- `model.train()` は学習開始
- `model.eval()` は評価開始

でも、正確には少し違います。  
本当の役割は、**モデル内部の一部の層の動作モードを切り替えること**です。

代表的なのは次の 2 つです。

- `Dropout`
- `BatchNorm`

まだ詳しくは扱いませんが、まずはこの習慣を覚えてください。

```python
model.train()   # 学習前
model.eval()    # 検証 / テスト前
```

### 初学段階では、これはしっかり覚えておく価値がある

今はまだ、次の内容を完全には理解していなくても大丈夫です。

- Dropout
- BatchNorm

でも、次の 2 つは条件反射のようにできるようにしておきましょう。

- 学習前に `model.train()`
- 検証前に `model.eval()`

ネットワークが複雑になるほど、この習慣が助けになります。

---

## 完全な小さな例：成績を予測する

以下は、そのまま実行できる小さなネットワークです。  
入力する特徴量は 2 つです。

- 毎週の学習時間
- 毎週こなした問題数

出力は予測スコアです。

```python
import torch
from torch import nn

torch.manual_seed(42)

X = torch.tensor([
    [2.0, 1.0],
    [3.0, 2.0],
    [4.0, 3.0],
    [5.0, 5.0],
    [6.0, 6.0],
    [7.0, 8.0]
])

y = torch.tensor([
    [55.0],
    [60.0],
    [68.0],
    [78.0],
    [85.0],
    [92.0]
])

class ScorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ScorePredictor()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch={epoch:3d}, loss={loss.item():.4f}")

test = torch.tensor([[6.5, 7.0]])
print("予測スコア:", round(model(test).item(), 2))
```

---

## `Sequential` を使うべきとき、自分で `Module` を定義すべきとき

### `nn.Sequential` が向いている場合

次のようなときに向いています。

- 層が完全に順番通りに並んでいる
- 分岐がない
- 特殊な制御ロジックがない

### 自作の `nn.Module` が向いている場合

次のようなときに向いています。

- 複数入力 / 複数出力がある
- スキップ接続や分岐がある
- 条件分岐がある
- 構造をより見やすく、保守しやすくしたい

実際の開発では、自作の `Module` のほうがよく使われます。

---

## 初心者がよくやる間違い

### `forward()` の中で一時的に新しい層を作る

おすすめしません。  
層は `__init__()` の中で定義したほうが、パラメータが正しく登録されます。

### `Sequential` しか書けず、クラス定義ができない

`Sequential` は便利ですが、いずれは自作 `Module` を書けるようになる必要があります。  
後で学ぶ CNN や Transformer も、それなしでは作れません。

### モデルにどんなパラメータがあるか分からない

`named_parameters()` を使う習慣をつけましょう。  
デバッグのときにとても役立ちます。

---

## まとめ

この節で押さえるべき核心は次の 3 つです。

1. `nn.Module` はモデルをまとめる標準的な方法
2. `forward()` は学習手順ではなく、データの流れを表す
3. モデルのパラメータは自動で集められ、オプティマイザが一括で更新できる

モデルの箱ができたら、次はデータを少しずつ入れていく段階です。

## この節でいちばん持ち帰るべきこと

ひとことで言うなら、こうです。

> **`nn.Module` の本当の価値は、コードをオブジェクト指向っぽくすることではなく、「層、パラメータ、順伝播ロジック、学習モード」をまとめて管理できるようにすることです。**

---

## 練習

1. `ScorePredictor` の隠れ層を `8` から `16` に変えて、loss の変化を観察してみましょう。
2. `ReLU()` を外して、モデルがまだ規則を学習できるか確認してみましょう。
3. `named_parameters()` を使って各層のパラメータ名と形状を表示し、それぞれの層を理解できているか確認しましょう。

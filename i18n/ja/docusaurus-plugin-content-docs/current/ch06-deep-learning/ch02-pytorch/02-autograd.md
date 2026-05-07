---
title: "6.2.4 自動微分"
sidebar_position: 2
description: "requires_grad、backward、勾配の累積と no_grad を理解して、学習時にパラメータがなぜ更新されるのかを本当に理解する。"
keywords: [autograd, backward, gradient, requires_grad, no_grad, PyTorch]
---

# 6.2.4 自動微分

![PyTorch Autograd 計算グラフ](/img/course/pytorch-autograd-graph-ja.png)

## 学習目標

- 勾配とは何かを理解する
- `requires_grad=True` の役割を理解する
- `loss.backward()` が何をしているのかを理解する
- 勾配の累積、クリア、`torch.no_grad()` を理解する

---

## まずは全体像をつかもう

この節は、初心者がいちばん「ただ `backward()` が書けるかどうか」の話だと勘違いしやすい部分です。  
でも本当に大事なのは、学習の流れの中でどこに位置するかを先に見ることです。

```mermaid
flowchart LR
    A["順伝播で loss を計算"] --> B["autograd が計算グラフを記録"]
    B --> C["backward() でグラフを逆向きにたどって勾配を求める"]
    C --> D["パラメータが grad を受け取る"]
    D --> E["最適化器がパラメータを更新する"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#333
    style E fill:#e8f5e9,stroke:#2e7d32,color:#333
```

つまり、この節で本当に理解したいのは単なる API ではなく、次の点です。

- 勾配がどうやって loss からパラメータへ伝わるのか

## この節は前の節・次の節とどうつながるのか

- 前の節の `Tensor` では「データの形」を学びました
- この節の `autograd` では「勾配がどう生まれるか」を学びます
- 次の節の `nn.Module` では「モデルをどう整理するか」を学びます

つまり、この節はちょうど真ん中の梁です。  
これがないと、学習は前向き計算だけで終わってしまい、「学ぶ」ことができません。

## 一、なぜ自動微分が必要なのか？

モデル学習の核心は、たった一文で言えます。

> **モデルのパラメータを「損失が小さくなる方向」に動かすこと。**

では、どうやってどの方向に動かせばいいのでしょうか。

答えが**勾配（gradient）**です。

勾配は、山の斜面の傾きのようなものだと考えると分かりやすいです。

- 勾配が大きい場所は、斜面が急です
- 勾配の向きは、損失が最も増える方向を表します
- 私たちは損失を下げたいので、**負の勾配方向**にパラメータを更新します

毎回手計算で勾配を導くのは、とても大変です。  
PyTorch の `autograd` は、自動で記録してくれる帳簿係のようなものです。

- あなたは「loss をどう計算するか」だけ書けばよい
- あとは勾配のつながりを記録してくれる
- `backward()` を呼ぶと、自動で勾配を計算してくれる

### なぜ複雑になると、手計算では無理なのか？

パラメータが 1 つだけなら、手で計算できるかもしれません。  
でもモデルの中に次のようなものが入ると、一気に大変になります。

- 何千、何万ものパラメータ
- 多層構造
- さまざまな中間 Tensor

手計算ではすぐに破綻します。  
だから自動微分の本当の価値は「少し楽になる」ことではなく、

- 大きなモデルを実際に学習できるようにすること

にあります。

---

## 二、最小の例を見てみよう

```python
import torch

# 学習したいパラメータ
w = torch.tensor(2.0, requires_grad=True)

# 簡単な関数を定義: loss = (w * 3 - 10)^2
loss = (w * 3 - 10) ** 2

print("loss:", loss.item())

# 自動微分
loss.backward()

print("w の勾配:", w.grad.item())
```

### ここでは何が起きているのか？

PyTorch はこの計算の流れを記録しています。

```text
w -> w*3 -> w*3-10 -> (w*3-10)^2
```

次のコードを実行すると、

```python
loss.backward()
```

このつながりを逆向きにたどって、連鎖律に従って勾配を順番に計算し、最後に次の値を得ます。

```python
w.grad
```

これは、「今の `w` を少し動かしたら loss がどう変わるか」という情報です。

### この例で最初に押さえるべきことは？

まず押さえるべきなのは次の 3 つです。

- `loss` は最終結果である
- `backward()` は、その最終結果が `w` に与える影響を逆向きに計算する
- `w.grad` には「どう修正すべきか」の情報が入る

この 3 つが分かれば、もっと複雑なネットワークでも、あとはこの鎖が長くなるだけです。

---

## 三、勾配からパラメータ更新へ

勾配があれば、最も基本的な勾配降下法ができます。

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
lr = 0.1

for step in range(5):
    loss = (w * 3 - 10) ** 2
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad

    print(f"step={step}, w={w.item():.4f}, loss={loss.item():.4f}")

    w.grad.zero_()
```

### このコードは毎回何をしているのか？

| コード | 役割 |
|---|---|
| `loss = ...` | 現在の損失を計算する |
| `loss.backward()` | 現在の損失の `w` に対する勾配を求める |
| `w -= lr * w.grad` | 勾配を使ってパラメータを更新する |
| `w.grad.zero_()` | 古い勾配を消して、次の計算に備える |

---

## 四、なぜ勾配をクリアする必要があるのか？

これは PyTorch 初心者が最もつまずきやすいポイントの 1 つです。

PyTorch はデフォルトで勾配を**累積**します。  
自動で上書きはしません。

次の例を見てみましょう。

```python
import torch

x = torch.tensor(3.0, requires_grad=True)

y1 = x ** 2
y1.backward()
print("1回目の backward 後の勾配:", x.grad.item())

y2 = 2 * x
y2.backward()
print("2回目の backward 後の勾配:", x.grad.item())
```

すると、2 回目の勾配は新しい値ではなく、「1 回目 + 2 回目」の合計になっていることが分かります。

だから学習ループでは、普通は次のように書きます。

```python
optimizer.zero_grad()
```

または、

```python
tensor.grad.zero_()
```

### なぜ PyTorch は最初から「累積」にしているのか？

それは、あえてそう使う高度な学習手法があるからです。たとえば、

- 勾配累積
- 複数の loss をまとめて逆伝播する

などです。

なので設計としては正しいです。  
ただし初心者のうちは、まず次の動作を習慣にするとよいです。

- 各更新の前に、必ず勾配をクリアする

![PyTorch 自動微分の勾配ライフサイクル図](/img/course/ch06-autograd-gradient-lifecycle-map-ja.png)

:::tip 図の読み方
この図は 1 回の学習ステップとして読みます。まず順伝播で loss を計算し、`backward()` が勾配を `.grad` に書き込み、`optimizer.step()` がその勾配でパラメータを更新します。最後に必ず `zero_grad()` で古い勾配を消します。PyTorch はデフォルトで勾配を累積するので、「消し忘れ」は初心者に最も多い見えにくいバグです。
:::

---

## 五、`requires_grad=True` は何を制御しているのか？

`requires_grad=True` とマークされた Tensor だけが、PyTorch に勾配追跡されます。

```python
import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=False)

y = a * b + 1
y.backward()

print("a.grad:", a.grad.item())
print("b.grad:", b.grad)
```

出力では次のようになります。

- `a.grad` には値が入る
- `b.grad` は `None`

これは直感的にも自然です。  
もしある値が「学習したいパラメータ」ではないなら、その値の勾配を求める必要はありません。

---

## 六、`torch.no_grad()` は何をするのか？

学習時には勾配を記録します。  
でも推論、評価、パラメータの手動更新では、たいてい**勾配は必要ありません**。

そんなときに使うのが次のコードです。

```text
with torch.no_grad():
    # inference or parameter update code goes here
```

これには次の効果があります。

- 勾配追跡をオフにする
- メモリを節約する
- 推論を速くする

### 初心者が最初に見落としやすい点: パラメータ更新時にも勾配を切ることが多い

手書きの更新コードを見ると、次のように包まれていることがよくあります。

```text
with torch.no_grad():
    # inference or parameter update code goes here
```

理由は次のとおりです。

- パラメータ更新そのものは、学習グラフの中で次に微分したい対象ではない
- なので普通は autograd に追跡させる必要がない

```python
import torch

w = torch.tensor(5.0, requires_grad=True)

with torch.no_grad():
    y = w * 2

print("y.requires_grad:", y.requires_grad)
```

---

## 七、これを「モデル学習」の文脈に戻してみよう

実際の学習では、1 つの数字 `w` だけを更新するのではなく、たくさんのパラメータをまとめて更新します。

たとえば線形モデルなら、

> `y = wx + b`

ここで `w` と `b` はどちらもパラメータで、学習対象です。  
学習時に起きることは、実はずっと同じです。

1. 現在のパラメータで予測する
2. 予測と正解の差から損失を計算する
3. 各パラメータの勾配を自動で求める
4. 最適化器が勾配の方向に沿ってパラメータを更新する

つまり、自動微分は「追加機能」ではなく、深層学習の学習エンジンです。

---

## 八、2 つのパラメータを使った実行可能な例

```python
import torch

# モデルに y = 2x + 1 を学習させたい
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.05

for epoch in range(200):
    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if epoch % 40 == 0:
        print(
            f"epoch={epoch:3d}, loss={loss.item():.4f}, "
            f"w={w.item():.4f}, b={b.item():.4f}"
        )
```

うまくいけば、`w` は `2` に近づき、`b` は `1` に近づきます。

---

## 九、よくある誤解

### `backward()` が自動でパラメータを更新してくれる

いいえ。  
`backward()` は**勾配を求めるだけ**です。実際にパラメータを更新するのは、あなたが書く更新処理か、最適化器の `step()` です。

### 毎回勾配を消さなくても大丈夫

ダメです。  
クリアしないと勾配がどんどん足し合わされてしまい、学習結果がおかしくなることが多いです。

### 推論でもそのまま勾配を有効にしておく

動きはしますが、無駄が多いです。  
評価やデプロイ時には、できるだけ `torch.no_grad()` を使いましょう。

---

## まとめ

この節でいちばん大事な結論は、次の 3 つです。

1. 勾配は「パラメータをどちらに動かすべきか」を教えてくれる
2. `backward()` は勾配を求めるだけで、更新はしない
3. PyTorch はデフォルトで勾配を累積するので、学習ループでは必ずクリアする

自動微分を理解できると、ようやく「モデルを学習する」ということの本質に入れます。

## この節で何を持ち帰るべきか

もっと短く言うなら、次の一文です。

> **autograd の本質は、`loss` がパラメータに与える影響を、計算グラフを使って自動で逆向きに計算すること。**

だから本当に押さえるべきなのは次の点です。

- どの値が勾配を必要とするのか
- 勾配はいつ計算されるのか
- 勾配はいつ累積されるのか
- どの場面で勾配をオフにすべきか

---

## 練習

1. 上の `y = 2x + 1` の例を `y = 3x - 2` に変えて、もう一度学習してみましょう。
2. `w.grad.zero_()` と `b.grad.zero_()` を削除して、学習がどうなるか観察してみましょう。
3. 学習率 `lr` を `0.5` と `0.005` に変えて、収束速度と安定性を比較してみましょう。

---
title: "6.2.8 実用テクニック"
sidebar_position: 6
description: "デバイス切り替え、乱数シード、AMP、勾配クリッピングから checkpoint まで、PyTorch の学習でよく使う実践的な工程テクニックを身につけましょう。"
keywords: [PyTorch, AMP, 混合精度, 勾配クリッピング, checkpoint, device, reproducibility]
---

# 6.2.8 実用テクニック

## 学習目標

この節を終えると、次のことができるようになります。

- CPU / GPU のデバイス切り替えを正しく扱える
- 乱数シードを使って実験の再現性を高められる
- 混合精度学習と勾配クリッピングの役割を理解できる
- モデルの checkpoint を保存・復元できる
- PyTorch のデバッグ用チェックリストを作れる

---

## 一、まずはよくある工程上の問題を解決しよう

### デバイス切り替え：GPU が必ずあるとは思わない

多くの初心者は、コードをそのまま `cuda()` に固定して書いてしまい、GPU のないマシンではすぐにエラーになります。

より安全な書き方は次の通りです。

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("現在のデバイス:", device)

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(device)
print(x)
print("テンソルがあるデバイス:", x.device)
```

`device` は「学習がどの作業台で行われるか」と考えるとわかりやすいです。

- CPU：ふつうの机
- GPU：並列計算が得意な大きな作業台

### 乱数シードを固定する：実験をできるだけ再現可能にする

学習が安定しないとき、最初にやるべきことはモデルを変えることではなく、まず乱数性を固定することです。

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

print(torch.randn(3))
set_seed(42)
print(torch.randn(3))
```

2回の出力が同じなら、この部分の乱数性は固定できています。

:::info なぜ「完全に」ではなく「できるだけ」なの？
GPU の演算や並列処理の細かい違いによって、わずかな差が出ることがあります。  
そのため、再現性はたいてい「かなり近づけられる」ものであって、「完全に同じ」とは限りません。
:::

---

## 二、学習をより安定させる

### `train()`、`eval()`、`no_grad()` は体で覚える

学習と検証で最も書き間違えやすいのは、モデル構造ではなくモードの切り替えです。

基本の習慣は次の通りです。

```text
model.train()   # 学習前
# 学習コードを書く
model.eval()    # 検証 / 推論前
with torch.no_grad():
    # 検証 / 推論コードを書く
```

次のように理解するとよいです。

- `train()`：モデルを「練習モード」にする
- `eval()`：モデルを「試験モード」にする
- `no_grad()`：試験中は逆伝播の下書きをしないので、メモリを節約できる

### 勾配クリッピング：勾配が急に爆発するのを防ぐ

RNN、Transformer、または深いネットワークでは、勾配が大きくなりすぎて学習が不安定になることがあります。  
勾配クリッピングは、勾配に上限を設ける方法です。

```python
import torch
from torch import nn

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

x = torch.randn(32, 10)
y = torch.randn(32, 1) * 50

loss_fn = nn.MSELoss()
pred = model(x)
loss = loss_fn(pred, y)
loss.backward()

def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return total ** 0.5

before = grad_norm(model)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
after = grad_norm(model)

print("クリッピング前の勾配ノルム:", round(before, 4))
print("クリッピング後の勾配ノルム:", round(after, 4))
```

これは、下り坂の自転車に速度制限を付けて、飛ばしすぎないようにするイメージです。

---

## 三、学習を速くする

### 混合精度学習（AMP）：メモリを節約し、速くする

AMP の核心は次の通りです。

> 適切な場所で低い精度を使い、速度向上とメモリ使用量の削減を実現する。

特に GPU 学習に向いています。  
以下のコードは、GPU があるときだけ AMP を使い、GPU がないときは普通に学習するようにしてあります。

```python
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

x = torch.randn(64, 16).to(device)
y = torch.randn(64, 1).to(device)

if device == "cuda":
    scaler = torch.amp.GradScaler("cuda")
    for _ in range(3):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            pred = model(x)
            loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print("AMP を使って GPU 上で 3 ステップ学習しました")
else:
    for _ in range(3):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    print("現在 GPU がないため、通常精度で 3 ステップ学習しました")
```

### Batch が大きすぎるときは？

よくメモリ不足になるなら、次の順で考えましょう。

1. まず `batch_size` を小さくする
2. 次に AMP を検討する
3. さらに勾配累積を検討する

勾配累積の考え方はこうです。

> 一度に大きな batch は食べきれなくても、数回に分けて食べて、最後にまとめて 1 回更新する。

---

## 四、学習の進み具合を保存・復元する

### checkpoint はなぜ大事？

学習は次のような理由で、いつ中断してもおかしくありません。

- 停電
- Notebook のタイムアウト
- GPU の回収
- プログラムエラー

checkpoint は「ゲームのセーブデータ」のようなものです。

### 最小の実行例

```python
import torch
from torch import nn

model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

checkpoint_path = "demo_checkpoint.pt"

# 保存
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": 5
}, checkpoint_path)

print("checkpoint を保存しました:", checkpoint_path)

# 復元
new_model = nn.Linear(2, 1)
new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)

ckpt = torch.load(checkpoint_path, map_location="cpu")
new_model.load_state_dict(ckpt["model_state_dict"])
new_optimizer.load_state_dict(ckpt["optimizer_state_dict"])

print("復元した epoch:", ckpt["epoch"])
```

実際のプロジェクトでは、通常さらに次のものも保存します。

- 最良の検証セット指標
- 学習設定
- tokenizer / label mapping

---

## 五、デバッグするときはどこを見る？

### shape を最初に見る

PyTorch の多くのバグは、実は「モデルが難しい」からではなく、次のような問題です。

- shape が違う
- dtype が違う
- device が一致していない

学習前に、次のような表示をいくつか入れるとよいです。

```python
print("x shape:", x.shape)
print("y shape:", y.shape)
print("x dtype:", x.dtype)
print("x device:", x.device)
```

### 学習が下がらないときの確認順

次の順番で確認できます。

1. データが正しく読み込めているか
2. ラベルがそろっているか
3. loss が正しく計算されているか
4. `optimizer.zero_grad()` を書いているか
5. `backward()` と `step()` の順番が正しいか
6. 学習率が大きすぎたり小さすぎたりしないか

### `nan` が出たらどうする？

よくある原因は次の通りです。

- 学習率が大きすぎる
- 入力スケールが大きすぎる
- 勾配爆発
- ゼロ除算や `log(0)` などの数値問題

最初にやるとよいことは次の 3 つです。

1. 学習率を下げる
2. loss とパラメータの範囲を表示する
3. 勾配クリッピングを有効にする

---

## 六、保存しておける学習テンプレート

```python
model.train()
for batch_x, batch_y in train_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    pred = model(batch_x)
    loss = loss_fn(pred, batch_y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

model.eval()
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        val_loss = loss_fn(pred, batch_y)
```

このテンプレートは派手ではありませんが、とても実用的です。

---

## まとめ

この節で大切なのは、新しい API そのものではなく、学習工程の感覚です。

- デバイスを固定で書かない
- 乱数シードを先に固定する
- `train / eval / no_grad` を区別する
- 勾配が大きすぎたらクリップする
- 学習の進み具合を保存する

多くのモデル学習が止まる原因は、アルゴリズムがわからないからではなく、こうした「小さな工程上の工夫」が足りないからです。

---

## 練習

1. 自分の PyTorch 学習コードに `device` 処理を追加し、CPU と GPU の両方で動くようにしてください。
2. 既存の学習ループに勾配クリッピングを追加し、クリッピング前後の勾配ノルムを表示してください。
3. checkpoint 保存のロジックを追加し、中断後に復元を試してください。

---
title: "4.4 シーケンスモデリング実践"
sidebar_position: 3
description: "本当に学習できる小さな時系列タスクを使って、ウィンドウ構成、RNN/LSTM の学習、検証、予測までをつなげて学びます。"
keywords: [sequence modeling, time series, RNN, LSTM, sliding window, forecast]
---

# シーケンスモデリング実践

:::tip この節の位置づけ
前の2節で、あなたはすでに次を理解しました。

- RNN は「読みながら覚える」
- LSTM / GRU は「もっと賢く記憶をコントロールする」

この節では、これらの概念を小さなプロジェクトに落とし込みます。

> **ある系列を与えて、後ろの値を予測する。**
:::

## 学習目標

- 連続した系列を学習用サンプルに分割できるようになる
- LSTM を使って最小限の時系列予測器を作れるようになる
- 訓練用データ、検証用データ、予測の流れを理解する
- モデルが規則を学んでいるのか、それともただ暗記しているだけかを見分ける
- 実践でよくある落とし穴を知る

---

## 一、なぜ「時系列予測」を実践課題に選ぶのか？

### 1.1 シーケンスモデリングの基礎練習に最適だから

多くの系列タスクは、次のように抽象化できます。

- 前の一部分を入力する
- 後ろの1つを出力する

時系列予測は、その代表例です。

たとえば：

- 過去7日間の売上から、8日目の売上を予測する
- 過去12個の気温から、次の気温を予測する

### 1.2 とても大事な直感

この種のタスクでは、モデルは個々の数字を覚えるのではなく、次を学びます。

> **変化のパターン。**

たとえば：

- 周期
- 傾向
- 変動

これは普通の分類タスクとはかなり違います。

---

## 二、すぐに実行できるデータを先に作る

### 2.1 正弦波 + ノイズで最小限の系列を作る

こうすると、次の利点があります。

- 外部データセットに依存しない
- パターンがわかりやすい
- 教学にとても向いている

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

t = np.arange(0, 200)
series = np.sin(t * 0.1) + np.random.randn(200) * 0.05

plt.figure(figsize=(10, 4))
plt.plot(t, series)
plt.title("Toy Time Series")
plt.grid(True, alpha=0.3)
plt.show()
```

### 2.2 このデータはどんな見た目？

特徴は2つあります。

- 全体として波打っている
- 少しランダムなノイズが入っている

そのため、完全に規則的な系列よりも、実際のタスクに少し近いです。

---

## 三、スライディングウィンドウ：長い系列をどうやってサンプルに切るのか？

### 3.1 核心の考え方

モデルは、そのまま「1本の無限に続く系列」を直接受け取ることはできません。  
通常は、これをたくさんの小さな断片に分けます。

- `window_size` 個の点を入力にする
- `window_size + 1` 個目の点をラベルにする

これをスライディングウィンドウと呼びます。

### 3.2 実行可能な例

```python
import numpy as np

series = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
window_size = 3

X, y = [], []
for i in range(len(series) - window_size):
    X.append(series[i:i + window_size])
    y.append(series[i + window_size])

X = np.array(X)
y = np.array(y)

print("X =\n", X)
print("y =", y)
```

### 3.3 なぜこのステップがそんなに重要なのか？

なぜなら、ここで系列タスクのサンプル定義が決まるからです。  
ウィンドウの作り方を間違えると、その後の学習、検証、予測もすべてずれてしまいます。

---

## 四、データを PyTorch で学習できる形に整える

### 4.1 完全なデータ準備

```python
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

t = np.arange(0, 200)
series = np.sin(t * 0.1) + np.random.randn(200) * 0.05
series = series.astype(np.float32)

window_size = 12
X, y = [], []

for i in range(len(series) - window_size):
    X.append(series[i:i + window_size])
    y.append(series[i + window_size])

X = np.array(X)
y = np.array(y)

# [batch, seq_len, input_size] に変換
X = torch.tensor(X).unsqueeze(-1)
y = torch.tensor(y).unsqueeze(-1)

print("X shape:", X.shape)
print("y shape:", y.shape)
```

### 4.2 なぜ `unsqueeze(-1)` が必要なのか？

LSTM が期待する入力は、通常次の形です。

- `[batch, seq_len, input_size]`

ここでは各時刻に1つの特徴量しかないので、

- `input_size = 1`

になります。

---

## 五、本当に学習できる小さな LSTM 予測器

### 5.1 モデルを定義する

```python
import torch
from torch import nn

class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)
```

### 5.2 なぜ最後の時刻だけを取るのか？

このタスクは次のようなものだからです。

> 前のウィンドウを使って、次の値を予測する

そのため、系列の最後の時刻の表現を、ウィンドウ全体の要約として使うのが最も自然です。

---

## 六、完全な学習フロー

### 6.1 学習 + 検証

```python
import numpy as np
import torch
from torch import nn

np.random.seed(42)
torch.manual_seed(42)

t = np.arange(0, 200)
series = np.sin(t * 0.1) + np.random.randn(200) * 0.05
series = series.astype(np.float32)

window_size = 12
X, y = [], []
for i in range(len(series) - window_size):
    X.append(series[i:i + window_size])
    y.append(series[i + window_size])

X = torch.tensor(np.array(X)).unsqueeze(-1)
y = torch.tensor(np.array(y)).unsqueeze(-1)

train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMForecaster(hidden_size=32)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 40 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val)
        print(f"epoch={epoch:3d}, train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")
```

### 6.2 この学習コードで本当に見るべきポイント

最も重要なのは次の3つです。

- 入力の shape が正しいか
- 最後に `out[:, -1, :]` だけを取っているか
- 損失がちゃんと下がっているか

この3点を理解できれば、あなたはもうシーケンスモデリング実践の入口を越えています。

---

## 七、実際に予測してみる

### 7.1 1つのウィンドウで予測する

```python
model.eval()
with torch.no_grad():
    sample_x = X_val[0:1]
    pred = model(sample_x)
    print("予測値:", float(pred.item()))
    print("正解値:", float(y_val[0].item()))
```

### 7.2 予測値と正解値を描画する

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    val_pred = model(X_val).squeeze(-1).numpy()
    val_true = y_val.squeeze(-1).numpy()

plt.figure(figsize=(10, 4))
plt.plot(val_true, label="true")
plt.plot(val_pred, label="pred")
plt.legend()
plt.title("Validation Prediction")
plt.grid(True, alpha=0.3)
plt.show()
```

実際の系列タスクでは、1つの指標よりもグラフのほうが問題を見つけやすいことがよくあります。

- 波の山や谷に追いつけているか
- 全体的に遅れていないか
- ずっと平らな線になっていないか

---

## 八、シーケンスモデリング実践でよくある落とし穴

### 8.1 データリーク

訓練用データと検証用データの分け方が悪いと、未来の情報がモデルに漏れてしまうことがあります。

時系列タスクでは、基本的に次の方針が安全です。

> 時系列順に分ける。むやみにシャッフルしない。

### 8.2 ウィンドウが短すぎる、または長すぎる

- 短すぎる：モデルが十分な過去を見られない
- 長すぎる：学習が難しくなり、ノイズも増える

### 8.3 loss だけ見て、曲線を見ない

系列予測では、グラフを描くことがとても大事です。  
なぜなら、loss が近い2つのモデルでも、予測の形はまったく違うことがあるからです。

### 8.4 「因果」を学んだつもりで、実は短期パターンしか学んでいない

これはすべての系列予測で注意が必要です。  
モデルが予測できることと、本当に仕組みを理解していることは別です。

---

## 九、とても大事な実務感覚

実際のプロジェクトでは、系列タスクに必ずしも RNN / LSTM を使うとは限りません。  
今では、次のような手法もよく使われます。

- Transformer
- Temporal Convolution
- 伝統的な統計モデル

ただし、将来どんなモデルを使うとしても、この節で学んだウィンドウ構成、時系列の分割、検証方法は、ずっと基礎になります。

---

## まとめ

この節で最も大事なのは、「LSTM を動かすこと」そのものではなく、次を理解することです。

> **シーケンス実践の鍵は、連続データをどう学習サンプルに切り分け、未来情報を漏らさない前提でモデルに変化の規則を学ばせるかにある。**

データの作り方、学習の流れ、検証、予測のグラフ化、これらを一つの流れとしてつなげられるようになってはじめて、シーケンスモデリングは本当に使えるものになります。

---

## 練習

1. `window_size` を 12 から 6 と 24 に変えて、予測結果を比べてみましょう。
2. モデルを LSTM から GRU に変えて、学習曲線が違うか確認してみましょう。
3. あえて訓練用データと検証用データをランダムにシャッフルして、なぜ時系列では危険なのか考えてみましょう。
4. もし系列に明確な週周期があるなら、ウィンドウ長はどう設計すべきか考えてみましょう。

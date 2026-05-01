---
title: "4.3 LSTM と GRU"
sidebar_position: 2
description: "RNN がなぜ忘れてしまうのか、ゲート機構がどのように情報の流れを制御するのかを通して、LSTM と GRU が系列モデリングで果たす役割を理解します。"
keywords: [LSTM, GRU, ゲート機構, cell state, update gate, forget gate]
---

# LSTM と GRU

![LSTM ゲート記憶フロー図](/img/course/lstm-gate-memory-flow-ja.png)

:::tip この節の位置づけ
前の節で、RNN が「読みながら覚える」ことを見ました。  
この節で解決したいのは、もっと現実的な問題です。

> **普通の RNN が長く覚えていられないなら、どうすればいいのか？**

LSTM と GRU は、この「読めるけれど、忘れやすい」という問題を解決するために生まれました。
:::

## 学習目標

- 普通の RNN が遠い情報を忘れやすい理由を理解する
- ゲート機構が何をしているのかを直感的に理解する
- LSTM の cell state と 3 つのゲートを理解する
- GRU の update gate と reset gate を理解する
- PyTorch の `nn.LSTM` と `nn.GRU` の入出力を読めるようになる
- どんなときに LSTM が向いていて、どんなときに GRU で十分かを理解する

## 歴史的背景：なぜ最終的に LSTM にたどり着いたのか？

この節で最も重要な歴史的ポイントは次の 2 つです。

| 年 | できごと | 主な研究者 | 最も重要だった解決内容 |
|---|---|---|---|
| 1994 | Learning Long-Term Dependencies is Difficult | Bengio, Simard, Frasconi | 普通の RNN における長期依存学習での勾配消失問題を体系的に示した |
| 1997 | LSTM | Hochreiter, Schmidhuber | ゲート付き記憶機構で長期依存と勾配問題を緩和した |

初学者の方がまず覚えておくとよいのは、次の点です。

> **LSTM は「RNN を少し複雑にしたもの」ではなく、普通の RNN が長距離情報を安定して覚えるのが難しいという根本問題を解くための仕組みです。**

つまり、この節で本当に大事なのは：

- いくつかのゲート名を覚えること

ではなく、

- なぜそれらのゲートが必要だったのかを理解すること

### なぜ当時、多くの人が LSTM を「正面から助けに来た仕組み」だと見たのか？

それは、当時の RNN の流れが誰にも支持されていなかったわけではないからです。  
むしろ、とても魅力的に見えました。

- テキストは系列
- 音声も系列
- 時系列データも系列

直感的には、RNN はこうしたタスクにぴったりのように見えます。

しかし、実際に学習させると、多くの人が同じ壁にぶつかりました。

- 早い時点の情報を保てない
- 勾配を過去へ伝えるほど弱くなる
- 系列が長くなると「読めるけれど忘れる」ようになる

そのため、LSTM が当時大きな注目を集めたのは、「ゲートが賢いから」だけではありません。  
それ以前に、次のことをはっきり示したからです。

> **RNN の方向性が間違っていたのではなく、記憶をもっと丁寧に管理する仕組みが必要だった。**

### なぜ「勾配消失」は、RNN を主役にしたい人たちを不安にさせたのか？

紙の上では、RNN はほとんどどんな系列にも使えそうに見えます。

- テキスト
- 音声
- 時系列

でも、実際に長い系列を学習させると、次のことが起こります。

- 早い情報ほど残りにくい
- 勾配が時間をさかのぼるほど小さくなる

これは、長い物語を最初から最後まで完璧に言い直せると思っていたのに、  
話が進むほど最初の細部があいまいになっていく感じに似ています。

だからこそ、LSTM の登場は「ゲートが増えた」こと以上に意味がありました。  
それは、次のメッセージを伝えていたからです。

> **普通の RNN だけで記憶させるのが難しいなら、記憶そのものを管理する仕組みを加えよう。**

これが、LSTM が後に大きな転換点として見られる理由です。

---

## 一、なぜ普通の RNN では足りないのか？

### 1.1 典型的な問題：長距離依存

次の文を見てください。

> 「私は子どものころ、上海に長年住んでいたので、今はもう引っ越したけれど、一番よく知っている都市はやはり上海です。」

モデルが最後に「上海」を読んだとき、前半でどの都市の話をしていたかを知るには、かなり前の情報を覚えておく必要があります。

普通の RNN は理論上は可能ですが、実際には次の問題がよく起こります。

- 後ろに行くほど、最初の情報が薄れる
- 学習中に勾配が消えやすい
- 系列が長いと、記憶が不安定になる

### 1.2 直感的なたとえ

普通の RNN は、紙に書いた短い要約を何度も書き換えているようなものです。

- 新しい文が来るたびに、古い要約を更新する

問題は、

- 要約できるスペースが小さい
- 古い情報が上書きされやすい

ということです。

そこで登場したのが、もっと賢い考え方です。

> **単なる「変わる要約」に頼るのではなく、モデルに「何を忘れ、何を残し、何を出力するか」を学ばせる。**

これがゲート機構です。

---

## 二、LSTM の核心となる直感：記憶に「門」をつける

### 2.1 LSTM は何が増えたのか？

LSTM は普通の RNN に比べて、主に次の点が強化されています。

- より安定した記憶の通り道: `cell state`
- 情報の流れを制御する複数のゲート

まずは次のように理解するとよいです。

> **普通の RNN が小さなノート 1 冊だけを持っているとしたら、LSTM はもっと細かく記憶を管理する仕組みを持っています。**

### 2.2 LSTM の 3 つのゲート

| ゲート | 役割 |
|---|---|
| Forget Gate | 古い記憶をどれだけ残すかを決める |
| Input Gate | 新しい情報をどれだけ書き込むかを決める |
| Output Gate | 現在どれだけ外に出力するかを決める |

これらのゲートは、人が手でルールを書くのではなく、モデルが学習して得るものです。

---

## 三、まずは「スカラー版」LSTM で直感をつかむ

### 3.1 なぜスカラー版から見るのか？

実際の LSTM は最初から行列とベクトルだらけで、初心者には少し見づらいからです。  
小さくした形で見ると、本質がつかみやすくなります。

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# これは前の時刻の記憶だとする
c_prev = 0.8

# 現在の入力と前の隠れ状態
x_t = 1.2
h_prev = 0.5

# ここではゲート値を手で決める。実際のモデルではネットワークがこれを学習する
forget_gate = sigmoid(1.0)   # 約 0.73
input_gate = sigmoid(0.2)    # 約 0.55
output_gate = sigmoid(0.7)   # 約 0.67

# 新しい候補情報
c_tilde = np.tanh(0.9)

# cell state を更新
c_t = forget_gate * c_prev + input_gate * c_tilde

# hidden state を更新
h_t = output_gate * np.tanh(c_t)

print("forget_gate =", round(float(forget_gate), 4))
print("input_gate  =", round(float(input_gate), 4))
print("output_gate =", round(float(output_gate), 4))
print("c_t         =", round(float(c_t), 4))
print("h_t         =", round(float(h_t), 4))
```

### 3.2 このコードは何を教えているのか？

このコードが教えているのは、次のことです。

- `forget_gate` は古い記憶をどれだけ捨てるかを決める
- `input_gate` は新しい情報をどれだけ書くかを決める
- `output_gate` は外にどれだけ見せるかを決める

つまり、LSTM が本当に強いのは「複雑だから」ではなく、

> **情報の流れを制御できるようになったこと**

にあります。

![LSTM ゲートによる情報流制御図](/img/course/ch06-lstm-gates-information-control-map-ja.png)

:::tip 図の見方
この図では、まず 3 つだけ注目してください。Forget Gate は古い記憶をどれだけ残すか、Input Gate は新しい情報をどれだけ書くか、Output Gate は外にどれだけ出すかを決めます。LSTM の本質はゲート名ではなく、「記憶を管理する」ことです。
:::

---

## 四、LSTM の 2 つの状態: `c_t` と `h_t`

### 4.1 なぜ 2 つの状態があるのか？

LSTM には通常、次の 2 つがあります。

- `c_t`：cell state。より長期記憶の主な通り道
- `h_t`：hidden state。現在時刻の外向きの出力に近い

### 4.2 覚えやすい比喩

次のように考えると覚えやすいです。

- `c_t`：長期用の下書きノート
- `h_t`：今、外に向かって話している内容

下書きノートの内容は全部口に出すわけではありませんが、これが後の記憶の土台になります。

---

## 五、GRU：より軽量なゲート付きモデル

### 5.1 なぜ GRU が登場したのか？

LSTM は強力ですが、構造は少し複雑です。  
その後、人々は GRU（Gated Recurrent Unit）を提案しました。目的は次のような、よりシンプルな仕組みを作ることでした。

- より簡単
- パラメータが少ない
- それでも性能は大きく落ちない

### 5.2 GRU の 2 つの主要ゲート

| ゲート | 役割 |
|---|---|
| Update Gate | 古い状態をどれだけ残し、新しい状態をどれだけ混ぜるかを決める |
| Reset Gate | 新しい状態を計算するときに、どれだけ古い情報を忘れるかを決める |

### 5.3 最小限の GRU の直感例

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

h_prev = 0.7
x_t = 1.1

update_gate = sigmoid(0.8)
reset_gate = sigmoid(-0.3)

h_candidate = np.tanh(x_t + reset_gate * h_prev)
h_t = (1 - update_gate) * h_prev + update_gate * h_candidate

print("update_gate =", round(float(update_gate), 4))
print("reset_gate  =", round(float(reset_gate), 4))
print("h_candidate =", round(float(h_candidate), 4))
print("h_t         =", round(float(h_t), 4))
```

### 5.4 LSTM との直感的な違い

- LSTM：より細かく記憶を管理する仕組み
- GRU：それを少し簡略化した仕組み

なので、よく次のように覚えられます。

> **GRU = 少し軽い LSTM**

---

## 六、LSTM と GRU はどう選ぶ？

### 6.1 一般的な目安

系列モデルのベースラインを作るだけなら、

- まず GRU を試すと手早いことが多いです

長距離依存が特に重要なタスクなら、

- LSTM を試す価値が高いことがあります

### 6.2 ただし、過剰に神格化しない

今の大規模モデルの時代では、多くの長文タスクは Transformer に任されることが増えています。  
それでも、LSTM / GRU は次のような場面で今でもよく使われます。

- 比較的短い系列のモデリング
- 小規模データ
- 時系列タスクのベースライン
- 系列モデリングの本質を学ぶための教材

---

## 七、PyTorch で LSTM と GRU をどう使うか？

### 7.1 最小の実行例

```python
import torch

torch.manual_seed(42)

# batch=4, seq_len=6, input_size=8
x = torch.randn(4, 6, 8)

lstm = torch.nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
gru = torch.nn.GRU(input_size=8, hidden_size=16, batch_first=True)

lstm_out, (lstm_h, lstm_c) = lstm(x)
gru_out, gru_h = gru(x)

print("lstm_out shape:", lstm_out.shape)
print("lstm_h shape  :", lstm_h.shape)
print("lstm_c shape  :", lstm_c.shape)
print("gru_out shape :", gru_out.shape)
print("gru_h shape   :", gru_h.shape)
```

### 7.2 出力はそれぞれ何か？

LSTM では、次の値が返ります。

- `lstm_out`：各時刻の出力
- `lstm_h`：最後の hidden state
- `lstm_c`：最後の cell state

GRU では、次の値が返ります。

- `gru_out`：各時刻の出力
- `gru_h`：最後の hidden state

ここからも違いがすぐ分かります。

> **LSTM は GRU より 1 つ多く `c` 状態を持っています。**

---

## 八、小さなタスク: 系列の先頭情報を覚えさせる

次のような小さなタスクを作ってみます。

- 入力系列の 1 番目の位置が `+1` または `-1`
- ラベルはその 1 番目の値で決まる
- 途中にはたくさんノイズを入れる

つまり、モデルは「かなり前の情報」を覚えていないといけません。

```python
import torch
from torch import nn

torch.manual_seed(42)

def build_dataset(n=100):
    X, y = [], []
    for _ in range(n):
        first = 1.0 if torch.rand(1).item() > 0.5 else -1.0
        seq = torch.randn(8, 1) * 0.2
        seq[0, 0] = first
        X.append(seq)
        y.append(1 if first > 0 else 0)
    return torch.stack(X), torch.tensor(y)

X, y = build_dataset(120)

class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=8, batch_first=True)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        out, h = self.gru(x)
        return self.fc(h[-1])

model = GRUClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

for epoch in range(80):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        acc = (pred.argmax(dim=1) == y).float().mean().item()
        print(f"epoch={epoch:3d}, loss={loss.item():.4f}, acc={acc:.3f}")

with torch.no_grad():
    final_acc = (model(X).argmax(dim=1) == y).float().mean().item()
    print("final acc =", round(final_acc, 3))
```

この小さなタスクは、とても単純ですが、次のことを教えてくれます。

> ゲート付きの循環ネットワークは、普通の RNN よりも早い時点の重要な情報を保つのが得意です。

---

## 九、初学者がよくつまずく点

### 9.1 LSTM / GRU を「RNN より深いもの」と考える

そうではありません。  
本質は「より深い」ではなく、「記憶の管理が賢い」ことです。

### 9.2 `out`、`h`、`c` を混同する

覚え方は次の通りです。

- `out`：各ステップの出力
- `h`：最後の hidden state
- `c`：LSTM の長期記憶状態

### 9.3 LSTM を使えば自動的に忘れなくなると思う

そうではありません。  
普通の RNN より忘れにくいだけで、無限に長い依存関係を簡単に解けるわけではありません。

---

## まとめ

この節で一番大事なのは、数式を丸暗記することではなく、次の考え方を理解することです。

> **LSTM と GRU の本質は、ゲート機構で「何を忘れるか、何を残すか、何を出力するか」を学習できるようにしたことです。**

これは普通の RNN に対する重要な改良であり、後の注意機構や Transformer を理解するための良い土台にもなります。

---

## 練習

1. LSTM のスカラー例でゲートの値を変えて、`c_t` と `h_t` がどう変わるか見てみましょう。
2. GRU の分類タスクを「ラベルが最後の値で決まる」ように変更して、学習しやすさを比べてみましょう。
3. 同じタスクを LSTM と GRU の両方で試し、学習速度とコードの複雑さを比較してみましょう。
4. 自分の言葉で説明してみましょう。なぜ LSTM / GRU の重要な点は「複雑さ」ではなく、「情報の流れをより細かく制御できること」なのか？

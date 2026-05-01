---
title: "3.5 転移学習 🔧"
sidebar_position: 4
description: "なぜゼロから学習しないのか、バックボーンを凍結すること、分類ヘッドを置き換えること、段階的に微調整することまで、ビジョンにおける転移学習をしっかり理解します。"
keywords: [転移学習, fine-tuning, feature extractor, freeze backbone, transfer learning, CNN]
---

# 転移学習

:::tip この節の位置づけ
もしあなたがすでに CNN が特徴を抽出することや、代表的なアーキテクチャがどう進化してきたかを知っているなら、次にとても自然に出てくる実務上の疑問はこれです。

> **自分の画像タスクをやるとき、本当に一つの CNN をゼロから学習する必要があるの？**

多くの場合、答えは「いいえ」です。  
転移学習は、「ほかのタスクで学んだ視覚の知識をどう借りてくるか」に答える考え方です。
:::

## 学習目標

- 画像タスクで転移学習が、ゼロから学習するより現実的な理由を理解する
- 「固定された特徴抽出器」と「微調整」という2つの代表的な方法を区別する
- 分類ヘッドの置き換えと、バックボーンのパラメータ凍結を学ぶ
- 実際に動く小さな転移学習の例を読めるようになる
- どのタイミングでヘッドだけを学習し、どのタイミングでより多くの層を解凍するべきかを理解する

---

## 一、なぜ転移学習は、ほぼビジョンタスクのデフォルトになったのか？

### 1.1 ゼロから学習するのはどれくらい大変？

それなりの画像モデルをゼロから学習しようとすると、通常は少なくとも次の問題にぶつかります。

- データが十分に多くない
- ラベル付けのコストが高い
- 学習に時間がかかる
- 過学習しやすい

たとえば、手元に 2000 枚の画像があって、5 クラスに分けたいとします。  
これは実際のプロジェクトでは極端に少ないわけではありませんが、深い CNN をゼロから学習するには、それでも安定しない可能性があります。

### 1.2 事前学習モデルは、いったい何を「事前学習」しているのか？

大規模画像データで学習されたモデルは、たいてい多くの汎用的な視覚特徴をすでに学んでいます。

- エッジ
- テクスチャ
- 色の組み合わせ
- 部品の形
- よくある物体パターン

これらは「猫と犬のタスク専用」の能力ではなく、多くの画像タスクで使える基礎的な視覚知識です。

つまり、転移学習の核心となる直感はこうです。

> **まずすでに学習済みの低レベル視覚能力を再利用し、最後の数層を自分のタスクに合うように調整する。**

### 1.3 覚えやすい例え

転移学習は、一般的な絵の技術をすでに学んでいる人に、専門的なイラストを描いてもらうようなものです。

- 「鉛筆の持ち方」から学び直す必要がない
- あなたは、その人に自分のスタイルや題材に合わせてもらえばよい

だからこそ、ビジョンタスクでは転移学習がとても有利なのです。

---

## 二、転移学習でよく使われる2つの方法

### 2.1 方法1：固定された特徴抽出器（feature extractor）

やり方：

- 事前学習済みバックボーンのパラメータは更新しない
- 最後の分類ヘッドだけを学習する

メリット：

- 速い
- 事前学習の能力を壊しにくい
- データがとても少ない場面に向いている

デメリット：

- 新しいタスクへの適応力は限られる

### 2.2 方法2：微調整（fine-tuning）

やり方：

- 最後の分類ヘッドを置き換える
- ヘッドだけでなく、バックボーンの一部、あるいは全部を段階的に解凍して学習する

メリット：

- 目的のタスクにより適応しやすい

デメリット：

- 過学習しやすい
- 学習が遅くなる
- 学習率により注意が必要になる

### 2.3 一言で覚えるなら

- データが少ない：まずは固定された特徴抽出器を優先
- データが多い / タスク差が大きい：段階的な微調整を検討

![転移学習で backbone を凍結するか、段階的に微調整するかの判断図](/img/course/ch06-transfer-learning-freeze-finetune-map.png)

:::tip 図の読み方
この図を見るときは、まず2つを確認します。データ量がどれくらいあるか、そして新しいタスクが事前学習タスクとどれくらい似ているかです。データが少なく、タスクが近いなら、まず backbone を凍結して head だけを学習します。データが多い、またはタスクの違いが大きいなら、後ろの層を少しずつ解凍して、より小さい学習率で微調整します。
:::

---

## 三、「そのまま動かせる」転移学習の小さな例

外部モデルのダウンロードなしでも動くように、ここでは「すでに事前学習済みの backbone」を自分で模擬します。

### 3.1 まず小さな backbone を定義する

```python
import torch
from torch import nn

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        return x.flatten(1)
```

この backbone の出力は、固定長の特徴ベクトルです。  
これは、多くの実際の事前学習モデルが「バックボーンの特徴を出力する」ときの形によく似ています。

---

## 四、まずは「固定された特徴抽出器」版を作る

### 4.1 分類ヘッドを置き換えて backbone を凍結する

```python
import torch
from torch import nn

class TransferClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = nn.Linear(16, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

model = TransferClassifier(num_classes=3)

# backbone を凍結
for param in model.backbone.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    print(name, "trainable =", param.requires_grad)
```

### 4.2 出力から何が分かるべきか？

次のことが分かります。

- `backbone` のパラメータはすべて学習されない
- 学習されるのは `head` のパラメータだけ

これが、最も標準的な「ヘッドだけを学習する」転移学習です。

---

## 五、本当に学習できる小さな画像分類タスクを作る

### 5.1 合成データで小さなタスクを模擬する

3 クラスの簡単な画像を作ります。

- 縦線
- 横線
- 斜め線

こうすれば、外部データセットなしでも学習の流れを一通り動かせます。

```python
import numpy as np
import torch

def make_image(label, size=12):
    img = np.zeros((size, size), dtype=np.float32)

    if label == 0:  # 縦線
        img[:, size // 2] = 1.0
    elif label == 1:  # 横線
        img[size // 2, :] = 1.0
    else:  # 斜め線
        for i in range(size):
            img[i, i] = 1.0

    img += np.random.randn(size, size).astype(np.float32) * 0.05
    return np.clip(img, 0.0, 1.0)

X, y = [], []
for label in range(3):
    for _ in range(80):
        X.append(make_image(label))
        y.append(label)

X = torch.tensor(np.array(X)).unsqueeze(1)
y = torch.tensor(np.array(y))

print(X.shape, y.shape)
```

---

## 六、完全な学習：ヘッドだけを学習する

```python
import torch
from torch import nn

torch.manual_seed(42)

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.features(x).flatten(1)

class TransferClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = nn.Linear(16, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

model = TransferClassifier(num_classes=3)

# backbone を凍結
for param in model.backbone.parameters():
    param.requires_grad = False

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.head.parameters(), lr=0.05)

for epoch in range(80):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        acc = (pred.argmax(dim=1) == y).float().mean().item()
        print(f"epoch={epoch:3d}, loss={loss.item():.4f}, acc={acc:.3f}")
```

### 6.2 このコードで本当に学んでほしいこと

学んでほしいのは、「パラメータを凍結する」という文法そのものではありません。  
大事なのは次の考え方です。

> 転移学習の最初の一歩は、モデル全体を再学習することではなく、今ある特徴だけで自分のタスクを支えられるかを見ることです。

---

## 七、いつさらに微調整すべきか？

### 7.1 よくある次の一手

ヘッドだけの学習で十分な性能が出ないなら、次を検討します。

- 最後の畳み込みブロックを解凍する
- もっと小さい学習率で学習を続ける

### 7.2 最小限の微調整の例

```python
# 最後の畳み込み層を解凍する
for param in model.backbone.features[3].parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.005
)

for epoch in range(40):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        acc = (pred.argmax(dim=1) == y).float().mean().item()
        print(f"finetune epoch={epoch:3d}, loss={loss.item():.4f}, acc={acc:.3f}")
```

### 7.3 なぜ微調整では、たいてい学習率を小さくするのか？

バックボーンには、すでに「もともと学んだ特徴」のセットがあります。  
学習率が大きすぎると、それなりに良かった表現を一気に壊してしまいやすいです。

そのため、よくある経験則はこうです。

- ヘッドの学習率は少し大きめ
- backbone の学習率は小さめ

---

## 八、実際のプロジェクトでは、転移学習は普通どう進める？

### 8.1 最もよくある流れ

1. 事前学習済みの backbone を選ぶ
2. 最後の分類ヘッドを置き換える
3. まずはヘッドだけを学習する
4. 必要なら段階的に解凍する
5. 検証データの結果を見続ける

### 8.2 なぜこの流れがよく使われるのか？

次の要素のバランスが良いからです。

- 学習速度
- 安定性
- 最終的な性能

これは、最初から全部を学習するより、たいてい安定しています。

---

## 九、初心者がよくハマる落とし穴

### 9.1 転移学習は「大きなモデルをコピーすること」だと思ってしまう

本当に大事なのは次です。

- どの層を凍結するか
- どの層を解凍するか
- 学習率をどう設定するか

### 9.2 最初から全部を微調整する

これは、特にデータが少ないタスクでは、遅くて不安定になりがちです。

### 9.3 どのパラメータが学習されているか確認し忘れる

これはとてもよくあるミスです。  
学習前に `requires_grad` の状態を一度出力して確認するのがおすすめです。

---

## まとめ

この節で最も大事なのは、「転移学習」という言葉を覚えることではなく、安定した実務感覚を身につけることです。

> **まず事前学習済みモデルが学んだ汎用特徴を再利用し、その後で自分のタスクに応じて、ヘッドだけを学習するか、部分的に学習するか、全部学習するかを決める。**

これが、現実の多くのビジョンプロジェクトで転移学習が「テクニック」ではなく、ほぼ出発点になっている理由です。

---

## 練習

1. 例の 3 クラスを 4 クラスに増やし、新しい画像パターンを 1 つ設計してみてください。
2. 「ヘッドだけを学習する場合」と「さらに 1 層解凍する場合」の学習曲線を比べてみてください。
3. モデルのすべてのパラメータの `requires_grad` を出力して、本当にどの層が学習されているか確認してください。
4. 考えてみてください。もし目標タスクと元の事前学習タスクの差が非常に大きいなら、なぜより多くの層を解凍する必要があるのでしょうか。

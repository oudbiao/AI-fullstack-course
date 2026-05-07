---
title: "6.3.6 CNN 実践：画像分類"
sidebar_position: 5
description: "データ作成、ネットワーク設計、学習、検証、予測まで、ひとつの小さな CNN 画像分類プロジェクトを最後まで通します。"
keywords: [image classification, CNN, PyTorch, train loop, validation, synthetic dataset]
---

# 6.3.6 CNN 実践：画像分類

:::tip この節の位置づけ
畳み込み、CNN の構造、代表的なアーキテクチャ、転移学習まで学んだあとで、いちばん大事なのは次のことです。

> **これらの概念を、本当にひとつの学習の流れとしてつなげること。**

この節では「大規模モデルの性能」を目指すのではなく、もっと大事なことを目指します。

> 画像分類プロジェクトを、最初から最後まで一通りやり切れるようになること。
:::

![CNN 画像分類の実践ループ](/img/course/ch06-cnn-image-classification-practice-loop-ja.png)

:::tip この図の読み方
コードを実行する前に、まず図で流れを確認しましょう。合成画像とラベルを確認し、テンソル形状 `N x C x H x W` を追い、最後に CNN、損失、検証曲線、誤分類確認をひとつの閉じたループとしてつなげます。
:::

## 学習目標

- 最小構成で学習できる画像分類タスクを作る
- CNN を使って、学習・検証・予測を一通り動かす
- 画像分類プロジェクトで、データ・モデル・損失関数・指標がどう連携するかを理解する
- 結果から、モデルが本当に学べているかを判断できるようになる

---

## 一、画像分類プロジェクトの最小の流れとは？

画像分類プロジェクトには、少なくとも次の要素が必要です。

1. データ
2. クラスラベル
3. モデル
4. 損失関数
5. 学習ループ
6. 検証 / テスト

初学者が「なんとなく分かった気がする」で止まりやすいのは、モデルの構造だけを見て、全体の流れをつなげていないからです。

この節のポイントは、この流れをひとつずつ通していくことです。

---

## 二、まずはそのまま動かせるデータを用意する

### なぜ引き続き合成画像を使うのか？

理由は次のとおりです。

- 外部ダウンロードに依存しない
- クラスの規則がとても分かりやすい
- 教学に最適

### 3 種類の小さな画像を作る

次の 3 クラスを作ります。

- 縦線
- 横線
- 斜め線

```python
import numpy as np
import matplotlib.pyplot as plt

def make_image(label, size=12):
    img = np.zeros((size, size), dtype=np.float32)

    if label == 0:
        img[:, size // 2] = 1.0
    elif label == 1:
        img[size // 2, :] = 1.0
    else:
        for i in range(size):
            img[i, i] = 1.0

    img += np.random.randn(size, size).astype(np.float32) * 0.05
    return np.clip(img, 0.0, 1.0)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for label in range(3):
    axes[label].imshow(make_image(label), cmap="gray")
    axes[label].set_title(f"class {label}")
    axes[label].axis("off")
plt.tight_layout()
plt.show()
```

### このデータセットは単純ですが、何を学ぶには十分ですか？

このデータセットで学べることは次のとおりです。

- 画像テンソルをどう整理するか
- 分類ラベルをどう対応づけるか
- CNN が局所的なパターンをどう学ぶか

入門段階では、いきなり大きなデータセットを使うより、こちらのほうがずっと学びやすいです。

---

## 三、データを学習用と検証用に分ける

```python
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

X, y = [], []
for label in range(3):
    for _ in range(100):
        X.append(make_image(label))
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

# シャッフル
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# テンソルに変換
X = torch.tensor(X).unsqueeze(1)  # [N, 1, H, W]
y = torch.tensor(y)

split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print("train:", X_train.shape, y_train.shape)
print("val  :", X_val.shape, y_val.shape)
```

### なぜ `unsqueeze(1)` が必要なのか？

PyTorch の畳み込み層は、入力を次の形で受け取ります。

- `[batch, channel, height, width]`

ここではグレースケール画像なので、チャネル数は 1 です。

---

## 四、最小構成の CNN 分類器を定義する

```python
import torch
from torch import nn

class TinyCNNClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = TinyCNNClassifier(num_classes=3)
sample_out = model(X_train[:4])
print("sample output shape:", sample_out.shape)
```

### なぜここで `16 * 3 * 3` になるのか？

元の画像サイズは `12x12` です。

- 1 回目の `MaxPool2d(2)` のあとで `6x6`
- 2 回目の `MaxPool2d(2)` のあとで `3x3`

最後の出力チャネルは 16 なので、Flatten したあとのサイズは次のとおりです。

> `16 * 3 * 3`

これは CNN 実践でとてもよく出る shape 計算です。

---

## 五、完全な学習ループ

```python
import torch
from torch import nn

torch.manual_seed(42)

model = TinyCNNClassifier(num_classes=3)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    train_logits = model(X_train)
    train_loss = loss_fn(train_logits, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val)
            train_acc = (train_logits.argmax(dim=1) == y_train).float().mean().item()
            val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()

        print(
            f"epoch={epoch:3d}, "
            f"train_loss={train_loss.item():.4f}, "
            f"val_loss={val_loss.item():.4f}, "
            f"train_acc={train_acc:.3f}, "
            f"val_acc={val_acc:.3f}"
        )
```

### このコードで、いちばん注目すべきものは？

画像分類を学ぶときに、まず見るべきなのは次の 4 つです。

- `train_loss`
- `val_loss`
- `train_acc`
- `val_acc`

これらを見ることで、次のことが分かります。

- モデルが学べているか
- 過学習していないか
- 安定して収束しているか

---

## 六、実際に予測してみる

### 1 つのサンプルを見る

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    sample = X_val[0:1]
    pred = model(sample).argmax(dim=1).item()
    true = y_val[0].item()

plt.imshow(sample[0, 0].numpy(), cmap="gray")
plt.title(f"pred={pred}, true={true}")
plt.axis("off")
plt.show()
```

### なぜこのステップが大事なのか？

よくあるのは次のようなケースです。

- 指標は良さそうに見える
- でも、モデルが何を見ているのか分からない

実際にいくつかの予測結果を見ると、感覚をつかみやすくなります。

---

## 七、モデルが本当に学習できたかをどう判断するか？

### 典型的な良いサイン

モデルがちゃんと学べているなら、次のような変化が見られます。

- train loss が下がる
- val loss もたいてい下がる
- train / val acc が上がる
- 1 サンプルごとの予測がだんだん安定する

### 典型的な異常

### 学習用も検証用も悪い

考えられる原因は次のとおりです。

- モデルが弱すぎる
- 学習率が合っていない
- データの作り方に問題がある

### 学習用は良いが、検証用が悪い

考えられる原因は次のとおりです。

- 過学習
- データ数が少なすぎる
- ノイズが大きすぎる

### loss が動かない

考えられる原因は次のとおりです。

- shape が間違っている
- ラベルが間違っている
- 学習率が小さすぎる

---

## 八、実際の画像分類プロジェクトでは何を追加するのか？

この教材の例は、意図的にかなり小さくしています。  
実際のプロジェクトでは、通常さらに次の要素が必要です。

- DataLoader
- データ拡張
- より実データに近いデータセット
- より強い backbone
- もっと体系的な評価指標
- モデルの保存と復元

つまり、ここで学ぶのは次のことです。

> 「本番向けの最終解」ではなく、「ひと通りの流れを通すこと」。

---

## 九、初学者がよくハマる落とし穴

### モデルだけをまねして、データの shape を確認しない

画像タスクでは、shape の確認がほぼ最優先です。

### train loss だけを見る

検証用の指標も同じくらい大事です。

### 動けば終わりだと思ってしまう

本当のプロジェクトは、「動く」だけでなく、「結果を説明できる」ことが大事です。

---

## まとめ

この節で最も大事なのは、CNN を動かすことそのものではなく、画像分類プロジェクトの全体の流れをつなげることです。

> **データを作る / 整理する / モデルを定義する / 学習する / 検証する / 1 サンプルを予測する。**

この流れを一度きちんと通せるようになると、あとでより複雑なデータセットや強いモデルに変えても、慌てずに進められます。

---

## 練習

1. 今のデータに、4 つ目の画像パターンとして「反対対角線」を追加してみましょう。
2. `TinyCNNClassifier` のチャネル数を大きくして、収束速度が変わるか見てみましょう。
3. `Dropout` を追加して、検証用データでの挙動がより安定するか試してみましょう。
4. 考えてみましょう。なぜ画像分類プロジェクトでは、ネットワークを何層か増やすことよりも、データの作り方と検証方法のほうが大事なことが多いのでしょうか。

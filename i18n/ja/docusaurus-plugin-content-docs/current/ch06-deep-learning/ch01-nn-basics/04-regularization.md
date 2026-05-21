---
title: "6.1.6 正則化"
sidebar_position: 6
description: "手を動かして学ぶ正則化：dropout、weight decay、early stopping、train-validation gap、PyTorch での過学習診断"
keywords: [正則化, dropout, weight decay, early stopping, 過学習, PyTorch, AdamW]
---

# 6.1.6 正則化

![正則化で過学習を抑える図](/img/course/regularization-overfitting-controls-ja.webp)

:::tip この節の概要
正則化は、training loss をできるだけ低くするためだけのものではありません。validation や未来のデータでよく汎化させるためのものです。
:::

## 作るもの

この節では、PyTorch 実験で次を比較します。

- 正則化なし；
- dropout；
- weight decay；
- dropout + weight decay；
- `best_epoch` による early stopping の挙動。

![過学習の問題から正則化の行動を選ぶ図](/img/course/ch06-regularization-overfit-action-map-ja.webp)

## セットアップ

```bash
python -m pip install -U torch scikit-learn
```

## 完全な実験を実行する

`regularization_lab.py` を作成します。

```python
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_data():
    X, y = make_moons(n_samples=500, noise=0.28, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.35, random_state=42, stratify=y
    )
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32),
    )


class MLP(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def accuracy(logits, y):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return (pred == y).float().mean().item()


def train_case(name, dropout=0.0, weight_decay=0.0, epochs=120):
    torch.manual_seed(42)
    X_train, y_train, X_val, y_val = make_data()
    model = MLP(dropout=dropout)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)
    best_val = 10**9
    patience = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss = loss_fn(model(X_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= 20:
            break

    model.eval()
    with torch.no_grad():
        train_loss = loss_fn(model(X_train), y_train).item()
        val_loss = loss_fn(model(X_val), y_val).item()
        train_acc = accuracy(model(X_train), y_train)
        val_acc = accuracy(model(X_val), y_val)
    print(
        f"{name:<14} epochs={epoch:<3} best_epoch={best_epoch:<3} "
        f"train_loss={train_loss:.3f} val_loss={val_loss:.3f} "
        f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
    )


print("regularization_lab")
train_case("plain", dropout=0.0, weight_decay=0.0)
train_case("dropout", dropout=0.25, weight_decay=0.0)
train_case("weight_decay", dropout=0.0, weight_decay=0.05)
train_case("both", dropout=0.25, weight_decay=0.05)
```

実行します。

```bash
python regularization_lab.py
```

期待される出力：

```text
regularization_lab
plain          epochs=87  best_epoch=67  train_loss=0.141 val_loss=0.155 train_acc=0.945 val_acc=0.931
dropout        epochs=101 best_epoch=81  train_loss=0.158 val_loss=0.162 train_acc=0.945 val_acc=0.943
weight_decay   epochs=87  best_epoch=67  train_loss=0.141 val_loss=0.154 train_acc=0.948 val_acc=0.931
both           epochs=101 best_epoch=81  train_loss=0.159 val_loss=0.162 train_acc=0.942 val_acc=0.949
```

![正則化実験結果図](/img/course/ch06-regularization-generalization-result-map-ja.webp)

## 結果を読む

plain モデルは training loss が低いです。

```text
plain train_loss=0.141 val_acc=0.931
```

しかし、組み合わせた正則化モデルは validation accuracy が高くなっています。

```text
both train_loss=0.159 val_acc=0.949
```

これが正則化のポイントです。training fit が少し悪くなっても、汎化が良くなるなら価値があります。

## 残す証拠

正則化では、最低の training loss だけを保存しないでください。トレードオフを残します。

```text
単純：train_loss を下げる、validation accuracy を下げる
正則化済み：train_loss はやや高いが、validation accuracy はより良い
判断: validation の挙動が良いモデルを選ぶ
次の確認：dropout か weight_decay を一度に1つずつ変更する
```

ここで大切なのは、深層学習の最適化は「training loss を最小にすること」だけではなく、「将来の性能をより信頼できるものにすること」だと理解することです。

## Dropout

`nn.Dropout(0.25)` は、訓練中に活性化をランダムに落とします。

```python
nn.Linear(2, 32), nn.ReLU(), nn.Dropout(dropout)
```

ネットワークが特定の隠れユニットに依存しすぎないようにします。主に隠れ層で使います。`model.eval()` 中は dropout は自動的に無効になります。

## Weight Decay

Weight decay は optimizer が適用する L2 風の正則化です。

```python
torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.05)
```

大きすぎる重みを抑えます。現代の PyTorch では、weight decay が適応的な勾配更新から分離されるため、古い Adam + L2 より `AdamW` が好まれることが多いです。

## Early Stopping

実験では次を追跡しています。

```text
best_epoch=67
```

Early stopping は、validation が最も良い checkpoint を保持し、validation loss がしばらく改善しなければ停止する考え方です。validation performance が止まったあとに無駄に訓練し続けるのを防ぎます。

## まず試すこと

| 問題 | 最初の行動 |
|---|---|
| training loss 低い、validation loss 高い | weight decay または dropout |
| validation が改善後に悪化する | early stopping |
| train も validation も underfit | 正則化を減らす、モデルを強くする |
| validation が noisy | LR を下げる、データを増やす、fold 平均を見る |

## よくあるトラブル

| 症状 | よくある原因 | 修正 |
|---|---|---|
| dropout が訓練を大きく悪化させる | dropout が高すぎる、またはモデルが小さい | dropout を下げる |
| train も validation も悪い | アンダーフィット | 正則化を減らす |
| best epoch がかなり早い | 訓練しすぎ | best checkpoint を保存する |
| weight decay が効かない | 値が小さい、またはモデルが単純 | 少しずつ増やす |
| eval 結果がランダムに変わる | `model.eval()` を忘れている | validation 前に eval mode にする |

## 練習

1. dropout を `0.1`、`0.5`、`0.7` に変えてください。
2. weight decay を `0.001`、`0.01`、`0.1` に変えてください。
3. 20 epoch ごとに train loss と validation loss を表示してください。
4. `val_loss` が改善したときに best model state を保存してください。
5. validation 時に `model.eval()` を外し、何が変わるか説明してください。

<details>
<summary>参考実装と解説</summary>

1. dropout が小さいと効果は弱く、大きすぎると学習に必要な信号まで落ちます。`0.5` 前後は差を観察しやすい設定です。
2. weight decay を大きくすると重みが抑えられますが、大きすぎると underfitting になります。
3. train loss と validation loss を並べると、過学習、過小適合、学習停滞を区別しやすくなります。
4. best model state は `val_loss` が最小のときに保存します。最後の epoch が最良とは限りません。
5. `model.eval()` を外すと dropout や batch norm が訓練モードのまま動き、検証結果が揺れたり不公平になったりします。

</details>

## 合格チェック

次を説明できれば、この節はクリアです。

- 正則化は training loss だけでなく validation performance を見る；
- dropout は訓練中に隠れ活性化をランダムに無効化する；
- weight decay は大きな重みを抑える；
- early stopping は validation の最良点を保持する；
- 正則化が強すぎるとアンダーフィットになる。

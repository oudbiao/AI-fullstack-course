---
title: "6.7.2 ハイパーパラメータ調整戦略"
description: "learning rate、batch size、正則化、early stopping を、勘ではなく制御された実験として調整します。"
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "hyperparameter tuning, learning rate, batch size, regularization, experiment tracking"
---
:::tip[この節の位置づけ]
ハイパーパラメータ調整は実験設計です。重要な変数を 1 つ変え、記録を残し、validation の証拠を比較して、次の一手を決めます。
:::
## 学習目標

- 何でも同時に変えず、安定した順序で調整できる。
- PyTorch で小さな learning-rate sweep を実行できる。
- validation loss、validation accuracy、訓練の安定性を一緒に読める。
- 再利用できる表で実験 evidence を記録できる。
- learning rate、batch size、正則化、early stopping のどれを次に調整すべきか判断できる。

---

## まずルートを見る

![深層学習の tuning と診断ルート](/img/course/ch06-training-tuning-diagnosis-route-ja.webp)

実践的な順序：

```text
訓練を動かす -> learning rate を調整する -> validation を見る -> overfitting を抑える -> 局所的に細かく詰める
```

最初からすべての knob を回さないでください。有用な調整実験は、1 つの質問に答えるべきです。

| 質問 | まず試すパラメータ | 見るもの |
|---|---|---|
| モデルはそもそも学習するか | learning rate | train loss の傾向 |
| 訓練が不安定か | learning rate、gradient clipping、batch size | spike や divergence |
| validation が training より悪いか | weight decay、dropout、augmentation、early stopping | generalization gap |
| 訓練が遅すぎるか | batch size、model size、precision | 時間とメモリ |
| deploy には重すぎるか | architecture、pruning、quantization | latency と size |

## 実験：Learning-Rate Sweep を走らせる

この toy classification task は小さくてすぐ動きますが、調整の流れを学べます。

`lr_sweep.py` を作成します。

```python
import torch
from torch import nn

torch.manual_seed(11)

X = torch.randn(240, 2)
y = ((X[:, 0] * 0.8 + X[:, 1] * -0.5) > 0).long()

train_x, val_x = X[:180], X[180:]
train_y, val_y = y[:180], y[180:]


def run(lr):
    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(40):
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_loss = loss_fn(model(train_x), train_y).item()
        val_logits = model(val_x)
        val_loss = loss_fn(val_logits, val_y).item()
        val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()

    return train_loss, val_loss, val_acc


results = []
for lr in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    train_loss, val_loss, val_acc = run(lr)
    results.append((lr, train_loss, val_loss, val_acc))

print("lr_sweep")
for lr, train_loss, val_loss, val_acc in results:
    print(
        f"lr={lr:g} "
        f"train_loss={train_loss:.3f} "
        f"val_loss={val_loss:.3f} "
        f"val_acc={val_acc:.3f}"
    )

best = min(results, key=lambda row: row[2])
print("best_lr:", best[0])
```

実行します。

```bash
python lr_sweep.py
```

期待される出力：

```text
lr_sweep
lr=0.001 train_loss=0.763 val_loss=0.733 val_acc=0.450
lr=0.01 train_loss=0.675 val_loss=0.663 val_acc=0.533
lr=0.1 train_loss=0.340 val_loss=0.373 val_acc=0.967
lr=1 train_loss=0.053 val_loss=0.072 val_acc=0.983
lr=10 train_loss=0.280 val_loss=0.291 val_acc=0.883
best_lr: 1.0
```

![LR sweep 出力結果図](/img/course/ch06-lr-sweep-result-map-ja.webp)

読み方：

- `0.001` と `0.01` は、この budget では遅すぎます。
- `0.1` と `1.0` はよく学習しています。
- `10.0` はまだ訓練できますが悪化しているので、大きければよいわけではありません。
- ここでは training loss ではなく validation loss で選びます。

## 次に何を調整するか

![Hyperparameter tuning の探索図](/img/course/hyperparameter-tuning-search-ja.webp)

妥当な learning rate を見つけたら、次の順で進めます。

1. Batch size：メモリ、速度、gradient noise を調整する。
2. Epochs と early stopping：validation が伸びなくなったら止める。
3. Weight decay と dropout：overfitting を抑える。
4. Architecture size：loop が安定してから容量を変える。
5. Optimizer details：必要に応じて betas、scheduler、warmup、momentum を調整する。

ルール：

```text
まず広く探し、あとで近くを細かく詰める
```

## 最小の実験ログ

小さな project でも log を残します。

```text
experiment_id:
code_version:
data_version:
seed:
lr:
batch_size:
optimizer:
weight_decay:
dropout:
epochs:
best_val_metric:
train_time:
decision:
```

decision の例：

```text
quick sweep では lr=1.0 が最も良い validation loss。
次は lr=1.0 を固定し、batch_size=32 と 64 を比較する。
```

## 残す証拠

tuning decision card を 1 つ残します。

```text
質問：どの単一変数がテストされたか？
固定済み: データ分割、seed、モデル、optimizer の種類、学習予算
変更点：学習率の値
選択指標: 検証損失または検証精度
最良設定：速い探索で lr=1.0
次の実験：一度に多くの設定を変えず、ローカルな微調整を1つ行う
```

## 診断パターン

| パターン | ありそうな原因 | 次の実験 |
|---|---|---|
| train loss が動かない | LR が低い、モデルが小さい、label が悪い | LR を上げる、data を確認する、モデルを大きくする |
| train loss が発散する | LR が高い、gradient が不安定 | LR を下げる、clipping を入れる |
| train は良いが validation が悪い | overfitting または leakage | 正則化を足し、split を確認する |
| validation が良くなった後で悪化する | best epoch の後に overfitting | early stopping |
| seed で結果が大きく変わる | 訓練が不安定、または data が少ない | 3 seed で mean/std を出す |

## よくある間違い

| 間違い | 直し方 |
|---|---|
| LR、batch size、optimizer、model を同時に変える | 1 実験で主変数を 1 つに絞る |
| training metric で選ぶ | validation metric で選ぶ |
| runtime を無視する | 時間とメモリも記録する |
| lucky seed を信じる | 重要な run は複数 seed で繰り返す |
| data が汚いまま調整する | label、leakage、preprocessing を先に確認する |

## 練習

1. sweep に `lr=0.3` と `lr=3.0` を追加してください。どちらが良い領域に近いですか。
2. training budget を `40` step から `10` step に変えてください。best LR は変わりますか。
3. 各 LR を 2 つの seed で実行し、`seed` 列を追加してください。
4. LR sweep の後に行う次の実験を 1 文で書いてください。
5. 1 つの実験が 1 つの質問に答える形だと、なぜ調整が簡単になるのか説明してください。

<details>
<summary>参考実装と解説</summary>

1. 多くの場合 `lr=0.3` の方が使える領域に近く、`lr=3.0` は発散や振動を起こしやすいです。最終判断は検証曲線で行います。
2. 予算が短いと、大きめの学習率が良く見えることがあります。長い予算では安定性も重要です。
3. seed を増やすと平均とばらつきを見られます。1 回だけの良い結果に引っ張られにくくなります。
4. 次の実験は具体的に書きます。例: `0.03` から `0.3` の間を細かく調べ、各設定を 2 seed で走らせる。
5. 1 回の実験で 1 つの問いに答えると、結果の原因を追いやすく、記録も再現しやすくなります。

</details>

## まとめ

- 調整は制御された実験設計であり、勘ではありません。
- Learning rate は多くの場合、最初に試す knob です。
- 判断は validation の evidence で行います。
- Log があれば、実験は再現しやすく解釈しやすくなります。
- まず広く調整し、あとで局所的に詰めます。

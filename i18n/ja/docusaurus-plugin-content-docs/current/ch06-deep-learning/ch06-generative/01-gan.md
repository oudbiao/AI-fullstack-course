---
title: "6.6.2 GAN 基礎 [任意]"
sidebar_position: 1
description: "小さな adversarial game を動かしながら GAN を理解し、generator/discriminator の信号、不安定さ、mode collapse を診断します。"
keywords: [GAN, generator, discriminator, adversarial training, mode collapse, PyTorch]
---

# 6.6.2 GAN 基礎 [任意]

:::tip この節の位置づけ
GAN は 2 人のプレイヤーによる学習ループです。generator は本物らしい偽物を作ろうとし、discriminator は本物と偽物を見分けようとします。強さと不安定さは同じ理由から生まれます。両方のプレイヤーが変わり続けるからです。
:::

## 学習目標

- generator と discriminator の役割を説明できる。
- 1D データで最小の PyTorch GAN を動かせる。
- `loss_d`、`loss_g`、`fake_mean`、`fake_std` を訓練信号として読める。
- mode collapse と D/G の不均衡を見つけられる。
- GAN が役立つ場面と、diffusion など他の生成手法を優先すべき場面を区別できる。

---

## まずゲームを見る

![GAN generator discriminator adversarial 図](/img/course/gan-adversarial-loop-ja.webp)

| 部品 | 入力 | 出力 | 目標 |
|---|---|---|---|
| Generator `G` | ランダムノイズ `z` | fake sample | fake を本物らしくする |
| Discriminator `D` | real または fake sample | real/fake score | 本物と偽物を分ける |
| Training loop | `G` と `D` の更新 | 変化し続けるゲーム | 両者を学習させ続ける |

GAN は通常の分類のように、固定されたラベル目標だけで学習するわけではありません。discriminator は「だましにくさ」を変え、generator は fake sample の姿を変えます。

![GAN adversarial training balance と mode collapse 図](/img/course/ch06-gan-adversarial-balance-map-ja.webp)

## 実践ループ

GAN の 1 step は 2 回の更新として読みます。

```text
1. D を訓練する：real -> real, G(z).detach() -> fake
2. G を訓練する：G(z) が D に real と言わせる
```

discriminator 側の step にある `.detach()` は重要です。D を更新するときに、generator まで誤って変わるのを防ぎます。

## 実験：小さな 1D GAN を訓練する

この例は画像を生成しません。中心が `2.0` 付近の 1D 分布で、訓練の仕組みを学ぶためのものです。

`tiny_gan_1d.py` を作成します。

```python
import torch
from torch import nn

torch.manual_seed(7)


def real_batch(n):
    return torch.randn(n, 1) * 0.2 + 2.0


G = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
D = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))

loss_fn = nn.BCEWithLogitsLoss()
opt_g = torch.optim.Adam(G.parameters(), lr=0.01)
opt_d = torch.optim.Adam(D.parameters(), lr=0.01)

for step in range(1, 301):
    real = real_batch(64)
    z = torch.randn(64, 2)
    fake = G(z).detach()

    d_real = D(real)
    d_fake = D(fake)
    loss_d = loss_fn(d_real, torch.ones_like(d_real)) + loss_fn(
        d_fake, torch.zeros_like(d_fake)
    )

    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()

    z = torch.randn(64, 2)
    fake = G(z)
    d_fake = D(fake)
    loss_g = loss_fn(d_fake, torch.ones_like(d_fake))

    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()

    if step in [1, 100, 200, 300]:
        with torch.no_grad():
            sample = G(torch.randn(256, 2))
            print(
                f"step={step:03d} "
                f"loss_d={loss_d.item():.3f} "
                f"loss_g={loss_g.item():.3f} "
                f"fake_mean={sample.mean().item():.3f} "
                f"fake_std={sample.std().item():.3f}"
            )
```

実行します。

```bash
python tiny_gan_1d.py
```

期待される出力：

```text
step=001 loss_d=1.579 loss_g=0.844 fake_mean=0.025 fake_std=0.117
step=100 loss_d=1.287 loss_g=0.654 fake_mean=1.093 fake_std=0.204
step=200 loss_d=1.460 loss_g=0.835 fake_mean=2.988 fake_std=0.291
step=300 loss_d=1.307 loss_g=0.630 fake_mean=1.384 fake_std=0.056
```

![1D GAN 実験結果図](/img/course/ch06-gan-1d-distribution-result-map-ja.webp)

最後の行が一番良い、とは読まないでください。これは診断用の実験です。

- real sample は `2.0` 付近にあります。
- `fake_mean` は動きます。`G` と `D` が追いかけ合うからです。
- `fake_std` がとても小さい場合、多様性不足に注意します。
- GAN の loss は、sample の確認なしでは解釈しにくいです。

## 残す証拠

GAN 実行では loss だけでなく、生成分布のメモを残します。

```text
real_center: about 2.0
step_001: fake_mean=0.025, fake_std=0.117
step_300: fake_mean=1.384, fake_std=0.056
diagnosis: final line is not automatically best
collapse_signal: fake_std becomes very small
review_rule: compare samples/distribution, not loss alone
```

## Mode Collapse とは

Mode collapse は、generator が discriminator をだませる狭いパターンを見つけ、その似た sample ばかりを作る状態です。

画像では、ほぼ同じ向きの顔が何度も出るように見えるかもしれません。1D 実験では、とても小さい `fake_std` が単純な collapse 信号になります。

```text
本物らしく見えるが多様性がない -> mode collapse を疑う
```

## GAN 訓練が難しい理由

| 問題 | 症状 | 最初の対応 |
|---|---|---|
| Discriminator が強すぎる | `G` が有用な feedback を受けにくい | `D` の更新回数や容量を下げる |
| Generator が弱すぎる | fake sample が良くならない | 学習率、構造、正規化を調整する |
| Mode collapse | sample が繰り返しになる | 多様性を監視し、loss や正則化を改善する |
| Loss が紛らわしい | loss は変わるが sample が悪化する | sample grid を保存して比較する |
| 評価が曖昧 | “良さそう”が主観的 | 視覚確認、多様性、タスク指標を組み合わせる |

## GAN を学ぶ価値

GAN は今でも学ぶ価値があります。adversarial learning、distribution matching、失敗診断をとてもはっきり見せてくれるからです。

現代の画像生成プロジェクトでは、diffusion model の方が安定し、制御しやすいことが多いです。それでも GAN は次の理解に役立ちます。

- 訓練後の高速 sampling。
- adversarial realism signal。
- 生成 sample の多様性。
- 不安定な multi-objective training の具体例。

## よくある間違い

| 間違い | 直し方 |
|---|---|
| `loss_g` と `loss_d` だけで判断する | 生成 sample と多様性も確認する |
| `D` の step で `G` も更新してしまう | `G(z).detach()` を使う |
| `D` が早すぎる段階で完璧になる | 容量、更新比率、学習率を調整する |
| 繰り返し出力を無視する | 多様性と mode collapse を追跡する |
| すべての生成タスクで GAN をデフォルトにする | VAE、diffusion、autoregressive method と比較する |

## 練習

1. real data の中心を `2.0` から `-1.0` に変えてください。`fake_mean` は動きますか。
2. `lr` を `0.01` から `0.001` に下げてください。訓練は滑らかになりますか、それとも遅くなりますか。
3. hidden size を `16` から `64` に増やしてください。ゲームは安定しますか。
4. 25 step ごとに `fake_std` を表示し、collapse らしい点を記録してください。
5. GAN の出力品質を 1 つの loss だけで判断できない理由を説明してください。

## まとめ

- GAN 訓練は変化し続ける 2 人ゲームです。
- `G` は sample を作り、`D` は real/fake を判定します。
- 不安定さは単なるコードミスではなく、訓練設定そのものに含まれます。
- Mode collapse は「本物らしいが多様性がない」状態です。
- 本番で別の生成モデルを選ぶ場合でも、GAN は理解する価値があります。

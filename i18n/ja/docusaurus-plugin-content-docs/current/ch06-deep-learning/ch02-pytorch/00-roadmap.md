---
title: "6.2.1 PyTorch ロードマップ：Tensor、Autograd、Module、DataLoader、Loop"
sidebar_position: 0
description: "短い PyTorch ロードマップです。テンソル、自動微分、nn.Module、Dataset/DataLoader、学習ループ、デバッグを扱います。"
keywords: [PyTorch ガイド, tensor, autograd, nn.Module, DataLoader, training loop]
---

# 6.2.1 PyTorch ロードマップ：Tensor、Autograd、Module、DataLoader、Loop

PyTorch は深層学習ループを実行できるコードにします。まず実行順を覚えると、細部が後から楽になります。

## まずワークフローを見る

![PyTorch 章フローチャート](/img/course/ch06-pytorch-chapter-flow-ja.png)

![NumPy から PyTorch への学習ループマップ](/img/course/ch06-numpy-to-pytorch-training-loop-map-ja.png)

```text
tensor -> model -> loss -> backward -> optimizer.step -> repeat
```

## Autograd を一度動かす

`pytorch_first_loop.py` を作り、`torch` をインストールしてから実行します。

```python
import torch

w = torch.tensor([0.0], requires_grad=True)
learning_rate = 0.2

for step in range(1, 5):
    loss = (w - 3).pow(2)
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        w.grad.zero_()
    print(step, "w=", round(w.item(), 3), "loss=", round(loss.item(), 3))
```

出力：

```text
1 w= 1.2 loss= 9.0
2 w= 1.92 loss= 3.24
3 w= 2.352 loss= 1.166
4 w= 2.611 loss= 0.42
```

ここで PyTorch の重要な習慣が見えます。loss を計算し、`backward()` を呼び、勾配追跡なしで更新し、古い勾配を消します。

## この順番で学ぶ

| 順番 | 読む | 練習すること |
|---|---|---|
| 1 | [6.2.2 sklearn から PyTorch へ](./00-sklearn-to-pytorch-bridge.md) | なぜ学習ループが明示的になるか |
| 2 | [6.2.3 PyTorch 基礎](./01-pytorch-basics.md) | tensor、dtype、shape、device |
| 3 | [6.2.4 Autograd](./02-autograd.md) | `requires_grad`、`backward`、`grad` |
| 4 | [6.2.5 nn Module](./03-nn-module.md) | モデルクラス、パラメータ |
| 5 | [6.2.6 データ読み込み](./04-data-loading.md) | Dataset、DataLoader、batch |
| 6 | [6.2.7 学習ループ](./05-training-loop.md) | train/eval ループ、loss ログ |
| 7 | [6.2.8 実践 Tips](./06-practical-tips.md) | shape、device、seed、デバッグ |
| 8 | [6.2.9 PyTorch ワークショップ](./07-pytorch-matplotlib-workshop.md) | 小さなモデルを動かして可視化する |

## 合格ライン

PyTorch ループを読み、データ batch、モデル出力、loss、`backward()`、optimizer 更新の5つを見つけられれば合格です。

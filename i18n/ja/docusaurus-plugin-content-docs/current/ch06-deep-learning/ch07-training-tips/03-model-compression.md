---
title: "6.7.4 モデル圧縮 [任意]"
sidebar_position: 3
description: "デプロイ制約から quantization、pruning、distillation を選び、size、latency、task quality を測り直します。"
keywords: [model compression, quantization, pruning, distillation, deployment, model size]
---

# 6.7.4 モデル圧縮 [任意]

:::tip この節の位置づけ
モデル圧縮はデプロイ上の trade-off であり、魔法の縮小ボタンではありません。memory、latency、throughput、device limit があるから圧縮を検討します。
:::

## 学習目標

- quantization、pruning、distillation を「何を変えるか」で説明できる。
- parameter count と numeric precision から model size を見積もれる。
- 小さな例で quantization error を測れる。
- deployment bottleneck から compression path を選べる。
- model size だけで圧縮を評価しないようになる。

---

## Deployment Bottleneck から始める

![モデル圧縮のトレードオフ図](/img/course/ch06-model-compression-tradeoff-ja.webp)

| Bottleneck | まず考える方法 | 理由 |
|---|---|---|
| memory が大きすぎる | quantization | parameter count は同じでも、1 値あたりの bit を減らせる |
| weight/channel に冗長性がある | pruning | ほとんど貢献しない structure を取り除く |
| 大きな teacher があり再学習できる | distillation | 小さな student に behavior をまねさせる |
| 圧縮後も latency が高い | まず profiling | data transfer や非対応 kernel が bottleneck かもしれない |

重要な習慣：

```text
bottleneck を測る -> 方法を選ぶ -> size, latency, metric を測り直す
```

## 3 つの圧縮ルート

| 方法 | 何を変えるか | よくある benefit | 主な risk |
|---|---|---|---|
| Quantization | numeric precision | memory が小さくなり、推論が速くなることもある | accuracy drop、hardware support 問題 |
| Pruning | weights、channels、blocks | structure が本当に消えれば計算が減る | sparse speedup は全 hardware で出るわけではない |
| Distillation | training objective | teacher に近い小さな model | retraining と teacher outputs が必要 |

圧縮は、圧縮後も task が使える状態で初めて完了です。

## 実験 1：Quantization Error

```python
weights = [0.12, -1.87, 3.44, -0.03]


def fake_quantize(values, scale):
    return [round(v * scale) / scale for v in values]


def mae(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


q8_like = fake_quantize(weights, scale=16)
q4_like = fake_quantize(weights, scale=4)

print("quant_error_lab")
print("original:", weights)
print("q8_like:", q8_like)
print("q4_like:", q4_like)
print("q8_mae:", round(mae(weights, q8_like), 4))
print("q4_mae:", round(mae(weights, q4_like), 4))
```

期待される出力：

```text
quant_error_lab
original: [0.12, -1.87, 3.44, -0.03]
q8_like: [0.125, -1.875, 3.4375, 0.0]
q4_like: [0.0, -1.75, 3.5, 0.0]
q8_mae: 0.0106
q4_mae: 0.0825
```

より強い quantization は、ふつうより大きな numerical error を生みます。実務上の問いは、その後の task metric がまだ許容範囲かどうかです。

## 実験 2：Model Size を見積もる

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

param_count = sum(p.numel() for p in model.parameters())

print("model_size_lab")
print("params:", param_count)

for name, bits in [("fp32", 32), ("fp16", 16), ("int8", 8), ("int4", 4)]:
    mb = param_count * bits / 8 / 1024 / 1024
    print(f"{name:>4}: {mb:.4f} MB")
```

期待される出力：

```text
model_size_lab
params: 8906
fp32: 0.0340 MB
fp16: 0.0170 MB
int8: 0.0085 MB
int4: 0.0042 MB
```

![モデル圧縮の量子化誤差とパラメータサイズ結果図](/img/course/ch06-model-compression-quant-size-result-map-ja.webp)

これは parameter size だけの概算です。実際の deploy size には metadata、tokenizer files、runtime overhead、engine-specific packaging も含まれる場合があります。

## ルートの選び方

| 状況 | 最初の action |
|---|---|
| model が memory に入らない | まず quantization を試す |
| model は入るが latency が高い | pruning 前に latency を profile する |
| channel に冗長性が多そう | structured pruning を考える |
| 小さい model に teacher の behavior を残したい | teacher model から distillation する |
| 圧縮後に metric が落ちすぎる | 圧縮を弱める、または fine-tune する |

pruning では、deploy では structured pruning が扱いやすいことが多いです。channel や block ごと取り除くほうが、random sparse weights より hardware に利用されやすいからです。

distillation のよくある pattern：

```text
teacher logits or outputs -> student learns labels + teacher behavior
```

## 圧縮実験で報告するもの

| Metric | Before | After | なぜ重要か |
|---|---|---|---|
| model size | 必須 | 必須 | memory は改善したか |
| latency | 必須 | 必須 | 推論は本当に速くなったか |
| throughput | あるとよい | あるとよい | service がより多くの request を扱えるか |
| task metric | 必須 | 必須 | quality は許容範囲か |
| hardware/runtime | 必須 | 必須 | compression は deployment stack に依存する |

task metric と latency なしに「int8 が動いた」とだけ報告しないでください。小さいことは、自動的に良いことではありません。

## よくある間違い

| 間違い | 直し方 |
|---|---|
| bottleneck を測る前に圧縮する | 先に memory、latency、metric を測る |
| quantization は必ず速くなると思う | hardware と runtime support を確認する |
| parameter size だけ数える | 必要なら tokenizer、runtime、packaging も含める |
| unstructured pruning で自動的な speedup を期待する | target hardware で benchmark する |
| 圧縮後の accuracy を無視する | task metric を before/after で比較する |

## 練習

1. 実験 1 で `scale=16` を `scale=32` に変えてください。MAE は下がりますか。
2. 実験 2 に 3 つ目の Linear layer を追加し、model size を再計算してください。
3. memory には入るが latency が高すぎる model には、どの compression strategy を選びますか。
4. size、latency、throughput、metric を含む before/after report template を書いてください。
5. structured pruning が unstructured pruning より deploy しやすい理由を説明してください。

## まとめ

- 圧縮は deployment constraints から始まります。
- Quantization は numeric precision を変えます。
- Pruning は model structure を変えます。
- Distillation は training process を変えます。
- 圧縮が成功したと言えるのは、deploy 後の quality と latency が要求を満たすときです。

---
title: "6.7.1 学習 Tips ロードマップ：全部変える前に診断する"
description: "短い深層学習 Tips ロードマップです。チューニング、診断、圧縮、証拠に基づく判断を扱います。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "深層学習 Tips, ハイパーパラメータ調整, 学習診断, モデル圧縮"
---
学習 Tips は、診断に答えるときだけ役立ちます。optimizer、学習率、モデルサイズ、データを同時に変えないでください。

## まず診断フローを見る

![深層学習 Tips 章関係図](/img/course/ch06-training-tips-chapter-flow-ja.webp)

![学習診断ダッシュボードマップ](/img/course/ch06-training-diagnosis-dashboard-map-ja.webp)

| 症状 | 最初に確認すること |
|---|---|
| training loss が高い | モデルが小さすぎる、学習率が低い、データ問題 |
| training は良いが validation が悪い | 過学習、リーク、弱い augmentation |
| loss が不安定 | 学習率が高い、bad batch、勾配爆発 |
| 遅すぎる | batch size、device、モデルサイズ |
| デプロイには重い | 圧縮、量子化、枝刈り |

## 小さな loss ログを読む

`training_tips_first_loop.py` を作ります。

```python
val_loss = [0.62, 0.51, 0.48, 0.49, 0.53]
best_epoch = min(range(len(val_loss)), key=val_loss.__getitem__) + 1

print("best_epoch:", best_epoch)
print("best_val_loss:", val_loss[best_epoch - 1])
print("action: stop or reduce learning rate if validation keeps worsening")
```

出力：

```text
best_epoch: 3
best_val_loss: 0.48
action: stop or reduce learning rate if validation keeps worsening
```

![学習 Tips 最初の loss 出力結果図](/img/course/ch06-training-tips-first-loop-result-map-ja.webp)

工夫を足す前に曲線を読みます。単純なログでも、次に試すことが見える場合が多いです。

## 残す証拠

この小章の終わりには、診断にもとづく意思決定記録を 1 つ残します。

```text
可視症状：曲線や出力は何を示したか？
最初の確認: データ、形状、勾配、または validation split
1つの変更：どの単一設定が変わったか？
前後比較: 指標または成果物の比較
判断: 維持、調整、ロールバック、または調査する
```

目的は、学習の変更を戻せる形にすることです。5 つを同時に変えて良くなっても、どれが効いたのかは分かりません。

## この順番で学ぶ

| 順番 | 読む | 練習すること |
|---|---|---|
| 1 | [6.7.2 ハイパーパラメータ調整](./01-hyperparameter-tuning.md) | 学習率、batch size、optimizer |
| 2 | [6.7.3 学習診断](./02-training-diagnosis.md) | loss 曲線、過学習、不安定さ |
| 3 | [6.7.4 モデル圧縮](./03-model-compression.md) | 小さく、速く、デプロイしやすいモデル |

## 合格ライン

training/validation 曲線を見て、理由付きで次のアクションを1つ選べれば合格です。

<details>
<summary>確認の考え方と解説</summary>

1. 合格レベルの答えでは、tensor、model layer、loss、`backward()`、optimizer update を1つの学習ループとしてつなげます。
2. 証拠には、動く小さな実験、tensor shape の確認、説明できる loss または validation curve を含めます。
3. shape mismatch、loss が下がらない、過学習、data leakage、Attention/Transformer の data flow を説明できない、といった失敗例を1つ言えればよいです。

</details>

---
title: "6.8.4 プロジェクト：生成モデル実践 [任意]"
description: "sample checkpoints、quality notes、diversity checks、failure cases、portfolio presentation を含む generative project review loop を作ります。"
sidebar:
  order: 3
head:
  - tag: meta
    attrs:
      name: keywords
      content: "generative project, GAN, VAE, generation quality, diversity, evaluation"
---
:::tip[この節の位置づけ]
生成 project は、きれいな sample を 1 つ出したら終わりではありません。quality、diversity、stability、failures、そしてなぜその checkpoint を残すのかを示す必要があります。
:::
## 学習目標

- 生成 project の評価が分類と違う理由を説明できる。
- quality と diversity を一緒に追跡できる。
- 小さな checkpoint review table を作れる。
- mode collapse と blurry-output failure を見分けられる。
- generated samples を project evidence としてまとめられる。

---

## まず評価の流れを見る

![Generative model project evaluation loop](/img/course/ch06-project-generative-eval-loop-ja.webp)

```text
学習 -> checkpoint からサンプル生成 -> quality と diversity を確認 -> 失敗例を残す -> 次の一手を選ぶ
```

練習 project では、次のような生成 target を選びます。

- visual inspection できる。
- 訓練または simulation が小さくできる。
- checkpoint 間の比較がしやすい。

digits、icons、simple shapes、小さな grayscale patterns は、open-ended photorealistic generation より最初の project に向いています。

## 実験：チェックポイント評価表

`generative_review_dashboard.py` を作成します。

```python
checkpoints = [
    {"epoch": 1, "quality": 0.20, "diversity": 0.80, "note": "mostly noise"},
    {"epoch": 10, "quality": 0.45, "diversity": 0.72, "note": "outlines appear"},
    {"epoch": 30, "quality": 0.68, "diversity": 0.60, "note": "usable but varied"},
    {"epoch": 60, "quality": 0.75, "diversity": 0.48, "note": "possible collapse"},
]

print("generation_review")
for row in checkpoints:
    status = "candidate" if row["quality"] >= 0.6 and row["diversity"] >= 0.55 else "review"
    print(
        f"epoch={row['epoch']:03d} "
        f"quality={row['quality']:.2f} "
        f"diversity={row['diversity']:.2f} "
        f"status={status}"
    )

selected = max(
    [row for row in checkpoints if row["diversity"] >= 0.55],
    key=lambda row: row["quality"],
)
print("selected_epoch:", selected["epoch"])
```

実行します。

```bash
python generative_review_dashboard.py
```

期待される出力：

```text
generation_review
epoch=001 quality=0.20 diversity=0.80 status=review
epoch=010 quality=0.45 diversity=0.72 status=review
epoch=030 quality=0.68 diversity=0.60 status=candidate
epoch=060 quality=0.75 diversity=0.48 status=review
selected_epoch: 30
```

![生成モデル checkpoint 評価結果図](/img/course/ch06-generative-checkpoint-selection-result-map-ja.webp)

なぜ epoch 60 を選ばないのでしょうか。quality は高いですが diversity が低いからです。良い生成 project は、最もきれいな 1 枚だけを選びません。

## 保存するもの

| 証拠 | 理由 |
|---|---|
| checkpoint ごとのサンプル | 学習の進み方を示す |
| 失敗サンプル | 限界を正直に示す |
| 多様性メモ | 繰り返し出力を見つける |
| 品質メモ | 視覚的な改善を説明する |
| 学習ログ | 安定性や collapse を示す |
| 最終選択ルール | 選択を再現可能にする |

## 品質、多様性、安定性

| 観点 | 良い sign | Warning sign |
|---|---|---|
| Quality | samples が target data らしい | noisy、blurry、broken structure |
| Diversity | samples に意味のある variation がある | repeated outputs または 1 つの style だけ |
| Stability | checkpoints が徐々に改善する | sudden collapse または oscillation |
| Interpretability | failures が記録されている | best samples だけを見せる |

よくある trade-off：

```text
best-looking single sample != best project checkpoint
```

## プロジェクトの拡張ルート

| バージョン | 追加するもの |
|---|---|
| basic | one model、fixed sampling seed、checkpoint samples |
| standard | quality/diversity table と failure samples |
| challenge | VAE、GAN、diffusion-style outputs の比較 |
| portfolio | data、model、samples、failures、次の一手の明確なストーリー |

## 残す証拠

generative project では、最低限この evidence を残します。

```text
チェックポイントサンプル：各 epoch の固定 seed サンプル
品質ノート：何が見た目として改善したか
多様性メモ：出力が繰り返すかどうか
失敗サンプル: ぼやけた、壊れた、崩壊した、または非現実的な出力
選択ルール: このチェックポイントが保持された理由
次の行動：データ、目的、アーキテクチャ、またはサンプリングの変更
```

## よくある間違い

| 間違い | 直し方 |
|---|---|
| best samples だけ見せる | average samples と failure samples も見せる |
| diversity を無視する | repeated outputs や unique patterns を追う |
| checkpoint 比較で条件を揃えない | 同じ fixed seed set を使う |
| dataset が最初から複雑すぎる | 小さな visual target から始める |
| model choice を説明しない | なぜ VAE、GAN、または別手法なのかを書く |

## 練習

1. epoch `90`、quality `0.80`、diversity `0.30` を追加してください。選ぶべきですか。
2. 各 checkpoint に `failure` field を追加してください。
3. 自分の生成 project idea について 4 行の表を書いてください。
4. checkpoint table を使って mode collapse を説明してください。
5. 「なぜこの checkpoint を選んだのか」というポートフォリオ小節を下書きしてください。

<details>
<summary>プロジェクト参考とレビュー観点</summary>

1. 通常は選びません。ただし、project が quality を diversity より極端に重視するなら例外はあります。`0.30` の diversity は、出力が反復的または狭い範囲に偏る警告です。
2. `failure` field には、反復、アーティファクト、prompt mismatch、安全でない出力、diversity 不足など、見える問題を書きます。
3. 役立つ表には、idea、data/source、evaluation signal、main risk の 4 行を入れます。その表で、生成 project を評価できるか判断できるようにします。
4. Mode collapse は、モデルが少数の似た出力ばかり出す状態です。checkpoint table では、quality は悪くないのに diversity が低い状態として見えます。
5. ポートフォリオ小節では、quality、diversity、failure notes、sample outputs、採用しなかった checkpoint の弱点を根拠にして選定理由を書きます。

</details>

## まとめ

- Generative project には gallery ではなく評価ストーリーが必要です。
- Quality と diversity は一緒に読みます。
- Failure samples はプロジェクトをより信頼できるものにします。
- 明確な checkpoint selection rule も成果物の一部です。

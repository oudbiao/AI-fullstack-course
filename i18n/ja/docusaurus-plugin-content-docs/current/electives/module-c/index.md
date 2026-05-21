---
title: "E.C 古典的 ML ロードマップ"
sidebar_position: 0
description: "古典的機械学習の補足ロードマップ。SVM、KNN、ナイーブベイズ、LDA を、小中規模データの強いベースラインとして使います。"
---

# E.C 古典的 ML ロードマップ

データセットが小さい、特徴量がはっきりしている、重いモデルを試す前に強いベースラインがほしい。そんなときに使う選択モジュールです。

## まずベースラインの地図を見る

![古典的 ML 補足アルゴリズムモジュールマップ](/img/course/elective-classic-ml-module-map-ja.webp)

![KNN の近傍投票図](/img/course/elective-knn-neighbor-voting-ja.webp)

古典的 ML は、「この問題はシンプルな特徴量だけで解けるのか」を先に確かめる助けになります。

## 最小の KNN ベースラインを動かす

```python
def distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

train = [
    ([0.1, 0.2], "low"),
    ([0.2, 0.1], "low"),
    ([0.8, 0.9], "high"),
    ([0.9, 0.8], "high"),
]

point = [0.75, 0.85]
nearest = min(train, key=lambda row: distance(row[0], point))
print("prediction:", nearest[1])
print("neighbor:", nearest[0])
```

期待される出力：

```text
prediction: high
neighbor: [0.8, 0.9]
```

これはベースライン作成の最小習慣です。特徴量を決め、距離を比べ、予測し、その結果を後で比較できるように残します。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
model_family: SVM, KNN, Naive Bayes, LDA, or another classical baseline
dataset_view: feature scale, class balance, decision boundary, and train/test split
metric: accuracy/F1, confusion matrix, margin, neighbor behavior, or projection quality
failure_check: scaling, high dimensionality, weak assumptions, leakage, or poor baseline fit
Expected_output: classical-ML baseline result with one limitation note
```

## この順番で学ぶ

| ステップ | レッスン | 実践で残す成果 |
|---|---|---|
| 1 | [E.C.1 SVM](./01-svm.md) | マージン、サポートベクトル、`C`、カーネル選択を説明する |
| 2 | [E.C.2 KNN](./02-knn.md) | 距離と投票によるベースラインを作る |
| 3 | [E.C.3 ナイーブベイズ](./03-naive-bayes.md) | 証拠の件数をクラス確率に変換する |
| 4 | [E.C.4 LDA](./04-lda.md) | 特徴量を投影してクラスを分ける |

## 合格チェック

古典的なベースラインを 1 つ作り、それがなぜ適切か説明し、より重いモデルまたは後続プロジェクトの結果と比較できれば合格です。

<details>
<summary>参考解答と解説</summary>

合格する答えは、なぜこのベースラインが適切なのかを先に説明します。たとえば、データが小さい、特徴量が明確、距離や境界に意味がある、といった理由です。そのうえで、より重いモデルや後続プロジェクトの結果と比較し、どこに限界があるかを述べます。

単に accuracy を 1 つ報告するだけでは足りません。古典的 ML の価値は、速く、説明しやすく、比較しやすい基準を作ることです。

</details>

---
title: "5.1.1 機械学習基礎ロードマップ：タスク、データ、モデル、スコア"
sidebar_position: 0
description: "短い機械学習基礎ロードマップです。タスク種別、データ分割、fit/predict/score、baseline、sklearn の流れを扱います。"
keywords: [機械学習ガイド, ML 入門, sklearn ガイド, 教師あり学習, 教師なし学習]
---

# 5.1.1 機械学習基礎ロードマップ：タスク、データ、モデル、スコア

機械学習は、すべてのルールを手書きせず、モデルにデータからパターンを学ばせるところから始まります。最初の習慣はアルゴリズム暗記ではなく、小さなプロジェクトループです。

## 5.1.1.1 まずマップを見る

![機械学習基礎学習マップ](/img/course/ml-basics-roadmap-ja.png)

![機械学習基礎章フロー](/img/course/ch05-basics-chapter-flow-ja.png)

このループを覚えます。

```text
タスク定義 -> データ分割 -> モデル学習 -> 予測 -> 評価 -> 次の判断
```

| 用語 | 最初の意味 |
|---|---|
| feature | モデルが使う入力列 |
| label / target | モデルが予測する答え |
| train set | 学習に使うデータ |
| test set | 汎化を確認するために取っておくデータ |
| baseline | 比較用のシンプルな最初のモデル |

## 5.1.1.2 最小 sklearn ループを動かす

`ml_first_loop.py` を作り、`scikit-learn` をインストールしてから実行します。

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("task: classification")
print("test_accuracy:", round(model.score(X_test, y_test), 3))
print("prediction_count:", len(predictions))
```

出力：

```text
task: classification
test_accuracy: 0.967
prediction_count: 30
```

これが最小の有用な機械学習ループです。先に分割し、学習データだけで学習し、テストデータで評価します。

## 5.1.1.3 この順番で学ぶ

| 順番 | 読む | 練習すること |
|---|---|---|
| 1 | [5.1.2 機械学習とは](./01-what-is-ml.md) | タスク種別、特徴量、ラベル |
| 2 | [5.1.3 Scikit-learn 入門](./02-sklearn-intro.md) | `fit`、`predict`、`score` |
| 3 | [5.1.4 数学が機械学習へ入る流れ](./03-math-to-ml-bridge.md) | ベクトル、確率、loss、最適化 |
| 4 | [5.1.5 機械学習の発展史](./04-history-breakthroughs.md) | 主なアルゴリズムが生まれた理由 |
| 5 | [5.1.6 sklearn と Matplotlib ワークショップ](./05-sklearn-matplotlib-workshop.md) | 実行、可視化、baseline の説明 |

## 5.1.1.4 合格ライン

タスク種別を言える、`X` と `y` を識別できる、train/test 分割の理由を説明できる、baseline スコアを証拠として残せるなら合格です。

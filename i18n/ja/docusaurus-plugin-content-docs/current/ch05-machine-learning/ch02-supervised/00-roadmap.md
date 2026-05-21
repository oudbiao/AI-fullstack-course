---
title: "5.2.1 教師あり学習ロードマップ：ラベル付き例から学ぶ"
sidebar_position: 2
description: "短い教師あり学習ロードマップです。回帰、分類、木モデル、アンサンブル、SVM、モデル選択を扱います。"
keywords: [教師あり学習ガイド, 線形回帰, ロジスティック回帰, 決定木, アンサンブル学習]
---

# 5.2.1 教師あり学習ロードマップ：ラベル付き例から学ぶ

教師あり学習は、ラベル付きの例があるとき、新しい例のラベルを予測するモデルをどう学ぶかを扱います。

## まずモデル選択マップを見る

![教師あり学習ロードマップ](/img/course/supervised-learning-roadmap-ja.webp)

![教師あり学習章フロー](/img/course/ch05-supervised-chapter-flow-ja.webp)

| モデル系統 | 最初の用途 |
|---|---|
| 線形回帰 | 連続値を予測する |
| ロジスティック回帰 | シンプルな確率モデルで分類する |
| 決定木 | 読みやすいルールでデータを分ける |
| アンサンブルモデル | 多くのモデルを組み合わせ、表データの強い baseline を作る |
| SVM | マージンの直感で安定した境界を学ぶ |

## 回帰 baseline を1つ動かす

`supervised_first_loop.py` を作り、`scikit-learn` をインストールしてから実行します。

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_test)

print("task: regression")
print("r2:", round(r2_score(y_test, predictions), 3))
print("first_prediction:", round(predictions[0], 1))
```

出力：

```text
task: regression
r2: 0.485
first_prediction: 137.9
```

スコアが完璧でなくても価値があります。baseline は、後のモデルや特徴量改善がどこを超えるべきかを教えてくれます。

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
タスク：target 定義のある regression または classification 問題
モデル：線形/ロジスティック/木/アンサンブル/SVM の構成と train/test 分割
指標：回帰誤差、accuracy/F1、閾値曲線、または confusion matrix
失敗確認: 過学習、学習不足、特徴量スケーリング、閾値選択、またはクラス不均衡
期待される成果: モデル結果とエラーサンプル、または残差レビュー
```

## この順番で学ぶ

| 順番 | 読む | 比較すること |
|---|---|---|
| 1 | [5.2.2 線形回帰](./01-linear-regression.md) | シンプルな数値予測 |
| 2 | [5.2.3 ロジスティック回帰](./02-logistic-regression.md) | 分類確率 |
| 3 | [5.2.4 決定木](./03-decision-trees.md) | ルール、非線形、過学習 |
| 4 | [5.2.5 アンサンブル学習](./04-ensemble-learning.md) | bagging、boosting、強い表データモデル |
| 5 | [5.2.6 サポートベクターマシン](./05-svm.md) | マージン、境界、古典的分類器の直感 |

## 合格ライン

ラベル付きタスクが回帰か分類かを判断でき、baseline を1つ動かし、モデルが失敗しそうな理由を1つ説明できれば合格です。

<details>
<summary>確認の考え方と解説</summary>

1. ラベルが連続値なら回帰から始めます。ラベルがクラスなら分類から始めます。
2. baseline は単純な線形/ロジスティックモデルでも、dummy ルールでもかまいません。複雑なモデルが超えるべき基準点を作ることが目的です。
3. よくある失敗理由は、弱い特徴量、target リーク、クラス不均衡、スケーリング不足、過学習、実目的と合わない指標です。

</details>

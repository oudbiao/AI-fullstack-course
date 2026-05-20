---
title: "5.4.4 バイアス・バリアンスのトレードオフ"
sidebar_position: 4
description: "手を動かして学ぶバイアス・バリアンス：アンダーフィット、オーバーフィット、モデル複雑度、train-test gap、学習曲線、実用的な対処"
keywords: [バイアス, バリアンス, アンダーフィット, オーバーフィット, 学習曲線, 検証曲線, モデル複雑度]
---

# 5.4.4 バイアス・バリアンスのトレードオフ

![偏差方差トレードオフ三連図](/img/course/bias-variance-tradeoff-ja.webp)

:::tip この節の概要
バイアスとバリアンスは理論用語だけではありません。モデルが単純すぎるのか、不安定すぎるのか、データ品質に制限されているのかを診断する道具です。
:::

## 作るもの

この節では、決定木を使って次を確認します。

- モデル複雑度が train/test score をどう変えるか；
- train-test gap からアンダーフィットとオーバーフィットを見分ける方法；
- 学習曲線で、データ追加が効きそうかどうかを見る方法；
- high bias と high variance で取るべき行動。

![偏差方差アクション診断図](/img/course/ch05-bias-variance-action-map-ja.webp)

## セットアップ

```bash
python -m pip install -U scikit-learn numpy
```

## 完全な実験を実行する

`bias_variance_lab.py` を作成します。

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.tree import DecisionTreeClassifier


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("complexity_lab")
for depth in [1, 3, 5, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    gap = train_acc - test_acc
    print(
        f"max_depth={str(depth):<4} "
        f"train={train_acc:.3f} test={test_acc:.3f} gap={gap:.3f} "
        f"leaves={model.get_n_leaves()}"
    )

print("learning_curve_lab")
model = DecisionTreeClassifier(max_depth=3, random_state=42)
train_sizes, train_scores, val_scores = learning_curve(
    model,
    X,
    y,
    cv=5,
    train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    scoring="accuracy",
)
for size, train_mean, val_mean in zip(train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
    print(f"train_size={size:<3} train={train_mean:.3f} cv={val_mean:.3f} gap={train_mean - val_mean:.3f}")
```

実行します。

```bash
python bias_variance_lab.py
```

期待される出力：

```text
complexity_lab
max_depth=1    train=0.923 test=0.923 gap=-0.001 leaves=2
max_depth=3    train=0.977 test=0.944 gap=0.032 leaves=7
max_depth=5    train=0.995 test=0.937 gap=0.058 leaves=15
max_depth=None train=1.000 test=0.923 gap=0.077 leaves=18
learning_curve_lab
train_size=91  train=0.989 cv=0.847 gap=0.142
train_size=182 train=0.986 cv=0.870 gap=0.116
train_size=273 train=0.978 cv=0.903 gap=0.075
train_size=364 train=0.975 cv=0.917 gap=0.057
train_size=455 train=0.974 cv=0.919 gap=0.055
```

![バイアス・バリアンス実験結果図](/img/course/ch05-bias-variance-result-map-ja.webp)

## 複雑度の実験を読む

`max_depth` が大きいほど、木は複雑になります。

```text
max_depth=1    train=0.923 test=0.923 gap=-0.001 leaves=2
max_depth=None train=1.000 test=0.923 gap=0.077 leaves=18
```

`max_depth=1` は単純です。train と test は近いですが、スコアは最高ではありません。これは high bias、つまりモデルが単純すぎる状態かもしれません。

`max_depth=None` は訓練データを完全に覚えますが、test accuracy は下がります。これは high variance、つまり訓練の細部に合わせすぎて汎化できない状態です。

実用上は中間がよく効くことが多いです。

```text
max_depth=3 train=0.977 test=0.944 gap=0.032
```

訓練データで満点ではありませんが、よりよく汎化しています。

## 学習曲線

![学習曲線診断図](/img/course/ch05-learning-curve-diagnosis-map-ja.webp)

学習曲線は、訓練データが増えると何が起きるかを示します。

```text
train_size=91  train=0.989 cv=0.847 gap=0.142
train_size=455 train=0.974 cv=0.919 gap=0.055
```

データが増えると、検証スコアが上がり、gap が小さくなっています。これは、追加データが役立つ可能性を示します。ただし、特徴量や調整による改善余地も残っています。

## 診断ルール

| パターン | ありそうな問題 | 試すこと |
|---|---|---|
| train 低い、validation 低い | high bias / アンダーフィット | より表現力のあるモデル、良い特徴量、弱い正則化 |
| train 高い、validation 低い | high variance / オーバーフィット | モデルを単純にする、正則化を強める、データを増やす |
| train 高い、validation 高い | 良いフィット | final holdout で確認し、ドリフトを監視 |
| validation が fold ごとに大きく揺れる | 不安定またはセグメント差 | fold を調べ、データ追加や頑健なモデルを検討 |

1 つの指標だけで診断しないでください。train score、validation score、gap、そしてミスが特定セグメントに集中していないかを見ます。

## 実用的な対処

high bias のとき：

- 有用な特徴量を追加する；
- より表現力のあるモデルを使う；
- 強すぎる正則化を弱める；
- 反復型モデルなら学習を長くする。

high variance のとき：

- モデル複雑度を下げる；
- 正則化を強める；
- より多様で代表的なデータを集める；
- クロスバリデーションと final holdout を使う；
- バリアンスを下げる集成モデルを検討する。

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
evaluation_setup: split, cross-validation, metric, baseline, and comparison target
result: score table, curve, confusion matrix, validation result, or search outcome
decision: whether to change data, features, model, threshold, or hyperparameters
failure_check: leakage, unstable validation, wrong metric, or tuning on the test set
Expected_output: evaluation record that supports a next modeling decision
```

## よくあるトラブル

| 症状 | よくある原因 | 修正 |
|---|---|---|
| train も validation も悪い | モデルがパターンを表現できない | 特徴量またはモデルクラスを改善する |
| train は満点、validation は悪い | オーバーフィット | 深さ制限、枝刈り、正則化 |
| データ追加で validation が上がる | バリアンスまたはデータ不足 | より代表的なデータを集める |
| データ追加が効かない | high bias またはラベルノイズ | 特徴量、ラベル、モデルを改善する |
| validation が fold ごとに跳ねる | データが不均一 | セグメント分布を確認する |

## 練習

1. 木に `min_samples_leaf=5` を追加してください。gap はどう変わりますか？
2. `max_depth=2, 4, 6, 8` を試してください。test accuracy はどこで最大になりますか？
3. 木をロジスティック回帰に置き換えてください。問題は bias と variance のどちらに近いですか？
4. 複雑度実験を 1 回の test split ではなく 5-fold CV に変えてください。
5. 最良の木の誤分類を確認してください。特定クラスに集中していますか？

<details>
<summary>参考解答と解説</summary>

1. `min_samples_leaf=5` は小さな葉を作りにくくするため、train/test gap は縮まりやすいです。両方のスコアが下がるなら単純にしすぎています。
2. test accuracy は中間の深さで最大になることが多いです。浅すぎると underfitting、深すぎると訓練 accuracy だけ上がって overfitting します。
3. ロジスティック回帰は bias の確認に使えます。木だけ訓練で高くテストで悪いなら variance、両方低いなら特徴量やモデル族が弱い可能性があります。
4. 5-fold CV は 1 回の分割より安定して複雑度を比較できます。平均スコアが高く、ばらつきも許容できる深さを選びます。
5. 誤分類が特定クラスに集中するなら、そのクラスの特徴不足、ラベル曖昧さ、クラス不均衡を疑います。散らばっているなら残るノイズの影響かもしれません。

</details>

## 合格チェック

次を説明できれば、この節はクリアです。

- high bias は、モデルが単純すぎるか信号が足りない状態；
- high variance は、モデルが訓練の細部に敏感すぎる状態；
- train-validation gap は実用的な診断手がかり；
- 学習曲線は追加データが効くかを示す；
- 対処は用語ではなく、観察されたパターンで決める。

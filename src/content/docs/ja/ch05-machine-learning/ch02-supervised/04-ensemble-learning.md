---
title: "5.2.5 アンサンブル学習：Forest、Boosting、Stacking"
description: "短いアンサンブル学習ハンズオン。単一木、Random Forest、Gradient Boosting、リークを避けた Stacking を比較します。"
sidebar:
  order: 6
head:
  - tag: meta
    attrs:
      name: keywords
      content: "アンサンブル学習, Random Forest, Bagging, Boosting, GBDT, Stacking, XGBoost, LightGBM, CatBoost"
---
![Bagging と Boosting の比較図](/img/course/ch05-ensemble-bagging-boosting-flow-ja.webp)

アンサンブル学習は複数のモデルを組み合わせ、1つのモデルの弱点が最終予測を支配しにくくします。表形式データでは、古典的機械学習の中でも特に強い手法群です。

## まず二つの主線を見る

![アンサンブル学習ファミリー漫画](/img/course/ch05-ensemble-family-comic-ja.webp)

最初からモデル名を暗記しないでください。まず二つの考え方を分けます。

| ルート | イメージ | 代表モデル | 主な利点 | 主なリスク |
|---|---|---|---|---|
| Bagging | 複数モデルを並列に学習して投票 | Random Forest | 安定し、分散を下げる | 大きくなり、説明しづらい |
| Boosting | 後続モデルが前の誤りを補正 | GBDT、XGBoost、LightGBM、CatBoost | 精度が強い | 制御しないと過学習しやすい |
| Stacking | 基本モデルの予測をメタモデルへ渡す | `StackingClassifier` | 異なるモデル群を組み合わせる | 交差検証なしだとリークする |

## 比較実験を動かす

`ch05_ensemble_lab.py` を作成します。

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.25,
    random_state=42,
    stratify=data.target,
)

models = {
    "single_tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    ),
    "gradient_boost": GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    ),
}

models["stacking_cv"] = StackingClassifier(
    estimators=[
        ("rf", models["random_forest"]),
        ("gb", models["gradient_boost"]),
        ("lr", make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        )),
    ],
    final_estimator=LogisticRegression(max_iter=2000, random_state=42),
    cv=5,
)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name:<15} accuracy={accuracy_score(y_test, pred):.3f} f1={f1_score(y_test, pred):.3f}")

rf = models["random_forest"]
importances = rf.feature_importances_
top = importances.argsort()[-3:][::-1]
print("top_rf_features=")
for idx in top:
    print(f"- {data.feature_names[idx]}: {importances[idx]:.3f}")
```

実行します。

```bash
python ch05_ensemble_lab.py
```

期待される出力：

```text
single_tree     accuracy=0.944 f1=0.956
random_forest   accuracy=0.958 f1=0.967
gradient_boost  accuracy=0.944 f1=0.956
stacking_cv     accuracy=0.986 f1=0.989
top_rf_features=
- worst perimeter: 0.146
- worst area: 0.140
- worst concave points: 0.109
```

![アンサンブル学習実験結果図](/img/course/ch05-ensemble-comparison-result-map-ja.webp)

sklearn のバージョンによりスコアが少し変わることがあります。比較表と重要特徴量をプロジェクト証拠として残します。

## 結果を読む

単一木は baseline です。Random Forest は多くの異なる木を平均するため、たいてい安定します。

Boosting は小さなデータセットで常に勝つわけではありません。木の深さ、learning rate、木の本数、検証性能を制御する必要があります。

Stacking は異なるモデル群を組み合わせるため、この例では勝つことがあります。ただし、交差検証が必須です。メタモデルが基モデルの学習行に対する予測を直接見ると、情報リークになります。

## Bagging：Random Forest

![アンサンブル学習の投票と森の図](/img/course/ensemble-learning-voting-forest-ja.webp)

Random Forest はランダム化されたデータ視点で多くの決定木を学習し、その予測を平均または投票します。

最初に見る設定：

| パラメータ | 何を制御するか | 初心者の目安 |
|---|---|---|
| `n_estimators` | 木の数 | `100` から `300` |
| `max_depth` | 木の深さ | 小さく始めて増やす |
| `min_samples_leaf` | 葉に必要な最小サンプル数 | 過学習時に増やす |
| `random_state` | 再現性 | 学習中は必ず設定 |

## Boosting：GBDT とツール群

![GBDT 残差補正漫画](/img/course/ch05-ensemble-gbdt-residual-correction-ja.webp)

Boosting は順番にモデルを作ります。

```text
最初の小さな木 -> 誤りを見る -> 次の小さな木が誤りに集中 -> 繰り返す
```

sklearn では、まず `GradientBoostingClassifier` または `HistGradientBoostingClassifier` から始めます。実際の表形式プロジェクトでは XGBoost、LightGBM、CatBoost もよく使いますが、sklearn baseline が見える前に外部ライブラリへ飛ばないでください。

![Boosting ツール選択漫画](/img/course/ch05-ensemble-boosting-toolkit-ja.webp)

Boosting の最初の調整順：

| 手順 | 変えるもの | 理由 |
|---|---|---|
| 1 | `learning_rate` と `n_estimators` | 歩幅と学習ラウンド数を制御 |
| 2 | `max_depth` / leaf 設定 | 複雑さを制御 |
| 3 | 検証または early stopping | 過学習を止める |
| 4 | 特徴量前処理 | 信号品質を上げる |

## Stacking を安全に使う

![リークを避ける Stacking ワークフロー漫画](/img/course/ch05-ensemble-stacking-leakage-safe-ja.webp)

Stacking が信頼できるのは、メタモデルが out-of-fold 予測を見る場合です。

```text
CV fold 内で基モデルを学習 -> out-of-fold 予測を集める -> メタモデルを学習 -> holdout test で評価
```

手作業で学習行の予測をそのままメタモデルに渡すのではなく、sklearn の `StackingClassifier(cv=5)` を優先します。

## モデルをどう選ぶか

| 状況 | まず使う |
|---|---|
| 強く安定した baseline がほしい | Random Forest |
| 表形式データに非線形が多い | Gradient Boosting / XGBoost / LightGBM |
| カテゴリ特徴量が多い | baseline 後に CatBoost |
| 複数のモデル群が補完し合う | 交差検証つき Stacking |
| 説明しやすさを優先 | 浅い木または Random Forest の特徴量重要度 |

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
タスク：target 定義のある regression または classification 問題
モデル：線形/ロジスティック/木/アンサンブル/SVM の構成と train/test 分割
指標：回帰誤差、accuracy/F1、閾値曲線、または confusion matrix
失敗確認: 過学習、学習不足、特徴量スケーリング、閾値選択、またはクラス不均衡
期待される成果: モデル結果とエラーサンプル、または残差レビュー
```

## よくある失敗

| 症状 | 最初に確認 | よくある修正 |
|---|---|---|
| アンサンブルが単一木とあまり変わらない | 特徴量が弱い、分割が不安定 | 特徴量追加、交差検証 |
| 学習は良いがテストが悪い | 過学習 | 深さを下げ、leaf を増やし、検証を入れる |
| Boosting が木を増やすほど悪化 | ラウンド数が多すぎる | learning rate を下げる、early stopping |
| Stacking が異常に完璧 | 情報リーク | out-of-fold 予測または `StackingClassifier(cv=...)` |
| 特徴量重要度を読みすぎる | 特徴量が相関している | permutation importance や ablation で確認 |

## 練習

1. Random Forest の `max_depth` を `6` から `3` と `None` に変える。
2. Gradient Boosting の `learning_rate` を `0.05` から `0.2` に変える。
3. Stacking が交差検証なしだとなぜリークするか説明する。
4. モデル比較表を保存し、最初に本番へ出すモデルを一段落で説明する。

<details>
<summary>参考実装と解説</summary>

1. `max_depth=3` は各木を単純にし、安定しやすい一方で underfitting の可能性があります。`None` は深い木を許すため訓練スコアは上がりやすいですが、テスト性能と安定性を確認する必要があります。
2. `learning_rate=0.2` は各 Boosting ラウンドの修正を大きくします。早く改善することもありますが、過学習も早く進むため、検証データや交差検証で判断します。
3. Stacking で二段目のモデルが、基モデルが自分の訓練データ上で出した予測を見ると、楽観的な信号を学んでしまいます。out-of-fold 予測や `StackingClassifier(cv=...)` を使う理由はここにあります。
4. 本番候補は最高スコアだけで選びません。テスト性能の安定性、学習/推論コスト、説明しやすさ、失敗時の影響を含めて理由を書きます。

</details>

## 通過チェック

次を説明できれば先へ進めます。

- Bagging と Boosting の違い。
- Random Forest が単一木より安定しやすい理由。
- Boosting に検証制御が必要な理由。
- Stacking に交差検証が必須な理由。
- leaderboard 最高スコアが、常に本番最適とは限らない理由。

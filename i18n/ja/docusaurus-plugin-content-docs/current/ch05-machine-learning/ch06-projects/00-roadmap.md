---
title: "5.6.1 機械学習プロジェクトロードマップ：baseline、証拠、改善"
sidebar_position: 18
description: "短い機械学習プロジェクトロードマップです。問題定義、baseline、評価、改善、失敗分析、証拠整理を扱います。"
keywords: [機械学習プロジェクトガイド, 住宅価格予測, 顧客離脱, ユーザーセグメンテーション, Kaggle, 機械学習ポートフォリオ]
---

# 5.6.1 機械学習プロジェクトロードマップ：baseline、証拠、改善

この小章は第5章の出口です。データ問題を、評価でき、説明でき、ポートフォリオに見せられるモデリングフローへ変えられることを確認します。

## まずプロジェクトループを見る

![機械学習プロジェクト実践ロードマップ](/img/course/ml-projects-roadmap-ja.webp)

![機械学習プロジェクトポートフォリオループ](/img/course/ch05-projects-portfolio-loop-ja.webp)

このプロジェクトループを覚えます。

```text
問題 -> データ -> baseline -> 指標 -> 改善 -> 失敗例 -> レポート
```

いきなり複雑なモデルに進まないでください。ベースライン、指標、失敗分析がないプロジェクトは、ただのデモ実行になりがちです。

## 実験ログを1つ残す

`ml_project_log_first_loop.py` を作ります。これはモデルではなく、すべてのモデルプロジェクトに必要な習慣です。

```python
experiments = [
    {"version": "v1_baseline", "metric": 0.72, "change": "default model"},
    {"version": "v2_features", "metric": 0.78, "change": "add ratio features"},
    {"version": "v3_tuned", "metric": 0.80, "change": "tune max_depth"},
]

best = max(experiments, key=lambda row: row["metric"])

print("best_version:", best["version"])
print("best_metric:", best["metric"])
print("next_step: inspect failure cases before adding more models")
```

出力：

```text
best_version: v3_tuned
best_metric: 0.8
next_step: inspect failure cases before adding more models
```

ここでの変化は、「モデルを動かした」から「バージョンを比較し、次の一手を説明できる」へ移ることです。

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
project_goal: prediction, segmentation, Kaggle, or end-to-end ML portfolio target
pipeline: data split, preprocessing, model, evaluation, and report artifacts
result: metric table, chart, predictions, failure samples, and README note
failure_check: non-reproducible run, leakage, overfitting, weak baseline, or missing deployment boundary
Expected_output: ML project folder with pipeline, metrics, and failure review
```

## この順番で学ぶ

| 順番 | 読む | 提出するもの |
|---|---|---|
| 1 | [5.6.2 住宅価格予測](./01-house-price.md) | 回帰 baseline と改善 |
| 2 | [5.6.3 顧客離脱予測](./02-customer-churn.md) | 分類指標としきい値の考え方 |
| 3 | [5.6.4 ユーザーセグメンテーション](./03-user-segmentation.md) | クラスタ解釈と業務ラベル |
| 4 | [5.6.5 Kaggle 実践](./04-kaggle.md) | 実際の提出フロー |
| 5 | [5.6.6 ML 実践ワークショップ](./05-hands-on-ml-workshop.md) | 完全な証拠パックの練習 |

ワークショップを最後に置くのは、前のプロジェクト習慣を再現可能な証拠パックにまとめるためです。

## プロジェクト成果物基準

![機械学習プロジェクトレポートストーリーボード](/img/course/ch05-project-report-storyboard-ja.webp)

少なくとも1つのプロジェクトで、`README.md`、実行コマンド、指標表、実験ログ、失敗例1つ、グラフ1つ、次の改善案を残します。

## 合格ライン

タスクをどう定義したか、どの baseline を使ったか、どの指標を信頼したか、何が改善したか、どこで失敗したか、次に何をするかを説明できれば合格です。

<details>
<summary>参考解答と解説</summary>

1. 完全な答えでは、モデル名より先にタスク種類、目的変数、成功指標を定義します。
2. baseline は、固定 split、最小限の前処理、1 つのモデル、1 つの指標表からなる、最も単純で再現可能な版です。
3. 改善は同じ split または同じ検証方法で比べたときだけ信頼できます。split とモデルを同時に変えると、何が効いたか説明しにくくなります。
4. 失敗分析では、モデルが弱いサンプル種類やセグメントを少なくとも 1 つ挙げ、それを次の制御された実験に変えます。
5. 合格するプロジェクトフォルダには、実行コマンド、README、実験ログ、指標表、グラフ、失敗例、次の改善案が含まれます。

</details>

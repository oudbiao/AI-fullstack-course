---
title: "3.0 学習ガイドとタスクリスト：データ分析と可視化"
sidebar_position: 1
description: "第 3 章の主な学習ルートは章の入口ページへ統合済みです。このページは短い印刷用チェックリストです。"
keywords: [データ分析学習ガイド, データ分析タスクリスト, NumPy, Pandas, 可視化]
---

# 3.0 学習ガイドとタスクリスト：データ分析と可視化

![データ分析学習ガイドの最小ループ](/img/course/ch03-study-guide-data-loop-vertical-ja.webp)

主な学習ルートは [第 3 章の入口](./) にまとめました。このページは、練習中に見る短いチェックリストとして使います。

## 一行モデル

```text
読み込む -> 確認する -> 整える -> 集計する -> 可視化する -> 説明する
```

グラフを一文で説明できないなら、データの質問に戻ります。

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
data_source: raw records or small dataset used
processing_step: pure Python, NumPy, Pandas, charting, or SQL operation
output: cleaned data, statistic, chart, query result, or report note
failure_check: missing data, shape mismatch, wrong aggregation, or unclear question
Expected_output: data artifact plus the evidence needed to trust it
```

## 練習チェックリスト

| チェック | 証拠 |
|---|---|
| 行、列、型、欠損値を確認できる | `df.info()` と欠損メモ |
| 重複、欠損、明らかな外れ値を処理できる | クリーニングログ |
| `groupby` で質問に答えられる | 集計表 |
| 具体的な質問に合うグラフを選べる | 3 つのグラフファイル |
| 結論と限界を書ける | `report.md` |
| 再現可能なワークショップを完了できる | `ch03_output/` |


<details>
<summary>参考解答と解説</summary>

- このチェックリストは最終的な証拠監査として使います。各プロジェクトで、生ファイル、クリーン済みファイルまたはクリーニングスクリプト、要約表、グラフ、短い結論を指し示せる状態にします。
- 各結論には、支える証拠を 1 文、限界を 1 文で書きます。この習慣により、小さく汚いデータから言いすぎることを防げます。
- 別の学習者が新しいフォルダから notebook や script を再実行できないなら、次章へ進む前にパス、依存関係、README 手順を直します。

</details>


## 証拠基準

| 成果物 | 答えるべきこと |
|---|---|
| データ辞書 | 各列は何を意味し、単位は何で、どこから来たか。 |
| クリーニングログ | どの行や値を変え、その規則がなぜ受け入れられるか。 |
| 集計表 | どの数値パターンが答えを支えているか。 |
| グラフ | この可視化は 1 つのどの問いに答えるか。 |
| 限界メモ | 欠損データ、サンプリング、時間、リークにより、まだ何が間違いうるか。 |

## 次へ進めるサイン

1 つの CSV を、元データからクリーニング済みデータ、集計表、グラフ、短い結論まで進められたら、第 4 章へ進めます。

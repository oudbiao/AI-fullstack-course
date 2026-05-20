---
title: "7.4.1 事前学習ロードマップ：データ、目的、エンジニアリング"
sidebar_position: 0
description: "短い事前学習ロードマップです。データガバナンス、next-token 目的、エンジニアリングパイプライン、汚染、評価を扱います。"
keywords: [LLM 事前学習, 学習データ, next token prediction, データガバナンス, 事前学習エンジニアリング]
---

# 7.4.1 事前学習ロードマップ：データ、目的、エンジニアリング

事前学習は、モデルが広い言語パターンを最初に学ぶ工程です。エンジニアリング視点では、データを整え、目的を決め、大規模に学習し、リスクを追跡します。

## まず事前学習の三角形を見る

![事前学習章関係図](/img/course/ch07-pretraining-chapter-flow-ja.webp)

![事前学習データ、目的、エンジニアリング三角図](/img/course/ch07-pretraining-data-objective-engineering-map-ja.webp)

| 要素 | 最初に問うこと |
|---|---|
| データ | どのテキストを学習に入れ、何を除外するか |
| 目的 | どの予測タスクが学習信号を作るか |
| エンジニアリング | スケール、checkpoint、ログ、失敗をどう扱うか |
| 評価 | モデルに何ができ、どこで失敗するか |

## next-token ペアを作る

```python
tokens = ["AI", "learns", "from", "text"]
pairs = list(zip(tokens[:-1], tokens[1:]))

for source, target in pairs:
    print(f"{source} -> {target}")
```

期待される出力：

```text
AI -> learns
learns -> from
from -> text
```

![next-token ペア作成の実行結果図](/img/course/ch07-pretraining-next-token-pairs-result-map-ja.webp)

この小さな例が next-token prediction の形です。本物の事前学習では、これを巨大なテキストと厳密なデータガバナンスに広げます。

## この順番で学ぶ

| 順番 | 読む | まず見ること |
|---|---|---|
| 1 | [7.4.2 事前学習データ](./01-pretraining-data.md) | ソース、フィルタリング、重複除去、汚染 |
| 2 | [7.4.3 事前学習手法](./02-pretraining-methods.md) | next-token prediction、loss、scaling |
| 3 | [7.4.4 事前学習エンジニアリング](./03-pretraining-engineering.md) | 分散学習、checkpoint、監視 |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
triangle: data, objective, and engineering all matter
sample_pairs: next-token training pairs from one sentence
data_risk: contamination, duplication, or low-quality mixture
objective_note: objective shapes behavior and architecture fit
engineering_note: sharding, resume, throughput, and monitoring
```

## 合格ライン

データ、目的、エンジニアリングが最終モデルへどう影響するか、そして contamination が評価を誤解させる理由を説明できれば合格です。

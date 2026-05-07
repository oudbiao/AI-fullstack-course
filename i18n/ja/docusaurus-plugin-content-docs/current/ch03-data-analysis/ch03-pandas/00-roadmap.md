---
title: "3.3.1 Pandas ロードマップ：生の表から分析用テーブルへ"
sidebar_position: 8
description: "短い Pandas ロードマップです。表を読み、構造を確認し、整理し、集計し、グラフやモデルへ渡します。"
keywords: [Pandas 入門, DataFrame, データ処理, データクリーニング, groupby, Pandas 学習方法]
---

# 3.3.1 Pandas ロードマップ：生の表から分析用テーブルへ

Pandas は、このコースの表データ作業台です。CSV、Excel、ログ表、SQL の結果を、グラフや機械学習に渡せる形へ整えるときに使います。

## まずワークフローを見る

![Pandas データ処理ロードマップ](/img/course/ch03-pandas-roadmap-ja.png)

まずこの流れを覚えます。

```text
読み込み -> 確認 -> 抽出 -> クリーニング -> 変換 -> 集計 -> 結合 -> 出力
```

最初から API を全部覚えようとしなくて大丈夫です。今ある表は何か、必要な表は何か、どの手順で変わるのかを見ます。

## 小さな表を一度動かす

`pandas_first_loop.py` を作り、`pandas` をインストールしてから実行します。

```python
import pandas as pd

orders = pd.DataFrame(
    [
        {"date": "2026-05-01", "category": "book", "amount": 120},
        {"date": "2026-05-02", "category": "tool", "amount": 80},
        {"date": "2026-05-03", "category": "book", "amount": None},
        {"date": "2026-06-01", "category": "book", "amount": 150},
    ]
)

clean = (
    orders.dropna(subset=["amount"])
    .assign(month=lambda df: pd.to_datetime(df["date"]).dt.to_period("M").astype(str))
)
summary = clean.groupby(["month", "category"], as_index=False)["amount"].sum()

print(summary)
```

出力の形：

```text
     month category  amount
0  2026-05     book   120.0
1  2026-05     tool    80.0
2  2026-06     book   150.0
```

これは Pandas の基本ループです。データを作る/読む、欠損を落とす、列を追加する、グループ化して集計する、という流れです。

## この順番で学ぶ

| 順番 | 読む | 練習すること |
|---|---|---|
| 1 | [3.3.2 コアデータ構造](./01-core-structures.md) | `Series`、`DataFrame`、`Index` |
| 2 | [3.3.3 データの読み書き](./02-read-write.md) | CSV、Excel、JSON、出力 |
| 3 | [3.3.4 選択とフィルタリング](./03-selection-filter.md) | `loc`、`iloc`、条件抽出 |
| 4 | [3.3.5 データクリーニング](./04-data-cleaning.md) | 欠損値、重複、型 |
| 5 | [3.3.6 データ変換](./05-data-transform.md) | 新しい列、マッピング、文字列/日付処理 |
| 6 | [3.3.7 グループ化と集計](./06-groupby.md) | `groupby`、指標、カテゴリ/月別集計 |
| 7 | [3.3.8 データ結合](./07-merge.md) | 複数テーブルを安全に結合する |
| 8 | [3.3.9 時系列](./08-time-series.md) | 日付インデックス、リサンプリング、時間窓 |

## 合格ライン

生の表をきれいな集計表に変え、各列をなぜ処理したか説明でき、可視化や機械学習へ渡せるなら、この小節は合格です。

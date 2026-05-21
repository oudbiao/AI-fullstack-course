---
title: "3 データ分析と可視化"
sidebar_position: 0
description: "実用的なデータ分析ループを学びます：読み込み、品質確認、クリーニング、集計、可視化、結論の説明。"
keywords: [NumPy, Pandas, Matplotlib, Seaborn, データ分析, データ可視化, Pythonデータ分析]
---

# 3 データ分析と可視化

![データ分析と可視化のメインビジュアル](/img/course/ch03-data-visualization-ja.webp)

第 3 章の目的は 1 つです。乱れたデータを、**再現できるコードとグラフに支えられた、信頼できる結論**に変えることです。

## まずデータ分析ループを見る

![データ分析のメインループ](/img/course/ch03-data-analysis-backbone-ja.webp)

先に図を見てください。役に立つ分析の多くは、この流れです。

```text
読み込む -> 確認する -> 整える -> 集計する -> 可視化する -> 説明する
```

最初からグラフを描かないでください。まずフィールド、単位、欠損値、重複、サンプルの出所を確認します。

## 学習順序とタスクリスト

この表を、本章の学習ガイド兼タスクリストとして使います。

| ページ | 手を動かすこと | 残す証拠 |
|---|---|---|
| [3.1.1 純粋な Python データ処理](ch01-warmup/01-pure-python-data.md) | list と dict で小さな表を処理する | 純粋な Python で表処理がつらくなる理由のメモ |
| [3.2.1 NumPy 概要](ch02-numpy/01-overview.md) から [3.2.7 乱数と統計](ch02-numpy/07-random-stats.md) | 配列、shape、スライス、ブロードキャスト、ベクトル化を練習する | NumPy 練習ファイル |
| [3.3.1 Pandas の中心構造](ch03-pandas/01-core-structures.md) から [3.3.8 時系列](ch03-pandas/08-time-series.md) | 表を読み、欠損を処理し、groupby、merge、書き出しを行う | クリーニング済みデータとログ |
| [3.4.1 Matplotlib](ch04-visualization/01-matplotlib.md) から [3.4.4 可視化ベストプラクティス](ch04-visualization/04-best-practices.md) | 明確な質問に答えるグラフを描く | 3 つのグラフと、それぞれ 1 つの結論 |
| [3.5.1 関係データベース](ch05-database/01-relational-db.md) から [3.5.4 データベース設計](ch05-database/04-db-design.md) | SQL で実データを絞り込み、集計し、結合する | クエリまたは join の例 |
| [3.6.1 EDA プロジェクト](ch06-projects/01-eda-project.md) と [3.6.3 ハンズオンワークショップ](ch06-projects/03-hands-on-data-workshop.md) | 再現可能なデータパイプラインとレポートを作る | 元データ、整形済みデータ、グラフ、レポート、README |

本章でよく使う用語：

| 用語 | 意味 |
|---|---|
| `CSV` | 各行が 1 件のレコードになるプレーンテキスト表 |
| `DataFrame` | 行、列、名前、インデックスを持つ Pandas の表 |
| `Series` | DataFrame の 1 列 |
| `dtype` | 列や配列のデータ型 |
| `EDA` | Exploratory Data Analysis：モデリング前の探索的データ分析 |
| `groupby` | カテゴリで分け、統計量を計算し、結果をまとめる操作 |
| `merge` / `join` | 共通キーで複数の表を結合する操作 |

## 最初の実行ループ

まず 2 つのパッケージを入れます。

```bash
python -m pip install pandas matplotlib
```

空の練習フォルダで次のスクリプトを実行します。汚れたデータを作り、整え、集計し、グラフを保存します。

```python
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt

raw = StringIO("""topic,minutes
Python,45
Pandas,30
Python,45
Visualization,
Pandas,300
""")

df = pd.read_csv(raw)
print("クリーニング前")
print(df)

clean_df = df.drop_duplicates()
clean_df["minutes"] = clean_df["minutes"].fillna(clean_df["minutes"].median())
clean_df = clean_df[clean_df["minutes"] <= 180]

summary = clean_df.groupby("topic")["minutes"].sum().sort_values(ascending=False)
print("\nクリーニング後")
print(summary)

summary.plot(kind="bar", title="Study minutes by topic")
plt.ylabel("minutes")
plt.tight_layout()
plt.savefig("topic_minutes.png")
print("\nグラフを保存しました: topic_minutes.png")
```

期待される形：

```text
クリーニング前
...
クリーニング後
topic
Python           45.0
Visualization    ...
グラフを保存しました: topic_minutes.png
```

合格ラインは「グラフがきれい」ではありません。どの行を変えたか、なぜ変えたか、結論にどう影響するかを説明できることです。

### この出力の読み方

- `クリーニング前` は、重複、欠損値、外れ値を含む生の証拠です。
- `クリーニング後` は、実際に分析に使う変換後の表です。
- `topic_minutes.png` はレポート成果物なので、それを生成したスクリプトと一緒に残します。
- 別のクリーニング規則で結論が変わるなら、その変化を隠さず記録します。

## 深度ラダー

| レベル | 証明できること |
|---|---|
| 最低合格 | 表を読み、shape・型・欠損値を確認し、明らかな問題を整え、グラフを 1 つ保存できる。 |
| プロジェクト利用可 | レポートに、問い、クリーニング規則、集計表、グラフ、結論、限界、再実行コマンドがある。 |
| 深い確認 | 別のクリーニング規則で結論が変わるかを試し、リークやサンプリング偏りを見つけ、グラフ種類が問いに合う理由を説明できる。 |

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
データソース: 生レコードまたは使用した小規模データセット
処理ステップ：純 Python、NumPy、Pandas、可視化、または SQL の操作
出力：cleaned data、statistic、chart、query result、またはreport note
失敗確認: データ不足、形状不一致、誤った集計、または不明確な質問
期待される成果：信頼性を支える証拠があるデータアーティファクト
```

## よくある失敗

| 症状 | 最初に確認すること | よくある修正 |
|---|---|---|
| グラフはきれいだが結論が弱い | 先に質問を書いたか | グラフの前に質問を書く |
| groupby の結果がおかしい | 空白、別名、大文字小文字の違い | `unique()` を表示してカテゴリを統一する |
| 欠損値で結論が変わる | どの行と列が欠損しているか | 削除、補完、保持のルールを記録する |
| 相関が完璧すぎる | 時間、規模、リーク、サンプリング偏り | グループ比較を行い、限界を書く |
| Notebook を再実行できない | データパス、依存関係、実行順 | 再起動して上から下へ実行する |

## 通過チェック

次の 5 つに答えられたら、第 4 章へ進めます。

- 各列は何を表し、単位は何ですか？
- どのクリーニングルールがデータを変えましたか？
- 各グラフは何の質問に答えていますか？
- どの結論はデータに支えられ、どこはまだ不確かですか？
- 他の人が README を見て分析を再実行できますか？

印刷用のチェックリストが必要なときは、[3.0 学習ガイドとタスクリスト](./study-guide.md) を使ってください。次の章では、このデータ感覚を使って確率、ベクトル、勾配、モデル評価を理解します。


<details>
<summary>確認の考え方と解説</summary>

- この 5 つの通過チェックは、ばらばらの標語ではなく小さなデータストーリーとして答えます。
- 十分な答えには、列名と単位、行や値を変えたクリーニングルール、各グラフが答える明確な質問、データで支えられる結論と不確かな点、他人が再実行できる README が含まれます。
- 表、グラフ、コマンド出力ではなく記憶に頼っている答えがあるなら、証拠パックはまだ完成していません。

</details>

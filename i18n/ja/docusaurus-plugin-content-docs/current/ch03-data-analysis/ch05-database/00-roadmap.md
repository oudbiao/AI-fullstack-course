---
title: "3.5.1 データベースロードマップ：1つのファイルを越えて残る表"
sidebar_position: 20
description: "データ分析学習者向けの短いデータベースロードマップです。表、キー、SQL、SQLite、Python、そしてデータベースが必要になる場面を扱います。"
keywords: [データベース概要, リレーショナルデータベース, SQL, sqlite, Pandas とデータベース]
---

# 3.5.1 データベースロードマップ：1つのファイルを越えて残る表

この節は第3章の選択内容です。実際のプロジェクトで、データが CSV や DataFrame になる前にどこへ置かれるのかを知りたいときに読みます。

## まずデータベースマップを見る

![データベース選択学習ロードマップ](/img/course/ch03-database-roadmap-ja.png)

まずこの比較を覚えます。

| CSV または DataFrame | データベース |
|---|---|
| 1回のローカル分析に向く | 長期的な共有データに向く |
| 移動しやすい | クエリ、権限、更新に向く |
| たいてい1ファイルまたは1表 | 関連する複数テーブルになりやすい |

ここで DBA になる必要はありません。データを問い合わせ、Python へつなげられれば十分です。

## SQLite クエリを一度動かす

`sqlite_first_loop.py` を作ります。Python 付属の `sqlite3` を使います。

```python
import sqlite3

conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE orders (category TEXT, amount INTEGER)")
conn.executemany(
    "INSERT INTO orders VALUES (?, ?)",
    [("book", 120), ("tool", 80), ("book", 150)],
)

rows = conn.execute(
    "SELECT category, SUM(amount) AS total FROM orders GROUP BY category ORDER BY category"
).fetchall()

for category, total in rows:
    print(category, total)
```

出力：

```text
book 270
tool 80
```

これはデータベースの基本ループです。テーブルを作り、行を入れ、SQL で質問し、結果表を受け取ります。

## この順番で学ぶ

| 順番 | 読む | 練習すること |
|---|---|---|
| 1 | [3.5.2 リレーショナルデータベース基礎](./01-relational-db.md) | 表、行、列、主キー、外部キー |
| 2 | [3.5.3 SQL 基礎](./02-sql-basics.md) | `SELECT`、`WHERE`、`JOIN`、`GROUP BY` |
| 3 | [3.5.4 Python データベース操作](./03-python-db.md) | `sqlite3`、Pandas の読み書き、クエリ結果 |
| 4 | [3.5.5 データベース設計](./04-db-design.md) | テーブル分割、重複削減、関係の整理 |

## 合格ライン

データベースと CSV の違いを説明し、`SELECT ... GROUP BY` クエリを1つ書き、Python から結果を読めれば、この選択小節は合格です。

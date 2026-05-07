---
title: "3.5.1 数据库路线图：不只存在一个文件里的表"
sidebar_position: 20
description: "给数据分析学习者的紧凑版数据库路线图：表、键、SQL、SQLite、Python，以及数据库什么时候重要。"
keywords: [数据库概览, 关系型数据库, SQL, sqlite, Pandas 与数据库]
---

# 3.5.1 数据库路线图：不只存在一个文件里的表

本节是第 3 章的选修内容。想知道真实项目的数据在变成 CSV 或 DataFrame 之前存在哪里，就读它。

## 3.5.1.1 先看数据库地图

![数据库选修学习路线图](/img/course/ch03-database-roadmap.png)

先记住这个对比：

| CSV 或 DataFrame | 数据库 |
|---|---|
| 适合一次本地分析 | 适合长期共享数据 |
| 方便移动 | 更适合查询、权限和更新 |
| 通常是一个文件或一张表 | 往往是多张有关联的表 |

你不需要在这里成为 DBA，只需要能查询数据，并把结果接到 Python。

## 3.5.1.2 先跑一次 SQLite 查询

创建 `sqlite_first_loop.py`。它使用 Python 自带的 `sqlite3`。

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

预期输出：

```text
book 270
tool 80
```

这就是数据库主循环：建表、插入记录、用 SQL 提问、得到结果表。

## 3.5.1.3 按这个顺序学

| 顺序 | 阅读 | 练什么 |
|---|---|---|
| 1 | [3.5.2 关系型数据库基础](./01-relational-db.md) | 表、行、列、主键、外键 |
| 2 | [3.5.3 SQL 基础](./02-sql-basics.md) | `SELECT`、`WHERE`、`JOIN`、`GROUP BY` |
| 3 | [3.5.4 Python 数据库操作](./03-python-db.md) | `sqlite3`、Pandas 读写、查询结果 |
| 4 | [3.5.5 数据库设计](./04-db-design.md) | 拆表、减少重复、保持关系清晰 |

## 3.5.1.4 通过标准

能解释数据库和 CSV 的区别，写出一个 `SELECT ... GROUP BY` 查询，并用 Python 读取结果，就算通过这个选修小节。

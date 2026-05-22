---
title: "3.5.1 数据库路线图：不只存在一个文件里的表"
description: "给数据分析学习者的紧凑版数据库路线图：表、键、SQL、SQLite、Python，以及数据库什么时候重要。"
sidebar:
  order: 20
head:
  - tag: meta
    attrs:
      name: keywords
      content: "数据库概览, 关系型数据库, SQL, sqlite, Pandas 与数据库"
---
本节是第 3 章的选修内容。想知道真实项目的数据在变成 CSV 或 DataFrame 之前存在哪里，就读它。

## 先看数据库地图

![数据库选修学习路线图](/img/course/ch03-database-roadmap.webp)

先记住这个对比：

| CSV 或 DataFrame | 数据库 |
|---|---|
| 适合一次本地分析 | 适合长期共享数据 |
| 方便移动 | 更适合查询、权限和更新 |
| 通常是一个文件或一张表 | 往往是多张有关联的表 |

你不需要在这里成为 DBA，只需要能查询数据，并把结果接到 Python。

## 先跑一次 SQLite 查询

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

## 按这个顺序学

| 顺序 | 阅读 | 练什么 |
|---|---|---|
| 1 | [3.5.2 关系型数据库基础](./01-relational-db.md) | 表、行、列、主键、外键 |
| 2 | [3.5.3 SQL 基础](./02-sql-basics.md) | `SELECT`、`WHERE`、`JOIN`、`GROUP BY` |
| 3 | [3.5.4 Python 数据库操作](./03-python-db.md) | `sqlite3`、Pandas 读写、查询结果 |
| 4 | [3.5.5 数据库设计](./04-db-design.md) | 拆表、减少重复、保持关系清晰 |

## 通过标准

能解释数据库和 CSV 的区别，写出一个 `SELECT ... GROUP BY` 查询，并用 Python 读取结果，就算通过这个选修小节。

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要先说清问题，再指出需要的表、DataFrame 或 SQL 查询，并让清洗步骤可以复现。
2. 证据至少包含一小段输出、必要的图表或查询结果，以及一句对结果的解释。
3. 自检时要能说出一个数据质量风险，例如缺失值、重复行、错误 join、聚合误导或图表难读。

</details>


## 留下的证据

学完这一页，至少保留这张证据卡：

```text
架构：表名、键、关系和示例行
查询：所使用的 SQL 或 Python 数据库代码
输出：结果行、行数，或保存的抽取结果
失败检查：错误的连接键、不安全查询、缺少事务，或 schema 不匹配
期望产出：查询、结果表和一条数据质量说明
```

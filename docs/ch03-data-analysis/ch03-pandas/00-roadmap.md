---
title: "3.3.1 Pandas Roadmap: From Raw Table to Analysis Table"
sidebar_position: 8
description: "A compact Pandas roadmap: read a table, inspect it, clean it, summarize it, and prepare it for charts or models."
keywords: [Pandas introduction, DataFrame, data processing, data cleaning, groupby, how to learn Pandas]
---

# 3.3.1 Pandas Roadmap: From Raw Table to Analysis Table

Pandas is the table workstation of this course. Use it when a plain CSV, Excel file, log table, or SQL query result must become clean enough for charts, machine learning, RAG evaluation, or reports.

## 3.3.1.1 Look at the Workflow First

![Pandas data processing roadmap](/img/course/ch03-pandas-roadmap-en.png)

Keep this one-line flow in mind:

```text
read -> inspect -> select -> clean -> transform -> group -> merge -> export
```

Do not memorize every API first. Ask: what table do I have, what table do I need, and which step changes one into the other?

## 3.3.1.2 Run a Tiny Table Once

Create `pandas_first_loop.py` and run it after installing `pandas`.

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

Expected output shape:

```text
     month category  amount
0  2026-05     book   120.0
1  2026-05     tool    80.0
2  2026-06     book   150.0
```

You just did the core Pandas loop: create/read data, clean missing values, add a derived column, group, and summarize.

## 3.3.1.3 Learn in This Order

| Order | Read | What to practice |
|---|---|---|
| 1 | [3.3.2 Core Data Structures](./01-core-structures.md) | `Series`, `DataFrame`, `Index` |
| 2 | [3.3.3 Data Reading and Writing](./02-read-write.md) | CSV, Excel, JSON, export |
| 3 | [3.3.4 Selection and Filtering](./03-selection-filter.md) | `loc`, `iloc`, conditions |
| 4 | [3.3.5 Data Cleaning](./04-data-cleaning.md) | missing values, duplicates, types |
| 5 | [3.3.6 Data Transformation](./05-data-transform.md) | new columns, mapping, string/date handling |
| 6 | [3.3.7 Grouping and Aggregation](./06-groupby.md) | `groupby`, metrics, category/month summaries |
| 7 | [3.3.8 Data Merging](./07-merge.md) | join multiple tables safely |
| 8 | [3.3.9 Time Series](./08-time-series.md) | date index, resampling, time windows |

## 3.3.1.4 Pass Check

You pass this subchapter when you can turn one raw table into one clean summary table, explain each column change, and save the result for visualization or machine learning.

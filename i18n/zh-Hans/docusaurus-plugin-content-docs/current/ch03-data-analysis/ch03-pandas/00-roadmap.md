---
title: "3.3.1 Pandas 路线图：从原始表到分析表"
sidebar_position: 8
description: "紧凑版 Pandas 路线图：读取表格、检查结构、清洗、汇总，并交给图表或模型继续使用。"
keywords: [Pandas 入门, DataFrame, 数据处理, 数据清洗, groupby, Pandas 学习方法]
---

# 3.3.1 Pandas 路线图：从原始表到分析表

Pandas 是本课程里的表格工作台。只要 CSV、Excel、日志表或 SQL 查询结果需要变成干净表格，就会用到它。

## 先看工作流

![Pandas 数据处理路线图](/img/course/ch03-pandas-roadmap.webp)

先记住这一行：

```text
读取 -> 检查 -> 筛选 -> 清洗 -> 转换 -> 分组 -> 合并 -> 导出
```

不要一开始背 API。先问自己：我现在有什么表、最后需要什么表、中间哪一步在改变它？

## 先跑一次小表格

创建 `pandas_first_loop.py`，安装 `pandas` 后运行。

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

预期输出形状：

```text
     month category  amount
0  2026-05     book   120.0
1  2026-05     tool    80.0
2  2026-06     book   150.0
```

这就是 Pandas 主循环：创建或读取数据，清理缺失值，增加派生列，分组并汇总。

## 按这个顺序学

| 顺序 | 阅读 | 练什么 |
|---|---|---|
| 1 | [3.3.2 核心数据结构](./01-core-structures.md) | `Series`、`DataFrame`、`Index` |
| 2 | [3.3.3 数据读取与写入](./02-read-write.md) | CSV、Excel、JSON、导出 |
| 3 | [3.3.4 数据选择与筛选](./03-selection-filter.md) | `loc`、`iloc`、条件筛选 |
| 4 | [3.3.5 数据清洗](./04-data-cleaning.md) | 缺失值、重复值、类型 |
| 5 | [3.3.6 数据转换](./05-data-transform.md) | 新列、映射、字符串和日期处理 |
| 6 | [3.3.7 分组与聚合](./06-groupby.md) | `groupby`、指标、类别/月度汇总 |
| 7 | [3.3.8 数据合并](./07-merge.md) | 安全合并多张表 |
| 8 | [3.3.9 时间序列](./08-time-series.md) | 日期索引、重采样、时间窗口 |

## 通过标准

能把一张原始表变成一张干净的汇总表，能解释每个字段为什么这样处理，并能把结果交给可视化或机器学习继续使用，就算通过。

<details>
<summary>参考答案与讲解</summary>

1. 合格答案要先说清问题，再指出需要的表、DataFrame 或 SQL 查询，并让清洗步骤可以复现。
2. 证据至少包含一小段输出、必要的图表或查询结果，以及一句对结果的解释。
3. 自检时要能说出一个数据质量风险，例如缺失值、重复行、错误 join、聚合误导或图表难读。

</details>


## 留下的证据

学完这一页，至少保留这张证据卡：

```text
dataframe_state: columns, dtypes, row count, missing values, and sample rows
operation: read/write, select/filter, clean, transform, groupby, merge, or time-series step
output: resulting table, saved file, aggregation, join result, or time index view
failure_check: dtype mismatch, missing data, duplicated keys, chained assignment, or wrong time frequency
Expected_output: before/after table sample with the transformation reason
```

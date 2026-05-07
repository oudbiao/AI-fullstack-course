---
title: "3 数据分析与可视化"
sidebar_position: 0
description: "学习实用数据闭环：读取数据、检查质量、清洗问题、统计规律、绘制图表并解释结论。"
keywords: [NumPy, Pandas, Matplotlib, Seaborn, 数据分析, 数据可视化, Python数据分析]
---

# 3 数据分析与可视化

![数据分析与可视化主视觉](/img/course/ch03-data-visualization.png)

第 3 章只解决一件事：把混乱数据变成**有代码可复现、有图表支撑、能讲清边界的可信结论**。

## 3.0.1 先看数据分析闭环

![数据分析主线闭环](/img/course/ch03-data-analysis-backbone.png)

先看图。大多数有价值的数据分析都遵循这个闭环：

```text
读取 -> 检查 -> 清洗 -> 统计 -> 可视化 -> 解释
```

不要一上来就画图。先看字段、单位、缺失值、重复数据和样本来源。

## 3.0.2 学习顺序与任务表

下面这一张表同时作为本章学习指南和任务清单。

| 页面 | 跟着做 | 留下的证据 |
|---|---|---|
| [3.1.1 纯 Python 数据处理](ch01-warmup/01-pure-python-data.md) | 用列表和字典处理一张小表 | 一段说明：为什么纯 Python 处理表格会变累 |
| [3.2.1 NumPy 概览](ch02-numpy/01-overview.md) 到 [3.2.7 随机数与统计](ch02-numpy/07-random-stats.md) | 练习数组、形状、切片、广播和向量化计算 | 一个 NumPy 练习文件 |
| [3.3.1 Pandas 核心结构](ch03-pandas/01-core-structures.md) 到 [3.3.8 时间序列](ch03-pandas/08-time-series.md) | 读取表格、清洗缺失值、分组、合并并导出结果 | 清洗后的数据和清洗记录 |
| [3.4.1 Matplotlib](ch04-visualization/01-matplotlib.md) 到 [3.4.4 可视化最佳实践](ch04-visualization/04-best-practices.md) | 画出能回答明确问题的图表 | 3 张图，每张配 1 条结论 |
| [3.5.1 关系型数据库](ch05-database/01-relational-db.md) 到 [3.5.4 数据库设计](ch05-database/04-db-design.md) | 学会用 SQL 筛选、分组和连接真实应用数据 | 一个查询或 join 示例 |
| [3.6.1 EDA 项目](ch06-projects/01-eda-project.md) 和 [3.6.3 跟做工作坊](ch06-projects/03-hands-on-data-workshop.md) | 搭建可复现的数据流水线和报告 | 原始数据、清洗数据、图表、报告和 README |

本章常见术语：

| 术语 | 含义 |
|---|---|
| `CSV` | 纯文本表格，每一行是一条记录 |
| `DataFrame` | Pandas 表格，有行、列、列名和索引 |
| `Series` | DataFrame 中的一列 |
| `dtype` | 列或数组的数据类型 |
| `EDA` | Exploratory Data Analysis，建模前的探索性数据分析 |
| `groupby` | 按类别拆分、计算统计量、再合并结果 |
| `merge` / `join` | 按共同键把多张表合并 |

## 3.0.3 第一个可运行闭环

先安装两个包：

```bash
python -m pip install pandas matplotlib
```

然后在空练习文件夹里运行下面脚本。它会创建脏数据、清洗、统计，并保存一张图。

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
print("清洗前")
print(df)

clean_df = df.drop_duplicates()
clean_df["minutes"] = clean_df["minutes"].fillna(clean_df["minutes"].median())
clean_df = clean_df[clean_df["minutes"] <= 180]

summary = clean_df.groupby("topic")["minutes"].sum().sort_values(ascending=False)
print("\n清洗后")
print(summary)

summary.plot(kind="bar", title="Study minutes by topic")
plt.ylabel("minutes")
plt.tight_layout()
plt.savefig("topic_minutes.png")
print("\n已保存图表: topic_minutes.png")
```

预期形态：

```text
清洗前
...
清洗后
topic
Python           45.0
Visualization    ...
已保存图表: topic_minutes.png
```

真正的通过标准不是“图好看”，而是你能说明：哪些行被改了，为什么改，对结论有什么影响。

## 3.0.4 常见失败

| 现象 | 先检查什么 | 常见修复 |
|---|---|---|
| 图表漂亮但结论弱 | 是否先写清问题 | 把问题写在图表前面 |
| 分组结果不对 | 类别空格、别名或大小写不一致 | 打印 `unique()` 并统一类别 |
| 缺失值影响结论 | 哪些行列缺失 | 记录规则：删除、填充或保留 |
| 相关性高得异常 | 时间、规模、泄漏或采样偏差 | 分组对比，并写清限制 |
| Notebook 无法重跑 | 数据路径、依赖或执行顺序 | 重启后从头运行 |

## 3.0.5 通关检查

能回答下面五个问题，就可以进入第 4 章：

- 每列是什么意思，单位是什么？
- 哪些清洗规则改变了数据？
- 每张图回答了什么问题？
- 哪些结论有数据支持，哪些仍不确定？
- 其他人能不能按 README 重新运行分析？

需要打印式清单时，打开 [3.0 学习指南与任务单](./study-guide.md)。下一章会用这些数据直觉理解概率、向量、梯度和模型评估。

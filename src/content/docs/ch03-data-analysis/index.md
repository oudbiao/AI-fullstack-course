---
title: "3 Data Analysis and Visualization"
description: "Learn the practical data loop: read data, inspect quality, clean problems, summarize patterns, draw charts, and explain conclusions."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "NumPy, Pandas, Matplotlib, Seaborn, data analysis, data visualization, Python data analysis"
---
![Main visual for data analysis and visualization](/img/course/ch03-data-visualization-en.webp)

Chapter 3 has one job: help you turn messy data into **a trustworthy conclusion with reproducible code and charts**.

## See The Data Analysis Loop

![Main workflow loop of data analysis](/img/course/ch03-data-analysis-backbone-en.webp)

Read the picture first. Most useful analysis follows this loop:

```text
read -> inspect -> clean -> summarize -> visualize -> explain
```

Do not draw charts first. First understand fields, units, missing values, duplicates, and sample sources.

## Learning Order And Task List

Use this checklist as both the chapter guide and the task sheet. Each step should make clear where the data came from, how it changed, and what supports the conclusion.

1. **[3.1.1 Pure Python Data Processing](/ch03-data-analysis/ch01-warmup/01-pure-python-data/)**
   Follow along: process a small table with lists and dictionaries.
   Evidence to keep: a note explaining why tables become awkward in pure Python.

2. **[3.2.1 NumPy Overview](/ch03-data-analysis/ch02-numpy/01-overview/) to [3.2.7 Random and Statistics](/ch03-data-analysis/ch02-numpy/07-random-stats/)**
   Follow along: practice arrays, shapes, slicing, broadcasting, and vectorized math.
   Evidence to keep: one NumPy practice file.

3. **[3.3.1 Pandas Core Structures](/ch03-data-analysis/ch03-pandas/01-core-structures/) to [3.3.8 Time Series](/ch03-data-analysis/ch03-pandas/08-time-series/)**
   Follow along: read a table, clean missing values, group rows, merge tables, and export results.
   Evidence to keep: cleaned data plus a cleaning log.

4. **[3.4.1 Matplotlib](/ch03-data-analysis/ch04-visualization/01-matplotlib/) to [3.4.4 Visualization Best Practices](/ch03-data-analysis/ch04-visualization/04-best-practices/)**
   Follow along: draw charts that answer named questions.
   Evidence to keep: 3 charts, each with one conclusion.

5. **[3.5.1 Relational Databases](/ch03-data-analysis/ch05-database/01-relational-db/) to [3.5.4 Database Design](/ch03-data-analysis/ch05-database/04-db-design/)**
   Follow along: learn enough SQL to filter, group, and join real application data.
   Evidence to keep: one query or join example.

6. **[3.6.1 EDA Project](/ch03-data-analysis/ch06-projects/01-eda-project/) and [3.6.3 Follow-Along Workshop](/ch03-data-analysis/ch06-projects/03-hands-on-data-workshop/)**
   Follow along: build a reproducible data pipeline and report.
   Evidence to keep: raw data, clean data, chart, report, and README.

Key terms for this chapter:

| Term | Meaning |
|---|---|
| `CSV` | A plain-text table where each row is a record |
| `DataFrame` | A Pandas table with rows, columns, names, and indexes |
| `Series` | One column from a DataFrame |
| `dtype` | The data type of a column or array |
| `EDA` | Exploratory Data Analysis: first-pass exploration before modeling |
| `groupby` | Split by category, calculate statistics, then combine |
| `merge` / `join` | Combine tables by shared keys |

## First Runnable Loop

Install the two packages once:

```bash
python -m pip install pandas matplotlib
```

Then run this script in an empty practice folder. It creates dirty data, cleans it, summarizes it, and saves one chart.

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
print("Before cleaning")
print(df)

clean_df = df.drop_duplicates()
clean_df["minutes"] = clean_df["minutes"].fillna(clean_df["minutes"].median())
clean_df = clean_df[clean_df["minutes"] <= 180]

summary = clean_df.groupby("topic")["minutes"].sum().sort_values(ascending=False)
print("\nAfter cleaning")
print(summary)

summary.plot(kind="bar", title="Study minutes by topic")
plt.ylabel("minutes")
plt.tight_layout()
plt.savefig("topic_minutes.png")
print("\nSaved chart: topic_minutes.png")
```

Expected shape:

```text
Before cleaning
...
After cleaning
topic
Python           45.0
Visualization    ...
Saved chart: topic_minutes.png
```

The pass line is not “the chart looks nice.” The pass line is: you can explain which rows changed, why they changed, and how that affects the conclusion.

### How to read this output

- `Before cleaning` shows the raw evidence, including duplicates, missing values, and outliers.
- `After cleaning` shows the transformed table you are actually using for analysis.
- `topic_minutes.png` is the report artifact; keep it with the script that generated it.
- If the conclusion changes after another cleaning rule, write that down instead of hiding it.

## Depth Ladder

| Level | What you can prove |
|---|---|
| Minimum pass | You can read a table, inspect shape/types/missing values, clean obvious problems, and save one chart. |
| Project-ready | Your report names the question, cleaning rules, summary table, chart, conclusion, limitation, and rerun command. |
| Deeper check | You can test whether the conclusion changes under another cleaning rule, spot leakage or sampling bias, and explain why a chart type fits the question. |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
data_source: raw records or small dataset used
processing_step: pure Python, NumPy, Pandas, charting, or SQL operation
output: cleaned data, statistic, chart, query result, or report note
failure_check: missing data, shape mismatch, wrong aggregation, or unclear question
Expected_output: data artifact plus the evidence needed to trust it
```

## Common Failures

| Symptom | First thing to check | Usual fix |
|---|---|---|
| Chart is pretty but conclusion is weak | Did you name the question first? | Write the question above the chart |
| Grouped result looks wrong | Category spaces, aliases, or inconsistent case | Print `unique()` and normalize categories |
| Missing values change the conclusion | Which rows and columns are missing? | Record the rule: drop, fill, or keep |
| Correlation looks too perfect | Time, scale, leakage, or sampling bias | Compare groups and add limitation notes |
| Notebook cannot rerun | Data path, dependency, or execution order | Restart and run from top to bottom |

## Pass Check

Move to Chapter 4 when you can answer these five questions:

- What does each column mean, and what unit does it use?
- Which cleaning rules changed the data?
- What question does each chart answer?
- What conclusion is supported, and what is still uncertain?
- Can another person rerun the analysis from the README?

For a printable checklist, use [3.0 Study Guide and Task Sheet](/ch03-data-analysis/study-guide/). The next chapter uses this data intuition to understand probability, vectors, gradients, and model evaluation.


<details>
<summary>Check reasoning and explanation</summary>

- Use the five pass-check questions as a small data story, not as five separate slogans.
- A complete answer names the columns and units, lists every cleaning rule that changed rows or values, ties each chart to one explicit question, separates supported conclusions from uncertainty, and includes a README that lets another person rerun the notebook or script.
- If any answer depends on memory instead of a saved table, chart, or command output, the evidence pack is not ready yet.

</details>

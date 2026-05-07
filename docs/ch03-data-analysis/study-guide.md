---
title: "3.0 Study Guide and Task Sheet: Data Analysis and Visualization"
sidebar_position: 1
description: "A compact study guide and task sheet for Chapter 3: move from raw data to cleaning, analysis, charts, and a reproducible report."
keywords: [data analysis study guide, data analysis task sheet, NumPy, Pandas, visualization]
---

# 3.0 Study Guide and Task Sheet: Data Analysis and Visualization

Use this page as the control panel for Chapter 3. Your goal is not to memorize every Pandas method. Your goal is to turn raw data into a conclusion that someone can trust.

## 3.0.1 Recommended Learning Order

![Minimum loop for the data analysis study guide](/img/course/ch03-study-guide-data-loop-vertical-en.png)

Keep one workflow in mind: **read -> inspect -> clean -> summarize -> visualize -> explain**.

| Order | Section | Focus | Evidence to leave behind |
|---|---|---|---|
| 1 | `3.1 From Python to Data Analysis` | Why pure Python becomes painful for tables | A small before/after note |
| 2 | `3.2 NumPy Scientific Computing` | Arrays, shape, broadcasting, vectorization | One array practice file |
| 3 | `3.3 Pandas Data Processing` | DataFrame, selection, cleaning, groupby, merge | A cleaned table and cleaning log |
| 4 | `3.4 Data Visualization` | Chart choice, labels, conclusions, limits | 3 charts tied to 3 questions |
| 5 | `3.5 Database Basics` | SQL and relational data, optional but useful | One query or join example |
| 6 | `3.6 Stage Projects` | EDA report and reproducible pipeline | Report, chart files, and workshop output |

Do not draw charts first. First understand the fields, units, missing values, and sample source.

## 3.0.2 Terms You Need Before You Start

| Term | First meaning in this chapter |
|---|---|
| `CSV` | A plain text table; each row is a record. |
| `JSON` | Nested data often returned by APIs. |
| `DataFrame` | A Pandas table with rows, columns, names, and indexes. |
| `Series` | One column of a DataFrame. |
| `dtype` | The data type of a column or array. |
| `EDA` | Exploratory Data Analysis: first-pass exploration before modeling. |
| `groupby` | Split by category, calculate statistics, then combine. |
| `merge` / `join` | Combine tables by shared keys. |
| `vectorization` | Let NumPy/Pandas process many values at once. |

When an API feels hard to remember, translate it back to the workflow step: read, inspect, clean, transform, visualize, or explain.

## 3.0.3 Tasks You Must Complete in This Stage

| Task | Deliverable | Pass criteria |
|---|---|---|
| Use NumPy arrays | Array practice file | Can explain shape, slicing, broadcasting, and vectorized operations |
| Load data with Pandas | Script or Notebook | Can read CSV/Excel/JSON and inspect rows, columns, types, and missing values |
| Clean data | Before/after cleaning record | Can handle missing values, duplicates, outliers, and type conversion |
| Explore data | EDA Notebook | Can use statistics and charts to answer clear questions |
| Complete the guided pipeline | `ch03_output/` | Can reproduce raw data, clean data, SQLite query, chart, and report |
| Finish the stage project | Analysis report | Includes question, process, charts, conclusions, and limitations |

## 3.0.4 Chart Decision Rule

Before drawing a chart, write the question first.

| Question | Chart to try first |
|---|---|
| Which category is largest? | Bar chart |
| How does a value change over time? | Line chart |
| Are two numbers related? | Scatter plot |
| What does the distribution look like? | Histogram or box plot |
| Which groups differ? | Grouped bar chart or box plot |

A chart without a question is decoration. A chart with a question becomes evidence.

## 3.0.5 Stage Portfolio Deliverables

| Deliverable | What it proves |
|---|---|
| `analysis.ipynb` | You can run a full EDA loop. |
| `data_dictionary.md` | You understand field meaning, type, unit, and missingness. |
| `cleaning_log.md` | You can explain data cleaning decisions. |
| `figures/` | Your charts answer specific questions. |
| `report.md` | You can turn analysis into conclusions and limits. |
| `ch03_output/` | You completed the follow-along reproducible pipeline. |

## 3.0.6 Stage Completion Questions

Before moving to Chapter 4, check that you can answer these questions:

- Why does array `shape` matter?
- What is the difference between a Series and a DataFrame?
- How can missing values change a conclusion?
- What question does each chart answer?
- Why should a data analysis report include limitations?

You are ready to continue when you can take one CSV file from loading to cleaning, analysis, visualization, and a short written conclusion.

---
title: "3 Data Analysis and Visualization"
sidebar_position: 0
description: "Learn NumPy, Pandas, data cleaning, statistical analysis, and visualization to build the data skills needed for machine learning, RAG evaluation, and AI application analytics."
keywords: [NumPy, Pandas, Matplotlib, Seaborn, data analysis, data visualization, Python data analysis]
---

# 3 Data Analysis and Visualization

![Main visual for data analysis and visualization](/img/course/ch03-data-visualization-en.png)

This stage is about answering: “Can I understand data, organize data, discover patterns, and explain conclusions?” Whether you later work on machine learning, RAG, Agents, or product analytics, data skills are a foundational capability.

## Story-based introduction: investigate like a data detective

As you enter this stage, imagine yourself as a data detective. CSV files, Excel sheets, logs, and database tables are like a pile of clues: some clues are missing, some are duplicated, and some look normal but are actually outliers. Your job is not to jump straight into a model. First, clean up the clues, then use statistics and charts to find patterns, and finally tell a trustworthy conclusion.

## Learning quest map

![Data analysis learning quest map](/img/course/ch03-learning-quest-map-en.png)

## Interactive practice: ask every dataset three questions

The first question asks “What is it?”: How many rows, how many columns, and what does each column mean? The second question asks “Is it clean?”: Where are the missing values, where are the outliers, and are the data types correct? The third question asks “What does it tell us?”: Which variable is highest, which trend is most obvious, and which group differences are worth further study?

## Project bonus

The bonus project for this stage is a truly presentable data analysis report. You can build it as a Notebook, or organize it as a visual report: start with the question, show the cleaning and analysis process in the middle, and end with conclusions and next-step modeling suggestions. When you later learn machine learning, this report can be upgraded directly into a complete “from data analysis to prediction model” project.

## Stage overview

| Information | Description |
|---|---|
| Suitable for | Learners who can already write basic Python and want to move into data and AI projects |
| Estimated time | 80–120 hours |
| Prerequisites | Completed Python programming basics |
| Stage output | A complete data analysis report, and a multi-source data organization project |

## Minimum path for beginners

Beginners should first learn how to read data, inspect fields, handle missing values, do basic filtering and grouping statistics, and then use Matplotlib or Seaborn to draw charts that explain the problem. As long as you can complete a Notebook report from reading data to summarizing conclusions, you have passed the minimum path.

## Advanced path

Experienced learners can go deeper into multi-source data merging, time series, database access, visualization storytelling, and analysis report structure. You can further try turning the data cleaning process into reusable functions and output a clean dataset for future machine learning projects.

## What beginners should do first, and what advanced learners should do later

When beginners study this stage for the first time, treat data as “tables that need cleaning.” If you can load data, understand column meanings, handle missing values, draw key charts, and write three conclusions, you have already grasped the main thread.

Experienced learners can focus more on the reliability of the analysis: Is the data source trustworthy? Are the cleaning rules explainable? Do the charts mislead? Do the conclusions have boundaries? Your goal is not to draw more charts, but to produce data analysis that can support later modeling and business decisions.

## Why AI cannot do without data

Models do not learn from thin air. Data quality, structure, distribution, and labeling methods directly affect model results. Even if you mainly work on large model applications in the future, you will still encounter document cleaning, log analysis, evaluation set construction, user feedback statistics, and retrieval quality analysis.

![Main workflow loop of data analysis](/img/course/ch03-data-analysis-backbone-en.png)

## Learning path for this stage

Chapter 1 uses pure Python to process data, helping you feel why NumPy and Pandas are needed.

Chapter 2 teaches NumPy. You will understand arrays, vectorization, broadcasting, matrix operations, and random statistics. These concepts will appear again and again in machine learning and deep learning.

Chapter 3 teaches Pandas. You will work with tabular data, read and write files, filter and select, clean missing values, group and aggregate, merge data, and handle time series.

Chapter 4 teaches visualization. You will use charts to express distributions, trends, relationships, and anomalies, rather than just printing results as tables.

Chapter 5 databases are optional, but if you want to work on real applications or enterprise data projects, it is recommended that you at least understand relational databases and the basics of SQL.

Chapter 6 is where you turn the scattered skills into visible evidence. If you want one course page to follow from start to finish, do [6.3 Follow-Along Workshop: Build a Reproducible Data Analysis Pipeline](./ch06-projects/03-hands-on-data-workshop.md): it creates dirty CSV data, cleans it, writes SQLite, generates a chart, and outputs a report.

## What you should be able to do after finishing this stage

- Read common data files such as CSV, Excel, and JSON
- Use Pandas to filter, clean, transform, aggregate, and merge data
- Use NumPy to understand array computation and basic statistics
- Choose appropriate charts to present data conclusions
- Organize the data analysis process into a Notebook or report
- Prepare clean datasets for future machine learning projects

## Common mistakes

Do not turn data analysis into “memorizing APIs.” Pandas has many methods, and you do not need to remember them all the first time. You should understand the workflow of data processing: first inspect what the data looks like, then check missing values and anomalies, then perform transformations and statistics, and finally use charts to verify your judgment.

Also, do not just make pretty charts. The purpose of a chart is not decoration, but answering questions. Each chart should correspond to a clear question, such as “Which feature is most related to the target?”, “Is there a clear anomaly?”, or “What differences exist between groups?”

## Data failure theater: why charts and conclusions can mislead

If a chart looks beautiful but the conclusion is strange, first check whether the data has missing values, duplicates, outliers, or inconsistent units. If grouped results do not match intuition, check whether the category fields contain spaces, case differences, or encoding issues. If the correlation looks very high, first ask whether it is only influenced by time, scale, or the sampling method.

## Minimum runnable experiment: from dirty data to one trustworthy chart

The minimum experiment for this stage is to prepare a small CSV with missing values, duplicate rows, and outliers, then complete reading, inspection, cleaning, statistics, and visualization. You need to prove that you are not just calling the Pandas API, but can explain why each step is handled that way.

```python
import pandas as pd

df = pd.read_csv("learning_log.csv")
print(df.info())
print(df.isna().sum())

clean_df = df.drop_duplicates()
summary = clean_df.groupby("topic")["minutes"].sum().sort_values(ascending=False)
print(summary)
```

The real pass criterion is not whether the code runs, but whether you can explain: which data was changed, why it was changed, and what impact it had on the conclusion.

## Data failure case library: check fields, units, and sample sources first

| Phenomenon | Common cause | How to locate it | Fix direction |
|---|---|---|---|
| Charts look nice but the conclusion is untrustworthy | Missing values, duplicates, or outliers were not handled | Check `info()`, missing-value statistics, and distributions | Record cleaning rules and the scope of impact |
| Grouped statistics look abnormal | Category fields contain spaces, case differences, or aliases | Print unique values and frequencies | Normalize field names and category mappings |
| Correlation looks very high | Influenced by time, scale, or sampling method | Compare by strata, draw scatter plots | Add business explanation and limitation notes |
| Notebook cannot be reproduced | Data path, dependencies, or execution order is messy | Restart and run from the beginning | Fix the data path, dependencies, and execution order |

## Stage evaluation rubric

| Level | Evaluation standard | Portfolio evidence |
|---|---|---|
| Minimum pass | Can read, clean, summarize, and draw key charts | Notebook, CSV, chart outputs |
| Recommended pass | Can write a data analysis report around a question | Data dictionary, cleaning log, conclusion notes |
| Portfolio pass | Can explain data limitations and future modeling risks | Failure samples, limitations, reproducible run notes |

## Stage project

The basic version is to choose a public CSV dataset and complete field understanding, missing-value checks, basic statistics, and 3–5 key charts. The standard version requires organizing the analysis process into a complete Notebook or report, including the question, cleaning, analysis, conclusions, and next-step suggestions. The challenge version can integrate multiple data sources, add database or web data, and form a data analysis workflow closer to real business scenarios.

## Fun task cards for this stage

| Play style | Task for this stage |
|---|---|
| Story mission | Let the assistant understand learning records: find missing values, duplicates, and outliers, and turn the conclusions into trustworthy charts. |
| Boss fight | **Dirty Data Detective** |
| Unlockable badges | Dirty Data Detective, Chart Storyteller |
| Beginner easy mode | Complete only one minimum input-to-output loop, and keep a run screenshot or command output |
| Portfolio evidence | Data quality checklist and one chart with explanation |

If this stage feels like a lot, treat this task card as your minimum target first. Once you can complete beginner easy mode, you can keep going; later, when preparing your portfolio, come back and upgrade to the standard and challenge versions.

## Stage deliverables

| Deliverable | Minimum version | Portfolio version |
|---|---|---|
| Data analysis Notebook | Complete reading, cleaning, statistics, and charts | Includes problem definition, cleaning rules, key findings, and limitations |
| Data quality record | Mark missing values, duplicates, and outliers | Explains the reason for handling, impact scope, and basis for keeping/deleting |
| Visualization results | Draw 3–5 key charts | Each chart includes the question, conclusion, and misleading-risk notes |
| Analysis report | Write 3 conclusions | Can connect business questions, future modeling, and data limitations |
| Reproducibility notes | Clearly state data source and how to run | Includes data dictionary, dependencies, random seed, or version records |
| Reproducible workshop evidence | Run the 6.3 pipeline and keep the generated files | Includes raw data, clean data, cleaning log, SQLite database, SVG chart, and HTML report |

## Relationship with the AI learning assistant throughout the project

This stage can correspond to AI Learning Assistant v0.3: analyze learning records, count study time, completion rate, and procrastination topics, and generate charts. If you are following the end-to-end project path, it is recommended that by the end of this stage you submit at least one version record: what capabilities were added, how to run it, what the sample inputs and outputs are, what problems were encountered, and what you plan to improve next.

## Stage completion criteria

| Pass level | What you need to do |
|---|---|
| Minimum pass | Can use NumPy, Pandas, SQL, and visualization tools to complete a data analysis report. |
| Recommended pass | Complete at least one runnable mini project for this stage, and record the run method, sample inputs and outputs, and problems encountered in the README. |
| Portfolio pass | Connect the output of this stage to the “AI Learning Assistant” end-to-end project, leaving screenshots, logs, evaluation examples, and a next-step plan. |

After finishing this stage, you do not need to memorize every detail. What matters more is that you can clearly explain: what problem this stage solves, how it relates to the previous stage, and how it supports future learning. The next stage will use this data intuition to understand probability, vectors, gradients, and model evaluation.

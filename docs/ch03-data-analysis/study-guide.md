---
title: "Study Guide: How to Learn Data Analysis and Visualization Without Getting Confused"
sidebar_position: 1
description: "A data analysis learning guide for AI full-stack beginners: NumPy, Pandas, visualization, the minimal analysis loop, project roadmap, and acceptance criteria."
keywords: [Data analysis learning guide, How to learn NumPy, How to learn Pandas, How to learn visualization]
---

# Study Guide: How to Learn Data Analysis and Visualization Without Getting Confused

If you arrive at `02 Data Analysis and Visualization` and feel like there are too many libraries and the APIs are too fragmented, don’t rush to memorize method names. What you really need to build at this stage is data-flow awareness.

## Overall principle for this stage

The first time you learn data analysis, focus on one main flow: read the data in, understand the fields first, then clean and organize it, then do statistical analysis, and finally use charts to present conclusions.

![Minimum loop for the data analysis study guide](/img/course/ch03-study-guide-data-loop-en.png)

## Recommended learning order

In the first round, do some pure Python data processing warm-up to feel why specialized tools are needed.

In the second round, learn NumPy, focusing on arrays, shape, indexing, broadcasting, vectorization, matrix operations, and basic statistics. Don’t try to memorize every function from the start.

In the third round, learn Pandas, focusing on DataFrame, reading and writing files, selecting and filtering, missing values, grouping and aggregation, merging, and time series.

In the fourth round, learn visualization. First learn how to choose the right chart, then learn how to polish it. Charts should serve the question, not be drawn just to look nice.

In the fifth round, do an EDA project and connect reading, cleaning, analysis, visualization, and conclusion writing into a complete report.

## Suggested learning pace

| Content type | Suggested time | Learning goal |
|---|---|---|
| NumPy basics | 1–2 hours | Understand arrays, shape, and vectorization |
| Pandas processing page | 2–4 hours | Be able to filter, clean, aggregate, and merge |
| Visualization page | 1–3 hours | Be able to choose the right chart and explain it |
| Project page | 6–12 hours | Complete a readable data analysis report |

## Stage project roadmap

For your first project, it is recommended to do an EDA on a single dataset, such as Titanic, housing prices, movie ratings, e-commerce orders, or public operations data. You need to complete field understanding, missing value handling, statistical summaries, key charts, and conclusions.

For your second project, you can do multi-source data analysis by combining multiple CSV files, web data, or database tables, and practice the data organization process that is more common in real work.

## Common stumbling blocks

The most common stumbling block is “there are too many APIs to remember.” That’s completely normal. You only need to first remember the common actions: read, inspect, select, filter, modify, group, merge, and plot. You can look up the other methods when you need them.

The second stumbling block is “I don’t know what the chart is saying.” Before drawing each chart, write down the question it is meant to answer, such as “which category is the largest,” “are price and area correlated,” or “where are the outliers?”

The third stumbling block is not recording your cleaning steps. It’s recommended that you keep the reason for each step in your Notebook; otherwise, it will be hard to review and trace back later.

## Passing criteria

After you finish this stage, you should be able to take a CSV file and independently complete reading, cleaning, exploration, visualization, and conclusion summarization.

If you can write a data analysis report that includes at least three key charts, one data cleaning process, and three clear conclusions, then you are ready to move on to the AI math and machine learning stage.

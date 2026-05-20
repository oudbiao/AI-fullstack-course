---
title: "5.6.1 机器学习项目路线图：baseline、证据、改进"
sidebar_position: 18
description: "紧凑版机器学习项目路线图：定义问题、建立 baseline、评估、改进、分析失败并整理证据。"
keywords: [机器学习项目指南, 房价预测, 客户流失, 用户分群, Kaggle, 机器学习作品集]
---

# 5.6.1 机器学习项目路线图：baseline、证据、改进

本小章是第 5 章出口。它证明你能把一个数据问题变成可评估、可解释、可展示的建模流程。

## 先看项目闭环

![机器学习项目实践路线图](/img/course/ml-projects-roadmap.webp)

![机器学习项目作品集闭环](/img/course/ch05-projects-portfolio-loop.webp)

记住这个项目闭环：

```text
问题 -> 数据 -> baseline -> 指标 -> 改进 -> 失败样本 -> 报告
```

不要一开始就冲复杂模型。没有 baseline、指标和失败分析的项目，只是一次演示性运行。

## 保留一份实验记录

创建 `ml_project_log_first_loop.py`。这不是模型，而是每个模型项目都需要的习惯。

```python
experiments = [
    {"version": "v1_baseline", "metric": 0.72, "change": "default model"},
    {"version": "v2_features", "metric": 0.78, "change": "add ratio features"},
    {"version": "v3_tuned", "metric": 0.80, "change": "tune max_depth"},
]

best = max(experiments, key=lambda row: row["metric"])

print("best_version:", best["version"])
print("best_metric:", best["metric"])
print("next_step: inspect failure cases before adding more models")
```

预期输出：

```text
best_version: v3_tuned
best_metric: 0.8
next_step: inspect failure cases before adding more models
```

这一步是在转换思维：从“我跑了模型”变成“我能比较版本并解释下一步”。

## 按这个顺序学

| 顺序 | 阅读 | 交付什么 |
|---|---|---|
| 1 | [5.6.2 房价预测](./01-house-price.md) | 回归 baseline 和改进 |
| 2 | [5.6.3 客户流失预测](./02-customer-churn.md) | 分类指标和阈值思维 |
| 3 | [5.6.4 用户分群](./03-user-segmentation.md) | 聚类解释和业务标签 |
| 4 | [5.6.5 Kaggle 实践](./04-kaggle.md) | 真实提交流程 |
| 5 | [5.6.6 ML 实操工作坊](./05-hands-on-ml-workshop.md) | 一份完整证据包演练 |

工作坊放在最后，因为它把前面项目习惯整理成一份可复现证据包。

## 项目交付物标准

![机器学习项目报告分镜图](/img/course/ch05-project-report-storyboard.webp)

至少为一个项目保留这些文件：`README.md`、运行命令、指标表、实验记录、一个失败样本、一张图、下一步计划。

## 通过标准

能说清：我如何定义任务、用了什么 baseline、信任哪个指标、哪里变好了、模型在哪里失败、下一步做什么，就算通过。

<details>
<summary>参考答案与讲解</summary>

1. 完整回答要先定义任务类型、目标列和成功指标，再讨论模型名称。
2. baseline 应该是最简单、可复现的一版：固定划分、最少预处理、一个模型和一张指标表。
3. 改进必须和同一个划分或验证方案比较。一次同时改划分和模型，结果就很难解释。
4. 失败分析至少要指出一个模型较弱的样本类型或业务分段，并把它变成下一轮受控实验。
5. 合格的项目文件夹应该包含运行命令、README、实验日志、指标表、图表、失败样本和下一步计划。

</details>

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
project_goal: prediction, segmentation, Kaggle, or end-to-end ML portfolio target
pipeline: data split, preprocessing, model, evaluation, and report artifacts
result: metric table, chart, predictions, failure samples, and README note
failure_check: non-reproducible run, leakage, overfitting, weak baseline, or missing deployment boundary
Expected_output: ML project folder with pipeline, metrics, and failure review
```

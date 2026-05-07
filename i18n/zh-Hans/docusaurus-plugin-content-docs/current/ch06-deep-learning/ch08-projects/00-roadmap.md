---
title: "6.8.1 深度学习项目路线图：训练、检查、打包"
sidebar_position: 0
description: "紧凑版深度学习项目路线图：图像分类、情感分析、生成实践、训练证据和作品集打包。"
keywords: [深度学习项目指南, 图像分类, 情感分析, 生成实践, PyTorch 作品集]
---

# 6.8.1 深度学习项目路线图：训练、检查、打包

本小章是第 6 章出口。深度学习项目不只是训练脚本，还需要数据证据、形状检查、loss 日志、预测样本、失败案例和 README。

## 6.8.1.1 先看项目闭环

![深度学习项目作品集路线图](/img/course/ch06-projects-portfolio-loop.png)

![深度学习项目训练复盘闭环](/img/course/ch06-deep-learning-project-cycle.png)

```text
数据集 -> 模型 -> 训练日志 -> 评估 -> 失败案例 -> 打包
```

## 6.8.1.2 保留一份证据记录

创建 `dl_project_evidence_first_loop.py`。

```python
evidence = {
    "task": "image classification",
    "baseline_accuracy": 0.71,
    "current_accuracy": 0.82,
    "failure_case_count": 5,
    "next_step": "inspect confused classes and add augmentation",
}

print("task:", evidence["task"])
print("improvement:", round(evidence["current_accuracy"] - evidence["baseline_accuracy"], 3))
print("failure_case_count:", evidence["failure_case_count"])
print("next_step:", evidence["next_step"])
```

预期输出：

```text
task: image classification
improvement: 0.11
failure_case_count: 5
next_step: inspect confused classes and add augmentation
```

这就是项目习惯：每次改进都要有 baseline、指标、失败证据和下一步。

## 6.8.1.3 按这个顺序学

| 顺序 | 阅读 | 交付什么 |
|---|---|---|
| 1 | [6.8.2 图像分类](./01-image-classification.md) | 数据集、CNN/迁移 baseline、预测样本 |
| 2 | [6.8.3 情感分析](./02-sentiment-analysis.md) | 文本流程、训练日志、错误样本 |
| 3 | [6.8.4 生成实践](./03-generative-practice.md) | 生成样本和审查记录 |
| 4 | [6.8.5 DL 实操工作坊](./04-hands-on-dl-workshop.md) | 一份可复现 PyTorch 证据包 |

## 6.8.1.4 项目交付物标准

至少为一个项目保留这些文件：`README.md`、运行命令、数据集说明、模型摘要、loss 曲线或日志、指标表、预测样本、失败案例、下一步计划。

## 6.8.1.5 通过标准

另一个学习者能运行你的项目、查看训练证据、看到成功和失败样本，并理解你下一步会怎么改，就算通过。

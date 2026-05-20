---
title: "5.0 学习指南与任务单：机器学习"
sidebar_position: 1
description: "第 5 章主学习路线已经合并到章节入口页，本页保留一张简短可打印清单。"
keywords: [机器学习学习指南, sklearn, 机器学习项目, baseline, 特征工程]
---

# 5.0 学习指南与任务单：机器学习

![机器学习学习指南项目闭环](/img/course/ch05-study-guide-project-loop.webp)

主要学习路线已经放在 [第 5 章入口](./)。本页只作为练习时快速查看的清单。

## 一句话模型

```text
定义任务 -> 划分数据 -> 训练 baseline -> 评估 -> 查看错误 -> 改进
```

不知道该用哪个模型时，先做 baseline。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
modeling_loop: data, features, model, metric, error review, and next experiment
artifact: code, score, chart, pipeline, or project README
failure_check: leakage, metric mismatch, unstable split, overfitting, or unclear business target
next_action: one controlled experiment rather than many parameter changes
Expected_output: reproducible ML evidence that prepares for deep learning
```

## 练习清单

| 检查项 | 证据 |
|---|---|
| 能定义任务类型 | 问题说明 |
| 能无泄漏地划分数据 | 训练/测试划分记录 |
| 能训练 dummy baseline 和一个真实模型 | baseline 对比 |
| 能为任务选择指标 | 指标说明 |
| 能查看错误样本 | 错误样本记录 |
| 能完成证据包工作坊 | `ml_workshop_run/` |

<details>
<summary>参考答案与讲解</summary>

1. 问题说明要写清这是回归、分类、聚类、评估还是特征工程任务，以及什么算成功。
2. 安全的划分说明要解释数据何时被划分，哪些预处理步骤只在训练数据上 fit。
3. baseline 对比应该包含 dummy 或简单模型，以及一个更强模型，并使用同一套评估方案。
4. 指标说明要根据任务目标解释为什么选这个指标。不平衡分类不能只看 accuracy。
5. 错误样本要变成下一步行动，而不是只截图留档。好的下一步是受控的特征、数据、阈值或模型改动。
6. 当别人能复跑你的证据包并理解建模决策时，就可以进入第 6 章。

</details>

## 证据标准

| 产物 | 应该回答什么 |
|---|---|
| 问题说明 | 任务类型是什么，什么算成功？ |
| 划分说明 | 你怎样把测试数据和训练过程隔开？ |
| baseline 对比 | 需要超过的最低分数是多少？ |
| 指标说明 | 为什么这个指标比单纯 accuracy 更适合目标？ |
| 错误记录 | 哪些错误最重要，可能是哪些特征或标签问题导致的？ |

## 可以继续的信号

当一个表格项目包含 baseline、真实模型、指标、错误分析和别人可复现的 README 时，就可以进入第 6 章。

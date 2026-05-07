---
title: "学习指南与任务单：深度学习与 Transformer 基础怎么学最不容易学乱"
sidebar_position: 1
description: "给 AI 全栈新人的深度学习学习指南：神经网络、PyTorch、CNN、RNN、Attention、Transformer、项目路线和验收标准。"
keywords: [深度学习学习指南, PyTorch 怎么学, CNN 怎么学, Transformer 怎么学, Attention]
---

# 学习指南与任务单：深度学习与 Transformer 基础怎么学最不容易学乱

如果你来到 `第 6 章 深度学习与 Transformer 基础` 后觉得代码变长、模型变多，先把注意力放回训练闭环。深度学习第一遍最重要的是知道数据如何经过模型、损失和梯度更新参数。

## 本阶段总原则

深度学习要抓住一条主线：数据进入网络，前向传播得到输出，损失函数衡量差距，反向传播计算梯度，优化器更新参数。

![深度学习学习指南训练闭环](/img/course/ch06-study-guide-training-loop.png)

## 本阶段必须完成的任务

把这张表当成第 6 章的实操路径。本章重点不是背完所有架构，而是让一个小模型真的训练起来，记录发生了什么，并解释它为什么失败或变好。

| 任务 | 产出物 | 通过标准 |
|---|---|---|
| 理解神经网络训练闭环 | 一个手写或截图整理的训练流程图 | 能解释前向传播、损失、反向传播和参数更新 |
| 跑通 PyTorch 基础 | 一个最小训练脚本 | 能使用 Dataset、DataLoader、`nn.Module`、loss 和 optimizer |
| 完成引导式 PyTorch 证据包工作坊 | 生成的 `deep_learning_workshop_run/` 证据包 | 能复跑脚本，并解释 `training_log.csv`、`model_comparison.csv`、`loss_curve.png` 和 `shape_trace.md` |
| 完成图像或文本小任务 | 一个可运行训练项目 | 能记录训练曲线、验证指标和错误样本 |
| 理解 Attention 与 Transformer | 一份结构说明笔记 | 能解释 Query、Key、Value、Self-Attention 和位置编码 |
| 完成一个阶段项目 | 一个深度学习实践项目 | 有训练日志、指标、可复现命令和复盘 |

## 推荐学习顺序

第一轮先学历史突破和神经网络基础。重点理解感知器为什么出现、XOR 为什么让单层模型受挫、反向传播为什么重要，再理解神经元、激活函数、前向传播、反向传播、损失函数、优化器和正则化。

第二轮学 PyTorch。不要只复制代码，要知道张量、自动求导、`nn.Module`、Dataset、DataLoader 和训练循环分别负责什么。

第三轮学 CNN。图像分类最直观，适合第一次把网络结构和任务联系起来。

第四轮学 RNN 和序列模型。它们帮助你理解序列任务，也为 Transformer 的出现提供历史背景。

第五轮学 Attention 和 Transformer。这是进入大模型主线前最关键的桥。

生成模型和训练技巧可以作为扩展，不必在第一遍全部吃透。

进入更大的项目之前，建议先完成 [实操工作坊：构建 PyTorch 训练证据包](./ch08-projects/04-hands-on-dl-workshop.md)。它会把抽象训练循环变成一个可运行脚本，并生成日志、曲线、checkpoint、shape trace 和复盘样本。

## 建议学习节奏

| 内容类型 | 建议时间 | 学习目标 |
|---|---|---|
| 神经网络基础 | 3～6 小时 | 能解释训练闭环 |
| PyTorch 基础 | 6～10 小时 | 能写最小训练循环 |
| CNN / RNN | 4～8 小时 | 能理解不同数据结构对应的网络 |
| Transformer | 4～8 小时 | 能解释 Attention 的基本直觉 |
| 项目页 | 10～20 小时 | 完成一个可训练、可评估的小模型 |

## 阶段项目路线

在选择更大的项目主题前，建议先跑一遍 [PyTorch 证据包工作坊](./ch08-projects/04-hands-on-dl-workshop.md)。把它当作热身：你会生成数据、追踪 shape、训练 baseline、训练 CNN、做验证、保存曲线，并写出项目证据。

第一个项目建议做手写数字或小型图像分类，练习 Dataset、DataLoader、CNN、训练和评估。

第二个项目建议做文本情感分类，练习序列输入、Embedding 和基础文本模型。

第三个项目可以做 Transformer 结构阅读或小实验，重点理解 Attention 输入输出和上下文建模。

## 常见卡点

最常见的卡点是 loss、梯度和优化器串不起来。你可以用一个极小模型和几条样本，打印每一步的输入、输出、loss 和参数变化。

第二个卡点是 PyTorch 代码模板太长。建议先写最小训练循环，再逐步封装函数，不要一开始就追求工程化。

第三个卡点是模型效果不好。先查数据、标签、学习率、loss 是否下降，再考虑换模型。

## 阶段作品集交付物

![深度学习训练证据流水线图](/img/course/ch06-hands-on-training-evidence-pipeline.png)

如果你想把本阶段成果沉淀到作品集，建议至少保留下面这些文件或等价材料。

| 交付物 | 说明 |
|---|---|
| `train.py` | PyTorch 训练脚本，包含 Dataset、DataLoader、模型、loss 和 optimizer |
| `config.yaml` | 学习率、batch size、epoch、模型结构等实验配置 |
| `training_log.csv` | 每轮 loss、指标、耗时和验证结果 |
| `curves/` | 训练曲线、验证曲线、混淆矩阵或预测可视化 |
| `failure_cases.md` | 错误样本、过拟合/欠拟合现象和改进动作 |
| `README.md` | 数据说明、运行命令、模型结果、限制和复盘 |

## 阶段通关问题

学完本阶段后，你应该能从零写出一个 PyTorch 训练脚本，训练一个简单模型，画出 loss 变化，并解释模型为什么这样更新。

进入第 7 章前，确认自己能回答这些问题：

- 为什么需要反向传播？
- optimizer 更新的是什么？
- Dataset 和 DataLoader 分别解决什么问题？
- Attention 为什么能建模上下文？
- Transformer 和后续大模型有什么关系？

如果你能回答这些问题，并说清楚 CNN、RNN、Transformer 分别解决什么问题，就可以进入大模型原理阶段。

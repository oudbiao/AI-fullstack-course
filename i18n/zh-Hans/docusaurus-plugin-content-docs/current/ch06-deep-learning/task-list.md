---
title: "阶段学习任务单"
description: "把深度学习与 Transformer 基础阶段拆成可执行的学习任务、练习产出和通关标准。"
keywords: [深度学习, PyTorch, Transformer, CNN, 学习任务单]
---

# 阶段学习任务单：深度学习与 Transformer 基础

这个阶段的目标是让你理解神经网络如何训练、如何用 PyTorch 搭建模型，以及 Transformer 为什么成为现代大模型的基础。不要急着追求大模型训练，先把张量、模型、损失、优化器、训练循环和评估流程跑清楚。

## 本阶段必须完成的任务

| 任务 | 产出物 | 通过标准 |
| --- | --- | --- |
| 理解神经网络训练闭环 | 一个手写训练流程图 | 能解释前向传播、损失、反向传播和参数更新 |
| 跑通 PyTorch 基础 | 一个最小训练脚本 | 能使用 Dataset、DataLoader、nn.Module 和 optimizer |
| 完成引导式 PyTorch 证据包工作坊 | 生成的 `deep_learning_workshop_run/` 证据包 | 能复跑脚本，并解释 `training_log.csv`、`model_comparison.csv`、`loss_curve.png` 和 `shape_trace.md` |
| 完成图像或文本小任务 | 一个可运行训练项目 | 能记录训练曲线、验证指标和错误样本 |
| 理解 Attention 与 Transformer | 一份结构说明笔记 | 能解释 Query、Key、Value、Self-Attention 和位置编码 |
| 完成阶段项目 | 一个深度学习实践项目 | 有训练日志、指标、可复现命令和复盘 |

## 推荐学习顺序

先理解神经网络训练过程，再学习 PyTorch 基础，然后学习 CNN/RNN/Attention/Transformer。不要把 Transformer 当成孤立公式，它本质上是在处理序列信息、上下文关系和并行计算效率。

当你需要一个具体可运行检查点时，可以在大项目之前先做 [实操工作坊：构建 PyTorch 训练证据包](./ch08-projects/04-hands-on-dl-workshop.md)。

写 PyTorch 代码时，优先关注数据形状。大多数初学错误都和 tensor shape、batch 维度、loss 输入格式、device 不一致有关。每写一个模块，都建议打印一次输入输出形状。

## 和 AI 学习助手项目的关系

本阶段对应 AI 学习助手的 v0.5 表示学习理解版。你不一定要为学习助手训练大模型，但应该理解 embedding、序列建模和 Transformer 的基础，这会直接影响后续 RAG、Prompt、微调和 Agent 的理解。

建议做一个小实验：用简单文本分类或相似度任务观察不同文本表示方法的效果。重点不是追求高分，而是理解“文本如何变成向量”“向量相似度为什么能用于检索”。

## 常见卡点

常见问题包括张量维度不匹配、训练 loss 不下降、验证集效果很差、学习率过大或过小、过拟合、GPU/CPU device 不一致、把训练指标误当成泛化能力。遇到训练问题时，先用小数据集过拟合测试，确认代码能学到东西，再扩大数据。


## 轻松版 / 标准版 / 挑战版任务

| 难度 | 你要完成什么 | 适合谁 |
|---|---|---|
| 轻松版 | 跑通一个最小训练循环 | 第一遍学习、时间少或刚入门的学习者 |
| 标准版 | 保存训练曲线和验证指标 | 希望把本阶段放进作品集的学习者 |
| 挑战版 | 制造并修复一次 shape mismatch 或 loss 不降问题 | 已有基础、想做更强项目证据的学习者 |

## 本阶段徽章与 Boss 战

| 类型 | 内容 |
|---|---|
| Boss 战 | Shape 巨兽 |
| 可解锁徽章 | Loss 观察员、Shape 追踪者 |
| 最小通关口号 | 先跑通、再解释、再记录失败 |
| 证据保存建议 | 把截图、日志、失败样本或评估表保存到 `reports/`、`evals/` 或 `logs/` |

完成轻松版就可以继续前进；完成标准版才建议写进作品集；挑战版只在你有余力时再做。

## 阶段作品集交付物

如果你想把本阶段成果沉淀到作品集，建议至少保留下面这些文件或等价材料。

| 交付物 | 说明 |
| --- | --- |
| `train.py` | PyTorch 训练脚本，包含 Dataset、DataLoader、模型、loss 和 optimizer |
| `config.yaml` | 学习率、batch size、epoch、模型结构等实验配置 |
| `training_log.csv` | 每轮 loss、指标、耗时和验证结果 |
| `curves/` | 训练曲线、验证曲线、混淆矩阵或预测可视化 |
| `failure_cases.md` | 错误样本、过拟合/欠拟合现象和改进动作 |
| `README.md` | 数据说明、运行命令、模型结果、限制和复盘 |

这些材料会让深度学习项目从“训练跑起来”升级成“能诊断训练过程、能复现实验、能解释模型失败”。

## 阶段通关问题

学完后，你应该能回答这些问题：为什么需要反向传播，optimizer 更新的是什么，Dataset 和 DataLoader 分别解决什么问题，Attention 为什么能建模上下文，Transformer 和后续大模型有什么关系。

## 完成状态 Checklist

- [ ] 我能解释前向传播、损失、反向传播和参数更新。
- [ ] 我能用 PyTorch 写出 Dataset、DataLoader、nn.Module 和训练循环。
- [ ] 我能记录训练曲线、验证指标和错误样本。
- [ ] 我能说明 Attention 和 Transformer 解决了什么问题。
- [ ] 我已经完成一个深度学习小项目，并能解释它的输入、输出和失败原因。

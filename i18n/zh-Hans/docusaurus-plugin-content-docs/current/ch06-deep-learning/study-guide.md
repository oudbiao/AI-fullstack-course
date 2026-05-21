---
title: "6.0 学习指南与任务单：深度学习与 Transformer 基础"
sidebar_position: 1
description: "第 6 章主学习路线已经合并到章节入口页，本页保留一张简短可打印清单。"
keywords: [深度学习学习指南, PyTorch, CNN, Transformer, Attention]
---

# 6.0 学习指南与任务单：深度学习与 Transformer 基础

![深度学习学习指南训练闭环](/img/course/ch06-study-guide-training-loop.webp)

主要学习路线已经放在 [第 6 章入口](./)。本页只作为练习时快速查看的清单。

## 一句话模型

```text
batch 数据 -> 模型前向 -> 损失 -> 反向传播梯度 -> 优化器更新 -> 曲线
```

如果代码看起来很长，先找出这六步。

## 预期最终输出

第 6 章结束时，你应该留下一个小证据文件夹，而不只是读完笔记：

```text
deep_learning_evidence/
  shape_trace.txt
  training_log.csv
  loss_curve.png
  best_checkpoint_note.md
  attention_note.md
  failure_sample_note.md
```

如果这个文件夹还没有出现，即使页面都读完了，第 6 章也还没有真正完成。

## 练习清单

| 检查项 | 证据 |
|---|---|
| 能解释 forward、loss、backward、optimizer | 训练闭环说明 |
| 能运行一个最小 PyTorch 脚本 | `train.py` |
| 能打印模型中的 tensor shape | shape trace |
| 能对比训练曲线和验证曲线 | 曲线图片或 CSV |
| 能解释 Attention 改变了什么 | attention 说明 |
| 能完成证据包工作坊 | `deep_learning_workshop_run/` |

<details>
<summary>检查思路与讲解</summary>

这张清单的目标不是让你背概念，而是确认你能留下可检查的学习证据：

1. 训练闭环说明应包含 forward、loss、backward、optimizer 四步，并说明每一步改变了什么。
2. 最小 PyTorch 脚本应能独立运行，至少包含数据、模型、loss、optimizer 和训练循环。
3. Shape trace 要覆盖输入、关键中间层和输出，能解释 batch、channel、sequence 等维度含义。
4. 曲线或 CSV 应能支持诊断，例如过拟合、欠拟合、学习率不稳定或数据问题。
5. Attention 说明应讲清楚它如何让模型按上下文动态选择信息，而不只是写出公式。
6. 证据包工作坊应包含代码、运行日志、图表和复盘说明，方便别人复现你的结论。

</details>

## 证据标准

| 产物 | 应该回答什么 |
|---|---|
| 训练闭环说明 | forward、loss、backward、optimizer step 分别发生了什么？ |
| shape trace | tensor shape 在模型里怎样变化？ |
| 曲线图片或 CSV | 模型是在欠拟合、过拟合，还是稳定变好？ |
| attention 说明 | Attention 增加了什么信息，还有什么依然困难？ |
| 失败样本记录 | 哪个样本失败了，这说明数据、模型还是标签哪里有问题？ |

## 留下的证据

离开第 6 章前，保留一个紧凑证据包：

```text
形状 trace：一个模型及打印出的张量形状
训练日志：随时间变化的训练和验证损失
最佳检查点：如何选择最佳模型
注意力说明：Q/K/V、mask 和下一 token 桥接
失败样本：一个错误或较弱的预测及下一步动作
项目文件夹：可运行的证据包或 README
```

## 可以继续的信号

当你能训练一个小模型、保存训练日志、查看失败样本，并解释模型为什么变好或失败时，就可以进入第 7 章。

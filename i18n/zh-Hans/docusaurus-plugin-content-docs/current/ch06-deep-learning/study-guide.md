---
title: "6.0 学习指南与任务单：深度学习与 Transformer 基础"
sidebar_position: 1
description: "第 6 章主学习路线已经合并到章节入口页，本页保留一张简短可打印清单。"
keywords: [深度学习学习指南, PyTorch, CNN, Transformer, Attention]
---

# 6.0 学习指南与任务单：深度学习与 Transformer 基础

![深度学习学习指南训练闭环](/img/course/ch06-study-guide-training-loop.png)

主要学习路线已经放在 [第 6 章入口](./)。本页只作为练习时快速查看的清单。

## 一句话模型

```text
batch 数据 -> 模型前向 -> 损失 -> 反向传播梯度 -> 优化器更新 -> 曲线
```

如果代码看起来很长，先找出这六步。

## 练习清单

| 检查项 | 证据 |
|---|---|
| 能解释 forward、loss、backward、optimizer | 训练闭环说明 |
| 能运行一个最小 PyTorch 脚本 | `train.py` |
| 能打印模型中的 tensor shape | shape trace |
| 能对比训练曲线和验证曲线 | 曲线图片或 CSV |
| 能解释 Attention 改变了什么 | attention 说明 |
| 能完成证据包工作坊 | `deep_learning_workshop_run/` |

## 可以继续的信号

当你能训练一个小模型、保存训练日志、查看失败样本，并解释模型为什么变好或失败时，就可以进入第 7 章。

---
title: "6 深度学习与 Transformer 基础"
sidebar_position: 0
description: "学习实用深度学习闭环：张量、模型、损失、反向传播、优化器、曲线、CNN/RNN/Transformer 和小项目。"
keywords: [深度学习, PyTorch, 神经网络, CNN, RNN, Transformer, Attention]
---

# 6 深度学习与 Transformer 基础

![深度学习与 Transformer 主视觉](/img/course/ch06-deep-learning.png)

第 6 章只解决一件事：理解模型怎样通过**损失、梯度和反复训练步骤**学会东西。

## 6.0.1 先看训练闭环

![深度学习训练闭环主图](/img/course/ch06-training-loop-backbone.png)

先看图。大多数深度学习训练代码都是这个闭环：

```text
batch 数据 -> 模型前向 -> 损失 -> 反向传播梯度 -> 优化器更新 -> 曲线
```

不要一开始追大模型。先让一个小模型跑起来，记录发生了什么，并解释它为什么变好或失败。

## 6.0.2 学习顺序与任务表

下面这一张表同时作为本章学习指南和任务清单。

| 页面 | 跟着做 | 留下的证据 |
|---|---|---|
| [6.1 神经网络基础](ch01-nn-basics/00-roadmap.md) | 理解神经元、激活函数、前向/反向传播、优化器、正则化和初始化 | 一份手写训练闭环说明 |
| [6.1.2 深度学习历史](ch01-nn-basics/06-history-breakthroughs.md) | 可选背景：浏览 backprop、CNN、RNN、Attention、Transformer 为什么出现 | 一条“这个架构为什么存在”的说明 |
| [6.2 PyTorch](ch02-pytorch/00-roadmap.md) | 练习 tensor、autograd、`nn.Module`、Dataset、DataLoader 和最小训练循环 | 一个可运行 PyTorch 脚本 |
| [6.3 CNN](ch03-cnn/00-roadmap.md) | 用图像分类理解数据形状、卷积、池化和迁移学习 | shape 记录和一次图像分类运行 |
| [6.4 RNN](ch04-rnn/00-roadmap.md) | 理解序列数据为什么需要记忆，以及 LSTM/GRU 在 Transformer 前解决了什么 | 一条序列模型说明 |
| [6.5 Transformer](ch05-transformer/00-roadmap.md) | 学 Query、Key、Value、自注意力、位置编码和 Transformer block | 一张 attention 输入/输出图 |
| [6.6 生成模型](ch06-generative/00-roadmap.md) 和 [6.7 训练技巧](ch07-training-tips/00-roadmap.md) | 在训练闭环稳定后作为扩展学习 | 一条调参或诊断记录 |
| [6.8 项目](ch08-projects/00-roadmap.md) 和 [6.8.5 工作坊](ch08-projects/04-hands-on-dl-workshop.md) | 在图像、情感或生成项目之前，先做 PyTorch 证据包 | 日志、曲线、checkpoint、shape trace、README |

本章常见术语：

| 术语 | 含义 |
|---|---|
| `tensor` | PyTorch 使用的多维数组 |
| `forward` | 数据经过模型得到预测 |
| `loss` | 衡量预测错误的数字 |
| `backward` | 从损失计算梯度 |
| `optimizer` | 使用梯度更新参数 |
| `epoch` | 完整看完一遍训练数据 |
| `batch` | 一次一起处理的一小组样本 |

## 6.0.3 第一个可运行闭环

如果还没有 PyTorch，请先用官方选择器安装。PyTorch 可用后，运行下面这个极小训练循环：

```python
import torch
from torch import nn

torch.manual_seed(42)
x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
y = torch.tensor([[0.0], [2.0], [4.0], [6.0]])

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(20):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch in {0, 1, 5, 19}:
        print(epoch, round(loss.item(), 4))
```

预期形态：

```text
0 ...
1 ...
5 ...
19 ...
```

具体数字可能不同，但 loss 应该整体下降。只要下降，你就看到了训练闭环在工作。

## 6.0.4 常见失败

| 现象 | 先检查什么 | 常见修复 |
|---|---|---|
| shape 不匹配 | 输入形状、batch 维度、输出类别数 | 在每层打印 tensor shape |
| loss 不下降 | 学习率、标签、归一化、损失函数 | 先尝试过拟合一个小 batch |
| 训练好、验证差 | 过拟合或数据划分问题 | 加验证曲线、数据增强、正则化、early stopping |
| 显存不足 | batch 大小、图像尺寸、模型规模 | 降低 batch/分辨率，或用更小模型 |
| Transformer 抽象 | Q/K/V 和序列长度 | 写代码前先画一张 attention 表 |

## 6.0.5 通关检查

能回答下面五个问题，就可以进入第 7 章：

- `forward`、`loss.backward()`、`optimizer.step()` 分别做什么？
- Dataset 和 DataLoader 分别解决什么问题？
- 训练曲线和验证曲线怎样暴露过拟合？
- Attention 为什么能建模上下文？
- Transformer 和后面的大模型有什么关系？

需要打印式清单时，打开 [6.0 学习指南与任务单](./study-guide.md)。后面的大模型、RAG 和多模态模型都会建立在这些表示学习概念上。

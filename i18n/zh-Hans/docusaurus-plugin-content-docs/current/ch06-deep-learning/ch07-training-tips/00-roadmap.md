---
title: "6.7.1 训练技巧路线图：先诊断，再改东西"
sidebar_position: 0
description: "紧凑版深度学习训练技巧路线图：调参、诊断、压缩和基于证据的决策。"
keywords: [深度学习训练技巧, 超参数调优, 训练诊断, 模型压缩]
---

# 6.7.1 训练技巧路线图：先诊断，再改东西

训练技巧只有在回答诊断问题时才有用。不要同时改优化器、学习率、模型大小和数据。

## 先看诊断流程

![深度学习训练技巧章节关系图](/img/course/ch06-training-tips-chapter-flow.webp)

![训练诊断仪表盘图](/img/course/ch06-training-diagnosis-dashboard-map.webp)

| 现象 | 先检查 |
|---|---|
| 训练 loss 很高 | 模型太小、学习率太低、数据有问题 |
| 训练好、验证差 | 过拟合、泄漏、增强不足 |
| loss 不稳定 | 学习率太高、batch 异常、梯度爆炸 |
| 太慢 | batch size、device、模型大小 |
| 部署太重 | 压缩、量化、剪枝 |

## 读一段极小 loss 记录

创建 `training_tips_first_loop.py`。

```python
val_loss = [0.62, 0.51, 0.48, 0.49, 0.53]
best_epoch = min(range(len(val_loss)), key=val_loss.__getitem__) + 1

print("best_epoch:", best_epoch)
print("best_val_loss:", val_loss[best_epoch - 1])
print("action: stop or reduce learning rate if validation keeps worsening")
```

预期输出：

```text
best_epoch: 3
best_val_loss: 0.48
action: stop or reduce learning rate if validation keeps worsening
```

![训练技巧首段 loss 输出结果图](/img/course/ch06-training-tips-first-loop-result-map.webp)

加技巧之前，先读曲线。一段简单日志通常已经能告诉你下一步该试什么。

## 按这个顺序学

| 顺序 | 阅读 | 练什么 |
|---|---|---|
| 1 | [6.7.2 超参数调优](./01-hyperparameter-tuning.md) | 学习率、batch size、优化器 |
| 2 | [6.7.3 训练诊断](./02-training-diagnosis.md) | loss 曲线、过拟合、不稳定 |
| 3 | [6.7.4 模型压缩](./03-model-compression.md) | 更小、更快、更适合部署的模型 |

## 留下的证据

训练技巧小章结束后，保留一条诊断决策记录：

```text
可见症状：曲线或输出显示了什么？
首次检查：数据、形状、梯度，或验证集划分
单一变更：哪个单个设置发生了变化？
前后对比：指标或工件的比较
决策：保留、调整、回滚或调查
```

## 通过标准

能看一条训练/验证曲线，并带理由选择一个下一步动作，就算通过。

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要把 tensor、模型层、loss、`backward()` 和 optimizer 更新连成一个训练闭环。
2. 证据应包含可运行的小实验、tensor shape 检查，以及能解释的 loss 或验证曲线。
3. 自检时要能指出一个失败模式，例如 shape 不匹配、loss 不下降、过拟合、数据泄漏，或只会说 Attention/Transformer 名词却讲不出数据流。

</details>

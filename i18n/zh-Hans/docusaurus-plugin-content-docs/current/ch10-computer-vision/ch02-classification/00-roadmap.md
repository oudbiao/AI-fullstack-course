---
title: "10.2.1 图像分类路线图：图像输入、标签输出"
sidebar_position: 0
description: "图像分类的简短实操路线：学习数据增强、网络结构、训练检查和整图标签的失败分析。"
keywords: [图像分类指南, 数据增强, ResNet, 训练技巧]
---

# 10.2.1 图像分类路线图：图像输入、标签输出

图像分类回答一个问题：给定一整张图，它最像哪个类别？

## 10.2.1.1 先看分类闭环

![图像分类章节学习流程图](/img/course/ch10-classification-chapter-flow.png)

![图像分类架构演化图](/img/course/ch10-classification-architecture-evolution-map.png)

![分类训练诊断图](/img/course/ch10-classification-training-diagnosis-map.png)

分类是最简单的视觉输出，但它仍然依赖数据划分、增强、架构、loss、指标和错误样例。

## 10.2.1.2 跑一个预测检查

这个脚本模拟分类器最后一步：选择分数最高的标签。

```python
labels = ["cat", "dog", "panda"]
scores = [0.12, 0.74, 0.14]

best_index = max(range(len(scores)), key=lambda index: scores[index])

print("prediction:", labels[best_index])
print("confidence:", scores[best_index])
```

预期输出：

```text
prediction: dog
confidence: 0.74
```

真实项目里不要只展示 top class。保留 confidence、错误样例和混淆模式。

## 10.2.1.3 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 数据增强 | 解释哪些变化保持类别，哪些变化会带来风险 |
| 2 | 现代架构 | 比较特征提取器、分类头和预训练 backbone |
| 3 | 训练技巧 | 追踪划分、loss、accuracy、过拟合和错误样例 |

## 10.2.1.4 通过标准

如果你能运行一个最小分类器，展示训练/验证指标，并解释至少一张失败图片，就通过了本章。

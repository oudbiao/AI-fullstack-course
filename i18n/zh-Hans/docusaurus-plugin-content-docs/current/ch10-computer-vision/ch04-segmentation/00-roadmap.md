---
title: "10.4.1 图像分割路线图：像素级区域"
sidebar_position: 0
description: "图像分割的简短实操路线：理解 mask、语义分割、实例分割、IoU 和边界失败。"
keywords: [图像分割指南, 语义分割, 实例分割, mask]
---

# 10.4.1 图像分割路线图：像素级区域

分割比检测更细。它不是输出框，而是输出 mask，说明哪些像素属于某个类别或实例。

## 10.4.1.1 先看 Mask 工作流

![图像分割章节学习顺序图](/img/course/ch10-segmentation-chapter-flow.png)

![语义分割 mask 示例](/img/course/semantic-segmentation-mask.png)

![语义分割 IoU 与边界图](/img/course/ch10-semantic-segmentation-iou-boundary-map.png)

本章最重要的对象是 mask。常见失败通常是边界质量、小目标、遮挡或类别混淆。

## 10.4.1.2 跑一个 Mask IoU 检查

这个脚本比较两个极小二值 mask。

```python
truth = [
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
]

pred = [
    [1, 0, 0],
    [1, 1, 0],
    [0, 0, 0],
]

intersection = 0
union = 0
for y in range(3):
    for x in range(3):
        intersection += truth[y][x] == 1 and pred[y][x] == 1
        union += truth[y][x] == 1 or pred[y][x] == 1

print("mask_iou:", round(intersection / union, 3))
```

预期输出：

```text
mask_iou: 0.5
```

分割报告要展示 mask、指标和边界错误，而不只是彩色叠加图。

## 10.4.1.3 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 语义分割 | 为每个像素预测一个类别 |
| 2 | 实例分割 | 分开同类别的不同对象 |
| 3 | 分割实战 | 比较 mask、IoU/Dice、边界错误和失败样例 |

## 10.4.1.4 通过标准

如果你能创建或检查一个 mask，计算简单重叠指标，并解释一个边界或类别混淆失败，就通过了本章。

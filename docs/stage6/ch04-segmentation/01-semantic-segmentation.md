---
title: "4.1 语义分割"
sidebar_position: 11
description: "从“整图一个标签”走到“每个像素一个标签”，理解语义分割为什么是更细粒度的视觉理解任务。"
keywords: [semantic segmentation, pixel classification, mask, IoU, vision]
---

# 语义分割

:::tip 本节定位
分类回答的是：

- 这张图是什么

检测回答的是：

- 图里有什么、它在哪

语义分割再进一步回答：

> **图里的每个像素属于什么类别。**

这让视觉理解进入更细粒度层面。
:::

## 学习目标

- 理解语义分割和分类/检测的区别
- 理解分割 mask 为什么更细粒度
- 通过可运行示例理解像素级标签和 IoU
- 建立语义分割的基本任务直觉

---

## 一、语义分割到底在做什么？

它的目标是：

- 给图像中每个像素分一个类别

例如：

- 天空
- 路面
- 人
- 车

### 为什么它比检测更细？

因为检测只给框，  
分割会更精确地给出区域边界。

---

## 二、先跑一个最小分割 mask 示例

```python
pred_mask = [
    [0, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
]

gt_mask = [
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 1],
]


def iou_for_class(pred, gt, target_class):
    inter = 0
    union = 0
    for pred_row, gt_row in zip(pred, gt):
        for p, g in zip(pred_row, gt_row):
            if p == target_class and g == target_class:
                inter += 1
            if p == target_class or g == target_class:
                union += 1
    return inter / union if union else 0.0


print("IoU for class 1:", round(iou_for_class(pred_mask, gt_mask, 1), 4))
```

### 2.1 这个例子最关键的直觉是什么？

分割评估不是看“整图对不对”，  
而是看：

- 区域重叠得好不好

这就是为什么 IoU 在分割里也非常重要。

---

## 三、最容易踩的坑

### 3.1 边界不准

分割模型很容易在物体边缘出错。

### 3.2 类别极度不平衡

背景往往太多，  
小目标类别很容易被忽略。

### 3.3 只看总体像素准确率

像素准确率高，不代表小类别真的分得好。

---

## 小结

这节最重要的是建立一个判断：

> **语义分割是在做像素级分类，因此它比分类和检测都更细粒度，也更依赖区域级评估。**

---

## 练习

1. 自己改一组 `pred_mask`，观察 IoU 会怎么变。
2. 为什么像素准确率高，不一定说明分割模型真的好？
3. 语义分割和目标检测最大的差别是什么？
4. 如果类别非常不平衡，你最担心什么问题？

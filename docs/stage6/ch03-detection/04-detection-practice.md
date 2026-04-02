---
title: "3.4 检测实战"
sidebar_position: 10
description: "围绕一个安防检测任务，走完目标定义、标注、评估和最小部署思路，建立检测项目闭环。"
keywords: [detection practice, computer vision project, bounding box, evaluation, IoU, mAP]
---

# 检测实战

:::tip 本节定位
真正做检测项目时，困难往往不只是模型本身。  
更常见的问题是：

- 标注怎么做
- 正负样本怎么定义
- 小目标怎么评估
- 检测框到底算不算对

所以这一节的重点，是把一个最小检测项目从问题定义走到评估闭环。
:::

## 学习目标

- 学会定义一个最小目标检测项目
- 理解检测项目里的标注、框匹配和评估逻辑
- 通过可运行示例建立 IoU 驱动的评估直觉
- 建立检测项目的展示骨架

---

## 一、一个检测项目最先要定什么？

### 1.1 类别边界

例如安防场景里你可能只先做：

- person
- helmet

而不是一开始就把所有目标都做进来。

### 1.2 标注规范

必须先说清：

- 框紧不紧
- 遮挡怎么标
- 小目标怎么算

### 1.3 评估标准

至少要明确：

- IoU 阈值
- 召回 / 精确率

---

## 二、先跑一个最小匹配评估示例

```python
ground_truth = [
    {"label": "person", "box": (10, 10, 30, 50)},
    {"label": "helmet", "box": (14, 8, 24, 18)},
]

predictions = [
    {"label": "person", "box": (11, 12, 31, 48), "score": 0.92},
    {"label": "helmet", "box": (15, 9, 23, 17), "score": 0.81},
    {"label": "helmet", "box": (40, 40, 50, 50), "score": 0.30},
]


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0.0


matches = []
for pred in predictions:
    best_iou = 0.0
    best_gt = None
    for gt in ground_truth:
        if gt["label"] != pred["label"]:
            continue
        cur_iou = iou(pred["box"], gt["box"])
        if cur_iou > best_iou:
            best_iou = cur_iou
            best_gt = gt
    matches.append(
        {
            "label": pred["label"],
            "score": pred["score"],
            "best_iou": round(best_iou, 4),
            "matched": best_iou >= 0.5,
        }
    )

print(matches)
```

### 2.1 这段代码最重要的地方是什么？

它让你看到检测评估不是：

- 分类对了就行

而是：

- 类别对
- 框也要足够准

### 2.2 为什么这就是很多检测项目的核心判断？

因为真实检测结果好不好，  
最终常常就体现在：

- 匹配阈值
- 框质量

---

## 三、检测项目最容易踩的坑

### 3.1 标注标准不一致

这会直接把训练和评估一起拖乱。

### 3.2 小目标和遮挡没单独分析

很多系统在这些场景下会明显掉表现。

### 3.3 只展示一两张漂亮图

真实项目更该展示：

- 哪些情况容易漏检
- 哪些情况容易误检

---

## 四、小结

这节最重要的是建立一个项目意识：

> **检测项目的关键，不只是模型名，而是类别定义、标注规范和框级评估方法是否清楚。**

---

## 练习

1. 调整 IoU 阈值到 `0.7`，看看匹配结果会怎么变。
2. 想一想：为什么检测项目比分类项目更依赖清晰标注规范？
3. 如果项目里总漏小目标，你会优先检查数据、输入分辨率还是模型结构？
4. 你会如何把这个检测项目包装成作品集？

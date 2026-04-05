---
title: "3.4 YOLO 系列"
sidebar_position: 9
description: "从单阶段检测思路讲起，理解 YOLO 为什么能把目标检测做得更实时，以及它在工程上为什么这么受欢迎。"
keywords: [YOLO, one-stage detector, object detection, NMS, realtime vision]
---

# YOLO 系列

:::tip 本节定位
YOLO 之所以火，不只是因为它能检测目标，  
而是因为它把目标检测做成了更偏工程可用的形式：

- 更快
- 更直接
- 更适合实时场景

所以这节要抓的不是版本号，而是它代表的这条路线：

> **把检测尽量做成一步到位。**
:::

## 学习目标

- 理解 YOLO 属于哪类检测器
- 理解单阶段检测和两阶段检测的主要差别
- 理解置信度、框筛选和 NMS 的基本作用
- 建立 YOLO 在工程部署里的价值判断

---

## 一、YOLO 的核心思路是什么？

### 1.1 单阶段检测

YOLO 想做的是：

- 不分两步
- 直接从图像一次性输出类别和框

### 1.2 为什么这很有吸引力？

因为它减少了：

- 额外 proposal 阶段
- 更复杂的检测流水线

所以更容易做到：

- 实时

### 1.3 一个类比

两阶段检测像先圈可疑区域，再派人逐一核查。  
YOLO 更像一眼扫过去，同时报出：

- 哪有目标
- 是什么

---

## 二、YOLO 的输出大致长什么样？

通常可以粗略理解成一组候选框，每个候选都带：

- 类别
- 置信度
- 边界框坐标

后面再通过筛选和 NMS，得到最终结果。

---

## 三、先跑一个最小 NMS 直觉示例

```python
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
    return inter_area / union if union > 0 else 0.0


predictions = [
    {"box": (10, 10, 30, 30), "score": 0.95},
    {"box": (12, 12, 31, 31), "score": 0.88},
    {"box": (60, 60, 90, 90), "score": 0.91},
]


def nms(preds, iou_threshold=0.5):
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    kept = []

    while preds:
        best = preds.pop(0)
        kept.append(best)
        preds = [
            pred for pred in preds
            if iou(best["box"], pred["box"]) < iou_threshold
        ]

    return kept


print(nms(predictions))
```

### 3.1 这个例子最关键的价值是什么？

它说明检测输出并不是直接就能用。  
很多时候模型会给出：

- 一堆重叠候选框

而 NMS 的作用就是：

- 保留最有代表性的那几个

### 3.2 为什么这对 YOLO 特别重要？

因为 YOLO 这种单阶段路线天然会产生很多候选，  
后处理筛选就是整个检测链的一部分。

---

## 四、为什么 YOLO 在工程上这么受欢迎？

### 4.1 实时性强

很多场景直接要求：

- 摄像头实时检测
- 边缘设备快速响应

YOLO 这类路线很适合这种需求。

### 4.2 结构相对统一

对很多工程同学来说，它比复杂多阶段管线更容易落地。

### 4.3 社区和工程生态成熟

这让它在真实项目里更常被优先尝试。

---

## 五、最容易踩的坑

### 5.1 误区一：YOLO 就等于目标检测

YOLO 是重要路线，但不是全部。

### 5.2 误区二：速度快就一定最适合

还要看：

- 小目标表现
- 框定位质量
- 部署约束

### 5.3 误区三：后处理不重要

NMS、阈值设置这些后处理会直接影响最终体验。

---

## 小结

这节最重要的是建立一个工程判断：

> **YOLO 代表的是单阶段、实时友好的检测路线，它之所以广泛流行，不只是因为“能检测”，而是因为“更容易在工程里快速检测”。**

---

## 练习

1. 调整示例里的 `iou_threshold`，看看保留框数如何变化。
2. 用自己的话解释：为什么单阶段检测更容易做到实时？
3. 为什么 NMS 对检测任务很重要？
4. 想一想：什么时候你可能不会优先选 YOLO？

---
title: "10.3.1 目标检测路线图：类别加框"
sidebar_position: 0
description: "目标检测的简短实操路线：理解框、IoU、阈值、YOLO 式输出和检测失败分析。"
keywords: [目标检测指南, YOLO, IoU, mAP]
---

# 10.3.1 目标检测路线图：类别加框

目标检测在分类上增加位置：图里有什么对象，它在哪里？

## 先看框工作流

![目标检测章节学习流程图](/img/course/ch10-detection-chapter-flow.webp)

![目标检测输出图](/img/course/object-detection-output.webp)

![检测输出 IoU 错误图](/img/course/ch10-detection-output-iou-error-map.webp)

重要概念是 bounding box、class、confidence、IoU、threshold、false positive、false negative 和 mAP。

## 跑一个 IoU 检查

IoU 衡量预测框和真实框重叠多少。

```python
truth = (10, 10, 50, 50)
pred = (20, 20, 60, 60)

def area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

ix1 = max(truth[0], pred[0])
iy1 = max(truth[1], pred[1])
ix2 = min(truth[2], pred[2])
iy2 = min(truth[3], pred[3])
intersection = area((ix1, iy1, ix2, iy2))
union = area(truth) + area(pred) - intersection

print("iou:", round(intersection / union, 3))
```

预期输出：

```text
iou: 0.391
```

检测调试从打印框和指标开始。不要凭一张漂亮截图判断检测质量。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 检测概览 | 解释 box、class、confidence、IoU、mAP |
| 2 | 经典检测器 | 比较 two-stage 和 one-stage 思路 |
| 3 | YOLO | 理解网格预测、阈值、NMS 和速度取舍 |
| 4 | 检测实战 | 记录误报、漏检和阈值变化 |

## 通过标准

如果你能用框、confidence、IoU 和至少一个误报或漏检案例解释检测结果，就通过了本章。

<details>
<summary>参考答案与讲解</summary>

1. 合格答案要把任务映射到正确的视觉输出：类别标签、检测框、mask、OCR 文本、embedding 或视频事件。
2. 证据应包含渲染后的视觉产物，以及一个指标或定性错误说明。
3. 自检时要能指出一个视觉失败模式，例如类别混淆、漏检、mask 边界差、光照变化、领域偏移或标注质量弱。

</details>


## 留下的证据

学完这一页，至少保留这张证据卡：

```text
input_image: detection sample with ground-truth or expected objects
prediction: boxes, labels, confidence scores, IoU, and threshold settings
metric: precision/recall, mAP, false positives, and false negatives
failure_check: small object, overlap, NMS, poor labels, or confidence threshold
Expected_output: annotated image plus detection metrics or error buckets
```

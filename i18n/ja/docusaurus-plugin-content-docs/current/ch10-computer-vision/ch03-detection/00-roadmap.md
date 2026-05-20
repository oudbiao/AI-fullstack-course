---
title: "10.3.1 Object Detection ロードマップ：Class plus Box"
sidebar_position: 0
description: "Object detection の短い実践ロードマップ：boxes、IoU、thresholds、YOLO-style outputs、detection failure analysis を理解する。"
keywords: [object detection guide, YOLO, IoU, mAP]
---

# 10.3.1 Object Detection ロードマップ：Class plus Box

Object detection は classification に location を加えます。どの object があり、image のどこにあるかを答えます。

## まず box workflow を見る

![Object detection 章の学習フローチャート](/img/course/ch10-detection-chapter-flow-ja.webp)

![Object detection output diagram](/img/course/object-detection-output-ja.webp)

![Detection output IoU error map](/img/course/ch10-detection-output-iou-error-map-ja.webp)

重要概念は bounding box、class、confidence、IoU、threshold、false positive、false negative、mAP です。

## IoU check を動かす

IoU は predicted box と ground-truth box の重なりを測ります。

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

出力：

```text
iou: 0.391
```

Detection debug は boxes と metrics の表示から始めます。きれいな screenshot 1 枚だけで detection quality を判断しないでください。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | Detection overview | box、class、confidence、IoU、mAP を説明する |
| 2 | Classic detectors | two-stage と one-stage の考え方を比較する |
| 3 | YOLO | grid prediction、threshold、NMS、speed trade-off を理解する |
| 4 | Detection practice | false positives、missed detections、threshold changes を記録する |

## 合格ライン

boxes、confidence、IoU、少なくとも 1 つの false-positive または false-negative case で detection result を説明できれば、この章は合格です。

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
input_image: detection sample with ground-truth or expected objects
prediction: boxes, labels, confidence scores, IoU, and threshold settings
metric: precision/recall, mAP, false positives, and false negatives
failure_check: small object, overlap, NMS, poor labels, or confidence threshold
Expected_output: annotated image plus detection metrics or error buckets
```

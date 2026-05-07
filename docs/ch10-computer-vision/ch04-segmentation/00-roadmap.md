---
title: "10.4.1 Segmentation Roadmap: Pixel-Level Regions"
sidebar_position: 0
description: "A concise hands-on roadmap for image segmentation: understand masks, semantic segmentation, instance segmentation, IoU, and boundary failures."
keywords: [image segmentation guide, semantic segmentation, instance segmentation, mask]
---

# 10.4.1 Segmentation Roadmap: Pixel-Level Regions

Segmentation is finer than detection. Instead of a box, it outputs a mask that says which pixels belong to a class or instance.

## 10.4.1.1 See the Mask Workflow First

![Image segmentation chapter learning order diagram](/img/course/ch10-segmentation-chapter-flow-en.png)

![Semantic segmentation mask example](/img/course/semantic-segmentation-mask-en.png)

![Semantic segmentation IoU and boundary map](/img/course/ch10-semantic-segmentation-iou-boundary-map-en.png)

The main object in this chapter is the mask. The main failure is often boundary quality, tiny objects, occlusion, or class confusion.

## 10.4.1.2 Run a Mask IoU Check

This script compares two tiny binary masks.

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

Expected output:

```text
mask_iou: 0.5
```

Segmentation reports should show masks, metrics, and boundary errors, not only a colored overlay.

## 10.4.1.3 Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | Semantic segmentation | Predict one class for every pixel |
| 2 | Instance segmentation | Separate different objects of the same class |
| 3 | Segmentation practice | Compare masks, IoU/Dice, boundary errors, and failed samples |

## 10.4.1.4 Pass Check

You pass this chapter when you can create or inspect a mask, compute a simple overlap metric, and explain one boundary or class-confusion failure.

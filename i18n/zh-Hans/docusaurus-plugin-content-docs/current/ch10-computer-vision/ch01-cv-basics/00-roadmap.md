---
title: "10.1.1 视觉基础路线图：像素、通道、处理"
sidebar_position: 0
description: "视觉基础的简短实操路线：理解像素、图像形状、颜色通道、OpenCV 坐标和基础处理。"
keywords: [视觉基础指南, OpenCV 指南, 图像处理指南]
---

# 10.1.1 视觉基础路线图：像素、通道、处理

计算机视觉从输入直觉开始。在分类、检测或分割之前，你需要知道图像在计算机里是什么数字形态。

## 先看图像流水线

![视觉基础章节学习流程](/img/course/ch10-cv-basics-chapter-flow.png)

![像素 RGB 网格图](/img/course/cv-pixel-rgb-grid.png)

![图像数组形状与通道图](/img/course/ch10-image-array-shape-channel-map.png)

第一个心智模型很简单：图像 = 高度 × 宽度 × 通道。后面的很多 bug 都来自 shape、通道顺序、坐标或颜色空间混淆。

## 跑一个极小图像形状检查

这个玩具图像有 2 行、3 列和 RGB 值。

```python
image = [
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    [[255, 255, 255], [0, 0, 0], [128, 128, 128]],
]

height = len(image)
width = len(image[0])
channels = len(image[0][0])
top_left_pixel = image[0][0]

print("shape:", (height, width, channels))
print("top_left_pixel:", top_left_pixel)
```

预期输出：

```text
shape: (2, 3, 3)
top_left_pixel: [255, 0, 0]
```

如果真实图片读取后的形状或通道顺序错了，后面每个模型结果都会更难信任。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 图像表示 | 解释像素、通道、高度、宽度、RGB/BGR |
| 2 | OpenCV 基础 | 加载、查看、裁剪、缩放、保存图片 |
| 3 | 基础处理 | 尝试灰度、阈值、模糊、边缘和简单滤波 |

## 通过标准

如果你能检查图像 shape，按坐标裁剪区域，解释通道顺序，并为 README 保存一张处理结果，就通过了本章。

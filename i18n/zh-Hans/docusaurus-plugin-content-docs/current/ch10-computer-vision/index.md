---
title: "10 计算机视觉（方向选修）"
sidebar_position: 0
description: "通过输出粒度、图像像素、预处理、分类、检测、分割、指标和失败复盘学习计算机视觉。"
keywords: [计算机视觉, OpenCV, 图像分类, 目标检测, YOLO, 图像分割]
---

# 10 计算机视觉（方向选修）

![计算机视觉主视觉](/img/course/ch10-computer-vision.webp)

这一选修章回答一个简单问题：**模型“看见一张图”到底是什么意思？** 先从像素开始，再按输出粒度逐步深入：识别整张图、定位目标、分割像素，最后连接 OCR、视频或多模态系统。

如果你的主线是 LLM 应用和 Agent，可以之后再补；如果你关心 OCR、工业检测、医学影像、视觉搜索或多模态产品，就建议系统学习。

## 按输出粒度看视觉任务

![视觉任务输出粒度阶梯](/img/course/ch10-vision-task-granularity-ladder.webp)

对同一张图问三个问题：

| 问题 | 任务 | 输出 |
|---|---|---|
| 这张图主要是什么？ | 分类 | 一个或多个标签 |
| 每个物体在哪里？ | 检测 | 框、标签、置信度 |
| 哪些像素属于哪个物体或区域？ | 分割 | mask 或像素类别 |
| 能提取什么文字或视觉含义？ | OCR / 视觉理解 | 文本、表格、描述、答案 |

## 学习顺序与任务表

先理解输出类型，再做项目。同一张图可以变成多种任务。

| 步骤 | 阅读内容 | 要动手做什么 | 留下什么证据 |
|---|---|---|---|
| 10.1 | 图像基础与 OpenCV | 检查像素、通道、缩放、灰度、边缘 | 输入图、处理后输出 |
| 10.2 | 图像分类 | 运行或训练一个小分类器 | 标签、accuracy/F1、错误图片 |
| 10.3 | 目标检测 | 理解框、置信度、IoU、mAP、YOLO | 预测框和阈值记录 |
| 10.4 | 图像分割 | 理解 mask 和像素级标签 | mask 可视化和 IoU/Dice 记录 |
| 10.5 | 进阶专题 | 只在需要时选择 OCR、视频、人脸、3D 或医学方向 | 方向说明和场景边界 |
| 10.6 | 阶段项目 | 运行 [10.6.4 实操：构建一个可复现的视觉迷你流水线](./ch06-projects/03-hands-on-vision-workshop.md) | 生成图像、mask、框、指标、失败报告 |

## 第一个可运行循环：零依赖检查像素

这个零依赖小实验会创建一张很小的彩色图，把它转成灰度图，并保存成大多数图片查看器可打开的文件。它要教的核心是：图像就是有结构的数字数据。

新建 `ch10_pixel_lab.py`，用 Python 3.10 或更新版本运行。

```python
from pathlib import Path

width, height = 8, 8

pixels = [
    [(x * 32, y * 32, 128) for x in range(width)]
    for y in range(height)
]

gray = [
    [round(0.299 * r + 0.587 * g + 0.114 * b) for r, g, b in row]
    for row in pixels
]

ppm_body = "\n".join(" ".join(f"{r} {g} {b}" for r, g, b in row) for row in pixels)
pgm_body = "\n".join(" ".join(str(value) for value in row) for row in gray)

Path("synthetic_rgb.ppm").write_text(f"P3\n{width} {height}\n255\n{ppm_body}\n")
Path("synthetic_gray.pgm").write_text(f"P2\n{width} {height}\n255\n{pgm_body}\n")

print("size:", (width, height))
print("channels:", 3)
print("top_left_rgb:", pixels[0][0])
print("center_gray:", gray[height // 2][width // 2])
print("saved:", "synthetic_rgb.ppm", "synthetic_gray.pgm")
```

预期输出：

```text
size: (8, 8)
channels: 3
top_left_rgb: (0, 0, 128)
center_gray: 128
saved: synthetic_rgb.ppm synthetic_gray.pgm
```

操作提示：修改 `width`、`height` 或 RGB 公式。保存的图像改变了，就说明你已经在做图像预处理。后续章节会把这个小实验替换成 OpenCV、Pillow、PyTorch，以及检测或分割模型。

### 如何读这个输出

- `size` 和 `channels` 说明模型看到图像之前，数据本身是什么形状。
- `top_left_rgb` 是真实像素值，不是对图片的文字描述。
- `center_gray` 证明预处理已经把 RGB 数据变成了一个灰度数值。
- 保存下来的文件就是证据。如果不能展示处理前后文件，后面很难调试预处理是否正确。

## 深度阶梯

| 层级 | 你能证明什么 |
|---|---|
| 最低通过 | 能运行像素实验，并解释图像尺寸、通道、RGB 值、灰度转换和保存输出。 |
| 项目可用 | 能按输出选择任务，保留原图、处理图和预测图，报告正确指标，并保存失败样本。 |
| 深度检查 | 换架构前，能把错误追到数据、标注、预处理、模型、阈值、指标或部署限制。 |

## 调试视觉结果

![视觉流水线与失败复盘闭环](/img/course/ch10-vision-pipeline-loop.webp)

视觉模型出错时，先检查输入和标签，再怀疑模型架构。

| 现象 | 先打印或可视化什么 | 可能修复 |
|---|---|---|
| 分类不稳定 | 误判图片和类别数量 | 清洗数据、平衡类别、调整增强 |
| 小目标漏检 | 图片分辨率、框、置信度阈值 | 改进标注、提高分辨率、调阈值 |
| 分割边界粗糙 | mask 叠加到原图的效果 | 改进标注，使用合适的 IoU/Dice 指标 |
| 演示图好，真实图差 | 光照、角度、背景、相机来源 | 补真实样本和场景说明 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
任务输出：分类标签、检测框、分割掩膜、OCR 文本或视频事件
工件：原始图像、处理后图像、预测叠加图、指标文件和失败样本
指标：准确率/F1、mAP、IoU、Dice、延迟或场景特定审查分数
失败检查：数据质量、标签错误、预处理不匹配、阈值或部署约束
期望产出：一个可复现的运行文件夹，包含可视化输出和简短失败报告
```

## 常见错误

- 还没检查数据质量就追模型名字。
- 只报 accuracy，却不保存错误图片。
- 混淆分类、检测和分割的输出。
- 使用会改变标签含义的数据增强。
- 忽略图像尺寸、延迟和设备内存等部署限制。

## 通关检查

离开这个选修章前，你应该能做到：

- 按输出解释分类、检测、分割、OCR 和视觉理解；
- 运行像素小实验，并解释图像尺寸、通道、RGB 值和灰度值；
- 保留输入图、处理图、预测图、指标和失败样本；
- 选择 accuracy/F1、mAP、IoU 或 Dice 等合适指标；
- 跑通可复现视觉迷你流水线，并写一段失败分析。

可打印清单见 [10.0 学习检查表](./study-guide.md)。如果想直接做项目，从 [10.6.4 实操：构建一个可复现的视觉迷你流水线](./ch06-projects/03-hands-on-vision-workshop.md) 开始。

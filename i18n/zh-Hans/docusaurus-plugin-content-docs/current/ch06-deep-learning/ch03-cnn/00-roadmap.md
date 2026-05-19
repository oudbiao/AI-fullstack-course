---
title: "6.3.1 CNN 路线图：把图像变成特征图"
sidebar_position: 0
description: "紧凑版 CNN 路线图：卷积、通道、特征图、经典架构、迁移学习和图像分类实战。"
keywords: [CNN 指南, 卷积, 图像分类, 迁移学习, 特征图]
---

# 6.3.1 CNN 路线图：把图像变成特征图

CNN 学习局部视觉模式。它不是把图像直接摊平成一行，而是扫描小区域并生成特征图。

## 先看图像流

![CNN 章节关系图](/img/course/ch06-cnn-chapter-flow.webp)

![CNN 感受野增长图](/img/course/ch06-cnn-receptive-field-growth-map.webp)

| 概念 | 第一层意思 |
|---|---|
| channel | 颜色或学到的特征维度 |
| kernel | 小的滑动滤波器 |
| feature map | 滤波器扫过图像后的输出 |
| pooling / stride | 缩小空间尺寸 |
| transfer learning | 复用预训练视觉骨干 |

## 跑一次卷积

创建 `cnn_first_loop.py`，安装 `torch` 后运行。

```python
import torch

image = torch.randn(1, 3, 32, 32)
conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
features = conv(image)

print("input_shape:", tuple(image.shape))
print("feature_shape:", tuple(features.shape))
```

预期输出：

```text
input_shape: (1, 3, 32, 32)
feature_shape: (1, 8, 32, 32)
```

把形状读成 `[batch, channels, height, width]`。卷积把 `3` 个输入通道变成了 `8` 个学习到的特征通道。

## 按这个顺序学

| 顺序 | 阅读 | 练什么 |
|---|---|---|
| 1 | [6.3.2 卷积基础](./01-convolution-basics.md) | kernel、stride、padding、channel |
| 2 | [6.3.3 CNN 结构](./02-cnn-structure.md) | 卷积块、池化、分类头 |
| 3 | [6.3.4 经典架构](./03-classic-architectures.md) | LeNet、AlexNet、VGG、ResNet 直觉 |
| 4 | [6.3.5 迁移学习](./04-transfer-learning.md) | 冻结骨干、微调 |
| 5 | [6.3.6 图像分类实战](./05-image-classification-practice.md) | 数据集、训练、预测样例 |

## 留下的证据

保留一条 CNN shape 笔记：

```text
input: [batch, channels, height, width]
conv_output: out_channels becomes new feature maps
spatial_change: stride/padding/pooling change height and width
classifier_bridge: conv features eventually become class logits
transfer_choice: freeze first, fine-tune only if validation improves
```

## 通过标准

能解释输入图像形状和特征图形状之间发生了什么变化，并知道为什么小数据集常复用预训练 CNN 骨干，就算通过。

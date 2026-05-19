---
title: "6.3.2 卷积基础"
sidebar_position: 1
description: "通过手算和 PyTorch 学会卷积：kernel、局部模式、stride、padding、channel、输出 shape 和感受野。"
keywords: [convolution, 卷积, 卷积核, CNN, stride, padding, receptive field, 图像特征]
---

# 6.3.2 卷积基础

:::tip 本节定位
卷积是 CNN 看图像的方式，它不会过早把空间结构压平成一长串数字。本页先看图，再手算卷积，最后用 `nn.Conv2d` 验证同样的概念。
:::

## 学习目标

- 解释为什么太早 flatten 图像很浪费。
- 手算一个卷积输出值。
- 理解 kernel、stride、padding、channel 和 feature map。
- 用 PyTorch 验证输出 shape。
- 解释为什么堆叠卷积会扩大感受野。

---

## 先看滑动窗口

![CNN 卷积核滑动图](/img/course/cnn-convolution-kernel.webp)

按这个顺序读图：

```text
小窗口 -> 和 kernel 逐元素相乘 -> 求和 -> 一个输出值 -> 滑动并重复
```

卷积核就是一个小型模式检测器。它不是一次看完整张图，而是扫描局部区域，并把得分写进 feature map。

## 为什么不先把图像 flatten？

一张 `32 x 32` 灰度图有 `1024` 个像素。如果接一个输出为 `512` 的全连接层，需要：

```text
1024 * 512 = 524288 个权重
```

一张 `224 x 224 x 3` 彩色图有 `150528` 个输入值。朴素全连接层参数会爆炸，而且会忽略像素的位置关系。

卷积解决两个问题：

| 过早 flatten 的问题 | 卷积的做法 |
|---|---|
| 附近像素的空间关系丢失 | 看局部窗口 |
| 每个位置都需要单独权重 | 同一个 kernel 到处复用 |
| 参数数量增长很快 | 在整张图上共享参数 |

两个核心术语：

- 局部连接：每个输出只看一小块区域；
- 参数共享：同一个 kernel 扫描多个位置。

## 实验 1：手算卷积

```python
import numpy as np

image = np.array(
    [
        [1, 2, 0, 0],
        [5, 3, 0, 4],
        [2, 1, 3, 1],
        [0, 2, 1, 2],
    ],
    dtype=np.float32,
)

kernel = np.array(
    [
        [1, 0],
        [0, -1],
    ],
    dtype=np.float32,
)

out = np.zeros((3, 3), dtype=np.float32)
for i in range(3):
    for j in range(3):
        patch = image[i : i + 2, j : j + 2]
        out[i, j] = np.sum(patch * kernel)

print("manual_conv_lab")
print(out)
```

预期输出：

```text
manual_conv_lab
[[-2.  2. -4.]
 [ 4.  0. -1.]
 [ 0.  0.  1.]]
```

左上角输出值：

```text
patch = [[1, 2],
         [5, 3]]

kernel = [[ 1,  0],
          [ 0, -1]]

score = 1*1 + 2*0 + 5*0 + 3*(-1) = -2
```

这就是卷积的核心计算。

## 实验 2：把 kernel 当边缘检测器

这个横向 kernel 会比较相邻像素从左到右的变化。

```python
import numpy as np

image = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.float32,
)

kernel = np.array([[-1, 1]], dtype=np.float32)

out = np.zeros((5, 4), dtype=np.float32)
for i in range(5):
    for j in range(4):
        patch = image[i : i + 1, j : j + 2]
        out[i, j] = np.sum(patch * kernel)

print("edge_lab")
print(out)
```

预期输出：

```text
edge_lab
[[0. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 1. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 0. 0.]]
```

`1` 出现在图像从 `0` 变成 `1` 的边界位置。这也是为什么早期 CNN 层经常学到类似边缘的滤波器。

## Stride、Padding 和输出尺寸

![卷积 stride padding 和输出尺寸变化图](/img/course/ch06-conv-stride-padding-size-map.webp)

| 术语 | 含义 | 影响 |
|---|---|---|
| `kernel_size` | 窗口大小 | kernel 越大，看见的局部区域越大 |
| `stride` | 每次移动多远 | stride 越大，输出越小 |
| `padding` | 给输入周围加边框 | 保留边缘信息并控制尺寸 |

单个空间维度的输出尺寸：

```text
output = floor((input + 2*padding - kernel_size) / stride) + 1
```

例子：

```text
input=6, kernel_size=3, padding=1, stride=2
output = floor((6 + 2*1 - 3) / 2) + 1 = 3
```

用 PyTorch 验证：

```python
import torch
from torch import nn

x = torch.randn(1, 1, 6, 6)
conv = nn.Conv2d(
    in_channels=1,
    out_channels=2,
    kernel_size=3,
    stride=2,
    padding=1,
)
y = conv(x)

print("size_lab")
print("input:", tuple(x.shape))
print("output:", tuple(y.shape))
```

预期输出：

```text
size_lab
input: (1, 1, 6, 6)
output: (1, 2, 3, 3)
```

shape 按 `[batch, channels, height, width]` 读。

## 多通道卷积

彩色图像有三个输入通道：红、绿、蓝。PyTorch 中一批 RGB 图像通常是：

```text
[batch, 3, height, width]
```

RGB 图像上的 `3 x 3` 卷积核实际 shape 是：

```text
[out_channels, in_channels, kernel_height, kernel_width]
```

运行：

```python
import torch
from torch import nn

x = torch.randn(2, 3, 32, 32)
conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
y = conv(x)

print("channel_lab")
print("input:", tuple(x.shape))
print("output:", tuple(y.shape))
print("weight:", tuple(conv.weight.shape))
print("bias:", tuple(conv.bias.shape))
```

预期输出：

```text
channel_lab
input: (2, 3, 32, 32)
output: (2, 8, 32, 32)
weight: (8, 3, 3, 3)
bias: (8,)
```

解释：

- `2`：batch 里有两张图；
- `3`：RGB 输入通道；
- `8`：八个学到的输出 feature map；
- `(8, 3, 3, 3)`：八个 kernel，每个都看三个输入通道。

## 感受野：CNN 如何随深度看得更大

![CNN 感受野逐层增长图](/img/course/ch06-cnn-receptive-field-growth-map.webp)

一层 `3 x 3` 卷积只能看很小的局部区域。如果堆叠多层，后面的特征会间接依赖原图中更大的区域。

直觉：

| 层深度 | 常学到什么 |
|---|---|
| 浅层 | 边缘、颜色变化、纹理 |
| 中层 | 角点、简单形状、部件 |
| 深层 | 更大的物体部件和语义模式 |

这就是 CNN 适合图像的原因：小局部线索可以逐层组合成更大的视觉概念。

## 基础 `Conv2d` 检查清单

```python
import torch
from torch import nn

x = torch.randn(1, 1, 8, 8)
conv = nn.Conv2d(
    in_channels=1,
    out_channels=4,
    kernel_size=3,
    stride=1,
    padding=1,
)
y = conv(x)

print("conv2d_lab")
print("input:", tuple(x.shape))
print("output:", tuple(y.shape))
print("weight:", tuple(conv.weight.shape))
print("bias:", tuple(conv.bias.shape))
```

预期输出：

```text
conv2d_lab
input: (1, 1, 8, 8)
output: (1, 4, 8, 8)
weight: (4, 1, 3, 3)
bias: (4,)
```

读任何 `Conv2d` 时，先问：

1. 输入 shape `[N, C, H, W]` 是什么？
2. `in_channels` 是否等于输入的 `C`？
3. `out_channels` 会创建多少个 feature map？
4. `kernel_size`、`stride`、`padding` 会怎样改变 `H` 和 `W`？

## 留下的证据

每个卷积实验都保存一条 shape 方程：

```text
input_shape: [N, C_in, H, W]
kernel: [C_out, C_in, kH, kW]
output_shape: [N, C_out, H_out, W_out]
meaning: C_out feature maps scan local regions
```

如果这条记录清楚，卷积就会变成 shape 和局部模式操作，而不是神秘图像层。

## 常见错误

| 错误 | 为什么有问题 | 修复 |
|---|---|---|
| 在 PyTorch 里用 `[H, W, C]` | PyTorch 期待 `[N, C, H, W]` | 从图像库转换时用 `permute` |
| `in_channels` 写错 | `Conv2d` 接不上输入 | 层前打印 `x.shape` |
| 忘记 padding | feature map 意外变小 | 计算输出尺寸或打印 shape |
| 把卷积当魔法 | 很难调试 feature | 记住 patch * kernel -> sum |
| 太早 flatten | 空间结构丢失 | 先用 conv block，再接 classifier head |

## 练习

1. 改变手写 `2 x 2` kernel，观察输出如何变化。
2. 手算实验 1 的 `out[1, 0]`，和输出对照。
3. 把 size lab 里的 `stride` 改成 `1`，输出 shape 变成什么？
4. 把 channel lab 里的 `out_channels` 改成 `16`，哪些 shape 会变？
5. 用 `permute` 把图像样式张量从 `[N, H, W, C]` 转成 `[N, C, H, W]`。

## 小结

- 卷积比过早 flatten 更能保留局部空间结构。
- kernel 是在不同位置共享的小型模式检测器。
- `stride` 和 `padding` 控制 kernel 如何移动，以及输出尺寸如何变化。
- 多通道卷积会综合多个输入通道的信息。
- 堆叠卷积层会扩大感受野，并建立视觉层级。

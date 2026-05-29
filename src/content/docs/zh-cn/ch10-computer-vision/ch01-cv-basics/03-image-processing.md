---
title: "10.1.4 图像处理技术"
description: "从滤波、边缘检测到形态学操作，用可运行的 OpenCV 示例理解经典图像处理的工作方式。"
sidebar:
  order: 3
head:
  - tag: meta
    attrs:
      name: keywords
      content: "滤波, 边缘检测, 形态学, OpenCV, Canny, blur"
---
![图像处理流水线图](/img/course/cv-image-processing-pipeline.webp)

## 学习目标

完成本节后，你将能够：

- 理解图像滤波在做什么
- 使用 OpenCV 进行平滑、边缘检测和二值化
- 理解膨胀、腐蚀等形态学操作的直觉
- 看懂经典图像处理任务的基础代码

---

## 什么时候会想到经典图像处理？

经典图像处理不是“深度学习之前的旧东西”。它更像一套便宜、稳定、可解释的像素工具，常常用在模型之前、模型之后，或者数据不够时的规则方案里。

你在这些场景里会经常想到它：

| 场景 | 为什么需要图像处理 |
|---|---|
| 文档扫描 / OCR | 先去噪、二值化、矫正，让文字更容易被识别 |
| 工业质检 | 用边缘、阈值、轮廓快速发现划痕、缺口、污染 |
| 医学影像 | 先增强对比、去噪、分割区域，再交给医生或模型判断 |
| 摄像头监控 | 先做背景分离、轮廓检测，减少后续模型压力 |
| 深度学习前处理 | 统一尺寸、通道、亮度范围，让训练和推理更稳定 |

简单判断：如果你需要的是“按明确规则改变像素”，优先想到图像处理；如果你需要从大量样本里学习复杂模式，再考虑 CNN、ViT 或检测模型。

## 一、图像处理在处理什么？

经典图像处理可以理解成：

> **用一套规则，重新调整像素。**

和深度学习不同，它不是“从数据中学规则”，而是我们先写好规则。

典型任务包括：

- 去噪
- 模糊
- 边缘提取
- 二值化
- 轮廓增强

:::note[安装依赖]
下面代码可以直接运行：

```bash
pip install opencv-python numpy
```
:::
---

## 二、先造一张测试图

为了让示例不依赖外部图片，我们先自己生成一张简单图像。

```python
import cv2
import numpy as np

img = np.zeros((240, 320), dtype=np.uint8)

# 画一个白色矩形和一个灰色圆
cv2.rectangle(img, (30, 40), (140, 180), 255, -1)
cv2.circle(img, (230, 120), 45, 180, -1)

cv2.imwrite("processing_original.png", img)
print("已保存 processing_original.png")
```

预期输出：

```text
已保存 processing_original.png
```

这里我们直接用灰度图，后面做边缘和阈值会更方便。

---

## 三、滤波：把图像“揉平一点”

滤波的直觉很像：

> 把一个像素周围邻居的值也考虑进来，让图像更平滑。

### 均值滤波

```python
import cv2
import numpy as np

img = cv2.imread("processing_original.png", cv2.IMREAD_GRAYSCALE)
blurred = cv2.blur(img, (7, 7))

cv2.imwrite("processing_blur.png", blurred)
print("已保存 processing_blur.png")
```

预期输出：

```text
已保存 processing_blur.png
```

均值滤波会让边缘变软，但也可能让细节损失。

### 高斯滤波

```python
import cv2

img = cv2.imread("processing_original.png", cv2.IMREAD_GRAYSCALE)
gaussian = cv2.GaussianBlur(img, (7, 7), 0)

cv2.imwrite("processing_gaussian.png", gaussian)
print("已保存 processing_gaussian.png")
```

预期输出：

```text
已保存 processing_gaussian.png
```

高斯滤波比简单均值滤波更常用，因为它更自然一些。

---

## 四、边缘检测：找出“变化最明显的地方”

边缘可以理解成：

> 亮度变化很突然的位置

比如黑底上的白色矩形边界，就是典型边缘。

### Canny 边缘检测

```python
import cv2

img = cv2.imread("processing_original.png", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, threshold1=50, threshold2=150)

cv2.imwrite("processing_edges.png", edges)
print("已保存 processing_edges.png")
```

预期输出：

```text
已保存 processing_edges.png
```

### 两个阈值怎么理解？

可以粗略记成：

- 小于低阈值：基本不是边缘
- 大于高阈值：大概率是边缘
- 中间区域：结合邻域再判断

---

## 五、阈值化：把灰度图变成黑白图

阈值化就是设一条线：

- 大于这个值的变白
- 小于这个值的变黑

```python
import cv2

img = cv2.imread("processing_original.png", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

cv2.imwrite("processing_binary.png", binary)
print("已保存 processing_binary.png")
```

预期输出：

```text
已保存 processing_binary.png
```

这种操作常用于：

- 文档扫描
- 前景 / 背景分离
- 轮廓提取前处理

---

## 六、形态学操作：对形状做加工

形态学操作特别适合处理二值图像。

可以把它理解成“对白色区域做揉一揉、扩一扩、缩一缩”。

### 腐蚀（Erosion）

白色区域会变小。

```python
import cv2
import numpy as np

img = cv2.imread("processing_binary.png", cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)
eroded = cv2.erode(img, kernel, iterations=1)

cv2.imwrite("processing_eroded.png", eroded)
print("已保存 processing_eroded.png")
```

预期输出：

```text
已保存 processing_eroded.png
```

### 膨胀（Dilation）

白色区域会变大。

```python
import cv2
import numpy as np

img = cv2.imread("processing_binary.png", cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(img, kernel, iterations=1)

cv2.imwrite("processing_dilated.png", dilated)
print("已保存 processing_dilated.png")
```

预期输出：

```text
已保存 processing_dilated.png
```

### 开运算和闭运算

- 开运算 = 先腐蚀再膨胀，适合去小噪点
- 闭运算 = 先膨胀再腐蚀，适合补小孔洞

```python
import cv2
import numpy as np

img = cv2.imread("processing_binary.png", cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)

opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imwrite("processing_opened.png", opened)
cv2.imwrite("processing_closed.png", closed)
print("已保存 processing_opened.png 和 processing_closed.png")
```

预期输出：

```text
已保存 processing_opened.png 和 processing_closed.png
```

![经典图像处理操作选择图](/img/course/ch10-image-processing-operation-decision-map.webp)

:::tip[读图提示]
经典图像处理不是一堆 API 清单，而是一组目的明确的像素规则：去噪先滤波，找变化看边缘，分前景用阈值，清理小噪点和孔洞用形态学。
:::
---

## 七、把这些操作串起来

真实任务里，这些操作经常是连起来用的。

比如你想提取一个目标轮廓，可能会这样：

1. 转灰度
2. 滤波去噪
3. 二值化
4. 形态学清理
5. 再做边缘检测或轮廓分析

下面给一个完整小流程：

```python
import cv2
import numpy as np

img = cv2.imread("processing_original.png", cv2.IMREAD_GRAYSCALE)

# 去噪
smoothed = cv2.GaussianBlur(img, (5, 5), 0)

# 二值化
_, binary = cv2.threshold(smoothed, 100, 255, cv2.THRESH_BINARY)

# 闭运算填补小空隙
kernel = np.ones((5, 5), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 边缘提取
edges = cv2.Canny(cleaned, 50, 150)

cv2.imwrite("processing_pipeline_smoothed.png", smoothed)
cv2.imwrite("processing_pipeline_binary.png", binary)
cv2.imwrite("processing_pipeline_cleaned.png", cleaned)
cv2.imwrite("processing_pipeline_edges.png", edges)
print("完整处理流程结果已保存")
```

预期输出：

```text
完整处理流程结果已保存
```

![图像处理流水线运行结果对照图](/img/course/ch10-image-processing-pipeline-output-montage.webp)

### 输出应该怎么看？

运行完本节代码后，不要只看终端有没有报错。请把生成的图片打开，对照下面的现象检查：

| 文件 | 应该观察什么 |
|---|---|
| `processing_original.png` | 黑底上有白色矩形和灰色圆，是所有处理的起点 |
| `processing_blur.png` / `processing_gaussian.png` | 边界变软，噪声和突变被抹平一些 |
| `processing_edges.png` | 主要只留下形状轮廓，内部大面积区域不应该全白 |
| `processing_binary.png` | 灰度被压成黑白两类，前景和背景更清楚 |
| `processing_eroded.png` | 白色区域变小，细小白点更容易消失 |
| `processing_dilated.png` | 白色区域变大，断开的细线更容易连起来 |
| `processing_pipeline_edges.png` | 经过完整流程后，边缘应该比直接乱调参数更干净 |

如果输出看起来“不对”，优先检查三件事：读图路径是否正确、灰度/彩色通道是否符合预期、阈值和 kernel 大小是否太激进。

---

## 八、这些经典方法为什么今天还要学？

因为它们依然非常有用：

- 作为深度学习前处理
- 在小项目里快速出效果
- 在工业场景里做规则补充
- 帮你建立“图像是怎么被处理的”直觉

很多新手上来只想学 CNN，但如果连灰度、边缘、阈值都没概念，后面对视觉模型的理解会发虚。

---

## 九、初学者常见误区

### 以为滤波就是“让图更好看”

不只是。
滤波常常是为了让后面的算法更稳定。

### 以为阈值可以固定不变

真实图像里光照变化大，阈值常常要结合场景调。

### 只学 API，不理解目的

你要始终问自己：

- 这一步是在去噪？
- 还是在增强边界？
- 还是在清理形状？

---

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
输入图像: 本次运行中使用的源图像或合成图像
数组形状：宽、高、通道、dtype，以及坐标约定
处理后输出：灰度、裁剪、边缘、阈值或已保存的中间图像
失败检查：通道顺序、缩放失真、坐标错误或过度处理
期望产出：前后对比图片，以及打印出的形状或像素值
```

## 小结

这节课你要抓住的核心是：

> **经典图像处理，本质上是在用规则重新排列和筛选像素。**

它不等于深度学习，但它是理解视觉任务的重要台阶。

---

## 练习

1. 修改 `threshold()` 的阈值为 `60`、`120`、`180`，观察二值图变化。
2. 修改腐蚀和膨胀的核大小，从 `(3, 3)` 改到 `(7, 7)`，观察形状变化。
3. 在原始图像里再加一个小白点，试试开运算能不能把它去掉。

<details>
<summary>参考实现与讲解</summary>

1. 阈值越低，二值图中变白的像素通常越多；阈值越高，变白的像素越少。重点不是记住某张图，而是看懂变化方向。
2. `(7, 7)` 核比 `(3, 3)` 核腐蚀更强、膨胀也更强。细线结构可能消失，或者变得明显更粗。
3. 开运算可以去掉孤立的小白点，前提是核尺寸大于噪点。如果白点连在大结构上，或本身太大，就可能保留下来。

</details>

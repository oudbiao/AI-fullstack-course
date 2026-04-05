---
title: "1.3 OpenCV 基础操作"
sidebar_position: 2
description: "学会用 OpenCV 读写图像、缩放、裁剪、颜色转换和绘图，迈出 CV 工程实践的第一步。"
keywords: [OpenCV, cv2, 图像读取, 图像缩放, 绘图, 颜色转换]
---

# OpenCV 基础操作

## 学习目标

完成本节后，你将能够：

- 使用 OpenCV 创建、读取、保存图像
- 完成缩放、裁剪、翻转等基础变换
- 理解 OpenCV 中常见的颜色顺序问题
- 用 OpenCV 在图像上绘制矩形、圆和文字

---

## 一、为什么几乎每个 CV 入门都从 OpenCV 开始？

因为 OpenCV 就像计算机视觉里的“瑞士军刀”：

- 能读图、写图
- 能做缩放、旋转、裁剪
- 能做滤波、边缘检测
- 能做人脸检测、视频处理

而且它很适合初学者建立工程感。

:::info 安装依赖
下面代码可以直接运行：

```bash
pip install opencv-python numpy
```
:::

---

## 二、先创建一张图，而不是依赖外部文件

为了让代码直接运行，我们先自己生成一张空白图。

```python
import cv2
import numpy as np

# 创建一张黑色画布：高 240，宽 320，3 个颜色通道
img = np.zeros((240, 320, 3), dtype=np.uint8)

print("shape:", img.shape)
print("dtype:", img.dtype)

cv2.imwrite("opencv_blank.png", img)
print("已保存 opencv_blank.png")
```

这里的 `shape = (240, 320, 3)`，表示：

- 高度 240
- 宽度 320
- 3 个颜色通道

---

## 三、OpenCV 里的颜色顺序是 BGR，不是 RGB

这是非常经典的坑。

OpenCV 默认使用：

> **BGR**

不是我们更熟悉的 RGB。

```python
import cv2
import numpy as np

img = np.zeros((100, 100, 3), dtype=np.uint8)

# 这个颜色是 BGR，不是 RGB
img[:, :] = (255, 0, 0)

cv2.imwrite("opencv_blue.png", img)
print("保存了一张蓝色图片 opencv_blue.png")
```

如果你以为 `(255, 0, 0)` 是红色，就会得到“颜色不对”的图。

### 转成 RGB

```python
import cv2
import numpy as np

img_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
img_bgr[:, :] = (255, 0, 0)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print("BGR 像素:", img_bgr[0, 0].tolist())
print("RGB 像素:", img_rgb[0, 0].tolist())
```

---

## 四、常见基础操作：缩放、裁剪、翻转

```python
import cv2
import numpy as np

img = np.zeros((200, 300, 3), dtype=np.uint8)
img[:, :] = (40, 180, 240)

# 缩放
small = cv2.resize(img, (150, 100))

# 裁剪：先行后列，即 [y1:y2, x1:x2]
crop = img[50:150, 80:220]

# 翻转
flip_horizontal = cv2.flip(img, 1)

print("原图:", img.shape)
print("缩放后:", small.shape)
print("裁剪后:", crop.shape)
print("水平翻转后:", flip_horizontal.shape)

cv2.imwrite("opencv_small.png", small)
cv2.imwrite("opencv_crop.png", crop)
cv2.imwrite("opencv_flip.png", flip_horizontal)
```

### 裁剪为什么写成 `[y1:y2, x1:x2]`？

因为图像本质上是二维数组，数组访问顺序是：

1. 先行（高度方向，`y`）
2. 再列（宽度方向，`x`）

---

## 五、在图像上画图

很多视觉任务都需要在图片上标注结果，比如：

- 画检测框
- 标类别名
- 标中心点

```python
import cv2
import numpy as np

canvas = np.ones((300, 400, 3), dtype=np.uint8) * 255

# 画矩形
cv2.rectangle(canvas, (50, 50), (180, 180), (0, 255, 0), 2)

# 画圆
cv2.circle(canvas, (280, 120), 40, (255, 0, 0), -1)

# 画直线
cv2.line(canvas, (30, 250), (350, 250), (0, 0, 255), 3)

# 写文字
cv2.putText(
    canvas,
    "CV Demo",
    (120, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 0),
    2
)

cv2.imwrite("opencv_draw_demo.png", canvas)
print("已保存 opencv_draw_demo.png")
```

---

## 六、灰度图转换

许多经典视觉处理会先把彩色图转成灰度图，因为：

- 计算更快
- 去掉颜色干扰
- 只保留亮度信息

```python
import cv2
import numpy as np

img = np.zeros((100, 100, 3), dtype=np.uint8)
img[:, :50] = (0, 0, 255)      # 红
img[:, 50:] = (0, 255, 0)      # 绿

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("原图 shape:", img.shape)
print("灰度图 shape:", gray.shape)
print("灰度图前 5 个像素:", gray[0, :5].tolist())

cv2.imwrite("opencv_gray.png", gray)
```

---

## 七、一个小项目：做一张“信息卡片图”

这个例子会把前面的知识串起来：创建图像、绘图、写字、保存。

```python
import cv2
import numpy as np

card = np.ones((220, 420, 3), dtype=np.uint8) * 245

cv2.rectangle(card, (20, 20), (400, 200), (60, 120, 200), 2)
cv2.circle(card, (80, 85), 35, (60, 120, 200), -1)

cv2.putText(card, "AI Fullstack", (140, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
cv2.putText(card, "Stage 6: CV Basics", (140, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
cv2.putText(card, "OpenCV starter demo", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

cv2.imwrite("opencv_info_card.png", card)
print("已保存 opencv_info_card.png")
```

---

## 八、初学者常见误区

### 1. 用 `cv2.imshow()` 结果窗口打不开

在很多远程环境、Notebook、服务器环境中，`imshow()` 不方便用。  
教学和脚本场景里，推荐先用 `cv2.imwrite()` 保存结果。

### 2. 把 BGR 当成 RGB

这是 OpenCV 初学者最常见 bug 之一。

### 3. 裁剪时把 `x`、`y` 顺序写反

图像数组索引是 `[y, x]`，不是 `[x, y]`。

---

## 小结

这节课的重点不是背完所有 OpenCV API，而是建立“我已经能操作图像了”的感觉：

- 我能创建图像
- 我能变换图像
- 我能标注图像
- 我能把结果保存出来

有了这些基础，下一节做滤波、边缘检测和形态学操作就顺很多。

---

## 练习

1. 把画布颜色改成其他颜色，并重新生成一张卡片图。
2. 在同一张图上多画几个矩形和圆，练习坐标系。
3. 试着把图像缩放为不同分辨率，再保存结果。

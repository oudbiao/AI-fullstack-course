---
title: "2.1 数据增强策略"
sidebar_position: 4
description: "从翻转、裁剪、颜色扰动到 mixup，理解数据增强为什么是视觉任务里最便宜也最有效的泛化手段之一。"
keywords: [data augmentation, flip, crop, color jitter, mixup, vision]
---

# 数据增强策略

:::tip 本节定位
图像分类里最常见也最容易被低估的技巧之一，就是数据增强。

它解决的不是“模型不会学”，而是：

> **模型太容易把训练集里的偶然细节当真。**

通过合理增强，我们能让模型看到更多“合理变化后的同一张图”，从而学得更稳。
:::

## 学习目标

- 理解数据增强为什么能提升泛化能力
- 区分几种常见增强方式适合处理什么问题
- 理解“标签保持不变”这个增强前提
- 通过可运行示例建立增强链路的直觉

---

## 一、为什么图像任务特别需要数据增强？

### 1.1 真实世界本来就在变化

同一只猫在不同图片里会有：

- 角度变化
- 光照变化
- 背景变化
- 局部遮挡

如果训练集覆盖不够，模型就很容易把偶然背景当成真正特征。

### 1.2 增强不是“造更多数据”，而是“模拟合理变化”

一张图片经过合理变换后，  
语义通常还没变。

例如：

- 左右翻转后的猫还是猫
- 轻微裁剪后的狗还是狗

这就是为什么增强能帮助模型学得更稳。

### 1.3 一个类比

数据增强像考前练变式题。  
不是换知识点，而是让你别死记某一道题的表面样子。

---

## 二、最常见的几类增强

### 2.1 几何增强

例如：

- 翻转
- 平移
- 裁剪
- 旋转

它主要帮助模型应对：

- 视角和位置变化

### 2.2 颜色增强

例如：

- 亮度
- 对比度
- 饱和度

它主要帮助模型应对：

- 光照和拍摄条件变化

### 2.3 组合与混合增强

例如：

- Cutout
- Mixup
- CutMix

它们更激进，但也往往更有效。

---

## 三、先跑一个最小增强流水线示例

下面这个例子不依赖图像库，  
而是用二维列表模拟一张灰度图，帮助你抓住增强的核心思想。

```python
image = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]


def horizontal_flip(img):
    return [list(reversed(row)) for row in img]


def center_crop(img, size=2):
    return [row[:size] for row in img[:size]]


def brightness_shift(img, delta=1):
    return [[pixel + delta for pixel in row] for row in img]


print("original:")
for row in image:
    print(row)

print("\nflip:")
for row in horizontal_flip(image):
    print(row)

print("\ncrop:")
for row in center_crop(image):
    print(row)

print("\nbrightness:")
for row in brightness_shift(image):
    print(row)
```

### 3.1 这个例子最该抓住什么？

增强的本质不是图像库 API，  
而是：

- 对输入做合理变换
- 同时尽量不改变标签语义

### 3.2 为什么“合理”很重要？

如果你把“6”和“9”这类数字图像乱旋转，  
标签可能就真的变了。

所以增强不是无脑越强越好，  
而要考虑任务语义。

---

## 四、Mixup 为什么值得单独记住？

### 4.1 它不是简单改图，而是连标签也一起混

Mixup 的核心思想是：

- 两张图按比例混合
- 标签也按比例混合

### 4.2 一个纯数字直觉示例

```python
img_a = [1.0, 2.0, 3.0]
img_b = [7.0, 8.0, 9.0]
label_a = [1.0, 0.0]
label_b = [0.0, 1.0]
alpha = 0.7

mixed_img = [alpha * a + (1 - alpha) * b for a, b in zip(img_a, img_b)]
mixed_label = [alpha * a + (1 - alpha) * b for a, b in zip(label_a, label_b)]

print("mixed_img:", mixed_img)
print("mixed_label:", mixed_label)
```

### 4.3 为什么这种方法会有效？

它会让模型更少学到极端边界，  
更倾向于形成平滑决策面。

---

## 五、增强最容易踩的坑

### 5.1 误区一：增强越重越好

增强过头可能会把有效特征破坏掉。

### 5.2 误区二：所有任务共用同一套增强

分类、检测、分割对增强的敏感点并不完全一样。

### 5.3 误区三：只加增强，不做验证

增强是手段，不是目标。  
最终还是要看验证集是否真的受益。

---

## 小结

这节最重要的是建立一个判断：

> **数据增强的核心，是通过模拟合理变化，让模型学会抓住更稳定的视觉特征，而不是死记训练集里的偶然细节。**

只要这层直觉在，后面你看更复杂增强策略就不会迷路。

---

## 练习

1. 给示例再写一个 `vertical_flip` 函数。
2. 想一想：为什么某些任务里旋转增强可能是有害的？
3. 用自己的话解释：Mixup 和普通增强最大的不同是什么？
4. 如果验证集效果下降，你会先怀疑增强太弱还是太强？

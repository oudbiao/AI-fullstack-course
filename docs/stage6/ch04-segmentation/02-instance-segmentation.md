---
title: "4.2 实例分割"
sidebar_position: 12
description: "从语义分割继续往前，理解实例分割为什么不仅要分出类别，还要区分同类物体的不同个体。"
keywords: [instance segmentation, mask, object instance, vision]
---

# 实例分割

:::tip 本节定位
语义分割已经能回答：

- 哪些像素属于“人”

但如果图里有三个人，它还不够。  
实例分割更进一步：

> **不仅知道像素属于哪个类别，还要知道它属于哪一个具体实例。**
:::

## 学习目标

- 理解实例分割和语义分割的差别
- 理解“类别”与“实例”为什么是两个层次
- 通过可运行示例建立实例 mask 直觉
- 理解实例分割为什么更接近真实视觉场景

---

## 一、实例分割比语义分割多了什么？

语义分割：

- 只区分类别

实例分割：

- 类别 + 个体区分

也就是说，图里两个“person”不该混成一个整体。

---

## 二、先看一个最小实例 mask 示例

```python
instance_map = [
    [0, 1, 1, 0],
    [0, 1, 1, 2],
    [0, 0, 0, 2],
]


def pixels_of_instance(instance_map, target_id):
    pixels = []
    for r, row in enumerate(instance_map):
        for c, value in enumerate(row):
            if value == target_id:
                pixels.append((r, c))
    return pixels


print("instance 1:", pixels_of_instance(instance_map, 1))
print("instance 2:", pixels_of_instance(instance_map, 2))
```

### 2.1 这个例子最关键的地方是什么？

它说明实例分割不只是输出类别编号，  
还会区分：

- 第 1 个实例
- 第 2 个实例

这在计数、跟踪和交互场景里非常重要。

---

## 三、最容易踩的坑

### 3.1 相邻同类实例容易粘在一起

这是实例分割特别常见的错误。

### 3.2 小实例更难

个体越小、越拥挤，越难分清。

### 3.3 评估比语义分割更复杂

因为现在不仅要看 mask 质量，  
还要看实例是否正确拆开。

---

## 小结

这节最重要的是建立一个判断：

> **实例分割比语义分割多解决了一层“同类目标之间怎么区分”的问题，因此更接近真实多目标视觉场景。**

---

## 练习

1. 自己构造一个更大的 `instance_map`，再标出 3 个实例。
2. 为什么实例分割比语义分割更难？
3. 如果两个相邻目标总被粘成一个实例，你会首先怀疑什么？
4. 想一想：实例分割在自动驾驶或安防里为什么特别有价值？

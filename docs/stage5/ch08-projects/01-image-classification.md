---
title: "8.1 项目：图像分类系统"
sidebar_position: 1
description: "围绕一个最小图像分类项目，走完数据、训练、评估和上线展示这条完整闭环。"
keywords: [image classification project, CNN, dataset, evaluation, computer vision]
---

# 项目：图像分类系统

:::tip 本节定位
项目课的重点不是再讲新概念，  
而是把前面学过的东西组合起来，真正做成一个能展示、能评估、能解释取舍的系统。

这一节我们围绕一个最经典也最适合练手的视觉项目：

> **做一个最小图像分类系统。**
:::

## 学习目标

- 学会定义图像分类项目边界
- 学会组织数据和评估方式
- 学会先做 baseline，再谈更复杂模型
- 通过项目结构建立“从研究到交付”的直觉

---

## 一、为什么图像分类很适合作为第一个视觉项目？

因为它的任务形式很清楚：

- 输入：一张图片
- 输出：一个类别

这使得你能把注意力放在：

- 数据质量
- 模型结构
- 指标和错误分析

而不是一开始就被任务复杂度淹没。

---

## 二、一个最小图像分类项目通常要包含什么？

### 1. 数据集

- 类别定义
- 训练/验证划分

### 2. 模型

- baseline CNN
- 或迁移学习模型

### 3. 指标

- accuracy
- confusion matrix

### 4. 错误分析

- 哪些图最容易分错
- 是因为模糊、遮挡还是类别太像

---

## 三、先跑一个最小项目骨架示例

```python
from dataclasses import dataclass, field


@dataclass
class CVProjectPlan:
    name: str
    classes: list
    modules: list
    metrics: list
    risks: list = field(default_factory=list)


plan = CVProjectPlan(
    name="image_classification_system",
    classes=["cat", "dog", "bird"],
    modules=["dataset", "augmentation", "model", "evaluation", "demo"],
    metrics=["accuracy", "confusion_matrix"],
    risks=["类别不平衡", "背景泄漏", "验证集过小"],
)

print(plan)
```

### 3.1 这个示例为什么仍然有价值？

因为项目最先要清楚的是：

- 你在做什么
- 用什么模块做
- 用什么指标证明有效

这会比一上来堆很多代码更稳。

---

## 四、项目最容易踩的坑

### 4.1 只看总准确率

不看错误分布，很难知道系统到底哪里弱。

### 4.2 类别划分太随意

如果类别本身边界模糊，项目会很难做好。

### 4.3 一上来追求大模型

更稳的方式通常是：

- 先小模型 baseline
- 再迁移学习

---

## 五、小结

这节最重要的是建立一个项目意识：

> **图像分类项目的核心不只是把模型训出来，而是把任务边界、数据组织、评估指标和错误分析讲清楚。**

---

## 练习

1. 自己设计一个 3~5 类的小型图像分类题目。
2. 想一想：如果总准确率不错，但某两类总混淆，你会先看什么？
3. 为什么 confusion matrix 对图像分类项目特别有帮助？
4. 如果你要把这个项目做成作品集，你会展示哪三部分？

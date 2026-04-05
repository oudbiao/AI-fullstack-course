---
title: "8.2 项目：图像分类系统"
sidebar_position: 1
description: "围绕一个真正可展示的图像分类项目，从选题、数据、baseline、训练、评估到演示方式，走完整条交付闭环。"
keywords: [image classification project, CNN, confusion matrix, error analysis, computer vision]
---

# 项目：图像分类系统

:::tip 本节定位
图像分类很适合作为第一个视觉项目，不是因为它最简单，而是因为它最容易把完整工程链路讲清楚：

- 类别怎么定
- 数据怎么组织
- baseline 怎么做
- 指标怎么看
- 错误怎么分析

这一节的目标不是做一个“能跑的模型”，而是做一个**能讲清楚的项目**。
:::

## 学习目标

- 学会定义一个适合作品集展示的图像分类题目
- 学会把数据、baseline、评估和错误分析串成闭环
- 学会用最小可运行示例表达项目结构
- 理解图像分类项目里最值得展示的是什么

---

## 一、先别急着选模型，先把项目题目选对

### 1.1 一个适合练手的题目通常有三个特征

1. 类别边界清楚  
   例如 `猫 / 狗 / 鸟`、`苹果叶片病害分类`、`垃圾分类`
2. 数据能拿到  
   不要一开始就选你根本没法收集样本的题目
3. 错误能解释  
   分错后你能说出可能原因，而不是只剩“模型不行”

### 1.2 一个很稳的项目题目

例如：

> **做一个“宠物照片分类器”，把图片分成 `cat / dog / rabbit` 三类。**

它的优点是：

- 类别直观
- 数据相对容易收集
- 很适合做 confusion matrix 和错误样例分析

### 1.3 不建议一开始就做的题目

例如：

- 上百类细粒度分类
- 类别边界极其模糊
- 数据严重不平衡但你还没准备好处理

---

## 二、项目最小闭环长什么样？

一个最小但完整的图像分类项目，通常至少应包含：

1. 题目与标签定义
2. 数据集组织与划分
3. baseline
4. 训练与验证
5. 评估与错误分析
6. 演示方式

如果这 6 件事都说清了，即使模型不复杂，项目也会很有说服力。

---

## 三、先看一个最小项目规划对象

```python
from dataclasses import dataclass, field


@dataclass
class CVProjectPlan:
    name: str
    classes: list
    dataset_split: dict
    baseline: str
    metrics: list
    risks: list = field(default_factory=list)


plan = CVProjectPlan(
    name="pet_image_classifier",
    classes=["cat", "dog", "rabbit"],
    dataset_split={"train": 900, "val": 180, "test": 180},
    baseline="small_cnn",
    metrics=["accuracy", "confusion_matrix", "error_cases"],
    risks=["类别不平衡", "背景泄漏", "标签噪声"],
)

print(plan)
```

### 3.1 这个对象为什么重要？

因为项目一开始最容易缺的，不是代码，而是边界。  
这个最小对象逼你先说明：

- 在做什么
- 有哪些类别
- 用什么 baseline
- 用什么指标判断成败

---

## 四、先用一个“伪特征”基线理解项目评估

为了不引入额外依赖，我们用一个很小的 toy baseline 来模拟图像分类项目的验证流程。

这里假设每张图片已经有三个非常粗糙的统计特征：

- `fur`
- `ear_shape`
- `size`

当然真实项目不会这么做，但它非常适合帮助你看懂：

- 训练集
- 类别原型
- 预测
- confusion matrix

这条链。

```python
train_data = [
    ("cat", [0.9, 0.8, 0.4]),
    ("cat", [0.8, 0.7, 0.5]),
    ("dog", [0.7, 0.5, 0.8]),
    ("dog", [0.6, 0.4, 0.9]),
    ("rabbit", [0.5, 0.9, 0.3]),
    ("rabbit", [0.4, 0.8, 0.2]),
]

test_data = [
    ("cat", [0.85, 0.75, 0.45]),
    ("dog", [0.65, 0.45, 0.85]),
    ("rabbit", [0.45, 0.85, 0.25]),
    ("dog", [0.82, 0.72, 0.42]),  # 故意放一个更像 cat 的错误样本
]


def class_prototypes(data):
    grouped = {}
    for label, features in data:
        grouped.setdefault(label, []).append(features)

    prototypes = {}
    for label, rows in grouped.items():
        dim = len(rows[0])
        prototypes[label] = [
            sum(row[i] for row in rows) / len(rows)
            for i in range(dim)
        ]
    return prototypes


def l1_distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def predict(features, prototypes):
    distances = {label: l1_distance(features, proto) for label, proto in prototypes.items()}
    return min(distances, key=distances.get), distances


prototypes = class_prototypes(train_data)
print("prototypes:", prototypes)

results = []
for gold, features in test_data:
    pred, distances = predict(features, prototypes)
    results.append({"gold": gold, "pred": pred, "distances": distances})
    print(results[-1])
```

### 4.1 为什么这个示例仍然有项目价值？

因为项目最重要的不是库名，而是评估思路。  
这个 toy baseline 已经让你看到：

- train -> prototype
- test -> predict
- gold vs pred

这正是后面真正 CNN 项目也必须走的一条线。

---

## 五、一个最小 confusion matrix 和错误分析

```python
labels = ["cat", "dog", "rabbit"]


def confusion_matrix(rows, labels):
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for row in rows:
        matrix[row["gold"]][row["pred"]] += 1
    return matrix


cm = confusion_matrix(results, labels)
print("confusion matrix:")
for gold in labels:
    print(gold, cm[gold])

error_cases = [row for row in results if row["gold"] != row["pred"]]
print("\nerror cases:", error_cases)
```

### 5.1 为什么 confusion matrix 对图像分类特别重要？

因为总准确率只会告诉你：

- 对了多少

但 confusion matrix 会告诉你：

- 哪两类最容易混

这正是你下一步改数据和改模型最需要的信息。

### 5.2 错误样例为什么比总分更值钱？

因为你能真正去看：

- 是不是背景误导了模型
- 是不是某类照片角度不一致
- 是不是标签标错了

这才是图像项目里最有洞察力的部分。

---

## 六、真实项目里最该补的三层

### 6.1 数据层

你至少应该说明：

- 每类大概多少图
- train / val / test 怎么分
- 有没有类别不平衡

### 6.2 模型层

很推荐先做两层 baseline：

1. 小 CNN
2. 迁移学习模型

这样你才能说清：

- 更复杂模型到底换来了什么

### 6.3 展示层

图像分类项目做作品集时，最值得展示的通常是：

- 标签定义
- confusion matrix
- 典型正确样本
- 典型错误样本

而不只是贴一张“训练完成”的截图。

---

## 七、这个项目最容易踩的坑

### 7.1 只看总准确率

你会很容易错过真正的问题分布。

### 7.2 类别定义太随意

如果类别边界本身模糊，模型和评估都会一起发虚。

### 7.3 数据泄漏

如果相似图片同时出现在训练和测试，  
结果会被高估。

---

## 小结

这节最重要的是建立一个项目意识：

> **图像分类项目真正有说服力的地方，不是模型名，而是你能否把类别边界、数据组织、baseline、confusion matrix 和错误分析讲成一个完整闭环。**

只要这个闭环立住了，即使是一个小型项目，也会很像作品级课程。

---

## 练习

1. 把 toy 数据里的 `dog` 样本再加两条，看看 confusion matrix 会怎么变化。
2. 如果 `cat` 和 `rabbit` 总混淆，你会优先查数据、标签还是模型？为什么？
3. 想一想：为什么图像分类项目特别适合用 confusion matrix 做展示？
4. 如果你要把这个项目做成作品集页面，你会优先放哪 4 块内容？

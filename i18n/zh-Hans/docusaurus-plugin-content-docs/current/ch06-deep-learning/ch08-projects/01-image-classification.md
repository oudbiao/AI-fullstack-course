---
title: "6.8.2 项目：图像分类系统"
sidebar_position: 1
description: "完成一个图像分类项目闭环：标签定义、数据划分、baseline、confusion matrix、错误样本和作品集包装。"
keywords: [image classification project, CNN, confusion matrix, error analysis, computer vision]
---

# 6.8.2 项目：图像分类系统

:::tip 本节定位
图像分类项目能进入作品集，不是因为模型名字高级，而是因为别人能看懂标签、跑通 baseline、检查指标，并看到失败案例。
:::

## 学习目标

- 定义边界清晰的分类任务。
- 组织 train/validation/test 证据。
- 跑一个能产生预测的极小 baseline。
- 构建 confusion matrix 并提取错误样本。
- 知道真实 CNN 或 transfer-learning 项目应该展示什么。

---

## 先看项目闭环

![图像分类项目闭环图](/img/course/ch06-project-image-classification-loop.webp)

```text
labels -> data split -> baseline -> metrics -> error cases -> next data/model action
```

选题要满足：

- 类别边界清楚；
- 数据真的能收集到；
- 错误可以解释。

不要一开始就做几百个细分类别，也不要选择人类都难以一致标注的标签。

## 项目计划模板

```python
from dataclasses import dataclass, field


@dataclass
class CVProjectPlan:
    name: str
    classes: list[str]
    dataset_split: dict[str, int]
    baseline: str
    metrics: list[str]
    risks: list[str] = field(default_factory=list)


plan = CVProjectPlan(
    name="pet_image_classifier",
    classes=["cat", "dog", "rabbit"],
    dataset_split={"train": 900, "val": 180, "test": 180},
    baseline="small_cnn_then_transfer_learning",
    metrics=["accuracy", "confusion_matrix", "error_cases"],
    risks=["class imbalance", "background leakage", "label noise"],
)

print(plan)
```

这个对象是项目边界。如果你填不出来，说明还没到选模型的时候。

## 实验：Prototype Baseline 与 Confusion Matrix

这个 toy 示例用三个伪特征代替真实图片。它训练的是之后 CNN 项目也要用的评估闭环。

创建 `image_project_baseline.py`：

```python
from collections import defaultdict

train_data = [
    ("cat", [0.9, 0.8, 0.4]),
    ("cat", [0.8, 0.7, 0.5]),
    ("dog", [0.7, 0.5, 0.8]),
    ("dog", [0.6, 0.4, 0.9]),
    ("rabbit", [0.5, 0.9, 0.3]),
    ("rabbit", [0.4, 0.8, 0.2]),
]

val_data = [
    ("cat", [0.85, 0.75, 0.45]),
    ("dog", [0.65, 0.45, 0.85]),
    ("rabbit", [0.45, 0.85, 0.25]),
    ("dog", [0.82, 0.72, 0.42]),
]

labels = ["cat", "dog", "rabbit"]


def prototypes(data):
    groups = defaultdict(list)
    for label, features in data:
        groups[label].append(features)

    result = {}
    for label, rows in groups.items():
        result[label] = [
            round(sum(row[i] for row in rows) / len(rows), 3)
            for i in range(len(rows[0]))
        ]
    return result


def l1(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def predict(features, protos):
    distances = {label: round(l1(features, proto), 3) for label, proto in protos.items()}
    pred = min(distances, key=distances.get)
    return pred, distances


protos = prototypes(train_data)
print("prototypes")
for label in labels:
    print(label, protos[label])

rows = []
for gold, features in val_data:
    pred, distances = predict(features, protos)
    rows.append({"gold": gold, "pred": pred})
    print("prediction", gold, "->", pred, distances)

cm = {g: {p: 0 for p in labels} for g in labels}
for row in rows:
    cm[row["gold"]][row["pred"]] += 1

print("confusion_matrix")
for gold in labels:
    print(gold, [cm[gold][p] for p in labels])

errors = [row for row in rows if row["gold"] != row["pred"]]
print("accuracy:", round((len(rows) - len(errors)) / len(rows), 3))
print("errors:", errors)
```

运行：

```bash
python image_project_baseline.py
```

预期输出：

```text
prototypes
cat [0.85, 0.75, 0.45]
dog [0.65, 0.45, 0.85]
rabbit [0.45, 0.85, 0.25]
prediction cat -> cat {'cat': 0.0, 'dog': 0.9, 'rabbit': 0.7}
prediction dog -> dog {'cat': 0.9, 'dog': 0.0, 'rabbit': 1.2}
prediction rabbit -> rabbit {'cat': 0.7, 'dog': 1.2, 'rabbit': 0.0}
prediction dog -> cat {'cat': 0.09, 'dog': 0.87, 'rabbit': 0.67}
confusion_matrix
cat [1, 0, 0]
dog [1, 1, 0]
rabbit [0, 0, 1]
accuracy: 0.75
errors: [{'gold': 'dog', 'pred': 'cat'}]
```

读这个错误：

- 最后一个 `dog` 样本更接近 `cat` prototype；
- confusion matrix 显示 `dog -> cat`；
- 下一步不要立刻换大模型，先检查 dog 图片是否有像 cat 的背景、姿态或标签问题。

## 真实项目升级路线

| 版本 | 增加什么 | 展示证据 |
|---|---|---|
| baseline | small CNN 或 transfer-learning baseline | train/val 曲线、accuracy |
| evaluation | confusion matrix 和错误样本 | 类别级错误 |
| robustness | augmentation 和 leakage 检查 | before/after 对比 |
| portfolio | README 和 demo command | 可复现运行 |

真实 CNN 项目至少保留：

- 数据目录截图或类别数量表；
- train/validation/test 划分规则；
- baseline model summary；
- metric table；
- confusion matrix；
- 6 到 12 个正确和错误样本；
- 下一步计划。

## 常见错误

| 错误 | 修复 |
|---|---|
| 只报告 accuracy | 展示 confusion matrix 和错误样本 |
| 类别定义模糊 | 收集数据前先定义标签边界 |
| 相似图片泄漏到 train/test | 必要时按来源或主体划分 |
| 一开始就上大模型 | 先做小 baseline |
| 隐藏失败样本 | 用失败样本说明下一步改进 |

## 练习

1. 增加两个 `dog` validation 样本，再重跑 confusion matrix。
2. 增加新类别 `hamster`。`labels` 和 matrix 要怎么变？
3. 为 `dog -> cat` 错误写一个可能的数据原因。
4. 在 README 中把 prototype baseline 替换成 small CNN outline。
5. 做一个项目 checklist，包含 dataset、command、metric 和 failure cases。

## 小结

- 图像分类项目看的是完整闭环，不只是模型名。
- Confusion matrix 能显示类别级失败。
- 错误样本是项目证据，不是丢脸内容。
- 强作品集项目会展示哪里改进了，哪里仍然失败。

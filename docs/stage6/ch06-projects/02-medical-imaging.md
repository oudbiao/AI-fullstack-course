---
title: "6.3 项目：医学影像分析【选修】"
sidebar_position: 19
description: "围绕一个高风险视觉任务，从任务边界、标注协议、指标优先级和风险说明出发，建立更像真实临床辅助项目的闭环。"
keywords: [medical imaging, segmentation, sensitivity, risk, annotation, project]
---

# 项目：医学影像分析【选修】

:::tip 本节定位
医学影像项目和普通视觉项目最大的不同，不在于模型换了个名字，而在于：

- 错误代价更高
- 数据更贵
- 标注更难
- 上线边界更敏感

所以它特别适合用来练“高风险 AI 项目”的判断能力。
:::

## 学习目标

- 学会把医学影像项目范围定得足够清楚
- 学会把标注、指标和临床风险一起写进项目定义
- 学会设计更像临床辅助系统的评估展示方式
- 学会把这类项目做成作品级页面而不是炫图 demo

---

## 一、项目题目为什么一定要收窄？

一个适合作品集的题目可以是：

> **做一个“肺部病灶区域分割辅助系统”，输入 CT slice，输出病灶区域 mask 和风险说明。**

### 为什么这个题目好？

- 输入输出明确
- 指标可解释
- 风险边界清晰

### 为什么不建议一开始做太大？

例如：

- 覆盖多器官、多病种、多模态

这会让项目从一开始就失去可验证性。

---

## 二、作品级医学影像项目最小闭环

1. 定义任务和临床边界
2. 说明标注协议
3. 选 baseline
4. 定义高风险指标
5. 展示成功与失败样例
6. 明确人工复核与适用边界

如果这些没讲清，项目就很难让人信任。

---

## 三、先看一个更像真实项目的规划对象

```python
from dataclasses import dataclass, field


@dataclass
class MedicalProject:
    task: str
    input_type: str
    labels: list
    metrics: list
    clinical_constraints: list
    risks: list = field(default_factory=list)


project = MedicalProject(
    task="肺部病灶区域分割",
    input_type="CT slice",
    labels=["background", "lesion"],
    metrics=["dice", "iou", "sensitivity", "false_negative_rate"],
    clinical_constraints=[
        "高风险样本必须人工复核",
        "结果仅作辅助，不直接替代临床判断",
    ],
    risks=["标注不一致", "类别极度不平衡", "假阴性代价高"],
)

print(project)
```

### 3.1 为什么这里要把 `clinical_constraints` 单独列出来？

因为这类项目和普通视觉项目最大的差别之一就在于：

- 不是只看模型成绩
- 还要看临床使用边界

这也是它更像真实高风险项目的地方。

---

## 四、为什么这类项目最怕假阴性？

如果模型漏掉病灶，  
通常风险比多报一个可疑区域更大。

所以作品级项目里，  
很值得单独展示：

- sensitivity / recall
- false negative rate

而不是只放一个总体准确率。

---

## 五、一个最小“高风险指标优先级”示例

```python
metrics = {
    "dice": 0.81,
    "iou": 0.69,
    "sensitivity": 0.92,
    "false_negative_rate": 0.08,
}


def risk_summary(metrics):
    if metrics["false_negative_rate"] > 0.1:
        return "当前假阴性偏高，不适合直接作为高风险辅助系统。"
    if metrics["sensitivity"] < 0.9:
        return "召回仍偏低，建议优先继续优化病灶检出率。"
    return "指标初步可用，但仍需配合人工复核与临床验证。"


print(risk_summary(metrics))
```

### 5.1 这个例子为什么比只打印一堆分数更有价值？

因为它把指标翻译成了：

- 可用于项目判断的语言

这在医学项目里非常关键。

---

## 六、医学影像项目最值得展示什么？

建议至少展示：

1. 原图
2. 专家标注 mask
3. 模型预测 mask
4. 失败样例
5. 风险边界说明

### 为什么这些比“几张好看的成功图”更重要？

因为高风险项目最重要的是：

- 可信
- 可解释
- 边界清楚

而不是视觉演示效果。

---

## 七、最常见误区

### 7.1 只看总体准确率

### 7.2 不写标注一致性问题

### 7.3 不说明人工复核边界

---

## 八、小结

这节最重要的是建立一个作品级判断：

> **医学影像项目真正像项目的地方，不是模型多复杂，而是你能否把任务边界、标注协议、敏感指标和风险说明一起讲清楚。**

只要这一点做到位，这类项目会非常有说服力。

---

## 练习

1. 把项目再改成一个更小的二分类筛查任务，重写 `clinical_constraints`。
2. 为什么说医学影像项目里 `false_negative_rate` 往往比总体准确率更值得被单独展示？
3. 想一想：标注一致性不高时，模型结果该怎么被解读？
4. 如果把这个项目放进作品集，哪一段风险说明最值得你单独强调？

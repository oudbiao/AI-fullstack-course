---
title: "10.6.1 项目路线图：构建视觉证据包"
sidebar_position: 0
description: "计算机视觉项目的简短实操路线：连接数据、标注、模型输出、指标、失败案例和展示。"
keywords: [CV 项目指南, 安防检测, 医学影像, 图像分类项目, 目标检测项目]
---

# 10.6.1 项目路线图：构建视觉证据包

计算机视觉项目不是“我用了一个模型”，而是数据、标注、模型输出、指标、失败案例和展示的闭环。

## 先看项目闭环

![视觉任务输出粒度递进图](/img/course/ch10-visual-task-progression-map.png)

![视觉项目交付闭环图](/img/course/ch10-projects-delivery-loop.png)

![计算机视觉证据包图](/img/course/ch10-vision-evidence-pack.png)

如果需要最快跑通完整闭环，从分类开始；需要框就进入检测，需要 mask 就进入分割，OCR/视频/3D 适合专门场景。

## 跑一个项目就绪检查

在称为可展示项目前，先跑这个检查。

```python
project = {
    "task": "helmet detection",
    "has_data_note": True,
    "has_metric": True,
    "has_failure_case": True,
    "has_annotation_rule": True,
}

ready = all(project[key] for key in ["has_data_note", "has_metric", "has_failure_case", "has_annotation_rule"])

print("task:", project["task"])
print("presentable:", ready)
```

预期输出：

```text
task: helmet detection
presentable: True
```

如果项目没有标注规则或失败案例，它仍然只是 Demo，不是作品集项目。

## 按这个顺序学

| 步骤 | 项目类型 | 证据 |
|---|---|---|
| 1 | 分类 | 数据划分、accuracy/F1、混淆样例 |
| 2 | 检测 | 框标注、IoU/mAP、误报和漏检 |
| 3 | 分割 | masks、IoU/Dice、边界失败 |
| 4 | 行业场景 | 风险说明、用户影响、部署想法 |
| 5 | 实操工作坊 | 在大项目页前先跑可复现迷你流水线 |

扩展项目前，先运行 [10.6.4 实操：构建可复现视觉迷你流水线](./03-hands-on-vision-workshop.md)。

## 项目交付物标准

| 交付物 | 最低要求 | 更强的作品集版本 |
|---|---|---|
| README | 目标、运行命令、依赖、示例 | 增加任务边界、数据来源、部署想法 |
| 数据与标注 | 图像来源、类别列表、标注格式 | 增加标注示例、质量检查、偏差说明 |
| 结果 | 至少 1 张输入图和预测结果 | 增加正确、误报、漏检、边界案例 |
| 评估 | Accuracy、F1、mAP、IoU、Dice 或 OCR 命中率 | 按类别、场景、光照、清晰度做错误分析 |
| 失败分析 | 至少 1 个真实失败 | 增加疑似原因、修复动作、回归检查 |
| 展示 | 截图或短 GIF 证明能运行 | 构建清晰的视觉项目页面 |

## 通过标准

如果你的视觉项目可复现，有清晰数据和标注规则，报告合适指标，并展示模型在哪里失败，就通过了本章。

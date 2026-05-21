---
title: "10.5.1 高级视觉路线图：OCR、人脸、视频、3D"
sidebar_position: 0
description: "高级视觉方向的简短实操路线：根据输入、输出、风险和项目目标选择 OCR、人脸、视频或 3D。"
keywords: [高级视觉指南, OCR, 视频分析, 人脸识别, 3D 视觉]
---

# 10.5.1 高级视觉路线图：OCR、人脸、视频、3D

高级视觉不是模型名称集合，而是建立在同一视觉基础上的应用方向：输入更复杂、输出更复杂、约束和风险也更多。

## 先看方向地图

![高级视觉方向选择图](/img/course/ch10-advanced-vision-route-map.webp)

![OCR 版面阅读顺序图](/img/course/ch10-ocr-layout-reading-order-map.webp)

![视频帧跟踪时间窗口图](/img/course/ch10-video-frame-tracking-temporal-window-map.webp)

OCR 适合文档，人脸识别适合身份敏感场景，视频适合时间和运动，3D 视觉适合空间结构。

## 跑一个方向选择检查

选择一个方向深入，不要四个方向都浅尝辄止。

```python
requirement = {
    "input": "screenshot",
    "needs_text": True,
    "needs_identity": False,
    "needs_time": False,
    "needs_depth": False,
}

if requirement["needs_text"]:
    direction = "OCR"
elif requirement["needs_identity"]:
    direction = "Face"
elif requirement["needs_time"]:
    direction = "Video"
elif requirement["needs_depth"]:
    direction = "3D"
else:
    direction = "Classification or detection"

print("direction:", direction)
print("first_output:", "text with layout")
```

预期输出：

```text
direction: OCR
first_output: text with layout
```

做人脸、监控、医疗或身份项目时，先写清隐私和使用边界，再展示结果。

## 按这个顺序学

| 步骤 | 方向 | 实操产出 |
|---|---|---|
| 1 | OCR | 抽取文本、版面、字段、置信度和失败样例 |
| 2 | 人脸 | 检测人脸，解释阈值、隐私和偏见风险 |
| 3 | 视频 | 跨帧追踪事件并记录时间维度失败 |
| 4 | 3D 视觉 | 解释深度、点云、几何和传感器假设 |

## 通过标准

如果你能选择一个方向，定义输入/输出，运行最小项目，并记录失败案例和使用边界，就通过了本章。

<details>
<summary>参考答案与讲解</summary>

1. 合格答案要把任务映射到正确的视觉输出：类别标签、检测框、mask、OCR 文本、embedding 或视频事件。
2. 证据应包含渲染后的视觉产物，以及一个指标或定性错误说明。
3. 自检时要能指出一个视觉失败模式，例如类别混淆、漏检、mask 边界差、光照变化、领域偏移或标注质量弱。

</details>


## 留下的证据

学完这一页，至少保留这张证据卡：

```text
scenario_boundary: face, video, OCR, 3D, medical, or another vision scenario
input_sample: source image/frame/document and the expected output type
result_artifact: extracted text, tracked event, depth clue, diagnosis flag, or review note
failure_check: privacy, lighting, temporal drift, layout, calibration, or domain risk
Expected_output: scenario-specific artifact with metric or human-review note
```
